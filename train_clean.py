#! /usr/bin/env python

import os
import datetime
import math
import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data_loader
import argparse

# Args:
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=2, type=int, help='the gpu to use')
parser.add_argument('--dry-run', default=False, const=True, nargs='?', type=bool, help='run with small data file')
parser.add_argument('--cpu', default=False, const=True, nargs='?', type=bool, help='use cpu')
parser.add_argument('--continue-train', default=False, const=True, nargs='?', type=bool, help='continue with training')
parser.add_argument('--suffix', default='50', type=str, help='the suffix of the files to load')
parser.add_argument('--ww', default=360, type=int, help='window width')
parser.add_argument('--dims', default=13, type=int, help='hidden dimensions')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--test', default=False, const=True, nargs='?', type=bool, help='run test only')
parser.add_argument('--best', default=False, const=True, nargs='?', type=bool, help='load best model')
args = parser.parse_args()

# Assing GPU
if not args.cpu:
	torch.cuda.set_device(args.gpu)

# Parameters
# ==================================================
ftype = torch.cuda.FloatTensor if not args.cpu else torch.FloatTensor
ltype = torch.cuda.LongTensor if not args.cpu else torch.LongTensor

# Data loading params
poi2pos_file = './prepro_poi2pos_%s.txt'%args.suffix
train_file = "./prepro_train_%s.txt"%args.suffix
valid_file = "./prepro_valid_%s.txt"%args.suffix
test_file = "./prepro_test_%s.txt"%args.suffix
model_file = './latest_strnn_%s.pt'%args.suffix
best_model_file = './best_strnn_%s.pt'%args.suffix
#gowalla_file = '../dataset/loc-gowalla_totalCheckins.txt'
#if args.dry_run:
#	gowalla_file = '../dataset/small-10000.txt'

# Model Hyperparameters
dim = args.dims # latent dimensionality
ww = args.ww  # winodw width (6h)
up_time = 1440.  # min
lw_time = 0.
up_dist = 40. # distance in meters 
lw_dist = 0.
reg_lambda = 0 # 0.001
top_location_count = 1000 # consider those top locations as negative candidates

# Training Parameters
batch_size = 16 # we can not compute in batches, but at least the optimizer steps in batches.
num_epochs = args.epochs if not args.test else 0
learning_rate = 0.1 # = 0.1/top_location_count
momentum = 0.9
evaluate_every = 3

# best model score
best_model_score = 0.

try:
    xrange
except NameError:
    xrange = range

# Data Preparation
# ===========================================================

# Load data (ancient)
print("Loading data...")
# Todo: map user to id and map loc to id..
poi2pos = data_loader.load_poi2pos(poi2pos_file)
train_user, train_td, train_ld, train_loc, train_dst = data_loader.treat_prepro(train_file, step=1)
valid_user, valid_td, valid_ld, valid_loc, valid_dst = data_loader.treat_prepro(valid_file, step=2)
test_user, test_td, test_ld, test_loc, test_dst = data_loader.treat_prepro(test_file, step=3)

# persist validation errors:
negative_candidates = {}


#print("Loading data from {} ...   ".format(gowalla_file))
#user_cnt, poi2id, train_user, train_time, train_lati, train_longi, train_loc, valid_user, valid_time, valid_lati, valid_longi, valid_loc, test_user, test_time, test_lati, test_longi, test_loc = data_loader.load_data(gowalla_file)

#user_cnt = len(train_user) # 42241 # hardcoded by current preprocessing (42130 on dgx???)
user_cnt = 45000 # allow up to 45'000 users
loc_cnt = len(poi2pos)



print("User/Location: {:d}/{:d}".format(user_cnt, loc_cnt))
print("===========================================================")

###############################################################################################
zero = torch.tensor([0.]).type(ftype)

class STRNN(nn.Module):
    def __init__(self, hidden_size):
        super(STRNN, self).__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # C
        self.weight_th_upper = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # T
        self.weight_th_lower = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # T
        self.weight_sh_upper = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # S
        self.weight_sh_lower = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # S

        self.location_weight = nn.Embedding(loc_cnt, hidden_size)
        #self.permanet_weight = nn.Embedding(user_cnt+112, hidden_size) # wtf here is a bug...
        self.permanet_weight = nn.Embedding(user_cnt, hidden_size)
        
        # rnn stuff:
        self.h0shared = nn.Parameter(torch.randn(dim, 1)).type(ftype)
        self.h_0 = nn.Embedding(user_cnt, hidden_size)
        self.latest_user_state = Variable(torch.randn(user_cnt, hidden_size), requires_grad=False).type(ftype)

        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        self.h0shared = nn.Parameter(torch.randn(dim, 1)).type(ftype)

    def forward(self, td_upper, td_lower, ld_upper, ld_lower, loc, hx):
        
        # stay convex
        td_upper = torch.clamp(td_upper, min=0., max=up_time)
        td_lower = torch.clamp(td_lower, min=0., max=up_time)
        ld_upper = torch.clamp(ld_upper, min=0., max=up_dist)
        ld_lower = torch.clamp(ld_lower, min=0., max=up_dist)
        
        loc_len = len(loc)
        Ttd = [((self.weight_th_upper*td_upper[i] + self.weight_th_lower*td_lower[i])/(td_upper[i]+td_lower[i])) for i in xrange(loc_len)]
        Sld = [((self.weight_sh_upper*ld_upper[i] + self.weight_sh_lower*ld_lower[i])/(ld_upper[i]+ld_lower[i])) for i in xrange(loc_len)]
                
        self.weight_sh_lower.retain_grad()
        self.weight_sh_upper.retain_grad()

        loc = self.location_weight(loc).view(-1,self.hidden_size,1)
        loc_vec = torch.sum(torch.cat([torch.mm(Sld[i], torch.mm(Ttd[i], loc[i])).view(1,self.hidden_size,1) for i in xrange(loc_len)], dim=0), dim=0)
        usr_vec = torch.mm(self.weight_ih, hx)
        hx = loc_vec + usr_vec # hidden_size x 1
        return self.sigmoid(hx)
    
    def loss_candidates(self, user, q_v, Q_neg, h_tq):
        p_u = self.permanet_weight(user)
        user_vec = h_tq + torch.t(p_u)
        
        #positive sample:
        pos_o = torch.mm(q_v, user_vec)
        
        # negative samples:
        neg_os = torch.mm(Q_neg, user_vec)
        
        # loss (hinge):
        #return torch.max(zero, torch.neg(pos_o - neg_os - 1.))
        # loss (original):
        return torch.log(1. + torch.exp(torch.neg(pos_o - neg_os)))
        

    def loss(self, user, td_upper, td_lower, ld_upper, ld_lower, neg_ld_upper, neg_ld_lower, loc, dst, neg_dst, hx):        
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        #self.latest_user_state[user].data = torch.t(h_tq.detach()).data # persist latest state
        p_u = self.permanet_weight(user)
        #p_u.retain_grad() # retain grad for debug
        q_v = self.location_weight(dst)
        pos_o = torch.mm(q_v, (h_tq + torch.t(p_u)))

        neg_h_tq = self.forward(td_upper, td_lower, neg_ld_upper, neg_ld_lower, loc, hx)
        neg_q_v = self.location_weight(neg_dst)
        neg_o = torch.mm(neg_q_v, (neg_h_tq + torch.t(p_u)))
        
        #if (user.item() == 23):
        #    print('pos_o (u=23):', pos_o, neg_o, pos_o - neg_o)

        # variante: hinge loss: (not recommended)
        # return torch.max(zero, torch.neg(pos_o - neg_o - 1.))
        # Variante: original loss
        return torch.log(1. + torch.exp(torch.neg(pos_o - neg_o)))

    def validation_fast(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, dst, hx):
        # error exist in distance (ld_upper, ld_lower)
        # here we are missing the spatial impact on hx (use same everywhere!)
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        p_u = self.permanet_weight(user)
        user_vector = h_tq + torch.t(p_u)
        probs = torch.mm(self.location_weight.weight, user_vector).data.cpu().numpy()
        return np.argsort(np.squeeze(-probs))

    def validation_slow(self, user, td_upper, td_lower, loc, dst, hx):
        
        # go through all locations:
        probs = []
        for j in range(len(poi2pos)):
            ld = []
            curr_coord = poi2pos[j]
            for l in loc: # all latest locations:
                l_coord = poi2pos[l.item()]
                ld.append(euclidean_dist(curr_coord - l_coord))
            ld = np.asarray(ld)
            
            # restrict to local movements
            if (ld[-1] > up_dist):
                probs.append(0.)
                continue
            
            ld_upper = Variable(torch.from_numpy(np.asarray(up_dist-ld))).type(ftype)
            ld_lower = Variable(torch.from_numpy(np.asarray(ld-lw_dist))).type(ftype)
            loc_j = Variable(torch.from_numpy(np.asarray([j]))).type(ltype)
            
            # state for current location:
            h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
            p_u = self.permanet_weight(user)
            user_vector = h_tq + torch.t(p_u)
            probs.append(torch.mm(self.location_weight(loc_j), user_vector).cpu().item())
        probs = np.array(probs)
        return np.argsort(-probs) # reverse argsort!
        

###############################################################################################
def parameters():
    params = []
    for model in [strnn_model]:
        params += list(model.parameters())

    return params

def print_score(batches, step, label = 'validation', update_best=False):
    global best_model_score
    recall1 = 0.
    recall5 = 0.
    recall10 = 0.
    recall100 = 0.
    recall1000 = 0.
    recall10000 = 0.
    iter_cnt = 0
    
    users_in_10 = []
    with torch.no_grad():
        for batch in tqdm.tqdm(batches, desc=label):
            batch_user, batch_td, batch_ld, batch_loc, batch_dst = batch
            #if len(batch_loc) < 3: # skip with less than 3 history
            #    continue
            iter_cnt += 1
            batch_o, target = run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=step)

            # persist negative candidates:
            # negative_candidates[batch_user] = batch_o[:top_location_count]

            if (target in batch_o[:10]):
                users_in_10.append(batch_user)
                #print('user recall@10:', batch_user)
                #print('predicted:', batch_o[:10])
                #print('actual:', target)

            recall1 += target in batch_o[:1]
            recall5 += target in batch_o[:5]
            recall10 += target in batch_o[:10]
            recall100 += target in batch_o[:100]
            recall1000 += target in batch_o[:1000]
            #recall10000 += target in batch_o[:10000]

    print('users recall@10', users_in_10)
    print("recall@1: ", recall1/iter_cnt)
    print("recall@5: ", recall5/iter_cnt)
    print("recall@10: ", recall10/iter_cnt)
    print("recall@100: ", recall100/iter_cnt)
    print("recall@1000: ", recall1000/iter_cnt)
    #print("recall@10000: ", recall10000/iter_cnt)
    
    score = recall10/iter_cnt
    if (update_best == True and best_model_score <= score):
        best_model_score = score
        torch.save(strnn_model.state_dict(), best_model_file)    
        print('best model saved')
        

###############################################################################################
#def euclidean_dist(coord):
#    return torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))

def euclidean_dist(coords):
    return np.sqrt(np.power(coords[0], 2) + np.power(coords[1], 2) + np.power(coords[2], 2))    

def run(user, td, ld, loc, dst, step):
    # run for one user (and all his window width batches [seqlen])
    # preserve rnn_output among different patches
    # "interpolate" here

    #seqlen = len(td)
    if step > 1:
        seqlen = len(td)
    else:
        seqlen = int(torch.FloatTensor(1).uniform_(3, len(td)).long().item()) # update to random seqlen (should have at least 3 batches)
    
    user_idx = user
    user = Variable(torch.from_numpy(np.asarray([user]))).type(ltype)
    
    # Case 1: use trainig state for validation!
    #if step == 1:
    #    rnn_output = torch.t(strnn_model.h_0(user)) # update on h_0
    #else:
    #    rnn_output = torch.t(strnn_model.latest_user_state[user]) # reuse latest state
    
    # Case 2: use shared h_0:c
    rnn_output = strnn_model.h0shared        
        
    for idx in xrange(seqlen-1):
        #TODO: create once and reuse!    
        td_upper = Variable(torch.from_numpy(np.asarray(up_time-td[idx]))).type(ftype)
        td_lower = Variable(torch.from_numpy(np.asarray(td[idx]-lw_time))).type(ftype)
        ld_upper = Variable(torch.from_numpy(np.asarray(up_dist-ld[idx]))).type(ftype)
        ld_lower = Variable(torch.from_numpy(np.asarray(ld[idx]-lw_dist))).type(ftype)
        location = Variable(torch.from_numpy(np.asarray(loc[idx]))).type(ltype)
        rnn_output = strnn_model(td_upper, td_lower, ld_upper, ld_lower, location, rnn_output)#, neg_lati, neg_longi, neg_loc, step)

    # positive sample last round:
    td_upper = Variable(torch.from_numpy(np.asarray(up_time-td[seqlen-1]))).type(ftype)
    td_lower = Variable(torch.from_numpy(np.asarray(td[seqlen-1]-lw_time))).type(ftype)
    ld_upper = Variable(torch.from_numpy(np.asarray(up_dist-ld[seqlen-1]))).type(ftype)
    ld_lower = Variable(torch.from_numpy(np.asarray(ld[seqlen-1]-lw_dist))).type(ftype)
    location = Variable(torch.from_numpy(np.asarray(loc[seqlen-1]))).type(ltype)

    if step > 1:
        return strnn_model.validation_fast(user, td_upper, td_lower, ld_upper, ld_lower, location, dst[seqlen-1], rnn_output), dst[seqlen-1]
        #return strnn_model.validation_slow(user, td_upper, td_lower, location, dst[seqlen-1], rnn_output), dst[seqlen-1]

    # negative sample last round:
    # pick negative sample (regardless of current)
    # Strategy: uniform from top candidates:
    #if user_idx in negative_candidates: 
    #    neg_idx = int(torch.FloatTensor(1).uniform_(0, top_location_count-1).long().item())
    #    neg_loc = negative_candidates[user_idx][neg_idx]
    #    while (neg_loc == dst[seqlen-1]):
    #        neg_idx = int(torch.FloatTensor(1).uniform_(0, top_location_count-1).long().item())
    #        neg_loc = negative_candidates[user_idx][neg_idx]
    #    neg_coord = poi2pos[neg_loc]
    #else:
    #    #print('user not validated', user_idx)
    #    neg_loc = int(torch.FloatTensor(1).uniform_(0, len(poi2pos)-1).long().item())
    #    neg_coord = poi2pos[neg_loc]
    
    # Strategy: uniform inside upper distance distance
    #neg_loc = int(torch.FloatTensor(1).uniform_(0, len(poi2pos)-1).long().item())
    #neg_coord = poi2pos[neg_loc]
    #dest_coord = poi2pos[dst[-1]]
    #counter = 5000
    #while(counter > 0 and (euclidean_dist(neg_coord - dest_coord) > up_dist/4 or neg_loc == dst[-1])):
    #    neg_loc = int(torch.FloatTensor(1).uniform_(0, len(poi2pos)-1).long().item())
    #    neg_coord = poi2pos[neg_loc]
    #    counter -= 1
    #if (counter == 0):
    #    print('resampling counter exceeded!')
    
    Jtot = torch.tensor([[0.]]).type(ftype)
    
    # Strategie: all neg candidates at once!
    #Q = torch.zeros(top_location_count, dim).type(ftype) # negative candiate weights
    #for idx in range(top_location_count):
        #if user_idx in negative_candidates:
        #    neg_loc = negative_candidates[user_idx][idx]
        #    if (neg_loc == dst[seqlen-1]): # skip own location
        #        neg_loc = int(torch.FloatTensor(1).uniform_(0, len(poi2pos)-1).long().item())
        #else:
        #    neg_loc = int(torch.FloatTensor(1).uniform_(0, len(poi2pos)-1).long().item())
        #neg_loc = Variable(torch.from_numpy(np.asarray([neg_loc]))).type(ltype)
        #Q[idx, :] = strnn_model.location_weight(neg_loc)
        
    Q = strnn_model.location_weight.weight
    #print('Q', Q)
    destination = Variable(torch.from_numpy(np.asarray([dst[seqlen-1]]))).type(ltype) # positive destination
    destination = strnn_model.location_weight(destination)
    h_tq = strnn_model.forward(td_upper, td_lower, ld_upper, ld_lower, location, rnn_output)
    J = strnn_model.loss_candidates(user, destination, Q, h_tq)
    #print('J', J)
    return torch.mean(J)
            
        
    '''
    for idx in range(top_location_count):
        if user_idx in negative_candidates: 
            neg_loc = negative_candidates[user_idx][idx]
            if (neg_loc == dst[seqlen-1]):
                continue
            neg_coord = poi2pos[neg_loc]
        else:
            #print('user not validated', user_idx)
            neg_loc = int(torch.FloatTensor(1).uniform_(0, len(poi2pos)-1).long().item())
            neg_coord = poi2pos[neg_loc]
        
    
        neg_ld = []
        for l in loc[seqlen-1]: # all latest locations:
            l_coord = poi2pos[l]
            neg_ld.append(euclidean_dist(neg_coord - l_coord))
        neg_ld = np.asarray(neg_ld)
        
        #neg_ld_upper = Variable(torch.from_numpy(np.asarray(up_dist-neg_ld))).type(ftype)
        #neg_ld_lower = Variable(torch.from_numpy(np.asarray(neg_ld-lw_dist))).type(ftype)

        destination = Variable(torch.from_numpy(np.asarray([dst[seqlen-1]]))).type(ltype)
        neg_destination = Variable(torch.from_numpy(np.asarray([neg_loc]))).type(ltype)

        #J = strnn_model.loss(user, td_upper, td_lower, ld_upper, ld_lower, neg_ld_upper, neg_ld_lower, location, destination, neg_destination, rnn_output)#, neg_lati, neg_longi, neg_loc, step)
        # Fast validation: use same distance for negative samples all!
        J = strnn_model.loss(user, td_upper, td_lower, ld_upper, ld_lower, ld_upper, ld_lower, location, destination, neg_destination, rnn_output)
        Jtot = Jtot + J
    
    return Jtot
    '''

###############################################################################################
strnn_model = STRNN(dim).cuda() if not args.cpu else STRNN(dim)
optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=momentum, weight_decay=reg_lambda)
#optimizer = torch.optim.Adam(parameters(), lr=learning_rate, weight_decay=reg_lambda)

if args.continue_train or args.test:
    print('load model...')
    if args.best:
        strnn_model.load_state_dict(torch.load(best_model_file))
    else:
        strnn_model.load_state_dict(torch.load(model_file))

for i in xrange(num_epochs):
        
    # Evaluation (used to obtain negative candidates)
    if (i+1) % evaluate_every == 0 or i == 0:
        print("==================================================================================")
        print("Epoch:", i+1)
        #print("Evaluation at epoch #{:d}: ".format(i+1)), total_loss/j, datetime.datetime.now()
        valid_batches = list(zip(valid_user, valid_td, valid_ld, valid_loc, valid_dst))
        print_score(valid_batches, step=2, label='validation', update_best = True)
        train_batches = list(zip(train_user, train_td, train_ld, train_loc, train_dst))
        print_score(train_batches, step=2, label='train-set')
    
    # Training
        
    total_loss = 0.
    Jtot = None
    
    train_ids = list(range(len(train_user)))
    random.shuffle(train_ids)
    for j, idx in enumerate(tqdm.tqdm(train_ids, desc='train')):
        user = train_user[idx]
        td = train_td[idx]
        ld = train_ld[idx]
        loc = train_loc[idx]
        dst = train_dst[idx]
        
        if j % batch_size == 0:
            optimizer.zero_grad()
            Jtot = torch.tensor([[0]]).type(ftype)
        
        if len(loc) < 3:
            #print('skip user', user)
            continue # skip whatever..
        
        J = run(user, td, ld, loc, dst, step=1)
        Jtot += J
        
        if (j+1) % batch_size == 0:
            Jtot.backward()
            optimizer.step()
        
        #total_loss += J.cpu().detach().numpy()
        #if (j+1) % 250 == 0:
        #    print("batch #{:d}: ".format(j+1), "batch_loss :", total_loss/j, datetime.datetime.now())
    
    # end of epoch step
    print ('end of epoch step')
    Jtot.backward()
    optimizer.step()
    print("final step loss :", Jtot.item(), datetime.datetime.now())
    
    #print("last loss", J.item())
    
    # analyze user:
    for u in []:
        user_idx = torch.tensor(u).type(ltype)
        p_u = strnn_model.permanet_weight(user_idx)
        print('p_u', p_u.cpu())
        print('p_u.requires_grad', p_u.requires_grad)
        print('grad', p_u.grad)
        grad_u = p_u.grad.dot(p_u.grad)
        print('user ', user_idx.cpu().item(), ':', p_u, '||grad||^2', grad_u) 
    # analyze Spatial S:
    #print('*** spatial S ***')
    #grd = strnn_model.weight_sh_lower.cpu().grad
    #grd = grd.view(-1)
    #normgrd = grd.dot(grd)
    #print(strnn_model.weight_sh_lower.cpu())
    #print(grd)
    #print(normgrd)
    
    # safe model
    torch.save(strnn_model.state_dict(), model_file)    
    print('model saved')
    
    # old training
    #train_batches = list(zip(train_user, train_td, train_ld, train_loc, train_dst))
    #for j, train_batch in enumerate(tqdm.tqdm(train_batches, desc="train")):
    #    batch_user, batch_td, batch_ld, batch_loc, batch_dst = train_batch
    #    if len(batch_loc) < 3:
    #        continue # skip entries with less than 3 batches (why?, only 30 checkins are considered)
    #    total_loss += run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=1)
    #    if (j+1) % 2000 == 0:
    #        print("batch #{:d}: ".format(j+1), "batch_loss :", total_loss/j, datetime.datetime.now())
    

# Testing
print("Training End..")
print("==================================================================================")
print("Test: ")
test_batches = list(zip(test_user, test_td, test_ld, test_loc, test_dst))
print_score(test_batches, step=3, label='test')
