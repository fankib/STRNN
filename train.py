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
parser.add_argument('--cpu', default=False, const=True, nargs='?', type=bool, help='use cpu')
parser.add_argument('--continue-train', default=False, const=True, nargs='?', type=bool, help='continue with training')
parser.add_argument('--suffix', default='50', type=str, help='the suffix of the files to load')
parser.add_argument('--dims', default=13, type=int, help='hidden dimensions')
parser.add_argument('--epochs', default=30, type=int, help='number of epochs')
parser.add_argument('--test', default=False, const=True, nargs='?', type=bool, help='run test only')
parser.add_argument('--best', default=False, const=True, nargs='?', type=bool, help='load best model')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=16, type=int, help='the batch size')
parser.add_argument('--up-time', default = 1440., type=float, help='upper time bound in minutes')
parser.add_argument('--up-dist', default = 40., type=float, help='upper distance bound in km')
parser.add_argument('--silent', default=False, const=True, nargs='?', type=bool, help='run in silent mode')
parser.add_argument('--reg', default=0., type=float, help='regularization')
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

# Model Hyperparameters
dim = args.dims # latent dimensionality
up_time = args.up_time  # in minutes
lw_time = 0.
up_dist = args.up_dist # in km 
lw_dist = 0.
reg_lambda = args.reg

# Training Parameters
batch_size = args.batch_size # we can not compute in batches, but at least the optimizer steps in batches.
num_epochs = args.epochs if not args.test else 0
learning_rate = args.lr
momentum = 0.9
evaluate_every = 5

# best model score
best_model_score = 0.

# print log to std out if not silent
def log(*opts):
    if not args.silent:
        print(*opts)

# Data Preparation
# ===========================================================

log("Loading data...")
poi2pos = data_loader.load_poi2pos(poi2pos_file)
train_user, train_td, train_ld, train_loc, train_dst = data_loader.treat_prepro(train_file, step=1)
valid_user, valid_td, valid_ld, valid_loc, valid_dst = data_loader.treat_prepro(valid_file, step=2)
test_user, test_td, test_ld, test_loc, test_dst = data_loader.treat_prepro(test_file, step=3)

#user_cnt = len(train_user) # here was some problem..
user_cnt = 45000 # allow up to 45'000 users
loc_cnt = len(poi2pos)


log("User/Location: {:d}/{:d}".format(user_cnt, loc_cnt))
log("===========================================================")

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
        Ttd = [((self.weight_th_upper*td_upper[i] + self.weight_th_lower*td_lower[i])/(td_upper[i]+td_lower[i])) for i in range(loc_len)]
        Sld = [((self.weight_sh_upper*ld_upper[i] + self.weight_sh_lower*ld_lower[i])/(ld_upper[i]+ld_lower[i])) for i in range(loc_len)]
                
        self.weight_sh_lower.retain_grad()
        self.weight_sh_upper.retain_grad()

        loc = self.location_weight(loc).view(-1,self.hidden_size,1)
        loc_vec = torch.sum(torch.cat([torch.mm(Sld[i], torch.mm(Ttd[i], loc[i])).view(1,self.hidden_size,1) for i in range(loc_len)], dim=0), dim=0)
        usr_vec = torch.mm(self.weight_ih, hx)
        hx = loc_vec + usr_vec # hidden_size x 1
        return self.sigmoid(hx)
    
    def loss(self, user, q_v, h_tq):
        p_u = self.permanet_weight(user)
        user_vec = h_tq + torch.t(p_u)
        
        #positive sample:
        pos_o = torch.mm(q_v, user_vec)
        
        # negative samples:
        neg_os = torch.mm(self.location_weight.weight, user_vec)
        
        # loss (original):
        return torch.log(1. + torch.exp(torch.neg(pos_o - neg_os)))

    def validation_fast(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, dst, hx):
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

def print_score(batches, step, label = 'validation', update_best=False, slow_validation=False):
    global best_model_score
    
    recall1 = 0.
    recall5 = 0.
    recall10 = 0.
    recall100 = 0.
    recall1000 = 0.
    recall10000 = 0.
    average_precision = 0.
    iter_cnt = 0
    
    users_in_10 = []
    with torch.no_grad():
        for batch in tqdm.tqdm(batches, desc=label):
            batch_user, batch_td, batch_ld, batch_loc, batch_dst = batch
            if len(batch_loc) < 3: # skip with less than 3 history batches
                continue
            iter_cnt += 1
            batch_o, target = run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=step, slow_validation=slow_validation)

            if (target in batch_o[:10]):
                users_in_10.append(batch_user)
                
            idx_target = np.where(batch_o == target)[0][0]
            precision = 1./(idx_target+1)
            average_precision += precision

            recall1 += target in batch_o[:1]
            recall5 += target in batch_o[:5]
            recall10 += target in batch_o[:10]
            recall100 += target in batch_o[:100]
            recall1000 += target in batch_o[:1000]
            
            recall_at_1 = recall1/iter_cnt
            recall_at_5 = recall5/iter_cnt
            recall_at_10 = recall10/iter_cnt
            recall_at_100 = recall100/iter_cnt
            recall_at_1000 = recall1000/iter_cnt
            m_AP = average_precision/iter_cnt
            
            if slow_validation and iter_cnt % 5 == 0:
                log('validation for {}'.format(iter_cnt))
                log('users recall@10', users_in_10)
                log("recall@1: ", recall_at_1)
                log("recall@5: ", recall_at_5)
                log("recall@10: ", recall_at_10)
                log("recall@100: ", recall_at_100)
                log("recall@1000: ", recall_at_1000)
                log("MAP", m_AP)

    if args.silent:
        print('score: MAP {:0.4f} \t recall@1 {:0.3f} \t recall@5 {:0.3f} \t recall@10 {:0.3f}'.format(m_AP, recall_at_1, recall_at_5, recall_at_10))

    log('users recall@10', users_in_10)
    log("recall@1: ", recall_at_1)
    log("recall@5: ", recall_at_5)
    log("recall@10: ", recall_at_10)
    log("recall@100: ", recall_at_100)
    log("recall@1000: ", recall_at_1000)
    log("MAP", m_AP)
    
    score = recall10/iter_cnt
    if (update_best == True and best_model_score <= score):
        best_model_score = score
        torch.save(strnn_model.state_dict(), best_model_file)    
        log('best model saved')
        

###############################################################################################

def euclidean_dist(coords):
    return np.sqrt(np.power(coords[0], 2) + np.power(coords[1], 2) + np.power(coords[2], 2))    

def run(user, td, ld, loc, dst, step, slow_validation=False):
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
    
    # Case 2: use shared h_0:c
    rnn_output = strnn_model.h0shared        
        
    for idx in range(seqlen-1):
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

    if step > 1 and not slow_validation:
        return strnn_model.validation_fast(user, td_upper, td_lower, ld_upper, ld_lower, location, dst[seqlen-1], rnn_output), dst[seqlen-1]
    if step > 1 and slow_validation:
        return strnn_model.validation_slow(user, td_upper, td_lower, location, dst[seqlen-1], rnn_output), dst[seqlen-1]
    
    destination = Variable(torch.from_numpy(np.asarray([dst[seqlen-1]]))).type(ltype) # positive destination
    destination = strnn_model.location_weight(destination)
    h_tq = strnn_model.forward(td_upper, td_lower, ld_upper, ld_lower, location, rnn_output)
    J = strnn_model.loss(user, destination, h_tq)
    J = torch.mean(J)
    return J

###############################################################################################
strnn_model = STRNN(dim).cuda() if not args.cpu else STRNN(dim)
optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=momentum, weight_decay=reg_lambda)
#optimizer = torch.optim.Adam(parameters(), lr=learning_rate, weight_decay=reg_lambda)

if args.continue_train or args.test:
    log('load model...')
    if args.best:
        strnn_model.load_state_dict(torch.load(best_model_file))
    else:
        strnn_model.load_state_dict(torch.load(model_file))

for i in range(num_epochs):

    # Training
    total_loss = 0.
    total_loss_cnt = 0
    train_ids = list(range(len(train_user)))
    random.shuffle(train_ids)
    for j, idx in enumerate(tqdm.tqdm(train_ids, desc='train')):
        user = train_user[idx]
        td = train_td[idx]
        ld = train_ld[idx]
        loc = train_loc[idx]
        dst = train_dst[idx]
        if len(train_dst) < 3:
            continue # skip users with to little history
        
        if j % batch_size == 0:
            optimizer.zero_grad()
            Jtot = torch.tensor([[0]]).type(ftype)
        
        if len(loc) < 3:
            #log('skip user', user)
            continue # skip whatever..
        
        J = run(user, td, ld, loc, dst, step=1)
        Jtot += J
        
        if (j+1) % batch_size == 0:
            Jtot.backward()
            optimizer.step()
            total_loss += Jtot.item()
            total_loss_cnt += 1
            
        if (j+1) % 2000 == 0:
            log("batch #{:d}: ".format(j+1), "batch_loss :", total_loss/total_loss_cnt, datetime.datetime.now())

    
    # end of epoch step
    Jtot.backward()
    optimizer.step()
    log("final step loss :", Jtot.item(), datetime.datetime.now())
    
    # safe model
    torch.save(strnn_model.state_dict(), model_file)    
    
    # Evaluation (used to obtain negative candidates)
    if (i+1) % evaluate_every == 0:
        log("==================================================================================")
        log("Evaluation epoch:", i+1)
        valid_batches = list(zip(valid_user, valid_td, valid_ld, valid_loc, valid_dst))
        print_score(valid_batches, step=2, label='validation', update_best = True)

# Testing (only with --test)
log("Training End..")
if args.test:
    log("==================================================================================")
    log("Test: ")
    test_batches = list(zip(test_user, test_td, test_ld, test_loc, test_dst))
    print_score(test_batches, step=3, label='test', slow_validation=True)
