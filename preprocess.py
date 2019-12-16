#! /usr/bin/env python

import os
import datetime
import csv
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import sys

import data_loader

# fix recursion depth
sys.setrecursionlimit(10000)

# Args:
parser = argparse.ArgumentParser()
parser.add_argument('--max-users', default=0, type=int, help='users to process')
parser.add_argument('--suffix', default='50', type=str, help='the suffix of the files to load')
parser.add_argument('--ww', default=360, type=int, help='window width')
args = parser.parse_args()

# Prepare for small files:
restrict_max_users = True if args.max_users > 0 else False
max_users = args.max_users
max_users_str = args.suffix

# Parameters
# ==================================================
ftype = torch.FloatTensor
ltype = torch.LongTensor

# Data loading params
train_file = "../dataset/loc-gowalla_totalCheckins.txt"

# Model Hyperparameters
dim = 13    # dimensionality
ww = args.ww  # winodw width (6h)
up_time = 1440  # 1d
lw_time = 50    # 50m
up_dist = 100   # ??
lw_dist = 1

# Training Parameters
batch_size = 2
learning_rate = 0.001
momentum = 0.9
evaluate_every = 1

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
user_cnt, poi2id, poi2pos, train_user, train_time, train_coord, train_loc, valid_user, valid_time, valid_coord, valid_loc, test_user, test_time, test_coord, test_loc = data_loader.load_data(train_file, args.max_users)

#np.save("poi2id_30", poi2id)
print("User/Location: {:d}/{:d}".format(user_cnt, len(poi2id)))
print("==================================================================================")

class STRNNModule(nn.Module):
    def __init__(self):
        super(STRNNModule, self).__init__()

        # embedding:
        self.user_weight = Variable(torch.randn(user_cnt, dim), requires_grad=False).type(ftype)
        self.h_0 = Variable(torch.randn(dim, 1), requires_grad=False).type(ftype)
        self.location_weight = nn.Embedding(len(poi2id), dim)
        self.perm_weight = nn.Embedding(user_cnt, dim)
        # attributes:
        self.time_upper = nn.Parameter(torch.randn(dim, dim).type(ftype))
        self.time_lower = nn.Parameter(torch.randn(dim, dim).type(ftype))
        self.dist_upper = nn.Parameter(torch.randn(dim, dim).type(ftype))
        self.dist_lower = nn.Parameter(torch.randn(dim, dim).type(ftype))
        self.C = nn.Parameter(torch.randn(dim, dim).type(ftype))

        # modules:
        self.sigmoid = nn.Sigmoid()

    # find the most closest value to w, w_cap(index)
    # the index is used to denote t_i
    # returns the index for time/checkin corresponding to w_cap
    def find_w_cap(self, times, i):
        trg_t = times[i] - ww
        tmp_t = times[i]
        tmp_i = i-1
        for idx, t_w in enumerate(reversed(times[:i]), start=1):
            if t_w.data.cpu().numpy() == trg_t.data.cpu().numpy():
                return i-idx
            elif t_w.data.cpu().numpy() > trg_t.data.cpu().numpy():
                tmp_t = t_w
                tmp_i = i-idx
            elif t_w.data.cpu().numpy() < trg_t.data.cpu().numpy():
                if trg_t.data.cpu().numpy() - t_w.data.cpu().numpy() \
                    < tmp_t.data.cpu().numpy() - trg_t.data.cpu().numpy():
                    return i-idx
                else:
                    return tmp_i
        return 0

    # arrays for times, latis, longis, locs, one "row" is one checkin
    # splits the history of a user into window width patches.
    # each such patch will be placed on one row.
    # one row consists of 4 tab separated parts: time-differences (in minutes), distances (euclidean on WGS84), location-numbers, current location
    # each of those things is an array and comma separated
    def return_h_tw(self, times, coord, locs, idx):
        w_cap = self.find_w_cap(times, idx)
        if w_cap is 0:
            return self.h_0
        else:
            # recursion on previous windows!
            # so they are written first to the file
            self.return_h_tw(times, coord, locs, w_cap)
            #self.return_h_tw(times, coord, locs, idx-1) # recusively through each location!

        # idx denotes the current time/position at this check in
        td = times[idx] - times[w_cap:idx] # differences in time
        ld = self.euclidean_dist(coord[idx]-coord[w_cap:idx] )

        data = ','.join(str(e) for e in td.data.cpu().numpy())+"\t"
        f.write(data)
        data = ','.join(str(e) for e in ld.data.cpu().numpy())+"\t"
        f.write(data)
        data = ','.join(str(e.data.cpu().numpy()) for e in locs[w_cap:idx])+"\t"
        f.write(data)
        data = str(locs[idx].data.cpu().numpy())+"\n"
        f.write(data)

    # get transition matrices by linear interpolation
    def get_location_vector(self, td, ld, locs):
        tud = up_time - td
        tdd = td - lw_time
        lud = up_dist - ld
        ldd = ld - lw_dist
        loc_vec = 0
        for i in xrange(len(tud)):
            Tt = torch.div(torch.mul(self.time_upper, tud[i]) + torch.mul(self.time_lower, tdd[i]),
                            tud[i]+tdd[i])
            Sl = torch.div(torch.mul(self.dist_upper, lud[i]) + torch.mul(self.dist_lower, ldd[i]),
                            lud[i]+ldd[i])
            loc_vec += torch.mm(Sl, torch.mm(Tt, torch.t(self.location_weight(locs[i]))))
        return loc_vec
    
    def euclidean_dist(self, coords):
        return torch.sqrt(torch.pow(coords[:,0], 2) + torch.pow(coords[:,1], 2) + torch.pow(coords[:,2], 2))

    def forward(self, user, times, coord, locs, step):#neg_lati, neg_longi, neg_loc, step):
        f.write(str(user.data.cpu().numpy()[0])+"\n")
        # positive sampling
        pos_h = self.return_h_tw(times, coord, locs, len(times)-1)

###############################################################################################
def run(user, time, coord, loc, step):

    # pushes the corresponding batches to GPU as tensors!
    
    user = Variable(torch.from_numpy(np.asarray([user]))).type(ltype)
    time = Variable(torch.from_numpy(np.asarray(time))).type(ftype)
    coord = Variable(torch.from_numpy(np.asarray(coord))).type(ftype)
    loc = Variable(torch.from_numpy(np.asarray(loc))).type(ltype)

    rnn_output = strnn_model(user, time, coord, loc, step)#, neg_lati, neg_longi, neg_loc, step)

###############################################################################################
strnn_model = STRNNModule()

print('make poi2pos file...')
f = open('./prepro_poi2pos_%s.txt'%max_users_str, 'w')
for key in poi2pos:
    x,y,z = poi2pos[key]
    f.write('{}\t{}\t{}\t{}\n'.format(key, x,y,z))
f.close()

print("Making train file...")
f = open("./prepro_train_%s.txt"%max_users_str, 'w')
# Training
train_batches = list(zip(train_time, train_coord, train_loc))
for j, train_batch in enumerate(tqdm.tqdm(train_batches, desc="train")):
    #if (restrict_max_users and j >= max_users):
    #    break
    batch_time, batch_coord, batch_loc = train_batch#inner_batch)
    # run once per user and all its training_{time, coord, loc}:
    run(train_user[j], batch_time, batch_coord, batch_loc, step=1)
f.close()

print("Making valid file...")
f = open("./prepro_valid_%s.txt"%max_users_str, 'w')
# Eavludating
valid_batches = list(zip(valid_time, valid_coord, valid_loc))
for j, valid_batch in enumerate(tqdm.tqdm(valid_batches, desc="valid")):
    #if (restrict_max_users and j >= max_users):
    #    break
    batch_time, batch_coord, batch_loc = valid_batch#inner_batch)
    run(valid_user[j], batch_time, batch_coord, batch_loc, step=2)
f.close()

print("Making test file...")
f = open("./prepro_test_%s.txt"%max_users_str, 'w')
# Testing
test_batches = list(zip(test_time, test_coord, test_loc))
for j, test_batch in enumerate(tqdm.tqdm(test_batches, desc="test")):
    #if (restrict_max_users and j >= max_users):
    #    break
    batch_time, batch_coord, batch_loc = test_batch#inner_batch)
    run(test_user[j], batch_time, batch_coord, batch_loc, step=3)
f.close()
