import numpy as np
from datetime import datetime
import pandas as pd
import time

def load_poi2pos(poi2pos_file):
    poi2pos = {}
    f = open(poi2pos_file, 'r')
    lines = f.readlines()
    
    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        location = int(tokens[0])
        x = float(tokens[1])
        y = float(tokens[2])
        z = float(tokens[3])
        poi2pos[location] = np.array([x,y,z])
    
    return poi2pos

# use this to load a preprocessed file
def treat_prepro(train, step):
    train_f = open(train, 'r')
    lines = train_f.readlines()
    
    # fsb1: outdated
    # Need to change depending on threshold
    #if step==1:
    #    lines = train_f.readlines()#[:86445] #659 #[:309931]
    #elif step==2:
    #    lines = train_f.readlines()#[:13505]#[:309931]
    #elif step==3:
    #    lines = train_f.readlines()#[:30622]#[:309931]

    # update fsb1: ld's are computed only for positive samples
    
    train_user = []
    train_td = []
    train_ld = []
    train_loc = []
    train_dst = []

    user = 1
    user_td = []
    user_ld = []
    user_loc = []
    user_dst = []

    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        if len(tokens) < 3:
            if user_td: 
                train_user.append(user)
                train_td.append(user_td)
                train_ld.append(user_ld)
                train_loc.append(user_loc)
                train_dst.append(user_dst)
            user = int(tokens[0])
            user_td = []
            user_ld = []
            user_loc = []
            user_dst = []
            continue
        td = np.array([float(t) for t in tokens[0].split(',')])
        ld = np.array([float(t) for t in tokens[1].split(',')])
        loc = np.array([int(t) for t in tokens[2].split(',')])
        dst = int(tokens[3])
        user_td.append(td)
        user_ld.append(ld)
        user_loc.append(loc)
        user_dst.append(dst)

    # process last line
    if user_td: 
        train_user.append(user)
        train_td.append(user_td)
        train_ld.append(user_ld)
        train_loc.append(user_loc)
        train_dst.append(user_dst)

    return train_user, train_td, train_ld, train_loc, train_dst

# WGS84 to R3:
def WGS84_to_R3(lati, longi):
    ''' transfer lat lon to 3d points on earth approxiation '''
    R =  6371. # average radius of the earth
    lat_r, lon_r = lati*np.pi/180., longi*np.pi/180.
    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return (x, y, z)

# Note this is called from preprocess using gowalla full.
def load_data(train, max_users=0):
    
    # maps to resolve ids (ids are continous, numbers are not)
    user2id = {} # map a user number to its id
    poi2id = {} # map a location number to its id
    poi2pos = {} # map a location number to lat long

    # arrays to store data
    train_user = []
    train_time = []
    train_coord = []    
    train_loc = []
    valid_user = []
    valid_time = []
    valid_coord = []
    valid_loc = []
    test_user = []
    test_time = []
    test_coord = []
    test_loc = []

    user_time = []
    user_coord = []
    user_loc = []
    visit_thr = 30 # only consider users with 30 checkins
    
    # organize the data by users
    # (assign each line (checkin) to the corresponding user)
    # (ids are continous, users are distinctive numbers)

    # collect all users with visit_thr checkins:
    train_f = open(train, 'r')
    lines = train_f.readlines()
    
    prev_user = int(lines[0].split('\t')[0])
    visit_cnt = 0
    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        user = int(tokens[0])
        if user==prev_user:
            visit_cnt += 1
        else:
            if visit_cnt >= visit_thr:
                user2id[prev_user] = len(user2id)
            prev_user = user
            visit_cnt = 1
        if max_users > 0 and len(user2id) >= max_users:
            break # restrict to max users

    # we read the file again:
    # this time we collect the data for the interresting users 

    train_f = open(train, 'r')
    lines = train_f.readlines()

    prev_user = int(lines[0].split('\t')[0])
    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        user = int(tokens[0])
        if user2id.get(user) is None:
            continue
        user = user2id.get(user)

        time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")\
                -datetime(2009,1,1)).total_seconds()/60  # minutes since 1.1.2009
        lati = float(tokens[2]) # WGS84? Latitude
        longi = float(tokens[3]) # WGS84? Longitude
        coord = WGS84_to_R3(lati, longi)
        location = int(tokens[4]) # location nr
        if poi2id.get(location) is None: # get-or-set locations
            poi2id[location] = len(poi2id)
            poi2pos[poi2id.get(location)] = coord
        location = poi2id.get(location)

        if user == prev_user:
            # insert in front!
            user_time.insert(0, time)
            user_coord.insert(0, coord)
            user_loc.insert(0, location)
        else:
            # add each user once to train, once to validate and once to test:
            # 0%..70% to train
            # 70%..80% to validate
            # 80%..100% to test
            # ordered in time where training corresponds to the farthest in time.
            train_thr = int(len(user_time) * 0.7)
            valid_thr = int(len(user_time) * 0.8)
            train_user.append(user)
            train_time.append(user_time[:train_thr])
            train_coord.append(user_coord[:train_thr])
            train_loc.append(user_loc[:train_thr])
            valid_user.append(user)
            valid_time.append(user_time[train_thr:valid_thr])
            valid_coord.append(user_coord[train_thr:valid_thr])
            valid_loc.append(user_loc[train_thr:valid_thr])
            test_user.append(user)
            test_time.append(user_time[valid_thr:])
            test_coord.append(user_coord[valid_thr:])
            test_loc.append(user_loc[valid_thr:])

            prev_user = user
            user_time = [time]
            user_coord = [coord]
            user_loc = [location]

    # process also the latest user in the for loop
    if user2id.get(user) is not None:
        train_thr = int(len(user_time) * 0.7)
        valid_thr = int(len(user_time) * 0.8)
        train_user.append(user)
        train_time.append(user_time[:train_thr])
        train_coord.append(user_coord[:train_thr])
        train_loc.append(user_loc[:train_thr])
        valid_user.append(user)
        valid_time.append(user_time[train_thr:valid_thr])
        valid_coord.append(user_coord[train_thr:valid_thr])
        valid_loc.append(user_loc[train_thr:valid_thr])
        test_user.append(user)
        test_time.append(user_time[valid_thr:])
        test_coord.append(user_coord[valid_thr:])
        test_loc.append(user_loc[valid_thr:])

    return len(user2id), poi2id, poi2pos, train_user, train_time, train_coord, train_loc, valid_user, valid_time, valid_coord, valid_loc, test_user, test_time, test_coord, test_loc

def inner_iter(data, batch_size):
    data_size = len(data)
    num_batches = int(len(data)/batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
