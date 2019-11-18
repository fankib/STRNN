#! /usr/bin/env python
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt

import data_loader
import argparse

# Args:
parser = argparse.ArgumentParser()
parser.add_argument('--suffix', default='50', type=str, help='the suffix of the files to load')
args = parser.parse_args()

# Data loading params
poi2pos_file = './prepro_poi2pos_%s.txt'%args.suffix
train_file = "./prepro_train_%s.txt"%args.suffix
valid_file = "./prepro_valid_%s.txt"%args.suffix
test_file = "./prepro_test_%s.txt"%args.suffix

# Load data
print("Loading data...")
poi2pos = data_loader.load_poi2pos(poi2pos_file)
train_user, train_td, train_ld, train_loc, train_dst = data_loader.treat_prepro(train_file, step=1)
valid_user, valid_td, valid_ld, valid_loc, valid_dst = data_loader.treat_prepro(valid_file, step=2)
test_user, test_td, test_ld, test_loc, test_dst = data_loader.treat_prepro(test_file, step=3)

# compute the flat earth and visualize locations:
def show_flat_earch(locations_uv):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    
    X = []
    Y = []
    for key in locations_uv:
        lat, lon  = locations_uv[key]
        lat_r, lon_r = lat*np.pi/180., lon*np.pi/180.
        X.append(lat_r)
        Y.append(lon_r)
    ax.scatter(Y, X, color='r', s=1)
    plt.xlim((-np.pi, np.pi))
    plt.ylim((-np.pi/2, np.pi/2))
    
    plt.savefig('visualization/gowalla_flat_earth.png')
    #plt.show()
    
    

# compute earth and visualize locations
def show_earth(locations_xyz):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    
    # draw sphere
    R = 6371*0.99
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = R * np.cos(u)*np.sin(v)
    y = R * np.sin(u)*np.sin(v)
    z = R * np.cos(v)
    ax.plot_surface(x, y, z, color='w')
    #ax.plot_wireframe(x, y, z, color="b")
    
    # draw points:
    X = []
    Y = []
    Z = []
    for key in locations_xyz:
        (x,y,z) = locations_xyz[key]
        X.append(x)
        Y.append(y)
        Z.append(z)
    ax.scatter(X, Y, Z, color="r", s=1)
    
    # rotate the axes and update
    for angle in np.arange(0, 360, 20):
        ax.view_init(30, angle)
        #plt.savefig('visualization/gowalla_earth_{}.png'.format(angle))
        #plt.draw()        
        plt.pause(.001)
    

# compute historgram on temporal delays (using training set)
def show_stats(user, distances, x_lim, bins, xlabel, suffix):
    ds = []
    for i in range(len(user)):
        for j in range(len(distances[i])): # skip 0
            for k in range(0, len(distances[i][j])):
                ds.append(distances[i][j][k])
    ds = np.array(ds)
    print('measurements: ', ds.shape)
    print('median: ', np.median(ds))
    print('mean: ', np.mean(ds))
    print('min: ', np.min(ds))
    print('max: ', np.max(ds))

    # visualize histogram:
    plt.hist(ds, color = 'blue', edgecolor = 'black', bins = bins)
    plt.title('Histogram of Distances')
    plt.xlabel(xlabel)
    plt.ylabel('Amount')
    plt.xlim((0, x_lim))
    #plt.savefig('visualization/gowalla_distance_close_{}.png'.format(suffix))
    plt.show()  
 
#visualize locations 
show_earth(poi2pos) 

# visualize training data:
show_stats(train_user, train_td, 600, 50000, 'Delay in minutes', 'temporal_{}'.format(args.suffix))
show_stats(train_user, train_ld, 75, 5000, 'L2 in km', 'spatial_{}'.format(args.suffix))






