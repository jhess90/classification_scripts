#!/usr/bin/env python

#import packages
import scipy.io as sio
import h5py
import numpy as np
import pdb
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import sys
import xlsxwriter
import glob
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from matplotlib import cm
import xlsxwriter
import scipy.stats as stats
from scipy import ndimage
from math import isinf
import os
from scipy.stats.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
import math
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

########################
# params to set ########
########################

plot_bool = False
plot_hist_bool = True

bin_size = 10 #in ms
time_before = -0.5 #negative value
time_after = 1.0
res_window = [0.3,0.8]

run_pop_bool = False

#########################
# functions  ############
#########################


#########################
# run ###################
#########################

all_dict = np.load('v2_backup_cv.npy')[()]

for region_key,region_val in all_dict.iteritems():
    r_levels_succ_corr = np.zeros((5,2))
    p_levels_succ_corr = np.zeros((5,2))
    r_bin_succ_corr = np.zeros((5,2))
    p_bin_succ_corr = np.zeros((5,2))
    comb_succ_corr = np.zeros((5,2))

    for win_key,win_val in all_dict[region_key]['avg_and_corr'].iteritems():
        r_p_all = all_dict[region_key]['avg_and_corr'][win_key]['r_p_all']

        r_levels = r_p_all[:,0]
        p_levels = r_p_all[:,1]
        
        succ_targs = np.where(r_p_all[:,2] > 0, 1,0)

        r_nr_targs = np.where(r_p_all[:,0] > 0,1,0)
        p_np_targs = np.where(r_p_all[:,1] > 0,1,0)

        comb = np.zeros((np.shape(r_p_all)[0]))
        for i in range(np.shape(r_nr_targs)[0]):
            if r_nr_targs[i] == 0 and p_np_targs[i] == 0:
                comb[i] = 0
            elif r_nr_targs[i] == 1 and p_np_targs[i] == 0:
                comb[i] = 1
            elif r_nr_targs[i] == 0 and p_np_targs[i] == 1:
                comb[i] = 2
            elif r_nr_targs[i] == 1 and p_np_targs[i] == 1:
                comb[i] = 3

        r_bin_sf_targ = np.zeros((np.shape(r_p_all)[0]))
        for i in range(np.shape(r_nr_targs)[0]):
            if r_nr_targs[i] == 0 and succ_targs[i] != 1:
                r_bin_sf_targ[i] = 0
            elif r_nr_targs[i] == 1 and succ_targs[i] != 1:
                r_bin_sf_targ[i] = 1
            elif r_nr_targs[i] == 0 and succ_targs[i] == 1:
                r_bin_sf_targ[i] = 2
            elif r_nr_targs[i] == 1 and succ_targs[i] == 1:
                r_bin_sf_targ[i] = 3

        p_bin_sf_targ = np.zeros((np.shape(r_p_all)[0]))
        for i in range(np.shape(p_np_targs)[0]):
            if p_np_targs[i] == 0 and succ_targs[i] != 1:
                p_bin_sf_targ[i] = 0
            elif p_np_targs[i] == 1 and succ_targs[i] != 1:
                p_bin_sf_targ[i] = 1
            elif p_np_targs[i] == 0 and succ_targs[i] == 1:
                p_bin_sf_targ[i] = 2
            elif p_np_targs[i] == 1 and succ_targs[i] == 1:
                p_bin_sf_targ[i] = 3

        r_levels_succ = pearsonr(r_levels,succ_targs)
        p_levels_succ = pearsonr(p_levels,succ_targs)

        r_bin_succ = pearsonr(r_nr_targs,succ_targs)
        p_bin_succ = pearsonr(p_np_targs,succ_targs)
        
        comb_succ = pearsonr(comb,succ_targs)

        
        if win_key == 'aft_cue':
            r_levels_succ_corr[0,:] = r_levels_succ
            p_levels_succ_corr[0,:] = p_levels_succ
            r_bin_succ_corr[0,:] = r_bin_succ
            p_bin_succ_corr[0,:] = p_bin_succ
            comb_succ_corr[0,:] = comb_succ
        elif win_key == 'bfr_res':
            r_levels_succ_corr[1,:] = r_levels_succ
            p_levels_succ_corr[1,:] = p_levels_succ
            r_bin_succ_corr[1,:] = r_bin_succ
            p_bin_succ_corr[1,:] = p_bin_succ
            comb_succ_corr[1,:] = comb_succ
        elif win_key == 'aft_res':
            r_levels_succ_corr[2,:] = r_levels_succ
            p_levels_succ_corr[2,:] = p_levels_succ
            r_bin_succ_corr[2,:] = r_bin_succ
            p_bin_succ_corr[2,:] = p_bin_succ
            comb_succ_corr[2,:] = comb_succ
        elif win_key == 'res_win':
            r_levels_succ_corr[3,:] = r_levels_succ
            p_levels_succ_corr[3,:] = p_levels_succ
            r_bin_succ_corr[3,:] = r_bin_succ
            p_bin_succ_corr[3,:] = p_bin_succ
            comb_succ_corr[3,:] = comb_succ
        elif win_key == 'concat':
            r_levels_succ_corr[4,:] = r_levels_succ
            p_levels_succ_corr[4,:] = p_levels_succ
            r_bin_succ_corr[4,:] = r_bin_succ
            p_bin_succ_corr[4,:] = p_bin_succ
            comb_succ_corr[4,:] = comb_succ

        corr_dict = {'r_levels_succ_corr':r_levels_succ_corr,'p_levels_suc_corr':p_levels_succ_corr,'r_bin_succ_corr':r_bin_succ_corr,'p_bin_succ_corr':p_bin_succ_corr,'comb_succ_corr':comb_succ_corr}
