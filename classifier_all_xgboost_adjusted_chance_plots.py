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


#########################
# functions  ############
#########################

######run

data_dict_all = np.load('v2_backup_cv.npy')[()]


for region_key,region_val in data_dict_all.iteritems():
    print 'region: %s' %(region_key)
    
    for win_key,win_val in data_dict_all[region_key]['avg_and_corr'].iteritems():

        accuracy_spread_r_levels = data_dict_all[region_key]['accuracy_spread']['r_levels'][win_key]
        accuracy_spread_p_levels = data_dict_all[region_key]['accuracy_spread']['p_levels'][win_key]
        accuracy_spread_sf = data_dict_all[region_key]['accuracy_spread']['sf'][win_key]
        accuracy_spread_comb = data_dict_all[region_key]['accuracy_spread']['comb'][win_key]

        accuracy_spread_r_bin = data_dict_all[region_key]['accuracy_spread']['r_bin'][win_key]
        accuracy_spread_p_bin = data_dict_all[region_key]['accuracy_spread']['p_bin'][win_key]
        accuracy_spread_r_binsf = data_dict_all[region_key]['accuracy_spread']['r_bin_sf'][win_key]
        accuracy_spread_p_binsf = data_dict_all[region_key]['accuracy_spread']['p_bin_sf'][win_key]

        accuracy_list_total = data_dict_all[region_key]['classification_output'][win_key]


        #chance2_all_2 = np.zeros((data_dict_all[region_key]['avg_and_corr']['aft_cue']['total_unit_num']))

        #calc adjusted chance level for each block (then assign to each unit in that bloc, for 2 and 4
        alpha = 0.05
        num_classes2 = 2.0
        num_classes4 = 4.0
        
        for i in range(np.shape(data_dict_all['M1_dicts']['avg_and_corr']['aft_cue']['avg_fr_dict'].keys())[0]):

            [num_trials,num_units] = np.shape(data_dict_all[region_key]['avg_and_corr']['aft_cue']['avg_fr_dict'][i])
            num_trials = float(num_trials)
            chance2_all_2_temp = stats.binom.ppf(1-alpha,num_trials,1/num_classes2) / num_trials #not multiply by 100 b/ using probability not percentage
            chance2_all_4_temp = stats.binom.ppf(1-alpha,num_trials,1/num_classes4) / num_trials 
            
            if i == 0:
                chance2_all_2 = np.full(num_units,chance2_all_2_temp)
                chance2_all_4 = np.full(num_units,chance2_all_4_temp)
            else:
                chance2_all_2 = np.append(chance2_all_2,np.full(num_units,chance2_all_2_temp))
                chance2_all_4 = np.append(chance2_all_4,np.full(num_units,chance2_all_4_temp))

        #######################
        #accuracy spread plots
 
        #
        f,axarr = plt.subplots(2,sharex=True)

        f.suptitle('unit accuracies: %s %s' %(region_key,win_key))

        srt = np.mean(accuracy_spread_r_levels,axis=1).argsort()
        srt2 = np.mean(accuracy_spread_p_levels,axis=1).argsort()

        axarr[0].errorbar(np.mean(accuracy_spread_r_levels,axis=1)[srt],range(np.shape(accuracy_spread_r_levels)[0]),xerr = stats.sem(accuracy_spread_r_levels,axis=1)[srt],ecolor='dimgrey',elinewidth=0.25,marker='.',capsize=0.0,linewidth=0.0,mec='darkcyan',mfc='darkcyan',mew=0.5)
        axarr[0].set_xlim([0,1])
        axarr[0].axvline(accuracy_list_total[:,1][0],color='grey',linestyle='solid',linewidth=0.75)
        axarr[0].scatter(chance2_all_4[srt],range(np.shape(chance2_all_4)[0]),color='indianred',marker="|",linewidth=0.6,alpha=0.9,s=.75)
        axarr[0].set_ylim([-2,np.shape(chance2_all_4)[0]+2])

        axarr[0].set_title('reward level accuracy')
        axarr[0].yaxis.set_major_locator(plt.MaxNLocator(3))

        axarr[1].errorbar(np.mean(accuracy_spread_p_levels,axis=1)[srt2],range(np.shape(accuracy_spread_p_levels)[0]),xerr = stats.sem(accuracy_spread_p_levels,axis=1)[srt2],ecolor='dimgrey',elinewidth=0.25,marker='.',capsize=0.0,linewidth=0.0,mec='darkcyan',mfc='darkcyan',mew=0.5)
        axarr[1].axvline(accuracy_list_total[:,4][0],color='grey',linestyle='solid',linewidth=0.75)
        axarr[1].scatter(chance2_all_4[srt2],range(np.shape(chance2_all_4)[0]),color='indianred',marker="|",linewidth=0.6,alpha=0.9,s=0.75)
        axarr[1].set_ylim([-2,np.shape(chance2_all_4)[0]+2])
        axarr[1].yaxis.set_major_locator(plt.MaxNLocator(3))
        axarr[1].set_title('punishment level accuracy')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('chance2_v2_accuracy_spread_1_%s_%s' %(region_key,win_key))
        plt.clf()

        #
        f,axarr = plt.subplots(2,sharex=True)

        f.suptitle('unit accuracies: %s %s' %(region_key,win_key))
        srt = np.mean(accuracy_spread_sf,axis=1).argsort()
        srt2 = np.mean(accuracy_spread_comb,axis=1).argsort()

        axarr[0].errorbar(np.mean(accuracy_spread_sf,axis=1)[srt],range(np.shape(accuracy_spread_sf)[0]),xerr = stats.sem(accuracy_spread_sf,axis=1)[srt],ecolor='dimgrey',elinewidth=0.25,marker='.',capsize=0.0,linewidth=0.0,mec='darkcyan',mfc='darkcyan',mew=0.5)
        axarr[0].set_xlim([0,1])
        axarr[0].axvline(accuracy_list_total[:,7][0],color='grey',linestyle='solid',linewidth=0.75)
        axarr[0].scatter(chance2_all_2[srt],range(np.shape(chance2_all_2)[0]),color='indianred',marker="|",linewidth=0.6,alpha=0.9,s=.75)
        axarr[0].set_ylim([-2,np.shape(chance2_all_4)[0]+2])
        axarr[0].yaxis.set_major_locator(plt.MaxNLocator(3))
        axarr[0].set_title('succ/fail accuracy')
        axarr[1].errorbar(np.mean(accuracy_spread_comb,axis=1)[srt2],range(np.shape(accuracy_spread_comb)[0]),xerr = stats.sem(accuracy_spread_comb,axis=1)[srt2],ecolor='dimgrey',elinewidth=0.25,marker='.',capsize=0.0,linewidth=0.0,mec='darkcyan',mfc='darkcyan',mew=0.5)
        axarr[1].scatter(chance2_all_4[srt2],range(np.shape(chance2_all_4)[0]),color='indianred',marker="|",linewidth=0.6,alpha=0.9,s=.75)
        axarr[1].set_ylim([-2,np.shape(chance2_all_4)[0]+2])
        axarr[1].axvline(accuracy_list_total[:,10][0],color='grey',linestyle='solid',linewidth=0.75)
        axarr[1].yaxis.set_major_locator(plt.MaxNLocator(3))
        axarr[1].set_title('comb accuracy')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('chance2_v2_accuracy_spread_2_%s_%s' %(region_key,win_key))
        plt.clf()

        #
        f,axarr = plt.subplots(2,sharex=True)

        f.suptitle('unit accuracies: %s %s' %(region_key,win_key))
        srt = np.mean(accuracy_spread_r_bin,axis=1).argsort()
        srt2 = np.mean(accuracy_spread_p_bin,axis=1).argsort()

        axarr[0].errorbar(np.mean(accuracy_spread_r_bin,axis=1)[srt],range(np.shape(accuracy_spread_r_bin)[0]),xerr = stats.sem(accuracy_spread_r_bin,axis=1)[srt],ecolor='dimgrey',elinewidth=0.25,marker='.',capsize=0.0,linewidth=0.0,mec='darkcyan',mfc='darkcyan',mew=0.5)
        axarr[0].set_xlim([0,1])
        axarr[0].axvline(accuracy_list_total[:,13][0],color='grey',linestyle='solid',linewidth=0.75)
        axarr[0].scatter(chance2_all_2[srt],range(np.shape(chance2_all_2)[0]),color='indianred',marker="|",linewidth=0.6,alpha=0.9,s=.75)
        axarr[0].set_ylim([-2,np.shape(chance2_all_4)[0]+2])
        axarr[0].yaxis.set_major_locator(plt.MaxNLocator(3))
        axarr[0].set_title('reward binary accuracy')
        axarr[1].errorbar(np.mean(accuracy_spread_p_bin,axis=1)[srt2],range(np.shape(accuracy_spread_p_bin)[0]),xerr = stats.sem(accuracy_spread_p_bin,axis=1)[srt2],ecolor='dimgrey',elinewidth=0.25,marker='.',capsize=0.0,linewidth=0.0,mec='darkcyan',mfc='darkcyan',mew=0.5)
        axarr[1].axvline(accuracy_list_total[:,16][0],color='grey',linestyle='solid',linewidth=0.75)
        axarr[1].scatter(chance2_all_2[srt2],range(np.shape(chance2_all_2)[0]),color='indianred',marker="|",linewidth=0.6,alpha=0.9,s=.75)
        axarr[1].set_ylim([-2,np.shape(chance2_all_4)[0]+2])
        axarr[1].yaxis.set_major_locator(plt.MaxNLocator(3))
        axarr[1].set_title('punishment binary accuracy')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('chance2_v2_accuracy_spread_3_%s_%s' %(region_key,win_key))
        plt.clf()
        #
        f,axarr = plt.subplots(2,sharex=True)

        f.suptitle('unit accuracies: %s %s' %(region_key,win_key))
        srt = np.mean(accuracy_spread_r_binsf,axis=1).argsort()
        srt2 = np.mean(accuracy_spread_p_binsf,axis=1).argsort()

        axarr[0].errorbar(np.mean(accuracy_spread_r_binsf,axis=1)[srt],range(np.shape(accuracy_spread_r_binsf)[0]),xerr = stats.sem(accuracy_spread_r_binsf,axis=1)[srt],ecolor='dimgrey',elinewidth=0.25,marker='.',capsize=0.0,linewidth=0.0,mec='darkcyan',mfc='darkcyan',mew=0.5)
        axarr[0].set_xlim([0,1]) 
        axarr[0].axvline(accuracy_list_total[:,19][0],color='grey',linestyle='solid',linewidth=0.75)
        axarr[0].scatter(chance2_all_4[srt],range(np.shape(chance2_all_4)[0]),color='indianred',marker="|",linewidth=0.6,alpha=0.9,s=.75)
        axarr[0].set_ylim([-2,np.shape(chance2_all_4)[0]+2])
        axarr[0].yaxis.set_major_locator(plt.MaxNLocator(3))
        axarr[0].set_title('reward bin sf accuracy')
        axarr[1].errorbar(np.mean(accuracy_spread_p_binsf,axis=1)[srt2],range(np.shape(accuracy_spread_p_binsf)[0]),xerr = stats.sem(accuracy_spread_p_binsf,axis=1)[srt2],ecolor='dimgrey',elinewidth=0.25,marker='.',capsize=0.0,linewidth=0.0,mec='darkcyan',mfc='darkcyan',mew=0.5)
        axarr[1].axvline(accuracy_list_total[:,22][0],color='grey',linestyle='solid',linewidth=0.75)
        axarr[1].scatter(chance2_all_4[srt2],range(np.shape(chance2_all_4)[0]),color='indianred',marker="|",linewidth=0.6,alpha=0.9,s=.75)
        axarr[1].set_ylim([-2,np.shape(chance2_all_4)[0]+2])
        axarr[1].yaxis.set_major_locator(plt.MaxNLocator(3))
        axarr[1].set_title('punishment bin sf accuracy')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('chance2_v2_accuracy_spread_4_%s_%s' %(region_key,win_key))
        plt.clf()
 
