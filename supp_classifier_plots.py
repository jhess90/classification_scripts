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


def is_outlier(points, thresh = 4.0):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


######################
data_dict_all = np.load('backup_cv.npy')[()]



for region_key,region_val in data_dict_all.iteritems():
    print 'region: %s' %(region_key)
    
    for win_key,win_val in data_dict_all[region_key]['avg_and_corr'].iteritems():

        accuracy_list_total = data_dict_all[region_key]['classification_output'][win_key]
        
        print 'window: %s' %(win_key)

        f,axarr = plt.subplots(2,sharex=True)

        f.suptitle('unit accuracies: %s %s' %(region_key,win_key))

        axarr[0] = 





        '''
        f,axarr = plt.subplots(4,sharex=True)

        f.suptitle('unit accuracy distribution: %s %s' %(region_key,win_key))
        axarr[0].hist(accuracy_list_total[:,0],color='mediumturquoise',lw=0)
        axarr[0].axvline(accuracy_list_total[:,1][0],color='k',linestyle='dashed',linewidth=1)
        axarr[0].set_xlim([0,1])
        axarr[0].set_title('reward level accuracy')
        axarr[1].hist(accuracy_list_total[:,2],color='darkviolet',lw=0)
        axarr[1].axvline(accuracy_list_total[:,1][0],color='k',linestyle='dashed',linewidth=1)
        axarr[1].set_title('reward level accuracy: shuffled')

        axarr[2].hist(accuracy_list_total[:,3],color='mediumturquoise',lw=0)
        axarr[2].axvline(accuracy_list_total[:,4][0],color='k',linestyle='dashed',linewidth=1)
        axarr[2].set_title('punishment level accuracy')
        axarr[3].hist(accuracy_list_total[:,5],color='darkviolet',lw=0)
        axarr[3].axvline(accuracy_list_total[:,4][0],color='k',linestyle='dashed',linewidth=1)
        axarr[3].set_title('punishment level accuracy: shuffled')

        for axi in axarr.reshape(-1):
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('unit_accuracy_hist_xgboost_cv_1_%s_%s' %(region_key,win_key))
        plt.clf()

        #
        f,axarr = plt.subplots(4,sharex=True)

        f.suptitle('unit accuracy distribution 2: %s %s' %(region_key,win_key))
        axarr[0].hist(accuracy_list_total[:,6],color='mediumturquoise',lw=0)
        axarr[0].axvline(accuracy_list_total[:,7][0],color='k',linestyle='dashed',linewidth=1)
        axarr[0].set_title('succ/fail accuracy')
        axarr[1].hist(accuracy_list_total[:,8],color='darkviolet',lw=0)
        axarr[1].axvline(accuracy_list_total[:,7][0],color='k',linestyle='dashed',linewidth=1)
        axarr[1].set_title('succ/fail accuracy: shuffled')

        axarr[2].hist(accuracy_list_total[:,9],color='mediumturquoise',lw=0)
        axarr[2].axvline(accuracy_list_total[:,10][0],color='k',linestyle='dashed',linewidth=1)
        axarr[2].set_title('comb accuracy')
        axarr[3].hist(accuracy_list_total[:,11],color='darkviolet',lw=0)
        axarr[3].axvline(accuracy_list_total[:,10][0],color='k',linestyle='dashed',linewidth=1)
        axarr[3].set_title('comb accuracy: shuffled')

        for axi in axarr.reshape(-1):
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('unit_accuracy_hist_xgboost_cv_2_%s_%s' %(region_key,win_key))
        plt.clf()

        #
        f,axarr = plt.subplots(4,sharex=True)

        f.suptitle('unit accuracy distribution 3: %s %s' %(region_key,win_key))
        axarr[0].hist(accuracy_list_total[:,12],color='mediumturquoise',lw=0)
        axarr[0].axvline(accuracy_list_total[:,13][0],color='k',linestyle='dashed',linewidth=1)
        axarr[0].set_xlim([0,1])
        axarr[0].set_title('reward binary accuracy')
        axarr[1].hist(accuracy_list_total[:,14],color='darkviolet',lw=0)
        axarr[1].axvline(accuracy_list_total[:,13][0],color='k',linestyle='dashed',linewidth=1)
        axarr[1].set_title('reward binary accuracy: shuffled')

        axarr[2].hist(accuracy_list_total[:,15],color='mediumturquoise',lw=0)
        axarr[2].axvline(accuracy_list_total[:,16][0],color='k',linestyle='dashed',linewidth=1)
        axarr[2].set_title('punishment binary accuracy')
        axarr[3].hist(accuracy_list_total[:,17],color='darkviolet',lw=0)
        axarr[3].axvline(accuracy_list_total[:,16][0],color='k',linestyle='dashed',linewidth=1)
        axarr[3].set_title('punishment binary accuracy: shuffled')

        for axi in axarr.reshape(-1):
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('unit_accuracy_hist_xgboost_cv_3_%s_%s' %(region_key,win_key))
        plt.clf()

        '''
        #
        #f,axarr = plt.subplots(4,sharex=True)

        #f.suptitle('unit accuracy distribution 4: %s %s' %(region_key,win_key))

        #axarr[0].hist(accuracy_list_total[:,18],color='mediumturquoise',lw=0)
        #axarr[0].axvline(accuracy_list_total[:,19][0],color='k',linestyle='dashed',linewidth=1)
        #axarr[0].set_xlim([0,1])
        #axarr[0].set_title('reward binary succ fail accuracy')
        #axarr[1].hist(accuracy_list_total[:,20],color='darkviolet',lw=0)
        #axarr[1].axvline(accuracy_list_total[:,19][0],color='k',linestyle='dashed',linewidth=1)
        #axarr[1].set_title('reward binary succ fail accuracy: shuffled')

        #axarr[2].hist(accuracy_list_total[:,21],color='mediumturquoise',lw=0)
        #axarr[2].axvline(accuracy_list_total[:,22][0],color='k',linestyle='dashed',linewidth=1)
        #axarr[2].set_title('punishment binary succ fail accuracy')
        #axarr[3].hist(accuracy_list_total[:,23],color='darkviolet',lw=0)
        #axarr[3].axvline(accuracy_list_total[:,22][0],color='k',linestyle='dashed',linewidth=1)
        #axarr[3].set_title('punishment binary succ fail accuracy: shuffled')

        #for axi in axarr.reshape(-1):
        #    axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        #plt.tight_layout()
        #plt.subplots_adjust(top=0.9)
        #plt.savefig('unit_accuracy_hist_xgboost_cv_4_supp_%s_%s' %(region_key,win_key))
        #plt.clf()

        pdb.set_trace()








#################

#if xlsx currently exists delte, or else won't write properly
if os.path.isfile('xgb_output_cv.xlsx'):
    os.remove('xgb_output_cv.xlsx')

type_names = ['aft cue','bfr res','aft res','res win','concat']
comb_names = ['r levels','p levels','succ/fail','comb','r bin','p bin','r bin sf','p bin sf']

accuracy_workbook = xlsxwriter.Workbook('xgb_output_cv.xlsx',options={'nan_inf_to_errors':True})
worksheet = accuracy_workbook.add_worksheet('unit_accuracy')

# > chance and > shuff

temp = data_dict_all['M1_dicts']['classification_output']

r_level_accuracies = [np.sum(temp['aft_cue'][:,0] > temp['aft_cue'][:,2])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] > temp['bfr_res'][:,2])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] > temp['aft_res'][:,2])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] > temp['res_win'][:,2])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] > temp['concat'][:,2])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,3] > temp['aft_cue'][:,5])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] > temp['bfr_res'][:,5])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] > temp['aft_res'][:,5])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] > temp['res_win'][:,5])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] > temp['concat'][:,5])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,6] > temp['aft_cue'][:,8])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] > temp['bfr_res'][:,8])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] > temp['aft_res'][:,8])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] > temp['res_win'][:,8])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] > temp['concat'][:,8])/float(np.shape(temp['concat'])[0])]

comb_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,11])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,11])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,11])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,11])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,11])/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,12] > temp['aft_cue'][:,14])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,12] > temp['bfr_res'][:,14])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,12] > temp['aft_res'][:,14])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,12] > temp['res_win'][:,14])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,12] > temp['concat'][:,14])/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,15] > temp['aft_cue'][:,17])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,15] > temp['bfr_res'][:,17])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,15] > temp['aft_res'][:,17])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,15] > temp['res_win'][:,17])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,15] > temp['concat'][:,17])/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies =[np.sum(temp['aft_cue'][:,18] > temp['aft_cue'][:,20])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,18] > temp['bfr_res'][:,20])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,18] > temp['aft_res'][:,20])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,18] > temp['res_win'][:,20])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,18] > temp['concat'][:,20])/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,21] > temp['aft_cue'][:,23])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,21] > temp['bfr_res'][:,23])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,21] > temp['aft_res'][:,23])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,21] > temp['res_win'][:,23])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,21] > temp['concat'][:,23])/float(np.shape(temp['concat'])[0])]

worksheet.write(0,0,'M1: percent better than shuffled')
worksheet.write_row(0,1,comb_names)
worksheet.write_column(1,0,type_names)
worksheet.write_column(1,1,r_level_accuracies)
worksheet.write_column(1,2,p_level_accuracies)
worksheet.write_column(1,3,succ_accuracies)
worksheet.write_column(1,4,comb_accuracies)
worksheet.write_column(1,5,r_bin_accuracies)
worksheet.write_column(1,6,p_bin_accuracies)
worksheet.write_column(1,7,r_bin_sf_accuracies)
worksheet.write_column(1,8,p_bin_sf_accuracies)

r_level_accuracies = [np.sum(temp['aft_cue'][:,0] > temp['aft_cue'][:,1])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] > temp['bfr_res'][:,1])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] > temp['aft_res'][:,1])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] > temp['res_win'][:,1])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] > temp['concat'][:,1])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,3] > temp['aft_cue'][:,4])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] > temp['bfr_res'][:,4])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] > temp['aft_res'][:,4])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] > temp['res_win'][:,4])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] > temp['concat'][:,4])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,6] > temp['aft_cue'][:,7])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] > temp['bfr_res'][:,7])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] > temp['aft_res'][:,7])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] > temp['res_win'][:,7])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] > temp['concat'][:,7])/float(np.shape(temp['concat'])[0])]

comb_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,10])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,10])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,10])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,10])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,10])/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,12] > temp['aft_cue'][:,13])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,12] > temp['bfr_res'][:,13])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,12] > temp['aft_res'][:,13])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,12] > temp['res_win'][:,13])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,12] > temp['concat'][:,13])/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,15] > temp['aft_cue'][:,16])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,15] > temp['bfr_res'][:,16])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,15] > temp['aft_res'][:,16])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,15] > temp['res_win'][:,16])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,15] > temp['concat'][:,16])/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies =[np.sum(temp['aft_cue'][:,18] > temp['aft_cue'][:,19])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,18] > temp['bfr_res'][:,19])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,18] > temp['aft_res'][:,19])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,18] > temp['res_win'][:,19])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,18] > temp['concat'][:,19])/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,21] > temp['aft_cue'][:,22])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,21] > temp['bfr_res'][:,22])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,21] > temp['aft_res'][:,22])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,21] > temp['res_win'][:,22])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,21] > temp['concat'][:,22])/float(np.shape(temp['concat'])[0])]

worksheet.write(0,10,'M1: percent better than chance')
worksheet.write_row(0,11,comb_names)
worksheet.write_column(1,10,type_names)
worksheet.write_column(1,11,r_level_accuracies)
worksheet.write_column(1,12,p_level_accuracies)
worksheet.write_column(1,13,succ_accuracies)
worksheet.write_column(1,14,comb_accuracies)
worksheet.write_column(1,15,r_bin_accuracies)
worksheet.write_column(1,16,p_bin_accuracies)
worksheet.write_column(1,17,r_bin_sf_accuracies)
worksheet.write_column(1,18,p_bin_sf_accuracies)


######

temp = data_dict_all['M1_dicts']['classification_output_p_val']
r_level_accuracies = [np.sum(temp['aft_cue'][:,0] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] < 0.05)/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,1] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,1] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,1] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,1] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,1] < 0.05)/float(np.shape(temp['concat'])[0])]
succ_accuraracies = [np.sum(temp['aft_cue'][:,2] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,2] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,2] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,2] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,2] < 0.05)/float(np.shape(temp['concat'])[0])]
comb_accuracies = [np.sum(temp['aft_cue'][:,3] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] < 0.05)/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,4] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,4] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,4] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,4] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,4] < 0.05)/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,5] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,5] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,5] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,5] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,5] < 0.05)/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,6] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] < 0.05)/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,7] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,7] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,7] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,7] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,7] < 0.05)/float(np.shape(temp['concat'])[0])]

worksheet.write(0,20,'M1: percent sig diff')
worksheet.write_row(0,21,comb_names)
worksheet.write_column(1,20,type_names)
worksheet.write_column(1,21,r_level_accuracies)
worksheet.write_column(1,22,p_level_accuracies)
worksheet.write_column(1,23,succ_accuracies)
worksheet.write_column(1,24,comb_accuracies)
worksheet.write_column(1,25,r_bin_accuracies)
worksheet.write_column(1,26,p_bin_accuracies)
worksheet.write_column(1,27,r_bin_sf_accuracies)
worksheet.write_column(1,28,p_bin_sf_accuracies)


##
temp = data_dict_all['S1_dicts']['classification_output']

r_level_accuracies = [np.sum(temp['aft_cue'][:,0] > temp['aft_cue'][:,2])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] > temp['bfr_res'][:,2])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] > temp['aft_res'][:,2])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] > temp['res_win'][:,2])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] > temp['concat'][:,2])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,3] > temp['aft_cue'][:,5])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] > temp['bfr_res'][:,5])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] > temp['aft_res'][:,5])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] > temp['res_win'][:,5])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] > temp['concat'][:,5])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,6] > temp['aft_cue'][:,8])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] > temp['bfr_res'][:,8])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] > temp['aft_res'][:,8])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] > temp['res_win'][:,8])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] > temp['concat'][:,8])/float(np.shape(temp['concat'])[0])]

comb_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,11])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,11])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,11])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,11])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,11])/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,12] > temp['aft_cue'][:,14])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,12] > temp['bfr_res'][:,14])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,12] > temp['aft_res'][:,14])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,12] > temp['res_win'][:,14])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,12] > temp['concat'][:,14])/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,15] > temp['aft_cue'][:,17])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,15] > temp['bfr_res'][:,17])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,15] > temp['aft_res'][:,17])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,15] > temp['res_win'][:,17])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,15] > temp['concat'][:,17])/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies =[np.sum(temp['aft_cue'][:,18] > temp['aft_cue'][:,20])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,18] > temp['bfr_res'][:,20])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,18] > temp['aft_res'][:,20])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,18] > temp['res_win'][:,20])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,18] > temp['concat'][:,20])/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,21] > temp['aft_cue'][:,23])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,21] > temp['bfr_res'][:,23])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,21] > temp['aft_res'][:,23])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,21] > temp['res_win'][:,23])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,21] > temp['concat'][:,23])/float(np.shape(temp['concat'])[0])]


worksheet.write(8,0,'S1: percent better than shuffled')
worksheet.write_row(8,1,comb_names)
worksheet.write_column(9,0,type_names)
worksheet.write_column(9,1,r_level_accuracies)
worksheet.write_column(9,2,p_level_accuracies)
worksheet.write_column(9,3,succ_accuracies)
worksheet.write_column(9,4,comb_accuracies)
worksheet.write_column(9,5,r_bin_accuracies)
worksheet.write_column(9,6,p_bin_accuracies)
worksheet.write_column(9,7,r_bin_sf_accuracies)
worksheet.write_column(9,8,p_bin_sf_accuracies)

r_level_accuracies = [np.sum(temp['aft_cue'][:,0] > temp['aft_cue'][:,1])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] > temp['bfr_res'][:,1])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] > temp['aft_res'][:,1])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] > temp['res_win'][:,1])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] > temp['concat'][:,1])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,3] > temp['aft_cue'][:,4])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] > temp['bfr_res'][:,4])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] > temp['aft_res'][:,4])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] > temp['res_win'][:,4])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] > temp['concat'][:,4])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,6] > temp['aft_cue'][:,7])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] > temp['bfr_res'][:,7])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] > temp['aft_res'][:,7])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] > temp['res_win'][:,7])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] > temp['concat'][:,7])/float(np.shape(temp['concat'])[0])]

comb_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,10])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,10])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,10])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,10])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,10])/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,12] > temp['aft_cue'][:,13])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,12] > temp['bfr_res'][:,13])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,12] > temp['aft_res'][:,13])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,12] > temp['res_win'][:,13])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,12] > temp['concat'][:,13])/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,15] > temp['aft_cue'][:,16])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,15] > temp['bfr_res'][:,16])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,15] > temp['aft_res'][:,16])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,15] > temp['res_win'][:,16])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,15] > temp['concat'][:,16])/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies =[np.sum(temp['aft_cue'][:,18] > temp['aft_cue'][:,19])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,18] > temp['bfr_res'][:,19])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,18] > temp['aft_res'][:,19])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,18] > temp['res_win'][:,19])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,18] > temp['concat'][:,19])/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,21] > temp['aft_cue'][:,22])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,21] > temp['bfr_res'][:,22])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,21] > temp['aft_res'][:,22])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,21] > temp['res_win'][:,22])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,21] > temp['concat'][:,22])/float(np.shape(temp['concat'])[0])]

worksheet.write(8,10,'S1: percent better than chance')
worksheet.write_row(8,11,comb_names)
worksheet.write_column(9,10,type_names)
worksheet.write_column(9,11,r_level_accuracies)
worksheet.write_column(9,12,p_level_accuracies)
worksheet.write_column(9,13,succ_accuracies)
worksheet.write_column(9,14,comb_accuracies)
worksheet.write_column(9,15,r_bin_accuracies)
worksheet.write_column(9,16,p_bin_accuracies)
worksheet.write_column(9,17,r_bin_sf_accuracies)
worksheet.write_column(9,18,p_bin_sf_accuracies)

temp = data_dict_all['S1_dicts']['classification_output_p_val']
r_level_accuracies = [np.sum(temp['aft_cue'][:,0] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] < 0.05)/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,1] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,1] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,1] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,1] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,1] < 0.05)/float(np.shape(temp['concat'])[0])]
succ_accuraracies = [np.sum(temp['aft_cue'][:,2] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,2] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,2] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,2] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,2] < 0.05)/float(np.shape(temp['concat'])[0])]
comb_accuracies = [np.sum(temp['aft_cue'][:,3] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] < 0.05)/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,4] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,4] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,4] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,4] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,4] < 0.05)/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,5] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,5] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,5] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,5] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,5] < 0.05)/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,6] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] < 0.05)/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,7] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,7] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,7] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,7] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,7] < 0.05)/float(np.shape(temp['concat'])[0])]

worksheet.write(8,20,'S1: percent sig diff')
worksheet.write_row(8,21,comb_names)
worksheet.write_column(9,20,type_names)
worksheet.write_column(9,21,r_level_accuracies)
worksheet.write_column(9,22,p_level_accuracies)
worksheet.write_column(9,23,succ_accuracies)
worksheet.write_column(9,24,comb_accuracies)
worksheet.write_column(9,25,r_bin_accuracies)
worksheet.write_column(9,26,p_bin_accuracies)
worksheet.write_column(9,27,r_bin_sf_accuracies)
worksheet.write_column(9,28,p_bin_sf_accuracies)



##
temp = data_dict_all['PmD_dicts']['classification_output']

r_level_accuracies = [np.sum(temp['aft_cue'][:,0] > temp['aft_cue'][:,2])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] > temp['bfr_res'][:,2])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] > temp['aft_res'][:,2])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] > temp['res_win'][:,2])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] > temp['concat'][:,2])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,3] > temp['aft_cue'][:,5])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] > temp['bfr_res'][:,5])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] > temp['aft_res'][:,5])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] > temp['res_win'][:,5])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] > temp['concat'][:,5])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,6] > temp['aft_cue'][:,8])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] > temp['bfr_res'][:,8])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] > temp['aft_res'][:,8])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] > temp['res_win'][:,8])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] > temp['concat'][:,8])/float(np.shape(temp['concat'])[0])]

comb_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,11])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,11])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,11])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,11])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,11])/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,12] > temp['aft_cue'][:,14])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,12] > temp['bfr_res'][:,14])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,12] > temp['aft_res'][:,14])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,12] > temp['res_win'][:,14])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,12] > temp['concat'][:,14])/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,15] > temp['aft_cue'][:,17])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,15] > temp['bfr_res'][:,17])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,15] > temp['aft_res'][:,17])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,15] > temp['res_win'][:,17])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,15] > temp['concat'][:,17])/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies =[np.sum(temp['aft_cue'][:,18] > temp['aft_cue'][:,20])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,18] > temp['bfr_res'][:,20])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,18] > temp['aft_res'][:,20])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,18] > temp['res_win'][:,20])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,18] > temp['concat'][:,20])/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,21] > temp['aft_cue'][:,23])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,21] > temp['bfr_res'][:,23])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,21] > temp['aft_res'][:,23])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,21] > temp['res_win'][:,23])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,21] > temp['concat'][:,23])/float(np.shape(temp['concat'])[0])]

worksheet.write(16,0,'PMd: percent better than shuffled')
worksheet.write_row(16,1,comb_names)
worksheet.write_column(17,0,type_names)
worksheet.write_column(17,1,r_level_accuracies)
worksheet.write_column(17,2,p_level_accuracies)
worksheet.write_column(17,3,succ_accuracies)
worksheet.write_column(17,4,comb_accuracies)
worksheet.write_column(17,5,r_bin_accuracies)
worksheet.write_column(17,6,p_bin_accuracies)
worksheet.write_column(17,7,r_bin_sf_accuracies)
worksheet.write_column(17,8,p_bin_sf_accuracies)

r_level_accuracies = [np.sum(temp['aft_cue'][:,0] > temp['aft_cue'][:,1])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] > temp['bfr_res'][:,1])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] > temp['aft_res'][:,1])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] > temp['res_win'][:,1])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] > temp['concat'][:,1])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,3] > temp['aft_cue'][:,4])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] > temp['bfr_res'][:,4])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] > temp['aft_res'][:,4])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] > temp['res_win'][:,4])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] > temp['concat'][:,4])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,6] > temp['aft_cue'][:,7])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] > temp['bfr_res'][:,7])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] > temp['aft_res'][:,7])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] > temp['res_win'][:,7])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] > temp['concat'][:,7])/float(np.shape(temp['concat'])[0])]

comb_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,10])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,10])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,10])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,10])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,10])/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,12] > temp['aft_cue'][:,13])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,12] > temp['bfr_res'][:,13])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,12] > temp['aft_res'][:,13])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,12] > temp['res_win'][:,13])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,12] > temp['concat'][:,13])/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,15] > temp['aft_cue'][:,16])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,15] > temp['bfr_res'][:,16])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,15] > temp['aft_res'][:,16])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,15] > temp['res_win'][:,16])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,15] > temp['concat'][:,16])/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies =[np.sum(temp['aft_cue'][:,18] > temp['aft_cue'][:,19])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,18] > temp['bfr_res'][:,19])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,18] > temp['aft_res'][:,19])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,18] > temp['res_win'][:,19])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,18] > temp['concat'][:,19])/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,21] > temp['aft_cue'][:,22])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,21] > temp['bfr_res'][:,22])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,21] > temp['aft_res'][:,22])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,21] > temp['res_win'][:,22])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,21] > temp['concat'][:,22])/float(np.shape(temp['concat'])[0])]

worksheet.write(16,10,'PMd: percent better than chance')
worksheet.write_row(16,11,comb_names)
worksheet.write_column(17,10,type_names)
worksheet.write_column(17,11,r_level_accuracies)
worksheet.write_column(17,12,p_level_accuracies)
worksheet.write_column(17,13,succ_accuracies)
worksheet.write_column(17,14,comb_accuracies)
worksheet.write_column(17,15,r_bin_accuracies)
worksheet.write_column(17,16,p_bin_accuracies)
worksheet.write_column(17,17,r_bin_sf_accuracies)
worksheet.write_column(17,18,p_bin_sf_accuracies)


temp = data_dict_all['PmD_dicts']['classification_output_p_val']
r_level_accuracies = [np.sum(temp['aft_cue'][:,0] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] < 0.05)/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,1] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,1] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,1] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,1] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,1] < 0.05)/float(np.shape(temp['concat'])[0])]
succ_accuraracies = [np.sum(temp['aft_cue'][:,2] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,2] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,2] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,2] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,2] < 0.05)/float(np.shape(temp['concat'])[0])]
comb_accuracies = [np.sum(temp['aft_cue'][:,3] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] < 0.05)/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,4] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,4] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,4] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,4] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,4] < 0.05)/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,5] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,5] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,5] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,5] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,5] < 0.05)/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,6] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] < 0.05)/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,7] < 0.05)/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,7] < 0.05)/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,7] < 0.05)/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,7] < 0.05)/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,7] < 0.05)/float(np.shape(temp['concat'])[0])]

worksheet.write(16,20,'PMd: percent sig diff')
worksheet.write_row(16,21,comb_names)
worksheet.write_column(17,20,type_names)
worksheet.write_column(17,21,r_level_accuracies)
worksheet.write_column(17,22,p_level_accuracies)
worksheet.write_column(17,23,succ_accuracies)
worksheet.write_column(17,24,comb_accuracies)
worksheet.write_column(17,25,r_bin_accuracies)
worksheet.write_column(17,26,p_bin_accuracies)
worksheet.write_column(17,27,r_bin_sf_accuracies)
worksheet.write_column(17,28,p_bin_sf_accuracies)





#########
worksheet = accuracy_workbook.add_worksheet('all_accuracy')

out_names = ['accuracy','accuracy shuffled','chance']

#
temp = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['r_level_all']
temp2 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['p_level_all']
temp3 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['succ_all']
temp4 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['comb_all']
temp5 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['r_bin_all']
temp6 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['p_bin_all']
temp7 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['r_bin_sf_all']
temp8 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['p_bin_sf_all']
ac = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['r_level_all']
temp2 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['p_level_all']
temp3 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['succ_all']
temp4 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['comb_all']
temp5 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['r_bin_all']
temp6 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['p_bin_all']
temp7 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['r_bin_sf_all']
temp8 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['p_bin_sf_all']
br = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['r_level_all']
temp2 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['p_level_all']
temp3 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['succ_all']
temp4 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['comb_all']
temp5 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['r_bin_all']
temp6 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['p_bin_all']
temp7 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['r_bin_sf_all']
temp8 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['p_bin_sf_all']
ar = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['M1_dicts']['classification_output_all']['res_win']['r_level_all']
temp2 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['p_level_all']
temp3 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['succ_all']
temp4 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['comb_all']
temp5 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['r_bin_all']
temp6 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['p_bin_all']
temp7 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['r_bin_sf_all']
temp8 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['p_bin_sf_all']
rw = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['M1_dicts']['classification_output_all']['concat']['r_level_all']
temp2 = data_dict_all['M1_dicts']['classification_output_all']['concat']['p_level_all']
temp3 = data_dict_all['M1_dicts']['classification_output_all']['concat']['succ_all']
temp4 = data_dict_all['M1_dicts']['classification_output_all']['concat']['comb_all']
temp5 = data_dict_all['M1_dicts']['classification_output_all']['concat']['r_bin_all']
temp6 = data_dict_all['M1_dicts']['classification_output_all']['concat']['p_bin_all']
temp7 = data_dict_all['M1_dicts']['classification_output_all']['concat']['r_bin_sf_all']
temp8 = data_dict_all['M1_dicts']['classification_output_all']['concat']['p_bin_sf_all']
ct = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

worksheet.write(0,0,'M1')
worksheet.write(0,1,comb_names[0])
worksheet.write(0,4,comb_names[1])
worksheet.write(0,7,comb_names[2])
worksheet.write(0,10,comb_names[3])
worksheet.write(0,13,comb_names[4])
worksheet.write(0,16,comb_names[5])
worksheet.write(0,19,comb_names[6])
worksheet.write(0,22,comb_names[7])

worksheet.write_row(1,1,out_names)
worksheet.write_row(1,4,out_names)
worksheet.write_row(1,7,out_names)
worksheet.write_row(1,10,out_names)
worksheet.write_row(1,13,out_names)
worksheet.write_row(1,16,out_names)
worksheet.write_row(1,19,out_names)
worksheet.write_row(1,22,out_names)

worksheet.write_column(1,0,type_names)
worksheet.write_row(2,1,ac)
worksheet.write_row(3,1,br)
worksheet.write_row(4,1,ar)
worksheet.write_row(5,1,rw)
worksheet.write_row(6,1,ct)


#
temp = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['r_level_all']
temp2 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['p_level_all']
temp3 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['succ_all']
temp4 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['comb_all']
temp5 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['r_bin_all']
temp6 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['p_bin_all']
temp7 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['r_bin_sf_all']
temp8 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['p_bin_sf_all']
ac =[temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['r_level_all']
temp2 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['p_level_all']
temp3 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['succ_all']
temp4 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['comb_all']
temp5 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['r_bin_all']
temp6 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['p_bin_all']
temp7 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['r_bin_sf_all']
temp8 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['p_bin_sf_all']
br = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['r_level_all']
temp2 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['p_level_all']
temp3 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['succ_all']
temp4 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['comb_all']
temp5 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['r_bin_all']
temp6 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['p_bin_all']
temp7 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['r_bin_sf_all']
temp8 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['p_bin_sf_all']
ar = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['S1_dicts']['classification_output_all']['res_win']['r_level_all']
temp2 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['p_level_all']
temp3 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['succ_all']
temp4 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['comb_all']
temp5 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['r_bin_all']
temp6 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['p_bin_all']
temp7 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['r_bin_sf_all']
temp8 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['p_bin_sf_all']
rw = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['S1_dicts']['classification_output_all']['concat']['r_level_all']
temp2 = data_dict_all['S1_dicts']['classification_output_all']['concat']['p_level_all']
temp3 = data_dict_all['S1_dicts']['classification_output_all']['concat']['succ_all']
temp4 = data_dict_all['S1_dicts']['classification_output_all']['concat']['comb_all']
temp5 = data_dict_all['S1_dicts']['classification_output_all']['concat']['r_bin_all']
temp6 = data_dict_all['S1_dicts']['classification_output_all']['concat']['p_bin_all']
temp7 = data_dict_all['S1_dicts']['classification_output_all']['concat']['r_bin_sf_all']
temp8 = data_dict_all['S1_dicts']['classification_output_all']['concat']['p_bin_sf_all']
ct = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

worksheet.write(8,0,'S1')
worksheet.write(0,1,comb_names[0])
worksheet.write(0,4,comb_names[1])
worksheet.write(0,7,comb_names[2])
worksheet.write(0,10,comb_names[3])
worksheet.write(0,13,comb_names[4])
worksheet.write(0,16,comb_names[5])
worksheet.write(0,19,comb_names[6])
worksheet.write(0,22,comb_names[7])

worksheet.write_row(1,1,out_names)
worksheet.write_row(1,4,out_names)
worksheet.write_row(1,7,out_names)
worksheet.write_row(1,10,out_names)
worksheet.write_row(1,13,out_names)
worksheet.write_row(1,16,out_names)
worksheet.write_row(1,19,out_names)
worksheet.write_row(1,22,out_names)

worksheet.write_column(9,0,type_names)
worksheet.write_row(10,1,ac)
worksheet.write_row(11,1,br)
worksheet.write_row(12,1,ar)
worksheet.write_row(13,1,rw)
worksheet.write_row(14,1,ct)

#
temp = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['r_level_all']
temp2 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['p_level_all']
temp3 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['succ_all']
temp4 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['comb_all']
temp5 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['r_bin_all']
temp6 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['p_bin_all']
temp7 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['r_bin_sf_all']
temp8 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['p_bin_sf_all']
ac = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['r_level_all']
temp2 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['p_level_all']
temp3 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['succ_all']
temp4 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['comb_all']
temp5 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['r_bin_all']
temp6 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['p_bin_all']
temp7 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['r_bin_sf_all']
temp8 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['p_bin_sf_all']
br = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['r_level_all']
temp2 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['p_level_all']
temp3 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['succ_all']
temp4 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['comb_all']
temp5 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['r_bin_all']
temp6 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['p_bin_all']
temp7 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['r_bin_sf_all']
temp8 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['p_bin_sf_all']
ar = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['r_level_all']
temp2 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['p_level_all']
temp3 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['succ_all']
temp4 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['comb_all']
temp5 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['r_bin_all']
temp6 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['p_bin_all']
temp7 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['r_bin_sf_all']
temp8 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['p_bin_sf_all']
rw = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['PmD_dicts']['classification_output_all']['concat']['r_level_all']
temp2 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['p_level_all']
temp3 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['succ_all']
temp4 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['comb_all']
temp5 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['r_bin_all']
temp6 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['p_bin_all']
temp7 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['r_bin_sf_all']
temp8 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['p_bin_sf_all']
ct = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

worksheet.write(16,0,'PmD')
worksheet.write(0,1,comb_names[0])
worksheet.write(0,4,comb_names[1])
worksheet.write(0,7,comb_names[2])
worksheet.write(0,10,comb_names[3])
worksheet.write(0,13,comb_names[4])
worksheet.write(0,16,comb_names[5])
worksheet.write(0,19,comb_names[6])
worksheet.write(0,22,comb_names[7])

worksheet.write_row(1,1,out_names)
worksheet.write_row(1,4,out_names)
worksheet.write_row(1,7,out_names)
worksheet.write_row(1,10,out_names)
worksheet.write_row(1,13,out_names)
worksheet.write_row(1,16,out_names)
worksheet.write_row(1,19,out_names)
worksheet.write_row(1,22,out_names)

worksheet.write_column(17,0,type_names)
worksheet.write_row(18,1,ac)
worksheet.write_row(19,1,br)
worksheet.write_row(20,1,ar)
worksheet.write_row(21,1,rw)
worksheet.write_row(22,1,ct)



###########

plt.close('all')


