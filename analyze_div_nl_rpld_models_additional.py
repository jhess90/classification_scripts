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




###
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


#################

data_dict_1 = np.load('model_save_rpld.npy')[()]
data_dict_2 = np.load('model_save_rpld_pop.npy')[()]


for region_key,region_val in data_dict_1.iteritems():
    for win_key,win_val in data_dict_1[region_key]['comp_r2_mse_dicts'].iteritems():
        comp_r2_adj_all_1 = data_dict_1[region_key]['comp_r2_mse_dicts'][win_key]['comp_r2_adj_all']
        comp_r2_adj_sig_1 = data_dict_1[region_key]['comp_r2_mse_dicts'][win_key]['comp_r2_adj_sig']

        comp_r2_adj_all_2 = data_dict_2[region_key]['comp_r2_mse_dicts'][win_key]['comp_r2_adj_all']
        comp_r2_adj_sig_2 = data_dict_2[region_key]['comp_r2_mse_dicts'][win_key]['comp_r2_adj_sig']

        #r only reg, r only pop resonse, p only reg, p only pop response
        play_r2_adj_all = [comp_r2_adj_all_1[0],comp_r2_adj_all_2[0],comp_r2_adj_all_1[1],comp_r2_adj_all_2[1]]
        play_r2_adj_sig = [comp_r2_adj_sig_1[0],comp_r2_adj_sig_2[0],comp_r2_adj_sig_1[1],comp_r2_adj_sig_2[1]]


        play_r2_adj_all_temp = [item for sublist in play_r2_adj_all for item in sublist]
        play_r2_adj_sig_temp = [item for sublist in play_r2_adj_sig for item in sublist]

        #
        #play_r2_adj_all_range = [math.floor(np.nanmin(play_r2_adj_all_temp)*10)/10,math.ceil(np.nanmax(play_r2_adj_all_temp)*10)/10]
        #if np.shape(play_r2_adj_sig_temp)[0] != 0:
        #    play_r2_adj_sig_range = [math.floor(np.nanmin(play_r2_adj_sig_temp)*10)/10,math.ceil(np.nanmax(play_r2_adj_sig_temp)*10)/10]
        #else:
        #    play_r2_adj_sig_range = [0,1]

        play_r2_adj_all_range = [0,1]
        play_r2_adj_sig_range = [0,1]

        ##
        #all plots (not outlier)
        f,axarr = plt.subplots(4,sharex=True)
        f.suptitle('response types r2_adj %s all units: %s' %(region_key,win_key))
        try:
            axarr[0].hist(play_r2_adj_all[0][~is_outlier(play_r2_adj_all[0])],color='midnightblue',lw=0,bins=12)
            axarr[0].axvline(np.mean(play_r2_adj_all[0]))
        except:
            pass
        axarr[0].set_xlim([play_r2_adj_all_range[0],play_r2_adj_all_range[1]])
        axarr[0].set_title('r only',fontsize='small')
        try:
            axarr[1].hist(play_r2_adj_all[1][~is_outlier(play_r2_adj_all[1])],color='midnightblue',lw=0,bins=12)
            axarr[1].axvline(np.mean(play_r2_adj_all[1]))
        except:
            pass
        axarr[1].set_xlim([play_r2_adj_all_range[0],play_r2_adj_all_range[1]])
        axarr[1].set_title('r only pop response',fontsize='small')
        try:
            axarr[2].hist(play_r2_adj_all[2][~is_outlier(play_r2_adj_all[2])],color='midnightblue',lw=0,bins=12)
            axarr[2].axvline(np.mean(play_r2_adj_all[2]))
        except:
            pass
        axarr[2].set_xlim([play_r2_adj_all_range[0],play_r2_adj_all_range[1]])
        axarr[2].set_title('p only',fontsize='small')
        try:
            axarr[3].hist(play_r2_adj_all[3][~is_outlier(play_r2_adj_all[3])],color='midnightblue',lw=0,bins=12)
            axarr[3].axvline(np.mean(play_r2_adj_all[3]))
        except:
            pass
        axarr[3].set_xlim([play_r2_adj_all_range[0],play_r2_adj_all_range[1]])
        axarr[3].set_title('p only pop response',fontsize='small')
        for axi in axarr.reshape(-1):
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('rp_response_types_r2_adj_all_%s_%s' %(region_key,win_key))
        plt.clf()


        #sig plots
        f,axarr = plt.subplots(4,sharex=True)
        f.suptitle('response types r2_adj %s sig units: %s' %(region_key,win_key))
        try:
            axarr[0].hist(play_r2_adj_sig[0],color='seagreen',lw=0,bins=12)
            axarr[0].axvline(np.mean(play_r2_adj_sig[0]))
        except:
            pass
        axarr[0].set_xlim([play_r2_adj_sig_range[0],play_r2_adj_sig_range[1]])
        axarr[0].set_title('r only',fontsize='small')
        try:
            axarr[1].hist(play_r2_adj_sig[1],color='seagreen',lw=0,bins=12)
            axarr[1].axvline(np.mean(play_r2_adj_sig[1]))
        except:
            pass
        axarr[1].set_xlim([play_r2_adj_sig_range[0],play_r2_adj_sig_range[1]])
        axarr[1].set_title('r only pop response',fontsize='small')
        try:
            axarr[2].hist(play_r2_adj_sig[2],color='seagreen',lw=0,bins=12)
            axarr[2].axvline(np.mean(play_r2_adj_sig[2]))
        except:
            pass
        axarr[2].set_xlim([play_r2_adj_sig_range[0],play_r2_adj_sig_range[1]])
        axarr[2].set_title('p only',fontsize='small')
        try:
            axarr[3].hist(play_r2_adj_sig[3],color='seagreen',lw=0,bins=12)
            axarr[3].axvline(np.mean(play_r2_adj_sig[3]))
        except:
            pass
        axarr[3].set_xlim([play_r2_adj_sig_range[0],play_r2_adj_sig_range[1]])
        axarr[3].set_title('p only pop response',fontsize='small')
        for axi in axarr.reshape(-1):
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('rp_response_types_r2_adj_sig_%s_%s' %(region_key,win_key))
        plt.clf()


        #NOTE rp 0059: sig looks exactly the same but is not, same to a number of decimal points though
