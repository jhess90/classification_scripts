#!/usr/bin/env python

#targeted dimensionality reduction (based on Mante et al 2013)


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
from scipy.ndimage.filters import gaussian_filter
from sklearn.decomposition import PCA
from dPCA import dPCA


#######################
#params to set ########
#######################






###############################
### functions #################
###############################
def sort_and_avg(fr_array,sort_ind):

	#unit_fr_mean = np.squeeze(np.apply_over_axes(np.mean,fr_array,(0,2)))
	#unit_fr_std = np.squeeze(np.apply_over_axes(np.std,fr_array,(0,2)))

	




	zscore_array_all = []
	for i in range(int(np.max(sort_ind[:,1]) + 1)):
		temp = sort_ind[sort_ind[:,1] == i]
		for j in range(np.shape(temp)[0]):
			cond_trial_nums = temp[:,0]
			cond_trial_nums = cond_trial_nums.astype(int)

			fr_array_cond = fr_array[cond_trial_nums,:,:]
			mean_fr_cond = np.mean(fr_array_cond,axis=0)
			
			#gaussian_smoothed = gaussian_filter(mean_fr_cond,sigma=sigma_val)
			zscore_array = np.zeros((np.shape(mean_fr_cond)))
			for k in range(np.shape(mean_fr_cond)[0]):
				zscore_array[k,:] = (mean_fr_cond[k,:] - unit_fr_mean[k]) / unit_fr_std[k]

		zscore_array_all.append(zscore_array)

	zscore_array_all = np.asarray(zscore_array_all)
	dims = np.shape(zscore_array_all)
	zscore_array_all = zscore_array_all.reshape((dims[1],dims[0],dims[2]))



	return(zscore_array_all)







###############################################
#start ########################################
###############################################

#TODO load multiple days of data, collate data
data = np.load('master_fr_dict.npy')[()]

#For each region, 3D dict [trial x unit x binned data]
M1_fr_dict = data['M1_fr_dict']
S1_fr_dict = data['S1_fr_dict']
PmD_fr_dict = data['PmD_fr_dict']
condensed = data['condensed']

#condensed: col 0 = disp_rp, 1 = succ scene, 2 = failure scene, 3 = rnum, 4 = pnum, 5 = succ(1) / fail(-1), 6 = value, 7 = motivation

new_condensed = np.zeros((np.shape(condensed)[0],8))
new_condensed[:,0:6] = condensed
new_condensed[:,6] = new_condensed[:,3] - new_condensed[:,4]
new_condensed[:,7] = new_condensed[:,3] + new_condensed[:,4]
condensed = new_condensed

#stim = S
#for now seperate into all-R, no-R, no-P, and all-P, so four combinations
#TODO make work for multiple levels (16 combinations)

all_r = np.squeeze(np.where(condensed[:,3] > 0))
no_r = np.squeeze(np.where(condensed[:,3] == 0))
all_p = np.squeeze(np.where(condensed[:,4] > 0))
no_p = np.squeeze(np.where(condensed[:,4] == 0))





#result = Q
succ_trials = np.squeeze(np.where(condensed[:,5] == 1))
fail_trials = np.squeeze(np.where(condensed[:,5] == -1))
q_result = {'succ_trials':succ_trials,'fail_trials':fail_trials}

sort_ind = {'q_result':q_result}


test = sort_and_avg(M1_fr_dict['aft_result'],sort_ind)
test2 = sort_and_avg(M1_fr_dict['aft_result'],sort_ind)

#arrange 3d structure for PCA
coditions_dict = {}


pca=PCA(n_components = 20)

#for each unit
pca.fit(test[0,:,:])

fitted = pca.fit(test[0,:,:]).transform(test[0,:,:])


