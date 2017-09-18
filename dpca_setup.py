#!/usr/bin/env python

#dPCA, based on Kobak et al 2016

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
def sort_and_avg(fr_array,sort_dict):

	S = len(sort_dict['s_stim'].keys())
	Q = len(sort_dict['q_result'].keys())
	K = np.shape(fr_array)[0]
	N = np.shape(fr_array)[1]
	T = np.shape(fr_array)[2]

	#sq0 = r0p0, succ   sq1 = r0p0, fail
	#sq2 = rxp0, succ   sq3 = rxp0, fail
	#sq4 = r0px, succ   sq5 = r0px, fail
	#sq6 = rxpx, succ   sq7 = rxpx, fail
	
	#TODO make general for more than S = 4
	sq0 = sort_dict['s_stim']['r0p0'][np.in1d(sort_dict['s_stim']['r0p0'],sort_dict['q_result']['succ_trials'])]
	sq1 = sort_dict['s_stim']['r0p0'][np.in1d(sort_dict['s_stim']['r0p0'],sort_dict['q_result']['fail_trials'])]
	
	sq2 = sort_dict['s_stim']['rxp0'][np.in1d(sort_dict['s_stim']['rxp0'],sort_dict['q_result']['succ_trials'])]
	sq3 = sort_dict['s_stim']['rxp0'][np.in1d(sort_dict['s_stim']['rxp0'],sort_dict['q_result']['fail_trials'])]

	sq4 = sort_dict['s_stim']['r0px'][np.in1d(sort_dict['s_stim']['r0px'],sort_dict['q_result']['succ_trials'])]
	sq5 = sort_dict['s_stim']['r0px'][np.in1d(sort_dict['s_stim']['r0px'],sort_dict['q_result']['fail_trials'])]

	sq6 = sort_dict['s_stim']['rxpx'][np.in1d(sort_dict['s_stim']['rxpx'],sort_dict['q_result']['succ_trials'])]
	sq7 = sort_dict['s_stim']['rxpx'][np.in1d(sort_dict['s_stim']['rxpx'],sort_dict['q_result']['fail_trials'])]

	######### deal w/ unbalanced data #######
	fr_all_dict = {'comb0':fr_array[sq0,:,:],'comb1':fr_array[sq1,:,:],'comb2':fr_array[sq2,:,:],'comb3':fr_array[sq3,:,:],'comb4':fr_array[sq4,:,:],'comb5':fr_array[sq5,:,:],'comb6':fr_array[sq6,:,:],'comb7':fr_array[sq7,:,:]}

	min_trial_num = 1000
	for i in range(8):
		var_name = 'sq%s' %(i)
		if np.shape(fr_array[var_name,:,:])[0] < min_trial_num:
			min_trial_num = np.shape(fr_array[var_name,:,:])

	bal_fr_array = np.zeros((min_trial_num,np.shape(fr_array[sq0,:,:])[1],np.shape(fr_array[sq0,:,:])[2])
	for i in range(8):
		var_name = 'sq%s' %(i)
		comb_fr= fr_array[var_name,:,:]
		idx = np.random.randint(np.size(comb_fr)[0],size=min_trial_num)
	

	
	sq0_avg = np.mean(fr_array[sq0,:,:],axis=0)
	sq1_avg = np.mean(fr_array[sq1,:,:],axis=0)
	sq2_avg = np.mean(fr_array[sq2,:,:],axis=0)
	sq3_avg = np.mean(fr_array[sq3,:,:],axis=0)
	sq4_avg = np.mean(fr_array[sq4,:,:],axis=0)
	sq5_avg = np.mean(fr_array[sq5,:,:],axis=0)
	sq6_avg = np.mean(fr_array[sq6,:,:],axis=0)
	sq7_avg = np.mean(fr_array[sq7,:,:],axis=0)
	
	pdb.set_trace()
	#x = np.dstack((sq0_fr,sq1_fr,sq2_fr,sq3_fr,sq4_fr,sq5_fr,sq6_fr,sq7_fr))
	#xavg = avg across K trials for each of SQ conditions
	x_avg = np.dstack((sq0_avg,sq1_avg,sq2_avg,sq3_avg,sq4_avg,sq5_avg,sq6_avg,sq7_avg))
	x_avg = np.reshape(x_avg,(np.shape(x_avg)[0],np.shape(x_avg)[2],np.shape(x_avg)[1]))

	#TODO unhardcode conditions
	x_avg_shaped = np.zeros((N,T,4,2))
	x_avg_shaped[:,:,0,0] = sq0_avg
	x_avg_shaped[:,:,1,0] = sq2_avg
	x_avg_shaped[:,:,2,0] = sq4_avg
	x_avg_shaped[:,:,3,0] = sq6_avg
	x_avg_shaped[:,:,0,1] = sq1_avg
	x_avg_shaped[:,:,1,1] = sq3_avg
	x_avg_shaped[:,:,2,1] = sq5_avg
	x_avg_shaped[:,:,3,1] = sq7_avg


	
	
	
	return x_avg_shaped

	
def marginalization(x_avg):
	




	marg = []
	return(marg)






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

all_dict = {}
all_dict['M1'] = {}
all_dict['S1'] = {}
all_dict['PmD'] = {}
all_dict['M1']['M1_fr_dict'] = M1_fr_dict
all_dict['S1']['S1_fr_dict'] = S1_fr_dict
all_dict['PmD']['PmD_fr_dict'] = PmD_fr_dict

#condensed: col 0 = disp_rp, 1 = succ scene, 2 = failure scene, 3 = rnum, 4 = pnum, 5 = succ(1) / fail(-1), 6 = value, 7 = motivation
new_condensed = np.zeros((np.shape(condensed)[0],8))
new_condensed[:,0:6] = condensed
new_condensed[:,6] = new_condensed[:,3] - new_condensed[:,4]
new_condensed[:,7] = new_condensed[:,3] + new_condensed[:,4]
condensed = new_condensed

#stim = S
#for now seperate into all-R, no-R, no-P, and all-P, so four combinations
#TODO make work for multiple levels (16 combinations)
r0p0 = []
rxp0 = []
r0px = []
rxpx = []
for i in range(np.shape(condensed)[0]):
	if (condensed[i,3] == 0) and (condensed[i,4] == 0):
		r0p0.append(i)
	elif (condensed[i,3] > 0) and (condensed[i,4] ==0):
		rxp0.append(i)
	elif (condensed[i,3] == 0) and (condensed[i,4] > 0):
		r0px.append(i)
	elif (condensed[i,3] > 0) and (condensed[i,4] > 0):
		rxpx.append(i)
	else:
		pdb.set_trace()

s_stim = {'r0p0':np.asarray(r0p0),'rxp0':np.asarray(rxp0),'r0px':np.asarray(r0px),'rxpx':np.asarray(rxpx)}
		
#result = Q
succ_trials = np.squeeze(np.where(condensed[:,5] == 1))
fail_trials = np.squeeze(np.where(condensed[:,5] == -1))
q_result = {'succ_trials':np.asarray(succ_trials),'fail_trials':np.asarray(fail_trials)}

sort_dict = {'q_result':q_result,'s_stim':s_stim,'condensed':condensed}

for region_key,region_val in all_dict.iteritems():
	fr_dict_str = '%s_fr_dict' %(region_key)

	bfr_cue_avg = sort_and_avg(all_dict[region_key][fr_dict_str]['bfr_cue'],sort_dict)
	aft_cue_avg = sort_and_avg(all_dict[region_key][fr_dict_str]['aft_cue'],sort_dict)
	bfr_result_avg = sort_and_avg(all_dict[region_key][fr_dict_str]['bfr_result'],sort_dict)
	aft_result_avg = sort_and_avg(all_dict[region_key][fr_dict_str]['aft_result'],sort_dict)

	all_avg = np.zeros((np.shape(bfr_cue_avg)[0],6*np.shape(bfr_cue_avg)[1],np.shape(bfr_cue_avg)[2],np.shape(bfr_cue_avg)[3]))
	#TODO unhardcode
	all_avg[:,0:10,:,:] = bfr_cue_avg
	all_avg[:,10:30,:,:] = aft_cue_avg
	all_avg[:,30:40,:,:] = bfr_result_avg
	all_avg[:,40:60,:,:] = aft_result_avg



	
	#all_avg = np.dstack((bfr_cue_avg,aft_cue_avg,bfr_result_avg,aft_result_avg))
	#all_sort = np.dstack((bfr_cue_sort,aft_cue_sort,bfr_result_sort,aft_result_sort))
	#all_dict[region_key]['all_avg'] = all_avg
	#all_dict[region_key]['all_sort'] = all_sort
	
	#fr_all = np.dstack((all_dict[region_key][fr_dict_str]['bfr_cue'],all_dict[region_key][fr_dict_str]['aft_cue'],all_dict[region_key][fr_dict_str]['bfr_result'],all_dict[region_key][fr_dict_str]['aft_result']))
	
	#RESHAPING
	#fr_labeled = np.zeros((np.shape(all_avg)[0],np.shape(all_avg)[2],np.shape(all_avg)[1],np.shape()[1]))


	


######### from test ########
dpca = dPCA.dPCA(labels='tsd') #,regularizer='auto')
dpca.protect = ['t']
#Z = dpca.fit_transform(R,trialR)
Z = dpca.fit_transform(all_avg)
	

#pca=PCA(n_components = 20)
#for each unit
#pca.fit(test[0,:,:])
#fitted = pca.fit(test[0,:,:]).transform(test[0,:,:])


