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
#from dPCA import dPCA
import dPCA_new as dPCA

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

	#demean data
	unit_fr_mean = np.squeeze(np.apply_over_axes(np.mean,fr_array,(0,2)))
	demeaned_fr = np.zeros((K,N,T))
	for i in range(N):
		demeaned_fr[:,i,:] = fr_array[:,i,:] - unit_fr_mean[i]

	fr_array = demeaned_fr
	
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
	sq_dict = {'sq0':sq0,'sq1':sq1,'sq2':sq2,'sq3':sq3,'sq4':sq4,'sq5':sq5,'sq6':sq6,'sq7':sq7}

	
	min_trial_num = 1000
	for i in range(8):
		var_name = sq_dict['sq%s' %(i)]
		if np.shape(fr_array[var_name,:,:])[0] < min_trial_num:
			min_trial_num = np.shape(fr_array[var_name,:,:])[0]

	bal_fr_array = np.zeros((8,min_trial_num,np.shape(fr_array[sq0,:,:])[1],np.shape(fr_array[sq0,:,:])[2]))
	for i in range(8):
		var_name = sq_dict['sq%s' %(i)]
		comb_fr= fr_array[var_name,:,:]
		idx = np.random.randint(np.shape(comb_fr)[0],size=min_trial_num)
		bal_fr_array[i,:,:,:] = comb_fr[idx,:,:]

		
	sq0_avg = np.mean(fr_array[sq0,:,:],axis=0)
	sq1_avg = np.mean(fr_array[sq1,:,:],axis=0)
	sq2_avg = np.mean(fr_array[sq2,:,:],axis=0)
	sq3_avg = np.mean(fr_array[sq3,:,:],axis=0)
	sq4_avg = np.mean(fr_array[sq4,:,:],axis=0)
	sq5_avg = np.mean(fr_array[sq5,:,:],axis=0)
	sq6_avg = np.mean(fr_array[sq6,:,:],axis=0)
	sq7_avg = np.mean(fr_array[sq7,:,:],axis=0)
	
	#xavg = avg across K trials for each of SQ conditions
	x_avg = np.dstack((sq0_avg,sq1_avg,sq2_avg,sq3_avg,sq4_avg,sq5_avg,sq6_avg,sq7_avg))
	x_avg = np.reshape(x_avg,(np.shape(x_avg)[0],np.shape(x_avg)[2],np.shape(x_avg)[1]))

	#Reshape so x_avg shape = [N,stim,result,T]
	#           bal_fr shape = [min_trial_num,N,stim,result,T]  
	x_avg_shaped = np.zeros((N,4,2,T))
	x_avg_shaped[:,0,0,:] = sq0_avg
	x_avg_shaped[:,1,0,:] = sq2_avg
	x_avg_shaped[:,2,0,:] = sq4_avg
	x_avg_shaped[:,3,0,:] = sq6_avg
	x_avg_shaped[:,0,1,:] = sq1_avg
	x_avg_shaped[:,1,1,:] = sq3_avg
	x_avg_shaped[:,2,1,:] = sq5_avg
	x_avg_shaped[:,3,1,:] = sq7_avg

	temp = bal_fr_array.reshape((8,min_trial_num,N,T))
	bal_fr_shaped = np.zeros((min_trial_num,N,4,2,T))
	bal_fr_shaped[:,:,0,0,:] = temp[0,:,:,:]
	bal_fr_shaped[:,:,1,0,:] = temp[2,:,:,:]
	bal_fr_shaped[:,:,2,0,:] = temp[4,:,:,:]
	bal_fr_shaped[:,:,3,0,:] = temp[6,:,:,:]
	bal_fr_shaped[:,:,0,1,:] = temp[1,:,:,:]
	bal_fr_shaped[:,:,1,1,:] = temp[3,:,:,:]
	bal_fr_shaped[:,:,2,1,:] = temp[5,:,:,:]
	bal_fr_shaped[:,:,3,1,:] = temp[7,:,:,:]

	
	return x_avg_shaped,bal_fr_shaped

	
def marginalization(x_avg):
	




	marg = []
	return(marg)






###############################################
#start ########################################
###############################################

#data = np.load('master_fr_dict.npy')[()]

filelist = glob.glob('master_fr*.npy')

all_dict = {}
all_dict['M1'] = {}
all_dict['S1'] = {}
all_dict['PmD'] = {}

all_input = {}
condensed_all = []
bfr_cue_all = []
aft_cue_all = []
bfr_result_all = []
aft_result_all = []
min_unit_num = [10000,1000,1000]
for i in range(len(filelist)):
	data = np.load(filelist[i])[()]

	M1_fr_dict = data['M1_fr_dict']
	S1_fr_dict = data['S1_fr_dict']
	PmD_fr_dict = data['PmD_fr_dict']
	condensed = data['condensed']

	
	if np.shape(M1_fr_dict['bfr_cue'])[1] < min_unit_num[0]:
		min_unit_num[0] = np.shape(M1_fr_dict['bfr_cue'])[1]
	if np.shape(S1_fr_dict['bfr_cue'])[1] < min_unit_num[1]:
		min_unit_num[1] = np.shape(S1_fr_dict['bfr_cue'])[1]
	if np.shape(PmD_fr_dict['bfr_cue'])[1] < min_unit_num[2]:
		min_unit_num[2] = np.shape(PmD_fr_dict['bfr_cue'])[1]

	M1_dims = np.shape(M1_fr_dict)
	S1_dims = np.shape(S1_fr_dict)
	PmD_dims = np.shape(PmD_fr_dict)
	dims = {'M1_dims':M1_dims,'S1_dims':S1_dims,'PmD_dims':PmD_dims}
	all_input['%s'%(i)] = {'M1_fr_dict':M1_fr_dict,'S1_fr_dict':S1_fr_dict,'PmD_fr_dict':PmD_fr_dict,'condensed':condensed,'dims':dims}

	if i == 0:
		condensed_all = condensed
	else:
		condensed_all = np.append(condensed_all,condensed,axis=0)

#TODO different windows, check all
filenum = len(filelist)

M1_bfr_cue_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[0],10))
M1_aft_cue_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[0],20))
M1_bfr_result_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[0],10))
M1_aft_result_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[0],20))
S1_bfr_cue_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[1],10))
S1_aft_cue_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[1],20))
S1_bfr_result_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[1],10))
S1_aft_result_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[1],20))
PmD_bfr_cue_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[2],10))
PmD_aft_cue_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[2],20))
PmD_bfr_result_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[2],10))
PmD_aft_result_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[2],20))


count = 0
for i in range(len(filelist)):
	data = all_input[str(i)]
	M1_fr_dict = data['M1_fr_dict']
	S1_fr_dict = data['S1_fr_dict']
	PmD_fr_dict = data['PmD_fr_dict']
	condensed = data['condensed']

	M1_unit_ind = np.sort(np.random.choice(np.shape(M1_fr_dict['bfr_cue'])[1],size=min_unit_num[0],replace=False))
	S1_unit_ind = np.sort(np.random.choice(np.shape(S1_fr_dict['bfr_cue'])[1],size=min_unit_num[1],replace=False))
	PmD_unit_ind = np.sort(np.random.choice(np.shape(PmD_fr_dict['bfr_cue'])[1],size=min_unit_num[2],replace=False))
	
	M1_bfr_cue_all[count:count+np.shape(condensed)[0],i*min_unit_num[0]:(i+1)*min_unit_num[0]:,:] = M1_fr_dict['bfr_cue'][:,M1_unit_ind,:]
	M1_aft_cue_all[count:count+np.shape(condensed)[0],i*min_unit_num[0]:(i+1)*min_unit_num[0]:,:] = M1_fr_dict['aft_cue'][:,M1_unit_ind,:]
	M1_bfr_result_all[count:count+np.shape(condensed)[0],i*min_unit_num[0]:(i+1)*min_unit_num[0]:,:] = M1_fr_dict['bfr_result'][:,M1_unit_ind,:]
	M1_aft_result_all[count:count+np.shape(condensed)[0],i*min_unit_num[0]:(i+1)*min_unit_num[0]:,:] = M1_fr_dict['aft_result'][:,M1_unit_ind,:]
	S1_bfr_cue_all[count:count+np.shape(condensed)[0],i*min_unit_num[1]:(i+1)*min_unit_num[1]:,:] = S1_fr_dict['bfr_cue'][:,S1_unit_ind,:]
	S1_aft_cue_all[count:count+np.shape(condensed)[0],i*min_unit_num[1]:(i+1)*min_unit_num[1]:,:] = S1_fr_dict['aft_cue'][:,S1_unit_ind,:]
	S1_bfr_result_all[count:count+np.shape(condensed)[0],i*min_unit_num[1]:(i+1)*min_unit_num[1]:,:] = S1_fr_dict['bfr_result'][:,S1_unit_ind,:]
	S1_aft_result_all[count:count+np.shape(condensed)[0],i*min_unit_num[1]:(i+1)*min_unit_num[1]:,:] = S1_fr_dict['aft_result'][:,S1_unit_ind,:]
	PmD_bfr_cue_all[count:count+np.shape(condensed)[0],i*min_unit_num[2]:(i+1)*min_unit_num[2]:,:] = PmD_fr_dict['bfr_cue'][:,PmD_unit_ind,:]
	PmD_aft_cue_all[count:count+np.shape(condensed)[0],i*min_unit_num[2]:(i+1)*min_unit_num[2]:,:] = PmD_fr_dict['aft_cue'][:,PmD_unit_ind,:]
	PmD_bfr_result_all[count:count+np.shape(condensed)[0],i*min_unit_num[2]:(i+1)*min_unit_num[2]:,:] = PmD_fr_dict['bfr_result'][:,PmD_unit_ind,:]
	PmD_aft_result_all[count:count+np.shape(condensed)[0],i*min_unit_num[2]:(i+1)*min_unit_num[2]:,:] = PmD_fr_dict['aft_result'][:,PmD_unit_ind,:]

	count += np.shape(condensed)[0]
	
	

	
#For each region, 3D dict [trial x unit x binned data]

all_dict['M1']['M1_fr_dict'] = {'bfr_cue':M1_bfr_cue_all,'aft_cue':M1_aft_cue_all,'bfr_result':M1_bfr_result_all,'aft_result':M1_aft_result_all}
all_dict['S1']['S1_fr_dict'] = {'bfr_cue':S1_bfr_cue_all,'aft_cue':S1_aft_cue_all,'bfr_result':S1_bfr_result_all,'aft_result':S1_aft_result_all}
all_dict['PmD']['PmD_fr_dict'] = {'bfr_cue':PmD_bfr_cue_all,'aft_cue':PmD_aft_cue_all,'bfr_result':PmD_bfr_result_all,'aft_result':PmD_aft_result_all}

condensed = condensed_all

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
	print 'running region: %s' %(region_key)
	fr_dict_str = '%s_fr_dict' %(region_key)
	bfr_cue = all_dict[region_key][fr_dict_str]['bfr_cue']
	aft_cue = all_dict[region_key][fr_dict_str]['aft_cue']
	bfr_result = all_dict[region_key][fr_dict_str]['bfr_result']
	aft_result = all_dict[region_key][fr_dict_str]['aft_result']

	all_fr = np.zeros((np.shape(bfr_cue)[0],np.shape(bfr_cue)[1],60))
	all_fr[:,:,0:10] = bfr_cue
	all_fr[:,:,10:30] = aft_cue
	all_fr[:,:,30:40] = bfr_result
	all_fr[:,:,40:60] = aft_result

	all_avg,all_bal = sort_and_avg(all_fr,sort_dict)
	[bal_cond,N,S,R,T] = np.shape(all_bal)
	
        print 'N= %s, S= %s, R= %s, T= %s, bal_cond= %s' %(N,S,R,T,bal_cond)

	######### from test ########
	dpca = dPCA.dPCA(labels='sdt',regularizer='auto',n_components = 5)
	dpca.protect = ['t']
	Z = dpca.fit_transform(all_avg,all_bal)
	explained_var = dpca.explained_variance_ratio_
	
	bins = np.arange(T)

	# Z has keys ['sdt', 'd', 'st', 's', 't', 'dt', 'sd']
	#each key of shape [n_components,S,R,T]

	#PARAM
	components_plot = 3
	my_ticks = ['-10','0','10','-10','0','10','20']
	my_ticks_num = [0,10,20,30,40,50,60]
	labels = ['r0p0','rxp0','r0px','rxpx']
	lines = []
	for comb_ind,comb_val in Z.iteritems():
		ax = plt.gca()
		for i in range(components_plot):
			plt.subplot(2,components_plot,i+1)
			for s in range(S):
				line_name = 'line%s' %(s)
				line = plt.plot(bins,Z[comb_ind][i,s,0,:],label=labels[s])
				lines = np.append(lines,line)
			plt.axvline(x=10,color='g',linestyle='--')
			plt.axvline(x=30,color='k')
			plt.axvline(x=40,color='b',linestyle='--')
			plt.title('Component: %s, R: 0' %(i+1),fontsize='small')
			plt.xticks(my_ticks_num,my_ticks)

			plt.subplot(2,components_plot,i+components_plot+1)
			for s in range(S):
				plt.plot(bins,Z[comb_ind][i,s,1,:],label=labels[s])
				plt.xticks(my_ticks_num,my_ticks)
				
			plt.axvline(x=10,color='g',linestyle='--')
			plt.axvline(x=30,color='k')
			plt.axvline(x=40,color='b',linestyle='--')
			plt.title('Component: %s, R: 1' %(i+1),fontsize='small')

		plt.figlegend(lines,labels,loc='right',ncol=1,fontsize='small')
		plt.tight_layout(w_pad=0.1)
		plt.subplots_adjust(top=0.9,right=0.85)
		plt.rcParams['xtick.labelsize'] = 8
		plt.rcParams['ytick.labelsize'] = 8
		plt.suptitle('Region %s, comb ind = %s' %(region_key,comb_ind))
		plt.savefig('compmonents_%s_%s' %(region_key,comb_ind))
		plt.clf()

	
		sig_analysis = dpca.significance_analysis(all_avg,all_bal)
		transformed = dpca.transform(all_avg)

		dpca_results = {'Z':Z,'explained_var':explained_var,'sig_analysis':sig_analysis,'transformed':transformed,'all_avg':all_avg,'all_bal':all_bal}
		all_dict[region_key]['dpca_results'] = dpca_results

all_dict['sort_dict'] = sort_dict
np.save('dpca_results.npy',all_dict)
