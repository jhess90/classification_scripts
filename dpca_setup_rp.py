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

do_sig_analysis = False
inv_plot_bool = True




###############################
### functions #################
###############################
def sort_and_avg(fr_array,sort_dict):

	#S = len(sort_dict['s_stim'].keys())
	R = len(sort_dict['r_stim'].keys())
	P= len(sort_dict['p_stim'].keys())	
	Q = len(sort_dict['q_result'].keys())
	K = np.shape(fr_array)[0]
	N = np.shape(fr_array)[1]
	T = np.shape(fr_array)[2]

	count = 0
	trial_comb = np.zeros((R*P*Q,K))
	for r in range(R):
		if r == 0:
			r_trials = sort_dict['r_stim']['r0']
		elif r == 1:
			r_trials = sort_dict['r_stim']['rx']
			
		for p in range(P):
			if p == 0:
				p_trials = sort_dict['p_stim']['p0']
			elif p == 1:
				p_trials = sort_dict['p_stim']['px']

			for q in range(Q):
				if q == 0:
					q_trials = sort_dict['q_result']['fail_trials']
				elif q == 1:
					q_trials = sort_dict['q_result']['succ_trials']

				combined = ()
				for k in range(K):
					if k in r_trials and k in p_trials and k in q_trials:
						combined = np.append(combined,k)

				trial_comb[count,0:combined.size] = combined
				count += 1
				
	min_trial_size = K
	for i in range(np.shape(trial_comb)[0]):
		temp = np.size(np.nonzero(trial_comb[i]))
		if temp < min_trial_size:
			min_trial_size = temp

	bal_fr_shaped = np.zeros((min_trial_size,N,R,P,Q,T))
	bal_ind = np.zeros((np.shape(trial_comb)[0],min_trial_size))
	
	for i in range(np.shape(trial_comb)[0]):
		if np.size(np.nonzero(trial_comb[i])) > min_trial_size:
			bal_ind[i,:] = np.random.choice(trial_comb[i][np.nonzero(trial_comb[i])],min_trial_size,replace=False)
		else:
			bal_ind[i,:] = trial_comb[i,:][np.nonzero(trial_comb[i,:])]


	bal_fr_shaped[:,:,0,0,0,:] = fr_array[bal_ind[0,:].astype(int),:,:]
	bal_fr_shaped[:,:,0,0,1,:] = fr_array[bal_ind[1,:].astype(int),:,:]
	bal_fr_shaped[:,:,0,1,0,:] = fr_array[bal_ind[2,:].astype(int),:,:]
	bal_fr_shaped[:,:,0,1,1,:] = fr_array[bal_ind[3,:].astype(int),:,:]
	bal_fr_shaped[:,:,1,0,0,:] = fr_array[bal_ind[4,:].astype(int),:,:]
	bal_fr_shaped[:,:,1,0,1,:] = fr_array[bal_ind[5,:].astype(int),:,:]
	bal_fr_shaped[:,:,1,1,0,:] = fr_array[bal_ind[6,:].astype(int),:,:]
	bal_fr_shaped[:,:,1,1,1,:] = fr_array[bal_ind[7,:].astype(int),:,:]

	x_avg_shaped = np.mean(bal_fr_shaped,axis=0)
	
	return x_avg_shaped,bal_fr_shaped


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

#TODO make sure error thrown if different bin sizes, time before/after, etc
for i in range(len(filelist)):
	data = np.load(filelist[i])[()]

	M1_fr_dict = data['M1_fr_dict']
	S1_fr_dict = data['S1_fr_dict']
	PmD_fr_dict = data['PmD_fr_dict']
	condensed = data['condensed']
	params = data['params']

	bfr_bins = int(params['time_before']*-1*1000/params['bin_size'])
	aft_bins = int(params['time_after']*1000/params['bin_size'])
	
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

M1_bfr_cue_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[0],bfr_bins))
M1_aft_cue_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[0],aft_bins))
M1_bfr_result_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[0],bfr_bins))
M1_aft_result_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[0],aft_bins))
S1_bfr_cue_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[1],bfr_bins))
S1_aft_cue_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[1],aft_bins))
S1_bfr_result_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[1],bfr_bins))
S1_aft_result_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[1],aft_bins))
PmD_bfr_cue_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[2],bfr_bins))
PmD_aft_cue_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[2],aft_bins))
PmD_bfr_result_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[2],bfr_bins))
PmD_aft_result_all = np.zeros((np.shape(condensed_all)[0],filenum*min_unit_num[2],aft_bins))


count = 0
for i in range(len(filelist)):
	data = all_input[str(i)]
	M1_fr_dict = data['M1_fr_dict']
	S1_fr_dict = data['S1_fr_dict']
	PmD_fr_dict = data['PmD_fr_dict']
	condensed = data['condensed']

        print filelist[i]
        
        print np.shape(M1_fr_dict['bfr_cue'])

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
r0 = []
rx = []
p0 = []
px = []

for i in range(np.shape(condensed)[0]):
	if condensed[i,3] == 0:
		r0.append(i)
	elif condensed[i,3] > 0:
		rx.append(i)

	if condensed[i,4] == 0:
		p0.append(i)
	elif condensed[i,4] > 1:
		px.append(i)

r_stim = {'r0':r0,'rx':rx}
p_stim = {'p0':p0,'px':px}
		
#result = Q
succ_trials = np.squeeze(np.where(condensed[:,5] == 1))
fail_trials = np.squeeze(np.where(condensed[:,5] == -1))
q_result = {'succ_trials':np.asarray(succ_trials),'fail_trials':np.asarray(fail_trials)}

sort_dict = {'q_result':q_result,'r_stim':r_stim,'p_stim':p_stim,'condensed':condensed}




for region_key,region_val in all_dict.iteritems():
	print 'running region: %s' %(region_key)
	fr_dict_str = '%s_fr_dict' %(region_key)
	bfr_cue = all_dict[region_key][fr_dict_str]['bfr_cue']
	aft_cue = all_dict[region_key][fr_dict_str]['aft_cue']
	bfr_result = all_dict[region_key][fr_dict_str]['bfr_result']
	aft_result = all_dict[region_key][fr_dict_str]['aft_result']

	all_fr = np.zeros((np.shape(bfr_cue)[0],np.shape(bfr_cue)[1],2*(bfr_bins+aft_bins)))

	all_fr[:,:,0:bfr_bins] = bfr_cue
	all_fr[:,:,bfr_bins:bfr_bins+aft_bins] = aft_cue
	all_fr[:,:,bfr_bins+aft_bins:2*bfr_bins+aft_bins] = bfr_result
	all_fr[:,:,2*bfr_bins+aft_bins:2*bfr_bins+2*aft_bins] = aft_result

	all_avg,all_bal = sort_and_avg(all_fr,sort_dict)
	[bal_cond,N,R,P,D,T] = np.shape(all_bal)
	
	print 'N= %s, R= %s, P= %s, D= %s, T= %s, bal_cond= %s' %(N,R,P,D,T,bal_cond)

        join_comb = {'rt':['r','rt'],'pt':['p','pt'],'dt':['d','dt'],'rdt':['rd','rdt'],'pdt':['pd','pdt'],'rpt':['rp','rpt'],'rpdt':['rpd','rpdt']}

	######### from test ########
	dpca = dPCA.dPCA(labels='rpdt',regularizer='auto',n_components = 15,join=join_comb)
	dpca.protect = ['t']
	Z = dpca.fit_transform(all_avg,all_bal)

        inv_transform_dict = {}
        for comb_key,comb_val in Z.iteritems():
                inv_transform_dict[comb_key]  = dpca.inverse_transform(Z[comb_key],comb_key)
            
        if not do_sig_analysis:
                del all_bal

	explained_var = dpca.explained_variance_ratio_	
        bins = np.arange(T)

        #PARAM
        components_plot = 4 #even for now
        my_ticks = ['-0.5','0','0.5','-0.5','0','0.5','1.0']
        
        tot_bins = (bfr_bins+aft_bins)*2
        my_ticks_num = np.arange(0,tot_bins*7/6,tot_bins/6)
        
        labels = ['r0p0 succ','rxp0 succ','r0px succ','rxpx succ','r0p0 fail','rxp0 fail','r0px fail','rxpx fail']
        colors = ['black','green','maroon','blue','black','green','maroon','blue']

        rt_labels = ['r0','rx']
        pt_labels = ['p0','px']
        d_labels = ['fail','succ']

        t_labels = ['cond_indp']
        comb_labels = ['comb_labels']

        ###########################
        if inv_plot_bool:
            
            print 'inverse plotting'
            for unit_num in range(np.shape(inv_transform_dict['t'])[0]):
                    ax = plt.gca()
                    plt.subplot(2,1,2)
                    line0 = plt.plot(bins,inv_transform_dict['t'][unit_num,0,0,1,:],label=labels[0],color=colors[0])
                    line1 = plt.plot(bins,inv_transform_dict['t'][unit_num,1,0,1,:],label=labels[1],color=colors[1])
                    line2 = plt.plot(bins,inv_transform_dict['t'][unit_num,0,1,1,:],label=labels[2],color=colors[2])
                    line3 = plt.plot(bins,inv_transform_dict['t'][unit_num,1,1,1,:],label=labels[3],color=colors[3])
                    line4 = plt.plot(bins,inv_transform_dict['t'][unit_num,0,0,0,:],label=labels[4],color=colors[4],linestyle='--')
                    line5 = plt.plot(bins,inv_transform_dict['t'][unit_num,1,0,0,:],label=labels[5],color=colors[5],linestyle='--')
                    line6 = plt.plot(bins,inv_transform_dict['t'][unit_num,0,1,0,:],label=labels[6],color=colors[6],linestyle='--')
                    line7 = plt.plot(bins,inv_transform_dict['t'][unit_num,1,1,0,:],label=labels[7],color=colors[7],linestyle='--')
                    
                    lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    #
                    ax = plt.gca()
                    plt.subplot(2,1,1)
                    all_label = ['all']
                    line0 = plt.plot(bins,np.mean(inv_transform_dict['t'],axis=(1,2,3))[unit_num,:],label=all_label[0],color='k')

                    lines = line0

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.suptitle('Unit: %s t' %(unit_num),fontsize='small')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,t_labels,loc='upper right',ncol=1,fontsize='small')
                    plt.tight_layout(w_pad=0.1)
                    plt.subplots_adjust(top=0.9,right=0.82)
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    plt.savefig('%s_t_%s' %(region_key,unit_num))
                    plt.clf()

                    #####
                    ax = plt.gca()
                    plt.subplot(2,1,2)
                    line0 = plt.plot(bins,inv_transform_dict['rt'][unit_num,0,0,1,:],label=labels[0],color=colors[0])
                    line1 = plt.plot(bins,inv_transform_dict['rt'][unit_num,1,0,1,:],label=labels[1],color=colors[1])
                    line2 = plt.plot(bins,inv_transform_dict['rt'][unit_num,0,1,1,:],label=labels[2],color=colors[2])
                    line3 = plt.plot(bins,inv_transform_dict['rt'][unit_num,1,1,1,:],label=labels[3],color=colors[3])
                    line4 = plt.plot(bins,inv_transform_dict['rt'][unit_num,0,0,0,:],label=labels[4],color=colors[4],linestyle='--')
                    line5 = plt.plot(bins,inv_transform_dict['rt'][unit_num,1,0,0,:],label=labels[5],color=colors[5],linestyle='--')
                    line6 = plt.plot(bins,inv_transform_dict['rt'][unit_num,0,1,0,:],label=labels[6],color=colors[6],linestyle='--')
                    line7 = plt.plot(bins,inv_transform_dict['rt'][unit_num,1,1,0,:],label=labels[7],color=colors[7],linestyle='--')
                    
                    lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    #
                    ax = plt.gca()
                    plt.subplot(2,1,1)
                    rt_labels = ['r0','rx']
                    line0 = plt.plot(bins,np.mean(inv_transform_dict['rt'],axis=(2,3))[unit_num,0,:],label=rt_labels[0],color=colors[0])
                    line1 = plt.plot(bins,np.mean(inv_transform_dict['rt'],axis=(2,3))[unit_num,1,:],label=rt_labels[1],color=colors[1])

                    lines = np.squeeze(np.dstack((line0,line1)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.suptitle('Unit: %s rt' %(unit_num),fontsize='small')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,rt_labels,loc='upper right',ncol=1,fontsize='small')
                    plt.tight_layout(w_pad=0.1)
                    plt.subplots_adjust(top=0.9,right=0.82)
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    plt.savefig('%s_rt_%s' %(region_key,unit_num))
                    plt.clf()

                    #####
                    ax = plt.gca()
                    plt.subplot(2,1,2)
                    line0 = plt.plot(bins,inv_transform_dict['pt'][unit_num,0,0,1,:],label=labels[0],color=colors[0])
                    line1 = plt.plot(bins,inv_transform_dict['pt'][unit_num,1,0,1,:],label=labels[1],color=colors[1])
                    line2 = plt.plot(bins,inv_transform_dict['pt'][unit_num,0,1,1,:],label=labels[2],color=colors[2])
                    line3 = plt.plot(bins,inv_transform_dict['pt'][unit_num,1,1,1,:],label=labels[3],color=colors[3])
                    line4 = plt.plot(bins,inv_transform_dict['pt'][unit_num,0,0,0,:],label=labels[4],color=colors[4],linestyle='--')
                    line5 = plt.plot(bins,inv_transform_dict['pt'][unit_num,1,0,0,:],label=labels[5],color=colors[5],linestyle='--')
                    line6 = plt.plot(bins,inv_transform_dict['pt'][unit_num,0,1,0,:],label=labels[6],color=colors[6],linestyle='--')
                    line7 = plt.plot(bins,inv_transform_dict['pt'][unit_num,1,1,0,:],label=labels[7],color=colors[7],linestyle='--')
                    
                    lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    #
                    ax = plt.gca()
                    plt.subplot(2,1,1)
                    p_labels = ['p0','px']
                    line0 = plt.plot(bins,np.mean(inv_transform_dict['pt'],axis=(1,3))[unit_num,0,:],label=p_labels[0],color=colors[0])
                    line1 = plt.plot(bins,np.mean(inv_transform_dict['pt'],axis=(1,3))[unit_num,1,:],label=p_labels[1],color=colors[2])

                    lines = np.squeeze(np.dstack((line0,line1)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.suptitle('Unit: %s pt' %(unit_num),fontsize='small')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,p_labels,loc='upper right',ncol=1,fontsize='small')
                    plt.tight_layout(w_pad=0.1)
                    plt.subplots_adjust(top=0.9,right=0.82)
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    plt.savefig('%s_pt_%s' %(region_key,unit_num))
                    plt.clf()

                    #####
                    ax = plt.gca()
                    plt.subplot(2,1,2)
                    line0 = plt.plot(bins,inv_transform_dict['dt'][unit_num,0,0,1,:],label=labels[0],color=colors[0])
                    line1 = plt.plot(bins,inv_transform_dict['dt'][unit_num,1,0,1,:],label=labels[1],color=colors[1])
                    line2 = plt.plot(bins,inv_transform_dict['dt'][unit_num,0,1,1,:],label=labels[2],color=colors[2])
                    line3 = plt.plot(bins,inv_transform_dict['dt'][unit_num,1,1,1,:],label=labels[3],color=colors[3])
                    line4 = plt.plot(bins,inv_transform_dict['dt'][unit_num,0,0,0,:],label=labels[4],color=colors[4],linestyle='--')
                    line5 = plt.plot(bins,inv_transform_dict['dt'][unit_num,1,0,0,:],label=labels[5],color=colors[5],linestyle='--')
                    line6 = plt.plot(bins,inv_transform_dict['dt'][unit_num,0,1,0,:],label=labels[6],color=colors[6],linestyle='--')
                    line7 = plt.plot(bins,inv_transform_dict['dt'][unit_num,1,1,0,:],label=labels[7],color=colors[7],linestyle='--')
                    
                    lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    #
                    ax = plt.gca()
                    plt.subplot(2,1,1)
                    d_labels = ['fail','succ']
                    line0 = plt.plot(bins,np.mean(inv_transform_dict['dt'],axis=(1,2))[unit_num,0,:],label=labels[0],color=colors[0],linestyle='--')
                    line1 = plt.plot(bins,np.mean(inv_transform_dict['dt'],axis=(1,2))[unit_num,1,:],label=labels[1],color=colors[0])

                    lines = np.squeeze(np.dstack((line0,line1)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.suptitle('Unit: %s dt' %(unit_num),fontsize='small')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,d_labels,loc='upper right',ncol=1,fontsize='small')
                    plt.tight_layout(w_pad=0.1)
                    plt.subplots_adjust(top=0.9,right=0.82)
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    plt.savefig('%s_dt_%s' %(region_key,unit_num))
                    plt.clf()

                    #####
                    ax = plt.gca()
                    plt.subplot(2,1,2)
                    line0 = plt.plot(bins,inv_transform_dict['rdt'][unit_num,0,0,1,:],label=labels[0],color=colors[0])
                    line1 = plt.plot(bins,inv_transform_dict['rdt'][unit_num,1,0,1,:],label=labels[1],color=colors[1])
                    line2 = plt.plot(bins,inv_transform_dict['rdt'][unit_num,0,1,1,:],label=labels[2],color=colors[2])
                    line3 = plt.plot(bins,inv_transform_dict['rdt'][unit_num,1,1,1,:],label=labels[3],color=colors[3])
                    line4 = plt.plot(bins,inv_transform_dict['rdt'][unit_num,0,0,0,:],label=labels[4],color=colors[4],linestyle='--')
                    line5 = plt.plot(bins,inv_transform_dict['rdt'][unit_num,1,0,0,:],label=labels[5],color=colors[5],linestyle='--')
                    line6 = plt.plot(bins,inv_transform_dict['rdt'][unit_num,0,1,0,:],label=labels[6],color=colors[6],linestyle='--')
                    line7 = plt.plot(bins,inv_transform_dict['rdt'][unit_num,1,1,0,:],label=labels[7],color=colors[7],linestyle='--')
                    
                    lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    #
                    ax = plt.gca()
                    plt.subplot(2,1,1)
                    rd_labels = ['r0_fail','rx_fail','r0_succ','rx_succ']
                    line0 = plt.plot(bins,np.mean(inv_transform_dict['rdt'],axis=(2))[unit_num,0,0,:],label=rd_labels[0],color=colors[0],linestyle='--')
                    line1 = plt.plot(bins,np.mean(inv_transform_dict['rdt'],axis=(2))[unit_num,1,0,:],label=rd_labels[1],color=colors[1],linestyle='--')
                    line2 = plt.plot(bins,np.mean(inv_transform_dict['rdt'],axis=(2))[unit_num,0,1,:],label=rd_labels[2],color=colors[0])
                    line3 = plt.plot(bins,np.mean(inv_transform_dict['rdt'],axis=(2))[unit_num,1,1,:],label=rd_labels[3],color=colors[1])

                    lines = np.squeeze(np.dstack((line0,line1,line2,line3)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.suptitle('Unit: %s rdt' %(unit_num),fontsize='small')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,rd_labels,loc='upper right',ncol=1,fontsize='small')
                    plt.tight_layout(w_pad=0.1)
                    plt.subplots_adjust(top=0.9,right=0.82)
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    plt.savefig('%s_rdt_%s' %(region_key,unit_num))
                    plt.clf()
                    
                    #####
                    ax = plt.gca()
                    plt.subplot(2,1,2)
                    line0 = plt.plot(bins,inv_transform_dict['pdt'][unit_num,0,0,1,:],label=labels[0],color=colors[0])
                    line1 = plt.plot(bins,inv_transform_dict['pdt'][unit_num,1,0,1,:],label=labels[1],color=colors[1])
                    line2 = plt.plot(bins,inv_transform_dict['pdt'][unit_num,0,1,1,:],label=labels[2],color=colors[2])
                    line3 = plt.plot(bins,inv_transform_dict['pdt'][unit_num,1,1,1,:],label=labels[3],color=colors[3])
                    line4 = plt.plot(bins,inv_transform_dict['pdt'][unit_num,0,0,0,:],label=labels[4],color=colors[4],linestyle='--')
                    line5 = plt.plot(bins,inv_transform_dict['pdt'][unit_num,1,0,0,:],label=labels[5],color=colors[5],linestyle='--')
                    line6 = plt.plot(bins,inv_transform_dict['pdt'][unit_num,0,1,0,:],label=labels[6],color=colors[6],linestyle='--')
                    line7 = plt.plot(bins,inv_transform_dict['pdt'][unit_num,1,1,0,:],label=labels[7],color=colors[7],linestyle='--')
                    
                    lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    #
                    ax = plt.gca()
                    plt.subplot(2,1,1)
                    pd_labels = ['p0_fail','px_fail','p0_succ','px_succ']
                    line0 = plt.plot(bins,np.mean(inv_transform_dict['pdt'],axis=(1))[unit_num,0,0,:],label=pd_labels[0],color=colors[0],linestyle='--')
                    line1 = plt.plot(bins,np.mean(inv_transform_dict['pdt'],axis=(1))[unit_num,1,0,:],label=pd_labels[1],color=colors[2],linestyle='--')
                    line2 = plt.plot(bins,np.mean(inv_transform_dict['pdt'],axis=(1))[unit_num,0,1,:],label=pd_labels[2],color=colors[0])
                    line3 = plt.plot(bins,np.mean(inv_transform_dict['pdt'],axis=(1))[unit_num,1,1,:],label=pd_labels[3],color=colors[2])

                    lines = np.squeeze(np.dstack((line0,line1,line2,line3)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.suptitle('Unit: %s pdt' %(unit_num),fontsize='small')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,pd_labels,loc='upper right',ncol=1,fontsize='small')
                    plt.tight_layout(w_pad=0.1)
                    plt.subplots_adjust(top=0.9,right=0.82)
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    plt.savefig('%s_pdt_%s' %(region_key,unit_num))
                    plt.clf()

                    #####
                    ax = plt.gca()
                    plt.subplot(2,1,2)
                    line0 = plt.plot(bins,inv_transform_dict['rpt'][unit_num,0,0,1,:],label=labels[0],color=colors[0])
                    line1 = plt.plot(bins,inv_transform_dict['rpt'][unit_num,1,0,1,:],label=labels[1],color=colors[1])
                    line2 = plt.plot(bins,inv_transform_dict['rpt'][unit_num,0,1,1,:],label=labels[2],color=colors[2])
                    line3 = plt.plot(bins,inv_transform_dict['rpt'][unit_num,1,1,1,:],label=labels[3],color=colors[3])
                    line4 = plt.plot(bins,inv_transform_dict['rpt'][unit_num,0,0,0,:],label=labels[4],color=colors[4],linestyle='--')
                    line5 = plt.plot(bins,inv_transform_dict['rpt'][unit_num,1,0,0,:],label=labels[5],color=colors[5],linestyle='--')
                    line6 = plt.plot(bins,inv_transform_dict['rpt'][unit_num,0,1,0,:],label=labels[6],color=colors[6],linestyle='--')
                    line7 = plt.plot(bins,inv_transform_dict['rpt'][unit_num,1,1,0,:],label=labels[7],color=colors[7],linestyle='--')
                    
                    lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    #
                    ax = plt.gca()
                    plt.subplot(2,1,1)
                    rp_labels = ['r0_p0','r0_px','rx_p0','rx_px']
                    line0 = plt.plot(bins,np.mean(inv_transform_dict['rpt'],axis=(3))[unit_num,0,0,:],label=rp_labels[0],color=colors[0])
                    line1 = plt.plot(bins,np.mean(inv_transform_dict['rpt'],axis=(3))[unit_num,0,1,:],label=rp_labels[1],color=colors[1])
                    line2 = plt.plot(bins,np.mean(inv_transform_dict['rpt'],axis=(3))[unit_num,1,0,:],label=rp_labels[2],color=colors[2])
                    line3 = plt.plot(bins,np.mean(inv_transform_dict['rpt'],axis=(3))[unit_num,1,1,:],label=rp_labels[3],color=colors[3])

                    lines = np.squeeze(np.dstack((line0,line1,line2,line3)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.suptitle('Unit: %s rpt' %(unit_num),fontsize='small')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,rp_labels,loc='upper right',ncol=1,fontsize='small')
                    plt.tight_layout(w_pad=0.1)
                    plt.subplots_adjust(top=0.9,right=0.82)
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    plt.savefig('%s_rpt_%s' %(region_key,unit_num))
                    plt.clf()

                    #####
                    ax = plt.gca()
                    plt.subplot(2,1,2)
                    line0 = plt.plot(bins,inv_transform_dict['rpdt'][unit_num,0,0,1,:],label=labels[0],color=colors[0])
                    line1 = plt.plot(bins,inv_transform_dict['rpdt'][unit_num,1,0,1,:],label=labels[1],color=colors[1])
                    line2 = plt.plot(bins,inv_transform_dict['rpdt'][unit_num,0,1,1,:],label=labels[2],color=colors[2])
                    line3 = plt.plot(bins,inv_transform_dict['rpdt'][unit_num,1,1,1,:],label=labels[3],color=colors[3])
                    line4 = plt.plot(bins,inv_transform_dict['rpdt'][unit_num,0,0,0,:],label=labels[4],color=colors[4],linestyle='--')
                    line5 = plt.plot(bins,inv_transform_dict['rpdt'][unit_num,1,0,0,:],label=labels[5],color=colors[5],linestyle='--')
                    line6 = plt.plot(bins,inv_transform_dict['rpdt'][unit_num,0,1,0,:],label=labels[6],color=colors[6],linestyle='--')
                    line7 = plt.plot(bins,inv_transform_dict['rpdt'][unit_num,1,1,0,:],label=labels[7],color=colors[7],linestyle='--')
                    
                    lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    #
                    plt.subplot(2,1,1)
                    all_label = ['all']
                    line0 = plt.plot(bins,np.mean(inv_transform_dict['rpdt'],axis=(1,2,3))[unit_num,:],label=all_label,color='k')

                    lines = line0

                    plt.axvline(x=bfr_bins,color='g',linestyle='--')
                    plt.axvline(x=bfr_bins+aft_bins,color='k')
                    plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
                    plt.suptitle('Unit: %s rpdt' %(unit_num),fontsize='small')
                    plt.xticks(my_ticks_num,my_ticks)
                    
                    plt.figlegend(lines,t_labels,loc='upper right',ncol=1,fontsize='small')
                    plt.tight_layout(w_pad=0.1)
                    plt.subplots_adjust(top=0.9,right=0.82)
                    plt.rcParams['xtick.labelsize'] = 8
                    plt.rcParams['ytick.labelsize'] = 8

                    plt.savefig('%s_rpdt_%s' %(region_key,unit_num))
                    plt.clf()


        #avg over all units
        ax = plt.gca()
        plt.subplot(2,1,2)
        line0 = plt.plot(bins,np.mean(inv_transform_dict['t'][0,0,1,:],axis=0),label=labels[0],color=colors[0])
        line1 = plt.plot(bins,np.mean(inv_transform_dict['t'][1,0,1,:],axis=0),label=labels[1],color=colors[1])
        line2 = plt.plot(bins,np.mean(inv_transform_dict['t'][0,1,1,:],axis=0),label=labels[2],color=colors[2])
        line3 = plt.plot(bins,np.mean(inv_transform_dict['t'][1,1,1,:],axis=0),label=labels[3],color=colors[3])
        line4 = plt.plot(bins,np.mean(inv_transform_dict['t'][0,0,0,:],axis=0),label=labels[4],color=colors[4],linestyle='--')
        line5 = plt.plot(bins,np.mean(inv_transform_dict['t'][1,0,0,:],axis=0),label=labels[5],color=colors[5],linestyle='--')
        line6 = plt.plot(bins,np.mean(inv_transform_dict['t'][0,1,0,:],axis=0),label=labels[6],color=colors[6],linestyle='--')
        line7 = plt.plot(bins,np.mean(inv_transform_dict['t'][1,1,0,:],axis=0),label=labels[7],color=colors[7],linestyle='--')
        
        lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        #
        ax = plt.gca()
        plt.subplot(2,1,1)
        all_label = ['all']
        line0 = plt.plot(bins,np.mean(inv_transform_dict['t'],axis=(0,1,2,3)),label=all_label[0],color='k')
        
        lines = line0
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.suptitle('All t',fontsize='small')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,t_labels,loc='upper right',ncol=1,fontsize='small')
        plt.tight_layout(w_pad=0.1)
        plt.subplots_adjust(top=0.9,right=0.82)
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        plt.savefig('%s_t_all' %(region_key))
        plt.clf()
        
        #####
        ax = plt.gca()
        plt.subplot(2,1,2)
        
        line0 = plt.plot(bins,np.mean(inv_transform_dict['rt'][0,0,1,:],axis=0),label=labels[0],color=colors[0])
        line1 = plt.plot(bins,np.mean(inv_transform_dict['rt'][1,0,1,:],axis=0),label=labels[1],color=colors[1])
        line2 = plt.plot(bins,np.mean(inv_transform_dict['rt'][0,1,1,:],axis=0),label=labels[2],color=colors[2])
        line3 = plt.plot(bins,np.mean(inv_transform_dict['rt'][1,1,1,:],axis=0),label=labels[3],color=colors[3])
        line4 = plt.plot(bins,np.mean(inv_transform_dict['rt'][0,0,0,:],axis=0),label=labels[4],color=colors[4],linestyle='--')
        line5 = plt.plot(bins,np.mean(inv_transform_dict['rt'][1,0,0,:],axis=0),label=labels[5],color=colors[5],linestyle='--')
        line6 = plt.plot(bins,np.mean(inv_transform_dict['rt'][0,1,0,:],axis=0),label=labels[6],color=colors[6],linestyle='--')
        line7 = plt.plot(bins,np.mean(inv_transform_dict['rt'][1,1,0,:],axis=0),label=labels[7],color=colors[7],linestyle='--')
        
        lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))
                        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        #
        ax = plt.gca()
        plt.subplot(2,1,1)
        rt_labels = ['r0','rx']
        line0 = plt.plot(bins,np.mean(inv_transform_dict['rt'],axis=(0,2,3))[0,:],label=rt_labels[0],color=colors[0])
        line1 = plt.plot(bins,np.mean(inv_transform_dict['rt'],axis=(0,2,3))[1,:],label=rt_labels[1],color=colors[1])
        
        lines = np.squeeze(np.dstack((line0,line1)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.suptitle('All rt',fontsize='small')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,rt_labels,loc='upper right',ncol=1,fontsize='small')
        plt.tight_layout(w_pad=0.1)
        plt.subplots_adjust(top=0.9,right=0.82)
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        plt.savefig('%s_rt_all' %(region_key))
        plt.clf()
        
        #####
        ax = plt.gca()
        plt.subplot(2,1,2)
        
        line0 = plt.plot(bins,np.mean(inv_transform_dict['pt'][0,0,1,:],axis=0),label=labels[0],color=colors[0])
        line1 = plt.plot(bins,np.mean(inv_transform_dict['pt'][1,0,1,:],axis=0),label=labels[1],color=colors[1])
        line2 = plt.plot(bins,np.mean(inv_transform_dict['pt'][0,1,1,:],axis=0),label=labels[2],color=colors[2])
        line3 = plt.plot(bins,np.mean(inv_transform_dict['pt'][1,1,1,:],axis=0),label=labels[3],color=colors[3])
        line4 = plt.plot(bins,np.mean(inv_transform_dict['pt'][0,0,0,:],axis=0),label=labels[4],color=colors[4],linestyle='--')
        line5 = plt.plot(bins,np.mean(inv_transform_dict['pt'][1,0,0,:],axis=0),label=labels[5],color=colors[5],linestyle='--')
        line6 = plt.plot(bins,np.mean(inv_transform_dict['pt'][0,1,0,:],axis=0),label=labels[6],color=colors[6],linestyle='--')
        line7 = plt.plot(bins,np.mean(inv_transform_dict['pt'][1,1,0,:],axis=0),label=labels[7],color=colors[7],linestyle='--')
        
        lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        #
        ax = plt.gca()
        plt.subplot(2,1,1)
        p_labels = ['p0','px']
        line0 = plt.plot(bins,np.mean(inv_transform_dict['pt'],axis=(0,1,3))[0,:],label=p_labels[0],color=colors[0])
        line1 = plt.plot(bins,np.mean(inv_transform_dict['pt'],axis=(0,1,3))[1,:],label=p_labels[1],color=colors[2])
        
        lines = np.squeeze(np.dstack((line0,line1)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.suptitle('All pt',fontsize='small')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,p_labels,loc='upper right',ncol=1,fontsize='small')
        plt.tight_layout(w_pad=0.1)
        plt.subplots_adjust(top=0.9,right=0.82)
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        plt.savefig('%s_pt_all' %(region_key))
        plt.clf()
        
        #####
        ax = plt.gca()
        plt.subplot(2,1,2)
        
        line0 = plt.plot(bins,np.mean(inv_transform_dict['dt'][0,0,1,:],axis=0),label=labels[0],color=colors[0])
        line1 = plt.plot(bins,np.mean(inv_transform_dict['dt'][1,0,1,:],axis=0),label=labels[1],color=colors[1])
        line2 = plt.plot(bins,np.mean(inv_transform_dict['dt'][0,1,1,:],axis=0),label=labels[2],color=colors[2])
        line3 = plt.plot(bins,np.mean(inv_transform_dict['dt'][1,1,1,:],axis=0),label=labels[3],color=colors[3])
        line4 = plt.plot(bins,np.mean(inv_transform_dict['dt'][0,0,0,:],axis=0),label=labels[4],color=colors[4],linestyle='--')
        line5 = plt.plot(bins,np.mean(inv_transform_dict['dt'][1,0,0,:],axis=0),label=labels[5],color=colors[5],linestyle='--')
        line6 = plt.plot(bins,np.mean(inv_transform_dict['dt'][0,1,0,:],axis=0),label=labels[6],color=colors[6],linestyle='--')
        line7 = plt.plot(bins,np.mean(inv_transform_dict['dt'][1,1,0,:],axis=0),label=labels[7],color=colors[7],linestyle='--')
        
        lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        #
        ax = plt.gca()
        plt.subplot(2,1,1)
        d_labels = ['fail','succ']
        line0 = plt.plot(bins,np.mean(inv_transform_dict['dt'],axis=(0,1,2))[0,:],label=labels[0],color=colors[0],linestyle='--')
        line1 = plt.plot(bins,np.mean(inv_transform_dict['dt'],axis=(0,1,2))[1,:],label=labels[1],color=colors[0])
        
        lines = np.squeeze(np.dstack((line0,line1)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.suptitle('All dt',fontsize='small')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,d_labels,loc='upper right',ncol=1,fontsize='small')
        plt.tight_layout(w_pad=0.1)
        plt.subplots_adjust(top=0.9,right=0.82)
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        plt.savefig('%s_dt_all' %(region_key))
        plt.clf()
        
        #####
        ax = plt.gca()
        plt.subplot(2,1,2)
        
        line0 = plt.plot(bins,np.mean(inv_transform_dict['rdt'][0,0,1,:],axis=0),label=labels[0],color=colors[0])
        line1 = plt.plot(bins,np.mean(inv_transform_dict['rdt'][1,0,1,:],axis=0),label=labels[1],color=colors[1])
        line2 = plt.plot(bins,np.mean(inv_transform_dict['rdt'][0,1,1,:],axis=0),label=labels[2],color=colors[2])
        line3 = plt.plot(bins,np.mean(inv_transform_dict['rdt'][1,1,1,:],axis=0),label=labels[3],color=colors[3])
        line4 = plt.plot(bins,np.mean(inv_transform_dict['rdt'][0,0,0,:],axis=0),label=labels[4],color=colors[4],linestyle='--')
        line5 = plt.plot(bins,np.mean(inv_transform_dict['rdt'][1,0,0,:],axis=0),label=labels[5],color=colors[5],linestyle='--')
        line6 = plt.plot(bins,np.mean(inv_transform_dict['rdt'][0,1,0,:],axis=0),label=labels[6],color=colors[6],linestyle='--')
        line7 = plt.plot(bins,np.mean(inv_transform_dict['rdt'][1,1,0,:],axis=0),label=labels[7],color=colors[7],linestyle='--')
        
        lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        #
        ax = plt.gca()
        plt.subplot(2,1,1)
        rd_labels = ['r0_fail','rx_fail','r0_succ','rx_succ']
        line0 = plt.plot(bins,np.mean(inv_transform_dict['rdt'],axis=(0,2))[0,0,:],label=rd_labels[0],color=colors[0],linestyle='--')
        line1 = plt.plot(bins,np.mean(inv_transform_dict['rdt'],axis=(0,2))[1,0,:],label=rd_labels[1],color=colors[1],linestyle='--')
        line2 = plt.plot(bins,np.mean(inv_transform_dict['rdt'],axis=(0,2))[0,1,:],label=rd_labels[2],color=colors[0])
        line3 = plt.plot(bins,np.mean(inv_transform_dict['rdt'],axis=(0,2))[1,1,:],label=rd_labels[3],color=colors[1])
        
        lines = np.squeeze(np.dstack((line0,line1,line2,line3)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.suptitle('All rdt',fontsize='small')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,rd_labels,loc='upper right',ncol=1,fontsize='small')
        plt.tight_layout(w_pad=0.1)
        plt.subplots_adjust(top=0.9,right=0.82)
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        plt.savefig('%s_rdt_all' %(region_key))
        plt.clf()
        
        #####
        ax = plt.gca()
        plt.subplot(2,1,2)
        
        line0 = plt.plot(bins,np.mean(inv_transform_dict['pdt'][0,0,1,:],axis=0),label=labels[0],color=colors[0])
        line1 = plt.plot(bins,np.mean(inv_transform_dict['pdt'][1,0,1,:],axis=0),label=labels[1],color=colors[1])
        line2 = plt.plot(bins,np.mean(inv_transform_dict['pdt'][0,1,1,:],axis=0),label=labels[2],color=colors[2])
        line3 = plt.plot(bins,np.mean(inv_transform_dict['pdt'][1,1,1,:],axis=0),label=labels[3],color=colors[3])
        line4 = plt.plot(bins,np.mean(inv_transform_dict['pdt'][0,0,0,:],axis=0),label=labels[4],color=colors[4],linestyle='--')
        line5 = plt.plot(bins,np.mean(inv_transform_dict['pdt'][1,0,0,:],axis=0),label=labels[5],color=colors[5],linestyle='--')
        line6 = plt.plot(bins,np.mean(inv_transform_dict['pdt'][0,1,0,:],axis=0),label=labels[6],color=colors[6],linestyle='--')
        line7 = plt.plot(bins,np.mean(inv_transform_dict['pdt'][1,1,0,:],axis=0),label=labels[7],color=colors[7],linestyle='--')
        
        lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        #
        ax = plt.gca()
        plt.subplot(2,1,1)
        pd_labels = ['p0_fail','px_fail','p0_succ','px_succ']
        line0 = plt.plot(bins,np.mean(inv_transform_dict['pdt'],axis=(0,1))[0,0,:],label=pd_labels[0],color=colors[0],linestyle='--')
        line1 = plt.plot(bins,np.mean(inv_transform_dict['pdt'],axis=(0,1))[1,0,:],label=pd_labels[1],color=colors[2],linestyle='--')
        line2 = plt.plot(bins,np.mean(inv_transform_dict['pdt'],axis=(0,1))[0,1,:],label=pd_labels[2],color=colors[0])
        line3 = plt.plot(bins,np.mean(inv_transform_dict['pdt'],axis=(0,1))[1,1,:],label=pd_labels[3],color=colors[2])
        
        lines = np.squeeze(np.dstack((line0,line1,line2,line3)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.suptitle('All pdt',fontsize='small')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,pd_labels,loc='upper right',ncol=1,fontsize='small')
        plt.tight_layout(w_pad=0.1)
        plt.subplots_adjust(top=0.9,right=0.82)
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        plt.savefig('%s_pdt_all' %(region_key))
        plt.clf()
        
        #####
        ax = plt.gca()
        plt.subplot(2,1,2)
        
        line0 = plt.plot(bins,np.mean(inv_transform_dict['rpt'][0,0,1,:],axis=0),label=labels[0],color=colors[0])
        line1 = plt.plot(bins,np.mean(inv_transform_dict['rpt'][1,0,1,:],axis=0),label=labels[1],color=colors[1])
        line2 = plt.plot(bins,np.mean(inv_transform_dict['rpt'][0,1,1,:],axis=0),label=labels[2],color=colors[2])
        line3 = plt.plot(bins,np.mean(inv_transform_dict['rpt'][1,1,1,:],axis=0),label=labels[3],color=colors[3])
        line4 = plt.plot(bins,np.mean(inv_transform_dict['rpt'][0,0,0,:],axis=0),label=labels[4],color=colors[4],linestyle='--')
        line5 = plt.plot(bins,np.mean(inv_transform_dict['rpt'][1,0,0,:],axis=0),label=labels[5],color=colors[5],linestyle='--')
        line6 = plt.plot(bins,np.mean(inv_transform_dict['rpt'][0,1,0,:],axis=0),label=labels[6],color=colors[6],linestyle='--')
        line7 = plt.plot(bins,np.mean(inv_transform_dict['rpt'][1,1,0,:],axis=0),label=labels[7],color=colors[7],linestyle='--')
        
        lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        #
        ax = plt.gca()
        plt.subplot(2,1,1)
        rp_labels = ['r0_p0','r0_px','rx_p0','rx_px']
        line0 = plt.plot(bins,np.mean(inv_transform_dict['rpt'],axis=(0,3))[0,0,:],label=rp_labels[0],color=colors[0])
        line1 = plt.plot(bins,np.mean(inv_transform_dict['rpt'],axis=(0,3))[0,1,:],label=rp_labels[1],color=colors[1])
        line2 = plt.plot(bins,np.mean(inv_transform_dict['rpt'],axis=(0,3))[1,0,:],label=rp_labels[2],color=colors[2])
        line3 = plt.plot(bins,np.mean(inv_transform_dict['rpt'],axis=(0,3))[1,1,:],label=rp_labels[3],color=colors[3])
        
        lines = np.squeeze(np.dstack((line0,line1,line2,line3)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.suptitle('All rpt',fontsize='small')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,rp_labels,loc='upper right',ncol=1,fontsize='small')
        plt.tight_layout(w_pad=0.1)
        plt.subplots_adjust(top=0.9,right=0.82)
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        plt.savefig('%s_rpt_all' %(region_key))
        plt.clf()
        
        #####
        ax = plt.gca()
        plt.subplot(2,1,2)
        
        line0 = plt.plot(bins,np.mean(inv_transform_dict['rpdt'][0,0,1,:],axis=0),label=labels[0],color=colors[0])
        line1 = plt.plot(bins,np.mean(inv_transform_dict['rpdt'][1,0,1,:],axis=0),label=labels[1],color=colors[1])
        line2 = plt.plot(bins,np.mean(inv_transform_dict['rpdt'][0,1,1,:],axis=0),label=labels[2],color=colors[2])
        line3 = plt.plot(bins,np.mean(inv_transform_dict['rpdt'][1,1,1,:],axis=0),label=labels[3],color=colors[3])
        line4 = plt.plot(bins,np.mean(inv_transform_dict['rpdt'][0,0,0,:],axis=0),label=labels[4],color=colors[4],linestyle='--')
        line5 = plt.plot(bins,np.mean(inv_transform_dict['rpdt'][1,0,0,:],axis=0),label=labels[5],color=colors[5],linestyle='--')
        line6 = plt.plot(bins,np.mean(inv_transform_dict['rpdt'][0,1,0,:],axis=0),label=labels[6],color=colors[6],linestyle='--')
        line7 = plt.plot(bins,np.mean(inv_transform_dict['rpdt'][1,1,0,:],axis=0),label=labels[7],color=colors[7],linestyle='--')
        
        lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,labels,loc='lower right',ncol=1,fontsize='small')
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        #
        plt.subplot(2,1,1)
        all_label = ['all']
        line0 = plt.plot(bins,np.mean(inv_transform_dict['rpdt'],axis=(0,1,2,3)),label=all_label,color='k')
        
        lines = line0
        
        plt.axvline(x=bfr_bins,color='g',linestyle='--')
        plt.axvline(x=bfr_bins+aft_bins,color='k')
        plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
        plt.suptitle('All rpdt',fontsize='small')
        plt.xticks(my_ticks_num,my_ticks)
        
        plt.figlegend(lines,t_labels,loc='upper right',ncol=1,fontsize='small')
        plt.tight_layout(w_pad=0.1)
        plt.subplots_adjust(top=0.9,right=0.82)
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        plt.savefig('%s_rpdt_all' %(region_key))
        plt.clf()
        
        
        #make plots avg over units

                    

	print 'plotting'
	
	for comb_ind,comb_val in Z.iteritems():
		ax = plt.gca()
		for i in range(components_plot):
			plt.subplot(components_plot/2,2,i+1)
			#if i < components_plot/2:
			#	plt.subplot(components_plot/2,2,i+1)
			#else:
			#	plt.subplot(components_plot/2,2,i+1+components_plot/2)
							
			line0 = plt.plot(bins,Z[comb_ind][i,0,0,1,:],label=labels[0],color=colors[0])
			line1 = plt.plot(bins,Z[comb_ind][i,1,0,1,:],label=labels[1],color=colors[1])
			line2 = plt.plot(bins,Z[comb_ind][i,0,1,1,:],label=labels[2],color=colors[2])
			line3 = plt.plot(bins,Z[comb_ind][i,1,1,1,:],label=labels[3],color=colors[3])

			line4 = plt.plot(bins,Z[comb_ind][i,0,0,0,:],label=labels[4],color=colors[4],linestyle=':')
			line5 = plt.plot(bins,Z[comb_ind][i,1,0,0,:],label=labels[5],color=colors[5],linestyle=':')
			line6 = plt.plot(bins,Z[comb_ind][i,0,1,0,:],label=labels[6],color=colors[6],linestyle=':')
			line7 = plt.plot(bins,Z[comb_ind][i,1,1,0,:],label=labels[7],color=colors[7],linestyle=':')
			lines = np.squeeze(np.dstack((line0,line1,line2,line3,line4,line5,line6,line7)))
			
			plt.axvline(x=bfr_bins,color='g',linestyle='--')
			plt.axvline(x=bfr_bins+aft_bins,color='k')
			plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
			plt.title('Component: %s' %(i+1),fontsize='small')
			plt.xticks(my_ticks_num,my_ticks)

		plt.figlegend(lines,labels,loc='right',ncol=1,fontsize='small')
		plt.tight_layout(w_pad=0.1)
		plt.subplots_adjust(top=0.9,right=0.8)
		plt.rcParams['xtick.labelsize'] = 8
		plt.rcParams['ytick.labelsize'] = 8
		plt.suptitle('Region %s, comb ind = %s' %(region_key,comb_ind))
		plt.savefig('component_binaryrp_%s_%s' %(region_key,comb_ind))
		plt.clf()

		if do_sig_analysis:
			sig_analysis = dpca.significance_analysis(all_avg,all_bal,full=True)
		transformed = dpca.transform(all_avg)

		if do_sig_analysis:
			dpca_results = {'Z':Z,'explained_var':explained_var,'sig_analysis':sig_analysis,'transformed':transformed,'all_avg':all_avg,'all_bal':all_bal}
		else:
			dpca_results = {'Z':Z,'explained_var':explained_var,'transformed':transformed,'all_avg':all_avg}
		
			
		all_dict[region_key]['dpca_results'] = dpca_results

all_dict['sort_dict'] = sort_dict
np.save('dpca_results_binaryrp.npy',all_dict)
