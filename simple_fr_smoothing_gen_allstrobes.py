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
#from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import pandas as pd
from matplotlib import cm
import xlsxwriter
import scipy.stats as stats
from scipy import ndimage

#######################
#params to set ########
#######################

bin_size = 10 #in ms

zscore_bool = True
gaussian_bool = False
gauss_sigma = 30 #in ms

bfr_cue_time = 0.5
aft_cue_time = 1.0
bfr_result_time = 0.5
aft_result_time = 1.0


###############################
### functions #################
###############################
def hist_and_smooth_data(spike_data):

        max_spike_ts = 0
        for i in range(len(spike_data)):
                if np.amax(spike_data[i]) > max_spike_ts:
                        max_spike_ts = np.amax(spike_data[i])
        
        max_bin_num = int(np.ceil(max_spike_ts) / float(bin_size) * 1000)
        hist_data = np.zeros((len(spike_data),max_bin_num))
        hist_bins = np.zeros((len(spike_data),max_bin_num))
        for i in range(len(spike_data)):
                total_bin_range = np.arange(0,int(np.ceil(spike_data[i].max())),bin_size/1000.0)
                hist,bins = np.histogram(spike_data[i],bins=total_bin_range,range=(0,int(np.ceil(spike_data[i].max()))),normed=False,density=False)
                #pdb.set_trace()
                hist_data[i,0:len(hist)] = hist
                hist_bins[i,0:len(bins)] = bins

                #TODO fix so gaus divide by bin size and -> fr before smoothing
                #TODO make option for zscore and gaus togethre
        if zscore_bool and gaussian_bool:
                        smoothed = stats.zscore(hist_data,axis=1)
                        smoothed = ndimage.filters.gaussian_filter1d(smoothed,gauss_sigma,axis=1)
        elif zscore_bool:
                        smoothed = stats.zscore(hist_data,axis=1)
        elif gaussian_bool:

                        smoothed = ndimage.filters.gaussian_filter1d(hist_data,gauss_sigma,axis=1)
        else:
                        smoothed = {}

        return_dict = {'hist_data':hist_data,'hist_bins':hist_bins,'smoothed':smoothed}
        return(return_dict)
        
def run_breakdown(binned_data,condensed,region_key):
        last_ind = 0
        trim_bool = False

        bfr_cue_bins = int(bfr_cue_time * 1000/bin_size)
        aft_cue_bins = int(aft_cue_time * 1000/bin_size)
        bfr_res_bins = int(bfr_result_time * 1000/bin_size)
        aft_res_bins = int(aft_result_time * 1000/bin_size)

        all_cue_fr = np.zeros((np.shape(binned_data)[0],np.shape(condensed)[0],bfr_cue_bins + aft_cue_bins))
        all_res_fr = np.zeros((np.shape(binned_data)[0],np.shape(condensed)[0],bfr_res_bins + aft_res_bins))
        
        for unit_num in range(np.shape(binned_data)[0]):
                for i in range(np.shape(condensed)[0]):
                        try:
                                cue_temp = binned_data[unit_num,condensed[i,8]-bfr_cue_bins : condensed[i,8]+aft_cue_bins]
                                res_temp = binned_data[unit_num,condensed[i,9]-bfr_res_bins : condensed[i,9]+aft_res_bins]
                                all_cue_fr[unit_num,i,:] = cue_temp
                                all_res_fr[unit_num,i,:] = res_temp
                        except:
                                #if np.shape(condensed)[0] - i > 3:
                                #        pdb.set_trace()
                                #else:
                                #        last_ind = i
                                #        trim_bool = True
                                #        break
                                last_ind = i
                                trim_bool = True
                                break

        if trim_bool:
                all_cue_fr = all_cue_fr[:,0:last_ind,:]
                all_res_fr = all_res_fr[:,0:last_ind,:]
                
                condensed = condensed[0:last_ind,:]

        params = {'bfr_cue':bfr_cue_time,'aft_cue':aft_cue_time,'bfr_result':bfr_result_time,'aft_result':aft_result_time,'bin_size':bin_size}
        return_dict = {'all_cue_fr':all_cue_fr,'all_res_fr':all_res_fr,'condensed':condensed,'params':params}
        #sio.savemat('simple_output_%s' %(region_key),{'return_dict':return_dict},format='5')
        
        sio.savemat('simple_output_%s' %(region_key),{'return_dict':return_dict},format='5')


        return(return_dict)




###############################################
#start ########################################
###############################################

ts_filename = glob.glob('Extracted*_timestamps.mat')[0]
extracted_filename = ts_filename[:-15] + '.mat'

a = sio.loadmat(extracted_filename)
timestamps = sio.loadmat(ts_filename)

print extracted_filename

#create matrix of trial-by-trial info
trial_breakdown = timestamps['trial_breakdown']
condensed = np.zeros((np.shape(trial_breakdown)[0],14))

#0: disp_rp, 1: succ scene 2: failure scene, 3: rnum, 4: pnum, 5:succ/fail, 6: value, 7: motiv, 8: disp_rp bin

condensed[:,0] = trial_breakdown[:,1]
condensed[:,1] = trial_breakdown[:,2]
condensed[:,2] = trial_breakdown[:,3]
condensed[:,3] = trial_breakdown[:,5]
condensed[:,4] = trial_breakdown[:,7]
condensed[:,5] = trial_breakdown[:,10]

#condensed[:,8] = trial_breakdown[   ] Next reset, don't need for now          

#####new
condensed[:,10] = trial_breakdown[:,11]    #reach
condensed[:,11] = trial_breakdown[:,12]    #grasp
condensed[:,12] = trial_breakdown[:,13]    #transport
condensed[:,13] = trial_breakdown[:,14]    #release


bin_size_sec = bin_size / float(1000)
for i in range(np.shape(condensed)[0]):
        condensed[i,8] = int(np.around((round(condensed[i,0] / bin_size_sec) * bin_size_sec),decimals=2)/bin_size_sec)
        #condensed[i,8] = condensed[i,8].astype(int)
        condensed[i,9] = int(np.around((round((condensed[i,1] + condensed[i,2]) / bin_size_sec) * bin_size_sec),decimals=2)/bin_size_sec)
        #condensed[i,9] = condensed[i,9].astype(int)

#delete end trials if not fully finished
if condensed[-1,1] == condensed[-1,2] == 0:
	new_condensed = condensed[0:-1,:]
        condensed = new_condensed
condensed = condensed[condensed[:,0] != 0]

#remove trials with now succ or failure scene (not sure why, but saw in one)
condensed = condensed[np.invert(np.logical_and(condensed[:,1] == 0, condensed[:,2] == 0))]

#TODO if have both succ and punishment scene (look into why would occur, saw once)
condensed = condensed[np.invert(np.logical_and(condensed[:,1] != 0, condensed[:,2] != 0 ))]


#TODOD FOR NOW remove catch trials
condensed = condensed[condensed[:,5] == 0]
#col 5 all 0s now, replace with succ/fail vector: succ = 1, fail = 0
condensed[condensed[:,1] != 0, 5] = 1
condensed[condensed[:,2] != 0, 5] = 0

condensed[:,6] = condensed[:,3] - condensed[:,4]
condensed[:,7] = condensed[:,3] + condensed[:,4]


#Pull and arrange spike data
neural_data=a['neural_data']
Spikes = a['neural_data']['spikeTimes'];

#Break spikes into M1 S1 pmd 
M1_spikes = Spikes[0,0][0,1];
PmD_spikes = Spikes[0,0][0,0];
S1_spikes = Spikes[0,0][0,2];
#Find first channel count for pmv on map1
#PmV requires extra processing to generate as the data is the last 32 channels of each MAP system
unit_names={}
M1_unit_names = []
for i in range(0,M1_spikes.shape[1]):
    if int(M1_spikes[0,i]['signame'][0][0][0][3:-1]) > 96:
        M1_limit = i;
        print ("The number of units in M1 is: %s"%(M1_limit))
        break
    elif (int(M1_spikes[0,i]['signame'][0][0][0][3:-1]) == 96):
        M1_limit = i;
        print ("The number of units in M1 is: %s"%(M1_limit))
    M1_unit_names.append(M1_spikes[0,i]['signame'][0][0][0])
if not 'M1_limit' in locals():
	M1_limit = i
	print ("The number of units in M1 is: %s"%(M1_limit))
dummy = [];
#M1_limit not defined for 0526_0059, blocks 1 and 2
for i in range(M1_limit,M1_spikes.shape[1]):
    dummy.append(M1_spikes[0,i]['ts'][0,0][0])
unit_names['M1_unit_names']=M1_unit_names
#Find first channel count for pmv on map3
S1_unit_names = []  
for i in range(0,S1_spikes.shape[1]):
    if int(S1_spikes[0,i]['signame'][0][0][0][3:-1]) > 96:
        #if int(S1_spikes[0,i]['signame'][0][0][0][5:-1]) > 96:
        S1_limit = i;
        #print S1_limit
        print ("The number of units in S1 is: %s"%(S1_limit))
        break
    elif (int(S1_spikes[0,i]['signame'][0][0][0][3:-1]) == 96):
        S1_limit = i;
        #print S1_limit
        print ('The number of units in S1 is: %s' %(S1_limit))
    S1_unit_names.append(S1_spikes[0,i]['signame'][0][0][0])
if not 'S1_limit' in locals():
	S1_limit = i
	print ("The number of units in S1 is: %s"%(M1_limit))
for i in range(S1_limit,S1_spikes.shape[1]):
    dummy.append(S1_spikes[0,i]['ts'][0,0][0])
unit_names['S1_unit_names']=S1_unit_names
#Find first channel count for pmv on map2
pmd_unit_names = []
for i in range(0,PmD_spikes.shape[1]):
    if int(PmD_spikes[0,i]['signame'][0][0][0][3:-1]) > 96:
        PmD_limit =i;
        print ("The number of units in PmD is: %s"%(PmD_limit))
        break
    elif (int(PmD_spikes[0,i]['signame'][0][0][0][3:-1]) == 96):
        PmD_limit = i;
        print ("The number of units in PmD is: %s"%(PmD_limit))
    pmd_unit_names.append(PmD_spikes[0,i]['signame'][0][0][0])
if not 'pmd_limit' in locals():
	PmD_limit = i
	print ("The number of units in PmD is: %s"%(M1_limit))
for i in range(PmD_limit,PmD_spikes.shape[1]):
    dummy.append(PmD_spikes[0,i]['ts'][0,0][0])
#Generate Pmv Spiking data
PmV_spikes = np.array(dummy);
print("The number of units in PmV is: %s"%(len(PmV_spikes)))
unit_names['pmd_unit_names']=pmd_unit_names

#trim M1,S1,and PmD spikes to remove PmV spikes trailing at end; nerual data pruning now complete
M1_spikes = M1_spikes[0,0:M1_limit];
S1_spikes = S1_spikes[0,0:S1_limit];
PmD_spikes = PmD_spikes[0,0:PmD_limit];

spike_dict = {'M1':M1_spikes,'S1':S1_spikes,'PmD':PmD_spikes}

data_dict = {'M1':{},'S1':{},'PmD':{}}
for key,value in data_dict.iteritems():
        print 'running %s' %(key)
        spike_data = []
        for i in range(len(spike_dict[key])):
                spike_data.append(spike_dict[key][i]['ts'][0,0][0])

        #print 'binning and smoothing'
        binned_data = hist_and_smooth_data(spike_data)
       
        if gaussian_bool or zscore_bool:
                output = run_breakdown(binned_data['smoothed'],condensed,key)
        else:
                output = run_breakdown(binned_data['hist_data'],condensed,key)



#cue to reach
cue_reach = condensed[:,11] - condensed[:,0]

#num failed before reach

#cue to grasp
cue_grasp = condensed[:,12] - condensed[:,0]

#cue to transport
cue_transport = condensed[:,13] - condensed[:,0]

#cue to fail times
cue_fail = condensed[:,2] - condensed[:,0]
cue_fail = cue_fail[condensed[:,5] != 1]

#result to next cue
temp = np.zeros(np.shape(condensed[:,0]))
temp[0:-2] = condensed[1:-1,0]
res_time = condensed[:,1] + condensed[:,2]

result_next_cue = temp - res_time

#result_next_cue = result_next_cue[condensed[:,5] != 1]


save_dict = {'cue_reach':cue_reach,'cue_grasp':cue_grasp,'cue_transport':cue_transport,'cue_fail':cue_fail,'result_next_cue':result_next_cue,'condensed':condensed}

np.save("times.npy",save_dict)
