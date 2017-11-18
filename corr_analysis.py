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

zscore_bool = False
gaussian_bool = True







###############################
### functions #################
###############################
def hist_and_smooth_data(spike_vals):


	








	
	return()



def run_corr(binned_data):




	return()



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
condensed = np.zeros((np.shape(trial_breakdown)[0],8))

#0: disp_rp, 1: succ scene 2: failure scene, 3: rnum, 4: pnum, 5:succ/fail, 6: value, 7: motiv

condensed[:,0] = trial_breakdown[:,1]
condensed[:,1] = trial_breakdown[:,2]
condensed[:,2] = trial_breakdown[:,3]
condensed[:,3] = trial_breakdown[:,5]
condensed[:,4] = trial_breakdown[:,7]
condensed[:,5] = trial_breakdown[:,10]
#condensed[:,8] = trial_breakdown[   ] Next reset, don't need for now                                                                                                                                                                     

#delete end trials if not fully finished
if condensed[-1,1] == condensed[-1,2] == 0:
	new_condensed = condensed[0:-1,:]
        condensed = new_condensed
condensed = condensed[condensed[:,0] != 0]

#remove trials with now succ or failure scene (not sure why, but saw in one)
condensed = condensed[np.invert(np.logical_and(condensed[:,1] == 0, condensed[:,2] == 0))]

#TODOD FOR NOW remove catch trials
condensed = condensed[condensed[:,5] == 0]
#col 5 all 0s now, replace with succ/fail vector: succ = 1, fail = -1
condensed[condensed[:,1] != 0, 5] = 1
condensed[condensed[:,2] != 0, 5] = -1

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
#TODO printing S1 numbers more than once?
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

for spike_key,spike_val in spike_dict.iteritems():
	#hist and smooth first in a func (make bool for type of smooth)
	#then run corr func
	binned_data = hist_and_smooth_data(spike_val)

	corr_data = run_correlation(binned_data)
    



