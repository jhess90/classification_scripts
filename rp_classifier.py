#!/usr/bin/env python

#import packages
import scipy.io as sio
import h5py
import numpy as np
import Tkinter as tk
import tkFileDialog
import matplotlib.patches as mpatches
from sklearn.decomposition import RandomizedPCA
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
import pdb
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import sys
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#######################
#params to set ########
#######################

pca_num = 150
no_bins = 10
plot_bool = False

#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-12-48-52.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-13-02-45.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-12-48-19.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-13-10-44.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-14-23.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-33-52.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-25-20.mat'
filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-46-25.mat'

######################
print filename
a = sio.loadmat(filename);

#Pull reward timestamps. Separate timespamp file must exist in folder containing neural data, with same name+_timestamps
print filename[:-4]+"_timestamps"+filename[-4:]
Timestamps = sio.loadmat(filename[:-4]+"_timestamps"+filename[-4:]);

#Timestamp generation: generates arrays of timestamps depending on reward level, including offset for cue presentation and reward delivery
#one reward cue, one punishment cue, successful (rewarding)
rp_s_rdelivery = Timestamps['rp_s_rdelivery_ts'][0];
rp_s_cue = Timestamps['rp_s_cue_ts'][0];
print ('rp_s trials: %s'%(len(rp_s_rdelivery)))
#in case there happened to not be any particular trial during a block, this makes the arrays dimensions equal. TODO look into better way instead of creating a timepoint
if not rp_s_rdelivery.size:
	rp_s_rdelivery = np.array([0])
	rp_s_cue = np.array([0])

#one reward cue, one punishment cue, unsuccessful (punishing)
rp_f_pdelivery = Timestamps['rp_f_pdelivery_ts'][0];
rp_f_cue = Timestamps['rp_f_cue_ts'][0];
print ('rp_f trials: %s'%(len(rp_f_pdelivery)))
if not rp_f_pdelivery.size:
	rp_f_pdelivery = np.array([0])
	rp_f_cue = np.array([0])

#no reward cue, no punishment cue, successful (non-rewarding)
nrnp_s_nextreset = Timestamps['nrnp_s_nextreset'][0];
nrnp_s_cue = Timestamps['nrnp_s_cue_ts'][0];
print ('nrnp_s trials: %s'%(len(nrnp_s_cue)))
if not nrnp_s_nextreset.size:
	nrnp_s_nextreset = np.array([0])
	nrnp_s_cue = np.array([0])

#no reward cue, no punishment cue, unsuccessful (non-punishing)
nrnp_f_nextreset = Timestamps['nrnp_f_nextreset'][0];
nrnp_f_cue = Timestamps['nrnp_f_cue_ts'][0];
print ('nrnp_f trials: %s'%(len(nrnp_f_cue)))
if not nrnp_f_nextreset.size:
	nrnp_f_nextreset = np.array([0])
	nrnp_f_cue = np.array([0])

#reward cue only (no punishment cue), successful (rewarding)
r_only_s_rdelivery = Timestamps['r_only_s_rdelivery_ts'][0];
r_only_s_cue = Timestamps['r_only_s_cue_ts'][0];
print ('r_only_s trials: %s'%(len(r_only_s_cue)))
if not r_only_s_rdelivery.size:
	r_only_s_rdelivery = np.array([0])
	r_only_s_cue = np.array([0])

#reward cue only (no punishment cue), unsuccessful (non-rewarding)
r_only_f_nextreset = Timestamps['r_only_f_nextreset'][0];
r_only_f_cue = Timestamps['r_only_f_cue_ts'][0];
print ('r_only_f trials: %s'%(len(r_only_f_cue)))
if not r_only_f_nextreset.size:
	r_only_f_nextreset = np.array([0])
	r_only_f_cue = np.array([0])

#punishment cue only (no reward cue), successful (non-punishing)
p_only_s_nextreset = Timestamps['p_only_s_nextreset'][0];
p_only_s_cue = Timestamps['p_only_s_cue_ts'][0];
print ('p_only_s trials: %s'%(len(p_only_s_cue)))
if not p_only_s_nextreset.size:
	p_only_s_nextreset = np.array([0])
	p_only_s_cue = np.array([0])

#punishment cue only (no reward cue), unsuccessful (punishing)
p_only_f_pdelivery = Timestamps['p_only_f_pdelivery_ts'][0];
p_only_f_cue = Timestamps['p_only_f_cue_ts'][0];
print ('p_only_f trials: %s'%(len(p_only_f_cue)))
if not  p_only_f_pdelivery.size:
	p_only_f_pdelivery = np.array([0])
	p_only_f_cue = np.array([0])

#catch trials- sometimes empty, so that's why there's the rcatch/pcatch bools
#reward cue, successfully completed, but no reward delivered
r_s_catch_cue = Timestamps['r_s_catch_cue_ts'][0];
r_s_catch_nextreset = Timestamps['r_s_catch_nextreset'][0];
if not r_s_catch_cue.size:
	print ('No r_s catch trials')
	rcatch_bool = False
else:
	print ('r_s catch trials: %s' %(len(r_s_catch_cue)))
	rcatch_bool = True

#punishment cue, successfully completed, but no reward delivered
p_f_catch_cue = Timestamps['p_f_catch_cue_ts'][0];
p_f_catch_nextreset = Timestamps['p_f_catch_nextreset'][0];
if not p_f_catch_cue.size:
	print ('No p_f catch trials')
	pcatch_bool = False
else:
	print ('p_f catch trials: %s' %(len(p_f_catch_cue)))
	pcatch_bool = True

#Pull neural spikes

neural_data=a['neural_data']
#print neural_data

Spikes = a['neural_data']['spikeTimes'];
#Break spikes into M1 S1 pmd 
#Different for jack/dave- jack has MAP1=PMd, MAP2=M1, MAP3=S1, ALL=PMv
M1_spikes = Spikes[0,0][0,1];
PmD_spikes = Spikes[0,0][0,0];
S1_spikes = Spikes[0,0][0,2];
#Find first channel count for pmv on map1
#PmV is requires extra processing to generate as the data is the last 32 channels of each MAP system
unit_names={}
M1_unit_names = []
for i in range(0,M1_spikes.shape[1]):
    if int(M1_spikes[0,i]['signame'][0][0][0][3:-1]) > 96:
        M1_limit = i;
        print ("The number of units in M1 is: %s"%(M1_limit))
        break
    M1_unit_names.append(M1_spikes[0,i]['signame'][0][0][0])
dummy = [];
for i in range(M1_limit,M1_spikes.shape[1]):
    dummy.append(M1_spikes[0,i]['ts'][0,0][0])
unit_names['M1_unit_names']=M1_unit_names
#Find first channel count for pmv on map3
S1_unit_names = []  
for i in range(0,S1_spikes.shape[1]):
    if int(S1_spikes[0,i]['signame'][0][0][0][3:-1]) > 96:
        S1_limit =i;
        print ("The number of units in S1 is: %s"%(S1_limit))
        break
    S1_unit_names.append(S1_spikes[0,i]['signame'][0][0][0])
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
    pmd_unit_names.append(PmD_spikes[0,i]['signame'][0][0][0])
for i in range(PmD_limit,PmD_spikes.shape[1]):
    dummy.append(PmD_spikes[0,i]['ts'][0,0][0])
#Generate Pmv Spiking data
PmV_spikes = np.array(dummy);
print("The number of units in PmV is: %s"%(len(PmV_spikes)))
unit_names['pmd_unit_names']=pmd_unit_names
#PSTH input parameters
#no_bins = 10;

print ('%s bins (50 ms each)\n%s pca components\n2 LDA components' %(no_bins,pca_num))

#trim M1,S1,and PmD spikes to remove PmV spikes trailing at end; nerual data pruning now complete
M1_spikes = M1_spikes[0,0:M1_limit];
S1_spikes = S1_spikes[0,0:S1_limit];
PmD_spikes = PmD_spikes[0,0:PmD_limit];
#generate time intervals of interest for PSTH generation

#time_boundry={'-0.5-0.0':[-0.5,0.0],'0-0.5':[0.0,0.5], '0.5-1.0':[0.5,1.0], '1.0-1.5':[1.0,1.5], '1.5-2.0':[1.5,2.0],'all':[-0.5,2.0]}
#time_boundry={'-0.5-0.0':[-0.5,0.0],'0-0.5':[0.0,0.5],'0.5-1.0':[0.5,1.0],'all_incpre':[-0.5,2.0]}
time_boundry={'-0.5-0.0':[-0.5,0.0],'0-0.5':[0.0,0.5],'0.5-1.0':[0.5,1.0],'all':[0.0,2.0]}


#accuracy_total = {}
all_accuracy_total = {}
#Begin main loop of analysis
for name_of_bin,time_of_bin in time_boundry.iteritems():
    before_time = -time_of_bin[0]
    after_time = time_of_bin[1]
    print( 'bin %s' %(name_of_bin))
    #Build and save M1 PSTH

    if name_of_bin == 'all':
        #to keep 50 ms bins
        no_bins = 50
    else:
        no_bins = 10
        
    #Function that builds histograms of M1,S1,PmD spikes based on parameters entered above
    def Build_hist(data,timestamps,no_bins,time_before,time_after):
		dummy = []
		for i in range(0,timestamps.shape[0]):
			dummy1=[]
			b = timestamps[i]
			for ii in range(data.shape[0]):
				a = data[ii]['ts'][0,0][0]
				c = (np.histogram(a[np.where(np.logical_and(a>=b-time_before,a<=b+time_after))],bins=no_bins,normed=False,density=False))
				dummy1.append(np.nan_to_num(c[0]))
			dummy.append(MaxAbsScaler().fit_transform(dummy1))
		#print np.array(dummy).shape
		return np.array(dummy)
    
    #Build PmV histograms
    def Build_hist_PmV(data,timestamps,no_bins,time_before,time_after):
        dummy = []
        for i in range(0,timestamps.shape[0]):
            dummy1=[]
            b = timestamps[i]
            for ii in range(data.shape[0]):
                a = data[ii]
                c = (np.histogram(a[np.where(np.logical_and(a>=b-time_before,a<=b+time_after))],bins=no_bins,normed=False,density=False))
                dummy1.append(np.nan_to_num(c[0]))
            dummy.append(MaxAbsScaler().fit_transform(dummy1))
            #plt.hist(dummy)
            #plt.show()
        return np.array(dummy)
    
    #Assemble a final dictionary of the final dataset for saving
    def Make_final_data(data,start_position):
        dummy = []
        for i in range(0,data[start_position].shape[0]):
            dummy.append(np.hstack(data[start_position][i]))
        rp_s_data = np.array(dummy)
        rp_s_targets = np.ones(rp_s_data.shape[0])*0
        dummy = []
        for i in range(0,data[start_position+8].shape[0]):
            dummy.append(np.hstack(data[start_position+8][i]))
        rp_f_data = np.array(dummy)
        rp_f_targets = np.ones(rp_f_data.shape[0])*1
        dummy = []
        for i in range(0,data[start_position+16].shape[0]):
            dummy.append(np.hstack(data[start_position+16][i]))
        nrnp_s_data = np.array(dummy)
        nrnp_s_targets = np.ones(nrnp_s_data.shape[0])*2
        dummy = []
        for i in range(0,data[start_position+24].shape[0]):
            dummy.append(np.hstack(data[start_position+24][i]))
        nrnp_f_data = np.array(dummy)
        nrnp_f_targets = np.ones(nrnp_f_data.shape[0])*3
        dummy = []
        for i in range(0,data[start_position+32].shape[0]):
            dummy.append(np.hstack(data[start_position+32][i]))
        r_only_s_data = np.array(dummy)
        r_only_s_targets = np.ones(r_only_s_data.shape[0])*4
        dummy = []
        for i in range(0,data[start_position+40].shape[0]):
            dummy.append(np.hstack(data[start_position+40][i]))
        r_only_f_data = np.array(dummy)
        r_only_f_targets = np.ones(r_only_f_data.shape[0])*5
        dummy = []
        for i in range(0,data[start_position+48].shape[0]):
            dummy.append(np.hstack(data[start_position+48][i]))
        p_only_s_data = np.array(dummy)
        p_only_s_targets = np.ones(p_only_s_data.shape[0])*6
        dummy = []
        for i in range(0,data[start_position+56].shape[0]):
            dummy.append(np.hstack(data[start_position+56][i]))
        p_only_f_data = np.array(dummy)
        p_only_f_targets = np.ones(p_only_f_data.shape[0])*7


		
#        return (np.vstack([rp_s_data,rp_f_data,nrnp_s_data,nrnp_f_data,r_only_s_data,r_only_f_data,p_only_s_data,p_only_f_data]),np.hstack([rp_s_targets,rp_f_targets,nrnp_s_targets,nrnp_f_targets,r_only_s_targets,r_only_f_targets,p_only_s_targets,p_only_f_targets]))
        return(np.vstack([rp_s_data,rp_f_data,nrnp_s_data,nrnp_f_data,r_only_s_data,r_only_f_data,p_only_s_data,p_only_f_data]),np.hstack([rp_s_targets,rp_f_targets,nrnp_s_targets,nrnp_f_targets,r_only_s_targets,r_only_f_targets,p_only_s_targets,p_only_f_targets]))

    #one reward cue, one punishment cue, successful (rewarding)
    M1_rp_s_rdelivery_hists = Build_hist(M1_spikes,rp_s_rdelivery,no_bins,before_time,after_time)
    S1_rp_s_rdelivery_hists = Build_hist(S1_spikes,rp_s_rdelivery,no_bins,before_time,after_time)
    PmD_rp_s_rdelivery_hists = Build_hist(PmD_spikes,rp_s_rdelivery,no_bins,before_time,after_time)
    PmV_rp_s_rdelivery_hists = Build_hist_PmV(PmV_spikes,rp_s_rdelivery,no_bins,before_time,after_time)

    M1_rp_s_cue_hists = Build_hist(M1_spikes,rp_s_cue,no_bins,before_time,after_time)
    S1_rp_s_cue_hists = Build_hist(S1_spikes,rp_s_cue,no_bins,before_time,after_time)
    PmD_rp_s_cue_hists = Build_hist(PmD_spikes,rp_s_cue,no_bins,before_time,after_time)
    PmV_rp_s_cue_hists = Build_hist_PmV(PmV_spikes,rp_s_cue,no_bins,before_time,after_time)

    #one reward cue, one punishment cue, unsuccessful (punishing)
    M1_rp_f_pdelivery_hists = Build_hist(M1_spikes,rp_f_pdelivery,no_bins,before_time,after_time)
    S1_rp_f_pdelivery_hists = Build_hist(S1_spikes,rp_f_pdelivery,no_bins,before_time,after_time)
    PmD_rp_f_pdelivery_hists = Build_hist(PmD_spikes,rp_f_pdelivery,no_bins,before_time,after_time)
    PmV_rp_f_pdelivery_hists = Build_hist_PmV(PmV_spikes,rp_f_pdelivery,no_bins,before_time,after_time)

    M1_rp_f_cue_hists = Build_hist(M1_spikes,rp_f_cue,no_bins,before_time,after_time)
    S1_rp_f_cue_hists = Build_hist(S1_spikes,rp_f_cue,no_bins,before_time,after_time)
    PmD_rp_f_cue_hists = Build_hist(PmD_spikes,rp_f_cue,no_bins,before_time,after_time)
    PmV_rp_f_cue_hists = Build_hist_PmV(PmV_spikes,rp_f_cue,no_bins,before_time,after_time)

    #no reward cue, no punishment cue, successful (non-rewarding)
    M1_nrnp_s_nextreset_hists = Build_hist(M1_spikes,nrnp_s_nextreset,no_bins,before_time,after_time)
    S1_nrnp_s_nextreset_hists = Build_hist(S1_spikes,nrnp_s_nextreset,no_bins,before_time,after_time)
    PmD_nrnp_s_nextreset_hists = Build_hist(PmD_spikes,nrnp_s_nextreset,no_bins,before_time,after_time)
    PmV_nrnp_s_nextreset_hists = Build_hist_PmV(PmV_spikes,nrnp_s_nextreset,no_bins,before_time,after_time)

    M1_nrnp_s_cue_hists = Build_hist(M1_spikes,nrnp_s_cue,no_bins,before_time,after_time)
    S1_nrnp_s_cue_hists = Build_hist(S1_spikes,nrnp_s_cue,no_bins,before_time,after_time)
    PmD_nrnp_s_cue_hists = Build_hist(PmD_spikes,nrnp_s_cue,no_bins,before_time,after_time)
    PmV_nrnp_s_cue_hists = Build_hist_PmV(PmV_spikes,nrnp_s_cue,no_bins,before_time,after_time)

    #no reward cue, no punishment cue, fail (non-punishing)
    M1_nrnp_f_nextreset_hists = Build_hist(M1_spikes,nrnp_f_nextreset,no_bins,before_time,after_time)
    S1_nrnp_f_nextreset_hists = Build_hist(S1_spikes,nrnp_f_nextreset,no_bins,before_time,after_time)
    PmD_nrnp_f_nextreset_hists = Build_hist(PmD_spikes,nrnp_f_nextreset,no_bins,before_time,after_time)
    PmV_nrnp_f_nextreset_hists = Build_hist_PmV(PmV_spikes,nrnp_f_nextreset,no_bins,before_time,after_time)

    M1_nrnp_f_cue_hists = Build_hist(M1_spikes,nrnp_f_cue,no_bins,before_time,after_time)
    S1_nrnp_f_cue_hists = Build_hist(S1_spikes,nrnp_f_cue,no_bins,before_time,after_time)
    PmD_nrnp_f_cue_hists = Build_hist(PmD_spikes,nrnp_f_cue,no_bins,before_time,after_time)
    PmV_nrnp_f_cue_hists = Build_hist_PmV(PmV_spikes,nrnp_f_cue,no_bins,before_time,after_time)

    #reward cue only, successful (rewarding)
    M1_r_only_s_rdelivery_hists = Build_hist(M1_spikes,r_only_s_rdelivery,no_bins,before_time,after_time)
    S1_r_only_s_rdelivery_hists = Build_hist(S1_spikes,r_only_s_rdelivery,no_bins,before_time,after_time)
    PmD_r_only_s_rdelivery_hists = Build_hist(PmD_spikes,r_only_s_rdelivery,no_bins,before_time,after_time)
    PmV_r_only_s_rdelivery_hists = Build_hist_PmV(PmV_spikes,r_only_s_rdelivery,no_bins,before_time,after_time)

    M1_r_only_s_cue_hists = Build_hist(M1_spikes,r_only_s_cue,no_bins,before_time,after_time)
    S1_r_only_s_cue_hists = Build_hist(S1_spikes,r_only_s_cue,no_bins,before_time,after_time)
    PmD_r_only_s_cue_hists = Build_hist(PmD_spikes,r_only_s_cue,no_bins,before_time,after_time)
    PmV_r_only_s_cue_hists = Build_hist_PmV(PmV_spikes,r_only_s_cue,no_bins,before_time,after_time)

    #reward cue only, unsuccesssul (non-rewarding)
    M1_r_only_f_nextreset_hists = Build_hist(M1_spikes,r_only_f_nextreset,no_bins,before_time,after_time)
    S1_r_only_f_nextreset_hists = Build_hist(S1_spikes,r_only_f_nextreset,no_bins,before_time,after_time)
    PmD_r_only_f_nextreset_hists = Build_hist(PmD_spikes,r_only_f_nextreset,no_bins,before_time,after_time)
    PmV_r_only_f_nextreset_hists = Build_hist_PmV(PmV_spikes,r_only_f_nextreset,no_bins,before_time,after_time)

    M1_r_only_f_cue_hists = Build_hist(M1_spikes,r_only_f_cue,no_bins,before_time,after_time)
    S1_r_only_f_cue_hists = Build_hist(S1_spikes,r_only_f_cue,no_bins,before_time,after_time)
    PmD_r_only_f_cue_hists = Build_hist(PmD_spikes,r_only_f_cue,no_bins,before_time,after_time)
    PmV_r_only_f_cue_hists = Build_hist_PmV(PmV_spikes,r_only_f_cue,no_bins,before_time,after_time)

    #punishment cue only, successful (non-punishing)
    M1_p_only_s_nextreset_hists = Build_hist(M1_spikes,p_only_s_nextreset,no_bins,before_time,after_time)
    S1_p_only_s_nextreset_hists = Build_hist(S1_spikes,p_only_s_nextreset,no_bins,before_time,after_time)
    PmD_p_only_s_nextreset_hists = Build_hist(PmD_spikes,p_only_s_nextreset,no_bins,before_time,after_time)
    PmV_p_only_s_nextreset_hists = Build_hist_PmV(PmV_spikes,p_only_s_nextreset,no_bins,before_time,after_time)

    M1_p_only_s_cue_hists = Build_hist(M1_spikes,p_only_s_cue,no_bins,before_time,after_time)
    S1_p_only_s_cue_hists = Build_hist(S1_spikes,p_only_s_cue,no_bins,before_time,after_time)
    PmD_p_only_s_cue_hists = Build_hist(PmD_spikes,p_only_s_cue,no_bins,before_time,after_time)
    PmV_p_only_s_cue_hists = Build_hist_PmV(PmV_spikes,p_only_s_cue,no_bins,before_time,after_time)

    #punishment cue only, unsuccessful (punishing)
    M1_p_only_f_pdelivery_hists = Build_hist(M1_spikes,p_only_f_pdelivery,no_bins,before_time,after_time)
    S1_p_only_f_pdelivery_hists = Build_hist(S1_spikes,p_only_f_pdelivery,no_bins,before_time,after_time)
    PmD_p_only_f_pdelivery_hists = Build_hist(PmD_spikes,p_only_f_pdelivery,no_bins,before_time,after_time)
    PmV_p_only_f_pdelivery_hists = Build_hist_PmV(PmV_spikes,p_only_f_pdelivery,no_bins,before_time,after_time)

    M1_p_only_f_cue_hists = Build_hist(M1_spikes,p_only_f_cue,no_bins,before_time,after_time)
    S1_p_only_f_cue_hists = Build_hist(S1_spikes,p_only_f_cue,no_bins,before_time,after_time)
    PmD_p_only_f_cue_hists = Build_hist(PmD_spikes,p_only_f_cue,no_bins,before_time,after_time)
    PmV_p_only_f_cue_hists = Build_hist_PmV(PmV_spikes,p_only_f_cue,no_bins,before_time,after_time)#reward cue, successful, NO reward delivery (catch trial)

    if rcatch_bool or pcatch_bool:
        M1_r_s_catch_nextreset_hists = Build_hist(M1_spikes,r_s_catch_nextreset,no_bins,before_time,after_time)
        S1_r_s_catch_nextreset_hists = Build_hist(M1_spikes,r_s_catch_nextreset,no_bins,before_time,after_time)
        PmD_r_s_catch_nextreset_hists = Build_hist(M1_spikes,r_s_catch_nextreset,no_bins,before_time,after_time)
        PmV_r_s_catch_nextreset_hists = Build_hist(M1_spikes,r_s_catch_nextreset,no_bins,before_time,after_time)

        M1_r_s_catch_cue_hists = Build_hist(M1_spikes,r_s_catch_cue,no_bins,before_time,after_time)
        S1_r_s_catch_cue_hists = Build_hist(M1_spikes,r_s_catch_cue,no_bins,before_time,after_time)
        PmD_r_s_catch_cue_hists = Build_hist(M1_spikes,r_s_catch_cue,no_bins,before_time,after_time)
        PmV_r_s_catch_cue_hists = Build_hist(M1_spikes,r_s_catch_cue,no_bins,before_time,after_time)

        #punishment cue, unsuccessful, NO punishment delivery (catch trial)
        #if pcatch_bool:
        M1_p_f_catch_nextreset_hists = Build_hist(M1_spikes,p_f_catch_nextreset,no_bins,before_time,after_time)
        S1_p_f_catch_nextreset_hists = Build_hist(M1_spikes,p_f_catch_nextreset,no_bins,before_time,after_time)
        PmD_p_f_catch_nextreset_hists = Build_hist(M1_spikes,p_f_catch_nextreset,no_bins,before_time,after_time)
        PmV_p_f_catch_nextreset_hists = Build_hist(M1_spikes,p_f_catch_nextreset,no_bins,before_time,after_time)

        M1_p_f_catch_cue_hists = Build_hist(M1_spikes,p_f_catch_cue,no_bins,before_time,after_time)
        S1_p_f_catch_cue_hists = Build_hist(M1_spikes,p_f_catch_cue,no_bins,before_time,after_time)
        PmD_p_f_catch_cue_hists = Build_hist(M1_spikes,p_f_catch_cue,no_bins,before_time,after_time)
        PmV_p_f_catch_cue_hists = Build_hist(M1_spikes,p_f_catch_cue,no_bins,before_time,after_time)
    
    #This list is a consortium of all of the reward histograms, both cue and reward for each region at each level of reward
    #no overlapping data points
    pnt_data_2 = [M1_rp_s_rdelivery_hists, S1_rp_s_rdelivery_hists, PmD_rp_s_rdelivery_hists, PmV_rp_s_rdelivery_hists, M1_rp_s_cue_hists, S1_rp_s_cue_hists, PmD_rp_s_cue_hists, PmV_rp_s_cue_hists, M1_rp_f_pdelivery_hists, S1_rp_f_pdelivery_hists, PmD_rp_f_pdelivery_hists, PmV_rp_f_pdelivery_hists, M1_rp_f_cue_hists, S1_rp_f_cue_hists, PmD_rp_f_cue_hists, PmV_rp_f_cue_hists, M1_nrnp_s_nextreset_hists, S1_nrnp_s_nextreset_hists, PmD_nrnp_s_nextreset_hists, PmV_nrnp_s_nextreset_hists, M1_nrnp_s_cue_hists, S1_nrnp_s_cue_hists, PmD_nrnp_s_cue_hists, PmV_nrnp_s_cue_hists, M1_nrnp_f_nextreset_hists, S1_nrnp_f_nextreset_hists, PmD_nrnp_f_nextreset_hists, PmV_nrnp_f_nextreset_hists, M1_nrnp_f_cue_hists, S1_nrnp_f_cue_hists, PmD_nrnp_f_cue_hists, PmV_nrnp_f_cue_hists, M1_r_only_s_rdelivery_hists, S1_r_only_s_rdelivery_hists, PmD_r_only_s_rdelivery_hists, PmV_r_only_s_rdelivery_hists, M1_r_only_s_cue_hists, S1_r_only_s_cue_hists, PmD_r_only_s_cue_hists, PmV_r_only_s_cue_hists, M1_r_only_f_nextreset_hists, S1_r_only_f_nextreset_hists, PmD_r_only_f_nextreset_hists, PmV_r_only_f_nextreset_hists, M1_r_only_f_cue_hists, S1_r_only_f_cue_hists, PmD_r_only_f_cue_hists, PmV_r_only_f_cue_hists, M1_p_only_s_nextreset_hists, S1_p_only_s_nextreset_hists, PmD_p_only_s_nextreset_hists, PmV_p_only_s_nextreset_hists, M1_p_only_s_cue_hists, S1_p_only_s_cue_hists, PmD_p_only_s_cue_hists, PmV_p_only_s_cue_hists, M1_p_only_f_pdelivery_hists, S1_p_only_f_pdelivery_hists, PmD_p_only_f_pdelivery_hists, PmV_p_only_f_pdelivery_hists, M1_p_only_f_cue_hists, S1_p_only_f_cue_hists, PmD_p_only_f_cue_hists, PmV_p_only_f_cue_hists]
    if pcatch_bool or rcatch_bool:
        pnt_data_2 = [M1_rp_s_rdelivery_hists, S1_rp_s_rdelivery_hists, PmD_rp_s_rdelivery_hists, PmV_rp_s_rdelivery_hists, M1_rp_s_cue_hists, S1_rp_s_cue_hists, PmD_rp_s_cue_hists, PmV_rp_s_cue_hists, M1_rp_f_pdelivery_hists, S1_rp_f_pdelivery_hists, PmD_rp_f_pdelivery_hists, PmV_rp_f_pdelivery_hists, M1_rp_f_cue_hists, S1_rp_f_cue_hists, PmD_rp_f_cue_hists, PmV_rp_f_cue_hists, M1_nrnp_s_nextreset_hists, S1_nrnp_s_nextreset_hists, PmD_nrnp_s_nextreset_hists, PmV_nrnp_s_nextreset_hists, M1_nrnp_s_cue_hists, S1_nrnp_s_cue_hists, PmD_nrnp_s_cue_hists, PmV_nrnp_s_cue_hists, M1_nrnp_f_nextreset_hists, S1_nrnp_f_nextreset_hists, PmD_nrnp_f_nextreset_hists, PmV_nrnp_f_nextreset_hists, M1_nrnp_f_cue_hists, S1_nrnp_f_cue_hists, PmD_nrnp_f_cue_hists, PmV_nrnp_f_cue_hists, M1_r_only_s_rdelivery_hists, S1_r_only_s_rdelivery_hists, PmD_r_only_s_rdelivery_hists, PmV_r_only_s_rdelivery_hists, M1_r_only_s_cue_hists, S1_r_only_s_cue_hists, PmD_r_only_s_cue_hists, PmV_r_only_s_cue_hists, M1_r_only_f_nextreset_hists, S1_r_only_f_nextreset_hists, PmD_r_only_f_nextreset_hists, PmV_r_only_f_nextreset_hists, M1_r_only_f_cue_hists, S1_r_only_f_cue_hists, PmD_r_only_f_cue_hists, PmV_r_only_f_cue_hists, M1_p_only_s_nextreset_hists, S1_p_only_s_nextreset_hists, PmD_p_only_s_nextreset_hists, PmV_p_only_s_nextreset_hists, M1_p_only_s_cue_hists, S1_p_only_s_cue_hists, PmD_p_only_s_cue_hists, PmV_p_only_s_cue_hists, M1_p_only_f_pdelivery_hists, S1_p_only_f_pdelivery_hists, PmD_p_only_f_pdelivery_hists, PmV_p_only_f_pdelivery_hists, M1_p_only_f_cue_hists, S1_p_only_f_cue_hists, PmD_p_only_f_cue_hists, PmV_p_only_f_cue_hists, M1_r_s_catch_nextreset_hists, S1_r_s_catch_nextreset_hists, PmD_r_s_catch_nextreset_hists, PmV_r_s_catch_nextreset_hists, M1_r_s_catch_cue_hists, S1_r_s_catch_cue_hists, PmD_r_s_catch_cue_hists, PmV_r_s_catch_cue_hists, M1_p_f_catch_nextreset_hists, S1_p_f_catch_nextreset_hists, PmD_p_f_catch_nextreset_hists, PmV_p_f_catch_nextreset_hists, M1_p_f_catch_cue_hists, S1_p_f_catch_cue_hists, PmD_p_f_catch_cue_hists, PmV_p_f_catch_cue_hists]

    #Generates final data dictionary for each region based on pnt_data; _d_ is following reward delivery and _c_ is following cue
    M1_d_data,targets=Make_final_data(pnt_data_2,0)
    S1_d_data,targets=Make_final_data(pnt_data_2,1)
    pmd_d_data,targets=Make_final_data(pnt_data_2,2)
    pmv_d_data,targets=Make_final_data(pnt_data_2,3)

    M1_c_data,targets=Make_final_data(pnt_data_2,4)
    S1_c_data,targets=Make_final_data(pnt_data_2,5)
    pmd_c_data,targets=Make_final_data(pnt_data_2,6)
    pmv_c_data,targets=Make_final_data(pnt_data_2,7) 

    #Construct dictionary for saving array
    final_data = {'M1_d_data':M1_d_data,'S1_d_data':S1_d_data,'pmd_d_data':pmd_d_data,'pmv_d_data':pmv_d_data,'M1_c_data':M1_c_data,'S1_c_data':S1_c_data,'pmd_c_data':pmd_c_data,'pmv_c_data':pmv_c_data,'targets':targets}

    #Construct temparary dictionary for figure generation
    final_data_no_targets = {'M1_at_delivery':M1_d_data,'S1_at_delivery':S1_d_data,'PmD_at_delivery':pmd_d_data,'PmV_at_delivery':pmv_d_data,'M1_at_cue':M1_c_data,'S1_at_cue':S1_c_data,'PmD_at_cue':pmd_c_data,'PmV_at_cue':pmv_c_data}

    np.save("single_rp_"+filename[-15:-4]+"_hists_"+name_of_bin,(final_data,unit_names))

    #accuracy_dict={}
    #all_accuracy_total = {}
    accuracy_dict = {}
    accuracy_total = {}
	#Perform PCA on PSTH followed By LDA on PCA transform of PSTH data and save figure showing results for each bin
    for key,value in final_data.iteritems():     #_no_targets.iteritems():
        #print lda
        print key
        if key == 'targets':
            continue
        len_targets=len(targets)
        #randomize targets (uncomment to randomize)
        #targets=np.random.randint(0,7,len_targets)
        
        lda = LDA(n_components=2)
        pca = RandomizedPCA(n_components=pca_num)
        proj = pca.fit_transform(value)
        proj = lda.fit_transform(proj,targets)
        print proj.shape

        plt.clf()

        if plot_bool:
            fig=plt.figure()
            fig, ax = plt.subplots(1,1,figsize=(8,8))
            plt.title(key+" from "+name_of_bin)
            plt.xlabel("LD1")
            plt.ylabel("LD2")
        
            cmap=plt.cm.nipy_spectral
            cmaplist=[cmap(i) for i in range(cmap.N)]
            cmap=cmap.from_list('Custom cmap',cmaplist,cmap.N)

            if rcatch_bool or pcatch_bool:
                bounds = np.linspace(0,10,11)
            else:
                bounds=np.linspace(0,8,9)

            ticks=bounds + 0.5
            norm=colors.BoundaryNorm(bounds, cmap.N)

            scatter=ax.scatter(proj[:,0],proj[:,1],c=targets,cmap=cmap,norm=norm)
            ax2=fig.add_axes([0.7,0.1,0.03,0.7])       
            cb=colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,spacing='proportional',ticks=bounds,boundaries=bounds,format='%1i')
            cb.set_ticks(ticks)
            ticklabels=np.arange(0,8,1)
            cb.set_ticklabels(ticklabels)
            ax2.set_title=('category')
            plt.figtext(0.8,0.4,'0: rp_s\n1: rp_f\n2: nrnp_s\n3: nrnp_f\n4: r_only_s\n5: r_only_f\n6: p_only_s\n7: p_only_f', bbox=dict(facecolor='white'))
            fig.subplots_adjust(right=0.7)
            
            plt.savefig(key+"_from_"+name_of_bin+"s.png")
            #plt.savefig(key+"_from_"+name_of_bin+"s_random.png")
            
            plt.clf()

        ################################

        #make rewarding targs and punishing targs (ie condensed)
        #rewarding = 0, 4
        #punishing = 1, 7
        #nrnp_s and _f = 2, 3
        #get opposite = 5, 6
        #lump 2, 3, 5, 6 as get nothing  [also, try w/ lumping nrnp_s w/ rewarding, nrnp_f as punishing, etc]

        r_p_null = False
        succ_unsucc = False
        succr_succnr_other = False
        succr_succnr_unsuccp_unsuccnp = False
        lda_pre_all = False
        lda_pre_succr_succnr_other = False
        
        #accuracy_total = {}
        #TODO must be better way to do this
        for i in range(7):
            #accuracy_total = {}
            if i == 0:
                r_p_null = True
            elif i == 1:
                r_p_null = False
                succ_unsucc = True
            elif i == 2:
                succ_unsucc = False
                succr_succnr_other = True
            elif i == 3:
                succr_succnr_other = False
                succr_succnr_unsuccp_unsuccnp = True
            elif i == 4:
                succr_succnr_unsuccp_unsuccnp = False
                lda_pre_all = True
            elif i == 5:
                lda_pre_all = False
                lda_pre_succr_succnr_other = True
            elif i == 6:
                lda_pre_succr_succnr_other = False
                
            value_to_classify = np.copy(value)
            #TESTING target type
            #make dict with each type, iterate through
            if r_p_null:
                print 'classifying rewarding, punishing, null'
                condensed_targ = np.copy(targets)
                condensed_targ[condensed_targ == 4] = 0  #setting two rewarding types to 0
                condensed_targ[condensed_targ == 7] = 1  #setting two punishing types to 1
                condensed_targ[condensed_targ == 3] = 2  #set all rest to 2
                condensed_targ[condensed_targ == 5] = 2
                condensed_targ[condensed_targ == 6] = 2  
                targs_to_classify = condensed_targ
            elif succ_unsucc:
                print 'classifying successful and unsuccessful'
                condensed_targ = np.copy(targets)
                condensed_targ[condensed_targ == 0] = 0
                condensed_targ[condensed_targ == 1] = 1
                condensed_targ[condensed_targ == 2] = 0  #setting successful to 0
                condensed_targ[condensed_targ == 3] = 1  #setting unsuccessful to 1
                condensed_targ[condensed_targ == 4] = 0
                condensed_targ[condensed_targ == 5] = 1
                condensed_targ[condensed_targ == 6] = 0 
                condensed_targ[condensed_targ == 7] = 1
                targs_to_classify = condensed_targ
            elif succr_succnr_other:
                print 'classifying successful_rewarding, successful_nonrewarding, all else'
                condensed_targ = np.copy(targets)
                condensed_targ[condensed_targ == 0] = 0
                condensed_targ[condensed_targ == 1] = 2
                condensed_targ[condensed_targ == 2] = 1  #setting successful and rewarding to 0
                condensed_targ[condensed_targ == 3] = 2  #setting successful and nonrewarding to 1
                condensed_targ[condensed_targ == 4] = 0  #setting all other to 2
                condensed_targ[condensed_targ == 5] = 2
                condensed_targ[condensed_targ == 6] = 1
                condensed_targ[condensed_targ == 7] = 2
                targs_to_classify = condensed_targ
            elif succr_succnr_unsuccp_unsuccnp:
                print 'classifying successful_rewarding, successful_nonrewarding, unsuccessful_punishing, unsuccessful_nonpunishing'
                condensed_targ = np.copy(targets)
                condensed_targ[condensed_targ == 0] = 0
                condensed_targ[condensed_targ == 1] = 2
                condensed_targ[condensed_targ == 2] = 1  #setting successful, rewarding to 0
                condensed_targ[condensed_targ == 3] = 3  #setting successful, nonrewarding = 1
                condensed_targ[condensed_targ == 4] = 0  #unsuccessful punishing = 2
                condensed_targ[condensed_targ == 5] = 3  #unsuccessful nonpunishing = 3
                condensed_targ[condensed_targ == 6] = 1
                condensed_targ[condensed_targ == 7] = 2
                targs_to_classify = condensed_targ
            elif lda_pre_all:
                print 'classifying all targets, with pca/lda first'
                value_to_classify = proj
                targs_to_classify = np.copy(targets)
            elif lda_pre_succr_succnr_other:
                print 'classifying pca/lda first then successful_rewarding, successful_nonrewarding, and other'
                value_to_classify = proj
                condensed_targ = np.copy(targets)
                condensed_targ[condensed_targ == 0] = 0
                condensed_targ[condensed_targ == 1] = 2
                condensed_targ[condensed_targ == 2] = 1  #setting successful and rewarding to 0
                condensed_targ[condensed_targ == 3] = 2  #setting successful and nonrewarding to 1
                condensed_targ[condensed_targ == 4] = 0  #setting all other to 2
                condensed_targ[condensed_targ == 5] = 2
                condensed_targ[condensed_targ == 6] = 1
                condensed_targ[condensed_targ == 7] = 2
                targs_to_classify = condensed_targ
            else:
                print 'classifying all targets'
                targs_to_classify = np.copy(targets)
            
            #split into training and testing samples. test_size = proportion of data used for test
            x_train, x_test, y_train, y_test = train_test_split(value_to_classify, targs_to_classify, test_size = .4) 

            #########################
            #ADABoost Classifier
            #########################
            bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1)

            bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1.5,algorithm="SAMME")
            bdt_real.fit(x_train, y_train)
            bdt_discrete.fit(x_train, y_train)
            
            real_test_errors = []
            discrete_test_errors = []

            for real_test_predict, discrete_train_predict in zip(bdt_real.staged_predict(x_test), bdt_discrete.staged_predict(x_test)):
                real_test_errors.append(1. - accuracy_score(real_test_predict, y_test))
                discrete_test_errors.append(1. - accuracy_score(discrete_train_predict, y_test))

            n_trees_discrete = len(bdt_discrete)
            n_trees_real = len(bdt_real)

            # Boosting might terminate early, but the following arrays are always
            # n_estimators long. We crop them to the actual number of trees here:
            discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
            real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
            discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]
            
            # Test on the testing data set and display the accuracies
            ypred_r = bdt_real.predict(x_test)
            ypred_e = bdt_discrete.predict(x_test)
            accuracy_sammer = accuracy_score(ypred_r,y_test)
            accuracy_samme = accuracy_score(ypred_e,y_test)
        
            print 'Accuracy of SAMME.R = ', accuracy_sammer
            print 'Accuracy of SAMME = ', accuracy_samme

            #try:
            #    plt.figure(figsize=(15, 5))

            #    plt.subplot(131)
            #    plt.plot(range(1, n_trees_discrete + 1),discrete_test_errors, c='black', label='SAMME')
			#   plt.plot(range(1, n_trees_real + 1),real_test_errors, c='black',linestyle='dashed', label='SAMME.R')
            #    plt.legend()
            #    plt.ylim(0.18, 0.62)
            #    plt.ylabel('Test Error')
            #    plt.xlabel('Number of Trees')
                
            #    plt.subplot(132)
            #    plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,"b", label='SAMME', alpha=.5)
            #    plt.plot(range(1, n_trees_real + 1), real_estimator_errors,"r", label='SAMME.R', alpha=.5)
            #    plt.legend()
            #    plt.ylabel('Error')
            #    plt.xlabel('Number of Trees')
            #    plt.ylim((.2,max(real_estimator_errors.max(),discrete_estimator_errors.max()) * 1.2))
            #    plt.xlim((-20, len(bdt_discrete) + 20))
            
            #    plt.subplot(133)
            #    plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,"b", label='SAMME')
            #    plt.legend()
            #    plt.ylabel('Weight')
            #    plt.xlabel('Number of Trees')
            #    plt.ylim((0, discrete_estimator_weights.max() * 1.2))
            #    plt.xlim((-20, n_trees_discrete + 20))

            #    # prevent overlapping y-axis labels
            #    plt.subplots_adjust(wspace=0.25)
            #    #plt.show()
            #    plt.savefig(key+"_from_"+name_of_bin+"_"+i+"_SAMME.png")
            #    plt.clf()
            #except:
            #    print 'error plotting'

            #accuracy_dict[key]={'accuracy_sammer':accuracy_sammer,'accuracy_samme':accuracy_samme}
        #accuracy_total[i] = accuracy_dict
        #or is it:
            accuracy_dict[i] ={'accuracy_sammer':accuracy_sammer,'accuracy_samme':accuracy_samme}
            #print 'accuracy dict is:'
            #print i
        accuracy_total[key] = accuracy_dict
        #print 'accuracy total is:'
        #print key
		
		#print accuracy_total.keys()
		###########################
    all_accuracy_total[name_of_bin] = accuracy_total
    print all_accuracy_total.keys()

np.save('accuracy_report_all',all_accuracy_total)


plt.clf()
plt.close()

















