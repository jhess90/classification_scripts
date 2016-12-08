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

no_bins = 10
plot_bool = True
time_boundry={'-0.5-1.0':[-0.5,1.0]}


bin_size = 50 #in ms

#### laptop ##############

#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-12-48-52.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-13-02-45.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-12-48-19.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-13-10-44.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-14-23.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-33-52.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-25-20.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-46-25.mat'

### beaver ##############
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-12-48-52.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-13-02-45.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-12-48-19.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-13-10-44.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-14-23.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-33-52.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-25-20.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-46-25.mat'

#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_0059/Extracted_0059_2016-05-25-15-41-44.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_0059/Extracted_0059_2016-05-25-15-58-17.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_504/Extracted_504_2016-05-25-14-46-46.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_504/Extracted_504_2016-05-25-15-02-58.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_0059/Extracted_0059_2016-05-26-12-17-53.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_0059/Extracted_0059_2016-05-26-12-38-56.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_504/Extracted_504_2016-05-26-11-25-03.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_504/Extracted_504_2016-05-26-11-45-52.mat'


### beaver ##############
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-12-48-52.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-13-02-45.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-12-48-19.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-13-10-44.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-14-23.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-33-52.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-25-20.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-46-25.mat'

#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_0059/Extracted_0059_2016-05-25-15-41-44.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_0059/Extracted_0059_2016-05-25-15-58-17.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_504/Extracted_504_2016-05-25-14-46-46.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_504/Extracted_504_2016-05-25-15-02-58.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_0059/Extracted_0059_2016-05-26-12-17-53.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_0059/Extracted_0059_2016-05-26-12-38-56.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_504/Extracted_504_2016-05-26-11-25-03.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_504/Extracted_504_2016-05-26-11-45-52.mat'

#filename = '/home/jack/Dropbox/mult_rp_files/workspace/20160226_0059/block3/Extracted_0059_2016-02-26-16-28-27.mat'
#filename = '/home/jack/Dropbox/mult_rp_files/workspace/20160226_0059/block4/Extracted_0059_2016-02-26-16-38-13.mat'
#filename = '/home/jack/Dropbox/mult_rp_files/workspace/20160111_504/block1/Extracted_504_2016-01-11-13-56-44.mat'
#filename = '/home/jack/Dropbox/mult_rp_files/workspace/20160111_504/block2/Extracted_504_2016-01-11-14-10-01.mat'

filename = '/Users/johnhessburg/dropbox/mult_rp_files/workspace/20160226_0059/block3/Extracted_0059_2016-02-26-16-28-27.mat'


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
    elif (int(M1_spikes[0,i]['signame'][0][0][0][3:-1]) == 96):
        M1_limit = i;
        print ("The number of units in M1 is: %s"%(M1_limit))
    M1_unit_names.append(M1_spikes[0,i]['signame'][0][0][0])
dummy = [];
#M1_limit not defined for 0526_0059, blocks 1 and 2
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
    elif (int(PmD_spikes[0,i]['signame'][0][0][0][3:-1]) == 96):
        PmD_limit = i;
        print ("The number of units in PmD is: %s"%(PmD_limit))
    pmd_unit_names.append(PmD_spikes[0,i]['signame'][0][0][0])
for i in range(PmD_limit,PmD_spikes.shape[1]):
    dummy.append(PmD_spikes[0,i]['ts'][0,0][0])
#Generate Pmv Spiking data
PmV_spikes = np.array(dummy);
print("The number of units in PmV is: %s"%(len(PmV_spikes)))
unit_names['pmd_unit_names']=pmd_unit_names

#print ('%s bins (50 ms each)\n%s pca components\n2 LDA components' %(no_bins,pca_num))

#trim M1,S1,and PmD spikes to remove PmV spikes trailing at end; nerual data pruning now complete
M1_spikes = M1_spikes[0,0:M1_limit];
S1_spikes = S1_spikes[0,0:S1_limit];
PmD_spikes = PmD_spikes[0,0:PmD_limit];
#generate time intervals of interest for PSTH generation

#time_boundry={'-0.5-0.0':[-0.5,0.0],'0-0.5':[0.0,0.5], '0.5-1.0':[0.5,1.0], '1.0-1.5':[1.0,1.5], '1.5-2.0':[1.5,2.0],'all':[-0.5,2.0]}
#time_boundry={'-0.5-0.0':[-0.5,0.0],'0-0.5':[0.0,0.5],'0.5-1.0':[0.5,1.0],'all_incpre':[-0.5,2.0]}
#time_boundry={'-0.5-0.0':[-0.5,0.0],'0-0.5':[0.0,0.5],'0.5-1.0':[0.5,1.0],'all':[0.0,2.0]}


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

    ##one reward cue, one punishment cue, successful (rewarding)
    #M1_rp_s_rdelivery_hists = Build_hist(M1_spikes,rp_s_rdelivery,no_bins,before_time,after_time)
    #S1_rp_s_rdelivery_hists = Build_hist(S1_spikes,rp_s_rdelivery,no_bins,before_time,after_time)
    #PmD_rp_s_rdelivery_hists = Build_hist(PmD_spikes,rp_s_rdelivery,no_bins,before_time,after_time)
    #PmV_rp_s_rdelivery_hists = Build_hist_PmV(PmV_spikes,rp_s_rdelivery,no_bins,before_time,after_time)

    #M1_rp_s_cue_hists = Build_hist(M1_spikes,rp_s_cue,no_bins,before_time,after_time)
    #S1_rp_s_cue_hists = Build_hist(S1_spikes,rp_s_cue,no_bins,before_time,after_time)
    #PmD_rp_s_cue_hists = Build_hist(PmD_spikes,rp_s_cue,no_bins,before_time,after_time)
    #PmV_rp_s_cue_hists = Build_hist_PmV(PmV_spikes,rp_s_cue,no_bins,before_time,after_time)

    #one reward cue, one punishment cue, unsuccessful (punishing)
    #M1_rp_f_pdelivery_hists = Build_hist(M1_spikes,rp_f_pdelivery,no_bins,before_time,after_time)
    #S1_rp_f_pdelivery_hists = Build_hist(S1_spikes,rp_f_pdelivery,no_bins,before_time,after_time)
    #PmD_rp_f_pdelivery_hists = Build_hist(PmD_spikes,rp_f_pdelivery,no_bins,before_time,after_time)
    #PmV_rp_f_pdelivery_hists = Build_hist_PmV(PmV_spikes,rp_f_pdelivery,no_bins,before_time,after_time)

    #M1_rp_f_cue_hists = Build_hist(M1_spikes,rp_f_cue,no_bins,before_time,after_time)
    #S1_rp_f_cue_hists = Build_hist(S1_spikes,rp_f_cue,no_bins,before_time,after_time)
    #PmD_rp_f_cue_hists = Build_hist(PmD_spikes,rp_f_cue,no_bins,before_time,after_time)
    #PmV_rp_f_cue_hists = Build_hist_PmV(PmV_spikes,rp_f_cue,no_bins,before_time,after_time)

    #no reward cue, no punishment cue, successful (non-rewarding)
    #M1_nrnp_s_nextreset_hists = Build_hist(M1_spikes,nrnp_s_nextreset,no_bins,before_time,after_time)
    #S1_nrnp_s_nextreset_hists = Build_hist(S1_spikes,nrnp_s_nextreset,no_bins,before_time,after_time)
    #PmD_nrnp_s_nextreset_hists = Build_hist(PmD_spikes,nrnp_s_nextreset,no_bins,before_time,after_time)
    #PmV_nrnp_s_nextreset_hists = Build_hist_PmV(PmV_spikes,nrnp_s_nextreset,no_bins,before_time,after_time)

    #M1_nrnp_s_cue_hists = Build_hist(M1_spikes,nrnp_s_cue,no_bins,before_time,after_time)
    #S1_nrnp_s_cue_hists = Build_hist(S1_spikes,nrnp_s_cue,no_bins,before_time,after_time)
    #PmD_nrnp_s_cue_hists = Build_hist(PmD_spikes,nrnp_s_cue,no_bins,before_time,after_time)
    #PmV_nrnp_s_cue_hists = Build_hist_PmV(PmV_spikes,nrnp_s_cue,no_bins,before_time,after_time)

    #no reward cue, no punishment cue, fail (non-punishing)
    #M1_nrnp_f_nextreset_hists = Build_hist(M1_spikes,nrnp_f_nextreset,no_bins,before_time,after_time)
    #S1_nrnp_f_nextreset_hists = Build_hist(S1_spikes,nrnp_f_nextreset,no_bins,before_time,after_time)
    #PmD_nrnp_f_nextreset_hists = Build_hist(PmD_spikes,nrnp_f_nextreset,no_bins,before_time,after_time)
    #PmV_nrnp_f_nextreset_hists = Build_hist_PmV(PmV_spikes,nrnp_f_nextreset,no_bins,before_time,after_time)

    #M1_nrnp_f_cue_hists = Build_hist(M1_spikes,nrnp_f_cue,no_bins,before_time,after_time)
    #S1_nrnp_f_cue_hists = Build_hist(S1_spikes,nrnp_f_cue,no_bins,before_time,after_time)
    #PmD_nrnp_f_cue_hists = Build_hist(PmD_spikes,nrnp_f_cue,no_bins,before_time,after_time)
    #PmV_nrnp_f_cue_hists = Build_hist_PmV(PmV_spikes,nrnp_f_cue,no_bins,before_time,after_time)

    #reward cue only, successful (rewarding)
    #M1_r_only_s_rdelivery_hists = Build_hist(M1_spikes,r_only_s_rdelivery,no_bins,before_time,after_time)
    #S1_r_only_s_rdelivery_hists = Build_hist(S1_spikes,r_only_s_rdelivery,no_bins,before_time,after_time)
    #PmD_r_only_s_rdelivery_hists = Build_hist(PmD_spikes,r_only_s_rdelivery,no_bins,before_time,after_time)
    #PmV_r_only_s_rdelivery_hists = Build_hist_PmV(PmV_spikes,r_only_s_rdelivery,no_bins,before_time,after_time)

    #M1_r_only_s_cue_hists = Build_hist(M1_spikes,r_only_s_cue,no_bins,before_time,after_time)
    #S1_r_only_s_cue_hists = Build_hist(S1_spikes,r_only_s_cue,no_bins,before_time,after_time)
    #PmD_r_only_s_cue_hists = Build_hist(PmD_spikes,r_only_s_cue,no_bins,before_time,after_time)
    #PmV_r_only_s_cue_hists = Build_hist_PmV(PmV_spikes,r_only_s_cue,no_bins,before_time,after_time)

    #reward cue only, unsuccesssul (non-rewarding)
    #M1_r_only_f_nextreset_hists = Build_hist(M1_spikes,r_only_f_nextreset,no_bins,before_time,after_time)
    #S1_r_only_f_nextreset_hists = Build_hist(S1_spikes,r_only_f_nextreset,no_bins,before_time,after_time)
    #PmD_r_only_f_nextreset_hists = Build_hist(PmD_spikes,r_only_f_nextreset,no_bins,before_time,after_time)
    #PmV_r_only_f_nextreset_hists = Build_hist_PmV(PmV_spikes,r_only_f_nextreset,no_bins,before_time,after_time)

    #M1_r_only_f_cue_hists = Build_hist(M1_spikes,r_only_f_cue,no_bins,before_time,after_time)
    #S1_r_only_f_cue_hists = Build_hist(S1_spikes,r_only_f_cue,no_bins,before_time,after_time)
    #PmD_r_only_f_cue_hists = Build_hist(PmD_spikes,r_only_f_cue,no_bins,before_time,after_time)
    #PmV_r_only_f_cue_hists = Build_hist_PmV(PmV_spikes,r_only_f_cue,no_bins,before_time,after_time)

    #punishment cue only, successful (non-punishing)
    #M1_p_only_s_nextreset_hists = Build_hist(M1_spikes,p_only_s_nextreset,no_bins,before_time,after_tim#e)
    #S1_p_only_s_nextreset_hists = Build_hist(S1_spikes,p_only_s_nextreset,no_bins,before_time,after_tim#e)
    #PmD_p_only_s_nextreset_hists = Build_hist(PmD_spikes,p_only_s_nextreset,no_bins,before_time,after_time)
    #PmV_p_only_s_nextreset_hists = Build_hist_PmV(PmV_spikes,p_only_s_nextreset,no_bins,before_time,after_time)

    #M1_p_only_s_cue_hists = Build_hist(M1_spikes,p_only_s_cue,no_bins,before_time,after_time)
    #S1_p_only_s_cue_hists = Build_hist(S1_spikes,p_only_s_cue,no_bins,before_time,after_time)
    #PmD_p_only_s_cue_hists = Build_hist(PmD_spikes,p_only_s_cue,no_bins,before_time,after_time)
    #PmV_p_only_s_cue_hists = Build_hist_PmV(PmV_spikes,p_only_s_cue,no_bins,before_time,after_time)

    #punishment cue only, unsuccessful (punishing)
    #M1_p_only_f_pdelivery_hists = Build_hist(M1_spikes,p_only_f_pdelivery,no_bins,before_time,after_time)
    #S1_p_only_f_pdelivery_hists = Build_hist(S1_spikes,p_only_f_pdelivery,no_bins,before_time,after_time)
    #PmD_p_only_f_pdelivery_hists = Build_hist(PmD_spikes,p_only_f_pdelivery,no_bins,before_time,after_time)
    #PmV_p_only_f_pdelivery_hists = Build_hist_PmV(PmV_spikes,p_only_f_pdelivery,no_bins,before_time,after_time)

    #M1_p_only_f_cue_hists = Build_hist(M1_spikes,p_only_f_cue,no_bins,before_time,after_time)
    #S1_p_only_f_cue_hists = Build_hist(S1_spikes,p_only_f_cue,no_bins,before_time,after_time)
    #PmD_p_only_f_cue_hists = Build_hist(PmD_spikes,p_only_f_cue,no_bins,before_time,after_time)
    #PmV_p_only_f_cue_hists = Build_hist_PmV(PmV_spikes,p_only_f_cue,no_bins,before_time,after_time)#reward cue, successful, NO reward delivery (catch trial)

    #if rcatch_bool or pcatch_bool:
        #M1_r_s_catch_nextreset_hists = Build_hist(M1_spikes,r_s_catch_nextreset,no_bins,before_time,after_time)
        #S1_r_s_catch_nextreset_hists = Build_hist(M1_spikes,r_s_catch_nextreset,no_bins,before_time,after_time)
        #PmD_r_s_catch_nextreset_hists = Build_hist(M1_spikes,r_s_catch_nextreset,no_bins,before_time,after_time)
        #PmV_r_s_catch_nextreset_hists = Build_hist(M1_spikes,r_s_catch_nextreset,no_bins,before_time,after_time)

        #M1_r_s_catch_cue_hists = Build_hist(M1_spikes,r_s_catch_cue,no_bins,before_time,after_time)
        #S1_r_s_catch_cue_hists = Build_hist(M1_spikes,r_s_catch_cue,no_bins,before_time,after_time)
        #PmD_r_s_catch_cue_hists = Build_hist(M1_spikes,r_s_catch_cue,no_bins,before_time,after_time)#
        #PmV_r_s_catch_cue_hists = Build_hist(M1_spikes,r_s_catch_cue,no_bins,before_time,after_time)

        #punishment cue, unsuccessful, NO punishment delivery (catch trial)
        #if pcatch_bool:
        #M1_p_f_catch_nextreset_hists = Build_hist(M1_spikes,p_f_catch_nextreset,no_bins,before_time,after_time)
        #S1_p_f_catch_nextreset_hists = Build_hist(M1_spikes,p_f_catch_nextreset,no_bins,before_time,after_time)
        #PmD_p_f_catch_nextreset_hists = Build_hist(M1_spikes,p_f_catch_nextreset,no_bins,before_time,after_time)
        #PmV_p_f_catch_nextreset_hists = Build_hist(M1_spikes,p_f_catch_nextreset,no_bins,before_time,after_time)

        #M1_p_f_catch_cue_hists = Build_hist(M1_spikes,p_f_catch_cue,no_bins,before_time,after_time)
        #S1_p_f_catch_cue_hists = Build_hist(M1_spikes,p_f_catch_cue,no_bins,before_time,after_time)
        #PmD_p_f_catch_cue_hists = Build_hist(M1_spikes,p_f_catch_cue,no_bins,before_time,after_time)#
        #PmV_p_f_catch_cue_hists = Build_hist(M1_spikes,p_f_catch_cue,no_bins,before_time,after_time)
    
    #This list is a consortium of all of the reward histograms, both cue and reward for each region at each level of reward
    #no overlapping data points
    #pnt_data_2 = [M1_rp_s_rdelivery_hists, S1_rp_s_rdelivery_hists, PmD_rp_s_rdelivery_hists, PmV_rp_s_rdelivery_hists, M1_rp_s_cue_hists, S1_rp_s_cue_hists, PmD_rp_s_cue_hists, PmV_rp_s_cue_hists, M1_rp_f_pdelivery_hists, S1_rp_f_pdelivery_hists, PmD_rp_f_pdelivery_hists, PmV_rp_f_pdelivery_hists, M1_rp_f_cue_hists, S1_rp_f_cue_hists, PmD_rp_f_cue_hists, PmV_rp_f_cue_hists, M1_nrnp_s_nextreset_hists, S1_nrnp_s_nextreset_hists, PmD_nrnp_s_nextreset_hists, PmV_nrnp_s_nextreset_hists, M1_nrnp_s_cue_hists, S1_nrnp_s_cue_hists, PmD_nrnp_s_cue_hists, PmV_nrnp_s_cue_hists, M1_nrnp_f_nextreset_hists, S1_nrnp_f_nextreset_hists, PmD_nrnp_f_nextreset_hists, PmV_nrnp_f_nextreset_hists, M1_nrnp_f_cue_hists, S1_nrnp_f_cue_hists, PmD_nrnp_f_cue_hists, PmV_nrnp_f_cue_hists, M1_r_only_s_rdelivery_hists, S1_r_only_s_rdelivery_hists, PmD_r_only_s_rdelivery_hists, PmV_r_only_s_rdelivery_hists, M1_r_only_s_cue_hists, S1_r_only_s_cue_hists, PmD_r_only_s_cue_hists, PmV_r_only_s_cue_hists, M1_r_only_f_nextreset_hists, S1_r_only_f_nextreset_hists, PmD_r_only_f_nextreset_hists, PmV_r_only_f_nextreset_hists, M1_r_only_f_cue_hists, S1_r_only_f_cue_hists, PmD_r_only_f_cue_hists, PmV_r_only_f_cue_hists, M1_p_only_s_nextreset_hists, S1_p_only_s_nextreset_hists, PmD_p_only_s_nextreset_hists, PmV_p_only_s_nextreset_hists, M1_p_only_s_cue_hists, S1_p_only_s_cue_hists, PmD_p_only_s_cue_hists, PmV_p_only_s_cue_hists, M1_p_only_f_pdelivery_hists, S1_p_only_f_pdelivery_hists, PmD_p_only_f_pdelivery_hists, PmV_p_only_f_pdelivery_hists, M1_p_only_f_cue_hists, S1_p_only_f_cue_hists, PmD_p_only_f_cue_hists, PmV_p_only_f_cue_hists]
    #if pcatch_bool or rcatch_bool:
        #pnt_data_2 = [M1_rp_s_rdelivery_hists, S1_rp_s_rdelivery_hists, PmD_rp_s_rdelivery_hists, PmV_rp_s_rdelivery_hists, M1_rp_s_cue_hists, S1_rp_s_cue_hists, PmD_rp_s_cue_hists, PmV_rp_s_cue_hists, M1_rp_f_pdelivery_hists, S1_rp_f_pdelivery_hists, PmD_rp_f_pdelivery_hists, PmV_rp_f_pdelivery_hists, M1_rp_f_cue_hists, S1_rp_f_cue_hists, PmD_rp_f_cue_hists, PmV_rp_f_cue_hists, M1_nrnp_s_nextreset_hists, S1_nrnp_s_nextreset_hists, PmD_nrnp_s_nextreset_hists, PmV_nrnp_s_nextreset_hists, M1_nrnp_s_cue_hists, S1_nrnp_s_cue_hists, PmD_nrnp_s_cue_hists, PmV_nrnp_s_cue_hists, M1_nrnp_f_nextreset_hists, S1_nrnp_f_nextreset_hists, PmD_nrnp_f_nextreset_hists, PmV_nrnp_f_nextreset_hists, M1_nrnp_f_cue_hists, S1_nrnp_f_cue_hists, PmD_nrnp_f_cue_hists, PmV_nrnp_f_cue_hists, M1_r_only_s_rdelivery_hists, S1_r_only_s_rdelivery_hists, PmD_r_only_s_rdelivery_hists, PmV_r_only_s_rdelivery_hists, M1_r_only_s_cue_hists, S1_r_only_s_cue_hists, PmD_r_only_s_cue_hists, PmV_r_only_s_cue_hists, M1_r_only_f_nextreset_hists, S1_r_only_f_nextreset_hists, PmD_r_only_f_nextreset_hists, PmV_r_only_f_nextreset_hists, M1_r_only_f_cue_hists, S1_r_only_f_cue_hists, PmD_r_only_f_cue_hists, PmV_r_only_f_cue_hists, M1_p_only_s_nextreset_hists, S1_p_only_s_nextreset_hists, PmD_p_only_s_nextreset_hists, PmV_p_only_s_nextreset_hists, M1_p_only_s_cue_hists, S1_p_only_s_cue_hists, PmD_p_only_s_cue_hists, PmV_p_only_s_cue_hists, M1_p_only_f_pdelivery_hists, S1_p_only_f_pdelivery_hists, PmD_p_only_f_pdelivery_hists, PmV_p_only_f_pdelivery_hists, M1_p_only_f_cue_hists, S1_p_only_f_cue_hists, PmD_p_only_f_cue_hists, PmV_p_only_f_cue_hists, M1_r_s_catch_nextreset_hists, S1_r_s_catch_nextreset_hists, PmD_r_s_catch_nextreset_hists, PmV_r_s_catch_nextreset_hists, M1_r_s_catch_cue_hists, S1_r_s_catch_cue_hists, PmD_r_s_catch_cue_hists, PmV_r_s_catch_cue_hists, M1_p_f_catch_nextreset_hists, S1_p_f_catch_nextreset_hists, PmD_p_f_catch_nextreset_hists, PmV_p_f_catch_nextreset_hists, M1_p_f_catch_cue_hists, S1_p_f_catch_cue_hists, PmD_p_f_catch_cue_hists, PmV_p_f_catch_cue_hists]

    #Generates final data dictionary for each region based on pnt_data; _d_ is following reward delivery and _c_ is following cue
    #M1_d_data,targets=Make_final_data(pnt_data_2,0)
    #S1_d_data,targets=Make_final_data(pnt_data_2,1)
    #pmd_d_data,targets=Make_final_data(pnt_data_2,2)
    #pmv_d_data,targets=Make_final_data(pnt_data_2,3)

    #M1_c_data,targets=Make_final_data(pnt_data_2,4)
    #S1_c_data,targets=Make_final_data(pnt_data_2,5)
    #pmd_c_data,targets=Make_final_data(pnt_data_2,6)
    #pmv_c_data,targets=Make_final_data(pnt_data_2,7) 

    #Construct dictionary for saving array
    #final_data = {'M1_d_data':M1_d_data,'S1_d_data':S1_d_data,'pmd_d_data':pmd_d_data,'pmv_d_data':pmv_d_data,'M1_c_data':M1_c_data,'S1_c_data':S1_c_data,'pmd_c_data':pmd_c_data,'pmv_c_data':pmv_c_data,'targets':targets}

    #Construct temparary dictionary for figure generation
    #final_data_no_targets = {'M1_at_delivery':M1_d_data,'S1_at_delivery':S1_d_data,'PmD_at_delivery':pmd_d_data,'PmV_at_delivery':pmv_d_data,'M1_at_cue':M1_c_data,'S1_at_cue':S1_c_data,'PmD_at_cue':pmd_c_data,'PmV_at_cue':pmv_c_data}

    #np.save("single_rp_"+filename[-15:-4]+"_hists_"+name_of_bin,(final_data,unit_names))



#spikes=[]
#for i in range(len(M1_spikes)):
#	spikes.append(M1_spikes[i]['ts'][0,0][0])

#rp_s_rdelivery
time_before = -0.5 #make sure if change to pos time before it still works
time_after = 1.0

no_bins = (time_after - time_before) * 1000 / bin_size

def make_raster(data,spikes,key,key2,unit_no,time_before,time_after):

	dummy = []
	for i in range(0,data.shape[0]):
		dummy1=[]
		b = data[i]
		for j in range(len(spikes)):
			#a = spikes[j]['ts'][0,0][0]
			a = spikes[j]
			c = a[np.where(np.logical_and(a>=b+time_before,a<=b+time_after))]
			dummy1.append(c)
			#dummy1.append(np.nan_to_num(c[0]))
		dummy.append(dummy1)

	dummy = np.asarray(dummy)

	ax = plt.gca()

	adjusted_times = []
	binned = []
	#firing_rate = []
	
	for i in range(dummy.shape[0]):
		adjusted_times=dummy[i][unit_no]-value[i]
		plt.vlines(adjusted_times, i+.5, i+1.5)
		plt.ylim(.5, dummy.shape[0] + .5)

		#TODO normalize here

		binned_temp = np.histogram(adjusted_times,bins=no_bins,normed=False,density=False)
		binned.append(np.nan_to_num(binned_temp[0]))

	binned = np.asarray(binned)
	firing_rate = np.zeros(len(binned),dtype=object)
	for i in range(len(binned)):
		firing_rate[i] = binned[i] / float(bin_size) * 1000 #because bin_size in ms, want to get Hz

	avg_firing_rate = np.mean(firing_rate)
	std_dev_firing_rate = np.std(firing_rate)
	
	#fig,axs = plt.subplots(nrows=1, ncols=2, sharex=True)

	plt.subplot(2,1,1)
	
	#ax=axs[0,0]
	
	plt.axvline(0.0)
	
	#plt.title('Ex raster plot,%s, %s, unit: %s' %(key,key2,str(unit_no)))
	#plt.set_title('Raster plot')
	#ax.xlabel('Time')
	#ax.ylabel('Trial')
	#plt.show()

	#ax=axs[0,1]
	plt.subplot(2,2,1)
	plt.errorbar(avg_firing_rate,std_dev_firing_rate) #,yerr=yerr,fmt=o)
	#plt.set_title('average firing rate')

	#fig.suptitle('%s, %s, unit: %s' %(key,key2,str(unit_no)))
	plt.title('%s, %s, unit: %s' %(key,key2,str(unit_no)))
	
	plt.savefig('raster_%s_%s_unit%s' %(key, key2, str(unit_no)))
	plt.clf()
	
	return(firing_rate)

#unit_no for loop
#unit_no = 20

#ts_dict={'rp_s_cue':rp_s_cue, 'rp_s_rdelivery':rp_s_rdelivery, 'rp_f_cue':rp_f_cue, 'rp_f_pdelivery':rp_f_pdelivery, 'nrnp_s_cue':nrnp_s_cue, 'nrnp_s_nextreset':nrnp_s_nextreset, 'nrnp_f_cue':nrnp_f_cue, 'nrnp_f_nextreset':nrnp_f_nextreset, 'r_only_s_cue':r_only_s_cue, 'r_only_s_rdelivery':r_only_s_rdelivery, 'r_only_f_cue':r_only_f_cue, 'r_only_f_nextreset':r_only_f_nextreset, 'p_only_s_cue':p_only_s_cue, 'p_only_s_nextreset':p_only_s_nextreset, 'p_only_f_cue':p_only_f_cue, 'p_only_f_pdelivery':p_only_f_pdelivery}

#testing dict
#ts_dict={'rp_s_cue':rp_s_cue, 'r_only_s':r_only_s_rdelivery,'nrnp_s_cue':nrnp_s_cue}
#ts_dict={'r_only_f_nextreset':r_only_f_nextreset}
ts_dict = {'rp_s_rdelivery':rp_s_rdelivery}

data_dict={'M1_spikes':M1_spikes,'S1_spikes':S1_spikes,'PmD_spikes':PmD_spikes,'PmV_spikes':PmV_spikes}

for key, value in ts_dict.iteritems():
	if not (len(value) == 1 and value[0] == 0):
		for key2, value2 in data_dict.iteritems():
			print 'making rasters %s, %s' %(key, key2)
			spikes=[]

			if key2 == 'PmV_spikes':
				for i in range(len(value2)):
					spikes.append(value2[i])
			else:
				for i in range(len(value2)):
					spikes.append(value2[i]['ts'][0,0][0])			
				for i in range(len(spikes)):
					rasters = make_raster(value,spikes,key,key2,i,time_before,time_after)
	else:
		print 'no instances of event %s, no plot generation' %(key)
