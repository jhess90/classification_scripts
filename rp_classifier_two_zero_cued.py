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


#-----------Pull neural data in Extracted_ Form----------------

#root = tk.Tk()
#root.withdraw()
#filename = tkFileDialog.askopenfilename();

filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20150928_504/Extracted_504_2015-09-28-12-07-26.mat'
print filename
a = sio.loadmat(filename);

#Pull reward timestamps. Separate timespamp file must exist in folder containing neural data, with same name+_timestamps
print filename[:-4]+"_timestamps"+filename[-4:]
Timestamps = sio.loadmat(filename[:-4]+"_timestamps"+filename[-4:]);

#Timestamp generation: generates arrays of timestamps depending on reward level, including offset for cue presentation and reward delivery
#one reward cue, completed unsuccessfully
r_f_nextreset = Timestamps['r_f_nextreset'][0];
r_f_cue = Timestamps['r_f_cue_ts'][0];

#one reward, completed successfully (rewarding)
r_s_rdelivery = Timestamps['r_s_rdelivery_ts'][0];
r_s_cue = Timestamps['r_s_cue_ts'][0];

#one punishment, completed unsuccessfully (punishing)
p_f_pdelivery = Timestamps['p_f_pdelivery_ts'][0];
p_f_cue = Timestamps['p_f_cue_ts'][0];

#one punishment, completed successcfully (non-punishing)
p_s_nextreset= Timestamps['p_s_nextreset'][0];
p_s_cue = Timestamps['p_s_cue_ts'][0];

#zero reward cue, successful (non-rewarding)
nr_s_nextreset = Timestamps['nr_s_nextreset'][0];
nr_s_cue = Timestamps['nr_s_cue_ts'][0];

#zero reward cue, unsuccessful (non-rewarding)
nr_f_nextreset = Timestamps['nr_f_nextreset'][0];
nr_f_cue = Timestamps['nr_f_cue_ts'][0];

#zero punishment cue, successful (non-punishing)
np_s_nextreset = Timestamps['np_s_nextreset'][0];
np_s_cue = Timestamps['np_s_cue_ts'][0];

#zero punishment cue, unsuccessful (non-punishing)
np_f_nextreset = Timestamps['np_f_nextreset'][0];
np_f_cue = Timestamps['np_f_cue_ts'][0];

#one reward cue, one punishment cue, successful (rewarding)
rp_s_rdelivery = Timestamps['rp_s_rdelivery_ts'][0];
rp_s_cue = Timestamps['rp_s_cue_ts'][0];
print ('rp_s trials: %s'%(len(rp_s_rdelivery)))

#one reward cue, one punishment cue, unsuccessful (punishing)
rp_f_pdelivery = Timestamps['rp_f_pdelivery_ts'][0];
rp_f_cue = Timestamps['rp_f_cue_ts'][0];
print ('rp_f trials: %s'%(len(rp_f_pdelivery)))

#no reward cue, no punishment cue, successful (non-rewarding)
nrnp_s_nextreset = Timestamps['nrnp_s_nextreset'][0];
nrnp_s_cue = Timestamps['nrnp_s_cue_ts'][0];
print ('nrnp_s trials: %s'%(len(nrnp_s_cue)))

#no reward cue, no punishment cue, unsuccessful (non-punishing)
nrnp_f_nextreset = Timestamps['nrnp_f_nextreset'][0];
nrnp_f_cue = Timestamps['nrnp_f_cue_ts'][0];
print ('nrnp_f trials: %s'%(len(nrnp_f_cue)))

#reward cue only (no punishment cue), successful (rewarding)
r_only_s_rdelivery = Timestamps['r_only_s_rdelivery_ts'][0];
r_only_s_cue = Timestamps['r_only_s_cue_ts'][0];
print ('r_only_s trials: %s'%(len(r_only_s_cue)))

#reward cue only (no punishment cue), unsuccessful (non-rewarding)
r_only_f_nextreset = Timestamps['r_only_f_nextreset'][0];
r_only_f_cue = Timestamps['r_only_f_cue_ts'][0];
print ('r_only_f trials: %s'%(len(r_only_f_cue)))

#punishment cue only (no reward cue), successful (non-punishing)
p_only_s_nextreset = Timestamps['p_only_s_nextreset'][0];
p_only_s_cue = Timestamps['p_only_s_cue_ts'][0];
print ('p_only_s trials: %s'%(len(p_only_s_cue)))

#punishment cue only (no reward cue), unsuccessful (punishing)
p_only_f_pdelivery = Timestamps['p_only_f_pdelivery_ts'][0];
p_only_f_cue = Timestamps['p_only_f_cue_ts'][0];
print ('p_only_f trials: %s'%(len(p_only_f_cue)))

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
no_bins = 10;

print ('10 bins (50 ms each)\n100 pca components\n2 LDA components')

#trim M1,S1,and PmD spikes to remove PmV spikes trailing at end; nerual data pruning now complete
M1_spikes = M1_spikes[0,0:M1_limit];
S1_spikes = S1_spikes[0,0:S1_limit];
PmD_spikes = PmD_spikes[0,0:PmD_limit];
#generate time intervals of interest for PSTH generation
#time_boundry={'0-2.5':[0.0,2.5],'2.5-5.0':[2.5,5.0],'0-1.5':[0.0,1.5],'1.5-3.5':[1.5,3.5],}
time_boundry={'0-0.5':[0.0,0.5], '0.5-1.0':[0.5,1.0], '1.0-1.5':[1.0,1.5], '1.5-2.0':[1.5,2.0]}

#Begin main loop of analysis
for name_of_bin,time_of_bin in time_boundry.iteritems():
    before_time = -time_of_bin[0]
    after_time = time_of_bin[1]
    print( 'bin %s' %(name_of_bin))
    #Build and save M1 PSTH

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
        #for i in range(0,data[start_position+32].shape[0]):
        #    dummy.append(np.hstack(data[start_position+32][i]))
        #r_only_s_data = np.array(dummy)
        #r_only_s_targets = np.ones(r_only_s_data.shape[0])*4
        #for i in range(0,data[start_position+40].shape[0]):
        #    dummy.append(np.hstack(data[start_position+40][i]))
        #r_only_f_data = np.array(dummy)
        #r_only_f_targets = np.ones(r_only_f_data.shape[0])*5
        #for i in range(0,data[start_position+48].shape[0]):
        #    dummy.append(np.hstack(data[start_position+48][i]))
        #p_only_s_data = np.array(dummy)
        #p_only_s_targets = np.ones(p_only_s_data.shape[0])*6
        #for i in range(0,data[start_position+56].shape[0]):
        #    dummy.append(np.hstack(data[start_position+56][i]))
        #p_only_f_data = np.array(dummy)
        #p_only_f_targets = np.ones(p_only_f_data.shape[0])*7
        
#        return (np.vstack([rp_s_data,rp_f_data,nrnp_s_data,nrnp_f_data,r_only_s_data,r_only_f_data,p_only_s_data,p_only_f_data]),np.hstack([rp_s_targets,rp_f_targets,nrnp_s_targets,nrnp_f_targets,r_only_s_targets,r_only_f_targets,p_only_s_targets,p_only_f_targets]))
        return(np.vstack([rp_s_data,rp_f_data,nrnp_s_data,nrnp_f_data]),np.hstack([rp_s_targets,rp_f_targets,nrnp_s_targets,nrnp_f_targets]))

    #One reward, failed (not rewarding)
    #M1_r_f_nextreset_hists = Build_hist(M1_spikes,r_f_nextreset,no_bins,before_time,after_time)
    #S1_r_f_nextreset_hists = Build_hist(S1_spikes,r_f_nextreset,no_bins,before_time,after_time)
    #PmD_r_f_nextreset_hists = Build_hist(PmD_spikes,r_f_nextreset,no_bins,before_time,after_time)
    #PmV_r_f_nextreset_hists = Build_hist_PmV(PmV_spikes,r_f_nextreset,no_bins,before_time,after_time)
    #M1_r_f_cue_hists = Build_hist(M1_spikes,r_f_cue,no_bins,before_time,after_time)
    #S1_r_f_cue_hists = Build_hist(S1_spikes,r_f_cue,no_bins,before_time,after_time)
    #PmD_r_f_cue_hists = Build_hist(PmD_spikes,r_f_cue,no_bins,before_time,after_time)
    #PmV_r_f_cue_hists = Build_hist_PmV(PmV_spikes,r_f_cue,no_bins,before_time,after_time)

    #One reward, successful (rewarding)
    #M1_r_s_rdelivery_hists = Build_hist(M1_spikes,r_s_rdelivery,no_bins,before_time,after_time)
    #S1_r_s_rdelivery_hists = Build_hist(S1_spikes,r_s_rdelivery,no_bins,before_time,after_time)
    #PmD_r_s_rdelivery_hists = Build_hist(PmD_spikes,r_s_rdelivery,no_bins,before_time,after_time)
    #PmV_r_s_rdelivery_hists = Build_hist_PmV(PmV_spikes,r_s_rdelivery,no_bins,before_time,after_time)
    #M1_r_s_cue_hists = Build_hist(M1_spikes,r_s_cue,no_bins,before_time,after_time)
    #S1_r_s_cue_hists = Build_hist(S1_spikes,r_s_cue,no_bins,before_time,after_time)
    #PmD_r_s_cue_hists = Build_hist(PmD_spikes,r_s_cue,no_bins,before_time,after_time)
    #PmV_r_s_cue_hists = Build_hist_PmV(PmV_spikes,r_s_cue,no_bins,before_time,after_time)
 
    #One punishment, unsuccessful (punishing)
    #M1_p_f_pdelivery_hists = Build_hist(M1_spikes,p_f_pdelivery,no_bins,before_time,after_time)
    #S1_p_f_pdelivery_hists = Build_hist(S1_spikes,p_f_pdelivery,no_bins,before_time,after_time)
    #PmD_p_f_pdelivery_hists = Build_hist(PmD_spikes,p_f_pdelivery,no_bins,before_time,after_time)
    #PmV_p_f_pdelivery_hists = Build_hist_PmV(PmV_spikes,p_f_pdelivery,no_bins,before_time,after_time)
    #M1_p_f_cue_hists = Build_hist(M1_spikes,p_f_cue,no_bins,before_time,after_time)
    #S1_p_f_cue_hists = Build_hist(S1_spikes,p_f_cue,no_bins,before_time,after_time)
    #PmD_p_f_cue_hists = Build_hist(PmD_spikes,p_f_cue,no_bins,before_time,after_time)
    #PmV_p_f_cue_hists = Build_hist_PmV(PmV_spikes,p_f_cue,no_bins,before_time,after_time)

    #One punishment, successful (non-punishing)
    #M1_p_s_nextreset_hists = Build_hist(M1_spikes,p_s_nextreset,no_bins,before_time,after_time)
    #S1_p_s_nextreset_hists = Build_hist(S1_spikes,p_s_nextreset,no_bins,before_time,after_time)
    #PmD_p_s_nextreset_hists = Build_hist(PmD_spikes,p_s_nextreset,no_bins,before_time,after_time)
    #PmV_p_s_nextreset_hists = Build_hist_PmV(PmV_spikes,p_s_nextreset,no_bins,before_time,after_time)
    #M1_p_s_cue_hists = Build_hist(M1_spikes,p_s_cue,no_bins,before_time,after_time)
    #S1_p_s_cue_hists = Build_hist(S1_spikes,p_s_cue,no_bins,before_time,after_time)
    #PmD_p_s_cue_hists = Build_hist(PmD_spikes,p_s_cue,no_bins,before_time,after_time)
    #PmV_p_s_cue_hists = Build_hist_PmV(PmV_spikes,p_s_cue,no_bins,before_time,after_time)

    #zero reward, successful (non-rewarding)
    #M1_nr_s_nextreset_hists = Build_hist(M1_spikes,nr_s_nextreset,no_bins,before_time,after_time)
    #S1_nr_s_nextreset_hists = Build_hist(S1_spikes,nr_s_nextreset,no_bins,before_time,after_time)
    #PmD_nr_s_nextreset_hists = Build_hist(PmD_spikes,nr_s_nextreset,no_bins,before_time,after_time)
    #PmV_nr_s_nextreset_hists = Build_hist_PmV(PmV_spikes,nr_s_nextreset,no_bins,before_time,after_time)

    #M1_nr_s_cue_hists = Build_hist(M1_spikes,nr_s_cue,no_bins,before_time,after_time)
    #S1_nr_s_cue_hists = Build_hist(S1_spikes,nr_s_cue,no_bins,before_time,after_time)
    #PmD_nr_s_cue_hists = Build_hist(PmD_spikes,nr_s_cue,no_bins,before_time,after_time)
    #PmV_nr_s_cue_hists = Build_hist_PmV(PmV_spikes,nr_s_cue,no_bins,before_time,after_time)

    #zero reward, failed (non-rewarding)
    #M1_nr_f_nextreset_hists = Build_hist(M1_spikes,nr_f_nextreset,no_bins,before_time,after_time)
    #S1_nr_f_nextreset_hists = Build_hist(S1_spikes,nr_f_nextreset,no_bins,before_time,after_time)
    #PmD_nr_f_nextreset_hists = Build_hist(PmD_spikes,nr_f_nextreset,no_bins,before_time,after_time)
    #PmV_nr_f_nextreset_hists = Build_hist_PmV(PmV_spikes,nr_f_nextreset,no_bins,before_time,after_time)

    #M1_nr_f_cue_hists = Build_hist(M1_spikes,nr_f_cue,no_bins,before_time,after_time)
    #S1_nr_f_cue_hists = Build_hist(S1_spikes,nr_f_cue,no_bins,before_time,after_time)
    #PmD_nr_f_cue_hists = Build_hist(PmD_spikes,nr_f_cue,no_bins,before_time,after_time)
    #PmV_nr_f_cue_hists = Build_hist_PmV(PmV_spikes,nr_f_cue,no_bins,before_time,after_time)

    #zero punishment cue, successful (non-punishing)
    #M1_np_s_nextreset_hists = Build_hist(M1_spikes,p_s_cue,no_bins,before_time,after_time)
    #S1_np_s_nextreset_hists = Build_hist(S1_spikes,p_s_nextreset,no_bins,before_time,after_time)
    #PmD_np_s_nextreset_hists = Build_hist(PmD_spikes,p_s_cue,no_bins,before_time,after_time)
    #PmV_np_s_nextreset_hists = Build_hist_PmV(PmV_spikes,p_s_cue,no_bins,before_time,after_time)

    #M1_np_s_cue_hists = Build_hist(M1_spikes,p_s_cue,no_bins,before_time,after_time)
    #S1_np_s_cue_hists = Build_hist(S1_spikes,p_s_nextreset,no_bins,before_time,after_time)
    #PmD_np_s_cue_hists = Build_hist(PmD_spikes,p_s_cue,no_bins,before_time,after_time)
    #PmV_np_s_cue_hists = Build_hist_PmV(PmV_spikes,p_s_cue,no_bins,before_time,after_time)

    #zero punishment cue, failed (non-punishing)
    #M1_np_f_nextreset_hists = Build_hist(M1_spikes,p_s_cue,no_bins,before_time,after_time)
    #S1_np_f_nextreset_hists = Build_hist(S1_spikes,p_s_nextreset,no_bins,before_time,after_time)
    #PmD_np_f_nextreset_hists = Build_hist(PmD_spikes,p_s_cue,no_bins,before_time,after_time)
    #PmV_np_f_nextreset_hists = Build_hist_PmV(PmV_spikes,p_s_cue,no_bins,before_time,after_time)

    #M1_np_f_cue_hists = Build_hist(M1_spikes,p_s_cue,no_bins,before_time,after_time)
    #S1_np_f_cue_hists = Build_hist(S1_spikes,p_s_nextreset,no_bins,before_time,after_time)
    #PmD_np_f_cue_hists = Build_hist(PmD_spikes,p_s_cue,no_bins,before_time,after_time)
    #PmV_np_f_cue_hists = Build_hist_PmV(PmV_spikes,p_s_cue,no_bins,before_time,after_time)

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
    #M1_p_only_s_nextreset_hists = Build_hist(M1_spikes,p_only_s_nextreset,no_bins,before_time,after_time)
    #S1_p_only_s_nextreset_hists = Build_hist(S1_spikes,p_only_s_nextreset,no_bins,before_time,after_time)
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
    #PmV_p_only_f_cue_hists = Build_hist_PmV(PmV_spikes,p_only_f_cue,no_bins,before_time,after_time)

    
    #This list is a consortium of all of the reward histograms, both cue and reward for each region at each level of reward
    #pnt_data = [M1_zero_reward_hists,S1_zero_reward_hists,PmD_zero_reward_hists,PmV_zero_reward_hists,M1_zero_reward_cue_hists,S1_zero_reward_cue_hists,PmD_zero_reward_cue_hists,PmV_zero_reward_cue_hists,M1_one_reward_hists,S1_one_reward_hists,PmD_one_reward_hists,PmV_one_reward_hists,M1_one_reward_cue_hists,S1_one_reward_cue_hists,PmD_one_reward_cue_hists,PmV_one_reward_cue_hists,M1_two_reward_hists,S1_two_reward_hists,PmD_two_reward_hists,PmV_two_reward_hists,M1_two_reward_cue_hists,S1_two_reward_cue_hists,PmD_two_reward_cue_hists,PmV_two_reward_cue_hists,M1_three_reward_hists,S1_three_reward_hists,PmD_three_reward_hists,PmV_three_reward_hists,M1_three_reward_cue_hists,S1_three_reward_cue_hists,PmD_three_reward_cue_hists,PmV_three_reward_cue_hists]

    #pnt_data = [M1_r_f_nextreset_hists,S1_r_f_nextreset_hists,PmD_r_f_nextreset_hists,PmV_r_f_nextreset_hists,M1_r_f_cue_hists,S1_r_f_cue_hists,PmD_r_f_cue_hists,PmV_r_f_cue_hists,M1_r_s_rdelivery_hists,S1_r_s_rdelivery_hists,PmD_r_s_rdelivery_hists,PmV_r_s_rdelivery_hists,M1_r_s_cue_hists,S1_r_s_cue_hists,PmD_r_s_cue_hists,PmV_r_s_cue_hists,M1_p_f_pdelivery_hists,S1_p_f_pdelivery_hists,PmD_p_f_pdelivery_hists,PmV_p_f_pdelivery_hists,M1_p_f_cue_hists,S1_p_f_cue_hists,PmD_p_f_cue_hists,PmV_p_f_cue_hists,M1_p_s_nextreset_hists,S1_p_s_nextreset_hists,PmD_p_s_nextreset_hists,PmV_p_s_nextreset_hists,M1_p_s_cue_hists,S1_p_s_cue_hists,PmD_p_s_cue_hists,PmV_p_s_cue_hists]

    #no overlapping data points
    pnt_data_2 = [M1_rp_s_rdelivery_hists, S1_rp_s_rdelivery_hists, PmD_rp_s_rdelivery_hists, PmV_rp_s_rdelivery_hists, M1_rp_s_cue_hists, S1_rp_s_cue_hists, PmD_rp_s_cue_hists, PmV_rp_s_cue_hists, M1_rp_f_pdelivery_hists, S1_rp_f_pdelivery_hists, PmD_rp_f_pdelivery_hists, PmV_rp_f_pdelivery_hists, M1_rp_f_cue_hists, S1_rp_f_cue_hists, PmD_rp_f_cue_hists, PmV_rp_f_cue_hists, M1_nrnp_s_nextreset_hists, S1_nrnp_s_nextreset_hists, PmD_nrnp_s_nextreset_hists, PmV_nrnp_s_nextreset_hists, M1_nrnp_s_cue_hists, S1_nrnp_s_cue_hists, PmD_nrnp_s_cue_hists, PmV_nrnp_s_cue_hists, M1_nrnp_f_nextreset_hists, S1_nrnp_f_nextreset_hists, PmD_nrnp_f_nextreset_hists, PmV_nrnp_f_nextreset_hists, M1_nrnp_f_cue_hists, S1_nrnp_f_cue_hists, PmD_nrnp_f_cue_hists, PmV_nrnp_f_cue_hists]

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

    #Perform PCA on PSTH followed By LDA on PCA transform of PSTH data and save figure showing results for each bin
    for key,value in final_data.iteritems():     #_no_targets.iteritems():
        #print lda
        print key
        if key == 'targets':
            continue
        lda = LDA(n_components=2)
        pca = RandomizedPCA(n_components=100)
        proj = pca.fit_transform(value)
        proj = lda.fit_transform(proj,targets)
        print proj.shape

        plt.clf()
        
        fig=plt.figure()
        fig, ax = plt.subplots(1,1,figsize=(8,8))
        plt.title(key+" from "+name_of_bin)
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        
        cmap=plt.cm.jet
        cmaplist=[cmap(i) for i in range(cmap.N)]
        cmap=cmap.from_list('Custom cmap',cmaplist,cmap.N)
        bounds=np.linspace(0,7,8)
        norm=colors.BoundaryNorm(bounds, cmap.N)

        scatter=ax.scatter(proj[:,0],proj[:,1],c=targets,cmap=cmap,norm=norm)
        ax2=fig.add_axes([0.7,0.1,0.03,0.7])       
        cb=colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,spacing='proportional',ticks=bounds,boundaries=bounds,format='%1i')
        ax2.set_title=('category')
        plt.figtext(0.8,0.4,'0: rp_s\n1: rp_f\n2: nrnp_s\n3: nrnp_f\n4: r_only_s\n5: r_only_f\n6: p_only_s\n7: p_only_f', bbox=dict(facecolor='white'))
        fig.subplots_adjust(right=0.7)
        
        plt.savefig(key+"_from_"+name_of_bin+"s.png")

        plt.clf()
     



                                                                                
