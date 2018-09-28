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
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

########################
# params to set ########
########################

plot_bool = False
plot_hist_bool = True

bin_size = 10 #in ms
time_before = -0.5 #negative value
time_after = 1.0
res_window = [0.3,0.8]

#########################
# functions  ############
#########################
def make_hist_all(spike_data):
        hist = []
        hist_data = []
        hist_bins = []
        
        #find max value (amax not working with data shape for some reason)
        max_spike_ts = 0
        for i in range(len(spike_data)):
                if np.max(spike_data[i]) > max_spike_ts:
                        max_spike_ts = np.max(spike_data[i])

        max_bin_num = int(np.ceil(max_spike_ts) / float(bin_size) * 1000)
        hist_data = np.zeros((len(spike_data),max_bin_num))
        hist_bins = np.zeros((len(spike_data),max_bin_num))
        for i in range(len(spike_data)):
                total_bin_range = np.arange(0,int(np.ceil(spike_data[i].max())),bin_size/1000.0)
                hist,bins = np.histogram(spike_data[i],bins=total_bin_range,range=(0,int(np.ceil(spike_data[i].max()))),normed=False,density=False)
                hist_data[i,0:len(hist)] = hist
                hist_bins[i,0:len(bins)] = bins
        
        return_dict = {'hist_data':hist_data,'hist_bins':hist_bins}
        return(return_dict)

def calc_firing_rates(hists,data_key,condensed):
        bin_size_sec = bin_size / 1000.0
        hists = stats.zscore(hists,axis=1)

        bfr_cue_fr = np.zeros((len(condensed),np.shape(hists)[0],-1*bins_before))
        aft_cue_fr = np.zeros((len(condensed),np.shape(hists)[0],bins_after))
        bfr_result_fr = np.zeros((len(condensed),np.shape(hists)[0],-1*bins_before))
        aft_result_fr = np.zeros((len(condensed),np.shape(hists)[0],bins_after))
        res_wind_fr = np.zeros((len(condensed),np.shape(hists)[0],res_wind_bins[1] - res_wind_bins[0]))
        concat_fr = np.zeros((len(condensed),np.shape(hists)[0],2*bins_after + -1*bins_before))

        bfr_cue_hist_all = np.zeros((len(condensed),np.shape(hists)[0],-1*bins_before))
        aft_cue_hist_all = np.zeros((len(condensed),np.shape(hists)[0],bins_after))
        bfr_result_hist_all = np.zeros((len(condensed),np.shape(hists)[0],-1*bins_before))
        aft_result_hist_all = np.zeros((len(condensed),np.shape(hists)[0],bins_after))
        res_wind_hist_all = np.zeros((len(condensed),np.shape(hists)[0],res_wind_bins[1] - res_wind_bins[0]))

        #col 0 = disp_rp, 1 = succ scene, 2 = failure scene, 3 = rnum, 4 = pnum, 5 = succ/fail
        for i in range(np.shape(condensed)[0]):
                closest_cue_start_time = np.around((round(condensed[i,0] / bin_size_sec) * bin_size_sec),decimals=2)
                cue_start_bin = int(closest_cue_start_time / bin_size_sec)
                if condensed[i,5] == 1:
                        closest_result_start_time = np.around(round((condensed[i,1] / bin_size_sec) * bin_size_sec),decimals=2)
                        result_start_bin = int(closest_result_start_time / bin_size_sec)
                else:
                        closest_result_start_time = np.around((round(condensed[i,2] / bin_size_sec) * bin_size_sec),decimals=2)
                        result_start_bin = int(closest_result_start_time / bin_size_sec)
                
                if not (result_start_bin + bins_after) > np.shape(hists)[1]:
                        for j in range(np.shape(hists)[0]):
                                bfr_cue_fr[i,j,:] = hists[j,bins_before + cue_start_bin : cue_start_bin]
                                aft_cue_fr[i,j,:] = hists[j,cue_start_bin : cue_start_bin + bins_after]
                                bfr_result_fr[i,j,:] = hists[j,bins_before + result_start_bin : result_start_bin]
                                aft_result_fr[i,j,:] = hists[j,result_start_bin : result_start_bin + bins_after]
                                res_wind_fr[i,j,:] = hists[j,result_start_bin + res_wind_bins[0] : result_start_bin + res_wind_bins[1]]

                                #aft cue + bfr res + aft res
                                concat_fr[i,j,:] = np.concatenate((hists[j,cue_start_bin : cue_start_bin + bins_after],hists[j,bins_before + result_start_bin : result_start_bin],hists[j,result_start_bin : result_start_bin + bins_after]))

                else:
                        continue

        return_dict = {'bfr_cue_fr':bfr_cue_fr,'aft_cue_fr':aft_cue_fr,'bfr_result_fr':bfr_result_fr,'aft_result_fr':aft_result_fr,'res_wind_fr':res_wind_fr,'concat_fr':concat_fr}


        return(return_dict)


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


################
#ada ###########
################
def run_adaboost(value,targets):
    #####can try and predict for each unit? Or pool entire pseudopopulation and then predict? run both?
    bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1)

    bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1.5,algorithm="SAMME")

    x_train, x_test, y_train, y_test = train_test_split(value,targets,test_size =.2)
    bdt_real.fit(x_train, y_train)
    bdt_discrete.fit(x_train, y_train)
    real_test_errors = []
    discrete_test_errors = []
    for real_test_predict, discrete_train_predict in zip(bdt_real.staged_predict(x_test), bdt_discrete.staged_predict(x_test)):
        real_test_errors.append(1. - accuracy_score(real_test_predict, y_test))
        discrete_test_errors.append(1. - accuracy_score(discrete_train_predict, y_test))

    n_trees_discrete = len(bdt_discrete)
    n_trees_real = len(bdt_real)
    discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
    real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
    discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]
    ypred_r = bdt_real.predict(x_test)
    ypred_e = bdt_discrete.predict(x_test)
    accuracy_sammer = accuracy_score(ypred_r,y_test)
    accuracy_samme = accuracy_score(ypred_e,y_test)

    ##
    chance = float(1) / np.shape(np.unique(targets))[0]

    return_dict = {'ypred_r':ypred_r,'ypred_e':ypred_e,'accuracy_sammer':accuracy_sammer,'accuracy_samme':accuracy_samme,'chance':chance}

    #run again with shuffled targets
    np.random.shuffle(targets)
    x_train, x_test, y_train, y_test = train_test_split(value,targets,test_size =.2)
    bdt_real.fit(x_train, y_train)
    bdt_discrete.fit(x_train, y_train)
    real_test_errors = []
    discrete_test_errors = []
    for real_test_predict, discrete_train_predict in zip(bdt_real.staged_predict(x_test), bdt_discrete.staged_predict(x_test)):
        real_test_errors.append(1. - accuracy_score(real_test_predict, y_test))
        discrete_test_errors.append(1. - accuracy_score(discrete_train_predict, y_test))

    n_trees_discrete = len(bdt_discrete)
    n_trees_real = len(bdt_real)
    discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
    real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
    discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]
    shuffle_ypred_r = bdt_real.predict(x_test)
    shuffle_ypred_e = bdt_discrete.predict(x_test)
    shuffle_accuracy_sammer = accuracy_score(ypred_r,y_test)
    shuffle_accuracy_samme = accuracy_score(ypred_e,y_test)

    return_dict['shuffle_accuracy_sammer'] = shuffle_accuracy_sammer
    return_dict['shuffle_accuracy_samme'] = shuffle_accuracy_samme

    return(return_dict)










#########################
# run ###################
#########################

bins_before = int(time_before / float(bin_size) * 1000)  #neg for now
bins_after = int(time_after / float(bin_size) * 1000)   
res_wind_bins = [int(res_window[0] / float(bin_size) * 1000),int(res_window[1] / float(bin_size)*1000)]

print 'bin size: %s' %(bin_size)
print 'time before: %s, time after: %s' %(time_before,time_after)

#load and then concat all rp/alt/uncued

file_ct = 0
file_dict = {}
for file in glob.glob('*timestamps.mat'):

    timestamps = sio.loadmat(file)
    a = sio.loadmat(file[:-15])
    
    neural_data=a['neural_data']
    Spikes = a['neural_data']['spikeTimes'];
    trial_breakdown = timestamps['trial_breakdown']

    condensed = np.zeros((np.shape(trial_breakdown)[0],6))

    #col 0 = disp_rp, 1 = succ scene, 2 = failure scene, 3 = rnum, 4 = pnum, 5 = catch_num
    condensed[:,0] = trial_breakdown[:,1]
    condensed[:,1] = trial_breakdown[:,2]
    condensed[:,2] = trial_breakdown[:,3]
    condensed[:,3] = trial_breakdown[:,5]
    condensed[:,4] = trial_breakdown[:,7]
    condensed[:,5] = trial_breakdown[:,10]

    #delete end trials if not fully finished
    if condensed[-1,1] == condensed[-1,2] == 0:
	new_condensed = condensed[0:-1,:]
        condensed = new_condensed
    condensed = condensed[condensed[:,0] != 0]

    #remove trials with no succ or failure scene (not sure why, but saw in one)
    condensed = condensed[np.invert(np.logical_and(condensed[:,1] == 0, condensed[:,2] == 0))]

    #TODOD FOR NOW remove catch trials
    condensed = condensed[condensed[:,5] == 0]
    #col 5 all 0s now, replace with succ/fail vector: succ = 1, fail = -1
    condensed[condensed[:,1] != 0, 5] = 1
    condensed[condensed[:,2] != 0, 5] = -1

    #succ only
    #condensed = condensed[condensed[:,5] == 1]

    #fail only
    #condensed = condensed[condensed[:,5] == -1]

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

    #take care of some misc params, save param dict
    no_bins = int((time_after - time_before) * 1000 / bin_size)
    print 'bin size = %s' %(bin_size)
    param_dict = {'time_before':time_before,'time_after':time_after,'bin_size':bin_size,'no_bins':no_bins}
    min_hist_len = 1000000 #set arbitrarily high number, because want to determin min length later
    data_dict={'M1_spikes':M1_spikes,'S1_spikes':S1_spikes,'PmD_spikes':PmD_spikes,'PmV_spikes':PmV_spikes}

    data_dict_hist_all = {}
    print 'making hist'
    for key, value in data_dict.iteritems():
	spike_data = []
	if key == 'PmV_spikes':
                continue
	else:
            for i in range(len(value)):
                spike_data.append(value[i]['ts'][0,0][0])

        hist_dict = make_hist_all(spike_data)
        data_dict_hist_all['%s_hist_dict' %(key)] = hist_dict

    M1_dicts = {'hist_all':data_dict_hist_all['M1_spikes_hist_dict']['hist_data']} 
    S1_dicts = {'hist_all':data_dict_hist_all['S1_spikes_hist_dict']['hist_data']}
    PmD_dicts = {'hist_all':data_dict_hist_all['PmD_spikes_hist_dict']['hist_data']}

    data_dict_all = {'M1_dicts':M1_dicts,'S1_dicts':S1_dicts,'PmD_dicts':PmD_dicts}

    file_dict[file_ct] = {}

    for region_key,region_value in data_dict_all.iteritems():
        hists = data_dict_all[region_key]['hist_all']
        fr_dict = calc_firing_rates(hists,region_key,condensed)
        data_dict_all[region_key]['fr_dict'] = fr_dict
        
        file_dict[file_ct][region_key] = {'aft_cue':fr_dict['aft_cue_fr'],'bfr_res':fr_dict['bfr_result_fr'],'aft_res':fr_dict['aft_result_fr'],'res_win':fr_dict['res_wind_fr'],'condensed':condensed,'concat':fr_dict['concat_fr']}

    file_ct += 1
    

file_length = np.shape(file_dict.keys())[0]

for region_key,region_val in file_dict[0].iteritems():
    data_dict_all[region_key]['avg_and_corr'] = {}
    total_unit_num = 0
    total_trial_num = 0
    for num in range(file_length):
        total_unit_num += np.shape(file_dict[num][region_key]['aft_cue'])[1]
        total_trial_num += np.shape(file_dict[num][region_key]['condensed'])[0]

    for type_key,type_val in file_dict[0][region_key].iteritems():
            
        if type_key != 'condensed':
            sig_vals = np.zeros((total_unit_num,4))
            r_p_all = np.zeros((total_trial_num,3))

            all_list = list()
            avg_fr_dict = {}
            unit_ct = 0
            trial_ct = 0
            sig_vals_by_file = {}
            
            for file_ind in range(file_length):
                frs = file_dict[file_ind][region_key][type_key]
                condensed = file_dict[file_ind][region_key]['condensed']
                r_vals = condensed[:,3]
                p_vals = condensed[:,4]
                succ_vals = condensed[:,5]
                
                for unit_num in range(np.shape(frs)[1]):
                    all_list.append((frs[:,unit_num,:],np.array((r_vals,p_vals,succ_vals))))



                unit_ct += frs.shape[1]
                r_p_all[trial_ct : trial_ct + np.shape(r_vals)[0],:] = np.column_stack((r_vals,p_vals,succ_vals))
                trial_ct += np.shape(r_vals)[0]
                avg_fr_dict[file_ind] = np.mean(frs,axis=2)
                
            save_dict = {'avg_fr_dict':avg_fr_dict,'r_p_all':r_p_all,'total_unit_num':total_unit_num,'total_trial_num':total_trial_num,'all_list':all_list}
            data_dict_all[region_key]['avg_and_corr'][type_key] = save_dict



################



for region_key,region_val in data_dict_all.iteritems():
    print 'region: %s' %(region_key)
    
    data_dict_all[region_key]['classification_output'] = {}

    #file_dict = data_dict_all[region_key]['file_dict']
    file_length = np.shape(file_dict.keys())[0]
    avg_and_corr = data_dict_all[region_key]['avg_and_corr']

    #if not sig_only_bool:
    total_unit_num = avg_and_corr['aft_cue']['total_unit_num']
        
    output_by_window = {}
    for win_key,win_val in data_dict_all[region_key]['avg_and_corr'].iteritems():
        all_list =  win_val['all_list']

        #TODO figure out how many variations running, then chance shape of this.
        #general pattern: [accuracy_samme, accuracy_sammer, chance, shuffled, .....]  (for now shuffle = samme.r?)
        
        #shorter for devel
        accuracy_list_total = np.zeros((np.shape(all_list)[0],32))
        #accuracy_list_total = np.zeros((3,32))
        
        print 'window: %s' %(win_key)

        
        #shorter for devel
        for unit_num in range(np.shape(all_list)[0]):
        #for unit_num in range(3):
            
            #TODO set up diff variations of target here
            #succ/fail high even aft cue- b/ of more succ? unequal distrib? 
            #how many times run? What best test/train split?
            #use SAMME.R for shuffled for now

            r_level_out = run_adaboost(all_list[unit_num][0],all_list[unit_num][1][0,:])
            p_level_out = run_adaboost(all_list[unit_num][0],all_list[unit_num][1][1,:])
            succ_out = run_adaboost(all_list[unit_num][0],all_list[unit_num][1][2,:])

            #r nr, p np, m, v, r/nr comb w/ and w/o succ

            r_nr_targs = np.where(all_list[unit_num][1][0,:] > 0,1,0)
            p_np_targs = np.where(all_list[unit_num][1][1,:] > 0,1,0)
            
            r_bin_out = run_adaboost(all_list[unit_num][0],r_nr_targs)
            p_bin_out = run_adaboost(all_list[unit_num][0],p_np_targs)
            
            comb = np.zeros((np.shape(r_nr_targs)[0]))
            for i in range(np.shape(r_nr_targs)[0]):
                if r_nr_targs[i] == 0 and p_np_targs[i] == 0:
                    comb[i] = 0
                elif r_nr_targs[i] == 1 and p_np_targs[i] == 0:
                    comb[i] = 1
                elif r_nr_targs[i] == 0 and p_np_targs[i] == 1:
                    comb[i] = 2
                elif r_nr_targs[i] == 1 and p_np_targs[i] == 1:
                    comb[i] = 3
                
            comb_out = run_adaboost(all_list[unit_num][0],comb)

            r_bin_sf_targ = np.zeros((np.shape(r_nr_targs)[0]))
            for i in range(np.shape(r_nr_targs)[0]):
                if r_nr_targs[i] == 0 and all_list[unit_num][1][2,:][i] == 0:
                    r_bin_sf_targ[i] = 0
                elif r_nr_targs[i] == 1 and all_list[unit_num][1][2,:][i] == 0:
                    r_bin_sf_targ[i] = 1
                elif r_nr_targs[i] == 0 and all_list[unit_num][1][2,:][i] == 1:
                    r_bin_sf_targ[i] = 2
                elif r_nr_targs[i] == 1 and all_list[unit_num][1][2,:][i] == 1:
                    r_bin_sf_targ[i] = 3

            p_bin_sf_targ = np.zeros((np.shape(p_np_targs)[0]))
            for i in range(np.shape(p_np_targs)[0]):
                if p_np_targs[i] == 0 and all_list[unit_num][1][2,:][i] == 0:
                    p_bin_sf_targ[i] = 0
                elif p_np_targs[i] == 1 and all_list[unit_num][1][2,:][i] == 0:
                    p_bin_sf_targ[i] = 1
                elif p_np_targs[i] == 0 and all_list[unit_num][1][2,:][i] == 1:
                    p_bin_sf_targ[i] = 2
                elif p_np_targs[i] == 1 and all_list[unit_num][1][2,:][i] == 1:
                    p_bin_sf_targ[i] = 3

            r_bin_sf_out = run_adaboost(all_list[unit_num][0],r_bin_sf_targ)
            p_bin_sf_out = run_adaboost(all_list[unit_num][0],p_bin_sf_targ)

            accuracy_list_total[unit_num,:] = [r_level_out['accuracy_samme'],r_level_out['accuracy_sammer'],r_level_out['chance'],r_level_out['shuffle_accuracy_sammer'],p_level_out['accuracy_samme'],p_level_out['accuracy_sammer'],p_level_out['chance'],p_level_out['shuffle_accuracy_sammer'],succ_out['accuracy_samme'],succ_out['accuracy_sammer'],succ_out['chance'],succ_out['shuffle_accuracy_sammer'],comb_out['accuracy_samme'],comb_out['accuracy_sammer'],comb_out['chance'],comb_out['shuffle_accuracy_sammer'],r_bin_out['accuracy_samme'],r_bin_out['accuracy_sammer'],r_bin_out['chance'],r_bin_out['shuffle_accuracy_sammer'],p_bin_out['accuracy_samme'],p_bin_out['accuracy_sammer'],p_bin_out['chance'],p_bin_out['shuffle_accuracy_sammer'],r_bin_sf_out['accuracy_samme'],r_bin_sf_out['accuracy_sammer'],r_bin_sf_out['chance'],r_bin_sf_out['shuffle_accuracy_sammer'],p_bin_sf_out['accuracy_samme'],p_bin_sf_out['accuracy_sammer'],p_bin_sf_out['chance'],p_bin_sf_out['shuffle_accuracy_sammer']]

            sys.stdout.write('.')
            sys.stdout.flush()

        output_by_window[win_key] = accuracy_list_total
        print ''

        f,axarr = plt.subplots(8,sharex=True)

        f.suptitle('unit accuracy distribution: %s %s (SAMME.R)' %(region_key,win_key))
        axarr[0].hist(accuracy_list_total[:,1],color='mediumturquoise',lw=0)
        axarr[0].axvline(accuracy_list_total[:,2][0],color='k',linestyle='dashed',linewidth=1)
        axarr[0].set_xlim([0,1])
        axarr[0].set_title('reward level accuracy')
        axarr[1].hist(accuracy_list_total[:,3],color='darkviolet',lw=0)
        axarr[1].axvline(accuracy_list_total[:,2][0],color='k',linestyle='dashed',linewidth=1)
        axarr[1].set_title('reward level accuracy: shuffled')

        axarr[2].hist(accuracy_list_total[:,5],color='mediumturquoise',lw=0)
        axarr[2].axvline(accuracy_list_total[:,6][0],color='k',linestyle='dashed',linewidth=1)
        axarr[2].set_title('punishment level accuracy')
        axarr[3].hist(accuracy_list_total[:,7],color='darkviolet',lw=0)
        axarr[3].axvline(accuracy_list_total[:,6][0],color='k',linestyle='dashed',linewidth=1)
        axarr[3].set_title('punishment level accuracy: shuffled')

        axarr[4].hist(accuracy_list_total[:,9],color='mediumturquoise',lw=0)
        axarr[4].axvline(accuracy_list_total[:,10][0],color='k',linestyle='dashed',linewidth=1)
        axarr[4].set_title('succ/fail accuracy')
        axarr[5].hist(accuracy_list_total[:,11],color='darkviolet',lw=0)
        axarr[5].axvline(accuracy_list_total[:,10][0],color='k',linestyle='dashed',linewidth=1)
        axarr[5].set_title('succ/fail accuracy: shuffled')

        axarr[6].hist(accuracy_list_total[:,13],color='mediumturquoise',lw=0)
        axarr[6].axvline(accuracy_list_total[:,14][0],color='k',linestyle='dashed',linewidth=1)
        axarr[6].set_title('comb accuracy')
        axarr[7].hist(accuracy_list_total[:,15],color='darkviolet',lw=0)
        axarr[7].axvline(accuracy_list_total[:,14][0],color='k',linestyle='dashed',linewidth=1)
        axarr[7].set_title('comb accuracy: shuffled')

        for axi in axarr.reshape(-1):
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('unit_accuracy_hist_%s_%s' %(region_key,win_key))
        plt.clf()

        #
        f,axarr = plt.subplots(8,sharex=True)

        f.suptitle('unit accuracy distribution 2: %s %s (SAMME.R)' %(region_key,win_key))
        axarr[0].hist(accuracy_list_total[:,1+16],color='mediumturquoise',lw=0)
        axarr[0].axvline(accuracy_list_total[:,2+16][0],color='k',linestyle='dashed',linewidth=1)
        axarr[0].set_xlim([0,1])
        axarr[0].set_title('reward binary accuracy')
        axarr[1].hist(accuracy_list_total[:,3+16],color='darkviolet',lw=0)
        axarr[1].axvline(accuracy_list_total[:,2+16][0],color='k',linestyle='dashed',linewidth=1)
        axarr[1].set_title('reward binary accuracy: shuffled')

        axarr[2].hist(accuracy_list_total[:,5+16],color='mediumturquoise',lw=0)
        axarr[2].axvline(accuracy_list_total[:,6+16][0],color='k',linestyle='dashed',linewidth=1)
        axarr[2].set_title('punishment binary accuracy')
        axarr[3].hist(accuracy_list_total[:,7+16],color='darkviolet',lw=0)
        axarr[3].axvline(accuracy_list_total[:,6+16][0],color='k',linestyle='dashed',linewidth=1)
        axarr[3].set_title('punishment binary accuracy: shuffled')

        axarr[4].hist(accuracy_list_total[:,9+16],color='mediumturquoise',lw=0)
        axarr[4].axvline(accuracy_list_total[:,10+16][0],color='k',linestyle='dashed',linewidth=1)
        axarr[4].set_title('reward binary succ fail accuracy')
        axarr[5].hist(accuracy_list_total[:,11+16],color='darkviolet',lw=0)
        axarr[5].axvline(accuracy_list_total[:,10+16][0],color='k',linestyle='dashed',linewidth=1)
        axarr[5].set_title('reward binary succ fail accuracy: shuffled')

        axarr[6].hist(accuracy_list_total[:,13+16],color='mediumturquoise',lw=0)
        axarr[6].axvline(accuracy_list_total[:,14+16][0],color='k',linestyle='dashed',linewidth=1)
        axarr[6].set_title('punishment binary succ fail accuracy')
        axarr[7].hist(accuracy_list_total[:,15+16],color='darkviolet',lw=0)
        axarr[7].axvline(accuracy_list_total[:,14+16][0],color='k',linestyle='dashed',linewidth=1)
        axarr[7].set_title('punishment binary succ fail accuracy: shuffled')

        for axi in axarr.reshape(-1):
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('unit_accuracy_hist_2_%s_%s' %(region_key,win_key))
        plt.clf()
            
    data_dict_all[region_key]['classification_output'] = output_by_window


np.save('backup.npy',data_dict_all)



#################3


#if xlsx currently exists delte, or else won't write properly
if os.path.isfile('classifier_output.xlsx'):
    os.remove('classifier_output.xlsx')

type_names = ['aft cue','bfr res','aft res','res win','concat']
comb_names = ['r levels','p levels','succ/fail','comb','r bin','p bin','r bin sf','p bin sf']

accuracy_workbook = xlsxwriter.Workbook('classifier_output.xlsx',options={'nan_inf_to_errors':True})
worksheet = accuracy_workbook.add_worksheet('unit_accuracy')

# > chance and > shuff

temp = data_dict_all['M1_dicts']['classification_output']

r_level_accuracies = [np.sum(temp['aft_cue'][:,1] > temp['aft_cue'][:,3])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,1] > temp['bfr_res'][:,3])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,1] > temp['aft_res'][:,3])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,1] > temp['res_win'][:,3])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,1] > temp['concat'][:,3])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,5] > temp['aft_cue'][:,7])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,5] > temp['bfr_res'][:,7])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,5] > temp['aft_res'][:,7])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,5] > temp['res_win'][:,7])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,5] > temp['concat'][:,7])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,11])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,11])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,11])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,11])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,11])/float(np.shape(temp['concat'])[0])]





worksheet.write(0,0,'M1: percent better than shuffled')
worksheet.write_row(0,1,comb_names)
worksheet.write_column(1,0,type_names)
worksheet.write_column(1,1,r_level_accuracies)
worksheet.write_column(1,2,p_level_accuracies)
worksheet.write_column(1,3,succ_accuracies)

r_level_accuracies = [np.sum(temp['aft_cue'][:,1] > temp['aft_cue'][:,2])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,1] > temp['bfr_res'][:,2])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,1] > temp['aft_res'][:,2])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,1] > temp['res_win'][:,2])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,1] > temp['concat'][:,2])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,5] > temp['aft_cue'][:,6])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,5] > temp['bfr_res'][:,6])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,5] > temp['aft_res'][:,6])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,5] > temp['res_win'][:,6])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,5] > temp['concat'][:,6])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,10])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,10])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,10])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,10])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,10])/float(np.shape(temp['concat'])[0])]


worksheet.write(0,8,'M1: percent better than chance')
worksheet.write_row(0,9,comb_names)
worksheet.write_column(1,8,type_names)
worksheet.write_column(1,9,r_level_accuracies)
worksheet.write_column(1,10,p_level_accuracies)
worksheet.write_column(1,11,succ_accuracies)

##
temp = data_dict_all['S1_dicts']['classification_output']

r_level_accuracies = [np.sum(temp['aft_cue'][:,1] > temp['aft_cue'][:,3])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,1] > temp['bfr_res'][:,3])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,1] > temp['aft_res'][:,3])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,1] > temp['res_win'][:,3])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,1] > temp['concat'][:,3])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,5] > temp['aft_cue'][:,7])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,5] > temp['bfr_res'][:,7])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,5] > temp['aft_res'][:,7])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,5] > temp['res_win'][:,7])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,5] > temp['concat'][:,7])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,11])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,11])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,11])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,11])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,11])/float(np.shape(temp['concat'])[0])]


worksheet.write(8,0,'S1: percent better than shuffled')
worksheet.write_row(8,1,comb_names)
worksheet.write_column(9,0,type_names)
worksheet.write_column(9,1,r_level_accuracies)
worksheet.write_column(9,2,p_level_accuracies)
worksheet.write_column(9,3,succ_accuracies)

r_level_accuracies = [np.sum(temp['aft_cue'][:,1] > temp['aft_cue'][:,2])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,1] > temp['bfr_res'][:,2])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,1] > temp['aft_res'][:,2])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,1] > temp['res_win'][:,2])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,1] > temp['concat'][:,2])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,5] > temp['aft_cue'][:,6])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,5] > temp['bfr_res'][:,6])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,5] > temp['aft_res'][:,6])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,5] > temp['res_win'][:,6])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,5] > temp['concat'][:,6])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,10])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,10])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,10])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,10])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,10])/float(np.shape(temp['concat'])[0])]


worksheet.write(8,8,'S1: percent better than chance')
worksheet.write_row(8,9,comb_names)
worksheet.write_column(9,8,type_names)
worksheet.write_column(9,9,r_level_accuracies)
worksheet.write_column(9,10,p_level_accuracies)
worksheet.write_column(9,11,succ_accuracies)

##
temp = data_dict_all['PmD_dicts']['classification_output']

r_level_accuracies = [np.sum(temp['aft_cue'][:,1] > temp['aft_cue'][:,3])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,1] > temp['bfr_res'][:,3])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,1] > temp['aft_res'][:,3])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,1] > temp['res_win'][:,3])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,1] > temp['concat'][:,3])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,5] > temp['aft_cue'][:,7])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,5] > temp['bfr_res'][:,7])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,5] > temp['aft_res'][:,7])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,5] > temp['res_win'][:,7])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,5] > temp['concat'][:,7])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,11])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,11])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,11])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,11])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,11])/float(np.shape(temp['concat'])[0])]


worksheet.write(16,0,'PMd: percent better than shuffled')
worksheet.write_row(16,1,comb_names)
worksheet.write_column(17,0,type_names)
worksheet.write_column(17,1,r_level_accuracies)
worksheet.write_column(17,2,p_level_accuracies)
worksheet.write_column(17,3,succ_accuracies)

r_level_accuracies = [np.sum(temp['aft_cue'][:,1] > temp['aft_cue'][:,2])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,1] > temp['bfr_res'][:,2])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,1] > temp['aft_res'][:,2])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,1] > temp['res_win'][:,2])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,1] > temp['concat'][:,2])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,5] > temp['aft_cue'][:,6])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,5] > temp['bfr_res'][:,6])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,5] > temp['aft_res'][:,6])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,5] > temp['res_win'][:,6])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,5] > temp['concat'][:,6])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,10])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,10])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,10])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,10])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,10])/float(np.shape(temp['concat'])[0])]


worksheet.write(16,8,'PMd: percent better than chance')
worksheet.write_row(16,9,comb_names)
worksheet.write_column(17,8,type_names)
worksheet.write_column(17,9,r_level_accuracies)
worksheet.write_column(17,10,p_level_accuracies)
worksheet.write_column(17,11,succ_accuracies)








###########

plt.close('all')


