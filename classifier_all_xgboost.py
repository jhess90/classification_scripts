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
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

########################
# params to set ########
########################

plot_bool = False
plot_hist_bool = True

bin_size = 10 #in ms
time_before = -0.5 #negative value
time_after = 1.0
res_window = [0.3,0.8]

run_pop_bool = False

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
def run_xgboost(value,targets):
    x_train, x_test, y_train, y_test = train_test_split(value,targets,test_size =.33)

    model = xgb.XGBClassifier()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    ##
    chance = float(1) / np.shape(np.unique(targets))[0]

    #run again with shuffled targets
    np.random.shuffle(targets)
    x_train, x_test, y_train, y_test = train_test_split(value,targets,test_size =.33)

    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy_shuffled = accuracy_score(y_test, y_pred)

    return_dict = {'accuracy':accuracy,'accuracy_shuffled':accuracy_shuffled,'chance':chance}

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
    data_dict_all[region_key]['classification_output_all'] = {}

    file_length = np.shape(file_dict.keys())[0]
    avg_and_corr = data_dict_all[region_key]['avg_and_corr']

    total_unit_num = avg_and_corr['aft_cue']['total_unit_num']
        
    output_by_window = {}
    output_by_window_all = {}
    for win_key,win_val in data_dict_all[region_key]['avg_and_corr'].iteritems():
        all_list =  win_val['all_list']
        
        #shorter for devel
        accuracy_list_total = np.zeros((np.shape(all_list)[0],24))
        #accuracy_list_total = np.zeros((4,24))
        
        print 'window: %s' %(win_key)

        #shorter for devel
        for unit_num in range(np.shape(all_list)[0]):
        #for unit_num in range(4):
            
            #succ/fail high even aft cue- b/ of more succ? unequal distrib? 
            #how many times run? What best test/train split?

            r_level_out = run_xgboost(all_list[unit_num][0],all_list[unit_num][1][0,:])
            p_level_out = run_xgboost(all_list[unit_num][0],all_list[unit_num][1][1,:])
            succ_out = run_xgboost(all_list[unit_num][0],all_list[unit_num][1][2,:])

            r_nr_targs = np.where(all_list[unit_num][1][0,:] > 0,1,0)
            p_np_targs = np.where(all_list[unit_num][1][1,:] > 0,1,0)
            
            r_bin_out = run_xgboost(all_list[unit_num][0],r_nr_targs)
            p_bin_out = run_xgboost(all_list[unit_num][0],p_np_targs)
            
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
                
            comb_out = run_xgboost(all_list[unit_num][0],comb)

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

            r_bin_sf_out = run_xgboost(all_list[unit_num][0],r_bin_sf_targ)
            p_bin_sf_out = run_xgboost(all_list[unit_num][0],p_bin_sf_targ)


            accuracy_list_total[unit_num,:] = [r_level_out['accuracy'],r_level_out['chance'],r_level_out['accuracy_shuffled'],p_level_out['accuracy'],p_level_out['chance'],p_level_out['accuracy_shuffled'],succ_out['accuracy'],succ_out['chance'],succ_out['accuracy_shuffled'],comb_out['accuracy'],comb_out['chance'],comb_out['accuracy_shuffled'],r_bin_out['accuracy'],r_bin_out['chance'],r_bin_out['accuracy_shuffled'],p_bin_out['accuracy'],p_bin_out['chance'],p_bin_out['accuracy_shuffled'],r_bin_sf_out['accuracy'],r_bin_sf_out['chance'],r_bin_sf_out['accuracy_shuffled'],p_bin_sf_out['accuracy'],p_bin_sf_out['chance'],p_bin_sf_out['accuracy_shuffled']]

            sys.stdout.write('.')
            sys.stdout.flush()

        output_by_window[win_key] = accuracy_list_total
        print ''

        f,axarr = plt.subplots(4,sharex=True)

        f.suptitle('unit accuracy distribution: %s %s' %(region_key,win_key))
        axarr[0].hist(accuracy_list_total[:,0],color='mediumturquoise',lw=0)
        axarr[0].axvline(accuracy_list_total[:,1][0],color='k',linestyle='dashed',linewidth=1)
        axarr[0].set_xlim([0,1])
        axarr[0].set_title('reward level accuracy')
        axarr[1].hist(accuracy_list_total[:,2],color='darkviolet',lw=0)
        axarr[1].axvline(accuracy_list_total[:,1][0],color='k',linestyle='dashed',linewidth=1)
        axarr[1].set_title('reward level accuracy: shuffled')

        axarr[2].hist(accuracy_list_total[:,3],color='mediumturquoise',lw=0)
        axarr[2].axvline(accuracy_list_total[:,4][0],color='k',linestyle='dashed',linewidth=1)
        axarr[2].set_title('punishment level accuracy')
        axarr[3].hist(accuracy_list_total[:,5],color='darkviolet',lw=0)
        axarr[3].axvline(accuracy_list_total[:,4][0],color='k',linestyle='dashed',linewidth=1)
        axarr[3].set_title('punishment level accuracy: shuffled')

        for axi in axarr.reshape(-1):
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('unit_accuracy_hist_xgboost_1_%s_%s' %(region_key,win_key))
        plt.clf()

        #
        f,axarr = plt.subplots(4,sharex=True)

        f.suptitle('unit accuracy distribution 2: %s %s' %(region_key,win_key))
        axarr[0].hist(accuracy_list_total[:,6],color='mediumturquoise',lw=0)
        axarr[0].axvline(accuracy_list_total[:,7][0],color='k',linestyle='dashed',linewidth=1)
        axarr[0].set_title('succ/fail accuracy')
        axarr[1].hist(accuracy_list_total[:,8],color='darkviolet',lw=0)
        axarr[1].axvline(accuracy_list_total[:,7][0],color='k',linestyle='dashed',linewidth=1)
        axarr[1].set_title('succ/fail accuracy: shuffled')

        axarr[2].hist(accuracy_list_total[:,9],color='mediumturquoise',lw=0)
        axarr[2].axvline(accuracy_list_total[:,10][0],color='k',linestyle='dashed',linewidth=1)
        axarr[2].set_title('comb accuracy')
        axarr[3].hist(accuracy_list_total[:,11],color='darkviolet',lw=0)
        axarr[3].axvline(accuracy_list_total[:,10][0],color='k',linestyle='dashed',linewidth=1)
        axarr[3].set_title('comb accuracy: shuffled')

        for axi in axarr.reshape(-1):
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('unit_accuracy_hist_xgboost_2_%s_%s' %(region_key,win_key))
        plt.clf()

        #
        f,axarr = plt.subplots(4,sharex=True)

        f.suptitle('unit accuracy distribution 3: %s %s' %(region_key,win_key))
        axarr[0].hist(accuracy_list_total[:,12],color='mediumturquoise',lw=0)
        axarr[0].axvline(accuracy_list_total[:,13][0],color='k',linestyle='dashed',linewidth=1)
        axarr[0].set_xlim([0,1])
        axarr[0].set_title('reward binary accuracy')
        axarr[1].hist(accuracy_list_total[:,14],color='darkviolet',lw=0)
        axarr[1].axvline(accuracy_list_total[:,13][0],color='k',linestyle='dashed',linewidth=1)
        axarr[1].set_title('reward binary accuracy: shuffled')

        axarr[2].hist(accuracy_list_total[:,15],color='mediumturquoise',lw=0)
        axarr[2].axvline(accuracy_list_total[:,16][0],color='k',linestyle='dashed',linewidth=1)
        axarr[2].set_title('punishment binary accuracy')
        axarr[3].hist(accuracy_list_total[:,17],color='darkviolet',lw=0)
        axarr[3].axvline(accuracy_list_total[:,16][0],color='k',linestyle='dashed',linewidth=1)
        axarr[3].set_title('punishment binary accuracy: shuffled')

        for axi in axarr.reshape(-1):
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('unit_accuracy_hist_xgboost_3_%s_%s' %(region_key,win_key))
        plt.clf()

        #
        f,axarr = plt.subplots(4,sharex=True)

        f.suptitle('unit accuracy distribution 4: %s %s' %(region_key,win_key))

        axarr[0].hist(accuracy_list_total[:,18],color='mediumturquoise',lw=0)
        axarr[0].axvline(accuracy_list_total[:,19][0],color='k',linestyle='dashed',linewidth=1)
        axarr[0].set_title('reward binary succ fail accuracy')
        axarr[1].hist(accuracy_list_total[:,20],color='darkviolet',lw=0)
        axarr[1].axvline(accuracy_list_total[:,19][0],color='k',linestyle='dashed',linewidth=1)
        axarr[1].set_title('reward binary succ fail accuracy: shuffled')

        axarr[2].hist(accuracy_list_total[:,21],color='mediumturquoise',lw=0)
        axarr[2].axvline(accuracy_list_total[:,22][0],color='k',linestyle='dashed',linewidth=1)
        axarr[2].set_title('punishment binary succ fail accuracy')
        axarr[3].hist(accuracy_list_total[:,23],color='darkviolet',lw=0)
        axarr[3].axvline(accuracy_list_total[:,22][0],color='k',linestyle='dashed',linewidth=1)
        axarr[3].set_title('punishment binary succ fail accuracy: shuffled')

        for axi in axarr.reshape(-1):
            axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('unit_accuracy_hist_xgboost_4_%s_%s' %(region_key,win_key))
        plt.clf()

        #run on whole data
        file_overview = np.zeros((np.shape(file_dict.keys())[0],2))
        for block_num in range(np.shape(file_dict.keys())[0]):
            file_overview[block_num] = [np.shape(file_dict[block_num][region_key][win_key])[0],np.shape(file_dict[block_num][region_key][win_key])[1]]
        
        ind = 0
        r_all = ()
        p_all = ()
        s_all = ()
        unit = int(0)
        for block_num in range(np.shape(file_dict.keys())[0]):
            for temp in range(int(file_overview[block_num,1])):
                r_all = np.append(r_all,all_list[unit][1][0,:])
                p_all = np.append(p_all,all_list[unit][1][1,:])
                s_all = np.append(s_all,all_list[unit][1][2,:])
            unit += int(file_overview[block_num,1])

        rps_all = [r_all,p_all,s_all]

        frs_all = ()
        for temp in range(np.shape(all_list)[0]):
            if temp == 0:
                frs_all = all_list[temp][0]
            else:
                frs_all = np.append(frs_all,all_list[temp][0],axis=0)

        if run_pop_bool:
            print 'running all'
            r_level_all_out = run_xgboost(frs_all,rps_all[0])
            p_level_all_out = run_xgboost(frs_all,rps_all[1])
            succ_all_out = run_xgboost(frs_all,rps_all[2])
        
        ##
        r_nr_targs = np.where(rps_all[0] > 0,1,0)
        p_np_targs = np.where(rps_all[1] > 0,1,0)
            
        if run_pop_bool:
            r_bin_all_out = run_xgboost(frs_all,r_nr_targs)
            p_bin_all_out = run_xgboost(frs_all,p_np_targs)
            
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

        if run_pop_bool:                
            comb_all_out = run_xgboost(frs_all,comb)

        r_bin_sf_targ = np.zeros((np.shape(r_nr_targs)[0]))
        for i in range(np.shape(r_nr_targs)[0]):
            if r_nr_targs[i] == 0 and rps_all[2][i] == 0:
                r_bin_sf_targ[i] = 0
            elif r_nr_targs[i] == 1 and rps_all[2][i] == 0:
                r_bin_sf_targ[i] = 1
            elif r_nr_targs[i] == 0 and rps_all[2][i] == 1:
                r_bin_sf_targ[i] = 2
            elif r_nr_targs[i] == 1 and rps_all[2][i] == 1:
                r_bin_sf_targ[i] = 3

        p_bin_sf_targ = np.zeros((np.shape(p_np_targs)[0]))
        for i in range(np.shape(p_np_targs)[0]):
            if p_np_targs[i] == 0 and rps_all[2][i] == 0:
                p_bin_sf_targ[i] = 0
            elif p_np_targs[i] == 1 and rps_all[2][i] == 0:
                p_bin_sf_targ[i] = 1
            elif p_np_targs[i] == 0 and rps_all[2][i] == 1:
                p_bin_sf_targ[i] = 2
            elif p_np_targs[i] == 1 and rps_all[2][i] == 1:
                p_bin_sf_targ[i] = 3

        if run_pop_bool:
            r_bin_sf_all_out = run_xgboost(frs_all,r_bin_sf_targ)
            p_bin_sf_all_out = run_xgboost(frs_all,p_bin_sf_targ)

        if not run_pop_bool:
            r_level_all_out = {'accuracy':1,'accuracy_shuffled':1,'chance':0}
            p_level_all_out = {'accuracy':1,'accuracy_shuffled':1,'chance':0}
            succ_all_out = {'accuracy':1,'accuracy_shuffled':1,'chance':0}
            r_bin_all_out = {'accuracy':1,'accuracy_shuffled':1,'chance':0}
            p_bin_all_out = {'accuracy':1,'accuracy_shuffled':1,'chance':0}
            comb_all_out = {'accuracy':1,'accuracy_shuffled':1,'chance':0}
            r_bin_sf_all_out = {'accuracy':1,'accuracy_shuffled':1,'chance':0}
            p_bin_sf_all_out = {'accuracy':1,'accuracy_shuffled':1,'chance':0}

        output_by_window_all[win_key] = {'r_level_all':r_level_all_out,'p_level_all':p_level_all_out,'succ_all':succ_all_out,'r_bin_all':r_bin_all_out,'p_bin_all':p_bin_all_out,'comb_all':comb_all_out,'r_bin_sf_all':r_bin_sf_all_out,'p_bin_sf_all':p_bin_sf_all_out}

    data_dict_all[region_key]['classification_output'] = output_by_window
    data_dict_all[region_key]['classification_output_all'] = output_by_window_all

np.save('backup.npy',data_dict_all)



#################


#if xlsx currently exists delte, or else won't write properly
if os.path.isfile('xgb_output.xlsx'):
    os.remove('xgb_output.xlsx')

type_names = ['aft cue','bfr res','aft res','res win','concat']
comb_names = ['r levels','p levels','succ/fail','comb','r bin','p bin','r bin sf','p bin sf']

accuracy_workbook = xlsxwriter.Workbook('xgb_output.xlsx',options={'nan_inf_to_errors':True})
worksheet = accuracy_workbook.add_worksheet('unit_accuracy')

# > chance and > shuff

temp = data_dict_all['M1_dicts']['classification_output']

r_level_accuracies = [np.sum(temp['aft_cue'][:,0] > temp['aft_cue'][:,2])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] > temp['bfr_res'][:,2])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] > temp['aft_res'][:,2])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] > temp['res_win'][:,2])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] > temp['concat'][:,2])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,3] > temp['aft_cue'][:,5])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] > temp['bfr_res'][:,5])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] > temp['aft_res'][:,5])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] > temp['res_win'][:,5])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] > temp['concat'][:,5])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,6] > temp['aft_cue'][:,8])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] > temp['bfr_res'][:,8])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] > temp['aft_res'][:,8])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] > temp['res_win'][:,8])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] > temp['concat'][:,8])/float(np.shape(temp['concat'])[0])]

comb_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,11])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,11])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,11])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,11])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,11])/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,12] > temp['aft_cue'][:,14])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,12] > temp['bfr_res'][:,14])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,12] > temp['aft_res'][:,14])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,12] > temp['res_win'][:,14])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,12] > temp['concat'][:,14])/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,15] > temp['aft_cue'][:,17])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,15] > temp['bfr_res'][:,17])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,15] > temp['aft_res'][:,17])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,15] > temp['res_win'][:,17])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,15] > temp['concat'][:,17])/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies =[np.sum(temp['aft_cue'][:,18] > temp['aft_cue'][:,20])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,18] > temp['bfr_res'][:,20])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,18] > temp['aft_res'][:,20])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,18] > temp['res_win'][:,20])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,18] > temp['concat'][:,20])/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,21] > temp['aft_cue'][:,23])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,21] > temp['bfr_res'][:,23])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,21] > temp['aft_res'][:,23])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,21] > temp['res_win'][:,23])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,21] > temp['concat'][:,23])/float(np.shape(temp['concat'])[0])]

worksheet.write(0,0,'M1: percent better than shuffled')
worksheet.write_row(0,1,comb_names)
worksheet.write_column(1,0,type_names)
worksheet.write_column(1,1,r_level_accuracies)
worksheet.write_column(1,2,p_level_accuracies)
worksheet.write_column(1,3,succ_accuracies)
worksheet.write_column(1,4,comb_accuracies)
worksheet.write_column(1,5,r_bin_accuracies)
worksheet.write_column(1,6,p_bin_accuracies)
worksheet.write_column(1,7,r_bin_sf_accuracies)
worksheet.write_column(1,8,p_bin_sf_accuracies)

r_level_accuracies = [np.sum(temp['aft_cue'][:,0] > temp['aft_cue'][:,1])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] > temp['bfr_res'][:,1])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] > temp['aft_res'][:,1])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] > temp['res_win'][:,1])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] > temp['concat'][:,1])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,3] > temp['aft_cue'][:,4])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] > temp['bfr_res'][:,4])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] > temp['aft_res'][:,4])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] > temp['res_win'][:,4])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] > temp['concat'][:,4])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,6] > temp['aft_cue'][:,7])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] > temp['bfr_res'][:,7])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] > temp['aft_res'][:,7])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] > temp['res_win'][:,7])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] > temp['concat'][:,7])/float(np.shape(temp['concat'])[0])]

comb_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,10])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,10])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,10])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,10])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,10])/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,12] > temp['aft_cue'][:,13])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,12] > temp['bfr_res'][:,13])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,12] > temp['aft_res'][:,13])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,12] > temp['res_win'][:,13])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,12] > temp['concat'][:,13])/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,15] > temp['aft_cue'][:,16])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,15] > temp['bfr_res'][:,16])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,15] > temp['aft_res'][:,16])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,15] > temp['res_win'][:,16])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,15] > temp['concat'][:,16])/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies =[np.sum(temp['aft_cue'][:,18] > temp['aft_cue'][:,19])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,18] > temp['bfr_res'][:,19])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,18] > temp['aft_res'][:,19])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,18] > temp['res_win'][:,19])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,18] > temp['concat'][:,19])/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,21] > temp['aft_cue'][:,22])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,21] > temp['bfr_res'][:,22])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,21] > temp['aft_res'][:,22])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,21] > temp['res_win'][:,22])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,21] > temp['concat'][:,22])/float(np.shape(temp['concat'])[0])]

worksheet.write(0,10,'M1: percent better than chance')
worksheet.write_row(0,11,comb_names)
worksheet.write_column(1,10,type_names)
worksheet.write_column(1,11,r_level_accuracies)
worksheet.write_column(1,12,p_level_accuracies)
worksheet.write_column(1,13,succ_accuracies)
worksheet.write_column(1,14,comb_accuracies)
worksheet.write_column(1,15,r_bin_accuracies)
worksheet.write_column(1,16,p_bin_accuracies)
worksheet.write_column(1,17,r_bin_sf_accuracies)
worksheet.write_column(1,18,p_bin_sf_accuracies)

##
temp = data_dict_all['S1_dicts']['classification_output']

r_level_accuracies = [np.sum(temp['aft_cue'][:,0] > temp['aft_cue'][:,2])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] > temp['bfr_res'][:,2])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] > temp['aft_res'][:,2])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] > temp['res_win'][:,2])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] > temp['concat'][:,2])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,3] > temp['aft_cue'][:,5])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] > temp['bfr_res'][:,5])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] > temp['aft_res'][:,5])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] > temp['res_win'][:,5])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] > temp['concat'][:,5])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,6] > temp['aft_cue'][:,8])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] > temp['bfr_res'][:,8])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] > temp['aft_res'][:,8])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] > temp['res_win'][:,8])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] > temp['concat'][:,8])/float(np.shape(temp['concat'])[0])]

comb_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,11])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,11])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,11])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,11])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,11])/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,12] > temp['aft_cue'][:,14])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,12] > temp['bfr_res'][:,14])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,12] > temp['aft_res'][:,14])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,12] > temp['res_win'][:,14])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,12] > temp['concat'][:,14])/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,15] > temp['aft_cue'][:,17])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,15] > temp['bfr_res'][:,17])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,15] > temp['aft_res'][:,17])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,15] > temp['res_win'][:,17])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,15] > temp['concat'][:,17])/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies =[np.sum(temp['aft_cue'][:,18] > temp['aft_cue'][:,20])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,18] > temp['bfr_res'][:,20])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,18] > temp['aft_res'][:,20])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,18] > temp['res_win'][:,20])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,18] > temp['concat'][:,20])/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,21] > temp['aft_cue'][:,23])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,21] > temp['bfr_res'][:,23])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,21] > temp['aft_res'][:,23])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,21] > temp['res_win'][:,23])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,21] > temp['concat'][:,23])/float(np.shape(temp['concat'])[0])]


worksheet.write(8,0,'S1: percent better than shuffled')
worksheet.write_row(8,1,comb_names)
worksheet.write_column(9,0,type_names)
worksheet.write_column(9,1,r_level_accuracies)
worksheet.write_column(9,2,p_level_accuracies)
worksheet.write_column(9,3,succ_accuracies)
worksheet.write_column(9,4,comb_accuracies)
worksheet.write_column(9,5,r_bin_accuracies)
worksheet.write_column(9,6,p_bin_accuracies)
worksheet.write_column(9,7,r_bin_sf_accuracies)
worksheet.write_column(9,8,p_bin_sf_accuracies)

r_level_accuracies = [np.sum(temp['aft_cue'][:,0] > temp['aft_cue'][:,1])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] > temp['bfr_res'][:,1])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] > temp['aft_res'][:,1])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] > temp['res_win'][:,1])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] > temp['concat'][:,1])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,3] > temp['aft_cue'][:,4])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] > temp['bfr_res'][:,4])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] > temp['aft_res'][:,4])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] > temp['res_win'][:,4])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] > temp['concat'][:,4])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,6] > temp['aft_cue'][:,7])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] > temp['bfr_res'][:,7])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] > temp['aft_res'][:,7])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] > temp['res_win'][:,7])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] > temp['concat'][:,7])/float(np.shape(temp['concat'])[0])]

comb_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,10])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,10])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,10])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,10])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,10])/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,12] > temp['aft_cue'][:,13])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,12] > temp['bfr_res'][:,13])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,12] > temp['aft_res'][:,13])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,12] > temp['res_win'][:,13])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,12] > temp['concat'][:,13])/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,15] > temp['aft_cue'][:,16])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,15] > temp['bfr_res'][:,16])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,15] > temp['aft_res'][:,16])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,15] > temp['res_win'][:,16])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,15] > temp['concat'][:,16])/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies =[np.sum(temp['aft_cue'][:,18] > temp['aft_cue'][:,19])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,18] > temp['bfr_res'][:,19])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,18] > temp['aft_res'][:,19])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,18] > temp['res_win'][:,19])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,18] > temp['concat'][:,19])/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,21] > temp['aft_cue'][:,22])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,21] > temp['bfr_res'][:,22])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,21] > temp['aft_res'][:,22])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,21] > temp['res_win'][:,22])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,21] > temp['concat'][:,22])/float(np.shape(temp['concat'])[0])]

worksheet.write(8,10,'S1: percent better than chance')
worksheet.write_row(8,11,comb_names)
worksheet.write_column(9,10,type_names)
worksheet.write_column(9,11,r_level_accuracies)
worksheet.write_column(9,12,p_level_accuracies)
worksheet.write_column(9,13,succ_accuracies)
worksheet.write_column(9,14,comb_accuracies)
worksheet.write_column(9,15,r_bin_accuracies)
worksheet.write_column(9,16,p_bin_accuracies)
worksheet.write_column(9,17,r_bin_sf_accuracies)
worksheet.write_column(9,18,p_bin_sf_accuracies)

##
temp = data_dict_all['PmD_dicts']['classification_output']

r_level_accuracies = [np.sum(temp['aft_cue'][:,0] > temp['aft_cue'][:,2])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] > temp['bfr_res'][:,2])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] > temp['aft_res'][:,2])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] > temp['res_win'][:,2])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] > temp['concat'][:,2])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,3] > temp['aft_cue'][:,5])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] > temp['bfr_res'][:,5])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] > temp['aft_res'][:,5])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] > temp['res_win'][:,5])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] > temp['concat'][:,5])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,6] > temp['aft_cue'][:,8])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] > temp['bfr_res'][:,8])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] > temp['aft_res'][:,8])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] > temp['res_win'][:,8])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] > temp['concat'][:,8])/float(np.shape(temp['concat'])[0])]

comb_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,11])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,11])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,11])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,11])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,11])/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,12] > temp['aft_cue'][:,14])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,12] > temp['bfr_res'][:,14])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,12] > temp['aft_res'][:,14])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,12] > temp['res_win'][:,14])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,12] > temp['concat'][:,14])/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,15] > temp['aft_cue'][:,17])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,15] > temp['bfr_res'][:,17])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,15] > temp['aft_res'][:,17])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,15] > temp['res_win'][:,17])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,15] > temp['concat'][:,17])/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies =[np.sum(temp['aft_cue'][:,18] > temp['aft_cue'][:,20])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,18] > temp['bfr_res'][:,20])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,18] > temp['aft_res'][:,20])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,18] > temp['res_win'][:,20])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,18] > temp['concat'][:,20])/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,21] > temp['aft_cue'][:,23])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,21] > temp['bfr_res'][:,23])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,21] > temp['aft_res'][:,23])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,21] > temp['res_win'][:,23])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,21] > temp['concat'][:,23])/float(np.shape(temp['concat'])[0])]

worksheet.write(16,0,'PMd: percent better than shuffled')
worksheet.write_row(16,1,comb_names)
worksheet.write_column(17,0,type_names)
worksheet.write_column(17,1,r_level_accuracies)
worksheet.write_column(17,2,p_level_accuracies)
worksheet.write_column(17,3,succ_accuracies)
worksheet.write_column(17,4,comb_accuracies)
worksheet.write_column(17,5,r_bin_accuracies)
worksheet.write_column(17,6,p_bin_accuracies)
worksheet.write_column(17,7,r_bin_sf_accuracies)
worksheet.write_column(17,8,p_bin_sf_accuracies)

r_level_accuracies = [np.sum(temp['aft_cue'][:,0] > temp['aft_cue'][:,1])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,0] > temp['bfr_res'][:,1])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,0] > temp['aft_res'][:,1])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,0] > temp['res_win'][:,1])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,0] > temp['concat'][:,1])/float(np.shape(temp['concat'])[0])]
p_level_accuracies = [np.sum(temp['aft_cue'][:,3] > temp['aft_cue'][:,4])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,3] > temp['bfr_res'][:,4])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,3] > temp['aft_res'][:,4])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,3] > temp['res_win'][:,4])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,3] > temp['concat'][:,4])/float(np.shape(temp['concat'])[0])]
succ_accuracies = [np.sum(temp['aft_cue'][:,6] > temp['aft_cue'][:,7])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,6] > temp['bfr_res'][:,7])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,6] > temp['aft_res'][:,7])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,6] > temp['res_win'][:,7])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,6] > temp['concat'][:,7])/float(np.shape(temp['concat'])[0])]

comb_accuracies = [np.sum(temp['aft_cue'][:,9] > temp['aft_cue'][:,10])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,9] > temp['bfr_res'][:,10])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,9] > temp['aft_res'][:,10])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,9] > temp['res_win'][:,10])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,9] > temp['concat'][:,10])/float(np.shape(temp['concat'])[0])]
r_bin_accuracies = [np.sum(temp['aft_cue'][:,12] > temp['aft_cue'][:,13])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,12] > temp['bfr_res'][:,13])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,12] > temp['aft_res'][:,13])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,12] > temp['res_win'][:,13])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,12] > temp['concat'][:,13])/float(np.shape(temp['concat'])[0])]
p_bin_accuracies = [np.sum(temp['aft_cue'][:,15] > temp['aft_cue'][:,16])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,15] > temp['bfr_res'][:,16])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,15] > temp['aft_res'][:,16])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,15] > temp['res_win'][:,16])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,15] > temp['concat'][:,16])/float(np.shape(temp['concat'])[0])]
r_bin_sf_accuracies =[np.sum(temp['aft_cue'][:,18] > temp['aft_cue'][:,19])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,18] > temp['bfr_res'][:,19])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,18] > temp['aft_res'][:,19])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,18] > temp['res_win'][:,19])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,18] > temp['concat'][:,19])/float(np.shape(temp['concat'])[0])]
p_bin_sf_accuracies = [np.sum(temp['aft_cue'][:,21] > temp['aft_cue'][:,22])/float(np.shape(temp['aft_cue'])[0]),np.sum(temp['bfr_res'][:,21] > temp['bfr_res'][:,22])/float(np.shape(temp['bfr_res'])[0]),np.sum(temp['aft_res'][:,21] > temp['aft_res'][:,22])/float(np.shape(temp['aft_res'])[0]),np.sum(temp['res_win'][:,21] > temp['res_win'][:,22])/float(np.shape(temp['res_win'])[0]),np.sum(temp['concat'][:,21] > temp['concat'][:,22])/float(np.shape(temp['concat'])[0])]

worksheet.write(16,10,'PMd: percent better than chance')
worksheet.write_row(16,11,comb_names)
worksheet.write_column(17,10,type_names)
worksheet.write_column(17,11,r_level_accuracies)
worksheet.write_column(17,12,p_level_accuracies)
worksheet.write_column(17,13,succ_accuracies)
worksheet.write_column(17,14,comb_accuracies)
worksheet.write_column(17,15,r_bin_accuracies)
worksheet.write_column(17,16,p_bin_accuracies)
worksheet.write_column(17,17,r_bin_sf_accuracies)
worksheet.write_column(17,18,p_bin_sf_accuracies)

worksheet = accuracy_workbook.add_worksheet('all_accuracy')

out_names = ['accuracy','accuracy shuffled','chance']

#
temp = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['r_level_all']
temp2 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['p_level_all']
temp3 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['succ_all']
temp4 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['comb_all']
temp5 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['r_bin_all']
temp6 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['p_bin_all']
temp7 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['r_bin_sf_all']
temp8 = data_dict_all['M1_dicts']['classification_output_all']['aft_cue']['p_bin_sf_all']
ac = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['r_level_all']
temp2 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['p_level_all']
temp3 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['succ_all']
temp4 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['comb_all']
temp5 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['r_bin_all']
temp6 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['p_bin_all']
temp7 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['r_bin_sf_all']
temp8 = data_dict_all['M1_dicts']['classification_output_all']['bfr_res']['p_bin_sf_all']
br = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['r_level_all']
temp2 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['p_level_all']
temp3 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['succ_all']
temp4 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['comb_all']
temp5 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['r_bin_all']
temp6 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['p_bin_all']
temp7 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['r_bin_sf_all']
temp8 = data_dict_all['M1_dicts']['classification_output_all']['aft_res']['p_bin_sf_all']
ar = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['M1_dicts']['classification_output_all']['res_win']['r_level_all']
temp2 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['p_level_all']
temp3 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['succ_all']
temp4 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['comb_all']
temp5 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['r_bin_all']
temp6 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['p_bin_all']
temp7 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['r_bin_sf_all']
temp8 = data_dict_all['M1_dicts']['classification_output_all']['res_win']['p_bin_sf_all']
rw = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['M1_dicts']['classification_output_all']['concat']['r_level_all']
temp2 = data_dict_all['M1_dicts']['classification_output_all']['concat']['p_level_all']
temp3 = data_dict_all['M1_dicts']['classification_output_all']['concat']['succ_all']
temp4 = data_dict_all['M1_dicts']['classification_output_all']['concat']['comb_all']
temp5 = data_dict_all['M1_dicts']['classification_output_all']['concat']['r_bin_all']
temp6 = data_dict_all['M1_dicts']['classification_output_all']['concat']['p_bin_all']
temp7 = data_dict_all['M1_dicts']['classification_output_all']['concat']['r_bin_sf_all']
temp8 = data_dict_all['M1_dicts']['classification_output_all']['concat']['p_bin_sf_all']
ct = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

worksheet.write(0,0,'M1')
worksheet.write(0,1,comb_names[0])
worksheet.write(0,4,comb_names[1])
worksheet.write(0,7,comb_names[2])
worksheet.write(0,10,comb_names[3])
worksheet.write(0,13,comb_names[4])
worksheet.write(0,16,comb_names[5])
worksheet.write(0,19,comb_names[6])
worksheet.write(0,22,comb_names[7])

worksheet.write_row(1,1,out_names)
worksheet.write_row(1,4,out_names)
worksheet.write_row(1,7,out_names)
worksheet.write_row(1,10,out_names)
worksheet.write_row(1,13,out_names)
worksheet.write_row(1,16,out_names)
worksheet.write_row(1,19,out_names)
worksheet.write_row(1,22,out_names)

worksheet.write_column(1,0,type_names)
worksheet.write_row(2,1,ac)
worksheet.write_row(3,1,br)
worksheet.write_row(4,1,ar)
worksheet.write_row(5,1,rw)
worksheet.write_row(6,1,ct)


#
temp = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['r_level_all']
temp2 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['p_level_all']
temp3 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['succ_all']
temp4 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['comb_all']
temp5 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['r_bin_all']
temp6 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['p_bin_all']
temp7 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['r_bin_sf_all']
temp8 = data_dict_all['S1_dicts']['classification_output_all']['aft_cue']['p_bin_sf_all']
ac =[temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['r_level_all']
temp2 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['p_level_all']
temp3 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['succ_all']
temp4 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['comb_all']
temp5 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['r_bin_all']
temp6 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['p_bin_all']
temp7 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['r_bin_sf_all']
temp8 = data_dict_all['S1_dicts']['classification_output_all']['bfr_res']['p_bin_sf_all']
br = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['r_level_all']
temp2 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['p_level_all']
temp3 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['succ_all']
temp4 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['comb_all']
temp5 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['r_bin_all']
temp6 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['p_bin_all']
temp7 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['r_bin_sf_all']
temp8 = data_dict_all['S1_dicts']['classification_output_all']['aft_res']['p_bin_sf_all']
ar = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['S1_dicts']['classification_output_all']['res_win']['r_level_all']
temp2 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['p_level_all']
temp3 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['succ_all']
temp4 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['comb_all']
temp5 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['r_bin_all']
temp6 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['p_bin_all']
temp7 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['r_bin_sf_all']
temp8 = data_dict_all['S1_dicts']['classification_output_all']['res_win']['p_bin_sf_all']
rw = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['S1_dicts']['classification_output_all']['concat']['r_level_all']
temp2 = data_dict_all['S1_dicts']['classification_output_all']['concat']['p_level_all']
temp3 = data_dict_all['S1_dicts']['classification_output_all']['concat']['succ_all']
temp4 = data_dict_all['S1_dicts']['classification_output_all']['concat']['comb_all']
temp5 = data_dict_all['S1_dicts']['classification_output_all']['concat']['r_bin_all']
temp6 = data_dict_all['S1_dicts']['classification_output_all']['concat']['p_bin_all']
temp7 = data_dict_all['S1_dicts']['classification_output_all']['concat']['r_bin_sf_all']
temp8 = data_dict_all['S1_dicts']['classification_output_all']['concat']['p_bin_sf_all']
ct = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

worksheet.write(8,0,'S1')
worksheet.write(0,1,comb_names[0])
worksheet.write(0,4,comb_names[1])
worksheet.write(0,7,comb_names[2])
worksheet.write(0,10,comb_names[3])
worksheet.write(0,13,comb_names[4])
worksheet.write(0,16,comb_names[5])
worksheet.write(0,19,comb_names[6])
worksheet.write(0,22,comb_names[7])

worksheet.write_row(1,1,out_names)
worksheet.write_row(1,4,out_names)
worksheet.write_row(1,7,out_names)
worksheet.write_row(1,10,out_names)
worksheet.write_row(1,13,out_names)
worksheet.write_row(1,16,out_names)
worksheet.write_row(1,19,out_names)
worksheet.write_row(1,22,out_names)

worksheet.write_column(9,0,type_names)
worksheet.write_row(10,1,ac)
worksheet.write_row(11,1,br)
worksheet.write_row(12,1,ar)
worksheet.write_row(13,1,rw)
worksheet.write_row(14,1,ct)

#
temp = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['r_level_all']
temp2 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['p_level_all']
temp3 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['succ_all']
temp4 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['comb_all']
temp5 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['r_bin_all']
temp6 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['p_bin_all']
temp7 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['r_bin_sf_all']
temp8 = data_dict_all['PmD_dicts']['classification_output_all']['aft_cue']['p_bin_sf_all']
ac = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['r_level_all']
temp2 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['p_level_all']
temp3 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['succ_all']
temp4 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['comb_all']
temp5 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['r_bin_all']
temp6 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['p_bin_all']
temp7 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['r_bin_sf_all']
temp8 = data_dict_all['PmD_dicts']['classification_output_all']['bfr_res']['p_bin_sf_all']
br = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['r_level_all']
temp2 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['p_level_all']
temp3 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['succ_all']
temp4 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['comb_all']
temp5 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['r_bin_all']
temp6 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['p_bin_all']
temp7 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['r_bin_sf_all']
temp8 = data_dict_all['PmD_dicts']['classification_output_all']['aft_res']['p_bin_sf_all']
ar = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['r_level_all']
temp2 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['p_level_all']
temp3 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['succ_all']
temp4 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['comb_all']
temp5 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['r_bin_all']
temp6 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['p_bin_all']
temp7 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['r_bin_sf_all']
temp8 = data_dict_all['PmD_dicts']['classification_output_all']['res_win']['p_bin_sf_all']
rw = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

temp = data_dict_all['PmD_dicts']['classification_output_all']['concat']['r_level_all']
temp2 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['p_level_all']
temp3 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['succ_all']
temp4 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['comb_all']
temp5 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['r_bin_all']
temp6 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['p_bin_all']
temp7 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['r_bin_sf_all']
temp8 = data_dict_all['PmD_dicts']['classification_output_all']['concat']['p_bin_sf_all']
ct = [temp['accuracy'],temp['accuracy_shuffled'],temp['chance'],temp2['accuracy'],temp2['accuracy_shuffled'],temp2['chance'],temp3['accuracy'],temp3['accuracy_shuffled'],temp3['chance'],temp4['accuracy'],temp4['accuracy_shuffled'],temp4['chance'],temp5['accuracy'],temp5['accuracy_shuffled'],temp5['chance'],temp6['accuracy'],temp6['accuracy_shuffled'],temp6['chance'],temp7['accuracy'],temp7['accuracy_shuffled'],temp7['chance'],temp8['accuracy'],temp8['accuracy_shuffled'],temp8['chance']]

worksheet.write(16,0,'PmD')
worksheet.write(0,1,comb_names[0])
worksheet.write(0,4,comb_names[1])
worksheet.write(0,7,comb_names[2])
worksheet.write(0,10,comb_names[3])
worksheet.write(0,13,comb_names[4])
worksheet.write(0,16,comb_names[5])
worksheet.write(0,19,comb_names[6])
worksheet.write(0,22,comb_names[7])

worksheet.write_row(1,1,out_names)
worksheet.write_row(1,4,out_names)
worksheet.write_row(1,7,out_names)
worksheet.write_row(1,10,out_names)
worksheet.write_row(1,13,out_names)
worksheet.write_row(1,16,out_names)
worksheet.write_row(1,19,out_names)
worksheet.write_row(1,22,out_names)

worksheet.write_column(17,0,type_names)
worksheet.write_row(18,1,ac)
worksheet.write_row(19,1,br)
worksheet.write_row(20,1,ar)
worksheet.write_row(21,1,rw)
worksheet.write_row(22,1,ct)


###########

plt.close('all')


