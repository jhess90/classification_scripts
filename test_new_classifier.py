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
        accuracy_list_total = np.zeros((np.shape(all_list)[0],12))
        
        print 'window: %s' %(win_key)

        for unit_num in range(np.shape(all_list)[0]):
            
            #TODO set up diff variations of target here
            #TODO report percentage of accuracy above chance. Look at each unit, then pop as whole
            #questions: better than chance, or determine chance by shuffling targets?
            #succ/fail high even aft cue- b/ of more succ? unequal distrib? 
            #how many times run? What best test/train split?
            #use SAMME.R for now

            r_level_out = run_adaboost(all_list[unit_num][0],all_list[unit_num][1][0,:])
            p_level_out = run_adaboost(all_list[unit_num][0],all_list[unit_num][1][1,:])
            succ_out = run_adaboost(all_list[unit_num][0],all_list[unit_num][1][2,:])
        
            #r nr, p np, m, v, r/nr comb w/ and w/o succ

            accuracy_list_total[unit_num,:] = [r_level_out['accuracy_samme'],r_level_out['accuracy_sammer'],r_level_out['chance'],r_level_out['shuffle_accuracy_sammer'],p_level_out['accuracy_samme'],p_level_out['accuracy_sammer'],p_level_out['chance'],p_level_out['shuffle_accuracy_sammer'],succ_out['accuracy_samme'],succ_out['accuracy_sammer'],succ_out['chance'],succ_out['shuffle_accuracy_sammer']]


            print 'r: %s, p: %s, succ: %s' %(r_level_out['accuracy_sammer'],p_level_out['accuracy_sammer'],succ_out['accuracy_sammer'])

        
        output_by_window[win_key] = accuracy_list_total
            
    data_dict_all[region_key]['classification_output'] = output_by_window
pdb.set_trace()


##saving 
if sig_only_bool:
    np.save('model_save_sig.npy',data_dict_all)
else:
    np.save('model_save.npy',data_dict_all)
data_dict_all['file_dict'] = file_dict


#for region_key,region_val in data_dict_all.iteritems():
type_names = ['aft cue','bfr res','aft res','res win','concat']
model_names = ['r only','p only','linear','difference','div nl Y','separate add','avg response separate']

if sig_only_bool:
    aic_workbook = xlsxwriter.Workbook('model_aic_sig.xlsx',options={'nan_inf_to_errors':True})
else:
    aic_workbook = xlsxwriter.Workbook('model_aic.xlsx',options={'nan_inf_to_errors':True})

worksheet = aic_workbook.add_worksheet('model_output')
# Light red fill with dark red text.
format1 = aic_workbook.add_format({'bg_color':'#FFC7CE','font_color': '#9C0006'})

lin_aics = [data_dict_all['M1_dicts']['models']['linear']['aft_cue']['AIC_overall'],data_dict_all['M1_dicts']['models']['linear']['bfr_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['linear']['aft_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['linear']['res_win']['AIC_overall'],data_dict_all['M1_dicts']['models']['linear']['concat']['AIC_overall']]
diff_aics = [data_dict_all['M1_dicts']['models']['diff']['aft_cue']['AIC_overall'],data_dict_all['M1_dicts']['models']['diff']['bfr_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['diff']['aft_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['diff']['res_win']['AIC_overall'],data_dict_all['M1_dicts']['models']['diff']['concat']['AIC_overall']]
div_nl_aics = [data_dict_all['M1_dicts']['models']['div_nl']['aft_cue']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl']['bfr_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl']['aft_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl']['res_win']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl']['concat']['AIC_overall']]
div_nl_noe_aics = [data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_cue']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_noe']['bfr_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_noe']['res_win']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_noe']['concat']['AIC_overall']]
div_nl_Y_aics = [data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_cue']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_Y']['bfr_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_Y']['res_win']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_Y']['concat']['AIC_overall']]
div_nl_separate_add_aics = [data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_cue']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['bfr_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['res_win']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['concat']['AIC_overall']]
div_nl_separate_multiply_aics = [data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_res']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['res_win']['AIC_overall'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['concat']['AIC_overall']]

worksheet.write(0,0,'M1')
worksheet.write_row(0,1,type_names)
worksheet.write_row(1,1,lin_aics)
worksheet.write_row(2,1,diff_aics)
worksheet.write_row(3,1,div_nl_aics)
worksheet.write_row(4,1,div_nl_noe_aics)
worksheet.write_row(5,1,div_nl_Y_aics)
worksheet.write_row(6,1,div_nl_separate_add_aics)
worksheet.write_row(7,1,div_nl_separate_multiply_aics)
worksheet.write_column(1,0,model_names)

worksheet.conditional_format('B2:B8', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('C2:C8', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('D2:D8', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('E2:E8', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('F2:F8', {'type': 'bottom', 'value':'1', 'format':format1})

lin_aics = [data_dict_all['S1_dicts']['models']['linear']['aft_cue']['AIC_overall'],data_dict_all['S1_dicts']['models']['linear']['bfr_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['linear']['aft_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['linear']['res_win']['AIC_overall'],data_dict_all['S1_dicts']['models']['linear']['concat']['AIC_overall']]
diff_aics = [data_dict_all['S1_dicts']['models']['diff']['aft_cue']['AIC_overall'],data_dict_all['S1_dicts']['models']['diff']['bfr_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['diff']['aft_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['diff']['res_win']['AIC_overall'],data_dict_all['S1_dicts']['models']['diff']['concat']['AIC_overall']]
div_nl_aics = [data_dict_all['S1_dicts']['models']['div_nl']['aft_cue']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl']['bfr_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl']['aft_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl']['res_win']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl']['concat']['AIC_overall']]
div_nl_noe_aics = [data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_cue']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_noe']['bfr_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_noe']['res_win']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_noe']['concat']['AIC_overall']]
div_nl_Y_aics = [data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_cue']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_Y']['bfr_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_Y']['res_win']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_Y']['concat']['AIC_overall']]
div_nl_separate_add_aics = [data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_cue']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['bfr_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['res_win']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['concat']['AIC_overall']]
div_nl_separate_multiply_aics = [data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_res']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['res_win']['AIC_overall'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['concat']['AIC_overall']]

worksheet.write(12,0,'S1')
worksheet.write_row(12,1,type_names)
worksheet.write_row(13,1,lin_aics)
worksheet.write_row(14,1,diff_aics)
worksheet.write_row(15,1,div_nl_aics)
worksheet.write_row(16,1,div_nl_noe_aics)
worksheet.write_row(17,1,div_nl_Y_aics)
worksheet.write_row(18,1,div_nl_separate_add_aics)
worksheet.write_row(19,1,div_nl_separate_multiply_aics)
worksheet.write_column(13,0,model_names)

worksheet.conditional_format('B14:B20', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('C14:C20', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('D14:D20', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('E14:E20', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('F14:F20', {'type': 'bottom', 'value':'1', 'format':format1})

lin_aics = [data_dict_all['PmD_dicts']['models']['linear']['aft_cue']['AIC_overall'],data_dict_all['PmD_dicts']['models']['linear']['bfr_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['linear']['aft_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['linear']['res_win']['AIC_overall'],data_dict_all['PmD_dicts']['models']['linear']['concat']['AIC_overall']]
diff_aics = [data_dict_all['PmD_dicts']['models']['diff']['aft_cue']['AIC_overall'],data_dict_all['PmD_dicts']['models']['diff']['bfr_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['diff']['aft_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['diff']['res_win']['AIC_overall'],data_dict_all['PmD_dicts']['models']['diff']['concat']['AIC_overall']]
div_nl_aics = [data_dict_all['PmD_dicts']['models']['div_nl']['aft_cue']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl']['bfr_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl']['aft_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl']['res_win']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl']['concat']['AIC_overall']]
div_nl_noe_aics = [data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_cue']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['bfr_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['res_win']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['concat']['AIC_overall']]
div_nl_Y_aics = [data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_cue']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['bfr_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['res_win']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['concat']['AIC_overall']]
div_nl_separate_add_aics = [data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_cue']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['bfr_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['res_win']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['concat']['AIC_overall']]
div_nl_separate_multiply_aics = [data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_cue']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['bfr_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_res']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['res_win']['AIC_overall'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['concat']['AIC_overall']]

worksheet.write(24,0,'PMd')
worksheet.write_row(24,1,type_names)
worksheet.write_row(25,1,lin_aics)
worksheet.write_row(26,1,diff_aics)
worksheet.write_row(27,1,div_nl_aics)
worksheet.write_row(28,1,div_nl_noe_aics)
worksheet.write_row(29,1,div_nl_Y_aics)
worksheet.write_row(30,1,div_nl_separate_add_aics)
worksheet.write_row(31,1,div_nl_separate_multiply_aics)
worksheet.write_column(25,0,model_names)

worksheet.conditional_format('B26:B32', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('C26:C32', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('D26:D32', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('E26:E32', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('F26:F32', {'type': 'bottom', 'value':'1', 'format':format1})


#perc of units best fit to each model
type_names_together = ['aft_cue', 'bfr_res', 'aft_res', 'res_win', 'concat']
for region_key,region_val in data_dict_all.iteritems():
    if not region_key == 'file_dict':
        #for type_key,type_val in data_dict_all[region_key]['models']['linear'].iteritems():
        

        all_percs = np.zeros((np.shape(type_names)[0],np.shape(model_names)[0]))    
        #
        sig_percs = np.zeros((np.shape(type_names)[0],np.shape(model_names)[0]))
        all_percs_r2 = np.zeros((np.shape(type_names)[0],np.shape(model_names)[0]))
        sig_percs_r2 = np.zeros((np.shape(type_names)[0],np.shape(model_names)[0]))
        num_any_sig_window = np.zeros((np.shape(type_names)[0]))
 
        all_no_fit = np.zeros((np.shape(type_names)[0],np.shape(model_names)[0]))
        total_num_units_window = np.zeros((np.shape(type_names)[0]))

        data_dict_all[region_key]['model_summary'] = {}
        for i in range(np.shape(type_names_together)[0]):
            type_key = type_names_together[i]
    
            all_model_AICs = np.column_stack((data_dict_all[region_key]['models']['linear'][type_key]['AIC_total'],data_dict_all[region_key]['models']['diff'][type_key]['AIC_total'],data_dict_all[region_key]['models']['div_nl'][type_key]['AIC_total'],data_dict_all[region_key]['models']['div_nl_noe'][type_key]['AIC_total'],data_dict_all[region_key]['models']['div_nl_Y'][type_key]['AIC_total'],data_dict_all[region_key]['models']['div_nl_separate_add'][type_key]['AIC_total'],data_dict_all[region_key]['models']['div_nl_separate_multiply'][type_key]['AIC_total']))

            #all_model_adj_r2
            all_model_adj_r2 = np.column_stack((data_dict_all[region_key]['models']['linear'][type_key]['r_sq_adj_total'],data_dict_all[region_key]['models']['diff'][type_key]['r_sq_adj_total'],data_dict_all[region_key]['models']['div_nl'][type_key]['r_sq_adj_total'],data_dict_all[region_key]['models']['div_nl_noe'][type_key]['r_sq_adj_total'],data_dict_all[region_key]['models']['div_nl_Y'][type_key]['r_sq_adj_total'],data_dict_all[region_key]['models']['div_nl_separate_add'][type_key]['r_sq_adj_total'],data_dict_all[region_key]['models']['div_nl_separate_multiply'][type_key]['r_sq_adj_total']))

            ###
            all_model_p_vals = np.column_stack((data_dict_all[region_key]['models']['linear'][type_key]['p_val_total'],data_dict_all[region_key]['models']['diff'][type_key]['p_val_total'],data_dict_all[region_key]['models']['div_nl'][type_key]['p_val_total'],data_dict_all[region_key]['models']['div_nl_noe'][type_key]['p_val_total'],data_dict_all[region_key]['models']['div_nl_Y'][type_key]['p_val_total'],data_dict_all[region_key]['models']['div_nl_separate_add'][type_key]['p_val_total'],data_dict_all[region_key]['models']['div_nl_separate_multiply'][type_key]['p_val_total']))
            
            all_p_bool = all_model_p_vals < 0.05
            #
            any_sig = np.sum(all_p_bool,axis=1) > 0            
            sig_multiple_models = np.sum(all_p_bool,axis=1)
            num_sig_multiple_models = np.sum(sig_multiple_models > 0)
            perc_sig_multiple_models = num_sig_multiple_models / float(np.shape(sig_multiple_models)[0])
            
            for j in range(np.shape(sig_multiple_models)[0]):
                if sig_multiple_models[j]:
                    p_vals = all_p_bool[j,:]
                    AICs = np.array((all_model_AICs[j,:],range(np.shape(model_names)[0])))
                    #keep model type (number) with AIC
                    sig_AICs = AICs[:,p_vals]
                    model_type = sig_AICs[1,np.argmin(sig_AICs[0,:])]
            
                    #TODO finish this part

            arg_mins = np.argmin(all_model_AICs,axis=1)
            #
            sig_arg_mins = arg_mins[any_sig]
            arg_max_r2_adj = np.argmax(all_model_adj_r2,axis=1)
            sig_arg_max_r2_adj = arg_max_r2_adj[any_sig]

            perc_min = np.zeros((np.shape(all_model_AICs)[1]))
            #
            sig_perc_min = np.zeros((np.shape(all_model_AICs)[1]))
            perc_max = np.zeros((np.shape(all_model_adj_r2)[1]))
            sig_perc_max = np.zeros((np.shape(all_model_adj_r2)[1]))
            for j in range(np.shape(all_model_AICs)[1]):
                perc_min[j] = np.sum(arg_mins == j) / float(np.shape(all_model_AICs)[0])
                #
                sig_perc_min[j] = np.sum(sig_arg_mins == j) / float(np.shape(sig_arg_mins)[0])
                perc_max[j] = np.sum(arg_max_r2_adj == j) / float(np.shape(all_model_adj_r2)[0])
                sig_perc_max[j] = np.sum(sig_arg_max_r2_adj == j) / float(np.shape(sig_arg_max_r2_adj)[0])
                
            all_percs[i,:] = perc_min
            #
            sig_percs[i,:] = sig_perc_min
            all_percs_r2[i,:] = perc_max
            sig_percs_r2[i,:] = sig_perc_max
            
            all_no_fit[i,:] = [data_dict_all[region_key]['models']['linear'][type_key]['perc_units_no_fit'],data_dict_all[region_key]['models']['diff'][type_key]['perc_units_no_fit'],data_dict_all[region_key]['models']['div_nl'][type_key]['perc_units_no_fit'],data_dict_all[region_key]['models']['div_nl_noe'][type_key]['perc_units_no_fit'],data_dict_all[region_key]['models']['div_nl_Y'][type_key]['perc_units_no_fit'],data_dict_all[region_key]['models']['div_nl_separate_add'][type_key]['perc_units_no_fit'],data_dict_all[region_key]['models']['div_nl_separate_multiply'][type_key]['perc_units_no_fit']]

            total_num_units_window[i] = data_dict_all[region_key]['models']['linear'][type_key]['total_num_units']
            
            #
            num_any_sig_window[i] = np.sum(any_sig)

        #
        data_dict_all[region_key]['model_summary'] = {'all_percs':all_percs,'all_no_fit':all_no_fit,'total_num_units_window':total_num_units_window,'all_percs_r2':all_percs_r2,'sig_percs_r2':sig_percs_r2,'num_any_sig':num_any_sig_window,'sig_percs':sig_percs}



worksheet = aic_workbook.add_worksheet('perc_unit_best_fit')
worksheet.write(0,0,'M1: all')
worksheet.write_row(0,1,model_names)
worksheet.write_column(1,0,type_names)
percs = data_dict_all['M1_dicts']['model_summary']['all_percs']
total_units = data_dict_all['M1_dicts']['model_summary']['total_num_units_window']
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+1,1,percs[i,:])
worksheet.write(0,8,'total units')
worksheet.write_column(1,8,total_units)
percs = data_dict_all['M1_dicts']['model_summary']['sig_percs']
total_units = data_dict_all['M1_dicts']['model_summary']['num_any_sig']
worksheet.write_row(0,11,model_names)
worksheet.write_column(1,10,type_names)
worksheet.write(0,10,'M1: sig')
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+1,11,percs[i,:])
worksheet.write(0,18,'total units')
worksheet.write_column(1,18,total_units)

worksheet.write(7,0,'S1: all')
worksheet.write_row(7,1,model_names)
worksheet.write_column(8,0,type_names)
percs = data_dict_all['S1_dicts']['model_summary']['all_percs']
total_units = data_dict_all['S1_dicts']['model_summary']['total_num_units_window']
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+8,1,percs[i,:])
worksheet.write(7,8,'total units')
worksheet.write_column(8,8,total_units)
percs = data_dict_all['S1_dicts']['model_summary']['sig_percs']
total_units = data_dict_all['S1_dicts']['model_summary']['num_any_sig']
worksheet.write_row(7,11,model_names)
worksheet.write_column(8,10,type_names)
worksheet.write(8,10,'S1: sig')
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+8,11,percs[i,:])
worksheet.write(7,18,'total units')
worksheet.write_column(8,18,total_units)

worksheet.write(14,0,'PMd: all')
worksheet.write_row(14,1,model_names)
worksheet.write_column(15,0,type_names)
percs = data_dict_all['PmD_dicts']['model_summary']['all_percs']
total_units = data_dict_all['S1_dicts']['model_summary']['total_num_units_window']
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+15,1,percs[i,:])
worksheet.write(14,8,'total units')
worksheet.write_column(15,8,total_units)
percs = data_dict_all['PmD_dicts']['model_summary']['sig_percs']
total_units = data_dict_all['PmD_dicts']['model_summary']['num_any_sig']
worksheet.write_row(14,11,model_names)
worksheet.write_column(15,10,type_names)
worksheet.write(14,10,'PmD: sig')
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+15,11,percs[i,:])
worksheet.write(14,18,'total units')
worksheet.write_column(15,18,total_units)

worksheet.conditional_format('B2:H2', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B3:H3', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B4:H4', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B5:H5', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B6:H6', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L2:R2', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L3:R3', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L4:R4', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L5:R5', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L6:R6', {'type': 'top', 'value':'1', 'format':format1})

worksheet.conditional_format('B9:H9', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B10:H10', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B11:H11', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B12:H12', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B13:H13', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L9:R9', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L10:R10', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L11:R11', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L12:R12', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L13:R13', {'type': 'top', 'value':'1', 'format':format1})

worksheet.conditional_format('B16:H16', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B17:H17', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B18:H18', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B19:H19', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B20:H20', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L16:R16', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L17:R17', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L18:R18', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L19:R19', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L20:R20', {'type': 'top', 'value':'1', 'format':format1})

#
worksheet = aic_workbook.add_worksheet('perc_unit_best_fit_adj_r2')
worksheet.write(0,0,'M1: all')
worksheet.write_row(0,1,model_names)
worksheet.write_column(1,0,type_names)
percs = data_dict_all['M1_dicts']['model_summary']['all_percs_r2']
total_units = data_dict_all['M1_dicts']['model_summary']['total_num_units_window']
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+1,1,percs[i,:])
worksheet.write(0,8,'total units')
worksheet.write_column(1,8,total_units)
percs = data_dict_all['M1_dicts']['model_summary']['sig_percs_r2']
total_units = data_dict_all['M1_dicts']['model_summary']['num_any_sig']
worksheet.write_row(0,11,model_names)
worksheet.write_column(1,10,type_names)
worksheet.write(0,10,'M1: sig')
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+1,11,percs[i,:])
worksheet.write(0,18,'total units')
worksheet.write_column(1,18,total_units)

worksheet.write(7,0,'S1')
worksheet.write_row(7,1,model_names)
worksheet.write_column(8,0,type_names)
percs = data_dict_all['S1_dicts']['model_summary']['all_percs_r2']
total_units = data_dict_all['S1_dicts']['model_summary']['total_num_units_window']
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+8,1,percs[i,:])
worksheet.write(7,8,'total units')
worksheet.write_column(8,8,total_units)
percs = data_dict_all['S1_dicts']['model_summary']['sig_percs_r2']
total_units = data_dict_all['S1_dicts']['model_summary']['num_any_sig']
worksheet.write_row(7,11,model_names)
worksheet.write_column(8,10,type_names)
worksheet.write(8,10,'S1: sig')
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+8,11,percs[i,:])
worksheet.write(7,18,'total units')
worksheet.write_column(8,18,total_units)

worksheet.write(14,0,'PMd')
worksheet.write_row(14,1,model_names)
worksheet.write_column(15,0,type_names)
percs = data_dict_all['PmD_dicts']['model_summary']['all_percs_r2']
total_units = data_dict_all['S1_dicts']['model_summary']['total_num_units_window']
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+15,1,percs[i,:])
worksheet.write(14,8,'total units')
worksheet.write_column(15,8,total_units)
percs = data_dict_all['PmD_dicts']['model_summary']['sig_percs_r2']
total_units = data_dict_all['PmD_dicts']['model_summary']['num_any_sig']
worksheet.write_row(14,11,model_names)
worksheet.write_column(15,10,type_names)
worksheet.write(14,10,'PmD: sig')
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+15,11,percs[i,:])
worksheet.write(14,18,'total units')
worksheet.write_column(15,18,total_units)

worksheet.conditional_format('B2:H2', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B3:H3', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B4:H4', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B5:H5', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B6:H6', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L2:R2', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L3:R3', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L4:R4', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L5:R5', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L6:R6', {'type': 'top', 'value':'1', 'format':format1})

worksheet.conditional_format('B9:H9', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B10:H10', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B11:H11', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B12:H12', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B13:H13', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L9:R9', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L10:R10', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L11:R11', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L12:R12', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L13:R13', {'type': 'top', 'value':'1', 'format':format1})

worksheet.conditional_format('B16:H16', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B17:H17', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B18:H18', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B19:H19', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('B20:H20', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L16:R16', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L17:R17', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L18:R18', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L19:R19', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('L20:R20', {'type': 'top', 'value':'1', 'format':format1})



#################

worksheet = aic_workbook.add_worksheet('perc_no_fit')
worksheet.write(0,0,'M1')
worksheet.write_row(0,1,model_names)
worksheet.write_column(1,0,type_names)
no_fit = data_dict_all['M1_dicts']['model_summary']['all_no_fit']
total_units = data_dict_all['M1_dicts']['model_summary']['total_num_units_window']
for i in range(np.shape(no_fit)[0]):
    worksheet.write_row(i+1,1,no_fit[i,:])
worksheet.write(0,8,'total units')
worksheet.write_column(1,8,total_units)

worksheet.write(7,0,'S1')
worksheet.write_row(7,1,model_names)
worksheet.write_column(8,0,type_names)
no_fit = data_dict_all['S1_dicts']['model_summary']['all_no_fit']
total_units = data_dict_all['S1_dicts']['model_summary']['total_num_units_window']
for i in range(np.shape(no_fit)[0]):
    worksheet.write_row(i+8,1,no_fit[i,:])
worksheet.write(7,8,'total units')
worksheet.write_column(8,8,total_units)

worksheet.write(14,0,'PMd')
worksheet.write_row(14,1,model_names)
worksheet.write_column(15,0,type_names)
no_fit = data_dict_all['PmD_dicts']['model_summary']['all_no_fit']
total_units = data_dict_all['PmD_dicts']['model_summary']['total_num_units_window']
for i in range(np.shape(no_fit)[0]):
    worksheet.write_row(i+15,1,no_fit[i,:])
worksheet.write(14,8,'total units')
worksheet.write_column(15,8,total_units)

#################
worksheet = aic_workbook.add_worksheet('AIC_averaged_over_sig')

lin_aics = [data_dict_all['M1_dicts']['models']['linear']['aft_cue']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['linear']['bfr_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['linear']['aft_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['linear']['res_win']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['linear']['concat']['AIC_sig_avg']]
diff_aics = [data_dict_all['M1_dicts']['models']['diff']['aft_cue']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['diff']['bfr_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['diff']['aft_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['diff']['res_win']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['diff']['concat']['AIC_sig_avg']]
div_nl_aics = [data_dict_all['M1_dicts']['models']['div_nl']['aft_cue']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl']['bfr_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl']['aft_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl']['res_win']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl']['concat']['AIC_sig_avg']]
div_nl_noe_aics = [data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_cue']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_noe']['bfr_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_noe']['res_win']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_noe']['concat']['AIC_sig_avg']]
div_nl_Y_aics = [data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_cue']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_Y']['bfr_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_Y']['res_win']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_Y']['concat']['AIC_sig_avg']]
div_nl_separate_add_aics = [data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_cue']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['bfr_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['res_win']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['concat']['AIC_sig_avg']]
div_nl_separate_multiply_aics = [data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_res']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['res_win']['AIC_sig_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['concat']['AIC_sig_avg']]

lin_aics_comb = [data_dict_all['M1_dicts']['models']['linear']['aft_cue']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['linear']['bfr_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['linear']['aft_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['linear']['res_win']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['linear']['concat']['AIC_sig_model']]
diff_aics_comb = [data_dict_all['M1_dicts']['models']['diff']['aft_cue']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['diff']['bfr_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['diff']['aft_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['diff']['res_win']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['diff']['concat']['AIC_sig_model']]
div_nl_aics_comb = [data_dict_all['M1_dicts']['models']['div_nl']['aft_cue']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl']['bfr_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl']['aft_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl']['res_win']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl']['concat']['AIC_sig_model']]
div_nl_noe_aics_comb = [data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_cue']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_noe']['bfr_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_noe']['res_win']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_noe']['concat']['AIC_sig_model']]
div_nl_Y_aics_comb = [data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_cue']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_Y']['bfr_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_Y']['res_win']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_Y']['concat']['AIC_sig_model']]
div_nl_separate_add_aics_comb = [data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_cue']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['bfr_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['res_win']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['concat']['AIC_sig_model']]
div_nl_separate_multiply_aics_comb = [data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_res']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['res_win']['AIC_sig_model'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['concat']['AIC_sig_model']]

lin_num_sig = [data_dict_all['M1_dicts']['models']['linear']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['linear']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['linear']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['linear']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['linear']['concat']['num_sig_fit']]
diff_num_sig = [data_dict_all['M1_dicts']['models']['diff']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['diff']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['diff']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['diff']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['diff']['concat']['num_sig_fit']]
div_nl_num_sig = [data_dict_all['M1_dicts']['models']['div_nl']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl']['concat']['num_sig_fit']]
div_nl_noe_num_sig = [data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_noe']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_noe']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_noe']['concat']['num_sig_fit']]
div_nl_Y_num_sig = [data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_Y']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_Y']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_Y']['concat']['num_sig_fit']]
div_nl_separate_add_num_sig = [data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['concat']['num_sig_fit']]
div_nl_separate_multiply_num_sig = [data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['concat']['num_sig_fit']]

worksheet.write(0,0,'M1')
worksheet.write(1,0,'AIC avg')
worksheet.write_row(1,1,type_names)
worksheet.write_row(2,1,lin_aics)
worksheet.write_row(3,1,diff_aics)
worksheet.write_row(4,1,div_nl_aics)
worksheet.write_row(5,1,div_nl_noe_aics)
worksheet.write_row(6,1,div_nl_Y_aics)
worksheet.write_row(7,1,div_nl_separate_add_aics)
worksheet.write_row(8,1,div_nl_separate_multiply_aics)
worksheet.write_column(2,0,model_names)

worksheet.write(1,7,'AIC combined')
worksheet.write_row(1,8,type_names)
worksheet.write_row(2,8,lin_aics_comb)
worksheet.write_row(3,8,diff_aics_comb)
worksheet.write_row(4,8,div_nl_aics_comb)
worksheet.write_row(5,8,div_nl_noe_aics_comb)
worksheet.write_row(6,8,div_nl_Y_aics_comb)
worksheet.write_row(7,8,div_nl_separate_add_aics_comb)
worksheet.write_row(8,8,div_nl_separate_multiply_aics_comb)
worksheet.write_column(2,7,model_names)

worksheet.write(1,14,'num sig')
worksheet.write_row(1,15,type_names)
worksheet.write_row(2,15,lin_num_sig)
worksheet.write_row(3,15,diff_num_sig)
worksheet.write_row(4,15,div_nl_num_sig)
worksheet.write_row(5,15,div_nl_noe_num_sig)
worksheet.write_row(6,15,div_nl_Y_num_sig)
worksheet.write_row(7,15,div_nl_separate_add_num_sig)
worksheet.write_row(8,15,div_nl_separate_multiply_num_sig)
worksheet.write_column(2,14,model_names)
worksheet.write(9,14,'total units')
worksheet.write(9,15,data_dict_all['M1_dicts']['avg_and_corr']['aft_cue']['total_unit_num'])

worksheet.conditional_format('B3:B9', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('C3:C9', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('D3:D9', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('E3:E9', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('F3:F9', {'type': 'bottom', 'value':'1', 'format':format1})

worksheet.conditional_format('I3:I9', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('J3:J9', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('K3:K9', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('L3:L9', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('M3:M9', {'type': 'bottom', 'value':'1', 'format':format1})

lin_aics = [data_dict_all['S1_dicts']['models']['linear']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['linear']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['linear']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['linear']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['linear']['concat']['AIC_sig_avg']]
diff_aics = [data_dict_all['S1_dicts']['models']['diff']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['diff']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['diff']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['diff']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['diff']['concat']['AIC_sig_avg']]
div_nl_aics = [data_dict_all['S1_dicts']['models']['div_nl']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl']['concat']['AIC_sig_avg']]
div_nl_noe_aics = [data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['concat']['AIC_sig_avg']]
div_nl_Y_aics = [data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['concat']['AIC_sig_avg']]
div_nl_separate_add_aics = [data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['concat']['AIC_sig_avg']]
div_nl_separate_multiply_aics = [data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['concat']['AIC_sig_avg']]

lin_aics_comb = [data_dict_all['S1_dicts']['models']['linear']['aft_cue']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['linear']['bfr_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['linear']['aft_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['linear']['res_win']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['linear']['concat']['AIC_sig_model']]
diff_aics_comb = [data_dict_all['S1_dicts']['models']['diff']['aft_cue']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['diff']['bfr_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['diff']['aft_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['diff']['res_win']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['diff']['concat']['AIC_sig_model']]
div_nl_aics_comb = [data_dict_all['S1_dicts']['models']['div_nl']['aft_cue']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl']['bfr_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl']['aft_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl']['res_win']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl']['concat']['AIC_sig_model']]
div_nl_noe_aics_comb = [data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_cue']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_noe']['bfr_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_noe']['res_win']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_noe']['concat']['AIC_sig_model']]
div_nl_Y_aics_comb = [data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_cue']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_Y']['bfr_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_Y']['res_win']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_Y']['concat']['AIC_sig_model']]
div_nl_separate_add_aics_comb = [data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_cue']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['bfr_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['res_win']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['concat']['AIC_sig_model']]
div_nl_separate_multiply_aics_comb = [data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_res']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['res_win']['AIC_sig_model'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['concat']['AIC_sig_model']]


lin_num_sig = [data_dict_all['S1_dicts']['models']['linear']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['linear']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['linear']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['linear']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['linear']['concat']['num_sig_fit']]
diff_num_sig = [data_dict_all['S1_dicts']['models']['diff']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['diff']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['diff']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['diff']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['diff']['concat']['num_sig_fit']]
div_nl_num_sig = [data_dict_all['S1_dicts']['models']['div_nl']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl']['concat']['num_sig_fit']]
div_nl_noe_num_sig = [data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_noe']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_noe']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_noe']['concat']['num_sig_fit']]
div_nl_Y_num_sig = [data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_Y']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_Y']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_Y']['concat']['num_sig_fit']]
div_nl_separate_add_num_sig = [data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['concat']['num_sig_fit']]
div_nl_separate_multiply_num_sig = [data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['concat']['num_sig_fit']]

worksheet.write(11,0,'S1')
worksheet.write(12,1,'AIC avg')
worksheet.write_row(12,1,type_names)
worksheet.write_row(13,1,lin_aics)
worksheet.write_row(14,1,diff_aics)
worksheet.write_row(15,1,div_nl_aics)
worksheet.write_row(16,1,div_nl_noe_aics)
worksheet.write_row(17,1,div_nl_Y_aics)
worksheet.write_row(18,1,div_nl_separate_add_aics)
worksheet.write_row(19,1,div_nl_separate_multiply_aics)
worksheet.write_column(13,0,model_names)

worksheet.write(12,0,'AIC combined')
worksheet.write_row(12,8,type_names)
worksheet.write_row(13,8,lin_aics_comb)
worksheet.write_row(14,8,diff_aics_comb)
worksheet.write_row(15,8,div_nl_aics_comb)
worksheet.write_row(16,8,div_nl_noe_aics_comb)
worksheet.write_row(17,8,div_nl_Y_aics_comb)
worksheet.write_row(18,8,div_nl_separate_add_aics_comb)
worksheet.write_row(19,8,div_nl_separate_multiply_aics_comb)
worksheet.write_column(13,7,model_names)

worksheet.write(12,14,'num sig')
worksheet.write_row(12,15,type_names)
worksheet.write_row(13,15,lin_num_sig)
worksheet.write_row(14,15,diff_num_sig)
worksheet.write_row(15,15,div_nl_num_sig)
worksheet.write_row(16,15,div_nl_noe_num_sig)
worksheet.write_row(17,15,div_nl_Y_num_sig)
worksheet.write_row(18,15,div_nl_separate_add_num_sig)
worksheet.write_row(19,15,div_nl_separate_multiply_num_sig)
worksheet.write_column(13,14,model_names)
worksheet.write(20,14,'total units')
worksheet.write(20,15,data_dict_all['S1_dicts']['avg_and_corr']['aft_cue']['total_unit_num'])

worksheet.conditional_format('B14:B20', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('C14:C20', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('D14:D20', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('E14:E20', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('F14:F20', {'type': 'bottom', 'value':'1', 'format':format1})

worksheet.conditional_format('I14:I20', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('J14:J20', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('K14:K20', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('L14:L20', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('M14:M20', {'type': 'bottom', 'value':'1', 'format':format1})

lin_aics = [data_dict_all['PmD_dicts']['models']['linear']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['linear']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['linear']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['linear']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['linear']['concat']['AIC_sig_avg']]
diff_aics = [data_dict_all['PmD_dicts']['models']['diff']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['diff']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['diff']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['diff']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['diff']['concat']['AIC_sig_avg']]
div_nl_aics = [data_dict_all['PmD_dicts']['models']['div_nl']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['concat']['AIC_sig_avg']]
div_nl_noe_aics = [data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['concat']['AIC_sig_avg']]
div_nl_Y_aics = [data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['concat']['AIC_sig_avg']]
div_nl_separate_add_aics = [data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['concat']['AIC_sig_avg']]
div_nl_separate_multiply_aics = [data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['concat']['AIC_sig_avg']]

lin_aics_comb = [data_dict_all['PmD_dicts']['models']['linear']['aft_cue']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['linear']['bfr_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['linear']['aft_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['linear']['res_win']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['linear']['concat']['AIC_sig_model']]
diff_aics_comb = [data_dict_all['PmD_dicts']['models']['diff']['aft_cue']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['diff']['bfr_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['diff']['aft_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['diff']['res_win']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['diff']['concat']['AIC_sig_model']]
div_nl_aics_comb = [data_dict_all['PmD_dicts']['models']['div_nl']['aft_cue']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl']['bfr_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl']['aft_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl']['res_win']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl']['concat']['AIC_sig_model']]
div_nl_noe_aics_comb = [data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_cue']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['bfr_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['res_win']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['concat']['AIC_sig_model']]
div_nl_Y_aics_comb = [data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_cue']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['bfr_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['res_win']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['concat']['AIC_sig_model']]
div_nl_separate_add_aics_comb = [data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_cue']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['bfr_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['res_win']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['concat']['AIC_sig_model']]
div_nl_separate_multiply_aics_comb = [data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_cue']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['bfr_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_res']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['res_win']['AIC_sig_model'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['concat']['AIC_sig_model']]

lin_num_sig = [data_dict_all['PmD_dicts']['models']['linear']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['linear']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['linear']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['linear']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['linear']['concat']['num_sig_fit']]
diff_num_sig = [data_dict_all['PmD_dicts']['models']['diff']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['diff']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['diff']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['diff']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['diff']['concat']['num_sig_fit']]
div_nl_num_sig = [data_dict_all['PmD_dicts']['models']['div_nl']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl']['concat']['num_sig_fit']]
div_nl_noe_num_sig = [data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['concat']['num_sig_fit']]
div_nl_Y_num_sig = [data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['concat']['num_sig_fit']]
div_nl_separate_add_num_sig = [data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['concat']['num_sig_fit']]
div_nl_separate_multiply_num_sig = [data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['concat']['num_sig_fit']]


worksheet.write(23,0,'PMd')
worksheet.write(24,0,'AIC avg')
worksheet.write_row(24,1,type_names)
worksheet.write_row(25,1,lin_aics)
worksheet.write_row(26,1,diff_aics)
worksheet.write_row(27,1,div_nl_aics)
worksheet.write_row(28,1,div_nl_noe_aics)
worksheet.write_row(29,1,div_nl_Y_aics)
worksheet.write_row(30,1,div_nl_separate_add_aics)
worksheet.write_row(31,1,div_nl_separate_multiply_aics)
worksheet.write_column(25,0,model_names)

worksheet.write(24,7,'AIC combined')
worksheet.write_row(24,8,type_names)
worksheet.write_row(25,8,lin_aics_comb)
worksheet.write_row(26,8,diff_aics_comb)
worksheet.write_row(27,8,div_nl_aics_comb)
worksheet.write_row(28,8,div_nl_noe_aics_comb)
worksheet.write_row(29,8,div_nl_Y_aics_comb)
worksheet.write_row(30,8,div_nl_separate_add_aics_comb)
worksheet.write_row(31,8,div_nl_separate_multiply_aics_comb)
worksheet.write_column(25,7,model_names)

worksheet.write(24,14,'num sig')
worksheet.write_row(24,15,type_names)
worksheet.write_row(25,15,lin_num_sig)
worksheet.write_row(26,15,diff_num_sig)
worksheet.write_row(27,15,div_nl_num_sig)
worksheet.write_row(28,15,div_nl_noe_num_sig)
worksheet.write_row(29,15,div_nl_Y_num_sig)
worksheet.write_row(30,15,div_nl_separate_add_num_sig)
worksheet.write_row(31,15,div_nl_separate_multiply_num_sig)
worksheet.write_column(25,14,model_names)
worksheet.write(32,14,'total units')
worksheet.write(32,15,data_dict_all['PmD_dicts']['avg_and_corr']['aft_cue']['total_unit_num'])

worksheet.conditional_format('B26:B32', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('C26:C32', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('D26:D32', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('E26:E32', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('F26:F32', {'type': 'bottom', 'value':'1', 'format':format1})

worksheet.conditional_format('I26:I32', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('J26:J32', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('K26:K32', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('L26:L32', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('M26:M32', {'type': 'bottom', 'value':'1', 'format':format1})


###################
#mean r2 vals

worksheet = aic_workbook.add_worksheet('mean_r2')
lin_r_sq = [data_dict_all['M1_dicts']['models']['linear']['aft_cue']['r_sq_avg'],data_dict_all['M1_dicts']['models']['linear']['bfr_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['linear']['aft_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['linear']['res_win']['r_sq_avg'],data_dict_all['M1_dicts']['models']['linear']['concat']['r_sq_avg']]
diff_r_sq = [data_dict_all['M1_dicts']['models']['diff']['aft_cue']['r_sq_avg'],data_dict_all['M1_dicts']['models']['diff']['bfr_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['diff']['aft_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['diff']['res_win']['r_sq_avg'],data_dict_all['M1_dicts']['models']['diff']['concat']['r_sq_avg']]
div_nl_r_sq = [data_dict_all['M1_dicts']['models']['div_nl']['aft_cue']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl']['bfr_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl']['aft_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl']['res_win']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl']['concat']['r_sq_avg']]
div_nl_noe_r_sq = [data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_cue']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_noe']['bfr_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_noe']['res_win']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_noe']['concat']['r_sq_avg']]
div_nl_Y_r_sq = [data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_cue']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_Y']['bfr_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_Y']['res_win']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_Y']['concat']['r_sq_avg']]
div_nl_separate_add_r_sq = [data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_cue']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['bfr_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['res_win']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['concat']['r_sq_avg']]
div_nl_separate_multiply_r_sq = [data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_res']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['res_win']['r_sq_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['concat']['r_sq_avg']]

worksheet.write(0,0,'M1: avg r2')
worksheet.write_row(0,1,type_names)
worksheet.write_row(1,1,lin_r_sq)
worksheet.write_row(2,1,diff_r_sq)
worksheet.write_row(3,1,div_nl_r_sq)
worksheet.write_row(4,1,div_nl_noe_r_sq)
worksheet.write_row(5,1,div_nl_Y_r_sq)
worksheet.write_row(6,1,div_nl_separate_add_r_sq)
worksheet.write_row(7,1,div_nl_separate_multiply_r_sq)
worksheet.write_column(1,0,model_names)

worksheet.conditional_format('B2:B8', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('C2:C8', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('D2:D8', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('E2:E8', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('F2:F8', {'type': 'top', 'value':'1', 'format':format1})

lin_r_sq = [data_dict_all['S1_dicts']['models']['linear']['aft_cue']['r_sq_avg'],data_dict_all['S1_dicts']['models']['linear']['bfr_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['linear']['aft_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['linear']['res_win']['r_sq_avg'],data_dict_all['S1_dicts']['models']['linear']['concat']['r_sq_avg']]
diff_r_sq = [data_dict_all['S1_dicts']['models']['diff']['aft_cue']['r_sq_avg'],data_dict_all['S1_dicts']['models']['diff']['bfr_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['diff']['aft_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['diff']['res_win']['r_sq_avg'],data_dict_all['S1_dicts']['models']['diff']['concat']['r_sq_avg']]
div_nl_r_sq = [data_dict_all['S1_dicts']['models']['div_nl']['aft_cue']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl']['bfr_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl']['aft_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl']['res_win']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl']['concat']['r_sq_avg']]
div_nl_noe_r_sq = [data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_cue']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['bfr_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['res_win']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['concat']['r_sq_avg']]
div_nl_Y_r_sq = [data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_cue']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['bfr_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['res_win']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['concat']['r_sq_avg']]
div_nl_separate_add_r_sq = [data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_cue']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['bfr_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['res_win']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['concat']['r_sq_avg']]
div_nl_separate_multiply_r_sq = [data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_res']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['res_win']['r_sq_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['concat']['r_sq_avg']]

worksheet.write(12,0,'S1: avg r2')
worksheet.write_row(12,1,type_names)
worksheet.write_row(13,1,lin_r_sq)
worksheet.write_row(14,1,diff_r_sq)
worksheet.write_row(15,1,div_nl_r_sq)
worksheet.write_row(16,1,div_nl_noe_r_sq)
worksheet.write_row(17,1,div_nl_Y_r_sq)
worksheet.write_row(18,1,div_nl_separate_add_r_sq)
worksheet.write_row(19,1,div_nl_separate_multiply_r_sq)
worksheet.write_column(13,0,model_names)

worksheet.conditional_format('B14:B20', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('C14:C20', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('D14:D20', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('E14:E20', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('F14:F20', {'type': 'top', 'value':'1', 'format':format1})

lin_r_sq = [data_dict_all['PmD_dicts']['models']['linear']['aft_cue']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['linear']['bfr_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['linear']['aft_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['linear']['res_win']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['linear']['concat']['r_sq_avg']]
diff_r_sq = [data_dict_all['PmD_dicts']['models']['diff']['aft_cue']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['diff']['bfr_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['diff']['aft_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['diff']['res_win']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['diff']['concat']['r_sq_avg']]
div_nl_r_sq = [data_dict_all['PmD_dicts']['models']['div_nl']['aft_cue']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['bfr_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['aft_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['res_win']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['concat']['r_sq_avg']]
div_nl_noe_r_sq = [data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_cue']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['bfr_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['res_win']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['concat']['r_sq_avg']]
div_nl_Y_r_sq = [data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_cue']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['bfr_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['res_win']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['concat']['r_sq_avg']]
div_nl_separate_add_r_sq = [data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_cue']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['bfr_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['res_win']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['concat']['r_sq_avg']]
div_nl_separate_multiply_r_sq = [data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_cue']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['bfr_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_res']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['res_win']['r_sq_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['concat']['r_sq_avg']]

worksheet.write(24,0,'PMd: Avg r2')
worksheet.write_row(24,1,type_names)
worksheet.write_row(25,1,lin_r_sq)
worksheet.write_row(26,1,diff_r_sq)
worksheet.write_row(27,1,div_nl_r_sq)
worksheet.write_row(28,1,div_nl_noe_r_sq)
worksheet.write_row(29,1,div_nl_Y_r_sq)
worksheet.write_row(30,1,div_nl_separate_add_r_sq)
worksheet.write_row(31,1,div_nl_separate_multiply_r_sq)
worksheet.write_column(25,0,model_names)

worksheet.conditional_format('B26:B32', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('C26:C32', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('D26:D32', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('E26:E32', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('F26:F32', {'type': 'top', 'value':'1', 'format':format1})


####
lin_r_sq = [data_dict_all['M1_dicts']['models']['linear']['aft_cue']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['linear']['bfr_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['linear']['aft_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['linear']['res_win']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['linear']['concat']['r_sq_adj_avg']]
diff_r_sq = [data_dict_all['M1_dicts']['models']['diff']['aft_cue']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['diff']['bfr_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['diff']['aft_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['diff']['res_win']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['diff']['concat']['r_sq_adj_avg']]
div_nl_r_sq = [data_dict_all['M1_dicts']['models']['div_nl']['aft_cue']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl']['bfr_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl']['aft_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl']['res_win']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl']['concat']['r_sq_adj_avg']]
div_nl_noe_r_sq = [data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_cue']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_noe']['bfr_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_noe']['res_win']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_noe']['concat']['r_sq_adj_avg']]
div_nl_Y_r_sq = [data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_cue']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_Y']['bfr_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_Y']['res_win']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_Y']['concat']['r_sq_adj_avg']]
div_nl_separate_add_r_sq = [data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_cue']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['bfr_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['res_win']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['concat']['r_sq_adj_avg']]
div_nl_separate_multiply_r_sq = [data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_res']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['res_win']['r_sq_adj_avg'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['concat']['r_sq_adj_avg']]

worksheet.write(0,0,'M1: avg adj r2')
worksheet.write_row(0,1,type_names)
worksheet.write_row(1,1,lin_r_sq)
worksheet.write_row(2,1,diff_r_sq)
worksheet.write_row(3,1,div_nl_r_sq)
worksheet.write_row(4,1,div_nl_noe_r_sq)
worksheet.write_row(5,1,div_nl_Y_r_sq)
worksheet.write_row(6,1,div_nl_separate_add_r_sq)
worksheet.write_row(7,1,div_nl_separate_multiply_r_sq)
worksheet.write_column(1,0,model_names)

worksheet.conditional_format('B2:B8', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('C2:C8', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('D2:D8', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('E2:E8', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('F2:F8', {'type': 'top', 'value':'1', 'format':format1})

lin_r_sq = [data_dict_all['S1_dicts']['models']['linear']['aft_cue']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['linear']['bfr_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['linear']['aft_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['linear']['res_win']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['linear']['concat']['r_sq_adj_avg']]
diff_r_sq = [data_dict_all['S1_dicts']['models']['diff']['aft_cue']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['diff']['bfr_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['diff']['aft_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['diff']['res_win']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['diff']['concat']['r_sq_adj_avg']]
div_nl_r_sq = [data_dict_all['S1_dicts']['models']['div_nl']['aft_cue']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl']['bfr_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl']['aft_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl']['res_win']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl']['concat']['r_sq_adj_avg']]
div_nl_noe_r_sq = [data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_cue']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['bfr_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['res_win']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['concat']['r_sq_adj_avg']]
div_nl_Y_r_sq = [data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_cue']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['bfr_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['res_win']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['concat']['r_sq_adj_avg']]
div_nl_separate_add_r_sq = [data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_cue']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['bfr_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['res_win']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['concat']['r_sq_adj_avg']]
div_nl_separate_multiply_r_sq = [data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_res']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['res_win']['r_sq_adj_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['concat']['r_sq_adj_avg']]

worksheet.write(12,0,'S1: avg adj r2')
worksheet.write_row(12,1,type_names)
worksheet.write_row(13,1,lin_r_sq)
worksheet.write_row(14,1,diff_r_sq)
worksheet.write_row(15,1,div_nl_r_sq)
worksheet.write_row(16,1,div_nl_noe_r_sq)
worksheet.write_row(17,1,div_nl_Y_r_sq)
worksheet.write_row(18,1,div_nl_separate_add_r_sq)
worksheet.write_row(19,1,div_nl_separate_multiply_r_sq)
worksheet.write_column(13,0,model_names)

worksheet.conditional_format('B14:B20', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('C14:C20', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('D14:D20', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('E14:E20', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('F14:F20', {'type': 'top', 'value':'1', 'format':format1})

lin_r_sq = [data_dict_all['PmD_dicts']['models']['linear']['aft_cue']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['linear']['bfr_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['linear']['aft_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['linear']['res_win']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['linear']['concat']['r_sq_adj_avg']]
diff_r_sq = [data_dict_all['PmD_dicts']['models']['diff']['aft_cue']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['diff']['bfr_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['diff']['aft_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['diff']['res_win']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['diff']['concat']['r_sq_adj_avg']]
div_nl_r_sq = [data_dict_all['PmD_dicts']['models']['div_nl']['aft_cue']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['bfr_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['aft_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['res_win']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['concat']['r_sq_adj_avg']]
div_nl_noe_r_sq = [data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_cue']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['bfr_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['res_win']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['concat']['r_sq_adj_avg']]
div_nl_Y_r_sq = [data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_cue']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['bfr_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['res_win']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['concat']['r_sq_adj_avg']]
div_nl_separate_add_r_sq = [data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_cue']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['bfr_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['res_win']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['concat']['r_sq_adj_avg']]
div_nl_separate_multiply_r_sq = [data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_cue']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['bfr_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_res']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['res_win']['r_sq_adj_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['concat']['r_sq_adj_avg']]

worksheet.write(24,0,'PMd: Avg adj r2')
worksheet.write_row(24,1,type_names)
worksheet.write_row(25,1,lin_r_sq)
worksheet.write_row(26,1,diff_r_sq)
worksheet.write_row(27,1,div_nl_r_sq)
worksheet.write_row(28,1,div_nl_noe_r_sq)
worksheet.write_row(29,1,div_nl_Y_r_sq)
worksheet.write_row(30,1,div_nl_separate_add_r_sq)
worksheet.write_row(31,1,div_nl_separate_multiply_r_sq)
worksheet.write_column(25,0,model_names)

worksheet.conditional_format('B26:B32', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('C26:C32', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('D26:D32', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('E26:E32', {'type': 'top', 'value':'1', 'format':format1})
worksheet.conditional_format('F26:F32', {'type': 'top', 'value':'1', 'format':format1})

aic_workbook.close()

###########

plt.close('all')


