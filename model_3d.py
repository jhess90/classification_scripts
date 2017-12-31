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
time_before = -0.5 #negative value
time_after = 1.0
baseline_time = -1.0 #negative value
normalize_bool = False
sqrt_bool = False
plot_3d_bool = False
mv_bool = True
zscore = False
abs_alphabeta = False

gaussian_bool = True
gauss_sigma = 30

ts_filename = glob.glob('Extracted*_timestamps.mat')[0]
extracted_filename = ts_filename[:-15] + '.mat'

###############################
### functions #################
###############################

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

        if zscore and gaussian_bool:
            hists = stats.zscore(hists,axis=1)
            hists = ndimage.filters.gaussian_filter1d(hists,gauss_sigma,axis=1)
        elif zscore:
            hists = stats.zscore(hists,axis=1)
        elif gaussian_bool:
            hists = ndimage.filters.gaussian_filter1d(hists,gauss_sigma,axis=1)

        #pdb.set_trace()
            
        baseline_fr = np.zeros((len(condensed),np.shape(hists)[0],-1*baseline_bins))
        bfr_cue_fr = np.zeros((len(condensed),np.shape(hists)[0],-1*bins_before))
        aft_cue_fr = np.zeros((len(condensed),np.shape(hists)[0],bins_after))
        bfr_result_fr = np.zeros((len(condensed),np.shape(hists)[0],-1*bins_before))
        aft_result_fr = np.zeros((len(condensed),np.shape(hists)[0],bins_after))
        
        bfr_cue_hist_all = np.zeros((len(condensed),np.shape(hists)[0],-1*bins_before))
        aft_cue_hist_all = np.zeros((len(condensed),np.shape(hists)[0],bins_after))
        bfr_result_hist_all = np.zeros((len(condensed),np.shape(hists)[0],-1*bins_before))
        aft_result_hist_all = np.zeros((len(condensed),np.shape(hists)[0],bins_after))
        
        

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
                
                #pdb.set_trace()
                if not (result_start_bin + bins_after) > np.shape(hists)[1]:
                        for j in range(np.shape(hists)[0]):
                                baseline_hist = hists[j,baseline_bins + cue_start_bin : cue_start_bin]
                                bfr_cue_hist = hists[j,bins_before + cue_start_bin : cue_start_bin]
                                aft_cue_hist = hists[j,cue_start_bin : cue_start_bin + bins_after]
                                bfr_result_hist = hists[j,bins_before + result_start_bin : result_start_bin]
                                aft_result_hist = hists[j,result_start_bin : result_start_bin + bins_after]
   
                                bfr_cue_hist_all[i,j,:] = bfr_cue_hist
                                aft_cue_hist_all[i,j,:] = aft_cue_hist
                                bfr_result_hist_all[i,j,:] = bfr_result_hist
                                aft_result_hist_all[i,j,:] = aft_result_hist
                     
                                if not zscore:
                                        baseline_fr[i,j,:] = baseline_hist / float(bin_size) * 1000
                                        bfr_cue_fr[i,j,:] = bfr_cue_hist / float(bin_size) * 1000
                                        aft_cue_fr[i,j,:] = aft_cue_hist / float(bin_size) * 1000
                                        bfr_result_fr[i,j,:] = bfr_result_hist / float(bin_size) * 1000
                                        aft_result_fr[i,j,:] = aft_result_hist / float(bin_size) * 1000                        
                                elif zscore:
                                        baseline_fr[i,j,:] = baseline_hist
                                        bfr_cue_fr[i,j,:] = bfr_cue_hist
                                        aft_cue_fr[i,j,:] = aft_cue_hist
                                        bfr_result_fr[i,j,:] = bfr_result_hist
                                        aft_result_fr[i,j,:] = aft_result_hist

                else:
                        continue
                        
        #normalize frs
        bfr_cue_nl_fr = np.zeros((np.shape(bfr_cue_fr)))
        aft_cue_nl_fr = np.zeros((np.shape(aft_cue_fr)))
        bfr_result_nl_fr = np.zeros((np.shape(bfr_result_fr)))
        aft_result_nl_fr = np.zeros((np.shape(aft_result_fr)))

        for i in range(bfr_cue_fr.shape[0]):
                for j in range(bfr_cue_fr.shape[1]):
                        baseline_max = np.max(baseline_fr[i,j,:])
                        baseline_min = np.min(baseline_fr[i,j,:])

                        denom = float(baseline_max - baseline_min)

                        #TODO better way to do this?
                        if denom == 0:
                                bfr_cue_nl_fr[i,j,:] = bfr_cue_fr[i,j,:]
                                aft_cue_nl_fr[i,j,:] = aft_cue_fr[i,j,:]
                                bfr_result_nl_fr[i,j,:] = bfr_result_fr[i,j,:]
                                aft_result_nl_fr[i,j,:] = aft_result_fr[i,j,:]

                        else:
                                num = np.subtract(bfr_cue_fr[i,j,:],baseline_min)
                                bfr_cue_nl_fr[i,j,:] = np.true_divide(num,denom)
                                num = np.subtract(aft_cue_fr[i,j,:],baseline_min)
                                aft_cue_nl_fr[i,j,:] = np.true_divide(num,denom)
                                num = np.subtract(bfr_result_fr[i,j,:],baseline_min)
                                bfr_result_nl_fr[i,j,:] = np.true_divide(num,denom)
                                num = np.subtract(aft_result_fr[i,j,:],baseline_min)
                                aft_result_nl_fr[i,j,:] = np.true_divide(num,denom)
                        
        return_dict = {'bfr_cue_fr':bfr_cue_fr,'aft_cue_fr':aft_cue_fr,'bfr_result_fr':bfr_result_fr,'aft_result_fr':aft_result_fr,'bfr_cue_nl_fr':bfr_cue_nl_fr,'aft_cue_nl_fr':aft_cue_nl_fr,'bfr_result_nl_fr':bfr_result_nl_fr,'aft_result_nl_fr':aft_result_nl_fr,'baseline_fr':baseline_fr,'bfr_cue_hist':bfr_cue_hist_all,'aft_cue_hist':aft_cue_hist_all,'bfr_result_hist':bfr_result_hist_all,'aft_result_hist':aft_result_hist_all}
        return(return_dict)

def make_3d_model(fr_data,condensed,region_key,type_key):
        
        avg_fr_data = np.mean(fr_data,axis=2)

        return_dict = {}
        conc_r_vals = []
        conc_p_vals = []
        conc_avg_frs = []
        sig_rsquared = []
        sig_fpvalue = []
        sig_pvals = []
        sig_const = []
        sig_alpha = []
        sig_beta = []

        for unit_num in range(fr_data.shape[1]):
                r_vals = condensed[:,3]
                p_vals = condensed[:,4]
                avg_frs = avg_fr_data[:,unit_num]
        
                conc_r_vals = np.append(conc_r_vals,r_vals)
                conc_p_vals = np.append(conc_p_vals,p_vals)
                conc_avg_frs = np.append(conc_avg_frs,avg_frs)

                df_dict = {'R':r_vals,'P':p_vals,'fr':avg_frs}
                df = pd.DataFrame(df_dict)

                X = df[['R','P']]
                y = df['fr']
                X = sm.add_constant(X) #adds k value
                
                #fitting to: avg firing rate = alpha * R + beta * P + k

                model = sm.OLS(y,X).fit()
                param_dict = model.params
                stat_result_dict = {'rsquared':model.rsquared,'rsquared_adj':model.rsquared_adj,'fvalue':model.fvalue,'f_pvalue':model.f_pvalue,'pvalues':model.pvalues,'conf_int':model.conf_int(),'std_err':model.bse}
                unit_dict = {'model':model,'param_dict':param_dict,'stat_result_dict':stat_result_dict}
                return_dict[unit_num] = unit_dict

                if model.rsquared >= 0.8:
                        sig_rsquared = np.append(sig_rsquared,unit_num)
                if model.f_pvalue <= 0.05:
                        sig_fpvalue = np.append(sig_fpvalue,unit_num)
                #if (model.pvalues < 0.05).any():
                if model.pvalues[1] <= 0.05 or model.pvalues[2] <= 0.05:
                        sig_pvals = np.append(sig_pvals,unit_num)
                if model.pvalues[0] <= 0.05:
                        sig_const = np.append(sig_const,unit_num)
                if model.pvalues[1] <= 0.05:
                        sig_alpha = np.append(sig_alpha,unit_num)  #alpha * R
                if model.pvalues[2] <= 0.05:
                        sig_beta = np.append(sig_beta,unit_num)   #beta * P

                if plot_3d_bool:
                        fig = plt.figure()
                        ax = fig.add_subplot(111,projection='3d')
                        ax.scatter(r_vals,p_vals,avg_frs,c='purple',marker='o')
                        
                        x_linspace = np.linspace(np.min(r_vals)-1,np.max(r_vals)+1)
                        y_linspace = np.linspace(np.min(p_vals)-1,np.max(p_vals)+1)
                        x,y = np.meshgrid(x_linspace,y_linspace)
                        z = param_dict['R'] * x + param_dict['P'] * y + param_dict['const']
                        surf = ax.plot_surface(x, y, z,cmap='Blues',alpha=0.4,linewidth=0)


                        plt.title('Region %s, %s, unit %s' %(region_key,type_key,unit_num))

                        ax.set_xlabel('R value')
                        ax.set_ylabel('P value')
                        ax.set_zlabel('avg firing rate')

                        #flipping z axis to left
                        tmp_planes = ax.zaxis._PLANES
                        ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3],
                                             tmp_planes[0], tmp_planes[1],
                                             tmp_planes[4], tmp_planes[5])
                        view_1 = (25,-135)
                        view_2 = (15,-55)
                        init_view = view_2
                        ax.view_init(*init_view)

                        ax.set_xlim(np.min(r_vals)-1,np.max(r_vals)+1)
                        ax.set_ylim(np.min(p_vals)-1,np.max(p_vals)+1)

                        plt.savefig('plot3dmodel_%s_%s_%s' %(region_key,type_key,str(unit_num).zfill(2)))
                        plt.clf()
        
        #pdb.set_trace()
        df_conc_dict = {'conc_R_vals':conc_r_vals,'conc_P_vals':conc_p_vals,'conc_avg_frs':conc_avg_frs}
        df_conc = pd.DataFrame(df_conc_dict)
        X = df_conc[['conc_R_vals','conc_P_vals']]
        y = df_conc['conc_avg_frs']
        X = sm.add_constant(X)
        conc_model = sm.OLS(y,X).fit()
        param_conc_dict = conc_model.params
        stat_conc_result_dict = {'rsquared':conc_model.rsquared,'rsquared_adj':conc_model.rsquared_adj,'fvalue':conc_model.fvalue,'f_pvalue':conc_model.f_pvalue,'pvalues':conc_model.pvalues,'conf_int':conc_model.conf_int(),'std_err':conc_model.bse}

        perc_sig_f_pvals = np.shape(sig_fpvalue)[0] / float(unit_num)
        perc_sig_rsquared = np.shape(sig_rsquared)[0] / float(unit_num)
        perc_sig_pvals = np.shape(sig_pvals)[0] / float(unit_num)
        perc_sig_const = np.shape(sig_const)[0] / float(unit_num)
        perc_sig_alpha = np.shape(sig_alpha)[0] / float(unit_num)
        perc_sig_beta = np.shape(sig_beta)[0] / float(unit_num)

        return_dict['conc'] = {'conc_model':conc_model,'param_conc_dict':param_conc_dict,'stat_conc_result_dict':stat_conc_result_dict,'sig_rsquared':sig_rsquared,'sig_fpvalue':sig_fpvalue,'sig_pvals':sig_pvals,'sig_const':sig_const,'sig_alpha':sig_alpha,'sig_beta':sig_beta,'perc_sic_f_pvals':perc_sig_f_pvals,'perc_sig_rsquared':perc_sig_rsquared,'perc_sig_pvals':perc_sig_pvals,'perc_sig_const':perc_sig_const,'perc_sig_alpha':perc_sig_alpha,'perc_sig_beta':perc_sig_beta,'df_conc_dict':df_conc_dict,'avg_frs':avg_frs,'avg_fr_data':avg_fr_data}
                
        return(return_dict)





###############################################
#start ########################################
###############################################

bins_before = int(time_before / float(bin_size) * 1000)  #neg for now
bins_after = int(time_after / float(bin_size) * 1000)   
baseline_bins = int(baseline_time / float(bin_size) * 1000) #neg

print 'bin size: %s' %(bin_size)
print 'time before: %s, time after: %s, baseline time: %s' %(time_before,time_after,baseline_time)
print 'nlize: %s, sqrt: %s, zscore: %s, gaussian: %s' %(normalize_bool,sqrt_bool,zscore,gaussian_bool)

#load files (from Extracted and timestamp files)
print extracted_filename
a = sio.loadmat(extracted_filename);
timestamps = sio.loadmat(ts_filename);

#create matrix of trial-by-trial info
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

#remove trials with now succ or failure scene (not sure why, but saw in one)

condensed = condensed[np.invert(np.logical_and(condensed[:,1] == 0, condensed[:,2] == 0))]


#TODOD FOR NOW remove catch trials
condensed = condensed[condensed[:,5] == 0]
#col 5 all 0s now, replace with succ/fail vector: succ = 1, fail = -1
condensed[condensed[:,1] != 0, 5] = 1
condensed[condensed[:,2] != 0, 5] = -1

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

#take care of some misc params, save param dict
no_bins = int((time_after - time_before) * 1000 / bin_size)
print 'bin size = %s' %(bin_size)
param_dict = {'time_before':time_before,'time_after':time_after,'bin_size':bin_size,'no_bins':no_bins}
#TODO add gripforce?
#gripforce_all = np.asarray(timestamps['gripforce'])
min_hist_len = 1000000 #set arbitrarily high number, because want to determin min length later
data_dict={'M1_spikes':M1_spikes,'S1_spikes':S1_spikes,'PmD_spikes':PmD_spikes,'PmV_spikes':PmV_spikes}

data_dict_hist_all = {}
print 'making hist'
for key, value in data_dict.iteritems():
	spike_data = []
	if key == 'PmV_spikes':
		#for i in range(len(value)):
		#	spike_data.append(value[i])
                continue
	else:
		for i in range(len(value)):
			spike_data.append(value[i]['ts'][0,0][0])

        hist_dict = make_hist_all(spike_data)
        data_dict_hist_all['%s_hist_dict' %(key)] = hist_dict
        
M1_dicts = {'spikes':data_dict['M1_spikes'],'hist_all':data_dict_hist_all['M1_spikes_hist_dict']['hist_data'],'bins_all':data_dict_hist_all['M1_spikes_hist_dict']['hist_bins']} 
S1_dicts = {'spikes':data_dict['S1_spikes'],'hist_all':data_dict_hist_all['S1_spikes_hist_dict']['hist_data'],'bins_all':data_dict_hist_all['S1_spikes_hist_dict']['hist_bins']}
PmD_dicts = {'spikes':data_dict['PmD_spikes'],'hist_all':data_dict_hist_all['PmD_spikes_hist_dict']['hist_data'],'bins_all':data_dict_hist_all['PmD_spikes_hist_dict']['hist_bins']}
#PmV_dicts = {'spikes':data_dict['PmV_spikes']} #,'nl':data_dict_nl['PmV_spikes_nl']}

data_dict_all = {'M1_dicts':M1_dicts,'S1_dicts':S1_dicts,'PmD_dicts':PmD_dicts}

print 'calc firing rate'
for region_key,region_value in data_dict_all.iteritems():
        hists = data_dict_all[region_key]['hist_all']
        fr_dict = calc_firing_rates(hists,region_key,condensed)
        data_dict_all[region_key]['fr_dict'] = fr_dict

        bfr_cue = fr_dict['bfr_cue_fr'][10,:,:]
        aft_cue = fr_dict['aft_cue_fr'][10,:,:]
        bfr_result = fr_dict['bfr_result_fr'][10,:,:]
        aft_result = fr_dict['aft_result_fr'][10,:,:]

        test_concat = np.append(bfr_cue,aft_cue,axis=1)
        test_concat = np.append(test_concat,bfr_result,axis=1)
        test_concat = np.append(test_concat,aft_result,axis=1)
        
        ax = plt.gca()
        plt.subplot(3,3,1)
        plt.plot(test_concat[0,:])
        plt.subplot(3,3,2)
        plt.plot(test_concat[1,:])
        plt.subplot(3,3,3)
        plt.plot(test_concat[2,:])
        plt.subplot(3,3,4)
        plt.plot(test_concat[3,:])
        plt.subplot(3,3,5)
        plt.plot(test_concat[4,:])
        plt.subplot(3,3,6)
        plt.plot(test_concat[5,:])
        plt.subplot(3,3,7)
        plt.plot(test_concat[6,:])
        plt.subplot(3,3,8)
        plt.plot(test_concat[7,:])
        plt.subplot(3,3,9)
        plt.plot(test_concat[8,:])

        plt.savefig('test_%s' %(region_key))
        plt.clf()


#print 'modeling'
for region_key,region_value in data_dict_all.iteritems():
        print 'modeling: %s' %(region_key)
        succ = condensed[condensed[:,5] == 1]
        fail = condensed[condensed[:,5] == -1]
        
        if normalize_bool:
                bfr_cue = data_dict_all[region_key]['fr_dict']['bfr_cue_nl_fr']
                aft_cue = data_dict_all[region_key]['fr_dict']['aft_cue_nl_fr']
                bfr_result = data_dict_all[region_key]['fr_dict']['bfr_result_nl_fr']
                aft_result = data_dict_all[region_key]['fr_dict']['aft_result_nl_fr']

        else:
                bfr_cue = data_dict_all[region_key]['fr_dict']['bfr_cue_fr']
                aft_cue = data_dict_all[region_key]['fr_dict']['aft_cue_fr']
                bfr_result = data_dict_all[region_key]['fr_dict']['bfr_result_fr']
                aft_result = data_dict_all[region_key]['fr_dict']['aft_result_fr']

        if sqrt_bool:
                bfr_cue = np.sqrt(bfr_cue)
                aft_cue = np.sqrt(aft_cue)
                bfr_result = np.sqrt(bfr_result)
                aft_result = np.sqrt(aft_result)

        bfr_cue_succ = bfr_cue[condensed[:,5] == 1]
        aft_cue_succ = aft_cue[condensed[:,5] == 1]
        bfr_result_succ = bfr_result[condensed[:,5] == 1]
        aft_result_succ = aft_result[condensed[:,5] == 1]
                
        bfr_cue_fail = bfr_cue[condensed[:,5] == -1]
        aft_cue_fail = aft_cue[condensed[:,5] == -1]
        bfr_result_fail = bfr_result[condensed[:,5] == -1]
        aft_result_fail = aft_result[condensed[:,5] == -1]
        

        bfr_cue_model = make_3d_model(bfr_cue,condensed,region_key,'bfr_cue')
        aft_cue_model = make_3d_model(aft_cue,condensed,region_key,'aft_cue')
        bfr_result_model = make_3d_model(bfr_result,condensed,region_key,'bfr_result')
        aft_result_model = make_3d_model(aft_result,condensed,region_key,'aft_result')


        #TODO succ vs fail
        model_return= {'bfr_cue_model':bfr_cue_model,'aft_cue_model':aft_cue_model,'bfr_result_model':bfr_result_model,'aft_result_model':aft_result_model}

        data_dict_all[region_key]['model_return'] = model_return



for region_key,region_value in data_dict_all.iteritems():
        data_dict_all[region_key]['slopes'] = {}
        data_dict_all[region_key]['sig_all_slopes'] = {}
        data_dict_all[region_key]['alpha_beta_only_sig'] = {}
        data_dict_all[region_key]['all_slopes'] = {}
        data_dict_all[region_key]['all_sig_all_slopes'] = {}
        data_dict_all[region_key]['all_alpha_beta_only_sig'] = {}
        for type_key,type_value in data_dict_all[region_key]['model_return'].iteritems():
                sig_rsquared = data_dict_all[region_key]['model_return'][type_key]['conc']['sig_rsquared']
                sig_fpvalue =  data_dict_all[region_key]['model_return'][type_key]['conc']['sig_fpvalue']
                sig_pvals =  data_dict_all[region_key]['model_return'][type_key]['conc']['sig_pvals']
                sig_const =  data_dict_all[region_key]['model_return'][type_key]['conc']['sig_const']
                sig_alpha =  data_dict_all[region_key]['model_return'][type_key]['conc']['sig_alpha']
                sig_beta =  data_dict_all[region_key]['model_return'][type_key]['conc']['sig_beta']

                alpha_only_sig = 0
                beta_only_sig = 0
                both_pos = 0
                both_neg = 0
                alpha_pos = 0
                beta_pos = 0

                if np.shape(sig_fpvalue)[0] > 0:
                        print 'Sig fpvalue- %s %s: %s' %(region_key,type_key,sig_fpvalue)
                if np.shape(sig_pvals)[0] > 0:
                        print 'Sig pvals- %s %s: %s' %(region_key,type_key,sig_pvals)

                        slopes = np.zeros((np.shape(sig_pvals)[0],7))
                        sig_all_slopes = np.zeros((np.shape(sig_pvals)[0],7))
                        sig_all_index = 0
                        
                        for i in range(np.shape(sig_pvals)[0]):
                                unit_num = sig_pvals[i]
                                slopes[i,0] = unit_num
                                #alpha
                                slopes[i,1] = data_dict_all[region_key]['model_return'][type_key][unit_num]['param_dict'][1]
                                #beta
                                slopes[i,2] = data_dict_all[region_key]['model_return'][type_key][unit_num]['param_dict'][2]
                                #const
                                slopes[i,3] = data_dict_all[region_key]['model_return'][type_key][unit_num]['param_dict'][0]

                                #alpha p val
                                slopes[i,4] = data_dict_all[region_key]['model_return'][type_key][unit_num]['stat_result_dict']['pvalues'][1]
                                #alpha p val
                                slopes[i,5] = data_dict_all[region_key]['model_return'][type_key][unit_num]['stat_result_dict']['pvalues'][2]
                                #alpha p val
                                slopes[i,6] = data_dict_all[region_key]['model_return'][type_key][unit_num]['stat_result_dict']['pvalues'][0]
                                
                                if slopes[i,4] <= 0.05 and slopes[i,5] <= 0.05:
                                        sig_all_slopes[sig_all_index,:] = slopes[i,:]
                                        sig_all_index = sig_all_index + 1
                                elif slopes[i,4] <= 0.05:
                                        alpha_only_sig +=1
                                elif slopes[i,5] <= 0.05:
                                        beta_only_sig <= 0.05

                                if slopes[i,1] > 0 and slopes[i,2] > 0:
                                        both_pos += 1
                                elif slopes[i,1] < 0 and slopes[i,2] < 0:
                                        both_neg += 1
                                elif slopes[i,1] > 0 and slopes[i,2] < 0:
                                        alpha_pos += 1
                                elif slopes[i,1] < 0 and slopes[i,2] > 0:
                                        beta_pos += 1

                        sig_all_slopes = sig_all_slopes[sig_all_slopes[:,1] != 0]
                        data_dict_all[region_key]['slopes'][type_key] = slopes
                        data_dict_all[region_key]['sig_all_slopes'][type_key] = sig_all_slopes
                        data_dict_all[region_key]['alpha_beta_only_sig'][type_key] = [alpha_only_sig,beta_only_sig,both_pos,both_neg,alpha_pos,beta_pos]

                #########

                all_alpha_only_sig = 0
                all_beta_only_sig = 0
                all_both_pos = 0
                all_both_neg = 0
                all_alpha_pos = 0
                all_beta_pos = 0

                all_slopes = np.zeros((np.shape(data_dict_all[region_key]['model_return'][type_key]['conc']['avg_fr_data'])[1],7))
                all_sig_all_slopes = np.zeros((np.shape(data_dict_all[region_key]['model_return'][type_key]['conc']['avg_fr_data'])[1],7))
                all_sig_all_index = 0

                for i in range(np.shape(data_dict_all[region_key]['model_return'][type_key]['conc']['avg_fr_data'])[1]):
                        unit_num = i
                        all_slopes[i,0] = unit_num
                        #alpha
                        all_slopes[i,1] = data_dict_all[region_key]['model_return'][type_key][unit_num]['param_dict'][1]
                        #beta
                        all_slopes[i,2] = data_dict_all[region_key]['model_return'][type_key][unit_num]['param_dict'][2]
                        #const
                        all_slopes[i,3] = data_dict_all[region_key]['model_return'][type_key][unit_num]['param_dict'][0]

                        #alpha p val
                        all_slopes[i,4] = data_dict_all[region_key]['model_return'][type_key][unit_num]['stat_result_dict']['pvalues'][1]
                        #alpha p val
                        all_slopes[i,5] = data_dict_all[region_key]['model_return'][type_key][unit_num]['stat_result_dict']['pvalues'][2]
                        #alpha p val
                        all_slopes[i,6] = data_dict_all[region_key]['model_return'][type_key][unit_num]['stat_result_dict']['pvalues'][0]
                
                        if all_slopes[i,4] <= 0.05 and all_slopes[i,5] <= 0.05:
                                all_sig_all_slopes[all_sig_all_index,:] = all_slopes[i,:]
                                all_sig_all_index = all_sig_all_index + 1
                        elif all_slopes[i,4] <= 0.05:
                                all_alpha_only_sig +=1
                        elif all_slopes[i,5] <= 0.05:
                                all_beta_only_sig <= 0.05
                        if all_slopes[i,1] > 0 and all_slopes[i,2] > 0:
                                all_both_pos += 1
                        elif all_slopes[i,1] < 0 and all_slopes[i,2] < 0:
                                all_both_neg += 1
                        elif all_slopes[i,1] > 0 and all_slopes[i,2] < 0:
                                all_alpha_pos += 1
                        elif all_slopes[i,1] < 0 and all_slopes[i,2] > 0:
                                all_beta_pos += 1


                all_sig_all_slopes = all_sig_all_slopes[sig_all_slopes[:,1] != 0]
                data_dict_all[region_key]['all_slopes'][type_key] = all_slopes
                data_dict_all[region_key]['all_sig_all_slopes'][type_key] = all_sig_all_slopes
                data_dict_all[region_key]['all_alpha_beta_only_sig'][type_key] = [all_alpha_only_sig,all_beta_only_sig,all_both_pos,all_both_neg,all_alpha_pos,all_beta_pos]

                        
                #do same for units where alpha and beta both sig

#######
for region_key,region_val in data_dict_all.iteritems():
        
        data_dict_all[region_key]['mv'] = {}

        bfr_cue_array = data_dict_all[region_key]['all_slopes']['bfr_cue_model']
        aft_cue_array = data_dict_all[region_key]['all_slopes']['aft_cue_model']
        bfr_result_array = data_dict_all[region_key]['all_slopes']['bfr_result_model']
        aft_result_array = data_dict_all[region_key]['all_slopes']['aft_result_model']

        avg_alphas = (bfr_cue_array[:,1] + aft_cue_array[:,1] + bfr_result_array[:,1] + aft_result_array[:,1]) / float(4)
        avg_betas = (bfr_cue_array[:,2] + aft_cue_array[:,2] + bfr_result_array[:,2] + aft_result_array[:,2]) / float(4)

        rnums = condensed[:,3]
        pnums = condensed[:,4]
        data_dict_all[region_key]['mv']['avg'] = {}
        for type_key,type_val in data_dict_all[region_key]['all_slopes'].iteritems():
                slopes = data_dict_all[region_key]['all_slopes'][type_key]
                #mv_array = np.zeros((np.shape(condensed)[0],np.shape(slopes)[0],5))
                temp = np.zeros((np.shape(avg_alphas)[0],np.shape(condensed)[0],8))
                rnums = condensed[:,3]
                pnums = condensed[:,4]

                for i in range(np.shape(avg_alphas)[0]):
                        unit = i
                        
                        alpha = avg_alphas[i]
                        beta= avg_betas[i]
                                                
                        unit_condensed = np.zeros((np.shape(condensed)[0],8))
                        unit_condensed[:,0] = rnums
                        unit_condensed[:,1] = pnums
                        unit_condensed[:,2] = alpha
                        unit_condensed[:,3] = beta
                        
                        if abs_alphabeta:
                                alpha = abs(alpha)
                                beta = abs(beta)

                        unit_condensed[:,4] = rnums * alpha - pnums * beta #value
                        unit_condensed[:,5] = rnums * alpha + pnums * beta #motiv
                
                        window_avg_unit_fr = data_dict_all[region_key]['model_return'][type_key]['conc']['avg_fr_data'][:,unit]
                        unit_condensed[:,6] = window_avg_unit_fr
                        unit_condensed[:,7] = unit
                        
                        temp[i,:,:] = unit_condensed
                
                        new_mv_array = temp
                
                        data_dict_all[region_key]['mv']['avg'][type_key] = new_mv_array
                
                        np.save('%s_%s_all_avg_mv_array.npy' %(region_key,type_key),new_mv_array)
                        sio.savemat('%s_%s_all_avg_mv_array.mat' %(region_key,type_key),{'mv_array':new_mv_array},format='5')




###### all ##########
        data_dict_all[region_key]['mv']['all'] = {}
        for type_key,type_val in data_dict_all[region_key]['all_slopes'].iteritems():
                slopes = data_dict_all[region_key]['all_slopes'][type_key]
                #mv_array = np.zeros((np.shape(condensed)[0],np.shape(slopes)[0],5))
                temp = np.zeros((np.shape(avg_alphas)[0],np.shape(condensed)[0],8))
                alphas = slopes[:,1]
                betas = slopes[:,2]
                rnums = condensed[:,3]
                pnums = condensed[:,4]

                for i in range(np.shape(avg_alphas)[0]):
                        unit = i
                        
                        alpha = alphas[i]
                        beta= betas[i]
                                                
                        unit_condensed = np.zeros((np.shape(condensed)[0],8))
                        unit_condensed[:,0] = rnums
                        unit_condensed[:,1] = pnums
                        unit_condensed[:,2] = alpha
                        unit_condensed[:,3] = beta
                        
                        if abs_alphabeta:
                                alpha = abs(alpha)
                                beta = abs(beta)

                        unit_condensed[:,4] = rnums * alpha - pnums * beta #value
                        unit_condensed[:,5] = rnums * alpha + pnums * beta #motiv
                
                        window_avg_unit_fr = data_dict_all[region_key]['model_return'][type_key]['conc']['avg_fr_data'][:,unit]
                        unit_condensed[:,6] = window_avg_unit_fr
                        unit_condensed[:,7] = unit
                        
                        temp[i,:,:] = unit_condensed
                
                        new_mv_array = temp
                
                        data_dict_all[region_key]['mv']['all'][type_key] = new_mv_array
                
                        np.save('%s_%s_all_mv_array.npy' %(region_key,type_key),new_mv_array)
                        sio.savemat('%s_%s_all_mv_array.mat' %(region_key,type_key),{'mv_array':new_mv_array},format='5')

###########




#slopes_workbook = xlsxwriter.Workbook('sig_slopes.xlsx',options={'nan_inf_to_errors':True})
for region_key,region_val in data_dict_all.iteritems():
        slopes_workbook = xlsxwriter.Workbook('sig_slopes_%s.xlsx' %(region_key),options={'nan_inf_to_errors':True})
        total_unit_num = np.shape(data_dict_all[region_key]['fr_dict']['baseline_fr'])[1]
        names = ['unit_num','alpha','beta','const','alpha_p','beta_p','const_p']

        percs = {}
        for type_key,type_val in data_dict_all[region_key]['slopes'].iteritems():
                
                slopes = np.asarray(data_dict_all[region_key]['slopes'][type_key])
                sig_slopes = np.asarray(data_dict_all[region_key]['sig_all_slopes'][type_key])
                sig_alpha = slopes[slopes[:,4] <= 0.05]
                sig_beta = slopes[slopes[:,5] <= 0.05]
                perc_sig_alpha = np.shape(sig_alpha)[0] / float(total_unit_num)
                perc_sig_beta = np.shape(sig_beta)[0] / float(total_unit_num)
                num_sig_alpha = np.shape(sig_alpha)[0]
                num_sig_beta = np.shape(sig_beta)[0]
                perc_slopes = np.shape(slopes)[0] / float(total_unit_num)
                perc_sig_slopes = np.shape(sig_slopes)[0] / float(total_unit_num)
                perc_sig_alpha_only = data_dict_all[region_key]['alpha_beta_only_sig'][type_key][0] / float(total_unit_num)
                num_sig_alpha_only = data_dict_all[region_key]['alpha_beta_only_sig'][type_key][0]
                perc_sig_beta_only = data_dict_all[region_key]['alpha_beta_only_sig'][type_key][1] / float(total_unit_num)
                num_sig_beta_only = data_dict_all[region_key]['alpha_beta_only_sig'][type_key][0]
                perc_both_pos = data_dict_all[region_key]['alpha_beta_only_sig'][type_key][2] / float(np.shape(slopes)[0]) #of at least one sig
                perc_both_neg = data_dict_all[region_key]['alpha_beta_only_sig'][type_key][3] / float(np.shape(slopes)[0])
                perc_alpha_pos = data_dict_all[region_key]['alpha_beta_only_sig'][type_key][4] / float(np.shape(slopes)[0])
                perc_beta_pos = data_dict_all[region_key]['alpha_beta_only_sig'][type_key][5] / float(np.shape(slopes)[0])

                #pdb.set_trace()
                worksheet = slopes_workbook.add_worksheet('slopes_%s' %(type_key))
                worksheet.write_row(0,0,names)
                if slopes.ndim == 2:
                        len_slopes = np.shape(slopes)[0]
                elif np.shape(slopes)[0] == 0:
                        len_slopes = -1
                else:
                        len_slopes = 0
                
                if not len_slopes == -1:
                        for i in range(len_slopes):
                                worksheet.write_row(i+1,0,slopes[i,:])
                
                worksheet = slopes_workbook.add_worksheet('sig_all_%s' %(type_key))
                worksheet.write_row(0,0,names)
                if sig_slopes.ndim == 2:
                        len_sig_slopes = np.shape(sig_slopes)[0]
                elif np.shape(sig_slopes)[0] == 0:
                        len_sig_slopes = -1
                else:
                        len_sig_slopes = 0
                                
                if not len_sig_slopes == -1:
                        for i in range(len_sig_slopes):
                                worksheet.write_row(i+1,0,sig_slopes[i,:])
                
                type_dict = {'perc_slopes':perc_slopes,'perc_sig_slopes':perc_sig_slopes,'num_slopes':np.shape(slopes)[0],'num_sig_slopes':np.shape(sig_slopes)[0],'total_unit_num':total_unit_num,'perc_sig_alpha':perc_sig_alpha,'perc_sig_beta':perc_sig_beta,'num_sig_alpha':num_sig_alpha,'num_sig_beta':num_sig_beta,'perc_sig_alpha_only':perc_sig_alpha_only,'num_sig_alpha_only':num_sig_alpha_only,'perc_sig_beta_only':perc_sig_beta_only,'num_sig_beta_only':num_sig_beta_only,'perc_both_pos':perc_both_pos,'perc_both_neg':perc_both_neg,'perc_alpha_pos':perc_alpha_pos,'perc_beta_pos':perc_beta_pos}
                percs[type_key] = type_dict
        data_dict_all[region_key]['percs'] = percs

        
percs_workbook = xlsxwriter.Workbook('percs_workbook.xlsx',options={'nan_inf_to_errors':True})
for region_key,region_val in data_dict_all.iteritems():
        worksheet = percs_workbook.add_worksheet('%s' %(region_key))
        worksheet.write(1,0,'perc slopes')
        worksheet.write(2,0,'perc sig slopes')
        worksheet.write(3,0,'num slopes')
        worksheet.write(4,0,'num sig slopes')
        worksheet.write(5,0,'total unit num')
        worksheet.write(6,0,'perc sig alpha')
        worksheet.write(7,0,'perc sig beta')
        worksheet.write(8,0,'num sig alpha')
        worksheet.write(9,0,'num sig beta')
        worksheet.write(10,0,'perc sig alpha only')
        worksheet.write(11,0,'perc sig beta only')
        worksheet.write(12,0,'num sig alpha only')
        worksheet.write(13,0,'num sig beta only')
        worksheet.write(14,0,'both alpha and beta pos')
        worksheet.write(15,0,'both alpha and beta neg')
        worksheet.write(16,0,'alpha pos beta neg')
        worksheet.write(17,0,'alpha neg beta pos')
        
        
        percs = data_dict_all[region_key]['percs']

        i = 1
        for type_key,val in percs.iteritems():
                worksheet.write(0,i,type_key)
                worksheet.write_column(1,i,[percs[type_key]['perc_slopes'],percs[type_key]['perc_sig_slopes'],percs[type_key]['num_slopes'],percs[type_key]['num_sig_slopes'],percs[type_key]['total_unit_num'],percs[type_key]['perc_sig_alpha'],percs[type_key]['perc_sig_beta'],percs[type_key]['num_sig_alpha'],percs[type_key]['num_sig_beta'],percs[type_key]['perc_sig_alpha_only'],percs[type_key]['perc_sig_beta_only'],percs[type_key]['num_sig_alpha_only'],percs[type_key]['num_sig_beta_only'],percs[type_key]['perc_both_pos'],percs[type_key]['perc_both_neg'],percs[type_key]['perc_alpha_pos'],percs[type_key]['perc_beta_pos']])
                i += 1



if mv_bool:
        #calc updated motiv and val vectors
        for region_key,region_val in data_dict_all.iteritems():
                #for now individ. should avg across windows? And if so does that affect any other of the analysis?
                #data_dict_all[region_key]['mv'] = {}
                for type_key,type_val in data_dict_all[region_key]['slopes'].iteritems():
                        slopes = data_dict_all[region_key]['slopes'][type_key]
                        sig_units = slopes[:,0]

                        alphas = slopes[:,1]
                        betas = slopes[:,2]
                        rnums = condensed[:,3]
                        pnums = condensed[:,4]

                        #mv_array = np.zeros((np.shape(condensed)[0],np.shape(slopes)[0],5))
                        temp = np.zeros((np.shape(slopes)[0],np.shape(condensed)[0],8))
                        for i in range(np.shape(slopes)[0]):
                                unit = sig_units[i]
                                
                                if abs_alphabeta:
                                        alpha = abs(alphas[i])
                                        beta = abs(betas[i])
                                else:
                                        alpha = alphas[i]
                                        beta = betas[i]

                                unit_condensed = np.zeros((np.shape(condensed)[0],8))
                                unit_condensed[:,0] = rnums
                                unit_condensed[:,1] = pnums
                                unit_condensed[:,2] = alpha
                                unit_condensed[:,3] = beta
                                unit_condensed[:,4] = rnums * alpha - pnums * beta #value
                                unit_condensed[:,5] = rnums * alpha + pnums * beta #motiv

                                window_avg_unit_fr = data_dict_all[region_key]['model_return'][type_key]['conc']['avg_fr_data'][:,unit]
                                unit_condensed[:,6] = window_avg_unit_fr
                                unit_condensed[:,7] = unit
                                temp[i,:,:] = unit_condensed

                        new_mv_array = temp
                                
                        data_dict_all[region_key]['mv'][type_key] = new_mv_array

                        np.save('%s_%s_mv_array.npy' %(region_key,type_key),new_mv_array)
                        sio.savemat('%s_%s_mv_array.mat' %(region_key,type_key),{'mv_array':new_mv_array},format='5')


#for region_key,region_val in data_dict_all.iteritems():
#        bfr_cue_array = data_dict_all[region_key]['all_slopes']['bfr_cue_model']
#        aft_cue_array = data_dict_all[region_key]['all_slopes']['aft_cue_model']
#        bfr_result_array = data_dict_all[region_key]['all_slopes']['bfr_result_model']
#        aft_result_array = data_dict_all[region_key]['all_slopes']['aft_result_model']
#
#        avg_alphas = (bfr_cue_array[:,1] + aft_cue_array[:,1] + bfr_result_array[:,1] + aft_result_array[:,1]) / float(4)
#        avg_betas = (bfr_cue_array[:,2] + aft_cue_array[:,2] + bfr_result_array[:,2] + aft_result_array[:,2]) / float(4)
#
#        rnums = condensed[:,3]
#        pnums = condensed[:,4]
#        data_dict_all[region_key]['mv']['all'] = {}

#        for type_key,type_val in data_dict_all[region_key]['all_slopes'].iteritems():
#
#                #mv_array = np.zeros((np.shape(condensed)[0],np.shape(slopes)[0],5))
#                temp = np.zeros((np.shape(avg_alphas)[0],np.shape(condensed)[0],8))
#                for i in range(np.shape(avg_alphas)[0]):
#                        unit = i
#                                
#                        if abs_alphabeta:
#                                alpha = abs(avg_alphas[i])
#                                beta = abs(avg_betas[i])
#                        else:
#                                alpha = avg_alphas[i]
#                                beta = avg_betas[i]
                        
#                        unit_condensed = np.zeros((np.shape(condensed)[0],8))
#                        unit_condensed[:,0] = rnums
#                        unit_condensed[:,1] = pnums
#                        unit_condensed[:,2] = alpha
#                        unit_condensed[:,3] = beta
#                        unit_condensed[:,4] = rnums * alpha - pnums * beta #value
#                        unit_condensed[:,5] = rnums * alpha + pnums * beta #motiv
#                
#                        window_avg_unit_fr = data_dict_all[region_key]['model_return'][type_key]['conc']['avg_fr_data'][:,unit]
#                        unit_condensed[:,6] = window_avg_unit_fr
#                        unit_condensed[:,7] = unit
#                        
#                        temp[i,:,:] = unit_condensed
                
#                        new_mv_array = temp
#                
#                        data_dict_all[region_key]['mv']['all'][type_key] = new_mv_array
#                
#                        np.save('%s_%s_all_mv_array.npy' %(region_key,type_key),new_mv_array)
#                        sio.savemat('%s_%s_all_mv_array.mat' %(region_key,type_key),{'mv_array':new_mv_array},format='5')


#For input into GPFA. Run on bin size of 1ms, other params = False (ie no normalizing, etc)
M1_hist_dict = {'bfr_cue':data_dict_all['M1_dicts']['fr_dict']['bfr_cue_hist'],'aft_cue':data_dict_all['M1_dicts']['fr_dict']['aft_cue_hist'],'bfr_result':data_dict_all['M1_dicts']['fr_dict']['bfr_result_hist'],'aft_result':data_dict_all['M1_dicts']['fr_dict']['aft_result_hist']}

S1_hist_dict = {'bfr_cue':data_dict_all['S1_dicts']['fr_dict']['bfr_cue_hist'],'aft_cue':data_dict_all['S1_dicts']['fr_dict']['aft_cue_hist'],'bfr_result':data_dict_all['S1_dicts']['fr_dict']['bfr_result_hist'],'aft_result':data_dict_all['S1_dicts']['fr_dict']['aft_result_hist']}

PmD_hist_dict = {'bfr_cue':data_dict_all['PmD_dicts']['fr_dict']['bfr_cue_hist'],'aft_cue':data_dict_all['PmD_dicts']['fr_dict']['aft_cue_hist'],'bfr_result':data_dict_all['PmD_dicts']['fr_dict']['bfr_result_hist'],'aft_result':data_dict_all['PmD_dicts']['fr_dict']['aft_result_hist']}

master_hist_dict = {'M1_hist_dict':M1_hist_dict,'S1_hist_dict':S1_hist_dict,'PmD_hist_dict':PmD_hist_dict,'condensed':condensed}

sio.savemat('master_hist',master_hist_dict)


#For input into tdr. Run with zscore normalization, whatever bin size.
M1_fr_dict = {'bfr_cue':data_dict_all['M1_dicts']['fr_dict']['bfr_cue_fr'],'aft_cue':data_dict_all['M1_dicts']['fr_dict']['aft_cue_fr'],'bfr_result':data_dict_all['M1_dicts']['fr_dict']['bfr_result_fr'],'aft_result':data_dict_all['M1_dicts']['fr_dict']['aft_result_fr']}
S1_fr_dict = {'bfr_cue':data_dict_all['S1_dicts']['fr_dict']['bfr_cue_fr'],'aft_cue':data_dict_all['S1_dicts']['fr_dict']['aft_cue_fr'],'bfr_result':data_dict_all['S1_dicts']['fr_dict']['bfr_result_fr'],'aft_result':data_dict_all['S1_dicts']['fr_dict']['aft_result_fr']}
PmD_fr_dict = {'bfr_cue':data_dict_all['PmD_dicts']['fr_dict']['bfr_cue_fr'],'aft_cue':data_dict_all['PmD_dicts']['fr_dict']['aft_cue_fr'],'bfr_result':data_dict_all['PmD_dicts']['fr_dict']['bfr_result_fr'],'aft_result':data_dict_all['PmD_dicts']['fr_dict']['aft_result_fr']}


params = {'time_before':time_before,'time_after':time_after,'bin_size':bin_size,'baseline_time':baseline_time,'normalize_bool':normalize_bool,'sqrt_bool':sqrt_bool,'zscore_bool':zscore,'gaussian_bool':gaussian_bool,'gauss_sigma':gauss_sigma}
master_fr_dict = {'M1_fr_dict':M1_fr_dict,'S1_fr_dict':S1_fr_dict,'PmD_fr_dict':PmD_fr_dict,'condensed':condensed,'params':params}



if extracted_filename == 'Extracted_504_2017-02-08-10-36-11.mat':
	np.save('master_fr_dict_5_8_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-02-08-11-02-03.mat':
	np.save('master_fr_dict_5_8_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-02-09-11-50-03.mat':
	np.save('master_fr_dict_5_9_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-02-09-12-15-57.mat':
	np.save('master_fr_dict_5_9_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-02-14-12-09-21.mat':
	np.save('master_fr_dict_5_14_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-02-14-12-35-41.mat':
	np.save('master_fr_dict_5_14_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-02-14-13-01-34.mat':
	np.save('master_fr_dict_5_14_3.npy',master_fr_dict)

elif extracted_filename == 'Extracted_0059_2017-02-08-11-43-22.mat':
	np.save('master_fr_dict_0_8_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-02-08-12-09-22.mat':
	np.save('master_fr_dict_0_8_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-02-09-12-52-17.mat':
	np.save('master_fr_dict_0_9_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-02-09-13-46-37.mat':
	np.save('master_fr_dict_0_9_2.npy',master_fr_dict)


elif extracted_filename == 'Extracted_0059_2017-03-09-13-47-33.mat':
        np.save('master_fr_dict_0_9_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-03-09-14-12-52.mat':
        np.save('master_fr_dict_0_9_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-03-09-14-36-42.mat':
        np.save('master_fr_dict_0_9_3.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-03-10-13-38-52.mat':
        np.save('master_fr_dict_0_10_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-03-10-13-54-44.mat':
        np.save('master_fr_dict_0_10_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-03-10-14-21-33.mat':
        np.save('master_fr_dict_0_10_3.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-03-13-14-26-36.mat':
        np.save('master_fr_dict_0_13_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-03-13-14-54-09.mat':
        np.save('master_fr_dict_0_13_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-03-13-15-24-14.mat':
        np.save('master_fr_dict_0_13_3.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-03-14-14-15-04.mat':
        np.save('master_fr_dict_0_14_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-03-14-14-51-42.mat':
        np.save('master_fr_dict_0_14_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2017-03-14-15-17-37.mat':
        np.save('master_fr_dict_0_14_3.npy',master_fr_dict)

elif extracted_filename == 'Extracted_504_2017-03-09-12-38-12.mat':
        np.save('master_fr_dict_5_9_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-03-09-13-01-02.mat':
        np.save('master_fr_dict_5_9_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-03-10-12-21-53.mat':
        np.save('master_fr_dict_5_10_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-03-10-12-32-43.mat':
        np.save('master_fr_dict_5_10_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-03-10-12-56-45.mat':
        np.save('master_fr_dict_5_10_3.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-03-13-12-50-41.mat':
        np.save('master_fr_dict_5_13_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-03-13-13-16-32.mat':
        np.save('master_fr_dict_5_13_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-03-13-13-42-37.mat':
        np.save('master_fr_dict_5_13_3.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-03-14-12-43-43.mat':
        np.save('master_fr_dict_5_14_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-03-14-13-09-46.mat':
        np.save('master_fr_dict_5_14_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2017-03-14-13-36-38.mat':
        np.save('master_fr_dict_5_14_3.npy',master_fr_dict)

elif extracted_filename == 'Extracted_0059_2015-10-19-16-46-25.mat':
        np.save('master_fr_dict_0_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_0059_2016-01-18-13-02-45.mat':
        np.save('master_fr_dict_0_2.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2015-09-29-12-48-19.mat':
        np.save('master_fr_dict_5_1.npy',master_fr_dict)
elif extracted_filename == 'Extracted_504_2016-01-11-14-10-01.mat':
        np.save('master_fr_dict_5_2.npy',master_fr_dict)


else:
	np.save('master_fr_dict.npy',master_fr_dict)














	
