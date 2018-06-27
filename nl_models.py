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

#######################
#params to set ########
#######################

bin_size = 10 #in ms
time_before = -0.5 #negative value
time_after = 1.0
#zscore = True

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

        #if zscore and gaussian_bool:
        #    hists = stats.zscore(hists,axis=1)
        #    hists = ndimage.filters.gaussian_filter1d(hists,gauss_sigma,axis=1)
        #elif zscore:
        #    hists = stats.zscore(hists,axis=1)
        #elif gaussian_bool:
        #    hists = ndimage.filters.gaussian_filter1d(hists,gauss_sigma,axis=1)

        hists = stats.zscore(hists,axis=1)

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
                
                if not (result_start_bin + bins_after) > np.shape(hists)[1]:
                        for j in range(np.shape(hists)[0]):
                                bfr_cue_hist = hists[j,bins_before + cue_start_bin : cue_start_bin]
                                aft_cue_hist = hists[j,cue_start_bin : cue_start_bin + bins_after]
                                bfr_result_hist = hists[j,bins_before + result_start_bin : result_start_bin]
                                aft_result_hist = hists[j,result_start_bin : result_start_bin + bins_after]
   
                                bfr_cue_hist_all[i,j,:] = bfr_cue_hist
                                aft_cue_hist_all[i,j,:] = aft_cue_hist
                                bfr_result_hist_all[i,j,:] = bfr_result_hist
                                aft_result_hist_all[i,j,:] = aft_result_hist
                     
                                bfr_cue_fr[i,j,:] = bfr_cue_hist
                                aft_cue_fr[i,j,:] = aft_cue_hist
                                bfr_result_fr[i,j,:] = bfr_result_hist
                                aft_result_fr[i,j,:] = aft_result_hist
                else:
                        continue

        return_dict = {'bfr_cue_fr':bfr_cue_fr,'aft_cue_fr':aft_cue_fr,'bfr_result_fr':bfr_result_fr,'aft_result_fr':aft_result_fr,'bfr_cue_hist':bfr_cue_hist_all,'aft_cue_hist':aft_cue_hist_all,'bfr_result_hist':bfr_result_hist_all,'aft_result_hist':aft_result_hist_all}

        return(return_dict)

def lin_func(x,a,b,c):
        r,p = x
        return a*r + b*p + c

def diff_func(x,a,b):
        r,p = x

        return a + b*(r - p)

def div_nl_func(x,a,b,c,d,e):
        #from Ruff, Cohen: relating normalization to neuronl populations across cortical areas
        r,p = x        
        #add an e to the whole thing for intercept?
        return (r * a + b * p) / (a + c * b + d) + e

def div_nl_noe_func(x,a,b,c,d):
        #from Ruff, Cohen: relating normalization to neuronl populations across cortical areas
        r,p = x        
        #add an e to the whole thing for intercept?
        return (r * a + b * p) / (a + c * b + d)

def div_nl_Y_func(x,a,b,c,d):
        #div nl func Yao is using
        r,p = x
        return a*(r + b*p) / (c + r + b * p) + d

####################
# models ###########
####################

#TODO combine and collate all

def make_lin_model(fr_data,condensed,region_key,type_key):
        #set each model
        num_params = 3

        avg_fr_data = np.mean(fr_data,axis=2)

        r_vals = condensed[:,3]
        p_vals = condensed[:,4]

        fit_params = np.zeros((fr_data.shape[1],num_params))
        cov_total = np.zeros((fr_data.shape[1],num_params,num_params))
        perr_total = np.zeros((fr_data.shape[1],num_params))
        AIC_total = np.zeros((fr_data.shape[1]))
        ss_res_total = np.zeros((fr_data.shape[1]))

        for unit_num in range(fr_data.shape[1]):
                avg_frs = avg_fr_data[:,unit_num]
        
                #firing rate = a*R + b*P + c
                params,covs = curve_fit(lin_func,[r_vals,p_vals],avg_frs)
                
                perr = np.sqrt(np.diag(covs))

                if np.size(covs) > 1:
                        perr = np.sqrt(np.diag(covs))
                elif isinf(covs):
                        perr = float('nan')
                else:
                        pdb.set_trace()

                fit_params[unit_num,:] = params
                cov_total[unit_num,:,:] = covs
                perr_total[unit_num,:] = perr

                model_out = lin_func([r_vals,p_vals],params[0],params[1],params[2])
                residuals = avg_frs - model_out
                ss_res = np.sum(residuals**2)
                k = num_params
                AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
                
                AIC_total[unit_num] = AIC
                ss_res_total[unit_num] = ss_res

        
        params,covs = curve_fit(lin_func,[np.tile(r_vals,fr_data.shape[1]),np.tile(p_vals,fr_data.shape[1])],np.ndarray.flatten(avg_fr_data))
        
        if np.size(covs) > 1:
                perr = np.sqrt(np.diag(covs))
        elif isinf(covs):
                perr = float('nan')
        else:
                pdb.set_trace()

        model_out = lin_func([r_vals,p_vals],params[0],params[1],params[2])
        residuals = avg_frs - model_out
        ss_res = np.sum(residuals**2)
        k = num_params
        AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)

        print '%s linear AIC: %s' %(type_key,AIC)

        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}
        
        return_dict = {'fit_params':fit_params,'cov_total':cov_total,'perr_total':perr_total,'AIC_total':AIC_total,'ss_res_total':ss_res_total,'combined':combined_dict}
                
        return(return_dict)


def make_diff_model(fr_data,condensed,region_key,type_key):
        num_params = 2
        
        avg_fr_data = np.mean(fr_data,axis=2)

        r_vals = condensed[:,3]
        p_vals = condensed[:,4]

        fit_params = np.zeros((fr_data.shape[1],num_params))
        cov_total = np.zeros((fr_data.shape[1],num_params,num_params))
        perr_total = np.zeros((fr_data.shape[1],num_params))
        AIC_total = np.zeros((fr_data.shape[1]))
        ss_res_total = np.zeros((fr_data.shape[1]))

        for unit_num in range(fr_data.shape[1]):
                avg_frs = avg_fr_data[:,unit_num]
        
                #firing rate = a + b(R - P)
                params,covs = curve_fit(diff_func,[r_vals,p_vals],avg_frs)
                
                perr = np.sqrt(np.diag(covs))

                fit_params[unit_num,:] = params
                cov_total[unit_num,:,:] = covs
                perr_total[unit_num,:] = perr
                if np.size(covs) > 1:
                        perr = np.sqrt(np.diag(covs))
                elif isinf(covs):
                        perr = float('nan')
                else:
                        pdb.set_trace()

                fit_params[unit_num,:] = params
                cov_total[unit_num,:,:] = covs
                perr_total[unit_num,:] = perr
                model_out = diff_func([r_vals,p_vals],params[0],params[1])
                residuals = avg_frs - model_out
                ss_res = np.sum(residuals**2)
                k = num_params
                AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
                
                AIC_total[unit_num] = AIC
                ss_res_total[unit_num] = ss_res

        params,covs = curve_fit(lin_func,[np.tile(r_vals,fr_data.shape[1]),np.tile(p_vals,fr_data.shape[1])],np.ndarray.flatten(avg_fr_data))
        
        if np.size(covs) > 1:
                perr = np.sqrt(np.diag(covs))
        elif isinf(covs):
                perr = float('nan')
        else:
                pdb.set_trace()

        model_out = diff_func([r_vals,p_vals],params[0],params[1])
        residuals = avg_frs - model_out
        ss_res = np.sum(residuals**2)
        k = num_params
        AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)

        print '%s difference AIC: %s' %(type_key,AIC)
        
        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}
        return_dict = {'fit_params':fit_params,'cov_total':cov_total,'perr_total':perr_total,'AIC_total':AIC_total,'ss_res_total':ss_res_total,'combined':combined_dict}
                
        return(return_dict)

def make_div_nl_model(fr_data,condensed,region_key,type_key):
        num_params = 5

        avg_fr_data = np.mean(fr_data,axis=2)

        r_vals = condensed[:,3]
        p_vals = condensed[:,4]

        cov_total = []
        fit_params = np.zeros((fr_data.shape[1],num_params))
        cov_total = np.zeros((fr_data.shape[1],num_params,num_params))
        perr_total = np.zeros((fr_data.shape[1],num_params))
        AIC_total = np.zeros((fr_data.shape[1]))
        ss_res_total = np.zeros((fr_data.shape[1]))

        for unit_num in range(fr_data.shape[1]):
                avg_frs = avg_fr_data[:,unit_num]

                #firing rate  = (R * a + b * P) / (a + c * b + d) + e
                params,covs = curve_fit(div_nl_func,[r_vals,p_vals],avg_frs)
                
                if np.size(covs) > 1:
                        perr = np.sqrt(np.diag(covs))
                elif isinf(covs):
                        perr = float('nan')
                else:
                        pdb.set_trace()

                fit_params[unit_num,:] = params
                cov_total[unit_num,:,:] = covs
                perr_total[unit_num,:] = perr

                model_out = div_nl_func([r_vals,p_vals],params[0],params[1],params[2],params[3],params[4])
                residuals = avg_frs - model_out
                ss_res = np.sum(residuals**2)
                k = num_params
                AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
                
                AIC_total[unit_num] = AIC
                ss_res_total[unit_num] = ss_res

                #calculate residuals, -> AIC
                #popt, pcov = curve_fit(f, xdata, ydata)

                #You can get the residual sum of squares (ss_tot) with
                #residuals = ydata- f(xdata, popt)
                #ss_res = numpy.sum(residuals**2)

                #from reddit
                #sse = sum(resid**2)
                #k= # of variables
                #AIC= 2k - 2ln(sse)
                
        params,covs = curve_fit(lin_func,[np.tile(r_vals,fr_data.shape[1]),np.tile(p_vals,fr_data.shape[1])],np.ndarray.flatten(avg_fr_data))
        
        if np.size(covs) > 1:
                perr = np.sqrt(np.diag(covs))
        elif isinf(covs):
                perr = float('nan')
        else:
                pdb.set_trace()

        model_out = div_nl_func([r_vals,p_vals],params[0],params[1],params[2],params[3],params[4])
        residuals = avg_frs - model_out
        ss_res = np.sum(residuals**2)
        k = num_params
        AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
        
        print '%s div nl AIC: %s' %(type_key,AIC)

        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}
        return_dict = {'fit_params':fit_params,'cov_total':cov_total,'perr_total':perr_total, 'AIC_total':AIC_total,'ss_res_total':ss_res_total,'combined':combined_dict}
        
        return(return_dict)

def make_div_nl_noe_model(fr_data,condensed,region_key,type_key):
        num_params = 4

        avg_fr_data = np.mean(fr_data,axis=2)

        r_vals = condensed[:,3]
        p_vals = condensed[:,4]

        cov_total = []
        fit_params = np.zeros((fr_data.shape[1],num_params))
        cov_total = np.zeros((fr_data.shape[1],num_params,num_params))
        perr_total = np.zeros((fr_data.shape[1],num_params))
        AIC_total = np.zeros((fr_data.shape[1]))
        ss_res_total = np.zeros((fr_data.shape[1]))

        for unit_num in range(fr_data.shape[1]):
                avg_frs = avg_fr_data[:,unit_num]

                #firing rate  = (R * a + b * P) / (a + c * b + d)
                params,covs = curve_fit(div_nl_noe_func,[r_vals,p_vals],avg_frs)
                
                if np.size(covs) > 1:
                        perr = np.sqrt(np.diag(covs))
                elif isinf(covs):
                        perr = float('nan')
                else:
                        pdb.set_trace()

                fit_params[unit_num,:] = params
                cov_total[unit_num,:,:] = covs
                perr_total[unit_num,:] = perr

                model_out = div_nl_noe_func([r_vals,p_vals],params[0],params[1],params[2],params[3])
                residuals = avg_frs - model_out
                ss_res = np.sum(residuals**2)
                k = num_params
                AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
                
                AIC_total[unit_num] = AIC
                ss_res_total[unit_num] = ss_res

                #calculate residuals, -> AIC
                #popt, pcov = curve_fit(f, xdata, ydata)

                #You can get the residual sum of squares (ss_tot) with
                #residuals = ydata- f(xdata, popt)
                #ss_res = numpy.sum(residuals**2)

                #from reddit
                #sse = sum(resid**2)
                #k= # of variables
                #AIC= 2k - 2ln(sse)

        params,covs = curve_fit(div_nl_noe_func,[np.tile(r_vals,fr_data.shape[1]),np.tile(p_vals,fr_data.shape[1])],np.ndarray.flatten(avg_fr_data))
        
        if np.size(covs) > 1:
                perr = np.sqrt(np.diag(covs))
        elif isinf(covs):
                perr = float('nan')
        else:
                pdb.set_trace()

        model_out = div_nl_noe_func([r_vals,p_vals],params[0],params[1],params[2],params[3])
        residuals = avg_frs - model_out
        ss_res = np.sum(residuals**2)
        k = num_params
        AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
        
        print '%s div nl noe AIC: %s' %(type_key,AIC)

        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}
        return_dict = {'fit_params':fit_params,'cov_total':cov_total,'perr_total':perr_total, 'AIC_total':AIC_total,'ss_res_total':ss_res_total,'combined':combined_dict}
        
        return(return_dict)

def make_div_nl_Y_model(fr_data,condensed,region_key,type_key):
        num_params = 4

        avg_fr_data = np.mean(fr_data,axis=2)

        r_vals = condensed[:,3]
        p_vals = condensed[:,4]

        cov_total = []
        fit_params = np.zeros((fr_data.shape[1],num_params))
        cov_total = np.zeros((fr_data.shape[1],num_params,num_params))
        perr_total = np.zeros((fr_data.shape[1],num_params))
        AIC_total = np.zeros((fr_data.shape[1]))
        ss_res_total = np.zeros((fr_data.shape[1]))

        for unit_num in range(fr_data.shape[1]):
                avg_frs = avg_fr_data[:,unit_num]

                # model = a*(r + b*P) / (c + r + b * p) + d
                try:
                        params,covs = curve_fit(div_nl_Y_func,[r_vals,p_vals],avg_frs)
                except:
                        print 'failure to fit Y nl %s %s unit %s' %(region_key,type_key,unit_num)
                        break

                if np.size(covs) > 1:
                        perr = np.sqrt(np.diag(covs))
                elif isinf(covs):
                        perr = float('nan')
                else:
                        pdb.set_trace()

                fit_params[unit_num,:] = params
                cov_total[unit_num,:,:] = covs
                perr_total[unit_num,:] = perr

                model_out = div_nl_Y_func([r_vals,p_vals],params[0],params[1],params[2],params[3])
                residuals = avg_frs - model_out
                ss_res = np.sum(residuals**2)
                k = num_params
                AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
                
                AIC_total[unit_num] = AIC
                ss_res_total[unit_num] = ss_res

        params,covs = curve_fit(div_nl_Y_func,[np.tile(r_vals,fr_data.shape[1]),np.tile(p_vals,fr_data.shape[1])],np.ndarray.flatten(avg_fr_data))
        
        if np.size(covs) > 1:
                perr = np.sqrt(np.diag(covs))
        elif isinf(covs):
                perr = float('nan')
        else:
                pdb.set_trace()

        model_out = div_nl_Y_func([r_vals,p_vals],params[0],params[1],params[2],params[3])
        residuals = avg_frs - model_out
        ss_res = np.sum(residuals**2)
        k = num_params
        AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
        
        print '%s div nl Y AIC: %s' %(type_key,AIC)

        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}
        return_dict = {'fit_params':fit_params,'cov_total':cov_total,'perr_total':perr_total, 'AIC_total':AIC_total,'ss_res_total':ss_res_total,'combined':combined_dict}
        
        return(return_dict)




###############################################
#start ########################################
###############################################

bins_before = int(time_before / float(bin_size) * 1000)  #neg for now
bins_after = int(time_after / float(bin_size) * 1000)   

print 'bin size: %s' %(bin_size)
print 'time before: %s, time after: %s' %(time_before,time_after)

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

#remove trials with no succ or failure scene (not sure why, but saw in one)
condensed = condensed[np.invert(np.logical_and(condensed[:,1] == 0, condensed[:,2] == 0))]

#TODOD FOR NOW remove catch trials
condensed = condensed[condensed[:,5] == 0]
#col 5 all 0s now, replace with succ/fail vector: succ = 1, fail = -1
condensed[condensed[:,1] != 0, 5] = 1
condensed[condensed[:,2] != 0, 5] = -1


###################
#calc times, can move to other files as well
post_cue_times = condensed[:,1] + condensed[:,2] - condensed[:,0]
post_delivery_times = np.zeros((np.shape(condensed)[0]-1))
cue_to_cue_times = np.zeros((np.shape(condensed)[0]-1))
for i in range(np.shape(condensed)[0]-1):
        post_delivery_times[i] = condensed[i+1,0] - (condensed[i,1] + condensed[i,2])
        cue_to_cue_times[i] = condensed[i+1,0] - condensed[i,0]

post_cue_dict = {'avg':np.mean(post_cue_times),'median':np.median(post_cue_times),'max':np.max(post_cue_times),'min':np.min(post_cue_times)}
post_delivery_dict = {'avg':np.mean(post_delivery_times),'median':np.median(post_delivery_times),'max':np.max(post_delivery_times),'min':np.min(post_delivery_times)}
cue_cue_dict = {'avg':np.mean(cue_to_cue_times),'median':np.median(cue_to_cue_times),'max':np.max(cue_to_cue_times),'min':np.min(cue_to_cue_times)}
size_dict = {'unit_num':np.shape(condensed)[0]}

times_dict = {'post_cue_times':post_cue_times,'post_delivery_times':post_delivery_times,'cue_to_cue_times':cue_to_cue_times,'post_cue_dict':post_cue_dict,'post_delivery_dict':post_delivery_dict,'cue_cue_dict':cue_cue_dict,'size_dict':size_dict}

np.save('times_%s.npy' %(extracted_filename[:-4]),times_dict)

f = open('times.txt','w')
f.write('post_cue: ' + repr(post_cue_dict) + '\n')
f.write('post_delivery: ' + repr(post_delivery_dict) + '\n')
f.write('cue_cue: ' + repr(cue_cue_dict) + '\n')
f.write('size: ' + repr(size_dict) + '\n')
f.close()

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
        
M1_dicts = {'spikes':data_dict['M1_spikes'],'hist_all':data_dict_hist_all['M1_spikes_hist_dict']['hist_data'],'bins_all':data_dict_hist_all['M1_spikes_hist_dict']['hist_bins']} 
S1_dicts = {'spikes':data_dict['S1_spikes'],'hist_all':data_dict_hist_all['S1_spikes_hist_dict']['hist_data'],'bins_all':data_dict_hist_all['S1_spikes_hist_dict']['hist_bins']}
PmD_dicts = {'spikes':data_dict['PmD_spikes'],'hist_all':data_dict_hist_all['PmD_spikes_hist_dict']['hist_data'],'bins_all':data_dict_hist_all['PmD_spikes_hist_dict']['hist_bins']}

data_dict_all = {'M1_dicts':M1_dicts,'S1_dicts':S1_dicts,'PmD_dicts':PmD_dicts}


############
#short filenames
if extracted_filename == 'Extracted_504_2017-02-08-10-36-11.mat':
	short_filename = '5_8_1'
elif extracted_filename == 'Extracted_504_2017-02-08-11-02-03.mat':
	short_filename = '5_8_2'
elif extracted_filename == 'Extracted_504_2017-02-09-11-50-03.mat':
	short_filename = '5_9_1'
elif extracted_filename == 'Extracted_504_2017-02-09-12-15-57.mat':
	short_filename = '5_9_2'
elif extracted_filename == 'Extracted_504_2017-02-14-12-09-21.mat':
	short_filename = '5_14_1'
elif extracted_filename == 'Extracted_504_2017-02-14-12-35-41.mat':
	short_filename = '5_14_2'
elif extracted_filename == 'Extracted_504_2017-02-14-13-01-34.mat':
	short_filename = '5_14_3'

elif extracted_filename == 'Extracted_0059_2017-02-08-11-43-22.mat':
	short_filename = '0_8_1'
elif extracted_filename == 'Extracted_0059_2017-02-08-12-09-22.mat':
	short_filename = '0_8_2'
elif extracted_filename == 'Extracted_0059_2017-02-09-12-52-17.mat':
	short_filename = '0_9_1'
elif extracted_filename == 'Extracted_0059_2017-02-09-13-46-37.mat':
	short_filename = '0_9_2'


elif extracted_filename == 'Extracted_0059_2015-10-19-16-46-25.mat':
	short_filename = '0_1'
elif extracted_filename == 'Extracted_0059_2016-01-18-13-02-45.mat':
	short_filename = '0_2'
elif extracted_filename == 'Extracted_504_2015-09-29-12-48-19.mat':
	short_filename = '5_1'
elif extracted_filename == 'Extracted_504_2016-01-11-14-10-01.mat':
	short_filename = '5_2'

###
		
elif extracted_filename == 'Extracted_0059_2017-03-13-14-26-36.mat':
	short_filename = '0_3_13_1'
elif extracted_filename == 'Extracted_0059_2017-03-13-14-54-09.mat':
	short_filename = '0_3_13_2'
elif extracted_filename == 'Extracted_0059_2017-03-13-15-24-14.mat':
	short_filename = '0_3_13_3'
elif extracted_filename == 'Extracted_0059_2017-03-14-14-15-04.mat':
	short_filename = '0_3_14_1'
elif extracted_filename == 'Extracted_0059_2017-03-14-14-51-42.mat':
	short_filename = '0_3_14_2'
elif extracted_filename == 'Extracted_0059_2017-03-14-15-17-37.mat':
	short_filename = '0_3_14_3'
elif extracted_filename == 'Extracted_0059_2017-03-27-12-50-24.mat':
	short_filename = '0_3_27_1'
elif extracted_filename == 'Extracted_0059_2017-03-27-13-16-34.mat':
	short_filename = '0_3_27_2'
elif extracted_filename == 'Extracted_0059_2017-03-28-13-37-57.mat':
	short_filename = '0_3_28_1'
elif extracted_filename == 'Extracted_0059_2017-03-28-14-04-05.mat':
	short_filename = '0_3_28_2'
elif extracted_filename == 'Extracted_0059_2017-03-28-14-32-14.mat':
	short_filename = '0_3_28_3'

elif extracted_filename == 'Extracted_504_2017-03-13-12-50-41.mat':
	short_filename = '5_3_13_1'
elif extracted_filename == 'Extracted_504_2017-03-13-13-16-32.mat':
	short_filename = '5_3_13_2'
elif extracted_filename == 'Extracted_504_2017-03-13-13-42-37.mat':
	short_filename = '5_3_13_3'
elif extracted_filename == 'Extracted_504_2017-03-14-12-43-43.mat':
	short_filename = '5_3_14_1'
elif extracted_filename == 'Extracted_504_2017-03-14-13-09-46.mat':
	short_filename = '5_3_14_2'
elif extracted_filename == 'Extracted_504_2017-03-14-13-36-38.mat':
	short_filename = '5_3_14_3'
elif extracted_filename == 'Extracted_504_2017-03-28-12-27-38.mat':
	short_filename = '5_3_28_2'
elif extracted_filename == 'Extracted_504_2017-03-28-12-54-41.mat':
	short_filename = '5_3_28_3'

else:
	short_filename = 'NAME'

print 'calc firing rate'
for region_key,region_value in data_dict_all.iteritems():
        hists = data_dict_all[region_key]['hist_all']
        fr_dict = calc_firing_rates(hists,region_key,condensed)
        data_dict_all[region_key]['fr_dict'] = fr_dict

        bfr_cue = fr_dict['bfr_cue_fr'][10,:,:]
        aft_cue = fr_dict['aft_cue_fr'][10,:,:]
        bfr_result = fr_dict['bfr_result_fr'][10,:,:]
        aft_result = fr_dict['aft_result_fr'][10,:,:]

#print 'modeling'
for region_key,region_value in data_dict_all.iteritems():
        print 'modeling: %s' %(region_key)
        data_dict_all[region_key]['models'] = {}
        
        aft_cue = data_dict_all[region_key]['fr_dict']['aft_cue_fr']
        bfr_res = data_dict_all[region_key]['fr_dict']['bfr_result_fr']
        aft_res = data_dict_all[region_key]['fr_dict']['aft_result_fr']
        
        aft_cue_bins = np.shape(aft_cue)[2]
        bfr_res_bins = np.shape(bfr_res)[2]
        aft_res_bins = np.shape(aft_res)[2]

        concat = np.zeros((np.shape(aft_cue)[0],np.shape(aft_cue)[1],aft_cue_bins+bfr_res_bins+aft_res_bins))
        concat[:,:,0:aft_cue_bins] = aft_cue
        concat[:,:,aft_cue_bins:aft_cue_bins+bfr_res_bins] = bfr_res_bins
        concat[:,:,aft_cue_bins+bfr_res_bins:aft_cue_bins+bfr_res_bins+aft_res_bins] = aft_res_bins

        linear_aft_cue_model = make_lin_model(aft_cue,condensed,region_key,'aft_cue')
        linear_bfr_res_model = make_lin_model(bfr_res,condensed,region_key,'bfr_res')
        linear_aft_res_model = make_lin_model(aft_res,condensed,region_key,'aft_res')
        linear_concat_model = make_lin_model(concat,condensed,region_key,'concat')

        linear_model_return= {'aft_cue':linear_aft_cue_model,'bfr_res':linear_bfr_res_model,'aft_res':linear_aft_res_model,'concat':linear_concat_model}

        data_dict_all[region_key]['models']['linear'] = linear_model_return
        
        diff_aft_cue_model = make_diff_model(aft_cue,condensed,region_key,'aft_cue')
        diff_bfr_res_model = make_diff_model(bfr_res,condensed,region_key,'bfr_res')
        diff_aft_res_model = make_diff_model(aft_res,condensed,region_key,'aft_res')
        diff_concat_model = make_diff_model(concat,condensed,region_key,'concat')
        
        diff_model_return = {'aft_cue':diff_aft_cue_model,'bfr_res':diff_bfr_res_model,'aft_res':diff_aft_res_model,'concat':diff_concat_model}
        
        data_dict_all[region_key]['models']['diff'] = diff_model_return

        div_aft_cue_model = make_div_nl_model(aft_cue,condensed,region_key,'aft_cue')
        div_bfr_res_model = make_div_nl_model(bfr_res,condensed,region_key,'bfr_res')
        div_aft_res_model = make_div_nl_model(aft_res,condensed,region_key,'aft_res')
        div_concat_model = make_div_nl_model(concat,condensed,region_key,'concat')
        
        div_model_return = {'aft_cue':div_aft_cue_model,'bfr_res':div_bfr_res_model,'aft_res':div_aft_res_model,'concat':div_concat_model}
        
        data_dict_all[region_key]['models']['div_nl'] = div_model_return

        div_noe_aft_cue_model = make_div_nl_noe_model(aft_cue,condensed,region_key,'aft_cue')
        div_noe_bfr_res_model = make_div_nl_noe_model(bfr_res,condensed,region_key,'bfr_res')
        div_noe_aft_res_model = make_div_nl_noe_model(aft_res,condensed,region_key,'aft_res')
        div_noe_concat_model = make_div_nl_noe_model(concat,condensed,region_key,'concat')
        
        div_noe_model_return = {'aft_cue':div_noe_aft_cue_model,'bfr_res':div_noe_bfr_res_model,'aft_res':div_noe_aft_res_model,'concat':div_noe_concat_model}
        
        data_dict_all[region_key]['models']['div_noe_nl'] = div_noe_model_return

        div_Y_aft_cue_model = make_div_nl_Y_model(aft_cue,condensed,region_key,'aft_cue')
        div_Y_bfr_res_model = make_div_nl_Y_model(bfr_res,condensed,region_key,'bfr_res')
        div_Y_aft_res_model = make_div_nl_Y_model(aft_res,condensed,region_key,'aft_res')
        div_Y_concat_model = make_div_nl_Y_model(concat,condensed,region_key,'concat')
        
        div_Y_model_return = {'aft_cue':div_Y_aft_cue_model,'bfr_res':div_Y_bfr_res_model,'aft_res':div_Y_aft_res_model,'concat':div_Y_concat_model}
        
        data_dict_all[region_key]['models']['div_Y_nl'] = div_Y_model_return

        

np.save('model_save.npy',data_dict_all)

#if want to save anything as mat file to -> R
#sio.savemat('%s_%s_all_avg_mv_array.mat' %(region_key,type_key),{'mv_array':new_mv_array},format='5')
