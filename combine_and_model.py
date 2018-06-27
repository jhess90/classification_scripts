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

########################
# params to set ########
########################

rp_bool = True
alt_bool = False
uncued_bool = False

bin_size = 10 #in ms
time_before = -0.5 #negative value
time_after = 1.0
res_window = [0.3,0.7]

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

def make_lin_model(fr_data_dict,region_key,type_key,file_length):
        total_unit_num = 0
        total_trial_num = 0
        for i in range(file_length):
            total_unit_num += np.shape(fr_data_dict[i][type_key])[1]
            total_trial_num += np.shape(fr_data_dict[i]['condensed'])[0]

        #set each model
        num_params = 3

        unit_ct = 0
        fit_params = np.zeros((total_unit_num,num_params))
        cov_total = np.zeros((total_unit_num,num_params,num_params))
        perr_total = np.zeros((total_unit_num,num_params))
        AIC_total = np.zeros((total_unit_num))
        ss_res_total = np.zeros((total_unit_num))

        for i in range(file_length):
            fr_data = fr_data_dict[i][type_key]
            condensed = fr_data_dict[i]['condensed']
            avg_fr_data = np.mean(fr_data,axis=2)

            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

            for unit_num in range(fr_data.shape[1]):
                avg_frs = avg_fr_data[:,unit_num]
        
                #firing rate = a*R + b*P + c
                params,covs = curve_fit(lin_func,[r_vals,p_vals],avg_frs)
                
                if np.size(covs) > 1:
                        perr = np.sqrt(np.diag(covs))
                elif isinf(covs):
                        perr = float('nan')
                else:
                        pdb.set_trace()

                fit_params[unit_ct,:] = params
                cov_total[unit_ct,:,:] = covs
                perr_total[unit_ct,:] = perr

                model_out = lin_func([r_vals,p_vals],params[0],params[1],params[2])
                residuals = avg_frs - model_out
                ss_res = np.sum(residuals**2)
                k = num_params
                AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
                
                AIC_total[unit_ct] = AIC
                ss_res_total[unit_ct] = ss_res
                unit_ct +=1

            if i == 0:
                r_flat_total = np.tile(r_vals,fr_data.shape[1])
                p_flat_total = np.tile(p_vals,fr_data.shape[1])
                fr_flat_total = np.ndarray.flatten(avg_fr_data)
            else:
                r_flat_total = np.concatenate((r_flat_total,np.tile(r_vals,fr_data.shape[1])))
                p_flat_total = np.concatenate((p_flat_total,np.tile(p_vals,fr_data.shape[1])))
                fr_flat_total = np.concatenate((fr_flat_total,np.ndarray.flatten(avg_fr_data)))

        params,covs = curve_fit(lin_func,[r_flat_total,p_flat_total],fr_flat_total)
        
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
        
        AIC_overall = 2*k - 2*np.log(sum(ss_res_total))

        print '%s linear overall AIC %s' %(type_key,AIC_overall)

        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}        
        return_dict = {'fit_params':fit_params,'cov_total':cov_total,'perr_total':perr_total,'AIC_combined':AIC,'ss_res_total':ss_res_total,'combined':combined_dict,'AIC_overall':AIC_overall}
                
        return(return_dict)

def make_diff_model(fr_data_dict,region_key,type_key,file_length):
        total_unit_num = 0
        total_trial_num = 0
        for i in range(file_length):
            total_unit_num += np.shape(fr_data_dict[i][type_key])[1]
            total_trial_num += np.shape(fr_data_dict[i]['condensed'])[0]

        #set each model
        num_params = 2

        unit_ct = 0
        fit_params = np.zeros((total_unit_num,num_params))
        cov_total = np.zeros((total_unit_num,num_params,num_params))
        perr_total = np.zeros((total_unit_num,num_params))
        AIC_total = np.zeros((total_unit_num))
        ss_res_total = np.zeros((total_unit_num))

        for i in range(file_length):
            fr_data = fr_data_dict[i][type_key]
            condensed = fr_data_dict[i]['condensed']
            avg_fr_data = np.mean(fr_data,axis=2)

            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

            for unit_num in range(fr_data.shape[1]):
                avg_frs = avg_fr_data[:,unit_num]
        
                #firing rate = a + b(R - P)
                params,covs = curve_fit(diff_func,[r_vals,p_vals],avg_frs)
                
                if np.size(covs) > 1:
                        perr = np.sqrt(np.diag(covs))
                elif isinf(covs):
                        perr = float('nan')
                else:
                        pdb.set_trace()

                fit_params[unit_ct,:] = params
                cov_total[unit_ct,:,:] = covs
                perr_total[unit_ct,:] = perr

                model_out = diff_func([r_vals,p_vals],params[0],params[1])
                residuals = avg_frs - model_out
                ss_res = np.sum(residuals**2)
                k = num_params
                AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
                
                AIC_total[unit_ct] = AIC
                ss_res_total[unit_ct] = ss_res
                unit_ct +=1

            if i == 0:
                r_flat_total = np.tile(r_vals,fr_data.shape[1])
                p_flat_total = np.tile(p_vals,fr_data.shape[1])
                fr_flat_total = np.ndarray.flatten(avg_fr_data)
            else:
                r_flat_total = np.concatenate((r_flat_total,np.tile(r_vals,fr_data.shape[1])))
                p_flat_total = np.concatenate((p_flat_total,np.tile(p_vals,fr_data.shape[1])))
                fr_flat_total = np.concatenate((fr_flat_total,np.ndarray.flatten(avg_fr_data)))

        params,covs = curve_fit(diff_func,[r_flat_total,p_flat_total],fr_flat_total)
        
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
        
        AIC_overall = 2*k - 2*np.log(sum(ss_res_total))

        print '%s diff overall AIC %s' %(type_key,AIC_overall)

        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}        
        return_dict = {'fit_params':fit_params,'cov_total':cov_total,'perr_total':perr_total,'AIC_combined':AIC,'ss_res_total':ss_res_total,'combined':combined_dict,'AIC_overall':AIC_overall}
                
        return(return_dict)

def make_div_nl_model(fr_data_dict,region_key,type_key,file_length):
        total_unit_num = 0
        total_trial_num = 0
        for i in range(file_length):
            total_unit_num += np.shape(fr_data_dict[i][type_key])[1]
            total_trial_num += np.shape(fr_data_dict[i]['condensed'])[0]

        #set each model
        num_params = 5

        unit_ct = 0
        fit_params = np.zeros((total_unit_num,num_params))
        cov_total = np.zeros((total_unit_num,num_params,num_params))
        perr_total = np.zeros((total_unit_num,num_params))
        AIC_total = np.zeros((total_unit_num))
        ss_res_total = np.zeros((total_unit_num))

        for i in range(file_length):
            fr_data = fr_data_dict[i][type_key]
            condensed = fr_data_dict[i]['condensed']
            avg_fr_data = np.mean(fr_data,axis=2)

            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

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

                fit_params[unit_ct,:] = params
                cov_total[unit_ct,:,:] = covs
                perr_total[unit_ct,:] = perr

                model_out = div_nl_func([r_vals,p_vals],params[0],params[1],params[2],params[3],params[4])
                residuals = avg_frs - model_out
                ss_res = np.sum(residuals**2)
                k = num_params
                AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
                
                AIC_total[unit_ct] = AIC
                ss_res_total[unit_ct] = ss_res
                unit_ct +=1

            if i == 0:
                r_flat_total = np.tile(r_vals,fr_data.shape[1])
                p_flat_total = np.tile(p_vals,fr_data.shape[1])
                fr_flat_total = np.ndarray.flatten(avg_fr_data)
            else:
                r_flat_total = np.concatenate((r_flat_total,np.tile(r_vals,fr_data.shape[1])))
                p_flat_total = np.concatenate((p_flat_total,np.tile(p_vals,fr_data.shape[1])))
                fr_flat_total = np.concatenate((fr_flat_total,np.ndarray.flatten(avg_fr_data)))

        params,covs = curve_fit(div_nl_func,[r_flat_total,p_flat_total],fr_flat_total)
        
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
        
        AIC_overall = 2*k - 2*np.log(sum(ss_res_total))

        print '%s div nl overall AIC %s' %(type_key,AIC_overall)

        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}        
        return_dict = {'fit_params':fit_params,'cov_total':cov_total,'perr_total':perr_total,'AIC_combined':AIC,'ss_res_total':ss_res_total,'combined':combined_dict,'AIC_overall':AIC_overall}
                
        return(return_dict)

def make_div_nl_noe_model(fr_data_dict,region_key,type_key,file_length):
        total_unit_num = 0
        total_trial_num = 0
        for i in range(file_length):
            total_unit_num += np.shape(fr_data_dict[i][type_key])[1]
            total_trial_num += np.shape(fr_data_dict[i]['condensed'])[0]

        #set each model
        num_params = 4

        unit_ct = 0
        fit_params = np.zeros((total_unit_num,num_params))
        cov_total = np.zeros((total_unit_num,num_params,num_params))
        perr_total = np.zeros((total_unit_num,num_params))
        AIC_total = np.zeros((total_unit_num))
        ss_res_total = np.zeros((total_unit_num))

        for i in range(file_length):
            fr_data = fr_data_dict[i][type_key]
            condensed = fr_data_dict[i]['condensed']
            avg_fr_data = np.mean(fr_data,axis=2)

            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

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

                fit_params[unit_ct,:] = params
                cov_total[unit_ct,:,:] = covs
                perr_total[unit_ct,:] = perr

                model_out = div_nl_noe_func([r_vals,p_vals],params[0],params[1],params[2],params[3])
                residuals = avg_frs - model_out
                ss_res = np.sum(residuals**2)
                k = num_params
                AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
                
                AIC_total[unit_ct] = AIC
                ss_res_total[unit_ct] = ss_res
                unit_ct +=1

            if i == 0:
                r_flat_total = np.tile(r_vals,fr_data.shape[1])
                p_flat_total = np.tile(p_vals,fr_data.shape[1])
                fr_flat_total = np.ndarray.flatten(avg_fr_data)
            else:
                r_flat_total = np.concatenate((r_flat_total,np.tile(r_vals,fr_data.shape[1])))
                p_flat_total = np.concatenate((p_flat_total,np.tile(p_vals,fr_data.shape[1])))
                fr_flat_total = np.concatenate((fr_flat_total,np.ndarray.flatten(avg_fr_data)))

        params,covs = curve_fit(div_nl_noe_func,[r_flat_total,p_flat_total],fr_flat_total)
        
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
        
        AIC_overall = 2*k - 2*np.log(sum(ss_res_total))

        print '%s div_nl_noe overall AIC %s' %(type_key,AIC_overall)

        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}        
        return_dict = {'fit_params':fit_params,'cov_total':cov_total,'perr_total':perr_total,'AIC_combined':AIC,'ss_res_total':ss_res_total,'combined':combined_dict,'AIC_overall':AIC_overall}
                
        return(return_dict)

def make_div_nl_Y_model(fr_data_dict,region_key,type_key,file_length):
        total_unit_num = 0
        total_trial_num = 0
        for i in range(file_length):
            total_unit_num += np.shape(fr_data_dict[i][type_key])[1]
            total_trial_num += np.shape(fr_data_dict[i]['condensed'])[0]

        #set each model
        num_params = 4

        unit_ct = 0
        fit_params = np.zeros((total_unit_num,num_params))
        cov_total = np.zeros((total_unit_num,num_params,num_params))
        perr_total = np.zeros((total_unit_num,num_params))
        AIC_total = np.zeros((total_unit_num))
        ss_res_total = np.zeros((total_unit_num))

        for i in range(file_length):
            fr_data = fr_data_dict[i][type_key]
            condensed = fr_data_dict[i]['condensed']
            avg_fr_data = np.mean(fr_data,axis=2)

            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

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

                fit_params[unit_ct,:] = params
                cov_total[unit_ct,:,:] = covs
                perr_total[unit_ct,:] = perr

                model_out = div_nl_Y_func([r_vals,p_vals],params[0],params[1],params[2],params[3])
                residuals = avg_frs - model_out
                ss_res = np.sum(residuals**2)
                k = num_params
                AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
                
                AIC_total[unit_ct] = AIC
                ss_res_total[unit_ct] = ss_res
                unit_ct +=1

            if i == 0:
                r_flat_total = np.tile(r_vals,fr_data.shape[1])
                p_flat_total = np.tile(p_vals,fr_data.shape[1])
                fr_flat_total = np.ndarray.flatten(avg_fr_data)
            else:
                r_flat_total = np.concatenate((r_flat_total,np.tile(r_vals,fr_data.shape[1])))
                p_flat_total = np.concatenate((p_flat_total,np.tile(p_vals,fr_data.shape[1])))
                fr_flat_total = np.concatenate((fr_flat_total,np.ndarray.flatten(avg_fr_data)))

        params,covs = curve_fit(div_nl_Y_func,[r_flat_total,p_flat_total],fr_flat_total)
        
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
        
        AIC_overall = 2*k - 2*np.log(sum(ss_res_total))

        print '%s div nl Y overall AIC %s' %(type_key,AIC_overall)

        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}        
        return_dict = {'fit_params':fit_params,'cov_total':cov_total,'perr_total':perr_total,'AIC_combined':AIC,'ss_res_total':ss_res_total,'combined':combined_dict,'AIC_overall':AIC_overall}
                
        return(return_dict)



#########################
# run ###################
#########################

if (rp_bool + alt_bool + uncued_bool > 1):
    print 'ERROR selecting too many types'

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

    for region_key,region_value in data_dict_all.iteritems():
        hists = data_dict_all[region_key]['hist_all']
        fr_dict = calc_firing_rates(hists,region_key,condensed)
        data_dict_all[region_key]['fr_dict'] = fr_dict
        
        file_dict[file_ct] = {'aft_cue':fr_dict['aft_cue_fr'],'bfr_res':fr_dict['bfr_result_fr'],'aft_res':fr_dict['aft_result_fr'],'res_win':fr_dict['res_wind_fr'],'condensed':condensed,'concat':fr_dict['concat_fr']}

        data_dict_all[region_key]['file_dict'] = file_dict
    file_ct += 1
    

for region_key,region_val in data_dict_all.iteritems():
    print 'modeling: %s' %(region_key)
    
    data_dict_all[region_key]['models'] = {}

    file_dict = data_dict_all[region_key]['file_dict']
    file_length = np.shape(file_dict.keys())[0]

    lin_aft_cue_model = make_lin_model(file_dict,region_key,'aft_cue',file_length)
    lin_bfr_res_model = make_lin_model(file_dict,region_key,'bfr_res',file_length)
    lin_aft_res_model = make_lin_model(file_dict,region_key,'aft_res',file_length)
    lin_res_win_model = make_lin_model(file_dict,region_key,'res_win',file_length)
    lin_concat_model = make_lin_model(file_dict,region_key,'concat',file_length)

    linear_model_return= {'aft_cue':lin_aft_cue_model,'bfr_res':lin_bfr_res_model,'aft_res':lin_aft_res_model,'res_win':lin_res_win_model,'concat':lin_concat_model}
    
    data_dict_all[region_key]['models']['linear'] = linear_model_return

    diff_aft_cue_model = make_diff_model(file_dict,region_key,'aft_cue',file_length)
    diff_bfr_res_model = make_diff_model(file_dict,region_key,'bfr_res',file_length)
    diff_aft_res_model = make_diff_model(file_dict,region_key,'aft_res',file_length)
    diff_res_win_model = make_diff_model(file_dict,region_key,'res_win',file_length)
    diff_concat_model = make_diff_model(file_dict,region_key,'concat',file_length)

    diff_model_return= {'aft_cue':diff_aft_cue_model,'bfr_res':diff_bfr_res_model,'aft_res':diff_aft_res_model,'res_win':diff_res_win_model,'concat':diff_concat_model}
    
    data_dict_all[region_key]['models']['diff'] = diff_model_return

    div_nl_aft_cue_model = make_div_nl_model(file_dict,region_key,'aft_cue',file_length)
    div_nl_bfr_res_model = make_div_nl_model(file_dict,region_key,'bfr_res',file_length)
    div_nl_aft_res_model = make_div_nl_model(file_dict,region_key,'aft_res',file_length)
    div_nl_res_win_model = make_div_nl_model(file_dict,region_key,'res_win',file_length)
    div_nl_concat_model = make_div_nl_model(file_dict,region_key,'concat',file_length)

    div_nl_model_return= {'aft_cue':div_nl_aft_cue_model,'bfr_res':div_nl_bfr_res_model,'aft_res':div_nl_aft_res_model,'res_win':div_nl_res_win_model,'concat':div_nl_concat_model}
    
    data_dict_all[region_key]['models']['div_nl'] = div_nl_model_return

    div_nl_noe_aft_cue_model = make_div_nl_noe_model(file_dict,region_key,'aft_cue',file_length)
    div_nl_noe_bfr_res_model = make_div_nl_noe_model(file_dict,region_key,'bfr_res',file_length)
    div_nl_noe_aft_res_model = make_div_nl_noe_model(file_dict,region_key,'aft_res',file_length)
    div_nl_noe_res_win_model = make_div_nl_noe_model(file_dict,region_key,'res_win',file_length)
    div_nl_noe_concat_model = make_div_nl_noe_model(file_dict,region_key,'concat',file_length)

    div_nl_noe_model_return= {'aft_cue':div_nl_noe_aft_cue_model,'bfr_res':div_nl_noe_bfr_res_model,'aft_res':div_nl_noe_aft_res_model,'res_win':div_nl_noe_res_win_model,'concat':div_nl_noe_concat_model}
    
    data_dict_all[region_key]['models']['div_nl_noe'] = div_nl_noe_model_return

    div_nl_Y_aft_cue_model = make_div_nl_Y_model(file_dict,region_key,'aft_cue',file_length)
    div_nl_Y_bfr_res_model = make_div_nl_Y_model(file_dict,region_key,'bfr_res',file_length)
    div_nl_Y_aft_res_model = make_div_nl_Y_model(file_dict,region_key,'aft_res',file_length)
    div_nl_Y_res_win_model = make_div_nl_Y_model(file_dict,region_key,'res_win',file_length)
    div_nl_Y_concat_model = make_div_nl_Y_model(file_dict,region_key,'concat',file_length)

    div_nl_Y_model_return= {'aft_cue':div_nl_Y_aft_cue_model,'bfr_res':div_nl_Y_bfr_res_model,'aft_res':div_nl_Y_aft_res_model,'res_win':div_nl_Y_res_win_model,'concat':div_nl_Y_concat_model}
    
    data_dict_all[region_key]['models']['div_nl_Y'] = div_nl_Y_model_return

np.save('model_save.npy',data_dict_all)


type_list = list('aft_cue','bfr_res','aft_res','res_win','concat')







