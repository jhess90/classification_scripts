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

########################
# params to set ########
########################

rp_bool = True
alt_bool = False
uncued_bool = False

sig_only_bool = False

plot_bool = True

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

def div_nl_separate_add_func(x,a,b,c,d,e):
        #separate and add
        r,p = x
        return a*r / (r + b) + c*p / (p + d) + e

def div_nl_separate_multiply_func(x,a,b,c,d,e):
        #separate and multiply
        r,p = x
        return (a*r / (r + b)) * (c*p / (p + d)) + e

def plot_fr_and_model(unit_num,fr_r_p,params,model_type,region_key,type_key):

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        
        avg_frs = fr_r_p[:,0]
        r_vals = fr_r_p[:,1]
        p_vals = fr_r_p[:,2]

        if 2 in r_vals:
            r_range = [0,1,2,3]
            p_range = [0,1,2,3]
        else:
            r_range = [0,3]
            p_range = [0,3]

        avg_avg = np.zeros((np.shape(r_range)[0]*np.shape(p_range)[0],3))
        avg_std = np.zeros((np.shape(r_range)[0]*np.shape(p_range)[0],3))
        ct = 0
        for i in range(np.shape(r_range)[0]):
            for j in range(np.shape(p_range)[0]):
                r = r_range[i]
                p = p_range[j]

                temp = fr_r_p[np.logical_and(fr_r_p[:,1] == r,fr_r_p[:,2] == p),:]

                avg_avg[ct,:] = [np.mean(temp,axis=0)[0],r,p]
                avg_std[ct,:] = [np.mean(temp,axis=0)[0],r,p]

                ct += 1

        ax.scatter(avg_avg[:,1],avg_avg[:,2],avg_avg[:,0],c='purple',marker='o')
        
        pos_std = avg_avg + avg_std
        neg_std = avg_avg - avg_std
        
        for i in range(np.shape(pos_std)[0]):
            ax.plot([avg_avg[i,1],avg_avg[i,1]],[avg_avg[i,2],avg_avg[i,2]],[pos_std[i,0],neg_std[i,0]],marker='_',color='grey',linewidth=0.5)

        x_linspace = np.linspace(np.min(r_vals)-1,np.max(r_vals)+1)
        y_linspace = np.linspace(np.min(p_vals)-1,np.max(p_vals)+1)
        x,y = np.meshgrid(x_linspace,y_linspace)
        
        if model_type == 'linear':
            z = lin_func([x,y],params[0],params[1],params[2])
        elif model_type == 'diff':
            z = diff_func([x,y],params[0],params[1])
        elif model_type == 'div_nl':
            z = div_nl_func([x,y],params[0],params[1],params[2],params[3],params[4])
        elif model_type == 'div_nl_noe':
            z = div_nl_noe_func([x,y],params[0],params[1],params[2],params[3])
        elif model_type == 'div_nl_Y':
            z = div_nl_Y_func([x,y],params[0],params[1],params[2],params[3])
        elif model_type == 'div_nl_separate_add':
            z = div_nl_separate_add_func([x,y],params[0],params[1],params[2],params[3],params[4])
        elif model_type == 'div_nl_separate_multiply':
             z = div_nl_separate_multiply_func([x,y],params[0],params[1],params[2],params[3],params[4])

        surf = ax.plot_surface(x, y, z,cmap='RdBu_r',alpha=0.4,linewidth=0)

        plt.title('%s: %s, %s, unit %s' %(model_type,region_key,type_key,unit_num))

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

        plt.savefig('sig_plt_%s_%s_%s' %(region_key,type_key,str(unit_num).zfill(2)))
        plt.clf()

        return 

####################
# models ###########
####################

def make_lin_model(fr_data_dict,region_key,type_key,file_length,avg_and_corr):
        #set each model
        num_params = 3

        unit_ct = 0
        fit_params = np.zeros((total_unit_num,num_params))
        cov_total = np.zeros((total_unit_num,num_params,num_params))
        perr_total = np.zeros((total_unit_num,num_params))
        AIC_total = np.zeros((total_unit_num))
        ss_res_total = np.zeros((total_unit_num))
        sig_unit_length = 0
        num_units_no_fit = 0
        p_val_total = np.zeros((total_unit_num))
        F_val_total = np.zeros((total_unit_num))
        r_sq_total = np.zeros((total_unit_num))
        fr_r_p_list = []
    
        for i in range(file_length):
            condensed = fr_data_dict[i][region_key]['condensed']
            avg_fr_data = avg_and_corr[type_key]['avg_fr_dict'][i]
            
            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

            if sig_only_bool:
                #[r_coeff,r_pval,p_coeff,p_pval]
                sig_vals = avg_and_corr[type_key]['sig_vals_by_file'][i]
                
                either_sig = np.logical_or(sig_vals[:,1] < 0.05,sig_vals[:,3] < 0.05)
                sig_unit_length += np.sum(either_sig)
                avg_fr_data = avg_fr_data[:,either_sig]
            else:
                sig_unit_length += np.shape(avg_fr_data)[1]

            for unit_num in range(avg_fr_data.shape[1]):
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

                ss_tot = np.sum((avg_frs-np.mean(avg_frs))**2)
                r_sq = 1 - (ss_res / ss_tot)

                ss_model = ss_tot - ss_res
                n = np.shape(avg_frs)[0]
                k = num_params #repeat for now
                #degrees of freedom
                df_total = n - 1
                df_residual = n - k
                df_model = k - 1

                #mean squares
                ms_residual = ss_res / float(df_residual)
                ms_model = ss_model / float(df_model)

                F_val = ms_model / ms_residual
                p_val = 1.0 - stats.f.cdf(F_val,df_residual,df_model)

                AIC_total[unit_ct] = AIC
                ss_res_total[unit_ct] = ss_res
                p_val_total[unit_ct] = p_val
                F_val_total[unit_ct] = F_val
                r_sq_total[unit_ct] = r_sq

                #here add summation array: avg fr, r, and p, by unit
                fr_r_p_list.append(np.transpose(np.array((avg_frs,r_vals,p_vals))))

                unit_ct +=1

            if i == 0:
                r_flat_total = np.tile(r_vals,avg_fr_data.shape[1])
                p_flat_total = np.tile(p_vals,avg_fr_data.shape[1])
                fr_flat_total = np.ndarray.flatten(avg_fr_data)
            else:
                r_flat_total = np.concatenate((r_flat_total,np.tile(r_vals,avg_fr_data.shape[1])))
                p_flat_total = np.concatenate((p_flat_total,np.tile(p_vals,avg_fr_data.shape[1])))
                fr_flat_total = np.concatenate((fr_flat_total,np.ndarray.flatten(avg_fr_data)))

        params,covs = curve_fit(lin_func,[r_flat_total,p_flat_total],fr_flat_total)
        
        if np.size(covs) > 1:
            perr = np.sqrt(np.diag(covs))
        elif isinf(covs):
            perr = float('nan')
        else:
            pdb.set_trace()

        model_out_total = lin_func([r_flat_total,p_flat_total],params[0],params[1],params[2])
        residuals = fr_flat_total - model_out_total
        ss_res = np.sum(residuals**2)
        k = num_params
        AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
        
        AIC_overall = 2*k - 2*np.log(sum(ss_res_total))

        #print '%s linear overall AIC %s' %(type_key,AIC_overall)

        num_sig_fit = np.sum(p_val_total < 0.05)
        if num_sig_fit > 0:
            AIC_sig_model = 2*k - 2*np.log(sum(ss_res_total[p_val_total < 0.05]))
            AIC_sig_avg = np.mean(AIC_total[p_val_total < 0.05])

            for i in range(num_sig_fit):
                sig_units = np.where((p_val_total < 0.05))
                unit = sig_units[0][i]
            
                if unit < unit_ct:
                    try:
                        if plot_bool:
                            plot_fr_and_model(unit,fr_r_p_list[unit],fit_params[unit],'linear',region_key,type_key)
                    except:
                        pdb.set_trace()
        else:
            AIC_sig_model = 1000
            AIC_sig_avg = 1000

        #this will give different values though depending on number of sig fits. That ok?
        #Or calc AICs for each unit then average? What's kosher
        
        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}        
        return_dict = {'fit_params':fit_params[0:sig_unit_length,:],'cov_total':cov_total[0:sig_unit_length,:],'perr_total':perr_total[0:sig_unit_length,:],'AIC_combined':AIC,'ss_res_total':np.trim_zeros(ss_res_total,'b'),'AIC_total':AIC_total[0:sig_unit_length],'combined':combined_dict,'AIC_overall':AIC_overall,'num_units_no_fit':num_units_no_fit,'perc_units_no_fit':num_units_no_fit/float(sig_unit_length),'total_num_units':sig_unit_length,'p_val_total':p_val_total[0:sig_unit_length],'F_val_total':F_val_total[0:sig_unit_length],'r_sq_total':r_sq_total[0:sig_unit_length],'AIC_sig_model':AIC_sig_model,'AIC_sig_avg':AIC_sig_avg,'num_sig_fit':num_sig_fit}

        return(return_dict)

def make_diff_model(fr_data_dict,region_key,type_key,file_length,avg_and_corr):
        #set each model
        num_params = 2

        unit_ct = 0
        fit_params = np.zeros((total_unit_num,num_params))
        cov_total = np.zeros((total_unit_num,num_params,num_params))
        perr_total = np.zeros((total_unit_num,num_params))
        AIC_total = np.zeros((total_unit_num))
        ss_res_total = np.zeros((total_unit_num))
        sig_unit_length = 0
        num_units_no_fit = 0
        p_val_total = np.zeros((total_unit_num))
        F_val_total = np.zeros((total_unit_num))
        r_sq_total = np.zeros((total_unit_num))
        fr_r_p_list = []


        for i in range(file_length):
            condensed = fr_data_dict[i][region_key]['condensed']
            avg_fr_data = avg_and_corr[type_key]['avg_fr_dict'][i]
            
            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

            if sig_only_bool:
                #[r_coeff,r_pval,p_coeff,p_pval]
                sig_vals = avg_and_corr[type_key]['sig_vals_by_file'][i]
                
                either_sig = np.logical_or(sig_vals[:,1] < 0.05,sig_vals[:,3] < 0.05)
                sig_unit_length += np.sum(either_sig)                
                avg_fr_data = avg_fr_data[:,either_sig]
            else:
                sig_unit_length += np.shape(avg_fr_data)[1]

            for unit_num in range(avg_fr_data.shape[1]):
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
                
                ss_tot = np.sum((avg_frs-np.mean(avg_frs))**2)
                r_sq = 1 - (ss_res / ss_tot)

                ss_model = ss_tot - ss_res
                n = np.shape(avg_frs)[0]
                k = num_params #repeat for now
                #degrees of freedom
                df_total = n - 1
                df_residual = n - k
                df_model = k - 1

                #mean squares
                ms_residual = ss_res / float(df_residual)
                ms_model = ss_model / float(df_model)

                F_val = ms_model / ms_residual
                p_val = 1.0 - stats.f.cdf(F_val,df_residual,df_model)

                AIC_total[unit_ct] = AIC
                ss_res_total[unit_ct] = ss_res
                p_val_total[unit_ct] = p_val
                F_val_total[unit_ct] = F_val
                r_sq_total[unit_ct] = r_sq

                #here add summation array: avg fr, r, and p, by unit
                fr_r_p_list.append(np.transpose(np.array((avg_frs,r_vals,p_vals))))

                unit_ct +=1

            if i == 0:
                r_flat_total = np.tile(r_vals,avg_fr_data.shape[1])
                p_flat_total = np.tile(p_vals,avg_fr_data.shape[1])
                fr_flat_total = np.ndarray.flatten(avg_fr_data)
            else:
                r_flat_total = np.concatenate((r_flat_total,np.tile(r_vals,avg_fr_data.shape[1])))
                p_flat_total = np.concatenate((p_flat_total,np.tile(p_vals,avg_fr_data.shape[1])))
                fr_flat_total = np.concatenate((fr_flat_total,np.ndarray.flatten(avg_fr_data)))

        params,covs = curve_fit(diff_func,[r_flat_total,p_flat_total],fr_flat_total)
        
        if np.size(covs) > 1:
            perr = np.sqrt(np.diag(covs))
        elif isinf(covs):
            perr = float('nan')
        else:
            pdb.set_trace()

        model_out_total = diff_func([r_flat_total,p_flat_total],params[0],params[1])
        residuals = fr_flat_total - model_out_total
        ss_res = np.sum(residuals**2)
        k = num_params
        AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
        
        AIC_overall = 2*k - 2*np.log(sum(ss_res_total))

        #print '%s diff overall AIC %s' %(type_key,AIC_overall)

        num_sig_fit = np.sum(p_val_total < 0.05)

        if num_sig_fit > 0:
            AIC_sig_model = 2*k - 2*np.log(sum(ss_res_total[p_val_total < 0.05]))
            AIC_sig_avg = np.mean(AIC_total[p_val_total < 0.05])

            for i in range(num_sig_fit):
                sig_units = np.where((p_val_total < 0.05))
                unit = sig_units[0][i]
            
                if unit < unit_ct:
                    if plot_bool:
                        plot_fr_and_model(unit,fr_r_p_list[unit],fit_params[unit],'diff',region_key,type_key)

        else:
            AIC_sig_model = 1000
            AIC_sig_avg = 1000

        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}        
        return_dict = {'fit_params':fit_params[0:sig_unit_length,:],'cov_total':cov_total[0:sig_unit_length,:],'perr_total':perr_total[0:sig_unit_length,:],'AIC_combined':AIC,'ss_res_total':np.trim_zeros(ss_res_total,'b'),'AIC_total':AIC_total[0:sig_unit_length],'combined':combined_dict,'AIC_overall':AIC_overall,'num_units_no_fit':num_units_no_fit,'perc_units_no_fit':num_units_no_fit/float(sig_unit_length),'total_num_units':sig_unit_length,'p_val_total':p_val_total[0:sig_unit_length],'F_val_total':F_val_total[0:sig_unit_length],'r_sq_total':r_sq_total[0:sig_unit_length],'AIC_sig_model':AIC_sig_model,'AIC_sig_avg':AIC_sig_avg,'num_sig_fit':num_sig_fit}
                
        return(return_dict)

def make_div_nl_model(fr_data_dict,region_key,type_key,file_length,avg_and_corr):
        #set each model
        num_params = 5

        unit_ct = 0
        fit_params = np.zeros((total_unit_num,num_params))
        cov_total = np.zeros((total_unit_num,num_params,num_params))
        perr_total = np.zeros((total_unit_num,num_params))
        AIC_total = np.zeros((total_unit_num))
        ss_res_total = np.zeros((total_unit_num))
        sig_unit_length = 0
        num_units_no_fit = 0
        p_val_total = np.zeros((total_unit_num))
        F_val_total = np.zeros((total_unit_num))
        r_sq_total = np.zeros((total_unit_num))
        fr_r_p_list = []

        for i in range(file_length):
            condensed = fr_data_dict[i][region_key]['condensed']
            avg_fr_data = avg_and_corr[type_key]['avg_fr_dict'][i]
            
            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

            if sig_only_bool:
                #[r_coeff,r_pval,p_coeff,p_pval]
                sig_vals = avg_and_corr[type_key]['sig_vals_by_file'][i]
                
                either_sig = np.logical_or(sig_vals[:,1] < 0.05,sig_vals[:,3] < 0.05)
                sig_unit_length += np.sum(either_sig)                
                avg_fr_data = avg_fr_data[:,either_sig]
            else:
                sig_unit_length += np.shape(avg_fr_data)[1]

            for unit_num in range(avg_fr_data.shape[1]):
                avg_frs = avg_fr_data[:,unit_num]

                try:
                        #firing rate  = (R * a + b * P) / (a + c * b + d) + e
                        params,covs = curve_fit(div_nl_func,[r_vals,p_vals],avg_frs)
                except:
                        #fake very high values
                        AIC_total[unit_ct] = 1000
                        ss_res_total[unit_ct] = 1000
                        num_units_no_fit += 1
                        fr_r_p_list.append(np.array((0,0,0)))

                        p_val_total[unit_ct] = 1
                        F_val_total[unit_ct] = 1000
                        r_sq_total[unit_ct] = 1000

                        unit_ct += 1
                        #print 'failure to fit div nl %s %s unit %s' %(region_key,type_key,unit_num)
                        continue

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
                
                ss_tot = np.sum((avg_frs-np.mean(avg_frs))**2)
                r_sq = 1 - (ss_res / ss_tot)

                ss_model = ss_tot - ss_res
                n = np.shape(avg_frs)[0]
                k = num_params #repeat for now
                #degrees of freedom
                df_total = n - 1
                df_residual = n - k
                df_model = k - 1

                #mean squares
                ms_residual = ss_res / float(df_residual)
                ms_model = ss_model / float(df_model)

                F_val = ms_model / ms_residual
                p_val = 1.0 - stats.f.cdf(F_val,df_residual,df_model)

                AIC_total[unit_ct] = AIC
                ss_res_total[unit_ct] = ss_res
                p_val_total[unit_ct] = p_val
                F_val_total[unit_ct] = F_val
                r_sq_total[unit_ct] = r_sq
                
                #here add summation array: avg fr, r, and p, by unit
                fr_r_p_list.append(np.transpose(np.array((avg_frs,r_vals,p_vals))))
 
                unit_ct +=1

            if i == 0:
                r_flat_total = np.tile(r_vals,avg_fr_data.shape[1])
                p_flat_total = np.tile(p_vals,avg_fr_data.shape[1])
                fr_flat_total = np.ndarray.flatten(avg_fr_data)
            else:
                r_flat_total = np.concatenate((r_flat_total,np.tile(r_vals,avg_fr_data.shape[1])))
                p_flat_total = np.concatenate((p_flat_total,np.tile(p_vals,avg_fr_data.shape[1])))
                fr_flat_total = np.concatenate((fr_flat_total,np.ndarray.flatten(avg_fr_data)))

        params,covs = curve_fit(div_nl_func,[r_flat_total,p_flat_total],fr_flat_total)
        
        if np.size(covs) > 1:
            perr = np.sqrt(np.diag(covs))
        elif isinf(covs):
            perr = float('nan')
        else:
            pdb.set_trace()
            
        #########
        #CHANGE ALL fixed
        model_out_total = div_nl_func([r_flat_total,p_flat_total],params[0],params[1],params[2],params[3],params[4])
        residuals = fr_flat_total - model_out_total
        ss_res = np.sum(residuals**2)
        k = num_params
        AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
        
        AIC_overall = 2*k - 2*np.log(sum(ss_res_total))

        #print '%s div nl overall AIC %s' %(type_key,AIC_overall)

        num_sig_fit = np.sum(p_val_total < 0.05)
        if num_sig_fit > 0:
            AIC_sig_model = 2*k - 2*np.log(sum(ss_res_total[p_val_total < 0.05]))
            AIC_sig_avg = np.mean(AIC_total[p_val_total < 0.05])

            for i in range(num_sig_fit):
                sig_units = np.where((p_val_total < 0.05))
                unit = sig_units[0][i]
            
                if unit < unit_ct:
                    if plot_bool:
                        plot_fr_and_model(unit,fr_r_p_list[unit],fit_params[unit],'div_nl',region_key,type_key)

        else:
            AIC_sig_model = 1000
            AIC_sig_avg = 1000


        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}        
        return_dict = {'fit_params':fit_params[0:sig_unit_length,:],'cov_total':cov_total[0:sig_unit_length,:],'perr_total':perr_total[0:sig_unit_length,:],'AIC_combined':AIC,'ss_res_total':np.trim_zeros(ss_res_total,'b'),'AIC_total':AIC_total[0:sig_unit_length],'combined':combined_dict,'AIC_overall':AIC_overall,'num_units_no_fit':num_units_no_fit,'perc_units_no_fit':num_units_no_fit/float(sig_unit_length),'total_num_units':sig_unit_length,'p_val_total':p_val_total[0:sig_unit_length],'F_val_total':F_val_total[0:sig_unit_length],'r_sq_total':r_sq_total[0:sig_unit_length],'AIC_sig_model':AIC_sig_model,'AIC_sig_avg':AIC_sig_avg,'num_sig_fit':num_sig_fit}
                
        return(return_dict)

def make_div_nl_noe_model(fr_data_dict,region_key,type_key,file_length,avg_and_corr):
        #set each model
        num_params = 4

        unit_ct = 0
        fit_params = np.zeros((total_unit_num,num_params))
        cov_total = np.zeros((total_unit_num,num_params,num_params))
        perr_total = np.zeros((total_unit_num,num_params))
        AIC_total = np.zeros((total_unit_num))
        ss_res_total = np.zeros((total_unit_num))
        sig_unit_length = 0
        num_units_no_fit = 0
        p_val_total = np.zeros((total_unit_num))
        F_val_total = np.zeros((total_unit_num))
        r_sq_total = np.zeros((total_unit_num))
        fr_r_p_list = []

        for i in range(file_length):
            condensed = fr_data_dict[i][region_key]['condensed']
            avg_fr_data = avg_and_corr[type_key]['avg_fr_dict'][i]
            
            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

            if sig_only_bool:
                #[r_coeff,r_pval,p_coeff,p_pval]
                sig_vals = avg_and_corr[type_key]['sig_vals_by_file'][i]
                
                either_sig = np.logical_or(sig_vals[:,1] < 0.05,sig_vals[:,3] < 0.05)
                sig_unit_length += np.sum(either_sig)                
                avg_fr_data = avg_fr_data[:,either_sig]
            else:
                sig_unit_length += np.shape(avg_fr_data)[1]

            for unit_num in range(avg_fr_data.shape[1]):
                avg_frs = avg_fr_data[:,unit_num]
        
                try:
                        #firing rate  = (R * a + b * P) / (a + c * b + d)
                        params,covs = curve_fit(div_nl_noe_func,[r_vals,p_vals],avg_frs)
                except:
                        #fake very high values
                        AIC_total[unit_ct] = 1000
                        ss_res_total[unit_ct] = 1000
                        num_units_no_fit += 1
                        fr_r_p_list.append(np.array((0,0,0)))

                        p_val_total[unit_ct] = 1
                        F_val_total[unit_ct] = 1000
                        r_sq_total[unit_ct] = 1000

                        unit_ct += 1

                        #print 'failure to fit div nl noe %s %s unit %s' %(region_key,type_key,unit_num)
                        continue

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
                
                ss_tot = np.sum((avg_frs-np.mean(avg_frs))**2)
                r_sq = 1 - (ss_res / ss_tot)

                ss_model = ss_tot - ss_res
                n = np.shape(avg_frs)[0]
                k = num_params #repeat for now
                #degrees of freedom
                df_total = n - 1
                df_residual = n - k
                df_model = k - 1

                #mean squares
                ms_residual = ss_res / float(df_residual)
                ms_model = ss_model / float(df_model)

                F_val = ms_model / ms_residual
                p_val = 1.0 - stats.f.cdf(F_val,df_residual,df_model)

                AIC_total[unit_ct] = AIC
                ss_res_total[unit_ct] = ss_res
                p_val_total[unit_ct] = p_val
                F_val_total[unit_ct] = F_val
                r_sq_total[unit_ct] = r_sq

                #here add summation array: avg fr, r, and p, by unit
                fr_r_p_list.append(np.transpose(np.array((avg_frs,r_vals,p_vals))))
                
                unit_ct +=1

            if i == 0:
                r_flat_total = np.tile(r_vals,avg_fr_data.shape[1])
                p_flat_total = np.tile(p_vals,avg_fr_data.shape[1])
                fr_flat_total = np.ndarray.flatten(avg_fr_data)
            else:
                r_flat_total = np.concatenate((r_flat_total,np.tile(r_vals,avg_fr_data.shape[1])))
                p_flat_total = np.concatenate((p_flat_total,np.tile(p_vals,avg_fr_data.shape[1])))
                fr_flat_total = np.concatenate((fr_flat_total,np.ndarray.flatten(avg_fr_data)))

        params,covs = curve_fit(div_nl_noe_func,[r_flat_total,p_flat_total],fr_flat_total)
        
        if np.size(covs) > 1:
            perr = np.sqrt(np.diag(covs))
        elif isinf(covs):
            perr = float('nan')
        else:
            pdb.set_trace()

        model_out_total = div_nl_noe_func([r_flat_total,p_flat_total],params[0],params[1],params[2],params[3])
        residuals = fr_flat_total - model_out_total
        ss_res = np.sum(residuals**2)
        k = num_params
        AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
        
        AIC_overall = 2*k - 2*np.log(sum(ss_res_total))

        #print '%s div_nl_noe overall AIC %s' %(type_key,AIC_overall)

        num_sig_fit = np.sum(p_val_total < 0.05)
        if num_sig_fit > 0:
            AIC_sig_model = 2*k - 2*np.log(sum(ss_res_total[p_val_total < 0.05]))
            AIC_sig_avg = np.mean(AIC_total[p_val_total < 0.05])

            for i in range(num_sig_fit):
                sig_units = np.where((p_val_total < 0.05))
                unit = sig_units[0][i]
            
                if unit < unit_ct:
                    if plot_bool:
                        plot_fr_and_model(unit,fr_r_p_list[unit],fit_params[unit],'div_nl_noe',region_key,type_key)

        else:
            AIC_sig_model = 1000
            AIC_sig_avg = 1000


        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}        
        return_dict = {'fit_params':fit_params[0:sig_unit_length,:],'cov_total':cov_total[0:sig_unit_length,:],'perr_total':perr_total[0:sig_unit_length,:],'AIC_combined':AIC,'ss_res_total':np.trim_zeros(ss_res_total,'b'),'AIC_total':AIC_total[0:sig_unit_length],'combined':combined_dict,'AIC_overall':AIC_overall,'num_units_no_fit':num_units_no_fit,'perc_units_no_fit':num_units_no_fit/float(sig_unit_length),'total_num_units':sig_unit_length,'p_val_total':p_val_total[0:sig_unit_length],'F_val_total':F_val_total[0:sig_unit_length],'r_sq_total':r_sq_total[0:sig_unit_length],'AIC_sig_model':AIC_sig_model,'AIC_sig_avg':AIC_sig_avg,'num_sig_fit':num_sig_fit}
                
        return(return_dict)

def make_div_nl_Y_model(fr_data_dict,region_key,type_key,file_length,avg_and_corr):
        #set each model
        num_params = 4

        unit_ct = 0
        fit_params = np.zeros((total_unit_num,num_params))
        cov_total = np.zeros((total_unit_num,num_params,num_params))
        perr_total = np.zeros((total_unit_num,num_params))
        AIC_total = np.zeros((total_unit_num))
        ss_res_total = np.zeros((total_unit_num))
        sig_unit_length = 0
        num_units_no_fit = 0
        p_val_total = np.zeros((total_unit_num))
        F_val_total = np.zeros((total_unit_num))
        r_sq_total = np.zeros((total_unit_num))
        fr_r_p_list = []

        for i in range(file_length):
            condensed = fr_data_dict[i][region_key]['condensed']
            avg_fr_data = avg_and_corr[type_key]['avg_fr_dict'][i]
            
            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

            if sig_only_bool:
                #[r_coeff,r_pval,p_coeff,p_pval]
                sig_vals = avg_and_corr[type_key]['sig_vals_by_file'][i]
                
                either_sig = np.logical_or(sig_vals[:,1] < 0.05,sig_vals[:,3] < 0.05)
                sig_unit_length += np.sum(either_sig)
                avg_fr_data = avg_fr_data[:,either_sig]
            else:
                sig_unit_length += np.shape(avg_fr_data)[1]

            for unit_num in range(avg_fr_data.shape[1]):
                avg_frs = avg_fr_data[:,unit_num]
        
                # model = a*(r + b*P) / (c + r + b * p) + d
                try:
                        params,covs = curve_fit(div_nl_Y_func,[r_vals,p_vals],avg_frs)
                except:
                        #fake very high values
                        AIC_total[unit_ct] = 1000
                        ss_res_total[unit_ct] = 1000
                        fr_r_p_list.append(np.array((0,0,0)))

                        num_units_no_fit += 1

                        p_val_total[unit_ct] = 1
                        F_val_total[unit_ct] = 1000
                        r_sq_total[unit_ct] = 1000

                        unit_ct += 1
                        #print 'failure to fit Y nl %s %s unit %s' %(region_key,type_key,unit_num)
                        continue
                
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
                
                ss_tot = np.sum((avg_frs-np.mean(avg_frs))**2)
                r_sq = 1 - (ss_res / ss_tot)

                ss_model = ss_tot - ss_res
                n = np.shape(avg_frs)[0]
                k = num_params #repeat for now
                #degrees of freedom
                df_total = n - 1
                df_residual = n - k
                df_model = k - 1

                #mean squares
                ms_residual = ss_res / float(df_residual)
                ms_model = ss_model / float(df_model)

                F_val = ms_model / ms_residual
                p_val = 1.0 - stats.f.cdf(F_val,df_residual,df_model)

                try:
                    AIC_total[unit_ct] = AIC
                    ss_res_total[unit_ct] = ss_res
                    p_val_total[unit_ct] = p_val
                    F_val_total[unit_ct] = F_val
                    r_sq_total[unit_ct] = r_sq
                except:
                    pdb.set_trace()
                    
                #here add summation array: avg fr, r, and p, by unit
                fr_r_p_list.append(np.transpose(np.array((avg_frs,r_vals,p_vals))))

                unit_ct +=1

            if i == 0:
                r_flat_total = np.tile(r_vals,avg_fr_data.shape[1])
                p_flat_total = np.tile(p_vals,avg_fr_data.shape[1])
                fr_flat_total = np.ndarray.flatten(avg_fr_data)
            else:
                r_flat_total = np.concatenate((r_flat_total,np.tile(r_vals,avg_fr_data.shape[1])))
                p_flat_total = np.concatenate((p_flat_total,np.tile(p_vals,avg_fr_data.shape[1])))
                fr_flat_total = np.concatenate((fr_flat_total,np.ndarray.flatten(avg_fr_data)))

        try:
                params,covs = curve_fit(div_nl_Y_func,[r_flat_total,p_flat_total],fr_flat_total)
        except:
                print 'failure to fit div nl Y on all data %s %s' %(region_key,type_key)

                combined_dict = {'params':0,'covs':0,'perr':0,'ss_res':0,'AIC':0}        
                return_dict = {'fit_params':fit_params,'cov_total':cov_total,'perr_total':perr_total,'AIC_combined':0,'ss_res_total':ss_res_total,'combined':combined_dict,'AIC_overall':0}


        if np.size(covs) > 1:
            perr = np.sqrt(np.diag(covs))
        elif isinf(covs):
            perr = float('nan')
        else:
            pdb.set_trace()

        model_out_total = div_nl_Y_func([r_flat_total,p_flat_total],params[0],params[1],params[2],params[3])
        residuals = fr_flat_total - model_out_total
        ss_res = np.sum(residuals**2)
        k = num_params
        AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
        
        AIC_overall = 2*k - 2*np.log(sum(ss_res_total))

        #print '%s div nl Y overall AIC %s' %(type_key,AIC_overall)

        num_sig_fit = np.sum(p_val_total < 0.05)
        if num_sig_fit > 0:
            AIC_sig_model = 2*k - 2*np.log(sum(ss_res_total[p_val_total < 0.05]))
            AIC_sig_avg = np.mean(AIC_total[p_val_total < 0.05])

            for i in range(num_sig_fit):
                sig_units = np.where((p_val_total < 0.05))
                unit = sig_units[0][i]
            
                if unit < unit_ct:
                    if plot_bool:
                        plot_fr_and_model(unit,fr_r_p_list[unit],fit_params[unit],'div_nl_Y',region_key,type_key)

        else:
            AIC_sig_model = 1000
            AIC_sig_avg = 1000


        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}        
        return_dict = {'fit_params':fit_params[0:sig_unit_length,:],'cov_total':cov_total[0:sig_unit_length,:],'perr_total':perr_total[0:sig_unit_length,:],'AIC_combined':AIC,'ss_res_total':np.trim_zeros(ss_res_total,'b'),'AIC_total':AIC_total[0:sig_unit_length],'combined':combined_dict,'AIC_overall':AIC_overall,'num_units_no_fit':num_units_no_fit,'perc_units_no_fit':num_units_no_fit/float(sig_unit_length),'total_num_units':sig_unit_length,'p_val_total':p_val_total[0:sig_unit_length],'F_val_total':F_val_total[0:sig_unit_length],'r_sq_total':r_sq_total[0:sig_unit_length],'AIC_sig_model':AIC_sig_model,'AIC_sig_avg':AIC_sig_avg,'num_sig_fit':num_sig_fit}
                
        return(return_dict)

def make_div_nl_separate_add_model(fr_data_dict,region_key,type_key,file_length,avg_and_corr):
        #set each model
        num_params = 5

        unit_ct = 0
        fit_params = np.zeros((total_unit_num,num_params))
        cov_total = np.zeros((total_unit_num,num_params,num_params))
        perr_total = np.zeros((total_unit_num,num_params))
        AIC_total = np.zeros((total_unit_num))
        ss_res_total = np.zeros((total_unit_num))
        sig_unit_length = 0
        num_units_no_fit = 0
        p_val_total = np.zeros((total_unit_num))
        F_val_total = np.zeros((total_unit_num))
        r_sq_total = np.zeros((total_unit_num))
        fr_r_p_list = []

        for i in range(file_length):
            condensed = fr_data_dict[i][region_key]['condensed']
            avg_fr_data = avg_and_corr[type_key]['avg_fr_dict'][i]
            
            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

            if sig_only_bool:
                #[r_coeff,r_pval,p_coeff,p_pval]
                sig_vals = avg_and_corr[type_key]['sig_vals_by_file'][i]
                
                either_sig = np.logical_or(sig_vals[:,1] < 0.05,sig_vals[:,3] < 0.05)
                sig_unit_length += np.sum(either_sig)       
                avg_fr_data = avg_fr_data[:,either_sig]
            else:
                sig_unit_length += np.shape(avg_fr_data)[1]

            for unit_num in range(avg_fr_data.shape[1]):
                avg_frs = avg_fr_data[:,unit_num]
                
                try:
                        params,covs = curve_fit(div_nl_separate_add_func,[r_vals,p_vals],avg_frs)
                except:
                        AIC_total[unit_ct] = 1000
                        ss_res_total[unit_ct] = 1000
                        fr_r_p_list.append(np.array((0,0,0)))

                        num_units_no_fit += 1
                        p_val_total[unit_ct] = 1
                        F_val_total[unit_ct] = 1000
                        r_sq_total[unit_ct] = 1000

                        unit_ct += 1
                        #print 'failure to fit separate add nl %s %s unit %s' %(region_key,type_key,unit_num)
                        continue

                if np.size(covs) > 1:
                        perr = np.sqrt(np.diag(covs))
                elif isinf(covs):
                        perr = float('nan')
                else:
                        pdb.set_trace()

                fit_params[unit_ct,:] = params
                cov_total[unit_ct,:,:] = covs
                perr_total[unit_ct,:] = perr

                model_out = div_nl_separate_add_func([r_vals,p_vals],params[0],params[1],params[2],params[3],params[4])
                residuals = avg_frs - model_out
                ss_res = np.sum(residuals**2)
                k = num_params
                AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
                
                ss_tot = np.sum((avg_frs-np.mean(avg_frs))**2)
                r_sq = 1 - (ss_res / ss_tot)

                ss_model = ss_tot - ss_res
                n = np.shape(avg_frs)[0]
                k = num_params #repeat for now
                #degrees of freedom
                df_total = n - 1
                df_residual = n - k
                df_model = k - 1

                #mean squares
                ms_residual = ss_res / float(df_residual)
                ms_model = ss_model / float(df_model)

                F_val = ms_model / ms_residual
                p_val = 1.0 - stats.f.cdf(F_val,df_residual,df_model)

                AIC_total[unit_ct] = AIC
                ss_res_total[unit_ct] = ss_res
                p_val_total[unit_ct] = p_val
                F_val_total[unit_ct] = F_val
                r_sq_total[unit_ct] = r_sq

                #here add summation array: avg fr, r, and p, by unit
                fr_r_p_list.append(np.transpose(np.array((avg_frs,r_vals,p_vals))))

                unit_ct +=1

            if i == 0:
                r_flat_total = np.tile(r_vals,avg_fr_data.shape[1])
                p_flat_total = np.tile(p_vals,avg_fr_data.shape[1])
                fr_flat_total = np.ndarray.flatten(avg_fr_data)
            else:
                r_flat_total = np.concatenate((r_flat_total,np.tile(r_vals,avg_fr_data.shape[1])))
                p_flat_total = np.concatenate((p_flat_total,np.tile(p_vals,avg_fr_data.shape[1])))
                fr_flat_total = np.concatenate((fr_flat_total,np.ndarray.flatten(avg_fr_data)))

        params,covs = curve_fit(div_nl_separate_add_func,[r_flat_total,p_flat_total],fr_flat_total)
        
        if np.size(covs) > 1:
            perr = np.sqrt(np.diag(covs))
        elif isinf(covs):
            perr = float('nan')
        else:
            pdb.set_trace()

        model_out_total = div_nl_separate_add_func([r_flat_total,p_flat_total],params[0],params[1],params[2],params[3],params[4])
        residuals = fr_flat_total - model_out_total
        ss_res = np.sum(residuals**2)
        k = num_params
        AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
        
        AIC_overall = 2*k - 2*np.log(sum(ss_res_total))

        #print '%s div nl separate add overall AIC %s' %(type_key,AIC_overall)

        num_sig_fit = np.sum(p_val_total < 0.05)
        if num_sig_fit > 0:
            AIC_sig_model = 2*k - 2*np.log(sum(ss_res_total[p_val_total < 0.05]))
            AIC_sig_avg = np.mean(AIC_total[p_val_total < 0.05])

            for i in range(num_sig_fit):
                sig_units = np.where((p_val_total < 0.05))
                unit = sig_units[0][i]
            
                if unit < unit_ct:
                    if plot_bool:
                        plot_fr_and_model(unit,fr_r_p_list[unit],fit_params[unit],'div_nl_separate_add',region_key,type_key)

        else:
            AIC_sig_model = 1000
            AIC_sig_avg = 1000


        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}        
        return_dict = {'fit_params':fit_params[0:sig_unit_length,:],'cov_total':cov_total[0:sig_unit_length,:],'perr_total':perr_total[0:sig_unit_length,:],'AIC_combined':AIC,'ss_res_total':np.trim_zeros(ss_res_total,'b'),'AIC_total':AIC_total[0:sig_unit_length],'combined':combined_dict,'AIC_overall':AIC_overall,'num_units_no_fit':num_units_no_fit,'perc_units_no_fit':num_units_no_fit/float(sig_unit_length),'total_num_units':sig_unit_length,'p_val_total':p_val_total[0:sig_unit_length],'F_val_total':F_val_total[0:sig_unit_length],'r_sq_total':r_sq_total[0:sig_unit_length],'AIC_sig_model':AIC_sig_model,'AIC_sig_avg':AIC_sig_avg,'num_sig_fit':num_sig_fit}
                
        return(return_dict)

def make_div_nl_separate_multiply_model(fr_data_dict,region_key,type_key,file_length,avg_and_corr):
        #set each model
        num_params = 5

        unit_ct = 0
        fit_params = np.zeros((total_unit_num,num_params))
        cov_total = np.zeros((total_unit_num,num_params,num_params))
        perr_total = np.zeros((total_unit_num,num_params))
        AIC_total = np.zeros((total_unit_num))
        ss_res_total = np.zeros((total_unit_num))
        sig_unit_length = 0
        num_units_no_fit = 0
        p_val_total = np.zeros((total_unit_num))
        F_val_total = np.zeros((total_unit_num))
        r_sq_total = np.zeros((total_unit_num))
        fr_r_p_list = []

        for i in range(file_length):
            fr_data = fr_data_dict[i][region_key][type_key]
            condensed = fr_data_dict[i][region_key]['condensed']
            avg_fr_data = np.mean(fr_data,axis=2)

            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

            condensed = fr_data_dict[i][region_key]['condensed']
            avg_fr_data = avg_and_corr[type_key]['avg_fr_dict'][i]
            
            r_vals = condensed[:,3]
            p_vals = condensed[:,4]

            if sig_only_bool:
                #[r_coeff,r_pval,p_coeff,p_pval]
                sig_vals = avg_and_corr[type_key]['sig_vals_by_file'][i]
                
                either_sig = np.logical_or(sig_vals[:,1] < 0.05,sig_vals[:,3] < 0.05)
                sig_unit_length += np.sum(either_sig)                
                avg_fr_data = avg_fr_data[:,either_sig]
            else:
                sig_unit_length += np.shape(avg_fr_data)[1]

            for unit_num in range(avg_fr_data.shape[1]):
                avg_frs = avg_fr_data[:,unit_num]

                try:
                        params,covs = curve_fit(div_nl_separate_multiply_func,[r_vals,p_vals],avg_frs)
                except:
                        AIC_total[unit_num] = 1000
                        ss_res_total[unit_num] = 1000
                        fr_r_p_list.append(np.array((0,0,0)))

                        num_units_no_fit += 1
                        p_val_total[unit_ct] = 1
                        F_val_total[unit_ct] = 1000
                        r_sq_total[unit_ct] = 1000

                        unit_ct += 1
                        #print 'failure to fit separate mult nl %s %s unit %s' %(region_key,type_key,unit_num)
                        continue

                if np.size(covs) > 1:
                        perr = np.sqrt(np.diag(covs))
                elif isinf(covs):
                        perr = float('nan')
                else:
                        pdb.set_trace()

                fit_params[unit_ct,:] = params
                cov_total[unit_ct,:,:] = covs
                perr_total[unit_ct,:] = perr

                model_out = div_nl_separate_multiply_func([r_vals,p_vals],params[0],params[1],params[2],params[3],params[4])
                residuals = avg_frs - model_out
                ss_res = np.sum(residuals**2)
                k = num_params
                AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
                
                ss_tot = np.sum((avg_frs-np.mean(avg_frs))**2)
                r_sq = 1 - (ss_res / ss_tot)

                ss_model = ss_tot - ss_res
                n = np.shape(avg_frs)[0]
                k = num_params #repeat for now
                #degrees of freedom
                df_total = n - 1
                df_residual = n - k
                df_model = k - 1

                #mean squares
                ms_residual = ss_res / float(df_residual)
                ms_model = ss_model / float(df_model)

                F_val = ms_model / ms_residual
                p_val = 1.0 - stats.f.cdf(F_val,df_residual,df_model)

                AIC_total[unit_ct] = AIC
                ss_res_total[unit_ct] = ss_res
                p_val_total[unit_ct] = p_val
                F_val_total[unit_ct] = F_val
                r_sq_total[unit_ct] = r_sq

                #here add summation array: avg fr, r, and p, by unit
                fr_r_p_list.append(np.transpose(np.array((avg_frs,r_vals,p_vals))))

                unit_ct +=1

            if i == 0:
                r_flat_total = np.tile(r_vals,avg_fr_data.shape[1])
                p_flat_total = np.tile(p_vals,avg_fr_data.shape[1])
                fr_flat_total = np.ndarray.flatten(avg_fr_data)
            else:
                r_flat_total = np.concatenate((r_flat_total,np.tile(r_vals,avg_fr_data.shape[1])))
                p_flat_total = np.concatenate((p_flat_total,np.tile(p_vals,avg_fr_data.shape[1])))
                fr_flat_total = np.concatenate((fr_flat_total,np.ndarray.flatten(avg_fr_data)))

        params,covs = curve_fit(div_nl_separate_multiply_func,[r_flat_total,p_flat_total],fr_flat_total)
        
        if np.size(covs) > 1:
            perr = np.sqrt(np.diag(covs))
        elif isinf(covs):
            perr = float('nan')
        else:
            pdb.set_trace()

        model_out_total = div_nl_separate_multiply([r_flat_total,p_flat_total],params[0],params[1],params[2],params[3],params[4])
        residuals = fr_flat_total - model_out_total
        ss_res = np.sum(residuals**2)
        k = num_params
        AIC = 2*k - 2*np.log(ss_res)  #np.log(x) = ln(x)
        
        AIC_overall = 2*k - 2*np.log(sum(ss_res_total))

        #print '%s div nl separate multiply overall AIC %s' %(type_key,AIC_overall)
        
        num_sig_fit = np.sum(p_val_total < 0.05)
        if num_sig_fit > 0:
            AIC_sig_model = 2*k - 2*np.log(sum(ss_res_total[p_val_total < 0.05]))
            AIC_sig_avg = np.mean(AIC_total[p_val_total < 0.05])

            for i in range(num_sig_fit):
                sig_units = np.where((p_val_total < 0.05))
                unit = sig_units[0][i]
            
                if unit < unit_ct:
                    if plot_bool:
                        plot_fr_and_model(unit,fr_r_p_list[unit],fit_params[unit],'div_nl_separate_multiply',region_key,type_key)

        else:
            AIC_sig_model = 1000
            AIC_sig_avg = 1000


        combined_dict = {'params':params,'covs':covs,'perr':perr,'ss_res':ss_res,'AIC':AIC}        
        return_dict = {'fit_params':fit_params[0:sig_unit_length,:],'cov_total':cov_total[0:sig_unit_length,:],'perr_total':perr_total[0:sig_unit_length,:],'AIC_combined':AIC,'ss_res_total':np.trim_zeros(ss_res_total,'b'),'AIC_total':AIC_total[0:sig_unit_length],'combined':combined_dict,'AIC_overall':AIC_overall,'num_units_no_fit':num_units_no_fit,'perc_units_no_fit':num_units_no_fit/float(sig_unit_length),'total_num_units':sig_unit_length,'p_val_total':p_val_total[0:sig_unit_length],'F_val_total':F_val_total[0:sig_unit_length],'r_sq_total':r_sq_total[0:sig_unit_length],'AIC_sig_model':AIC_sig_model,'AIC_sig_avg':AIC_sig_avg,'num_sig_fit':num_sig_fit}
                
        return(return_dict)




#########################
# run ###################
#########################

if (rp_bool + alt_bool + uncued_bool > 1):
    print 'ERROR selecting too many types'

if sig_only_bool:
    if os.path.isfile('model_aic_sig.xlsx'):
        os.remove('model_aic_sig.xlsx')
    if os.path.isfile('model_params_sig.xlsx'):
        os.remove('model_params_sig.xlsx')
else:
    if os.path.isfile('model_aic.xlsx'):
        os.remove('model_aic.xlsx')
    if os.path.isfile('model_params.xlsx'):
        os.remove('model_params.xlsx')

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

    file_dict[file_ct] = {}

    for region_key,region_value in data_dict_all.iteritems():
        hists = data_dict_all[region_key]['hist_all']
        fr_dict = calc_firing_rates(hists,region_key,condensed)
        data_dict_all[region_key]['fr_dict'] = fr_dict
        
        file_dict[file_ct][region_key] = {'aft_cue':fr_dict['aft_cue_fr'],'bfr_res':fr_dict['bfr_result_fr'],'aft_res':fr_dict['aft_result_fr'],'res_win':fr_dict['res_wind_fr'],'condensed':condensed,'concat':fr_dict['concat_fr']}

    file_ct += 1
    

########
# run correlations

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
            r_p_all = np.zeros((total_trial_num,2))

            avg_fr_dict = {}
            unit_ct = 0
            trial_ct = 0
            sig_vals_by_file = {}
            
            for file_ind in range(file_length):
                frs = file_dict[file_ind][region_key][type_key]
                condensed = file_dict[file_ind][region_key]['condensed']
                r_vals = condensed[:,3]
                p_vals = condensed[:,4]
                sig_vals_temp = np.zeros((np.shape(frs)[1],4))


                for unit_ind in range(frs.shape[1]):
                    avg_fr = np.mean(frs[:,unit_ind,:],axis=1)
                    r_coeff,r_pval = pearsonr(avg_fr,r_vals)
                    p_coeff,p_pval = pearsonr(avg_fr,p_vals)
                
                    sig_vals_temp[unit_ind,:] = [r_coeff,r_pval,p_coeff,p_pval]
                    sig_vals[unit_ct + unit_ind,:] = [r_coeff,r_pval,p_coeff,p_pval]
                 
                unit_ct += frs.shape[1]
                r_p_all[trial_ct : trial_ct + np.shape(r_vals)[0],:] = np.column_stack((r_vals,p_vals))
                trial_ct += np.shape(r_vals)[0]
                avg_fr_dict[file_ind] = np.mean(frs,axis=2)
                sig_vals_by_file[file_ind] = sig_vals_temp
                
            save_dict = {'avg_fr_dict':avg_fr_dict,'r_p_all':r_p_all,'sig_vals':sig_vals,'sig_vals_by_file':sig_vals_by_file}
            data_dict_all[region_key]['avg_and_corr'][type_key] = save_dict

#run modeling
for region_key,region_val in data_dict_all.iteritems():
    print 'modeling: %s' %(region_key)
    
    data_dict_all[region_key]['models'] = {}

    #file_dict = data_dict_all[region_key]['file_dict']
    file_length = np.shape(file_dict.keys())[0]
    avg_and_corr = data_dict_all[region_key]['avg_and_corr']

    print 'linear'
    lin_aft_cue_model = make_lin_model(file_dict,region_key,'aft_cue',file_length,avg_and_corr)
    lin_bfr_res_model = make_lin_model(file_dict,region_key,'bfr_res',file_length,avg_and_corr)
    lin_aft_res_model = make_lin_model(file_dict,region_key,'aft_res',file_length,avg_and_corr)
    lin_res_win_model = make_lin_model(file_dict,region_key,'res_win',file_length,avg_and_corr)
    lin_concat_model = make_lin_model(file_dict,region_key,'concat',file_length,avg_and_corr)

    linear_model_return= {'aft_cue':lin_aft_cue_model,'bfr_res':lin_bfr_res_model,'aft_res':lin_aft_res_model,'res_win':lin_res_win_model,'concat':lin_concat_model}
    
    data_dict_all[region_key]['models']['linear'] = linear_model_return

    print 'diff'
    diff_aft_cue_model = make_diff_model(file_dict,region_key,'aft_cue',file_length,avg_and_corr)
    diff_bfr_res_model = make_diff_model(file_dict,region_key,'bfr_res',file_length,avg_and_corr)
    diff_aft_res_model = make_diff_model(file_dict,region_key,'aft_res',file_length,avg_and_corr)
    diff_res_win_model = make_diff_model(file_dict,region_key,'res_win',file_length,avg_and_corr)
    diff_concat_model = make_diff_model(file_dict,region_key,'concat',file_length,avg_and_corr)

    diff_model_return= {'aft_cue':diff_aft_cue_model,'bfr_res':diff_bfr_res_model,'aft_res':diff_aft_res_model,'res_win':diff_res_win_model,'concat':diff_concat_model}
    
    data_dict_all[region_key]['models']['diff'] = diff_model_return

    print 'div nl'   
    div_nl_aft_cue_model = make_div_nl_model(file_dict,region_key,'aft_cue',file_length,avg_and_corr)
    div_nl_bfr_res_model = make_div_nl_model(file_dict,region_key,'bfr_res',file_length,avg_and_corr)
    div_nl_aft_res_model = make_div_nl_model(file_dict,region_key,'aft_res',file_length,avg_and_corr)
    div_nl_res_win_model = make_div_nl_model(file_dict,region_key,'res_win',file_length,avg_and_corr)
    div_nl_concat_model = make_div_nl_model(file_dict,region_key,'concat',file_length,avg_and_corr)

    div_nl_model_return= {'aft_cue':div_nl_aft_cue_model,'bfr_res':div_nl_bfr_res_model,'aft_res':div_nl_aft_res_model,'res_win':div_nl_res_win_model,'concat':div_nl_concat_model}
    
    data_dict_all[region_key]['models']['div_nl'] = div_nl_model_return

    print 'div nl noe'   
    div_nl_noe_aft_cue_model = make_div_nl_noe_model(file_dict,region_key,'aft_cue',file_length,avg_and_corr)
    div_nl_noe_bfr_res_model = make_div_nl_noe_model(file_dict,region_key,'bfr_res',file_length,avg_and_corr)
    div_nl_noe_aft_res_model = make_div_nl_noe_model(file_dict,region_key,'aft_res',file_length,avg_and_corr)
    div_nl_noe_res_win_model = make_div_nl_noe_model(file_dict,region_key,'res_win',file_length,avg_and_corr)
    div_nl_noe_concat_model = make_div_nl_noe_model(file_dict,region_key,'concat',file_length,avg_and_corr)

    div_nl_noe_model_return= {'aft_cue':div_nl_noe_aft_cue_model,'bfr_res':div_nl_noe_bfr_res_model,'aft_res':div_nl_noe_aft_res_model,'res_win':div_nl_noe_res_win_model,'concat':div_nl_noe_concat_model}
    
    data_dict_all[region_key]['models']['div_nl_noe'] = div_nl_noe_model_return

    print 'div nl Y'   
    div_nl_Y_aft_cue_model = make_div_nl_Y_model(file_dict,region_key,'aft_cue',file_length,avg_and_corr)
    div_nl_Y_bfr_res_model = make_div_nl_Y_model(file_dict,region_key,'bfr_res',file_length,avg_and_corr)
    div_nl_Y_aft_res_model = make_div_nl_Y_model(file_dict,region_key,'aft_res',file_length,avg_and_corr)
    div_nl_Y_res_win_model = make_div_nl_Y_model(file_dict,region_key,'res_win',file_length,avg_and_corr)
    div_nl_Y_concat_model = make_div_nl_Y_model(file_dict,region_key,'concat',file_length,avg_and_corr)

    div_nl_Y_model_return= {'aft_cue':div_nl_Y_aft_cue_model,'bfr_res':div_nl_Y_bfr_res_model,'aft_res':div_nl_Y_aft_res_model,'res_win':div_nl_Y_res_win_model,'concat':div_nl_Y_concat_model}
    
    data_dict_all[region_key]['models']['div_nl_Y'] = div_nl_Y_model_return

    print 'div nl separate add'   
    div_nl_separate_add_aft_cue_model = make_div_nl_separate_add_model(file_dict,region_key,'aft_cue',file_length,avg_and_corr)
    div_nl_separate_add_bfr_res_model = make_div_nl_separate_add_model(file_dict,region_key,'bfr_res',file_length,avg_and_corr)
    div_nl_separate_add_aft_res_model = make_div_nl_separate_add_model(file_dict,region_key,'aft_res',file_length,avg_and_corr)
    div_nl_separate_add_res_win_model = make_div_nl_separate_add_model(file_dict,region_key,'res_win',file_length,avg_and_corr)
    div_nl_separate_add_concat_model = make_div_nl_separate_add_model(file_dict,region_key,'concat',file_length,avg_and_corr)

    div_nl_separate_add_model_return= {'aft_cue':div_nl_separate_add_aft_cue_model,'bfr_res':div_nl_separate_add_bfr_res_model,'aft_res':div_nl_separate_add_aft_res_model,'res_win':div_nl_separate_add_res_win_model,'concat':div_nl_separate_add_concat_model}
    
    data_dict_all[region_key]['models']['div_nl_separate_add'] = div_nl_separate_add_model_return

    print 'div nl separate multiply'   
    div_nl_separate_multiply_aft_cue_model = make_div_nl_separate_multiply_model(file_dict,region_key,'aft_cue',file_length,avg_and_corr)
    div_nl_separate_multiply_bfr_res_model = make_div_nl_separate_multiply_model(file_dict,region_key,'bfr_res',file_length,avg_and_corr)
    div_nl_separate_multiply_aft_res_model = make_div_nl_separate_multiply_model(file_dict,region_key,'aft_res',file_length,avg_and_corr)
    div_nl_separate_multiply_res_win_model = make_div_nl_separate_multiply_model(file_dict,region_key,'res_win',file_length,avg_and_corr)
    div_nl_separate_multiply_concat_model = make_div_nl_separate_multiply_model(file_dict,region_key,'concat',file_length,avg_and_corr)

    div_nl_separate_multiply_model_return= {'aft_cue':div_nl_separate_multiply_aft_cue_model,'bfr_res':div_nl_separate_multiply_bfr_res_model,'aft_res':div_nl_separate_multiply_aft_res_model,'res_win':div_nl_separate_multiply_res_win_model,'concat':div_nl_separate_multiply_concat_model}
    
    data_dict_all[region_key]['models']['div_nl_separate_multiply'] = div_nl_separate_multiply_model_return



##saving 
if sig_only_bool:
    np.save('model_save_sig.npy',data_dict_all)
else:
    np.save('model_save.npy',data_dict_all)
data_dict_all['file_dict'] = file_dict


#for region_key,region_val in data_dict_all.iteritems():
type_names = ['aft cue','bfr res','aft res','res win','concat']
model_names = ['linear','difference','div nl','div nl noe','div nl Y','separate add','separate multiply']

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
        all_no_fit = np.zeros((np.shape(type_names)[0],np.shape(model_names)[0]))
        total_num_units_window = np.zeros((np.shape(type_names)[0]))

        data_dict_all[region_key]['model_summary'] = {}
        for i in range(np.shape(type_names_together)[0]):
            type_key = type_names_together[i]
    
            all_model_AICs = np.column_stack((data_dict_all[region_key]['models']['linear'][type_key]['AIC_total'],data_dict_all[region_key]['models']['diff'][type_key]['AIC_total'],data_dict_all[region_key]['models']['div_nl'][type_key]['AIC_total'],data_dict_all[region_key]['models']['div_nl_noe'][type_key]['AIC_total'],data_dict_all[region_key]['models']['div_nl_Y'][type_key]['AIC_total'],data_dict_all[region_key]['models']['div_nl_separate_add'][type_key]['AIC_total'],data_dict_all[region_key]['models']['div_nl_separate_multiply'][type_key]['AIC_total']))

            arg_mins = np.argmin(all_model_AICs,axis=1)
            perc_min = np.zeros((np.shape(all_model_AICs)[1]))
            for j in range(np.shape(all_model_AICs)[1]):
                perc_min[j] = np.sum(arg_mins == j) / float(np.shape(all_model_AICs)[0])

            all_percs[i,:] = perc_min
            
            all_no_fit[i,:] = [data_dict_all[region_key]['models']['linear'][type_key]['perc_units_no_fit'],data_dict_all[region_key]['models']['diff'][type_key]['perc_units_no_fit'],data_dict_all[region_key]['models']['div_nl'][type_key]['perc_units_no_fit'],data_dict_all[region_key]['models']['div_nl_noe'][type_key]['perc_units_no_fit'],data_dict_all[region_key]['models']['div_nl_Y'][type_key]['perc_units_no_fit'],data_dict_all[region_key]['models']['div_nl_separate_add'][type_key]['perc_units_no_fit'],data_dict_all[region_key]['models']['div_nl_separate_multiply'][type_key]['perc_units_no_fit']]

            total_num_units_window[i] = data_dict_all[region_key]['models']['linear'][type_key]['total_num_units']

        data_dict_all[region_key]['model_summary'] = {'all_percs':all_percs,'all_no_fit':all_no_fit,'total_num_units_window':total_num_units_window}



worksheet = aic_workbook.add_worksheet('perc_unit_best_fit')
worksheet.write(0,0,'M1')
worksheet.write_row(0,1,model_names)
worksheet.write_column(1,0,type_names)
percs = data_dict_all['M1_dicts']['model_summary']['all_percs']
total_units = data_dict_all['M1_dicts']['model_summary']['total_num_units_window']
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+1,1,percs[i,:])
worksheet.write(0,8,'total units')
worksheet.write_column(1,8,total_units)

worksheet.write(7,0,'S1')
worksheet.write_row(7,1,model_names)
worksheet.write_column(8,0,type_names)
percs = data_dict_all['S1_dicts']['model_summary']['all_percs']
total_units = data_dict_all['S1_dicts']['model_summary']['total_num_units_window']
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+8,1,percs[i,:])
worksheet.write(7,8,'total units')
worksheet.write_column(8,8,total_units)

worksheet.write(14,0,'PMd')
worksheet.write_row(14,1,model_names)
worksheet.write_column(15,0,type_names)
percs = data_dict_all['PmD_dicts']['model_summary']['all_percs']
total_units = data_dict_all['S1_dicts']['model_summary']['total_num_units_window']
for i in range(np.shape(percs)[0]):
    worksheet.write_row(i+15,1,percs[i,:])
worksheet.write(14,8,'total units')
worksheet.write_column(15,8,total_units)

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

lin_num_sig = [data_dict_all['M1_dicts']['models']['linear']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['linear']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['linear']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['linear']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['linear']['concat']['num_sig_fit']]
diff_num_sig = [data_dict_all['M1_dicts']['models']['diff']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['diff']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['diff']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['diff']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['diff']['concat']['num_sig_fit']]
div_nl_num_sig = [data_dict_all['M1_dicts']['models']['div_nl']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl']['concat']['num_sig_fit']]
div_nl_noe_num_sig = [data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_noe']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_noe']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_noe']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_noe']['concat']['num_sig_fit']]
div_nl_Y_num_sig = [data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_Y']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_Y']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_Y']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_Y']['concat']['num_sig_fit']]
div_nl_separate_add_num_sig = [data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_add']['concat']['num_sig_fit']]
div_nl_separate_multiply_num_sig = [data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['aft_res']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['res_win']['num_sig_fit'],data_dict_all['M1_dicts']['models']['div_nl_separate_multiply']['concat']['num_sig_fit']]

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

worksheet.write(0,6,'num sig')
worksheet.write_row(0,7,type_names)
worksheet.write_row(1,7,lin_num_sig)
worksheet.write_row(2,7,diff_num_sig)
worksheet.write_row(3,7,div_nl_num_sig)
worksheet.write_row(4,7,div_nl_noe_num_sig)
worksheet.write_row(5,7,div_nl_Y_num_sig)
worksheet.write_row(6,7,div_nl_separate_add_num_sig)
worksheet.write_row(7,7,div_nl_separate_multiply_num_sig)
#worksheet.write_column(1,6,model_names)

worksheet.conditional_format('B2:B8', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('C2:C8', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('D2:D8', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('E2:E8', {'type': 'bottom', 'value':'1', 'format':format1})
worksheet.conditional_format('F2:F8', {'type': 'bottom', 'value':'1', 'format':format1})

lin_aics = [data_dict_all['S1_dicts']['models']['linear']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['linear']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['linear']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['linear']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['linear']['concat']['AIC_sig_avg']]
diff_aics = [data_dict_all['S1_dicts']['models']['diff']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['diff']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['diff']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['diff']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['diff']['concat']['AIC_sig_avg']]
div_nl_aics = [data_dict_all['S1_dicts']['models']['div_nl']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl']['concat']['AIC_sig_avg']]
div_nl_noe_aics = [data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_noe']['concat']['AIC_sig_avg']]
div_nl_Y_aics = [data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_Y']['concat']['AIC_sig_avg']]
div_nl_separate_add_aics = [data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['concat']['AIC_sig_avg']]
div_nl_separate_multiply_aics = [data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_res']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['res_win']['AIC_sig_avg'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['concat']['AIC_sig_avg']]

lin_num_sig = [data_dict_all['S1_dicts']['models']['linear']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['linear']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['linear']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['linear']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['linear']['concat']['num_sig_fit']]
diff_num_sig = [data_dict_all['S1_dicts']['models']['diff']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['diff']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['diff']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['diff']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['diff']['concat']['num_sig_fit']]
div_nl_num_sig = [data_dict_all['S1_dicts']['models']['div_nl']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl']['concat']['num_sig_fit']]
div_nl_noe_num_sig = [data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_noe']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_noe']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_noe']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_noe']['concat']['num_sig_fit']]
div_nl_Y_num_sig = [data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_Y']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_Y']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_Y']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_Y']['concat']['num_sig_fit']]
div_nl_separate_add_num_sig = [data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_add']['concat']['num_sig_fit']]
div_nl_separate_multiply_num_sig = [data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_cue']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['bfr_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['aft_res']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['res_win']['num_sig_fit'],data_dict_all['S1_dicts']['models']['div_nl_separate_multiply']['concat']['num_sig_fit']]

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

lin_aics = [data_dict_all['PmD_dicts']['models']['linear']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['linear']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['linear']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['linear']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['linear']['concat']['AIC_sig_avg']]
diff_aics = [data_dict_all['PmD_dicts']['models']['diff']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['diff']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['diff']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['diff']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['diff']['concat']['AIC_sig_avg']]
div_nl_aics = [data_dict_all['PmD_dicts']['models']['div_nl']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl']['concat']['AIC_sig_avg']]
div_nl_noe_aics = [data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['concat']['AIC_sig_avg']]
div_nl_Y_aics = [data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['concat']['AIC_sig_avg']]
div_nl_separate_add_aics = [data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['concat']['AIC_sig_avg']]
div_nl_separate_multiply_aics = [data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_cue']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['bfr_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_res']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['res_win']['AIC_sig_avg'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['concat']['AIC_sig_avg']]

lin_num_sig = [data_dict_all['PmD_dicts']['models']['linear']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['linear']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['linear']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['linear']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['linear']['concat']['num_sig_fit']]
diff_num_sig = [data_dict_all['PmD_dicts']['models']['diff']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['diff']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['diff']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['diff']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['diff']['concat']['num_sig_fit']]
div_nl_num_sig = [data_dict_all['PmD_dicts']['models']['div_nl']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl']['concat']['num_sig_fit']]
div_nl_noe_num_sig = [data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_noe']['concat']['num_sig_fit']]
div_nl_Y_num_sig = [data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_Y']['concat']['num_sig_fit']]
div_nl_separate_add_num_sig = [data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_add']['concat']['num_sig_fit']]
div_nl_separate_multiply_num_sig = [data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_cue']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['bfr_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['aft_res']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['res_win']['num_sig_fit'],data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply']['concat']['num_sig_fit']]


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







###########
if sig_only_bool:
    param_workbook = xlsxwriter.Workbook('model_params_sig.xlsx',options={'nan_inf_to_errors':True})
else:
    param_workbook = xlsxwriter.Workbook('model_params.xlsx',options={'nan_inf_to_errors':True})

worksheet = param_workbook.add_worksheet('model_params')

param_names = ['a:mean','max','min','b:mean','max','min','c:mean','max','min','d:mean','max','min','e:mean','max','min']

#temp = np.full([7,75],np.nan)
temp = np.zeros((7,75))

for i in range(5):
    if i ==0:
        wind = 'aft_cue'
    elif i == 1:
        wind = 'bfr_res'
    elif i == 2:
        wind = 'aft_res'
    elif i == 3:
        wind = 'res_win'
    elif i == 4:
        wind = 'concat'

    ct = i*15

    temp[0,[0+ct,3+ct,6+ct]] = np.mean(data_dict_all['M1_dicts']['models']['linear'][wind]['fit_params'],axis=0)
    temp[0,[1+ct,4+ct,7+ct]] = np.max(data_dict_all['M1_dicts']['models']['linear'][wind]['fit_params'],axis=0)
    temp[0,[2+ct,5+ct,8+ct]] = np.min(data_dict_all['M1_dicts']['models']['linear'][wind]['fit_params'],axis=0)

    temp[1,[0+ct,3+ct]] = np.mean(data_dict_all['M1_dicts']['models']['diff'][wind]['fit_params'],axis=0)
    temp[1,[1+ct,4+ct]] = np.max(data_dict_all['M1_dicts']['models']['diff'][wind]['fit_params'],axis=0)
    temp[1,[2+ct,5+ct]] = np.min(data_dict_all['M1_dicts']['models']['diff'][wind]['fit_params'],axis=0)

    temp[2,[0+ct,3+ct,6+ct,9+ct,12+ct]] = np.mean(data_dict_all['M1_dicts']['models']['div_nl'][wind]['fit_params'],axis=0)
    temp[2,[1+ct,4+ct,7+ct,10+ct,13+ct]] = np.max(data_dict_all['M1_dicts']['models']['div_nl'][wind]['fit_params'],axis=0)
    temp[2,[2+ct,5+ct,8+ct,11+ct,14+ct]] = np.min(data_dict_all['M1_dicts']['models']['div_nl'][wind]['fit_params'],axis=0)

    temp[3,[0+ct,3+ct,6+ct,9+ct]] = np.mean(data_dict_all['M1_dicts']['models']['div_nl_noe'][wind]['fit_params'],axis=0)
    temp[3,[1+ct,4+ct,7+ct,10+ct]] = np.max(data_dict_all['M1_dicts']['models']['div_nl_noe'][wind]['fit_params'],axis=0)
    temp[3,[2+ct,5+ct,8+ct,11+ct]] = np.min(data_dict_all['M1_dicts']['models']['div_nl_noe'][wind]['fit_params'],axis=0)

    temp[4,[0+ct,3+ct,6+ct,9+ct]] = np.mean(data_dict_all['M1_dicts']['models']['div_nl_Y'][wind]['fit_params'],axis=0)
    temp[4,[1+ct,4+ct,7+ct,10+ct]] = np.max(data_dict_all['M1_dicts']['models']['div_nl_Y'][wind]['fit_params'],axis=0)
    temp[4,[2+ct,5+ct,8+ct,11+ct]] = np.min(data_dict_all['M1_dicts']['models']['div_nl_Y'][wind]['fit_params'],axis=0)

    temp[5,[0+ct,3+ct,6+ct,9+ct,12+ct]] = np.mean(data_dict_all['M1_dicts']['models']['div_nl_separate_add'][wind]['fit_params'],axis=0)
    temp[5,[1+ct,4+ct,7+ct,10+ct,13+ct]] = np.max(data_dict_all['M1_dicts']['models']['div_nl_separate_add'][wind]['fit_params'],axis=0)
    temp[5,[2+ct,5+ct,8+ct,11+ct,14+ct]] = np.min(data_dict_all['M1_dicts']['models']['div_nl_separate_add'][wind]['fit_params'],axis=0)

    temp[6,[0+ct,3+ct,6+ct,9+ct,12+ct]] = np.mean(data_dict_all['M1_dicts']['models']['div_nl_separate_multiply'][wind]['fit_params'],axis=0)
    temp[6,[1+ct,4+ct,7+ct,10+ct,13+ct]] = np.max(data_dict_all['M1_dicts']['models']['div_nl_separate_multiply'][wind]['fit_params'],axis=0)
    temp[6,[2+ct,5+ct,8+ct,11+ct,14+ct]] = np.min(data_dict_all['M1_dicts']['models']['div_nl_separate_multiply'][wind]['fit_params'],axis=0)

worksheet.write(0,0,'M1')
worksheet.write(0,1,'aft_cue')
worksheet.write_row(1,1,param_names)
worksheet.write(0,17,'bfr_res')
worksheet.write_row(1,17,param_names)
worksheet.write(0,33,'aft_res')
worksheet.write_row(1,33,param_names)
worksheet.write(0,49,'res_win')
worksheet.write_row(1,49,param_names)
worksheet.write(0,61,'concat')
worksheet.write_row(1,61,param_names)
worksheet.write_column(2,0,model_names)
for i in range(temp.shape[0]):
    worksheet.write_row(i + 2,1,temp[i,:])

#temp = np.full([7,75],np.nan)
temp = np.zeros((7,75))

for i in range(5):
    if i ==0:
        wind = 'aft_cue'
    elif i == 1:
        wind = 'bfr_res'
    elif i == 2:
        wind = 'aft_res'
    elif i == 3:
        wind = 'res_win'
    elif i == 4:
        wind = 'concat'

    ct = i*15

    temp[0,[0+ct,3+ct,6+ct]] = np.mean(data_dict_all['S1_dicts']['models']['linear'][wind]['fit_params'],axis=0)
    temp[0,[1+ct,4+ct,7+ct]] = np.max(data_dict_all['S1_dicts']['models']['linear'][wind]['fit_params'],axis=0)
    temp[0,[2+ct,5+ct,8+ct]] = np.min(data_dict_all['S1_dicts']['models']['linear'][wind]['fit_params'],axis=0)

    temp[1,[0+ct,3+ct]] = np.mean(data_dict_all['S1_dicts']['models']['diff'][wind]['fit_params'],axis=0)
    temp[1,[1+ct,4+ct]] = np.max(data_dict_all['S1_dicts']['models']['diff'][wind]['fit_params'],axis=0)
    temp[1,[2+ct,5+ct]] = np.min(data_dict_all['S1_dicts']['models']['diff'][wind]['fit_params'],axis=0)

    temp[2,[0+ct,3+ct,6+ct,9+ct,12+ct]] = np.mean(data_dict_all['S1_dicts']['models']['div_nl'][wind]['fit_params'],axis=0)
    temp[2,[1+ct,4+ct,7+ct,10+ct,13+ct]] = np.max(data_dict_all['S1_dicts']['models']['div_nl'][wind]['fit_params'],axis=0)
    temp[2,[2+ct,5+ct,8+ct,11+ct,14+ct]] = np.min(data_dict_all['S1_dicts']['models']['div_nl'][wind]['fit_params'],axis=0)

    temp[3,[0+ct,3+ct,6+ct,9+ct]] = np.mean(data_dict_all['S1_dicts']['models']['div_nl_noe'][wind]['fit_params'],axis=0)
    temp[3,[1+ct,4+ct,7+ct,10+ct]] = np.max(data_dict_all['S1_dicts']['models']['div_nl_noe'][wind]['fit_params'],axis=0)
    temp[3,[2+ct,5+ct,8+ct,11+ct]] = np.min(data_dict_all['S1_dicts']['models']['div_nl_noe'][wind]['fit_params'],axis=0)

    temp[4,[0+ct,3+ct,6+ct,9+ct]] = np.mean(data_dict_all['S1_dicts']['models']['div_nl_Y'][wind]['fit_params'],axis=0)
    temp[4,[1+ct,4+ct,7+ct,10+ct]] = np.max(data_dict_all['S1_dicts']['models']['div_nl_Y'][wind]['fit_params'],axis=0)
    temp[4,[2+ct,5+ct,8+ct,11+ct]] = np.min(data_dict_all['S1_dicts']['models']['div_nl_Y'][wind]['fit_params'],axis=0)

    temp[5,[0+ct,3+ct,6+ct,9+ct,12+ct]] = np.mean(data_dict_all['S1_dicts']['models']['div_nl_separate_add'][wind]['fit_params'],axis=0)
    temp[5,[1+ct,4+ct,7+ct,10+ct,13+ct]] = np.max(data_dict_all['S1_dicts']['models']['div_nl_separate_add'][wind]['fit_params'],axis=0)
    temp[5,[2+ct,5+ct,8+ct,11+ct,14+ct]] = np.min(data_dict_all['S1_dicts']['models']['div_nl_separate_add'][wind]['fit_params'],axis=0)

    temp[6,[0+ct,3+ct,6+ct,9+ct,12+ct]] = np.mean(data_dict_all['S1_dicts']['models']['div_nl_separate_multiply'][wind]['fit_params'],axis=0)
    temp[6,[1+ct,4+ct,7+ct,10+ct,13+ct]] = np.max(data_dict_all['S1_dicts']['models']['div_nl_separate_multiply'][wind]['fit_params'],axis=0)
    temp[6,[2+ct,5+ct,8+ct,11+ct,14+ct]] = np.min(data_dict_all['S1_dicts']['models']['div_nl_separate_multiply'][wind]['fit_params'],axis=0)

worksheet.write(12,0,'S1')
worksheet.write(12,1,'aft_cue')
worksheet.write_row(13,1,param_names)
worksheet.write(12,17,'bfr_res')
worksheet.write_row(13,17,param_names)
worksheet.write(12,33,'aft_res')
worksheet.write_row(13,33,param_names)
worksheet.write(12,49,'res_win')
worksheet.write_row(13,49,param_names)
worksheet.write(12,61,'concat')
worksheet.write_row(13,61,param_names)
worksheet.write_column(14,0,model_names)
for i in range(temp.shape[0]):
    worksheet.write_row(i + 14,1,temp[i,:])

#temp = np.full([7,75],np.nan)
temp = np.zeros((7,75))

for i in range(5):
    if i ==0:
        wind = 'aft_cue'
    elif i == 1:
        wind = 'bfr_res'
    elif i == 2:
        wind = 'aft_res'
    elif i == 3:
        wind = 'res_win'
    elif i == 4:
        wind = 'concat'

    ct = i*15

    temp[0,[0+ct,3+ct,6+ct]] = np.mean(data_dict_all['PmD_dicts']['models']['linear'][wind]['fit_params'],axis=0)
    temp[0,[1+ct,4+ct,7+ct]] = np.max(data_dict_all['PmD_dicts']['models']['linear'][wind]['fit_params'],axis=0)
    temp[0,[2+ct,5+ct,8+ct]] = np.min(data_dict_all['PmD_dicts']['models']['linear'][wind]['fit_params'],axis=0)

    temp[1,[0+ct,3+ct]] = np.mean(data_dict_all['PmD_dicts']['models']['diff'][wind]['fit_params'],axis=0)
    temp[1,[1+ct,4+ct]] = np.max(data_dict_all['PmD_dicts']['models']['diff'][wind]['fit_params'],axis=0)
    temp[1,[2+ct,5+ct]] = np.min(data_dict_all['PmD_dicts']['models']['diff'][wind]['fit_params'],axis=0)

    temp[2,[0+ct,3+ct,6+ct,9+ct,12+ct]] = np.mean(data_dict_all['PmD_dicts']['models']['div_nl'][wind]['fit_params'],axis=0)
    temp[2,[1+ct,4+ct,7+ct,10+ct,13+ct]] = np.max(data_dict_all['PmD_dicts']['models']['div_nl'][wind]['fit_params'],axis=0)
    temp[2,[2+ct,5+ct,8+ct,11+ct,14+ct]] = np.min(data_dict_all['PmD_dicts']['models']['div_nl'][wind]['fit_params'],axis=0)

    temp[3,[0+ct,3+ct,6+ct,9+ct]] = np.mean(data_dict_all['PmD_dicts']['models']['div_nl_noe'][wind]['fit_params'],axis=0)
    temp[3,[1+ct,4+ct,7+ct,10+ct]] = np.max(data_dict_all['PmD_dicts']['models']['div_nl_noe'][wind]['fit_params'],axis=0)
    temp[3,[2+ct,5+ct,8+ct,11+ct]] = np.min(data_dict_all['PmD_dicts']['models']['div_nl_noe'][wind]['fit_params'],axis=0)

    temp[4,[0+ct,3+ct,6+ct,9+ct]] = np.mean(data_dict_all['PmD_dicts']['models']['div_nl_Y'][wind]['fit_params'],axis=0)
    temp[4,[1+ct,4+ct,7+ct,10+ct]] = np.max(data_dict_all['PmD_dicts']['models']['div_nl_Y'][wind]['fit_params'],axis=0)
    temp[4,[2+ct,5+ct,8+ct,11+ct]] = np.min(data_dict_all['PmD_dicts']['models']['div_nl_Y'][wind]['fit_params'],axis=0)

    temp[5,[0+ct,3+ct,6+ct,9+ct,12+ct]] = np.mean(data_dict_all['PmD_dicts']['models']['div_nl_separate_add'][wind]['fit_params'],axis=0)
    temp[5,[1+ct,4+ct,7+ct,10+ct,13+ct]] = np.max(data_dict_all['PmD_dicts']['models']['div_nl_separate_add'][wind]['fit_params'],axis=0)
    temp[5,[2+ct,5+ct,8+ct,11+ct,14+ct]] = np.min(data_dict_all['PmD_dicts']['models']['div_nl_separate_add'][wind]['fit_params'],axis=0)

    temp[6,[0+ct,3+ct,6+ct,9+ct,12+ct]] = np.mean(data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply'][wind]['fit_params'],axis=0)
    temp[6,[1+ct,4+ct,7+ct,10+ct,13+ct]] = np.max(data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply'][wind]['fit_params'],axis=0)
    temp[6,[2+ct,5+ct,8+ct,11+ct,14+ct]] = np.min(data_dict_all['PmD_dicts']['models']['div_nl_separate_multiply'][wind]['fit_params'],axis=0)

worksheet.write(24,0,'PMd')
worksheet.write(24,1,'aft_cue')
worksheet.write_row(25,1,param_names)
worksheet.write(24,17,'bfr_res')
worksheet.write_row(25,17,param_names)
worksheet.write(24,33,'aft_res')
worksheet.write_row(25,33,param_names)
worksheet.write(24,49,'res_win')
worksheet.write_row(25,49,param_names)
worksheet.write(24,61,'concat')
worksheet.write_row(25,61,param_names)
worksheet.write_column(26,0,model_names)
for i in range(temp.shape[0]):
    worksheet.write_row(i + 26,1,temp[i,:])




