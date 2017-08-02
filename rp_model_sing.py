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
#import matplotlib.colors as colors

#######################
#params to set ########
#######################

bin_size = 50 #in ms
time_before = -0.5 #negative value
time_after = 1.0
baseline_time = -1.0 #negative value
normalize_bool = False
plot_model_bool = True
sqrt_bool = True

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

        baseline_fr = np.zeros((len(condensed),np.shape(hists)[0],-1*baseline_bins))
        bfr_cue_fr = np.zeros((len(condensed),np.shape(hists)[0],-1*bins_before))
        aft_cue_fr = np.zeros((len(condensed),np.shape(hists)[0],bins_after))
        bfr_result_fr = np.zeros((len(condensed),np.shape(hists)[0],-1*bins_before))
        aft_result_fr = np.zeros((len(condensed),np.shape(hists)[0],bins_after))

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
                        
                                baseline_fr[i,j,:] = baseline_hist / float(bin_size) * 1000
                                bfr_cue_fr[i,j,:] = bfr_cue_hist / float(bin_size) * 1000
                                aft_cue_fr[i,j,:] = aft_cue_hist / float(bin_size) * 1000
                                bfr_result_fr[i,j,:] = bfr_result_hist / float(bin_size) * 1000
                                aft_result_fr[i,j,:] = aft_result_hist / float(bin_size) * 1000                        
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
                        
        return_dict = {'bfr_cue_fr':bfr_cue_fr,'aft_cue_fr':aft_cue_fr,'bfr_result_fr':bfr_result_fr,'aft_result_fr':aft_result_fr,'bfr_cue_nl_fr':bfr_cue_nl_fr,'aft_cue_nl_fr':aft_cue_nl_fr,'bfr_result_nl_fr':bfr_result_nl_fr,'aft_result_nl_fr':aft_result_nl_fr,'baseline_fr':baseline_fr}
        return(return_dict)

def func(X,alpha,beta,k):
        R,P = X
        return (alpha * R + beta * P + k)

def make_model(fr_data,condensed,region_key,type_key):
    
        best_fit_params = np.zeros((fr_data.shape[0],fr_data.shape[1],5))
        pcov_noninf = []
        for unit_num in range(fr_data.shape[1]):
                for event_num in range(fr_data.shape[0]):
                
                        R = condensed[event_num,3]
                        P = condensed[event_num,4]
                        X=R,P
                
                        fr = fr_data[event_num,unit_num,:]
                        try:
                                popt,pcov = curve_fit(func,X,fr)
                        except:
                                print 'exception %s, unit %s, event %s' %(region_key,unit_num,event_num)
                                best_fit_params[event_num,unit_num,:] = [9999, 9999, 9999, R, P]

                        #popt 0 = alpha, 1 = beta, 2 = k, 3 = R, 4 = P
                        best_fit_params[event_num,unit_num,:] = [popt[0], popt[1], popt[2], R, P]
                        
                        #pcov 3x3 matrix, or inf
                        #one std dev error on params = perr = np.sqrt(np.diag(pcov))
                        if not np.isinf(pcov).any():
                                perr= np.sqrt(np.diag(pcov))
                                pcov_row = [event_num, unit_num, perr[0], perr[1], perr[2]]
                                pcov_noninf.append(pcov_row)

        #avg_temp = best_fit_params[best_fit_params != 9999]
        avg_temp = best_fit_params
        avg_temp[avg_temp == 9999] = np.nan

        unit_avg = np.nanmean(avg_temp,axis=0)

        return_dict = {'best_fit_params':best_fit_params,'pcov_noninf':pcov_noninf,'unit_avg':unit_avg}
        return(return_dict)


def plot_avgs(bfr_cue_model,aft_cue_model,bfr_result_model,aft_result_model,bfr_cue_fr,aft_cue_fr,bfr_result_fr,aft_result_fr,region_key,type_key):
        #if type_key == 'succ':
        #        pdb.set_trace()

        print 'plotting hists %s %s' %(region_key,type_key)

        #plot avg for each unit
        #plot hist for each unit

        bfr_cue_params = bfr_cue_model['best_fit_params']
        bfr_cue_unit_avg = bfr_cue_model['unit_avg']
        aft_cue_params = aft_cue_model['best_fit_params']
        aft_cue_unit_avg = aft_cue_model['unit_avg']
        bfr_result_params = bfr_result_model['best_fit_params']
        bfr_result_unit_avg = bfr_result_model['unit_avg']
        aft_result_params = aft_result_model['best_fit_params']
        aft_result_unit_avg = aft_result_model['unit_avg']

        bfr_cue_fr_avg = np.mean(bfr_cue_fr,axis=0)
        aft_cue_fr_avg = np.mean(aft_cue_fr,axis=0)
        bfr_result_fr_avg = np.mean(bfr_result_fr,axis=0)
        aft_result_fr_avg = np.mean(aft_result_fr,axis=0)
        
        #popt 0 = alpha, 1 = beta, 2 = k, 3 = R, 4 = P
        for unit_num in range(np.shape(bfr_cue_params)[1]):
                bfr_cue_unit_vals = bfr_cue_params[:,unit_num,:]
                aft_cue_unit_vals = aft_cue_params[:,unit_num,:]
                bfr_result_unit_vals = bfr_result_params[:,unit_num,:]
                aft_result_unit_vals = aft_result_params[:,unit_num,:]

                #bfr_cue_unit_vals = bfr_cue_unit_vals[bfr_cue_unit_vals != 9999]
                #aft_cue_unit_vals = aft_cue_unit_vals[aft_cue_unit_vals != 9999]
                #bfr_result_unit_vals = bfr_result_unit_vals[bfr_result_unit_vals != 9999]
                #aft_result_unit_vals = aft_result_unit_vals[aft_result_unit_vals != 9999]

                bfr_cue_unit_alpha = bfr_cue_unit_vals[:,0]
                bfr_cue_unit_beta = bfr_cue_unit_vals[:,1]
                bfr_cue_unit_k = bfr_cue_unit_vals[:,2]
                aft_cue_unit_alpha = aft_cue_unit_vals[:,0]
                aft_cue_unit_beta = aft_cue_unit_vals[:,1]
                aft_cue_unit_k = aft_cue_unit_vals[:,2]
                bfr_result_unit_alpha = bfr_result_unit_vals[:,0]
                bfr_result_unit_beta = bfr_result_unit_vals[:,1]
                bfr_result_unit_k = bfr_result_unit_vals[:,2]
                aft_result_unit_alpha = aft_result_unit_vals[:,0]
                aft_result_unit_beta = aft_result_unit_vals[:,1]
                aft_result_unit_k = aft_result_unit_vals[:,2]

                bfr_cue_unit_alpha = bfr_cue_unit_alpha[bfr_cue_unit_alpha != 9999]
                bfr_cue_unit_beta = bfr_cue_unit_beta[bfr_cue_unit_beta != 9999]
                bfr_cue_unit_k = bfr_cue_unit_k[bfr_cue_unit_k != 9999]
                aft_cue_unit_alpha = aft_cue_unit_alpha[aft_cue_unit_alpha != 9999]
                aft_cue_unit_beta = aft_cue_unit_beta[aft_cue_unit_beta != 9999]
                aft_cue_unit_k = aft_cue_unit_k[aft_cue_unit_k != 9999]
                bfr_result_unit_alpha = bfr_result_unit_alpha[bfr_result_unit_alpha != 9999]
                bfr_result_unit_beta = bfr_result_unit_beta[bfr_result_unit_beta != 9999]
                bfr_result_unit_k = bfr_result_unit_k[bfr_result_unit_k != 9999]
                aft_result_unit_alpha = aft_result_unit_alpha[aft_result_unit_alpha != 9999]
                aft_result_unit_beta = aft_result_unit_beta[aft_result_unit_beta != 9999]
                aft_result_unit_k = aft_result_unit_k[aft_result_unit_k != 9999]

                ax = plt.gca()
                plt.subplot(4,3,1)
                plt.hist(bfr_cue_unit_alpha)
                plt.title('before cue: alpha',fontsize='small')
                plt.subplot(4,3,2)
                plt.hist(bfr_cue_unit_beta)
                plt.title('beta',fontsize='small')
                plt.subplot(4,3,3)
                plt.hist(bfr_cue_unit_k)
                plt.title('k',fontsize='small')

                plt.subplot(4,3,4)
                plt.hist(aft_cue_unit_alpha)
                plt.title('after cue: alpha',fontsize='small')
                plt.subplot(4,3,5)
                plt.hist(aft_cue_unit_beta)
                plt.title('beta',fontsize='small')
                plt.subplot(4,3,6)
                plt.hist(aft_cue_unit_k)
                plt.title('k',fontsize='small')

                plt.subplot(4,3,7)
                plt.hist(bfr_result_unit_alpha)
                plt.title('before result: alpha',fontsize='small')
                plt.subplot(4,3,8)
                plt.hist(bfr_result_unit_beta)
                plt.title('beta',fontsize='small')
                plt.subplot(4,3,9)
                plt.hist(bfr_result_unit_k)
                plt.title('k',fontsize='small')

                plt.subplot(4,3,10)
                plt.hist(aft_result_unit_alpha)
                plt.title('after result: alpha',fontsize='small')
                plt.subplot(4,3,11)
                plt.hist(aft_result_unit_beta)
                plt.title('beta',fontsize='small')
                plt.subplot(4,3,12)
                plt.hist(aft_result_unit_k)
                plt.title('k',fontsize='small')
        
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                plt.rcParams['xtick.labelsize'] = 8
                plt.rcParams['ytick.labelsize'] = 8
                plt.suptitle('Region %s, unit %s: param hists %s' %(region_key,unit_num,type_key))
                plt.savefig('param_hists_%s_unit%s_%s' %(region_key,str(unit_num).zfill(2),type_key))
                plt.clf()

        #average
        ax = plt.gca()
        plt.subplot(4,3,1)
        plt.hist(bfr_cue_unit_avg[:,0])
        plt.title('before cue: alpha',fontsize='small')
        plt.subplot(4,3,2)
        plt.hist(bfr_cue_unit_avg[:,1])
        plt.title('beta',fontsize='small')
        plt.subplot(4,3,3)
        plt.hist(bfr_cue_unit_avg[:,2])
        plt.title('k',fontsize='small')
                
        plt.subplot(4,3,4)
        plt.hist(aft_cue_unit_avg[:,0])
        plt.title('after cue: alpha',fontsize='small')
        plt.subplot(4,3,5)
        plt.hist(aft_cue_unit_avg[:,1])
        plt.title('beta',fontsize='small')
        plt.subplot(4,3,6)
        plt.hist(aft_cue_unit_avg[:,2])
        plt.title('k',fontsize='small')

        plt.subplot(4,3,7)
        plt.hist(bfr_result_unit_avg[:,0])
        plt.title('before result: alpha',fontsize='small')
        plt.subplot(4,3,8)
        plt.hist(bfr_result_unit_avg[:,1])
        plt.title('beta',fontsize='small')
        plt.subplot(4,3,9)
        plt.hist(bfr_result_unit_avg[:,2])
        plt.title('k',fontsize='small')

        plt.subplot(4,3,10)
        plt.hist(aft_result_unit_avg[:,0])
        plt.title('after result: alpha',fontsize='small')
        plt.subplot(4,3,11)
        plt.hist(aft_result_unit_avg[:,1])
        plt.title('beta',fontsize='small')
        plt.subplot(4,3,12)
        plt.hist(aft_result_unit_avg[:,2])
        plt.title('k',fontsize='small')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.suptitle('Region %s average: param hists %s' %(region_key,type_key))
        plt.savefig('param_hists_avg_%s_%s' %(region_key,type_key))
        plt.clf()

        #concatenated hists
        bfr_cue_flattened_alpha = np.ndarray.flatten(bfr_cue_params[:,:,0])
        bfr_cue_flattened_beta = np.ndarray.flatten(bfr_cue_params[:,:,1])
        bfr_cue_flattened_k = np.ndarray.flatten(bfr_cue_params[:,:,2])
        bfr_cue_flattened_alpha = bfr_cue_flattened_alpha[bfr_cue_flattened_alpha != 9999]
        bfr_cue_flattened_beta = bfr_cue_flattened_beta[bfr_cue_flattened_beta != 9999]
        bfr_cue_flattened_k = bfr_cue_flattened_k[bfr_cue_flattened_k != 9999]

        aft_cue_flattened_alpha = np.ndarray.flatten(aft_cue_params[:,:,0])
        aft_cue_flattened_beta = np.ndarray.flatten(aft_cue_params[:,:,1])
        aft_cue_flattened_k = np.ndarray.flatten(aft_cue_params[:,:,2])
        aft_cue_flattened_alpha = aft_cue_flattened_alpha[aft_cue_flattened_alpha != 9999]
        aft_cue_flattened_beta = aft_cue_flattened_beta[aft_cue_flattened_beta != 9999]
        aft_cue_flattened_k = aft_cue_flattened_k[aft_cue_flattened_k != 9999]

        bfr_result_flattened_alpha = np.ndarray.flatten(bfr_result_params[:,:,0])
        bfr_result_flattened_beta = np.ndarray.flatten(bfr_result_params[:,:,1])
        bfr_result_flattened_k = np.ndarray.flatten(bfr_result_params[:,:,2])
        bfr_result_flattened_alpha = bfr_result_flattened_alpha[bfr_result_flattened_alpha != 9999]
        bfr_result_flattened_beta = bfr_result_flattened_beta[bfr_result_flattened_beta != 9999]
        bfr_result_flattened_k = bfr_result_flattened_k[bfr_result_flattened_k != 9999]

        aft_result_flattened_alpha = np.ndarray.flatten(aft_result_params[:,:,0])
        aft_result_flattened_beta = np.ndarray.flatten(aft_result_params[:,:,1])
        aft_result_flattened_k = np.ndarray.flatten(aft_result_params[:,:,2])
        aft_result_flattened_alpha = aft_result_flattened_alpha[aft_result_flattened_alpha != 9999]
        aft_result_flattened_beta = aft_result_flattened_beta[aft_result_flattened_beta != 9999]
        aft_result_flattened_k = aft_result_flattened_k[aft_result_flattened_k != 9999]

        ax = plt.gca()
        plt.subplot(4,3,1)
        plt.hist(bfr_cue_flattened_alpha)
        plt.title('before cue: alpha',fontsize='small')
        plt.subplot(4,3,2)
        plt.hist(bfr_cue_flattened_beta)
        plt.title('beta',fontsize='small')
        plt.subplot(4,3,3)
        plt.hist(bfr_cue_flattened_k)
        plt.title('k',fontsize='small')
                
        plt.subplot(4,3,4)
        plt.hist(aft_cue_flattened_alpha)
        plt.title('after cue: alpha',fontsize='small')
        plt.subplot(4,3,5)
        plt.hist(np.ndarray.flatten(aft_cue_params[:,:,1]))
        plt.hist(aft_cue_flattened_beta)
        plt.title('beta',fontsize='small')
        plt.subplot(4,3,6)
        plt.hist(aft_cue_flattened_k)
        plt.title('k',fontsize='small')

        plt.subplot(4,3,7)
        plt.hist(bfr_result_flattened_alpha)
        plt.title('before result: alpha',fontsize='small')
        plt.subplot(4,3,8)
        plt.hist(bfr_result_flattened_beta)
        plt.title('beta',fontsize='small')
        plt.subplot(4,3,9)
        plt.hist(bfr_result_flattened_k)
        plt.title('k',fontsize='small')

        plt.subplot(4,3,10)
        plt.hist(aft_result_flattened_alpha)
        plt.title('after result: alpha',fontsize='small')
        plt.subplot(4,3,11)
        plt.hist(aft_result_flattened_beta)
        plt.title('beta',fontsize='small')
        plt.subplot(4,3,12)
        plt.hist(aft_result_flattened_k)
        plt.title('k',fontsize='small')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.suptitle('Region %s conc: param hists %s' %(region_key,type_key))
        plt.savefig('param_hists_conc_%s_%s' %(region_key,type_key))
        plt.clf()
        
        #plt averages
        #if nl bool for labeling, already passing the correct one

        return_dict = {}
        return(return_dict)


def plot_stacked_avgs(bfr_cue_model,aft_cue_model,bfr_result_model,aft_result_model,bfr_cue_fr,aft_cue_fr,bfr_result_fr,aft_result_fr,condensed,region_key,type_key):

        print 'plotting stacked hists %s %s' %(region_key,type_key)

        bfr_cue_params = bfr_cue_model['best_fit_params']
        bfr_cue_unit_avg = bfr_cue_model['unit_avg']
        aft_cue_params = aft_cue_model['best_fit_params']
        aft_cue_unit_avg = aft_cue_model['unit_avg']
        bfr_result_params = bfr_result_model['best_fit_params']
        bfr_result_unit_avg = bfr_result_model['unit_avg']
        aft_result_params = aft_result_model['best_fit_params']
        aft_result_unit_avg = aft_result_model['unit_avg']

        bfr_cue_fr_avg = np.mean(bfr_cue_fr,axis=0)
        aft_cue_fr_avg = np.mean(aft_cue_fr,axis=0)
        bfr_result_fr_avg = np.mean(bfr_result_fr,axis=0)
        aft_result_fr_avg = np.mean(aft_result_fr,axis=0)
        
        bfr_cue_succ_params = bfr_cue_params[condensed[:,5] == 1]
        bfr_cue_fail_params = bfr_cue_params[condensed[:,5] == -1]
        aft_cue_succ_params = aft_cue_params[condensed[:,5] == 1]
        aft_cue_fail_params = aft_cue_params[condensed[:,5] == -1]
        bfr_result_succ_params = bfr_result_params[condensed[:,5] == 1]
        bfr_result_fail_params = bfr_result_params[condensed[:,5] == -1]
        aft_result_succ_params = aft_result_params[condensed[:,5] == 1]
        aft_result_fail_params = aft_result_params[condensed[:,5] == -1]


        #popt 0 = alpha, 1 = beta, 2 = k, 3 = R, 4 = P
        for unit_num in range(np.shape(bfr_cue_params)[1]):
                bfr_cue_succ_unit_vals = bfr_cue_succ_params[:,unit_num,:]
                aft_cue_succ_unit_vals = aft_cue_succ_params[:,unit_num,:]
                bfr_result_succ_unit_vals = bfr_result_succ_params[:,unit_num,:]
                aft_result_succ_unit_vals = aft_result_succ_params[:,unit_num,:]

                bfr_cue_fail_unit_vals = bfr_cue_fail_params[:,unit_num,:]
                aft_cue_fail_unit_vals = aft_cue_fail_params[:,unit_num,:]
                bfr_result_fail_unit_vals = bfr_result_fail_params[:,unit_num,:]
                aft_result_fail_unit_vals = aft_result_fail_params[:,unit_num,:]

                bfr_cue_succ_unit_alpha = bfr_cue_succ_unit_vals[:,0]
                bfr_cue_succ_unit_beta = bfr_cue_succ_unit_vals[:,1]
                bfr_cue_succ_unit_k = bfr_cue_succ_unit_vals[:,2]
                aft_cue_succ_unit_alpha = aft_cue_succ_unit_vals[:,0]
                aft_cue_succ_unit_beta = aft_cue_succ_unit_vals[:,1]
                aft_cue_succ_unit_k = aft_cue_succ_unit_vals[:,2]
                bfr_result_succ_unit_alpha = bfr_result_succ_unit_vals[:,0]
                bfr_result_succ_unit_beta = bfr_result_succ_unit_vals[:,1]
                bfr_result_succ_unit_k = bfr_result_succ_unit_vals[:,2]
                aft_result_succ_unit_alpha = aft_result_succ_unit_vals[:,0]
                aft_result_succ_unit_beta = aft_result_succ_unit_vals[:,1]
                aft_result_succ_unit_k = aft_result_succ_unit_vals[:,2]

                bfr_cue_succ_unit_alpha = bfr_cue_succ_unit_alpha[bfr_cue_succ_unit_alpha != 9999]
                bfr_cue_succ_unit_beta = bfr_cue_succ_unit_beta[bfr_cue_succ_unit_beta != 9999]
                bfr_cue_succ_unit_k = bfr_cue_succ_unit_k[bfr_cue_succ_unit_k != 9999]
                aft_cue_succ_unit_alpha = aft_cue_succ_unit_alpha[aft_cue_succ_unit_alpha != 9999]
                aft_cue_succ_unit_beta = aft_cue_succ_unit_beta[aft_cue_succ_unit_beta != 9999]
                aft_cue_succ_unit_k = aft_cue_succ_unit_k[aft_cue_succ_unit_k != 9999]
                bfr_result_succ_unit_alpha = bfr_result_succ_unit_alpha[bfr_result_succ_unit_alpha != 9999]
                bfr_result_succ_unit_beta = bfr_result_succ_unit_beta[bfr_result_succ_unit_beta != 9999]
                bfr_result_succ_unit_k = bfr_result_succ_unit_k[bfr_result_succ_unit_k != 9999]
                aft_result_succ_unit_alpha = aft_result_succ_unit_alpha[aft_result_succ_unit_alpha != 9999]
                aft_result_succ_unit_beta = aft_result_succ_unit_beta[aft_result_succ_unit_beta != 9999]
                aft_result_succ_unit_k = aft_result_succ_unit_k[aft_result_succ_unit_k != 9999]

                bfr_cue_fail_unit_alpha = bfr_cue_fail_unit_vals[:,0]
                bfr_cue_fail_unit_beta = bfr_cue_fail_unit_vals[:,1]
                bfr_cue_fail_unit_k = bfr_cue_fail_unit_vals[:,2]
                aft_cue_fail_unit_alpha = aft_cue_fail_unit_vals[:,0]
                aft_cue_fail_unit_beta = aft_cue_fail_unit_vals[:,1]
                aft_cue_fail_unit_k = aft_cue_fail_unit_vals[:,2]
                bfr_result_fail_unit_alpha = bfr_result_fail_unit_vals[:,0]
                bfr_result_fail_unit_beta = bfr_result_fail_unit_vals[:,1]
                bfr_result_fail_unit_k = bfr_result_fail_unit_vals[:,2]
                aft_result_fail_unit_alpha = aft_result_fail_unit_vals[:,0]
                aft_result_fail_unit_beta = aft_result_fail_unit_vals[:,1]
                aft_result_fail_unit_k = aft_result_fail_unit_vals[:,2]

                bfr_cue_fail_unit_alpha = bfr_cue_fail_unit_alpha[bfr_cue_fail_unit_alpha != 9999]
                bfr_cue_fail_unit_beta = bfr_cue_fail_unit_beta[bfr_cue_fail_unit_beta != 9999]
                bfr_cue_fail_unit_k = bfr_cue_fail_unit_k[bfr_cue_fail_unit_k != 9999]
                aft_cue_fail_unit_alpha = aft_cue_fail_unit_alpha[aft_cue_fail_unit_alpha != 9999]
                aft_cue_fail_unit_beta = aft_cue_fail_unit_beta[aft_cue_fail_unit_beta != 9999]
                aft_cue_fail_unit_k = aft_cue_fail_unit_k[aft_cue_fail_unit_k != 9999]
                bfr_result_fail_unit_alpha = bfr_result_fail_unit_alpha[bfr_result_fail_unit_alpha != 9999]
                bfr_result_fail_unit_beta = bfr_result_fail_unit_beta[bfr_result_fail_unit_beta != 9999]
                bfr_result_fail_unit_k = bfr_result_fail_unit_k[bfr_result_fail_unit_k != 9999]
                aft_result_fail_unit_alpha = aft_result_fail_unit_alpha[aft_result_fail_unit_alpha != 9999]
                aft_result_fail_unit_beta = aft_result_fail_unit_beta[aft_result_fail_unit_beta != 9999]
                aft_result_fail_unit_k = aft_result_fail_unit_k[aft_result_fail_unit_k != 9999]

                ax = plt.gca()
                plt.subplot(4,3,1)
                plt.hist([bfr_cue_succ_unit_alpha,bfr_cue_fail_unit_alpha], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
                plt.title('before cue: alpha',fontsize='small')
                plt.subplot(4,3,2)
                plt.hist([bfr_cue_succ_unit_beta,bfr_cue_fail_unit_beta], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
                plt.title('beta',fontsize='small')
                plt.subplot(4,3,3)
                plt.hist([bfr_cue_succ_unit_k,bfr_cue_fail_unit_k], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
                plt.title('k',fontsize='small')

                plt.subplot(4,3,4)
                plt.hist([aft_cue_succ_unit_alpha,aft_cue_fail_unit_alpha], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
                plt.title('after cue: alpha',fontsize='small')
                plt.subplot(4,3,5)
                plt.hist([aft_cue_succ_unit_beta,aft_cue_fail_unit_beta], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
                plt.title('beta',fontsize='small')
                plt.subplot(4,3,6)
                plt.hist([aft_cue_succ_unit_k,aft_cue_fail_unit_k], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
                plt.title('k',fontsize='small')

                plt.subplot(4,3,7)
                plt.hist([bfr_result_succ_unit_alpha,bfr_result_fail_unit_alpha], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
                plt.title('before result: alpha',fontsize='small')
                plt.subplot(4,3,8)
                plt.hist([bfr_result_succ_unit_beta,bfr_result_fail_unit_beta], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
                plt.title('beta',fontsize='small')
                plt.subplot(4,3,9)
                plt.hist([bfr_result_succ_unit_k,bfr_result_fail_unit_k], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
                plt.title('k',fontsize='small')

                plt.subplot(4,3,10)
                plt.hist([aft_result_succ_unit_alpha,aft_result_fail_unit_alpha], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
                plt.title('after result: alpha',fontsize='small')
                plt.subplot(4,3,11)
                plt.hist([aft_result_succ_unit_beta,aft_result_fail_unit_beta], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
                plt.title('beta',fontsize='small')
                plt.subplot(4,3,12)
                plt.hist([aft_result_succ_unit_k,aft_result_fail_unit_k], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
                plt.title('k',fontsize='small')
        
                plt.legend(bbox_to_anchor=[1.6,3.2],fontsize='small')
                plt.tight_layout()
                plt.subplots_adjust(top=0.9,right=0.85)
                plt.rcParams['xtick.labelsize'] = 8
                plt.rcParams['ytick.labelsize'] = 8
                plt.suptitle('Region %s, unit %s: param hists %s stacked' %(region_key,unit_num,type_key))
                plt.savefig('param_hists_%s_unit%s_%s_stacked' %(region_key,str(unit_num).zfill(2),type_key))
                plt.clf()

        #concatenated hists
        bfr_cue_succ_flattened_alpha = np.ndarray.flatten(bfr_cue_succ_params[:,:,0])
        bfr_cue_succ_flattened_beta = np.ndarray.flatten(bfr_cue_succ_params[:,:,1])
        bfr_cue_succ_flattened_k = np.ndarray.flatten(bfr_cue_succ_params[:,:,2])
        bfr_cue_succ_flattened_alpha = bfr_cue_succ_flattened_alpha[bfr_cue_succ_flattened_alpha != 9999]
        bfr_cue_succ_flattened_beta = bfr_cue_succ_flattened_beta[bfr_cue_succ_flattened_beta != 9999]
        bfr_cue_succ_flattened_k = bfr_cue_succ_flattened_k[bfr_cue_succ_flattened_k != 9999]

        aft_cue_succ_flattened_alpha = np.ndarray.flatten(aft_cue_succ_params[:,:,0])
        aft_cue_succ_flattened_beta = np.ndarray.flatten(aft_cue_succ_params[:,:,1])
        aft_cue_succ_flattened_k = np.ndarray.flatten(aft_cue_succ_params[:,:,2])
        aft_cue_succ_flattened_alpha = aft_cue_succ_flattened_alpha[aft_cue_succ_flattened_alpha != 9999]
        aft_cue_succ_flattened_beta = aft_cue_succ_flattened_beta[aft_cue_succ_flattened_beta != 9999]
        aft_cue_succ_flattened_k = aft_cue_succ_flattened_k[aft_cue_succ_flattened_k != 9999]

        bfr_result_succ_flattened_alpha = np.ndarray.flatten(bfr_result_succ_params[:,:,0])
        bfr_result_succ_flattened_beta = np.ndarray.flatten(bfr_result_succ_params[:,:,1])
        bfr_result_succ_flattened_k = np.ndarray.flatten(bfr_result_succ_params[:,:,2])
        bfr_result_succ_flattened_alpha = bfr_result_succ_flattened_alpha[bfr_result_succ_flattened_alpha != 9999]
        bfr_result_succ_flattened_beta = bfr_result_succ_flattened_beta[bfr_result_succ_flattened_beta != 9999]
        bfr_result_succ_flattened_k = bfr_result_succ_flattened_k[bfr_result_succ_flattened_k != 9999]

        aft_result_succ_flattened_alpha = np.ndarray.flatten(aft_result_succ_params[:,:,0])
        aft_result_succ_flattened_beta = np.ndarray.flatten(aft_result_succ_params[:,:,1])
        aft_result_succ_flattened_k = np.ndarray.flatten(aft_result_succ_params[:,:,2])
        aft_result_succ_flattened_alpha = aft_result_succ_flattened_alpha[aft_result_succ_flattened_alpha != 9999]
        aft_result_succ_flattened_beta = aft_result_succ_flattened_beta[aft_result_succ_flattened_beta != 9999]
        aft_result_succ_flattened_k = aft_result_succ_flattened_k[aft_result_succ_flattened_k != 9999]

        bfr_cue_fail_flattened_alpha = np.ndarray.flatten(bfr_cue_fail_params[:,:,0])
        bfr_cue_fail_flattened_beta = np.ndarray.flatten(bfr_cue_fail_params[:,:,1])
        bfr_cue_fail_flattened_k = np.ndarray.flatten(bfr_cue_fail_params[:,:,2])
        bfr_cue_fail_flattened_alpha = bfr_cue_fail_flattened_alpha[bfr_cue_fail_flattened_alpha != 9999]
        bfr_cue_fail_flattened_beta = bfr_cue_fail_flattened_beta[bfr_cue_fail_flattened_beta != 9999]
        bfr_cue_fail_flattened_k = bfr_cue_fail_flattened_k[bfr_cue_fail_flattened_k != 9999]

        aft_cue_fail_flattened_alpha = np.ndarray.flatten(aft_cue_fail_params[:,:,0])
        aft_cue_fail_flattened_beta = np.ndarray.flatten(aft_cue_fail_params[:,:,1])
        aft_cue_fail_flattened_k = np.ndarray.flatten(aft_cue_fail_params[:,:,2])
        aft_cue_fail_flattened_alpha = aft_cue_fail_flattened_alpha[aft_cue_fail_flattened_alpha != 9999]
        aft_cue_fail_flattened_beta = aft_cue_fail_flattened_beta[aft_cue_fail_flattened_beta != 9999]
        aft_cue_fail_flattened_k = aft_cue_fail_flattened_k[aft_cue_fail_flattened_k != 9999]

        bfr_result_fail_flattened_alpha = np.ndarray.flatten(bfr_result_fail_params[:,:,0])
        bfr_result_fail_flattened_beta = np.ndarray.flatten(bfr_result_fail_params[:,:,1])
        bfr_result_fail_flattened_k = np.ndarray.flatten(bfr_result_fail_params[:,:,2])
        bfr_result_fail_flattened_alpha = bfr_result_fail_flattened_alpha[bfr_result_fail_flattened_alpha != 9999]
        bfr_result_fail_flattened_beta = bfr_result_fail_flattened_beta[bfr_result_fail_flattened_beta != 9999]
        bfr_result_fail_flattened_k = bfr_result_fail_flattened_k[bfr_result_fail_flattened_k != 9999]

        aft_result_fail_flattened_alpha = np.ndarray.flatten(aft_result_fail_params[:,:,0])
        aft_result_fail_flattened_beta = np.ndarray.flatten(aft_result_fail_params[:,:,1])
        aft_result_fail_flattened_k = np.ndarray.flatten(aft_result_fail_params[:,:,2])
        aft_result_fail_flattened_alpha = aft_result_fail_flattened_alpha[aft_result_fail_flattened_alpha != 9999]
        aft_result_fail_flattened_beta = aft_result_fail_flattened_beta[aft_result_fail_flattened_beta != 9999]
        aft_result_fail_flattened_k = aft_result_fail_flattened_k[aft_result_fail_flattened_k != 9999]

        ax = plt.gca()
        plt.subplot(4,3,1)
        plt.hist([bfr_cue_succ_flattened_alpha,bfr_cue_fail_flattened_alpha], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
        plt.title('before cue: alpha',fontsize='small')
        plt.subplot(4,3,2)
        plt.hist([bfr_cue_succ_flattened_beta,bfr_cue_fail_flattened_beta], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
        plt.title('beta',fontsize='small')
        plt.subplot(4,3,3)
        plt.hist([bfr_cue_succ_flattened_k,bfr_cue_fail_flattened_k], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
        plt.title('k',fontsize='small')
                
        plt.subplot(4,3,4)
        plt.hist([aft_cue_succ_flattened_alpha,aft_cue_fail_flattened_alpha], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
        plt.title('after cue: alpha',fontsize='small')
        plt.subplot(4,3,5)
        plt.hist(np.ndarray.flatten(aft_cue_params[:,:,1]))
        plt.hist([aft_cue_succ_flattened_beta,aft_cue_fail_flattened_beta], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
        plt.title('beta',fontsize='small')
        plt.subplot(4,3,6)
        plt.hist([aft_cue_succ_flattened_k,aft_cue_fail_flattened_k], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
        plt.title('k',fontsize='small')

        plt.subplot(4,3,7)
        plt.hist([bfr_result_succ_flattened_alpha,bfr_result_fail_flattened_alpha], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
        plt.title('before result: alpha',fontsize='small')
        plt.subplot(4,3,8)
        plt.hist([bfr_result_succ_flattened_beta,bfr_result_fail_flattened_beta], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
        plt.title('beta',fontsize='small')
        plt.subplot(4,3,9)
        plt.hist([bfr_result_succ_flattened_k,bfr_result_fail_flattened_k], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
        plt.title('k',fontsize='small')

        plt.subplot(4,3,10)
        plt.hist([aft_result_succ_flattened_alpha,aft_result_fail_flattened_alpha], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
        plt.title('after result: alpha',fontsize='small')
        plt.subplot(4,3,11)
        plt.hist([aft_result_succ_flattened_beta,aft_result_fail_flattened_beta], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
        plt.title('beta',fontsize='small')
        plt.subplot(4,3,12)
        plt.hist([aft_result_succ_flattened_k,aft_result_fail_flattened_k], stacked=True, label=('succ','fail'),color=('cornflowerblue','darkmagenta'))
        plt.title('k',fontsize='small')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9,right=0.85)
        plt.legend(bbox_to_anchor=[1.6,3.2],fontsize='small')
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.suptitle('Region %s conc: param hists %s stacked' %(region_key,type_key))
        plt.savefig('param_hists_conc_%s_%s_stacked' %(region_key,type_key))
        plt.clf()
        
        return_dict = {}
        return(return_dict)

#TODO Mult val/motiv vectors as stacked hist? might be too much



###############################################
#start ########################################
###############################################

bins_before = int(time_before / float(bin_size) * 1000)  #neg for now
bins_after = int(time_after / float(bin_size) * 1000)   
baseline_bins = int(baseline_time / float(bin_size) * 1000) #neg

print 'bin size: %s' %(bin_size)
print 'time before: %s, time after: %s, baseline time: %s' %(time_before,time_after,baseline_time)
print 'nlize: %s, sqrt: %s' %(normalize_bool,sqrt_bool)

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
        

print 'modeling'
for region_key,region_value in data_dict_all.iteritems():
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

        bfr_cue_model = make_model(bfr_cue,condensed,region_key,'bfr_cue')
        aft_cue_model = make_model(aft_cue,condensed,region_key,'aft_cue')
        bfr_result_model = make_model(bfr_result,condensed,region_key,'bfr_result')
        aft_result_model = make_model(aft_result,condensed,region_key,'aft_result')

        bfr_cue_model_succ = make_model(bfr_cue_succ,succ,region_key,'bfr_cue_succ')
        aft_cue_model_succ = make_model(aft_cue_succ,succ,region_key,'aft_cue_succ')
        bfr_result_model_succ = make_model(bfr_result_succ,succ,region_key,'bfr_result_succ')
        aft_result_model_succ = make_model(aft_result_succ,succ,region_key,'aft_result_succ')

        bfr_cue_model_fail = make_model(bfr_cue_fail,fail,region_key,'bfr_cue_fail')
        aft_cue_model_fail = make_model(aft_cue_fail,fail,region_key,'aft_cue_fail')
        bfr_result_model_fail = make_model(bfr_result_fail,fail,region_key,'bfr_result_fail')
        aft_result_model_fail = make_model(aft_result_fail,fail,region_key,'aft_result_fail')

        model_return= {'bfr_cue_model':bfr_cue_model,'aft_cue_model':aft_cue_model,'bfr_result_model':bfr_result_model,'aft_result_model':aft_result_model,'bfr_cue_model_succ':bfr_cue_model_succ,'aft_cue_model_succ':aft_cue_model_succ,'bfr_result_model_succ':bfr_result_model_succ,'aft_result_model_succ':aft_result_model_succ,'bfr_cue_model_fail':bfr_cue_model_fail,'aft_cue_model_fail':aft_cue_model_fail,'bfr_result_model_fail':bfr_result_model_fail,'aft_result_model_fail':aft_result_model_fail}

        data_dict_all[region_key]['model_return'] = model_return

        #temp = plot_avgs(bfr_cue_model,aft_cue_model,bfr_result_model,aft_result_model,bfr_cue,aft_cue,bfr_result,aft_result,region_key,'all')
        #temp_succ = plot_avgs(bfr_cue_model_succ,aft_cue_model_succ,bfr_result_model_succ,aft_result_model_succ,bfr_cue,aft_cue,bfr_result,aft_result,region_key,'succ')
        #temp_fail = plot_avgs(bfr_cue_model_fail,aft_cue_model_fail,bfr_result_model_fail,aft_result_model_fail,bfr_cue,aft_cue,bfr_result,aft_result,region_key,'fail')

        temp_stacked = plot_stacked_avgs(bfr_cue_model,aft_cue_model,bfr_result_model,aft_result_model,bfr_cue,aft_cue,bfr_result,aft_result,condensed,region_key,'all_flat')

#save

