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

zscore_bool = False
gaussian_bool = True
gauss_sigma = 30 #in ms

plot_bool = True

aft_cue_time = 1.0
bfr_result_time = 0.5
aft_result_time = 1.0


###############################
### functions #################
###############################
def hist_and_smooth_data(spike_data):

        max_spike_ts = 0
        for i in range(len(spike_data)):
                if np.amax(spike_data[i]) > max_spike_ts:
                        max_spike_ts = np.amax(spike_data[i])
        
        max_bin_num = int(np.ceil(max_spike_ts) / float(bin_size) * 1000)
        hist_data = np.zeros((len(spike_data),max_bin_num))
        hist_bins = np.zeros((len(spike_data),max_bin_num))
        for i in range(len(spike_data)):
                total_bin_range = np.arange(0,int(np.ceil(spike_data[i].max())),bin_size/1000.0)
                hist,bins = np.histogram(spike_data[i],bins=total_bin_range,range=(0,int(np.ceil(spike_data[i].max()))),normed=False,density=False)
                #pdb.set_trace()
                hist_data[i,0:len(hist)] = hist
                hist_bins[i,0:len(bins)] = bins

        if zscore_bool and gaussian_bool:
                print 'ERROR: dont do two smoothing types at once'
        elif zscore_bool:
                smoothed = stats.zscore(hist_data,axis=1)
        elif gaussian_bool:
                smoothed = ndimage.filters.gaussian_filter1d(hist_data,gauss_sigma,axis=1)
        else:
                smoothed = {}

        return_dict = {'hist_data':hist_data,'hist_bins':hist_bins,'smoothed':smoothed}
        return(return_dict)
        


def run_corr(binned_data,condensed):
        
        #for each unit, build matrix. Col 0 = hist, col 1 =  r num for each bin, col2 = pnum, col3 = v num, col4 = mnum, col5 = succ/fail
        #TODO make combinations of rewarding and succ, punishing and fail, etc?

        #dim0 = num bins, dim1 = 5 (see above), dim2 = num units
        corr_input_matrix = np.zeros((np.shape(binned_data)[1],6,np.shape(binned_data)[0]))
        corr_output = np.zeros((np.shape(binned_data)[0],5))

        for unit_num in range(np.shape(binned_data)[0]):
                corr_input_matrix[:,0,unit_num] = binned_data[unit_num,:]
                for cond_ind in range(np.shape(condensed)[0]-1):
					#pdb.set_trace()
					corr_input_matrix[int(condensed[cond_ind,8]):int(condensed[cond_ind+1,8]),1,unit_num] = condensed[cond_ind,3]
					corr_input_matrix[int(condensed[cond_ind,8]):int(condensed[cond_ind+1,8]),2,unit_num] = condensed[cond_ind,4]
					corr_input_matrix[int(condensed[cond_ind,8]):int(condensed[cond_ind+1,8]),3,unit_num] = condensed[cond_ind,6]
					corr_input_matrix[int(condensed[cond_ind,8]):int(condensed[cond_ind+1,8]),4,unit_num] = condensed[cond_ind,7]
					corr_input_matrix[int(condensed[cond_ind,8]):int(condensed[cond_ind+1,8]),5,unit_num] = condensed[cond_ind,5]

                corr_df = pd.DataFrame({'binned_rate':corr_input_matrix[:,0,unit_num],'rnum':corr_input_matrix[:,1,unit_num],'pnum':corr_input_matrix[:,2,unit_num],'val':corr_input_matrix[:,3,unit_num],'mtv':corr_input_matrix[:,4,unit_num],'result':corr_input_matrix[:,5,unit_num]})
                corr_output[unit_num,0] = corr_df['binned_rate'].corr(corr_df['rnum'])
                corr_output[unit_num,1] = corr_df['binned_rate'].corr(corr_df['pnum'])
                corr_output[unit_num,2] = corr_df['binned_rate'].corr(corr_df['val'])
                corr_output[unit_num,3] = corr_df['binned_rate'].corr(corr_df['mtv'])
                corr_output[unit_num,4] = corr_df['binned_rate'].corr(corr_df['result'])
        
        return_dict = {'corr_input_matrix':corr_input_matrix,'corr_output':corr_output}

	return(return_dict)


def plot_corr(corr_input,corr_output,condensed,region_key):
        aft_cue_bins = int(aft_cue_time *1000/bin_size)
        bfr_res_bins = int(bfr_result_time * 1000/bin_size)
        aft_res_bins = int(aft_result_time * 1000/bin_size)

        r_corr_order = np.argsort(corr_output[:,0])
        p_corr_order = np.argsort(corr_output[:,1])
        val_corr_order = np.argsort(corr_output[:,2])
        mtv_corr_order = np.argsort(corr_output[:,3])
        res_corr_order = np.argsort(corr_output[:,4])

        order_dict = {'r_corr_order':r_corr_order,'p_corr_order':p_corr_order,'val_corr_order':val_corr_order,'mtv_corr_order':mtv_corr_order,'res_corr_order':res_corr_order,'condensed':condensed}
        sio.savemat('order_dict_%s' %(region_key),{'order_dict':order_dict},format='5')
		
        #plot top 10%? for now 10
        num_plot = 10
        
        top_inds = np.zeros((num_plot,5))
        btm_inds = np.zeros((num_plot,5))
        total_ind = np.shape(r_corr_order)[0] - 1

        for i in range(num_plot):
                top_inds[i,0] = np.where(r_corr_order == i)[0][0].astype(int)
                top_inds[i,1] = np.where(p_corr_order == i)[0][0].astype(int)
                top_inds[i,2] = np.where(val_corr_order == i)[0][0].astype(int)
                top_inds[i,3] = np.where(mtv_corr_order == i)[0][0].astype(int)
                top_inds[i,4] = np.where(res_corr_order == i)[0][0].astype(int)

                btm_inds[i,0] = np.where(r_corr_order == (total_ind - i))[0][0].astype(int)
                btm_inds[i,1] = np.where(p_corr_order == (total_ind - i))[0][0].astype(int)
                btm_inds[i,2] = np.where(val_corr_order == (total_ind - i))[0][0].astype(int)
                btm_inds[i,3] = np.where(mtv_corr_order == (total_ind - i))[0][0].astype(int)
                btm_inds[i,4] = np.where(res_corr_order == (total_ind - i))[0][0].astype(int)
        #plot r    
        r0 = np.where(condensed[:,3] == 0)[0].astype(int)
        r1 = np.where(condensed[:,3] == 1)[0].astype(int)
        r2 = np.where(condensed[:,3] == 2)[0].astype(int)
        r3 = np.where(condensed[:,3] == 3)[0].astype(int)

        #unit_num = 0
        r0_binned = []
        r1_binned = []
        r2_binned = []
        r3_binned = []

        top_r0_cue = []
        top_r0_res = []
        top_r1_cue = []
        top_r1_res = []
        top_r2_cue = []
        top_r2_res = []
        top_r3_cue = []
        top_r3_res = []

        btm_r0_cue = []
        btm_r0_res = []
        btm_r1_cue = []
        btm_r1_res = []
        btm_r2_cue = []
        btm_r2_res = []
        btm_r3_cue = []
        btm_r3_res = []

        top_r0_cue_avg = []
        top_r0_res_avg = []
        top_r1_cue_avg = []
        top_r1_res_avg = []
        top_r2_cue_avg = []
        top_r2_res_avg = []
        top_r3_cue_avg = []
        top_r3_res_avg = []

        btm_r0_cue_avg = []
        btm_r0_res_avg = []
        btm_r1_cue_avg = []
        btm_r1_res_avg = []
        btm_r2_cue_avg = []
        btm_r2_res_avg = []
        btm_r3_cue_avg = []
        btm_r3_res_avg = []

        #pdb.set_trace()
        
        for j in range(num_plot):
                top_unit_ind = top_inds[j,0].astype(int)
                btm_unit_ind = btm_inds[j,0].astype(int)

                for i in range(len(r0)):
                        if r0[i] != np.shape(condensed)[0]-1:
                                #print r0[i]
                                r0_binned = np.append(r0_binned,corr_input[condensed[r0[i].astype(int),8].astype(int):condensed[r0[i].astype(int)+1,8].astype(int),0,top_unit_ind])
                        
                                top_cue_temp = corr_input[condensed[r0[i],8].astype(int):condensed[r0[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[r0[i],9].astype(int)-bfr_res_bins:condensed[r0[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_r0_cue_avg = np.append(top_r0_cue_avg,np.mean(top_cue_temp))
                                top_r0_res_avg = np.append(top_r0_res_avg,np.mean(top_res_temp))
                                                        
                                top_r0_cue = np.append(top_r0_cue,top_cue_temp)
                                top_r0_res = np.append(top_r0_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[r0[i],8].astype(int):condensed[r0[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[r0[i],9].astype(int)-bfr_res_bins:condensed[r0[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_r0_cue_avg = np.append(btm_r0_cue_avg,np.mean(btm_cue_temp))
                                btm_r0_res_avg = np.append(btm_r0_res_avg,np.mean(btm_res_temp))
                        
                                btm_r0_cue = np.append(btm_r0_cue,btm_cue_temp)
                                btm_r0_res = np.append(btm_r0_res,btm_res_temp)

                for i in range(len(r1)):
                        if r1[i] != np.shape(condensed)[0]-1:
                                r1_binned = np.append(r1_binned,corr_input[condensed[r1[i],8].astype(int):condensed[r1[i]+1,8].astype(int),0,top_unit_ind])
                                
                                top_cue_temp = corr_input[condensed[r1[i],8].astype(int):condensed[r1[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[r1[i],9].astype(int)-bfr_res_bins:condensed[r1[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000

                                top_r1_cue_avg = np.append(top_r1_cue_avg,np.mean(top_cue_temp))
                                top_r1_res_avg = np.append(top_r1_res_avg,np.mean(top_res_temp))
                        
                                top_r1_cue = np.append(top_r1_cue,top_cue_temp)
                                top_r1_res = np.append(top_r1_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[r1[i],8].astype(int):condensed[r1[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[r1[i],9].astype(int)-bfr_res_bins:condensed[r1[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000

                                btm_r1_cue_avg = np.append(btm_r1_cue_avg,np.mean(btm_cue_temp))
                                btm_r1_res_avg = np.append(btm_r1_res_avg,np.mean(btm_res_temp))
                        
                                btm_r1_cue = np.append(btm_r1_cue,btm_cue_temp)
                                btm_r1_res = np.append(btm_r1_res,btm_res_temp)

                for i in range(len(r2)):
                        if r2[i] != np.shape(condensed)[0]-1:
                                r2_binned = np.append(r2_binned,corr_input[condensed[r2[i],8].astype(int):condensed[r2[i]+1,8].astype(int),0,top_unit_ind])

                                top_cue_temp = corr_input[condensed[r2[i],8].astype(int):condensed[r2[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[r2[i],9].astype(int)-bfr_res_bins:condensed[r2[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                        
                                top_r2_cue_avg = np.append(top_r2_cue_avg,np.mean(top_cue_temp))
                                top_r2_res_avg = np.append(top_r2_res_avg,np.mean(top_res_temp))

                                top_r2_cue = np.append(top_r2_cue,top_cue_temp)
                                top_r2_res = np.append(top_r2_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[r2[i],8].astype(int):condensed[r2[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[r2[i],9].astype(int)-bfr_res_bins:condensed[r2[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000

                                btm_r2_cue_avg = np.append(btm_r2_cue_avg,np.mean(btm_cue_temp))
                                btm_r2_res_avg = np.append(btm_r2_res_avg,np.mean(btm_res_temp))
                        
                                btm_r2_cue = np.append(btm_r2_cue,btm_cue_temp)
                                btm_r2_res = np.append(btm_r2_res,btm_res_temp)

                for i in range(len(r3)):
                        if r3[i] != np.shape(condensed)[0]-1:
                                r3_binned = np.append(r3_binned,corr_input[condensed[r3[i],8].astype(int):condensed[r3[i]+1,8].astype(int),0,top_unit_ind])

                                top_cue_temp = corr_input[condensed[r3[i],8].astype(int):condensed[r3[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[r3[i],9].astype(int)-bfr_res_bins:condensed[r3[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000

                                top_r3_cue_avg = np.append(top_r3_cue_avg,np.mean(top_cue_temp))
                                top_r3_res_avg = np.append(top_r3_res_avg,np.mean(top_res_temp))
                        
                                top_r3_cue = np.append(top_r3_cue,top_cue_temp)
                                top_r3_res = np.append(top_r3_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[r3[i],8].astype(int):condensed[r3[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[r3[i],9].astype(int)-bfr_res_bins:condensed[r3[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                        
                                btm_r3_cue_avg = np.append(btm_r3_cue_avg,np.mean(btm_cue_temp))
                                btm_r3_res_avg = np.append(btm_r3_res_avg,np.mean(btm_res_temp))

                                btm_r3_cue = np.append(btm_r3_cue,btm_cue_temp)
                                btm_r3_res = np.append(btm_r3_res,btm_res_temp)


        rlist = [r0_binned, r1_binned, r2_binned, r3_binned]
        top_r_cue_list = [top_r0_cue, top_r1_cue, top_r2_cue, top_r3_cue]
        top_r_res_list = [top_r0_res, top_r1_res, top_r2_res, top_r3_res]
        btm_r_cue_list = [btm_r0_cue, btm_r1_cue, btm_r2_cue, btm_r3_cue]
        btm_r_res_list = [btm_r0_res, btm_r1_res, btm_r2_res, btm_r3_res]

        
        r_avgs = [top_r0_cue_avg,top_r0_res_avg,top_r1_cue_avg,top_r1_res_avg,top_r2_cue_avg,top_r2_res_avg,top_r3_cue_avg,top_r3_res_avg,btm_r0_cue_avg,btm_r0_res_avg,btm_r1_cue_avg,btm_r1_res_avg,btm_r2_cue_avg,btm_r2_res_avg,btm_r3_cue_avg,btm_r3_res_avg]
        
        sio.savemat('r_avgs_%s' %(region_key),{'r_avgs':r_avgs},format='5')

        #ax = plt.gca()
        #plt.subplot(3,1,1)
        #plt.boxplot(rlist)
        #plt.subplot(3,1,2)
        #plt.boxplot(r_cue_list)
        #plt.subplot(3,1,3)
        #plt.boxplot(r_res_list)
        #plt.savefig('r_boxplot_test_%s' %(region_key))
        #plt.clf()

        #plot p
        p0 = np.where(condensed[:,4] == 0)[0].astype(int)
        p1 = np.where(condensed[:,4] == 1)[0].astype(int)
        p2 = np.where(condensed[:,4] == 2)[0].astype(int)
        p3 = np.where(condensed[:,4] == 3)[0].astype(int)

        #unit_num = 0
        top_p0_cue = []
        top_p0_res = []
        top_p1_cue = []
        top_p1_res = []
        top_p2_cue = []
        top_p2_res = []
        top_p3_cue = []
        top_p3_res = []

        btm_p0_cue = []
        btm_p0_res = []
        btm_p1_cue = []
        btm_p1_res = []
        btm_p2_cue = []
        btm_p2_res = []
        btm_p3_cue = []
        btm_p3_res = []

        top_p0_cue_avg = []
        top_p0_res_avg = []
        top_p1_cue_avg = []
        top_p1_res_avg = []
        top_p2_cue_avg = []
        top_p2_res_avg = []
        top_p3_cue_avg = []
        top_p3_res_avg = []

        btm_p0_cue_avg = []
        btm_p0_res_avg = []
        btm_p1_cue_avg = []
        btm_p1_res_avg = []
        btm_p2_cue_avg = []
        btm_p2_res_avg = []
        btm_p3_cue_avg = []
        btm_p3_res_avg = []

        for j in range(num_plot):
                top_unit_ind = top_inds[j,1].astype(int)
                btm_unit_ind = btm_inds[j,1].astype(int)

                for i in range(len(p0)):
                        if p0[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[p0[i],8].astype(int):condensed[p0[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[p0[i],9].astype(int)-bfr_res_bins:condensed[p0[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_p0_cue_avg = np.append(top_p0_cue_avg,np.mean(top_cue_temp))
                                top_p0_res_avg = np.append(top_p0_res_avg,np.mean(top_res_temp))
                                                        
                                top_p0_cue = np.append(top_p0_cue,top_cue_temp)
                                top_p0_res = np.append(top_p0_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[p0[i],8].astype(int):condensed[p0[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[p0[i],9].astype(int)-bfr_res_bins:condensed[p0[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_p0_cue_avg = np.append(btm_p0_cue_avg,np.mean(btm_cue_temp))
                                btm_p0_res_avg = np.append(btm_p0_res_avg,np.mean(btm_res_temp))
                        
                                btm_p0_cue = np.append(btm_p0_cue,btm_cue_temp)
                                btm_p0_res = np.append(btm_p0_res,btm_res_temp)

                for i in range(len(p1)):
                        if p1[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[p1[i],8].astype(int):condensed[p1[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[p1[i],9].astype(int)-bfr_res_bins:condensed[p1[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000

                                top_p1_cue_avg = np.append(top_p1_cue_avg,np.mean(top_cue_temp))
                                top_p1_res_avg = np.append(top_p1_res_avg,np.mean(top_res_temp))
                        
                                top_p1_cue = np.append(top_p1_cue,top_cue_temp)
                                top_p1_res = np.append(top_p1_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[p1[i],8].astype(int):condensed[p1[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[p1[i],9].astype(int)-bfr_res_bins:condensed[p1[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000

                                btm_p1_cue_avg = np.append(btm_p1_cue_avg,np.mean(btm_cue_temp))
                                btm_p1_res_avg = np.append(btm_p1_res_avg,np.mean(btm_res_temp))
                        
                                btm_p1_cue = np.append(btm_p1_cue,btm_cue_temp)
                                btm_p1_res = np.append(btm_p1_res,btm_res_temp)

                for i in range(len(p2)):
                        if p2[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[p2[i],8].astype(int):condensed[p2[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[p2[i],9].astype(int)-bfr_res_bins:condensed[p2[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                        
                                top_p2_cue_avg = np.append(top_p2_cue_avg,np.mean(top_cue_temp))
                                top_p2_res_avg = np.append(top_p2_res_avg,np.mean(top_res_temp))

                                top_p2_cue = np.append(top_p2_cue,top_cue_temp)
                                top_p2_res = np.append(top_p2_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[p2[i],8].astype(int):condensed[p2[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[p2[i],9].astype(int)-bfr_res_bins:condensed[p2[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000

                                btm_p2_cue_avg = np.append(btm_p2_cue_avg,np.mean(btm_cue_temp))
                                btm_p2_res_avg = np.append(btm_p2_res_avg,np.mean(btm_res_temp))
                        
                                btm_p2_cue = np.append(btm_p2_cue,btm_cue_temp)
                                btm_p2_res = np.append(btm_p2_res,btm_res_temp)

                for i in range(len(p3)):
                                top_cue_temp = corr_input[condensed[p3[i],8].astype(int):condensed[p3[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[p3[i],9].astype(int)-bfr_res_bins:condensed[p3[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000

                                top_p3_cue_avg = np.append(top_p3_cue_avg,np.mean(top_cue_temp))
                                top_p3_res_avg = np.append(top_p3_res_avg,np.mean(top_res_temp))
                        
                                top_p3_cue = np.append(top_p3_cue,top_cue_temp)
                                top_p3_res = np.append(top_p3_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[p3[i],8].astype(int):condensed[p3[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[p3[i],9].astype(int)-bfr_res_bins:condensed[p3[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                        
                                btm_p3_cue_avg = np.append(btm_p3_cue_avg,np.mean(btm_cue_temp))
                                btm_p3_res_avg = np.append(btm_p3_res_avg,np.mean(btm_res_temp))

                                btm_p3_cue = np.append(btm_p3_cue,btm_cue_temp)
                                btm_p3_res = np.append(btm_p3_res,btm_res_temp)


        top_p_cue_list = [top_p0_cue, top_p1_cue, top_p2_cue, top_p3_cue]
        top_p_res_list = [top_p0_res, top_p1_res, top_p2_res, top_p3_res]
        btm_p_cue_list = [btm_p0_cue, btm_p1_cue, btm_p2_cue, btm_p3_cue]
        btm_p_res_list = [btm_p0_res, btm_p1_res, btm_p2_res, btm_p3_res]
        
        p_avgs = [top_p0_cue_avg,top_p0_res_avg,top_p1_cue_avg,top_p1_res_avg,top_p2_cue_avg,top_p2_res_avg,top_p3_cue_avg,top_p3_res_avg,btm_p0_cue_avg,btm_p0_res_avg,btm_p1_cue_avg,btm_p1_res_avg,btm_p2_cue_avg,btm_p2_res_avg,btm_p3_cue_avg,btm_p3_res_avg]
        
        sio.savemat('p_avgs_%s' %(region_key),{'p_avgs':p_avgs},format='5')

        #plot v
        v_3 = np.where(condensed[:,6] == -3)[0].astype(int)
        v_2 = np.where(condensed[:,6] == -2)[0].astype(int)
        v_1 = np.where(condensed[:,6] == -1)[0].astype(int)
        v0 = np.where(condensed[:,6] == 0)[0].astype(int)
        v1 = np.where(condensed[:,6] == 1)[0].astype(int)
        v2 = np.where(condensed[:,6] == 2)[0].astype(int)
        v3 = np.where(condensed[:,6] == 3)[0].astype(int)

        #unit_num = 0
        top_v_3_cue = []
        top_v_3_res = []
        top_v_2_cue = []
        top_v_2_res = []
        top_v_1_cue = []
        top_v_1_res = []
        top_v0_cue = []
        top_v0_res = []
        top_v1_cue = []
        top_v1_res = []
        top_v2_cue = []
        top_v2_res = []
        top_v3_cue = []
        top_v3_res = []

        btm_v_3_cue = []
        btm_v_3_res = []
        btm_v_2_cue = []
        btm_v_2_res = []
        btm_v_1_cue = []
        btm_v_1_res = []
        btm_v0_cue = []
        btm_v0_res = []
        btm_v1_cue = []
        btm_v1_res = []
        btm_v2_cue = []
        btm_v2_res = []
        btm_v3_cue = []
        btm_v3_res = []

        top_v_3_cue_avg = []
        top_v_3_res_avg = []
        top_v_2_cue_avg = []
        top_v_2_res_avg = []
        top_v_1_cue_avg = []
        top_v_1_res_avg = []
        top_v0_cue_avg = []
        top_v0_res_avg = []
        top_v1_cue_avg = []
        top_v1_res_avg = []
        top_v2_cue_avg = []
        top_v2_res_avg = []
        top_v3_cue_avg = []
        top_v3_res_avg = []

        btm_v_3_cue_avg = []
        btm_v_3_res_avg = []
        btm_v_2_cue_avg = []
        btm_v_2_res_avg = []
        btm_v_1_cue_avg = []
        btm_v_1_res_avg = []
        btm_v0_cue_avg = []
        btm_v0_res_avg = []
        btm_v1_cue_avg = []
        btm_v1_res_avg = []
        btm_v2_cue_avg = []
        btm_v2_res_avg = []
        btm_v3_cue_avg = []
        btm_v3_res_avg = []

        for j in range(num_plot):
                top_unit_ind = top_inds[j,2].astype(int)
                btm_unit_ind = btm_inds[j,2].astype(int)

                for i in range(len(v_3)):
                        if v_3[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[v_3[i],8].astype(int):condensed[v_3[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[v_3[i],9].astype(int)-bfr_res_bins:condensed[v_3[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_v_3_cue_avg = np.append(top_v_3_cue_avg,np.mean(top_cue_temp))
                                top_v_3_res_avg = np.append(top_v_3_res_avg,np.mean(top_res_temp))
                                                        
                                top_v_3_cue = np.append(top_v_3_cue,top_cue_temp)
                                top_v_3_res = np.append(top_v_3_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[v_3[i],8].astype(int):condensed[v_3[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[v_3[i],9].astype(int)-bfr_res_bins:condensed[v_3[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_v_3_cue_avg = np.append(btm_v_3_cue_avg,np.mean(btm_cue_temp))
                                btm_v_3_res_avg = np.append(btm_v_3_res_avg,np.mean(btm_res_temp))
                        
                                btm_v_3_cue = np.append(btm_v_3_cue,btm_cue_temp)
                                btm_v_3_res = np.append(btm_v_3_res,btm_res_temp)

                for i in range(len(v_2)):
                        if v_2[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[v_2[i],8].astype(int):condensed[v_2[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[v_2[i],9].astype(int)-bfr_res_bins:condensed[v_2[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_v_2_cue_avg = np.append(top_v_2_cue_avg,np.mean(top_cue_temp))
                                top_v_2_res_avg = np.append(top_v_2_res_avg,np.mean(top_res_temp))
                                                        
                                top_v_2_cue = np.append(top_v_2_cue,top_cue_temp)
                                top_v_2_res = np.append(top_v_2_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[v_2[i],8].astype(int):condensed[v_2[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[v_2[i],9].astype(int)-bfr_res_bins:condensed[v_2[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_v_2_cue_avg = np.append(btm_v_2_cue_avg,np.mean(btm_cue_temp))
                                btm_v_2_res_avg = np.append(btm_v_2_res_avg,np.mean(btm_res_temp))
                        
                                btm_v_2_cue = np.append(btm_v_2_cue,btm_cue_temp)
                                btm_v_2_res = np.append(btm_v_2_res,btm_res_temp)

                for i in range(len(v_1)):
                        if v_1[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[v_1[i],8].astype(int):condensed[v_1[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[v_1[i],9].astype(int)-bfr_res_bins:condensed[v_1[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_v_1_cue_avg = np.append(top_v_1_cue_avg,np.mean(top_cue_temp))
                                top_v_1_res_avg = np.append(top_v_1_res_avg,np.mean(top_res_temp))
                                                        
                                top_v_1_cue = np.append(top_v_1_cue,top_cue_temp)
                                top_v_1_res = np.append(top_v_1_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[v_1[i],8].astype(int):condensed[v_1[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[v_1[i],9].astype(int)-bfr_res_bins:condensed[v_1[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_v_1_cue_avg = np.append(btm_v_1_cue_avg,np.mean(btm_cue_temp))
                                btm_v_1_res_avg = np.append(btm_v_1_res_avg,np.mean(btm_res_temp))
                        
                                btm_v_1_cue = np.append(btm_v_1_cue,btm_cue_temp)
                                btm_v_1_res = np.append(btm_v_1_res,btm_res_temp)

                for i in range(len(v0)):				
                        if v0[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[v0[i],8].astype(int):condensed[v0[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[v0[i],9].astype(int)-bfr_res_bins:condensed[v0[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_v0_cue_avg = np.append(top_v0_cue_avg,np.mean(top_cue_temp))
                                top_v0_res_avg = np.append(top_v0_res_avg,np.mean(top_res_temp))
                                                        
                                top_v0_cue = np.append(top_v0_cue,top_cue_temp)
                                top_v0_res = np.append(top_v0_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[v0[i],8].astype(int):condensed[v0[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[v0[i],9].astype(int)-bfr_res_bins:condensed[v0[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_v0_cue_avg = np.append(btm_v0_cue_avg,np.mean(btm_cue_temp))
                                btm_v0_res_avg = np.append(btm_v0_res_avg,np.mean(btm_res_temp))
                        
                                btm_v0_cue = np.append(btm_v0_cue,btm_cue_temp)
                                btm_v0_res = np.append(btm_v0_res,btm_res_temp)

                for i in range(len(v1)):
                        if v1[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[v1[i],8].astype(int):condensed[v1[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[v1[i],9].astype(int)-bfr_res_bins:condensed[v1[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_v1_cue_avg = np.append(top_v1_cue_avg,np.mean(top_cue_temp))
                                top_v1_res_avg = np.append(top_v1_res_avg,np.mean(top_res_temp))
                                                        
                                top_v1_cue = np.append(top_v1_cue,top_cue_temp)
                                top_v1_res = np.append(top_v1_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[v1[i],8].astype(int):condensed[v1[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[v1[i],9].astype(int)-bfr_res_bins:condensed[v1[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_v1_cue_avg = np.append(btm_v1_cue_avg,np.mean(btm_cue_temp))
                                btm_v1_res_avg = np.append(btm_v1_res_avg,np.mean(btm_res_temp))
                        
                                btm_v1_cue = np.append(btm_v1_cue,btm_cue_temp)
                                btm_v1_res = np.append(btm_v1_res,btm_res_temp)

                for i in range(len(v2)):
                        if v2[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[v2[i],8].astype(int):condensed[v2[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[v2[i],9].astype(int)-bfr_res_bins:condensed[v2[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_v2_cue_avg = np.append(top_v2_cue_avg,np.mean(top_cue_temp))
                                top_v2_res_avg = np.append(top_v2_res_avg,np.mean(top_res_temp))
                                                        
                                top_v2_cue = np.append(top_v2_cue,top_cue_temp)
                                top_v2_res = np.append(top_v2_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[v2[i],8].astype(int):condensed[v2[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[v2[i],9].astype(int)-bfr_res_bins:condensed[v2[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_v2_cue_avg = np.append(btm_v2_cue_avg,np.mean(btm_cue_temp))
                                btm_v2_res_avg = np.append(btm_v2_res_avg,np.mean(btm_res_temp))
                        
                                btm_v2_cue = np.append(btm_v2_cue,btm_cue_temp)
                                btm_v2_res = np.append(btm_v2_res,btm_res_temp)

                for i in range(len(v3)):
                        if v3[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[v3[i],8].astype(int):condensed[v3[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[v3[i],9].astype(int)-bfr_res_bins:condensed[v3[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_v3_cue_avg = np.append(top_v3_cue_avg,np.mean(top_cue_temp))
                                top_v3_res_avg = np.append(top_v3_res_avg,np.mean(top_res_temp))
                                                        
                                top_v3_cue = np.append(top_v3_cue,top_cue_temp)
                                top_v3_res = np.append(top_v3_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[v3[i],8].astype(int):condensed[v3[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[v3[i],9].astype(int)-bfr_res_bins:condensed[v3[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_v3_cue_avg = np.append(btm_v3_cue_avg,np.mean(btm_cue_temp))
                                btm_v3_res_avg = np.append(btm_v3_res_avg,np.mean(btm_res_temp))
                        
                                btm_v3_cue = np.append(btm_v3_cue,btm_cue_temp)
                                btm_v3_res = np.append(btm_v3_res,btm_res_temp)

        top_v_cue_list = [top_v_3_cue, top_v_2_cue, top_v_1_cue, top_v0_cue, top_v1_cue, top_v2_cue, top_v3_cue]
        top_v_res_list = [top_v_3_res, top_v_2_res, top_v_1_res, top_v0_res, top_v1_res, top_v2_res, top_v3_res]
        btm_v_cue_list = [btm_v_3_cue, btm_v_2_cue, btm_v_1_cue, btm_v0_cue, btm_v1_cue, btm_v2_cue, btm_v3_cue]
        btm_v_res_list = [btm_v_3_res, btm_v_2_res, btm_v_1_res, btm_v0_res, btm_v1_res, btm_v2_res, btm_v3_res]
        
        v_avgs = [top_v_3_cue_avg,top_v_3_res_avg,top_v_2_cue_avg,top_v_2_res_avg,top_v_1_cue_avg,top_v_1_res_avg,top_v0_cue_avg,top_v0_res_avg,top_v1_cue_avg,top_v1_res_avg,top_v2_cue_avg,top_v2_res_avg,top_v3_cue_avg,top_v3_res_avg,btm_v_3_cue_avg,btm_v_3_res_avg,btm_v_2_cue_avg,btm_v_2_res_avg,btm_v_1_cue_avg,btm_v_1_res_avg,btm_v0_cue_avg,btm_v0_res_avg,btm_v1_cue_avg,btm_v1_res_avg,btm_v2_cue_avg,btm_v2_res_avg,btm_v3_cue_avg,btm_v3_res_avg]

        #pdb.set_trace()
        sio.savemat('v_avgs_%s' %(region_key),{'v_avgs':v_avgs},format='5')

        #plot m
        m0 = np.where(condensed[:,7] == 0)[0].astype(int)
        m1 = np.where(condensed[:,7] == 1)[0].astype(int)
        m2 = np.where(condensed[:,7] == 2)[0].astype(int)
        m3 = np.where(condensed[:,7] == 3)[0].astype(int)
        m4 = np.where(condensed[:,7] == 4)[0].astype(int)
        m5 = np.where(condensed[:,7] == 5)[0].astype(int)
        m6 = np.where(condensed[:,7] == 6)[0].astype(int)

        top_m0_cue = []
        top_m0_res = []
        top_m1_cue = []
        top_m1_res = []
        top_m2_cue = []
        top_m2_res = []
        top_m3_cue = []
        top_m3_res = []
        top_m4_cue = []
        top_m4_res = []
        top_m5_cue = []
        top_m5_res = []
        top_m6_cue = []
        top_m6_res = []

        btm_m0_cue = []
        btm_m0_res = []
        btm_m1_cue = []
        btm_m1_res = []
        btm_m2_cue = []
        btm_m2_res = []
        btm_m3_cue = []
        btm_m3_res = []
        btm_m4_cue = []
        btm_m4_res = []
        btm_m5_cue = []
        btm_m5_res = []
        btm_m6_cue = []
        btm_m6_res = []

        top_m0_cue_avg = []
        top_m0_res_avg = []
        top_m1_cue_avg = []
        top_m1_res_avg = []
        top_m2_cue_avg = []
        top_m2_res_avg = []
        top_m3_cue_avg = []
        top_m3_res_avg = []
        top_m4_cue_avg = []
        top_m4_res_avg = []
        top_m5_cue_avg = []
        top_m5_res_avg = []
        top_m6_cue_avg = []
        top_m6_res_avg = []

        btm_m0_cue_avg = []
        btm_m0_res_avg = []
        btm_m1_cue_avg = []
        btm_m1_res_avg = []
        btm_m2_cue_avg = []
        btm_m2_res_avg = []
        btm_m3_cue_avg = []
        btm_m3_res_avg = []
        btm_m4_cue_avg = []
        btm_m4_res_avg = []
        btm_m5_cue_avg = []
        btm_m5_res_avg = []
        btm_m6_cue_avg = []
        btm_m6_res_avg = []

        for j in range(num_plot):
                top_unit_ind = top_inds[j,3].astype(int)
                btm_unit_ind = btm_inds[j,3].astype(int)

                for i in range(len(m0)):
                        if m0[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[m0[i],8].astype(int):condensed[m0[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[m0[i],9].astype(int)-bfr_res_bins:condensed[m0[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_m0_cue_avg = np.append(top_m0_cue_avg,np.mean(top_cue_temp))
                                top_m0_res_avg = np.append(top_m0_res_avg,np.mean(top_res_temp))
                                                        
                                top_m0_cue = np.append(top_m0_cue,top_cue_temp)
                                top_m0_res = np.append(top_m0_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[m0[i],8].astype(int):condensed[m0[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[m0[i],9].astype(int)-bfr_res_bins:condensed[m0[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_m0_cue_avg = np.append(btm_m0_cue_avg,np.mean(btm_cue_temp))
                                btm_m0_res_avg = np.append(btm_m0_res_avg,np.mean(btm_res_temp))
                        
                                btm_m0_cue = np.append(btm_m0_cue,btm_cue_temp)
                                btm_m0_res = np.append(btm_m0_res,btm_res_temp)

                for i in range(len(m1)):
                        if m1[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[m1[i],8].astype(int):condensed[m1[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[m1[i],9].astype(int)-bfr_res_bins:condensed[m1[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_m1_cue_avg = np.append(top_m1_cue_avg,np.mean(top_cue_temp))
                                top_m1_res_avg = np.append(top_m1_res_avg,np.mean(top_res_temp))
                                                        
                                top_m1_cue = np.append(top_m1_cue,top_cue_temp)
                                top_m1_res = np.append(top_m1_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[m1[i],8].astype(int):condensed[m1[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[m1[i],9].astype(int)-bfr_res_bins:condensed[m1[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_m1_cue_avg = np.append(btm_m1_cue_avg,np.mean(btm_cue_temp))
                                btm_m1_res_avg = np.append(btm_m1_res_avg,np.mean(btm_res_temp))
                        
                                btm_m1_cue = np.append(btm_m1_cue,btm_cue_temp)
                                btm_m1_res = np.append(btm_m1_res,btm_res_temp)

                for i in range(len(m2)):
                        if m2[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[m2[i],8].astype(int):condensed[m2[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[m2[i],9].astype(int)-bfr_res_bins:condensed[m2[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_m2_cue_avg = np.append(top_m2_cue_avg,np.mean(top_cue_temp))
                                top_m2_res_avg = np.append(top_m2_res_avg,np.mean(top_res_temp))
                                                        
                                top_m2_cue = np.append(top_m2_cue,top_cue_temp)
                                top_m2_res = np.append(top_m2_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[m2[i],8].astype(int):condensed[m2[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[m2[i],9].astype(int)-bfr_res_bins:condensed[m2[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_m2_cue_avg = np.append(btm_m2_cue_avg,np.mean(btm_cue_temp))
                                btm_m2_res_avg = np.append(btm_m2_res_avg,np.mean(btm_res_temp))
                        
                                btm_m2_cue = np.append(btm_m2_cue,btm_cue_temp)
                                btm_m2_res = np.append(btm_m2_res,btm_res_temp)

                for i in range(len(m3)):
                        if m3[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[m3[i],8].astype(int):condensed[m3[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[m3[i],9].astype(int)-bfr_res_bins:condensed[m3[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_m3_cue_avg = np.append(top_m3_cue_avg,np.mean(top_cue_temp))
                                top_m3_res_avg = np.append(top_m3_res_avg,np.mean(top_res_temp))
                                                        
                                top_m3_cue = np.append(top_m3_cue,top_cue_temp)
                                top_m3_res = np.append(top_m3_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[m3[i],8].astype(int):condensed[m3[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[m3[i],9].astype(int)-bfr_res_bins:condensed[m3[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_m3_cue_avg = np.append(btm_m3_cue_avg,np.mean(btm_cue_temp))
                                btm_m3_res_avg = np.append(btm_m3_res_avg,np.mean(btm_res_temp))
                        
                                btm_m3_cue = np.append(btm_m3_cue,btm_cue_temp)
                                btm_m3_res = np.append(btm_m3_res,btm_res_temp)

                for i in range(len(m4)):
                        if m4[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[m4[i],8].astype(int):condensed[m4[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[m4[i],9].astype(int)-bfr_res_bins:condensed[m4[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_m4_cue_avg = np.append(top_m4_cue_avg,np.mean(top_cue_temp))
                                top_m4_res_avg = np.append(top_m4_res_avg,np.mean(top_res_temp))
                                                        
                                top_m4_cue = np.append(top_m4_cue,top_cue_temp)
                                top_m4_res = np.append(top_m4_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[m4[i],8].astype(int):condensed[m4[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[m4[i],9].astype(int)-bfr_res_bins:condensed[m4[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_m4_cue_avg = np.append(btm_m4_cue_avg,np.mean(btm_cue_temp))
                                btm_m4_res_avg = np.append(btm_m4_res_avg,np.mean(btm_res_temp))
                        
                                btm_m4_cue = np.append(btm_m4_cue,btm_cue_temp)
                                btm_m4_res = np.append(btm_m4_res,btm_res_temp)

                for i in range(len(m5)):
                        if m5[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[m5[i],8].astype(int):condensed[m5[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[m5[i],9].astype(int)-bfr_res_bins:condensed[m5[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_m5_cue_avg = np.append(top_m5_cue_avg,np.mean(top_cue_temp))
                                top_m5_res_avg = np.append(top_m5_res_avg,np.mean(top_res_temp))
                                                        
                                top_m5_cue = np.append(top_m5_cue,top_cue_temp)
                                top_m5_res = np.append(top_m5_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[m5[i],8].astype(int):condensed[m5[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[m5[i],9].astype(int)-bfr_res_bins:condensed[m5[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_m5_cue_avg = np.append(btm_m5_cue_avg,np.mean(btm_cue_temp))
                                btm_m5_res_avg = np.append(btm_m5_res_avg,np.mean(btm_res_temp))
                        
                                btm_m5_cue = np.append(btm_m5_cue,btm_cue_temp)
                                btm_m5_res = np.append(btm_m5_res,btm_res_temp)

                for i in range(len(m6)):
                        if m6[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[m6[i],8].astype(int):condensed[m6[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[m6[i],9].astype(int)-bfr_res_bins:condensed[m6[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_m6_cue_avg = np.append(top_m6_cue_avg,np.mean(top_cue_temp))
                                top_m6_res_avg = np.append(top_m6_res_avg,np.mean(top_res_temp))
                                                        
                                top_m6_cue = np.append(top_m6_cue,top_cue_temp)
                                top_m6_res = np.append(top_m6_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[m6[i],8].astype(int):condensed[m6[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[m6[i],9].astype(int)-bfr_res_bins:condensed[m6[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_m6_cue_avg = np.append(btm_m6_cue_avg,np.mean(btm_cue_temp))
                                btm_m6_res_avg = np.append(btm_m6_res_avg,np.mean(btm_res_temp))
                        
                                btm_m6_cue = np.append(btm_m6_cue,btm_cue_temp)
                                btm_m6_res = np.append(btm_m6_res,btm_res_temp)

        top_m_cue_list = [top_m0_cue, top_m1_cue, top_m2_cue, top_m3_cue, top_m4_cue, top_m5_cue, top_m6_cue]
        top_m_res_list = [top_m0_res, top_m1_res, top_m2_res, top_m3_res, top_m4_res, top_m5_res, top_m6_res]
        btm_m_cue_list = [btm_m0_cue, btm_m1_cue, btm_m2_cue, btm_m3_cue, btm_m4_cue, btm_m5_cue, btm_m6_cue]
        btm_m_res_list = [btm_m0_res, btm_m1_res, btm_m2_res, btm_m3_res, btm_m4_res, btm_m5_res, btm_m6_res]
        
        m_avgs = [top_m0_cue_avg,top_m0_res_avg,top_m1_cue_avg,top_m1_res_avg,top_m2_cue_avg,top_m2_res_avg,top_m3_cue_avg,top_m3_res_avg,top_m4_cue_avg,top_m4_res_avg,top_m5_cue_avg,top_m5_res_avg,top_m6_cue_avg,top_m6_res_avg,btm_m0_cue_avg,btm_m0_res_avg,btm_m1_cue_avg,btm_m1_res_avg,btm_m2_cue_avg,btm_m2_res_avg,btm_m3_cue_avg,btm_m3_res_avg,btm_m4_cue_avg,btm_m4_res_avg,btm_m5_cue_avg,btm_m5_res_avg,btm_m6_cue_avg,btm_m6_res_avg]
				
        sio.savemat('m_avgs_%s' %(region_key),{'m_avgs':m_avgs},format='5')

        #plot res
		#TODO make sure :,5
        res0 = np.where(condensed[:,5] == 0)[0].astype(int)
        res1 = np.where(condensed[:,5] == 1)[0].astype(int)

        #unit_num = 0
        top_res0_cue = []
        top_res0_res = []
        top_res1_cue = []
        top_res1_res = []

        btm_res0_cue = []
        btm_res0_res = []
        btm_res1_cue = []
        btm_res1_res = []

        top_res0_cue_avg = []
        top_res0_res_avg = []
        top_res1_cue_avg = []
        top_res1_res_avg = []

        btm_res0_cue_avg = []
        btm_res0_res_avg = []
        btm_res1_cue_avg = []
        btm_res1_res_avg = []

        for j in range(num_plot):
                top_unit_ind = top_inds[j,3].astype(int)
                btm_unit_ind = btm_inds[j,3].astype(int)

                for i in range(len(res0)):
                        if res0[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[res0[i],8].astype(int):condensed[res0[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[res0[i],9].astype(int)-bfr_res_bins:condensed[res0[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000
                                
                                top_res0_cue_avg = np.append(top_res0_cue_avg,np.mean(top_cue_temp))
                                top_res0_res_avg = np.append(top_res0_res_avg,np.mean(top_res_temp))
                                                        
                                top_res0_cue = np.append(top_res0_cue,top_cue_temp)
                                top_res0_res = np.append(top_res0_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[res0[i],8].astype(int):condensed[res0[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[res0[i],9].astype(int)-bfr_res_bins:condensed[res0[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000
                                
                                btm_res0_cue_avg = np.append(btm_res0_cue_avg,np.mean(btm_cue_temp))
                                btm_res0_res_avg = np.append(btm_res0_res_avg,np.mean(btm_res_temp))
                        
                                btm_res0_cue = np.append(btm_res0_cue,btm_cue_temp)
                                btm_res0_res = np.append(btm_res0_res,btm_res_temp)

                for i in range(len(res1)):
                        if res1[i] != np.shape(condensed)[0]-1:
                                top_cue_temp = corr_input[condensed[res1[i],8].astype(int):condensed[res1[i],8].astype(int) + aft_cue_bins,0,top_unit_ind]
                                top_cue_temp = top_cue_temp /float(bin_size) * 1000 
                                top_res_temp = corr_input[condensed[res1[i],9].astype(int)-bfr_res_bins:condensed[res1[i],9].astype(int)+aft_res_bins,0,top_unit_ind]
                                top_res_temp = top_res_temp / float(bin_size) * 1000

                                top_res1_cue_avg = np.append(top_res1_cue_avg,np.mean(top_cue_temp))
                                top_res1_res_avg = np.append(top_res1_res_avg,np.mean(top_res_temp))
                        
                                top_res1_cue = np.append(top_res1_cue,top_cue_temp)
                                top_res1_res = np.append(top_res1_res,top_res_temp)

                                btm_cue_temp = corr_input[condensed[res1[i],8].astype(int):condensed[res1[i],8].astype(int) + aft_cue_bins,0,btm_unit_ind]
                                btm_cue_temp = btm_cue_temp /float(bin_size) * 1000 
                                btm_res_temp = corr_input[condensed[res1[i],9].astype(int)-bfr_res_bins:condensed[res1[i],9].astype(int)+aft_res_bins,0,btm_unit_ind]
                                btm_res_temp = btm_res_temp / float(bin_size) * 1000

                                btm_res1_cue_avg = np.append(btm_res1_cue_avg,np.mean(btm_cue_temp))
                                btm_res1_res_avg = np.append(btm_res1_res_avg,np.mean(btm_res_temp))
                        
                                btm_res1_cue = np.append(btm_res1_cue,btm_cue_temp)
                                btm_res1_res = np.append(btm_res1_res,btm_res_temp)


        top_res_cue_list = [top_res0_cue, top_res1_cue]
        top_res_res_list = [top_res0_res, top_res1_res]
        btm_res_cue_list = [btm_res0_cue, btm_res1_cue]
        btm_res_res_list = [btm_res0_res, btm_res1_res]

        
        res_avgs = [top_res0_cue_avg,top_res0_res_avg,top_res1_cue_avg,top_res1_res_avg,btm_res0_cue_avg,btm_res0_res_avg,btm_res1_cue_avg,btm_res1_res_avg]
        
        sio.savemat('res_avgs_%s' %(region_key),{'res_avgs':res_avgs},format='5')

        #all
        cue_avg = []
        res_avg = []
        cue_peaks = []
        res_peaks = []

        cue_avg = np.zeros((np.shape(corr_input)[2],np.shape(condensed)[0]))
        res_avg = np.zeros((np.shape(corr_input)[2],np.shape(condensed)[0]))
        cue_peaks = np.zeros((np.shape(corr_input)[2],np.shape(condensed)[0]))
        res_peaks = np.zeros((np.shape(corr_input)[2],np.shape(condensed)[0]))
        
        for unit_num in range(np.shape(corr_input)[2]):
            for i in range(np.shape(condensed)[0]):
                cue_temp = corr_input[condensed[i,8].astype(int):condensed[i,8].astype(int)+aft_cue_bins,0,unit_num]
                cue_temp = cue_temp / float(bin_size) * 1000

                res_temp = corr_input[condensed[i,9].astype(int):condensed[i,9].astype(int)+aft_cue_bins,0,unit_num]
                res_temp = res_temp / float(bin_size) * 1000
                
                cue_avg[unit_num,i] = np.mean(cue_temp)
                res_avg[unit_num,i] = np.mean(res_temp)

                try:
                        cue_peaks[unit_num,i] = np.max(cue_temp)
                        res_peaks[unit_num,i] = np.max(res_temp)
                except:
                        #pdb.set_trace()
                        pass


        fr_all_dict = {'cue_avg':cue_avg,'res_avg':res_avg,'cue_peaks':cue_peaks,'res_peaks':res_peaks}

        sio.savemat('fr_all_dict_%s' %(region_key),{'fr_all_dict':fr_all_dict},format='5')
		
        if any(len(element) == 0 for element in r_avgs) or any(len(element) == 0 for element in p_avgs) or any(len(element) == 0 for element in m_avgs) or any(len(element) == 0 for element in v_avgs) or any(len(element) == 0 for element in res_avgs):
            print "ERROR missing some elements"
        
        return_dict = {'top_inds':top_inds,'btm_inds':btm_inds,'top_r_cue_list':top_r_cue_list,'top_r_res_list':top_r_res_list,'btm_r_cue_list':btm_r_cue_list,'btm_r_res_list':btm_r_res_list,'r_avgs':r_avgs,'p_avgs':p_avgs,'v_avgs':v_avgs,'m_avgs':m_avgs,'res_avgs':res_avgs,'fr_all_dict':fr_all_dict,'order_dict':order_dict}

        return(return_dict)



###############################################
#start ########################################
###############################################

ts_filename = glob.glob('Extracted*_timestamps.mat')[0]
extracted_filename = ts_filename[:-15] + '.mat'

a = sio.loadmat(extracted_filename)
timestamps = sio.loadmat(ts_filename)

print extracted_filename

#create matrix of trial-by-trial info
trial_breakdown = timestamps['trial_breakdown']
condensed = np.zeros((np.shape(trial_breakdown)[0],10))

#0: disp_rp, 1: succ scene 2: failure scene, 3: rnum, 4: pnum, 5:succ/fail, 6: value, 7: motiv, 8: disp_rp bin

condensed[:,0] = trial_breakdown[:,1]
condensed[:,1] = trial_breakdown[:,2]
condensed[:,2] = trial_breakdown[:,3]
condensed[:,3] = trial_breakdown[:,5]
condensed[:,4] = trial_breakdown[:,7]
condensed[:,5] = trial_breakdown[:,10]
#condensed[:,8] = trial_breakdown[   ] Next reset, don't need for now          

bin_size_sec = bin_size / float(1000)
for i in range(np.shape(condensed)[0]):
        condensed[i,8] = int(np.around((round(condensed[i,0] / bin_size_sec) * bin_size_sec),decimals=2)/bin_size_sec)
        #condensed[i,8] = condensed[i,8].astype(int)
        condensed[i,9] = int(np.around((round((condensed[i,1] + condensed[i,2]) / bin_size_sec) * bin_size_sec),decimals=2)/bin_size_sec)
        #condensed[i,9] = condensed[i,9].astype(int)

#delete end trials if not fully finished
if condensed[-1,1] == condensed[-1,2] == 0:
	new_condensed = condensed[0:-1,:]
        condensed = new_condensed
condensed = condensed[condensed[:,0] != 0]

#remove trials with now succ or failure scene (not sure why, but saw in one)
condensed = condensed[np.invert(np.logical_and(condensed[:,1] == 0, condensed[:,2] == 0))]

#TODOD FOR NOW remove catch trials
condensed = condensed[condensed[:,5] == 0]
#col 5 all 0s now, replace with succ/fail vector: succ = 1, fail = 0
condensed[condensed[:,1] != 0, 5] = 1
condensed[condensed[:,2] != 0, 5] = 0

condensed[:,6] = condensed[:,3] - condensed[:,4]
condensed[:,7] = condensed[:,3] + condensed[:,4]


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

spike_dict = {'M1':M1_spikes,'S1':S1_spikes,'PmD':PmD_spikes}

data_dict = {'M1':{},'S1':{},'PmD':{}}
for key,value in data_dict.iteritems():
        print 'running %s' %(key)
        spike_data = []
        for i in range(len(spike_dict[key])):
                spike_data.append(spike_dict[key][i]['ts'][0,0][0])

        print 'binning and smoothing'
        binned_data = hist_and_smooth_data(spike_data)
       
        print 'correlation analysis'
        if gaussian_bool or zscore_bool:
                corr_data = run_corr(binned_data['smoothed'],condensed)
        else:
                corr_data = run_corr(binned_data['hist_data'],condensed)

        if plot_bool:
                print 'plotting'
                plot_return = plot_corr(corr_data['corr_input_matrix'],corr_data['corr_output'],condensed,key)
                data_dict[key]['plot_retun'] = plot_return

        data_dict[key]['spike_data'] = spike_data
        data_dict[key]['binned_data'] = binned_data
        data_dict[key]['corr_data'] = corr_data
		
        sio.savemat('corr_output_%s' %(key),{'corr_output':corr_data['corr_output']},format='5')

		
np.save('corr_analysis.npy',data_dict)
