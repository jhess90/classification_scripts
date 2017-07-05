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
import sys
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import glob
import scipy as sp
import statsmodels.api as sm
import pandas
import xlsxwriter
import pyexcel
import pyexcel_xlsx
import os
import subprocess

###################
# params to set ###
###################

plot_bool = True
plot_both_bool = True
stats_bool = True
save_bool = True
catch_bool = True

###################

filename = glob.glob('avg_fr_and_nlized_data*.npy')
filename = filename[0]

#with open("output_%s" %(filename), "w+") as output:
#        subprocess.call(["python", "/home/jack/workspace/classification_scripts/analyze_nl_data_mult.py"], stdout=output);

print 'loading %s' %(filename)

save_dict = np.load(filename)[()]

keys = save_dict.keys()

S1_dict_total = {}
M1_dict_total = {}
PmD_dict_total = {}
PmV_dict_total = {}

params = save_dict['param_dict']

bin_size = params['bin_size']
time_before = params['time_before']
time_after = params['time_after']
num_bins_before = np.int(time_before * 1000 / bin_size * -1) #because neg value
		
print 'arranging data'


def do_stats(event_key,event_data):
	#print 'stats %s' %(event_key)
	#stats_all = []
	stats_all = {}
	mann_w = []
	mann_w_sig_bool = []
	#for i in range(len(event_key)):
	for i in range(len(event_data)):
		stats = {}
                if np.shape(event_data[0]['firing_rate']) != (): 
                        before_bins = event_data[i]['avg_nl'][0:num_bins_before]
                        after_bins = event_data[i]['avg_nl'][num_bins_before:2*num_bins_before]
                else:
                        stats_all = {'mann_w':0,'mann_w_sig_bool':0,'perc_sig':0}
                        print 'skipping %s because doent exist' %(event_key)
                        return(stats_all)
		try:
                        #TODO should this be wilcoxon signed-rank test? B/ not indp samples
			mann_w_temp = sp.stats.mannwhitneyu(before_bins,after_bins,'two-sided')
			mann_w_sig_bool_temp = mann_w_temp[1] <= 0.05

                        ####TODO fix this for real, shitty jurry rigged version for now
                        mann_w_temp = sp.stats.wilcoxon(before_bins,after_bins,zero_method='wilcox')

		except:
			mann_w_temp = [99,99]   #TODO some other error?
			mann_w_sig_bool_temp = False

		mann_w.append(mann_w_temp)
		mann_w_sig_bool.append(mann_w_sig_bool_temp)

	stats_all['mann_w'] = mann_w
	stats_all['mann_w_sig_bool'] = mann_w_sig_bool

        try:
                perc_sig = sum(mann_w_sig_bool) / float(len(mann_w_sig_bool))
        except:
                perc_sig = 0.0
	stats_all['perc_sig'] = perc_sig
	perc_sig_str = np.around(perc_sig *100,decimals = 2)
	print '%s%% units in %s significantly different' %(perc_sig_str,event_key)

	return stats_all


def do_comp_stats(all_dict,region_key,type_key):
	nlized_comparison_data = {}
	index = 0
	d3_nl_array = []
	d3_index = {}
	for key,value in all_dict.iteritems():
		if (not 'stats' in key):      #and (not 'catch' in key):
			dummy = np.asarray(all_dict[key])
			nl_array=[]
			for i in range(len(dummy)):
				nl_array.append(dummy[i]['avg_nl'])
			nl_array = np.asarray(nl_array)
                        nlized_comparison_data[key] = nl_array

			d3_nl_array.append(nl_array)                        
                        d3_index[index] = key
			index+=1
        #pdb.set_trace()
	d3_nl_array = np.asarray(d3_nl_array)

        if len(d3_nl_array.shape) == 1:
                #pdb.set_trace()
                new_d3 = np.zeros((index,len(d3_nl_array),num_bins_before+2*num_bins_before))
                for i in range(d3_nl_array.shape[0]):
                        if not d3_nl_array[i].shape[0] == 0:
                                new_d3[i,:,:] = d3_nl_array[i][0]
                        else:
                                print 'error with %s, %s, %s' %(region_key,type_key,i)
                d3_nl_array = new_d3

        try:
                all_before = d3_nl_array[:,:,0:num_bins_before]
                all_after = d3_nl_array[:,:,num_bins_before:2*num_bins_before]
        except:
                #pdb.set_trace()
                pass

	anovas = []

        if d3_nl_array.shape[0] == 16:
                for i in range(d3_nl_array.shape[1]):
                        anovas.append(sp.stats.f_oneway(all_after[0][i],all_after[1][i],all_after[2][i],all_after[3][i],all_after[4][i],all_after[5][i],all_after[6][i],all_after[7][i],all_after[8][i],all_after[9][i],all_after[10][i],all_after[11][i],all_after[12][i],all_after[13][i],all_after[14][i],all_after[15][i]))
                anovas = np.asarray(anovas)
                
	elif d3_nl_array.shape[0] == 10:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(all_after[0][i],all_after[1][i],all_after[2][i],all_after[3][i],all_after[4][i],all_after[5][i],all_after[6][i],all_after[7][i],all_after[8][i],all_after[9][i]))
		anovas = np.asarray(anovas)
		
	elif d3_nl_array.shape[0] == 9:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(all_after[0][i],all_after[1][i],all_after[2][i],all_after[3][i],all_after[4][i],all_after[5][i],all_after[6][i],all_after[7][i],all_after[8][i]))
		anovas = np.asarray(anovas)
		
	elif d3_nl_array.shape[0] == 8:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(all_after[0][i],all_after[1][i],all_after[2][i],all_after[3][i],all_after[4][i],all_after[5][i],all_after[6][i],all_after[7][i]))
		anovas = np.asarray(anovas)

	elif d3_nl_array.shape[0] == 7:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(all_after[0][i],all_after[1][i],all_after[2][i],all_after[3][i],all_after[4][i],all_after[5][i],all_after[6][i]))
		anovas = np.asarray(anovas)
	elif d3_nl_array.shape[0] == 6:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(all_after[0][i],all_after[1][i],all_after[2][i],all_after[3][i],all_after[4][i],all_after[5][i]))
		anovas = np.asarray(anovas)
	elif d3_nl_array.shape[0] == 5:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(all_after[0][i],all_after[1][i],all_after[2][i],all_after[3][i],all_after[4][i]))
		anovas = np.asarray(anovas)
	elif d3_nl_array.shape[0] == 4:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(all_after[0][i],all_after[1][i],all_after[2][i],all_after[3][i]))
		anovas = np.asarray(anovas)
	elif d3_nl_array.shape[0] == 3:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(all_after[0][i],all_after[1][i],all_after[2][i]))
		anovas = np.asarray(anovas)
	elif d3_nl_array.shape[0] == 2:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(all_after[0][i],all_after[1][i]))
		anovas = np.asarray(anovas)
	else:
		print 'ANOVA error'
                anovas = np.zeros((1,2))
        
        #pdb.set_trace()
        if d3_nl_array.shape[1] == 0:

                return_dict={'d3_nl_array':d3_nl_array,'d3_index':d3_index,'nlized_comparison_data':nlized_comparison_data,'anovas':0,'anova_sig_pval':0,'anova_sig_pval_bool':0,'anova_sig_perc':0,'for_post_hoc':0}
								
                return return_dict

                #return
	anova_pval = anovas[:,1]
	anova_sig_pval = np.where(anova_pval <= 0.05)
	anova_sig_pval_bool = anova_pval <= 0.05

	anova_sig_perc = sum(anova_sig_pval_bool) / float(len(anova_sig_pval_bool))
	anova_sig_perc_str= np.around(anova_sig_perc *100,decimals = 2)
	print '%s%% units significantly different between events' %(anova_sig_perc_str)

	for_post_hoc = []
	for i in range(len(anova_sig_pval_bool)):
		if anova_sig_pval_bool[i] and not catch_bool:
			list_all = np.zeros(num_bins_before * d3_nl_array.shape[0])
			cat_all = np.zeros(num_bins_before * d3_nl_array.shape[0])
			vals = {}
			for j in range(d3_nl_array.shape[0]):
				list_all[j*num_bins_before:(j+1)*num_bins_before] = all_after[j][i][0:num_bins_before]
				cat_all[j*num_bins_before:(j+1)*num_bins_before] = np.ones(num_bins_before)*j

			list_all = np.asarray(list_all)
			cat_all = np.asarray(cat_all)

			mc = sm.stats.multicomp.MultiComparison(list_all,cat_all)
			tukey = mc.tukeyhsd()

			#print 'unit: %s' %(i)
			#print tukey.summary()

			pyexcel.save_as(array = tukey.summary(),dest_file_name='%s_%s_unit_%s.csv'%(type_key,region_key,i))
			
			vals['unit_no'] = i
			vals['list_all'] = list_all
			vals['cat_all'] = cat_all
			vals['mc'] = mc
			vals['tukey'] = tukey
			
			for_post_hoc.append(vals)

        #if catch_bool:
                #for_post_hoc = {'unit_no':[],'list_all':[],'cat_all':[],'mc':[],'tukey':[]}

	return_dict={'d3_nl_array':d3_nl_array,'d3_index':d3_index,'nlized_comparison_data':nlized_comparison_data,'anovas':anovas,'anova_sig_pval':anova_sig_pval,'anova_sig_pval_bool':anova_sig_pval_bool,'anova_sig_perc':anova_sig_perc,'for_post_hoc':for_post_hoc}
								
	return return_dict


#adding significance to plot
err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}

def label_sig(i,j,text,X,Y):
        #x and y = actual xy values on plot
        x = (X[i] + X[j]) / 2
        y = 1.3*max(Y[i],Y[j])

        props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':20,'shrinkB':20,'lw':2}
        #ax.annotate instead?
        plt.annotate(text, xy = (X[i],y+.2), zorder = 10)
        plt.annotate('',xy = (X[i],y), xytext = (X[j],y), arrowprops = props)


def plot_comparison(region_key,region_val):
        return_dicts = []
        print "plotting %s" %(region_key)
        suffix = '_%ss' %(region_key[0:-6])

        try:
                for i in range(3):
                        for j in range(len(all_regions_dict[region_key]['r0_succ_cue%s'%(suffix)])):
                                #print 'unit %s' %(j)
                                temp = np.zeros((8))
                                temp[0] = np.max(all_regions_dict[region_key]['r%s_succ_cue%s'%(i,suffix)][j]['avg_nl'])
                                temp[1] = np.max(all_regions_dict[region_key]['r%s_fail_cue%s'%(i,suffix)][j]['avg_nl'])
                                temp[2] = np.max(all_regions_dict[region_key]['p%s_succ_cue%s'%(i,suffix)][j]['avg_nl'])
                                temp[3] = np.max(all_regions_dict[region_key]['p%s_fail_cue%s'%(i,suffix)][j]['avg_nl'])
                                temp[4] = np.max(all_regions_dict[region_key]['r%s_succ_result%s'%(i,suffix)][j]['avg_nl'])
                                temp[5] = np.max(all_regions_dict[region_key]['r%s_fail_result%s'%(i,suffix)][j]['avg_nl'])
                                temp[6] = np.max(all_regions_dict[region_key]['p%s_succ_result%s'%(i,suffix)][j]['avg_nl'])
                                temp[7] = np.max(all_regions_dict[region_key]['p%s_fail_result%s'%(i,suffix)][j]['avg_nl'])

                                if np.max(temp) > 1.0:
                                        print 'nlization error: %s, unit %s, level %s, %s' %(region_key,j,i,np.argmax(temp))
        except:
                print 'error finding nl error'

        for i in range(len(all_regions_dict[region_key]['r0_succ_cue%s' %(suffix)])):
        
                try:
                        r0_succ_cue_avg = sum(all_regions_dict[region_key]['r0_succ_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        r1_succ_cue_avg = sum(all_regions_dict[region_key]['r1_succ_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        r2_succ_cue_avg = sum(all_regions_dict[region_key]['r2_succ_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        r3_succ_cue_avg = sum(all_regions_dict[region_key]['r3_succ_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p0_fail_cue_avg = sum(all_regions_dict[region_key]['p0_fail_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p1_fail_cue_avg = sum(all_regions_dict[region_key]['p1_fail_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p2_fail_cue_avg = sum(all_regions_dict[region_key]['p2_fail_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p3_fail_cue_avg = sum(all_regions_dict[region_key]['p3_fail_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before

                        r0_succ_result_avg = sum(all_regions_dict[region_key]['r0_succ_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        r1_succ_result_avg = sum(all_regions_dict[region_key]['r1_succ_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        r2_succ_result_avg = sum(all_regions_dict[region_key]['r2_succ_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        r3_succ_result_avg = sum(all_regions_dict[region_key]['r3_succ_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p0_fail_result_avg = sum(all_regions_dict[region_key]['p0_fail_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p1_fail_result_avg = sum(all_regions_dict[region_key]['p1_fail_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p2_fail_result_avg = sum(all_regions_dict[region_key]['p2_fail_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p3_fail_result_avg = sum(all_regions_dict[region_key]['p3_fail_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before

                        r0_fail_cue_avg = sum(all_regions_dict[region_key]['r0_fail_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        r1_fail_cue_avg = sum(all_regions_dict[region_key]['r1_fail_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        r2_fail_cue_avg = sum(all_regions_dict[region_key]['r2_fail_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        r3_fail_cue_avg = sum(all_regions_dict[region_key]['r3_fail_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p0_succ_cue_avg = sum(all_regions_dict[region_key]['p0_succ_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p1_succ_cue_avg = sum(all_regions_dict[region_key]['p1_succ_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p2_succ_cue_avg = sum(all_regions_dict[region_key]['p2_succ_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p3_succ_cue_avg = sum(all_regions_dict[region_key]['p3_succ_cue%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before

                        r0_fail_result_avg = sum(all_regions_dict[region_key]['r0_fail_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        r1_fail_result_avg = sum(all_regions_dict[region_key]['r1_fail_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        r2_fail_result_avg = sum(all_regions_dict[region_key]['r2_fail_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        r3_fail_result_avg = sum(all_regions_dict[region_key]['r3_fail_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p0_succ_result_avg = sum(all_regions_dict[region_key]['p0_succ_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p1_succ_result_avg = sum(all_regions_dict[region_key]['p1_succ_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p2_succ_result_avg = sum(all_regions_dict[region_key]['p2_succ_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before
                        p3_succ_result_avg = sum(all_regions_dict[region_key]['p3_succ_result%s' %(suffix)][i]['avg_nl'][num_bins_before:2*num_bins_before]) / num_bins_before

                        #std dev
                        r0_succ_cue_std_devs = all_regions_dict[region_key]['r0_succ_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        r1_succ_cue_std_devs = all_regions_dict[region_key]['r1_succ_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        r2_succ_cue_std_devs = all_regions_dict[region_key]['r2_succ_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        r3_succ_cue_std_devs = all_regions_dict[region_key]['r3_succ_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p0_fail_cue_std_devs = all_regions_dict[region_key]['p0_fail_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p1_fail_cue_std_devs = all_regions_dict[region_key]['p1_fail_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p2_fail_cue_std_devs = all_regions_dict[region_key]['p2_fail_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p3_fail_cue_std_devs = all_regions_dict[region_key]['p3_fail_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]

                        r0_succ_result_std_devs = all_regions_dict[region_key]['r0_succ_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        r1_succ_result_std_devs = all_regions_dict[region_key]['r1_succ_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        r2_succ_result_std_devs = all_regions_dict[region_key]['r2_succ_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        r3_succ_result_std_devs = all_regions_dict[region_key]['r3_succ_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p0_fail_result_std_devs = all_regions_dict[region_key]['p0_fail_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p1_fail_result_std_devs = all_regions_dict[region_key]['p1_fail_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p2_fail_result_std_devs = all_regions_dict[region_key]['p2_fail_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p3_fail_result_std_devs = all_regions_dict[region_key]['p3_fail_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]

                        r0_fail_cue_std_devs = all_regions_dict[region_key]['r0_fail_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        r1_fail_cue_std_devs = all_regions_dict[region_key]['r1_fail_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        r2_fail_cue_std_devs = all_regions_dict[region_key]['r2_fail_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        r3_fail_cue_std_devs = all_regions_dict[region_key]['r3_fail_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p0_succ_cue_std_devs = all_regions_dict[region_key]['p0_succ_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p1_succ_cue_std_devs = all_regions_dict[region_key]['p1_succ_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p2_succ_cue_std_devs = all_regions_dict[region_key]['p2_succ_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p3_succ_cue_std_devs = all_regions_dict[region_key]['p3_succ_cue%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]

                        r0_fail_result_std_devs = all_regions_dict[region_key]['r0_fail_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        r1_fail_result_std_devs = all_regions_dict[region_key]['r1_fail_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        r2_fail_result_std_devs = all_regions_dict[region_key]['r2_fail_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        r3_fail_result_std_devs = all_regions_dict[region_key]['r3_fail_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p0_succ_result_std_devs = all_regions_dict[region_key]['p0_succ_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p1_succ_result_std_devs = all_regions_dict[region_key]['p1_succ_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p2_succ_result_std_devs = all_regions_dict[region_key]['p2_succ_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]
                        p3_succ_result_std_devs = all_regions_dict[region_key]['p3_succ_result%s' %(suffix)][i]['std_dev_nl'][num_bins_before:2*num_bins_before]

                except:
                        print 'exception %s' %(i)
                        r0_succ_cue_avg=r0_fail_cue_avg=r1_succ_cue_avg=r1_fail_cue_avg=r2_succ_cue_avg=r2_fail_cue_avg=r3_succ_cue_avg=r3_fail_cue_avg=r0_succ_cue_avg=r0_fail_cue_avg=r1_succ_cue_avg=r1_fail_cue_avg=r2_succ_cue_avg=r2_fail_cue_avg=r3_succ_cue_avg=r3_fail_cue_avg = 0
                        r0_succ_result_avg=r0_fail_result_avg=r1_succ_result_avg=r1_fail_result_avg=r2_succ_result_avg=r2_fail_result_avg=r3_succ_result_avg=r3_fail_result_avg=r0_succ_result_avg=r0_fail_result_avg=r1_succ_result_avg=r1_fail_result_avg=r2_succ_result_avg=r2_fail_result_avg=r3_succ_result_avg=r3_fail_result_avg = 0
                        r0_succ_cue_std_devs=r0_fail_cue_std_devs=r1_succ_std_devs=r1_fail_cue_std_devs=r2_succ_cue_std_devs=r2_fail_cue_std_devs=r3_succ_cue_std_devs=r3_fail_cue_std_devs=r0_succ_cue_std_devs=r0_fail_cue_std_devs=r1_succ_cue_std_devs=r1_fail_cue_std_devs=r2_succ_cue_std_devs=r2_fail_cue_std_devs=r3_succ_cue_std_devs=r3_fail_cue_std_devs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        r0_succ_result_std_devs=r0_fail_result_std_devs=r1_succ_result_std_devs=r1_fail_result_std_devs=r2_succ_result_std_devs=r2_fail_result_std_devs=r3_succ_result_std_devs=r3_fail_result_std_devs=r0_succ_result_std_devs=r0_fail_result_std_devs=r1_succ_result_std_devs=r1_fail_result_std_devs=r2_succ_result_std_devs=r2_fail_result_std_devs=r3_succ_result_std_devs=r3_fail_result_std_devs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                        p0_succ_cue_avg=p0_fail_cue_avg=p1_succ_cue_avg=p1_fail_cue_avg=p2_succ_cue_avg=p2_fail_cue_avg=p3_succ_cue_avg=p3_fail_cue_avg=p0_succ_cue_avg=p0_fail_cue_avg=p1_succ_cue_avg=p1_fail_cue_avg=p2_succ_cue_avg=p2_fail_cue_avg=p3_succ_cue_avg=p3_fail_cue_avg = 0
                        p0_succ_result_avg=p0_fail_result_avg=p1_succ_result_avg=p1_fail_result_avg=p2_succ_result_avg=p2_fail_result_avg=p3_succ_result_avg=p3_fail_result_avg=p0_succ_result_avg=p0_fail_result_avg=p1_succ_result_avg=p1_fail_result_avg=p2_succ_result_avg=p2_fail_result_avg=p3_succ_result_avg=p3_fail_result_avg = 0
                        p0_succ_cue_std_devs=p0_fail_cue_std_devs=p1_succ_std_devs=p1_fail_cue_std_devs=p2_succ_cue_std_devs=p2_fail_cue_std_devs=p3_succ_cue_std_devs=p3_fail_cue_std_devs=p0_succ_cue_std_devs=p0_fail_cue_std_devs=p1_succ_cue_std_devs=p1_fail_cue_std_devs=p2_succ_cue_std_devs=p2_fail_cue_std_devs=p3_succ_cue_std_devs=p3_fail_cue_std_devs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        p0_succ_result_std_devs=p0_fail_result_std_devs=p1_succ_result_std_devs=p1_fail_result_std_devs=p2_succ_result_std_devs=p2_fail_result_std_devs=p3_succ_result_std_devs=rp_fail_result_std_devs=p0_succ_result_std_devs=p0_fail_result_std_devs=p1_succ_result_std_devs=p1_fail_result_std_devs=p2_succ_result_std_devs=p2_fail_result_std_devs=p3_succ_result_std_devs=p3_fail_result_std_devs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        #pdb.set_trace()
                                
                #pooled std devs
                r0_succ_cue_pooled_std = np.sqrt(sum(np.square(r0_succ_cue_std_devs))/len(r0_succ_cue_std_devs))
                r1_succ_cue_pooled_std = np.sqrt(sum(np.square(r1_succ_cue_std_devs))/len(r1_succ_cue_std_devs))
                r2_succ_cue_pooled_std = np.sqrt(sum(np.square(r2_succ_cue_std_devs))/len(r2_succ_cue_std_devs))
                r3_succ_cue_pooled_std = np.sqrt(sum(np.square(r3_succ_cue_std_devs))/len(r3_succ_cue_std_devs))
                p0_fail_cue_pooled_std = np.sqrt(sum(np.square(p0_fail_cue_std_devs))/len(p0_fail_cue_std_devs))
                p1_fail_cue_pooled_std = np.sqrt(sum(np.square(p1_fail_cue_std_devs))/len(p1_fail_cue_std_devs))
                p2_fail_cue_pooled_std = np.sqrt(sum(np.square(p2_fail_cue_std_devs))/len(p2_fail_cue_std_devs))
                p3_fail_cue_pooled_std = np.sqrt(sum(np.square(p3_fail_cue_std_devs))/len(p3_fail_cue_std_devs))

                r0_succ_result_pooled_std = np.sqrt(sum(np.square(r0_succ_result_std_devs))/len(r0_succ_result_std_devs))
                r1_succ_result_pooled_std = np.sqrt(sum(np.square(r1_succ_result_std_devs))/len(r1_succ_result_std_devs))
                r2_succ_result_pooled_std = np.sqrt(sum(np.square(r2_succ_result_std_devs))/len(r2_succ_result_std_devs))
                r3_succ_result_pooled_std = np.sqrt(sum(np.square(r3_succ_result_std_devs))/len(r3_succ_result_std_devs))
                p0_fail_result_pooled_std = np.sqrt(sum(np.square(p0_fail_result_std_devs))/len(p0_fail_result_std_devs))
                p1_fail_result_pooled_std = np.sqrt(sum(np.square(p1_fail_result_std_devs))/len(p1_fail_result_std_devs))
                p2_fail_result_pooled_std = np.sqrt(sum(np.square(p2_fail_result_std_devs))/len(p2_fail_result_std_devs))
                p3_fail_result_pooled_std = np.sqrt(sum(np.square(p3_fail_result_std_devs))/len(p3_fail_result_std_devs))

                r0_fail_cue_pooled_std = np.sqrt(sum(np.square(r0_fail_cue_std_devs))/len(r0_fail_cue_std_devs))
                r1_fail_cue_pooled_std = np.sqrt(sum(np.square(r1_fail_cue_std_devs))/len(r1_fail_cue_std_devs))
                r2_fail_cue_pooled_std = np.sqrt(sum(np.square(r2_fail_cue_std_devs))/len(r2_fail_cue_std_devs))
                r3_fail_cue_pooled_std = np.sqrt(sum(np.square(r3_fail_cue_std_devs))/len(r3_fail_cue_std_devs))
                p0_succ_cue_pooled_std = np.sqrt(sum(np.square(p0_succ_cue_std_devs))/len(p0_succ_cue_std_devs))
                p1_succ_cue_pooled_std = np.sqrt(sum(np.square(p1_succ_cue_std_devs))/len(p1_succ_cue_std_devs))
                p2_succ_cue_pooled_std = np.sqrt(sum(np.square(p2_succ_cue_std_devs))/len(p2_succ_cue_std_devs))
                p3_succ_cue_pooled_std = np.sqrt(sum(np.square(p3_succ_cue_std_devs))/len(p3_succ_cue_std_devs))

                r0_fail_result_pooled_std = np.sqrt(sum(np.square(r0_fail_result_std_devs))/len(r0_fail_result_std_devs))
                r1_fail_result_pooled_std = np.sqrt(sum(np.square(r1_fail_result_std_devs))/len(r1_fail_result_std_devs))
                r2_fail_result_pooled_std = np.sqrt(sum(np.square(r2_fail_result_std_devs))/len(r2_fail_result_std_devs))
                r3_fail_result_pooled_std = np.sqrt(sum(np.square(r3_fail_result_std_devs))/len(r3_fail_result_std_devs))
                p0_succ_result_pooled_std = np.sqrt(sum(np.square(p0_succ_result_std_devs))/len(p0_succ_result_std_devs))
                p1_succ_result_pooled_std = np.sqrt(sum(np.square(p1_succ_result_std_devs))/len(p1_succ_result_std_devs))
                p2_succ_result_pooled_std = np.sqrt(sum(np.square(p2_succ_result_std_devs))/len(p2_succ_result_std_devs))
                p3_succ_result_pooled_std = np.sqrt(sum(np.square(p3_succ_result_std_devs))/len(p3_succ_result_std_devs))

                ax = plt.gca()
                plt.subplot(2,1,1)

                rp_levels_p = [-3,-2,-1,0]
                rp_levels_r = [0,1,2,3]
                p_linspace = np.linspace(rp_levels_p[0],rp_levels_p[3])
                r_linspace = np.linspace(rp_levels_r[0],rp_levels_r[3])
                
                avg_cue_total_p_fail = [p3_fail_cue_avg,p2_fail_cue_avg,p1_fail_cue_avg,p0_fail_cue_avg]
                avg_cue_total_r_succ = [r0_succ_cue_avg,r1_succ_cue_avg,r2_succ_cue_avg,r3_succ_cue_avg]
                pooled_std_cue_total_p_fail = [p3_fail_cue_pooled_std,p2_fail_cue_pooled_std,p1_fail_cue_pooled_std,p0_fail_cue_pooled_std]
                pooled_std_cue_total_r_succ = [r0_succ_cue_pooled_std,r1_succ_cue_pooled_std,r2_succ_cue_pooled_std,r3_succ_cue_pooled_std]

                avg_cue_total_p_succ = [p3_succ_cue_avg,p2_succ_cue_avg,p1_succ_cue_avg,p0_succ_cue_avg]
                avg_cue_total_r_fail = [r0_fail_cue_avg,r1_fail_cue_avg,r2_fail_cue_avg,r3_fail_cue_avg]
                pooled_std_cue_total_p_succ = [p3_succ_cue_pooled_std,p2_succ_cue_pooled_std,p1_succ_cue_pooled_std,p0_succ_cue_pooled_std]
                pooled_std_cue_total_r_fail = [r0_fail_cue_pooled_std,r1_fail_cue_pooled_std,r2_fail_cue_pooled_std,r3_fail_cue_pooled_std]

                linestyle = {'marker':'o','color':'r','linestyle':'none'}
                plt.errorbar(rp_levels_p,avg_cue_total_p_fail,yerr=pooled_std_cue_total_p_fail,label='cued punishing,\nunsuccessful',**linestyle)
                linestyle = {'marker':'o','color':'g','linestyle':'none'}
                plt.errorbar(rp_levels_r,avg_cue_total_r_succ,yerr=pooled_std_cue_total_r_succ,label='cued rewarding,\nsuccessful',**linestyle)
 
                if plot_both_bool:
                        linestyle = {'marker':'o','color':'pink','linestyle':'none'} #'lw':0.5}
                        plt.errorbar(rp_levels_p,avg_cue_total_p_succ,yerr=pooled_std_cue_total_p_succ,label='cued punishing,\nsuccessful',**linestyle)
                        linestyle = {'marker':'o','color':'palegreen','linestyle':'none'} #'lw':0.5}
                        plt.errorbar(rp_levels_r,avg_cue_total_r_fail,yerr=pooled_std_cue_total_r_fail,label='cued rewarding,\nunsuccessful',**linestyle)
                        
                slope, intercept, r_value, p_value, std_err = sp.stats.linregress(rp_levels_p,avg_cue_total_p_fail)
                line = np.multiply(p_linspace,slope)+intercept
                plt.plot(p_linspace,line,color='r')
                p_fail_cue_linreg = {'slope':slope,'intercept':intercept,'r_value':r_value,'p_value':p_value,'std_err':std_err,'avg_total':avg_cue_total_p_fail,'pooled_std':pooled_std_cue_total_p_fail}
        
                slope, intercept, r_value, p_value, std_err = sp.stats.linregress(rp_levels_r,avg_cue_total_r_succ)
                line = np.multiply(r_linspace,slope)+intercept
                plt.plot(r_linspace,line,color='g')
                r_succ_cue_linreg = {'slope':slope,'intercept':intercept,'r_value':r_value,'p_value':p_value,'std_err':std_err,'avg_total':avg_cue_total_r_succ,'pooled_std':pooled_std_cue_total_r_succ}

                if plot_both_bool:
                        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(rp_levels_p,avg_cue_total_p_succ)
                        line = np.multiply(p_linspace,slope)+intercept
                        plt.plot(p_linspace,line,color='pink')
                        p_succ_cue_linreg = {'slope':slope,'intercept':intercept,'r_value':r_value,'p_value':p_value,'std_err':std_err,'avg_total':avg_cue_total_p_succ,'pooled_std':pooled_std_cue_total_p_succ}
        
                        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(rp_levels_r,avg_cue_total_r_fail)
                        line = np.multiply(r_linspace,slope)+intercept
                        plt.plot(r_linspace,line,color='palegreen')
                        r_fail_cue_linreg = {'slope':slope,'intercept':intercept,'r_value':r_value,'p_value':p_value,'std_err':std_err,'avg_total':avg_cue_total_r_fail,'pooled_std':pooled_std_cue_total_r_fail}

                plt.legend(loc='upper center', bbox_to_anchor=(1.30, 0.8),ncol=1)
                #plt.ylim(ymin = 0, ymax = 1.0)
                plt.xlim(xmin = -3.5, xmax = 3.5)
                plt.title('cue')

                plt.subplot(2,1,2)
                avg_result_total_p_fail = [p3_fail_result_avg,p2_fail_result_avg,p1_fail_result_avg,p0_fail_result_avg]
                avg_result_total_r_succ = [r0_succ_result_avg,r1_succ_result_avg,r2_succ_result_avg,r3_succ_result_avg]
                pooled_std_result_total_p_fail = [p3_fail_result_pooled_std,p2_fail_result_pooled_std,p1_fail_result_pooled_std,p0_fail_result_pooled_std]
                pooled_std_result_total_r_succ = [r0_succ_result_pooled_std,r1_succ_result_pooled_std,r2_succ_result_pooled_std,r3_succ_result_pooled_std]

                avg_result_total_p_succ = [p3_succ_result_avg,p2_succ_result_avg,p1_succ_result_avg,p0_succ_result_avg]
                avg_result_total_r_fail = [r0_fail_result_avg,r1_fail_result_avg,r2_fail_result_avg,r3_fail_result_avg]
                pooled_std_result_total_p_succ = [p3_succ_result_pooled_std,p2_succ_result_pooled_std,p1_succ_result_pooled_std,p0_succ_result_pooled_std]
                pooled_std_result_total_r_fail = [r0_fail_result_pooled_std,r1_fail_result_pooled_std,r2_fail_result_pooled_std,r3_fail_result_pooled_std]

                linestyle = {'marker':'o','color':'r','linestyle':'none'}
                plt.errorbar(rp_levels_p,avg_result_total_p_fail,yerr=pooled_std_result_total_p_fail,**linestyle) #label='cued punishing, unsuccessful',**linestyle)
                linestyle = {'marker':'o','color':'g','linestyle':'none'}
                plt.errorbar(rp_levels_r,avg_result_total_r_succ,yerr=pooled_std_result_total_r_succ,**linestyle) #label ='cued rewarding, successful',**linestyle)

                if plot_both_bool:
                        linestyle = {'marker':'o','color':'pink','linestyle':'none'}     #'lw':0.5}
                        plt.errorbar(rp_levels_p,avg_result_total_p_succ,yerr=pooled_std_result_total_p_succ,**linestyle) #,label='cued punishing, successful',**linestyle)
                        linestyle = {'marker':'o','color':'palegreen','linestyle':'none'}   #'lw':0.5}
                        plt.errorbar(rp_levels_r,avg_result_total_r_fail,yerr=pooled_std_result_total_r_fail,**linestyle) #label='cued rewarding, unsuccessful',**linestyle)

                slope, intercept, r_value, p_value, std_err = sp.stats.linregress(rp_levels_p,avg_result_total_p_fail)
                line = np.multiply(p_linspace,slope)+intercept
                plt.plot(p_linspace,line,color='r')
                p_fail_result_linreg = {'slope':slope,'intercept':intercept,'r_value':r_value,'p_value':p_value,'std_err':std_err,'avg_total':avg_result_total_p_fail,'pooled_std':pooled_std_result_total_p_fail}
        
                slope, intercept, r_value, p_value, std_err = sp.stats.linregress(rp_levels_r,avg_result_total_r_succ)
                line = np.multiply(r_linspace,slope)+intercept
                plt.plot(r_linspace,line,color='g')
                r_succ_result_linreg = {'slope':slope,'intercept':intercept,'r_value':r_value,'p_value':p_value,'std_err':std_err,'avg_total':avg_result_total_r_succ,'pooled_std':pooled_std_result_total_r_succ}

                if plot_both_bool:
                        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(rp_levels_p,avg_result_total_p_succ)
                        line = np.multiply(p_linspace,slope)+intercept
                        plt.plot(p_linspace,line,color='pink')
                        p_succ_result_linreg = {'slope':slope,'intercept':intercept,'r_value':r_value,'p_value':p_value,'std_err':std_err,'avg_total':avg_result_total_p_succ,'pooled_std':pooled_std_result_total_p_succ}
        
                        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(rp_levels_r,avg_result_total_r_fail)
                        line = np.multiply(r_linspace,slope)+intercept
                        plt.plot(r_linspace,line,color='palegreen')
                        r_fail_result_linreg = {'slope':slope,'intercept':intercept,'r_value':r_value,'p_value':p_value,'std_err':std_err,'avg_total':avg_result_total_r_fail,'pooled_std':pooled_std_result_total_r_fail}


                plt.xlim(xmin = -3.5, xmax = 3.5)
                #plt.ylim(ymin = 0, ymax = 1.0)
                plt.title('result')
                #does ylim obfuscate smaller changes?

                #plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),ncol=1)

                #plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
                plt.subplots_adjust(right=0.4)

                #label_sig(i,j,txt,range,means)
                #i and j the two to compare, range for r or p
                #                      np.asarray(rp_levels_r)
                #label_sig(0,1,'p=0.01',rp_levels_r,avg_result_total_r)
                #label_sig(0,2,'p=0.01',rp_levels_r,avg_result_total_r)
                #label_sig(2,3,'p=0.01',rp_levels_r,avg_result_total_r)
                
                #TODO unhardcode 0.5 sec here
                plt.suptitle('region: %s, unit: %s, 0.5s window' %(region_key, str(i).zfill(2)), fontsize = 20) 
                plt.tight_layout()
                plt.subplots_adjust(top = 0.85)
                plt.subplots_adjust(right = 0.65)
                plt.savefig('avg_rp_%s_%s' %(region_key,str(i).zfill(2)))
                plt.clf()
        
                avg_std_dict = {'avg_cue_total_p_fail':avg_cue_total_p_fail,'avg_cue_total_r_succ':avg_cue_total_r_succ,'avg_cue_total_p_succ':avg_cue_total_p_succ,'avg_cue_total_r_fail':avg_cue_total_r_fail,'pooled_std_cue_total_p_fail':pooled_std_cue_total_p_fail,'pooled_std_cue_total_r_succ':pooled_std_cue_total_r_succ,'pooled_std_cue_total_p_succ':pooled_std_cue_total_p_succ,'pooled_std_cue_total_r_fail':pooled_std_cue_total_r_fail,'avg_result_total_p_fail':avg_result_total_p_fail,'avg_result_total_r_succ':avg_result_total_r_succ,'avg_result_total_p_succ':avg_result_total_p_succ,'avg_result_total_r_fail':avg_result_total_r_fail,'pooled_std_result_total_p_fail':pooled_std_result_total_p_fail,'pooled_std_result_total_r_succ':pooled_std_result_total_r_succ,'pooled_std_result_total_p_succ':pooled_std_result_total_p_succ,'pooled_std_result_total_r_fail':pooled_std_result_total_r_fail}

                if plot_both_bool:
                        return_unit_dict = {'p_fail_cue_linreg':p_fail_cue_linreg,'r_succ_cue_linreg':r_succ_cue_linreg,'p_succ_cue_linreg':p_succ_cue_linreg,'r_fail_cue_linreg':r_fail_cue_linreg,'p_fail_result_linreg':p_fail_result_linreg,'r_succ_result_linreg':r_succ_result_linreg,'p_succ_result_linreg':p_succ_result_linreg,'r_fail_result_linreg':r_fail_result_linreg,'avg_std_dict':avg_std_dict}
                else:
                        return_unit_dict = {'p_fail_cue_linreg':p_fail_cue_linreg,'r_succ_cue_linreg':r_succ_cue_linreg,'p_fail_result_linreg':p_fail_result_linreg,'r_succ_result_linreg':r_succ_result_linreg,'avg_std_dict':avg_std_dict}
                return_dicts.append(return_unit_dict)
        return(return_dicts)


for i in range(len(keys)):
        if 'S1' in keys[i]:
		S1_dict_total[keys[i]] = save_dict[keys[i]]
	elif 'M1' in keys[i]:
		M1_dict_total[keys[i]] = save_dict[keys[i]]
	elif 'PmD' in keys[i]:
		PmD_dict_total[keys[i]] = save_dict[keys[i]]
	elif 'PmV' in keys[i]:
		PmV_dict_total[keys[i]] = save_dict[keys[i]]
		
all_regions_dict = {'S1_dict_total':S1_dict_total,'M1_dict_total':M1_dict_total,'PmD_dict_total':PmD_dict_total,'PmV_dict_total':PmV_dict_total}
#all_regions_dict = {'M1_dict_total':M1_dict_total}

catch_bool = False
for key,val in all_regions_dict['M1_dict_total'].iteritems():
	if 'catch' in key:
		catch_bool = True

print 'catch bool is %s' %(catch_bool)
		

if stats_bool:
        for region_key,region_dict in all_regions_dict.iteritems():
                cue_dicts = {}
                result_dicts = {}
                for event_key,event_value in region_dict.iteritems():
                        if 'cue' in event_key:
                                cue_dicts[event_key] = event_value
                                stats = do_stats(event_key,event_value)
                                cue_dicts['stats_%s' %(event_key)] = stats
                        elif 'result' in event_key:
                                result_dicts[event_key] = event_value
                                stats = do_stats(event_key,event_value)
                                result_dicts['stats_%s' %(event_key)] = stats
                        else:
                                print 'not stat cue: %s' %(event_key)

                print '%s cue:' %(region_key)
                cue_comp_stats = do_comp_stats(cue_dicts,region_key,'cue')
                print '%s result:' %(region_key)
                result_comp_stats = do_comp_stats(result_dicts,region_key,'result')

                all_regions_dict[region_key]['cue_dicts'] = cue_dicts
                all_regions_dict[region_key]['result_dicts'] = result_dicts

                all_regions_dict[region_key]['cue_comp_stats'] = cue_comp_stats
                all_regions_dict[region_key]['result_comp_stats'] = result_comp_stats
 

if stats_bool:
	print '\n################################################\n###Time to interpret this shit #################\n################################################\n'
	sig_compare = {}
	for region_key,region_val in all_regions_dict.iteritems():
		cue_temp = all_regions_dict[region_key]['cue_dicts']
		key_temp = region_key[0:-6]
		key_temp = key_temp + 's'
		sig_cued_rewarding = []
		sig_cued_punishing = []

		print key_temp
		for i in range(len(cue_temp['stats_r0_succ_cue_%s' %(key_temp)]['mann_w_sig_bool'])):
			try:
                                if cue_temp['stats_r0_succ_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_r1_succ_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_r2_succ_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_r3_succ_cue_%s' %(key_temp)]['mann_w_sig_bool'][i]:
                                        sig_cued_rewarding.append(i)
                                        print 'unit %s sig for all succ cued rewarding' %(i)
                                elif cue_temp['stats_p0_fail_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_p1_fail_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_p2_fail_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_p3_fail_cue_%s' %(key_temp)]['mann_w_sig_bool'][i]:
                                        sig_cued_punishing.append(i)
                                        print 'unit %s sig for all unsucc cued punishing' %(i)
                                        #TODO all cued rewarding and punishing, not succ or unsucc
                        except:
                                print 'error, index %s' %(i)

                #TODO combinations here

		#because indexes in anova not the same
		for ind,region in all_regions_dict[region_key]['cue_comp_stats']['d3_index'].items():
			if 'catch' in region:
                                continue

                        if 'r0_succ' in region:
				r0_succ_cue_ind = ind
			elif 'r0_fail' in region:
				r0_fail_cue_ind = ind
			elif 'r1_succ' in region:
				r1_succ_cue_ind = ind
			elif 'r1_fail' in region:
				r1_fail_cue_ind = ind
			elif 'r2_succ' in region:
				r2_succ_cue_ind = ind
			elif 'r2_fail' in region:
				r2_fail_cue_ind = ind
			elif 'r3_succ' in region:
				r3_succ_cue_ind = ind
			elif 'r3_fail' in region:
				r3_fail_cue_ind = ind
			elif 'p0_succ' in region:
				p0_succ_cue_ind = ind
			elif 'p0_fail' in region:
				p0_fail_cue_ind = ind
			elif 'p1_succ' in region:
				p1_succ_cue_ind = ind
			elif 'p1_fail' in region:
				p1_fail_cue_ind = ind
			elif 'p2_succ' in region:
				p2_succ_cue_ind = ind
			elif 'p2_fail' in region:
				p2_fail_cue_ind = ind
			elif 'p3_succ' in region:
				p3_succ_cue_ind = ind
			elif 'p3_fail' in region:
				p3_fail_cue_ind = ind

                        #TODO above add catch cues

			
		for ind,region in all_regions_dict[region_key]['result_comp_stats']['d3_index'].items():
			if 'catch' in region:
                                continue
                        
                        if 'r0_succ' in region:
				r0_succ_result_ind = ind
			elif 'r0_fail' in region:
				r0_fail_result_ind = ind
			elif 'r1_succ' in region:
				r1_succ_result_ind = ind
			elif 'r1_fail' in region:
				r1_fail_result_ind = ind
			elif 'r2_succ' in region:
				r2_succ_result_ind = ind
			elif 'r2_fail' in region:
				r2_fail_result_ind = ind
			elif 'r3_succ' in region:
				r3_succ_result_ind = ind
			elif 'r3_fail' in region:
				r3_fail_result_ind = ind
			elif 'p0_succ' in region:
				p0_succ_result_ind = ind
			elif 'p0_fail' in region:
				p0_fail_result_ind = ind
			elif 'p1_succ' in region:
				p1_succ_result_ind = ind
			elif 'p1_fail' in region:
				p1_fail_result_ind = ind
			elif 'p2_succ' in region:
				p2_succ_result_ind = ind
			elif 'p2_fail' in region:
				p2_fail_result_ind = ind
			elif 'p3_succ' in region:
				p3_succ_result_ind = ind
			elif 'p3_fail' in region:
				p3_fail_result_ind = ind

                        #TODO above add catch cues, below too

			# ind of tukey.reject bool array = val x vs val y in anova
			#0 = 0vs1   #10 = 0vs11   #20 = 1vs7    #30 = 2vs4    #40 = 2vs14   #50 = 3vs12   #60 = 4vs11   #70 = 5vs11   #80 = 6vs12   #90 = 7vs14   #100 = 9vs11   #110 = 11vs12 
			#1 = 0vs2   #11 = 0vs12   #21 = 1vs8    #31 = 2vs5    #41 = 2vs15   #51 = 3vs13   #61 = 4vs12   #71 = 5vs12   #81 = 6vs13   #91 = 7vs15   #101 = 9vs12   #111 = 11vs13
			#2 = 0vs3   #12 = 0vs13   #22 = 1vs9    #32 = 2vs6    #42 = 3vs4    #52 = 3vs14   #62 = 4vs13   #72 = 5vs13   #82 = 6vs14   #92 = 8vs9    #102 = 9vs13   #112 = 11vs14
			#3 = 0vs4   #13 = 0vs14   #23 = 1vs10   #33 = 2vs7    #43 = 3vs5    #53 = 3vs15   #63 = 4vs14   #73 = 5vs14   #83 = 6vs15   #93 = 8vs10   #103 = 9vs14   #113 = 11vs15
			#4 = 0vs5   #14 = 0vs15   #24 = 1vs11   #34 = 2vs8    #44 = 3vs6    #54 = 4vs5    #64 = 4vs15   #74 = 5vs15   #84 = 7vs8    #94 = 8vs11   #104 = 9vs15   #114 = 12vs13
			#5 = 0vs6   #15 = 1vs2    #25 = 1vs12   #35 = 2vs9    #45 = 3vs7    #55 = 4vs6    #65 = 5vs6    #75 = 6vs7    #85 = 7vs9    #95 = 8vs12   #105 = 10vs11  #115 = 12vs14
			#6 = 0vs7   #16 = 1vs3    #26 = 1vs13   #36 = 2vs10   #46 = 3vs8    #56 = 4vs7    #66 = 5vs7    #76 = 6vs8    #86 = 7vs10   #96 = 8vs13   #106 = 10vs12  #116 = 12vs15
			#7 = 0vs8   #17 = 1vs4    #27 = 1vs14   #37 = 2vs11   #47 = 3vs9    #57 = 4vs8    #67 = 5vs8    #77 = 6vs9    #87 = 7vs11   #97 = 8vs14   #107 = 10vs13  #117 = 13vs14
			#8 = 0vs9   #18 = 1vs5    #28 = 1vs15   #38 = 2vs12   #48 = 3vs10   #58 = 4vs9    #68 = 5vs9    #78 = 6vs10   #88 = 7vs12   #98 = 8vs15   #108 = 10vs14  #118 = 13vs15
			#9 = 0vs10  #19 = 1vs6    #29 = 2vs3    #39 = 2vs13   #49 = 3vs11   #59 = 4vs10   #69 = 5vs10   #79 = 6vs11   #89 = 7vs13   #99 = 9vs10   #109 = 10vs15  #119 = 14vs15

			all_inds = np.array(((0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(0,10),(0,11),(0,12),(0,13),(0,14),(0,15),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(1,10),(1,11),(1,12),(1,13),(1,14),(1,15),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(2,10),(2,11),(2,12),(2,13),(2,14),(2,15),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(3,12),(3,13),(3,14),(3,15),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),(4,13),(4,14),(4,15),(5,6),(5,7),(5,8),(5,9),(5,10),(5,11),(5,12),(5,13),(5,14),(5,16),(6,7),(6,8),(6,9),(6,10),(6,11),(6,12),(6,13),(6,14),(6,15),(7,8),(7,9),(7,10),(7,11),(7,12),(7,13),(7,14),(7,15),(8,9),(8,10),(8,11),(8,12),(8,13),(8,14),(8,15),(9,10),(9,11),(9,12),(9,13),(9,14),(9,15),(10,11),(10,12),(10,13),(10,14),(10,15),(11,12),(11,13),(11,14),(11,15),(12,13),(12,14),(12,15),(13,14),(13,15),(14,15)))

		print 'running tukey posthoc'
	
		#what to compare? if rewarding trials sig diff from nonrewarding and/or punishing, and if punishing tirals sig diff from nonpunishing and/or rewarding
		post_hoc_cue = all_regions_dict[region_key]['cue_comp_stats']['for_post_hoc']
		post_hoc_result = all_regions_dict[region_key]['result_comp_stats']['for_post_hoc']
		posthoc_diffs_dict_cue = {}
                
                if (not post_hoc_cue == 0) and (not len(post_hoc_cue)==0):
                        for i in range(len(post_hoc_cue)): #: if (len(post_hoc_cue) != 0):
                                tukey_cue = post_hoc_cue[i]['tukey']
                                reject_cue = tukey_cue.reject
                                unit = post_hoc_cue[i]['unit_no']

                                diffs = np.where(reject_cue)
                                paired_name_list = []
                                if len(diffs[0]) > 0:
                                        for j in range(len(diffs[0])):
                                                pair_temp = all_inds[diffs[0][j]]
                                                temp_a = pair_temp[0] 
                                                temp_b = pair_temp[1] 
				
                                                #print unit
                                                pair_names = []
                                                if temp_a == r0_succ_cue_ind or temp_b == r0_succ_cue_ind:
                                                        pair_names.append('r0_succ_cue')
                                                if temp_a == r0_fail_cue_ind or temp_b == r0_fail_cue_ind:
                                                        pair_names.append('r0_fail_cue')
                                                if temp_a == r1_succ_cue_ind or temp_b == r1_succ_cue_ind:
                                                        pair_names.append('r1_succ_cue')
                                                if temp_a == r1_fail_cue_ind or temp_b == r1_fail_cue_ind:
                                                        pair_names.append('r1_fail_cue')
                                                if temp_a == r2_succ_cue_ind or temp_b == r2_succ_cue_ind:
                                                        pair_names.append('r2_succ_cue')
                                                if temp_a == r2_fail_cue_ind or temp_b == r2_fail_cue_ind:
                                                        pair_names.append('r2_fail_cue')
                                                if temp_a == r3_succ_cue_ind or temp_b == r3_succ_cue_ind:
                                                        pair_names.append('r3_succ_cue')
                                                if temp_a == r3_fail_cue_ind or temp_b == r3_fail_cue_ind:
                                                        pair_names.append('r3_fail_cue')
                                                if temp_a == p0_succ_cue_ind or temp_b == p0_succ_cue_ind:
                                                        pair_names.append('p0_succ_cue')
                                                if temp_a == p0_fail_cue_ind or temp_b == p0_fail_cue_ind:
                                                        pair_names.append('p0_fail_cue')
                                                if temp_a == p1_succ_cue_ind or temp_b == p1_succ_cue_ind:
                                                        pair_names.append('p1_succ_cue')
                                                if temp_a == p1_fail_cue_ind or temp_b == p1_fail_cue_ind:
                                                        pair_names.append('p1_fail_cue')
                                                if temp_a == p2_succ_cue_ind or temp_b == p2_succ_cue_ind:
                                                        pair_names.append('p2_succ_cue')
                                                if temp_a == p2_fail_cue_ind or temp_b == p2_fail_cue_ind:
                                                        pair_names.append('p2_fail_cue')
                                                if temp_a == p3_succ_cue_ind or temp_b == p3_succ_cue_ind:
                                                        pair_names.append('p3_succ_cue')
                                                if temp_a == p3_fail_cue_ind or temp_b == p3_fail_cue_ind:
                                                        pair_names.append('p3_fail_cue')

                                        paired_name_list.append(pair_names)
 					
			posthoc_diffs_dict_cue[unit] = paired_name_list

		posthoc_diffs_dict_result = {}
		for i in range(len(post_hoc_result)):
			tukey_result = post_hoc_result[i]['tukey']
			reject_result = tukey_result.reject
			unit = post_hoc_result[i]['unit_no']
			
			diffs = np.where(reject_result)
			paired_name_list = []
			if len(diffs[0]) > 0:
				for j in range(len(diffs[0])):
					pair_temp = all_inds[diffs[0][j]]
					temp_a = pair_temp[0] 
					temp_b = pair_temp[1] 
					
					pair_names = []
					if temp_a == r0_succ_result_ind or temp_b == r0_succ_result_ind:
						pair_names.append('r0_succ_result')
					if temp_a == r0_fail_result_ind or temp_b == r0_fail_result_ind:
						pair_names.append('r0_fail_result')
					if temp_a == r1_succ_result_ind or temp_b == r1_succ_result_ind:
						pair_names.append('r1_succ_result')
					if temp_a == r1_fail_result_ind or temp_b == r1_fail_result_ind:
						pair_names.append('r1_fail_result')
					if temp_a == r2_succ_result_ind or temp_b == r2_succ_result_ind:
						pair_names.append('r2_succ_result')
					if temp_a == r2_fail_result_ind or temp_b == r2_fail_result_ind:
						pair_names.append('r2_fail_result')
					if temp_a == r3_succ_result_ind or temp_b == r3_succ_result_ind:
						pair_names.append('r3_succ_result')
					if temp_a == r3_fail_result_ind or temp_b == r3_fail_result_ind:
						pair_names.append('r3_fail_result')
					if temp_a == p0_succ_result_ind or temp_b == p0_succ_result_ind:
						pair_names.append('p0_succ_result')
					if temp_a == p0_fail_result_ind or temp_b == p0_fail_result_ind:
						pair_names.append('p0_fail_result')
					if temp_a == p1_succ_result_ind or temp_b == p1_succ_result_ind:
						pair_names.append('p1_succ_result')
					if temp_a == p1_fail_result_ind or temp_b == p1_fail_result_ind:
						pair_names.append('p1_fail_result')
					if temp_a == p2_succ_result_ind or temp_b == p2_succ_result_ind:
						pair_names.append('p2_succ_result')
					if temp_a == p2_fail_result_ind or temp_b == p2_fail_result_ind:
						pair_names.append('p2_fail_result')
					if temp_a == p3_succ_result_ind or temp_b == p3_succ_result_ind:
						pair_names.append('p3_succ_result')
					if temp_a == p3_fail_result_ind or temp_b == p3_fail_result_ind:
						pair_names.append('p3_fail_result')

					paired_name_list.append(pair_names)					
			posthoc_diffs_dict_result[unit] = paired_name_list

		
		sig_compare[key_temp] = {'sig_cued_rewarding':sig_cued_rewarding,'sig_cued_punishing':sig_cued_punishing,'posthoc_diffs_dict_cue':posthoc_diffs_dict_cue,'posthoc_diffs_dict_result':posthoc_diffs_dict_result}

#if not stats_bool:
#        sig_compare = {}

if plot_bool:
	print "begin plotting\n"

        plot_return = {}
	for region_key,region_val in all_regions_dict.iteritems():		
		plot_return[region_key] = plot_comparison(region_key,region_val)

print '\nanalyzing linear regression'
linreg_analysis = {}
for region_key,region_val in plot_return.iteritems():
        print region_key
        r_p_sig = []
        pos_slopes = []
        neg_slopes = []
        linreg_analysis[region_key]={}
        for unit_num in range(len(plot_return[region_key])):
                for type_key,type_data in plot_return[region_key][unit_num].iteritems():
                        if type_key == 'avg_std_dict':
                                continue

                        p_value = plot_return[region_key][unit_num][type_key]['p_value']
                        r_value = plot_return[region_key][unit_num][type_key]['r_value']
                        std_err = plot_return[region_key][unit_num][type_key]['std_err']
                        slope = plot_return[region_key][unit_num][type_key]['slope']
                        
                        if r_value > 0.6 and p_value < 0.1:
                                #print '%s unit %s: r = %s, p = %s' %(type_key,unit_num,r_value,p_value)
                                temp = [type_key,unit_num,r_value,p_value,slope]
                                r_p_sig.append(temp)
                        
                        #TODO instead of hardcoding, make units with greatest/lowest x% of slope
                        if slope > 0.01:
                                #print 'POS: %s unit %s: slope = %s' %(type_key,unit_num,slope)
                                temp = [type_key,unit_num,slope]
                                pos_slopes.append(temp)
                        elif slope < -0.01:
                                #print 'NEG: %s unit %s: slope = %s' %(type_key,unit_num,slope)
                                temp = [type_key,unit_num,slope]
                                neg_slopes.append(temp)
     

        linreg_analysis[region_key]['r_p_sig'] = r_p_sig
        linreg_analysis[region_key]['pos_slopes'] = pos_slopes
        linreg_analysis[region_key]['neg_slopes'] = neg_slopes

if save_bool:	
	#save npy and xls files
	print 'saving data'
        dict_to_save = {'all_regions_dict':all_regions_dict,'sig_compare':sig_compare,'plot_return':plot_return,'linreg_analysis':linreg_analysis}
	np.save('data_%s' %(filename),dict_to_save)

	workbook = xlsxwriter.Workbook('data_%s.xlsx' %(filename),options={'nan_inf_to_errors':True})
	worksheet = workbook.add_worksheet()

	row = 0
	for region_key,region_data in all_regions_dict.iteritems():
		worksheet.write(row,0,region_key)
		worksheet.write(row,1,'cue_dicts')
		for event_key,event_data in all_regions_dict[region_key]['cue_dicts'].iteritems():
			if 'stats' in event_key:
				worksheet.write(row,2,event_key)
				worksheet.write(row,3,all_regions_dict[region_key]['cue_dicts'][event_key]['perc_sig'])
				row +=1
		worksheet.write(row,1,'result_dicts')
		for event_key,event_data in all_regions_dict[region_key]['result_dicts'].iteritems():
			if 'stats' in event_key:
				worksheet.write(row,2,event_key)
				worksheet.write(row,3,all_regions_dict[region_key]['result_dicts'][event_key]['perc_sig'])
				row +=1

	row = 0
	worksheet2 = workbook.add_worksheet()
	for region_key,region_data in all_regions_dict.iteritems():
		worksheet2.write(row,0,region_key)
		worksheet2.write(row,1,'cue_comp_stats')
		worksheet2.write(row,2,all_regions_dict[region_key]['cue_comp_stats']['anova_sig_perc'])
		worksheet2.write(row,3,'result_comp_stats')
		worksheet2.write(row,4,all_regions_dict[region_key]['result_comp_stats']['anova_sig_perc'])
		row +=1

        row = 1
        worksheet3 = workbook.add_worksheet()
        worksheet3.write(0,0,'region')
        worksheet3.write(0,1,'unit')
        worksheet3.write(0,2,'event')
        worksheet3.write(0,3,'slope')
        worksheet3.write(0,4,'intercept')
        worksheet3.write(0,5,'p_value')
        worksheet3.write(0,6,'r_value')
        worksheet3.write(0,7,'std_err')
        for region_key,region_data in plot_return.iteritems():
                worksheet3.write(row,0,region_key)
                for i in range(len(plot_return[region_key])):
                        worksheet3.write(row,1,i)
                        for type_key,type_data in plot_return[region_key][i].iteritems():
                                if type_key == 'avg_std_dict':
                                        continue

                                worksheet3.write(row,2,type_key)
                                worksheet3.write(row,3,plot_return[region_key][i][type_key]['slope'])
                                worksheet3.write(row,4,plot_return[region_key][i][type_key]['intercept'])
                                worksheet3.write(row,5,plot_return[region_key][i][type_key]['p_value'])
                                worksheet3.write(row,6,plot_return[region_key][i][type_key]['r_value'])
                                worksheet3.write(row,7,plot_return[region_key][i][type_key]['std_err'])
                                row +=1
	pyexcel.merge_all_to_a_book(glob.glob('*.csv'),'post_hoc_results_%s.xlsx' %(filename))

for csv in glob.glob('*.csv'):
        os.remove(csv)

name_list = ['avg_cue_p_fail','std_cue_p_fail','avg_cue_r_succ','std_cue_r_succ','avg_cue_p_succ','std_cue_p_succ','avg_cue_r_fail','std_cue_r_fail','avg_result_p_fail','std_result_p_fail','avg_result_r_succ','std_result_r_succ','avg_result_p_succ','std_result_p_succ','avg_result_r_fail','std_result_r_fail']

for region_key,region_data in plot_return.iteritems():
        nlized_data_workbook = xlsxwriter.Workbook('avg_std_nlized_data_%s.xlsx' %(region_key),options={'nan_inf_to_errors':True})
        for i in range(len(plot_return[region_key])):
                
                temp_array = np.zeros((4,16))
                temp_array[:,0] = plot_return[region_key][i]['avg_std_dict']['avg_cue_total_p_fail']
                temp_array[:,1] = plot_return[region_key][i]['avg_std_dict']['pooled_std_cue_total_p_fail']
                temp_array[:,2] = plot_return[region_key][i]['avg_std_dict']['avg_cue_total_r_succ']
                temp_array[:,3] = plot_return[region_key][i]['avg_std_dict']['pooled_std_cue_total_r_succ']
                temp_array[:,4] = plot_return[region_key][i]['avg_std_dict']['avg_cue_total_p_succ']
                temp_array[:,5] = plot_return[region_key][i]['avg_std_dict']['pooled_std_cue_total_p_succ']
                temp_array[:,6] = plot_return[region_key][i]['avg_std_dict']['avg_cue_total_r_fail']
                temp_array[:,7] = plot_return[region_key][i]['avg_std_dict']['pooled_std_cue_total_r_fail']
                temp_array[:,8] = plot_return[region_key][i]['avg_std_dict']['avg_result_total_p_fail']
                temp_array[:,9] = plot_return[region_key][i]['avg_std_dict']['pooled_std_result_total_p_fail']
                temp_array[:,10] = plot_return[region_key][i]['avg_std_dict']['avg_result_total_r_succ']
                temp_array[:,11] = plot_return[region_key][i]['avg_std_dict']['pooled_std_result_total_r_succ']
                temp_array[:,12] = plot_return[region_key][i]['avg_std_dict']['avg_result_total_p_succ']
                temp_array[:,13] = plot_return[region_key][i]['avg_std_dict']['pooled_std_result_total_p_succ']
                temp_array[:,14] = plot_return[region_key][i]['avg_std_dict']['avg_result_total_r_fail']
                temp_array[:,15] = plot_return[region_key][i]['avg_std_dict']['pooled_std_result_total_r_fail']
                
                worksheet_nld = nlized_data_workbook.add_worksheet('unit_%s' %(i))
                worksheet_nld.write_row(0,0,name_list)
                for j in range(temp_array.shape[0]):
                        worksheet_nld.write_row(j+1,0,temp_array[j,:])
                        
for region_key,region_data in all_regions_dict.iteritems():
        nl_avg_data_workbook = xlsxwriter.Workbook('nl_avg_data_%s.xlsx' %(region_key),options={'nan_inf_to_errors':True})
        suffix = '_%ss' %(region_key[0:-6])

        nlized_cue_comparison_data = all_regions_dict[region_key]['cue_comp_stats']['nlized_comparison_data']
        nlized_result_comparison_data = all_regions_dict[region_key]['result_comp_stats']['nlized_comparison_data']

        for i in range(len(nlized_cue_comparison_data['r0_succ_cue%s' %(suffix)])):
                temp_array = np.zeros((30,40))
                #temp_array[:,0] = np.mean(nlized_cue_comparison_data['p3_fail_cue%s' %(suffix)], axis = 0)
                try:
                        temp_array[:,0] = nlized_cue_comparison_data['p3_fail_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,1] = nlized_cue_comparison_data['p2_fail_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,2] = nlized_cue_comparison_data['p1_fail_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,3] = nlized_cue_comparison_data['p0_fail_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,4] = nlized_cue_comparison_data['r0_succ_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,5] = nlized_cue_comparison_data['r1_succ_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,6] = nlized_cue_comparison_data['r2_succ_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,7] = nlized_cue_comparison_data['r3_succ_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,8] = nlized_cue_comparison_data['p3_succ_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,9] = nlized_cue_comparison_data['p2_succ_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,10] = nlized_cue_comparison_data['p1_succ_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,11] = nlized_cue_comparison_data['p0_succ_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,12] = nlized_cue_comparison_data['r0_fail_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,13] = nlized_cue_comparison_data['r1_fail_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,14] = nlized_cue_comparison_data['r2_fail_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,15] = nlized_cue_comparison_data['r3_fail_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,16] = nlized_result_comparison_data['p3_fail_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,17] = nlized_result_comparison_data['p2_fail_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,18] = nlized_result_comparison_data['p1_fail_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,19] = nlized_result_comparison_data['p0_fail_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,20] = nlized_result_comparison_data['r0_succ_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,21] = nlized_result_comparison_data['r1_succ_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,22] = nlized_result_comparison_data['r2_succ_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,23] = nlized_result_comparison_data['r3_succ_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,24] = nlized_result_comparison_data['p3_succ_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,25] = nlized_result_comparison_data['p2_succ_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,26] = nlized_result_comparison_data['p1_succ_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,27] = nlized_result_comparison_data['p0_succ_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,28] = nlized_result_comparison_data['r0_fail_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,29] = nlized_result_comparison_data['r1_fail_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,30] = nlized_result_comparison_data['r2_fail_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,31] = nlized_result_comparison_data['r3_fail_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,32] = nlized_result_comparison_data['ra_succ_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,33] = nlized_result_comparison_data['ra_fail_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,34] = nlized_result_comparison_data['pa_succ_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,35] = nlized_result_comparison_data['pa_fail_cue%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,36] = nlized_result_comparison_data['ra_succ_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,37] = nlized_result_comparison_data['ra_fail_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,38] = nlized_result_comparison_data['pa_succ_result%s' %(suffix)][i,:]
                except:
                        pass
                try:
                        temp_array[:,39] = nlized_result_comparison_data['pa_fail_result%s' %(suffix)][i,:]
                except:
                        pass

                names = ['p3_fail_cue','p2_fail_cue','p1_fail_cue','p0_fail_cue','r0_succ_cue','r1_succ_cue','r2_succ_cue','r3_succ_cue','p3_succ_cue','p2_succ_cue','p1_succ_cue','p0_succ_cue','r0_fail_cue','r1_fail_cue','r2_fail_cue','r3_fail_cue','p3_fail_result','p2_fail_result','p1_fail_result','p0_fail_result','r0_succ_result','r1_succ_result','r2_succ_result','r3_succ_result','p3_succ_result','p2_succ_result','p1_succ_result','p0_succ_result','r0_fail_result','r1_fail_result','r2_fail_result','r3_fail_result','ra_succ_cue','ra_fail_cue','pa_succ_cue','pa_fail_cue','ra_succ_result','ra_fail_result','pa_succ_result','pa_fail_result']
        
                avg_worksheet = nl_avg_data_workbook.add_worksheet('unit_%s' %(i))
                avg_worksheet.write_row(0,0,names)
                for j in range(temp_array.shape[0]):
                        avg_worksheet.write_row(j+1,0,temp_array[j,:])

###TODO where's PmD?
##TODO add std to above

#TODO should just do this once
for region_key,region_data in all_regions_dict.iteritems():
        gf_workbook = xlsxwriter.Workbook('gf_avg_data_%s.xlsx' %(region_key),options={'nan_inf_to_errors':True})
        suffix = '_%ss' %(region_key[0:-6])
        temp_array = np.zeros((300,120))
        try:
                temp_array[:,0] = all_regions_dict[region_key]['p3_fail_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,1] = all_regions_dict[region_key]['p2_fail_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,2] = all_regions_dict[region_key]['p1_fail_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,3] = all_regions_dict[region_key]['p0_fail_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,4] = all_regions_dict[region_key]['r0_succ_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,5] = all_regions_dict[region_key]['r1_succ_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,6] = all_regions_dict[region_key]['r2_succ_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,7] = all_regions_dict[region_key]['r3_succ_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,8] = all_regions_dict[region_key]['p3_succ_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,9] = all_regions_dict[region_key]['p2_succ_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,10] = all_regions_dict[region_key]['p1_succ_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,11] = all_regions_dict[region_key]['p0_succ_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,12] = all_regions_dict[region_key]['r0_fail_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,13] = all_regions_dict[region_key]['r1_fail_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,14] = all_regions_dict[region_key]['r2_fail_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,15] = all_regions_dict[region_key]['r3_fail_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,16] = all_regions_dict[region_key]['p3_fail_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,17] = all_regions_dict[region_key]['p2_fail_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,18] = all_regions_dict[region_key]['p1_fail_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,19] = all_regions_dict[region_key]['p0_fail_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,20] = all_regions_dict[region_key]['r0_succ_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,21] = all_regions_dict[region_key]['r1_succ_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,22] = all_regions_dict[region_key]['r2_succ_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,23] = all_regions_dict[region_key]['r3_succ_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,24] = all_regions_dict[region_key]['p3_succ_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,25] = all_regions_dict[region_key]['p2_succ_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,26] = all_regions_dict[region_key]['p1_succ_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,27] = all_regions_dict[region_key]['p0_succ_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,28] = all_regions_dict[region_key]['r0_fail_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,29] = all_regions_dict[region_key]['r1_fail_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,30] = all_regions_dict[region_key]['r2_fail_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,31] = all_regions_dict[region_key]['r3_fail_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,32] = all_regions_dict[region_key]['p3_catch_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,33] = all_regions_dict[region_key]['p2_catch_cuet%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,34] = all_regions_dict[region_key]['p1_catch_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,35] = all_regions_dict[region_key]['p0_catch_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,36] = all_regions_dict[region_key]['r0_catch_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,37] = all_regions_dict[region_key]['r1_catch_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,38] = all_regions_dict[region_key]['r2_catch_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,39] = all_regions_dict[region_key]['r3_catch_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,40] = all_regions_dict[region_key]['p3_catch_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,41] = all_regions_dict[region_key]['p2_catch_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,42] = all_regions_dict[region_key]['p1_catch_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,43] = all_regions_dict[region_key]['p0_catch_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,44] = all_regions_dict[region_key]['r0_catch_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,45] = all_regions_dict[region_key]['r1_catch_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,46] = all_regions_dict[region_key]['r2_catch_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,47] = all_regions_dict[region_key]['r3_catch_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,48] = all_regions_dict[region_key]['p3_fail_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,49] = all_regions_dict[region_key]['p2_fail_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,50] = all_regions_dict[region_key]['p1_fail_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,51] = all_regions_dict[region_key]['p0_fail_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,52] = all_regions_dict[region_key]['r0_succ_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,53] = all_regions_dict[region_key]['r1_succ_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,54] = all_regions_dict[region_key]['r2_succ_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,55] = all_regions_dict[region_key]['r3_succ_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,56] = all_regions_dict[region_key]['p3_succ_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,57] = all_regions_dict[region_key]['p2_succ_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,58] = all_regions_dict[region_key]['p1_succ_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,59] = all_regions_dict[region_key]['p0_succ_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,60] = all_regions_dict[region_key]['r0_fail_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,61] = all_regions_dict[region_key]['r1_fail_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,62] = all_regions_dict[region_key]['r2_fail_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,63] = all_regions_dict[region_key]['r3_fail_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,64] = all_regions_dict[region_key]['p3_fail_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,65] = all_regions_dict[region_key]['p2_fail_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,66] = all_regions_dict[region_key]['p1_fail_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,67] = all_regions_dict[region_key]['p0_fail_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,68] = all_regions_dict[region_key]['r0_succ_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,69] = all_regions_dict[region_key]['r1_succ_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,70] = all_regions_dict[region_key]['r2_succ_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,71] = all_regions_dict[region_key]['r3_succ_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,72] = all_regions_dict[region_key]['p3_succ_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,73] = all_regions_dict[region_key]['p2_succ_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,74] = all_regions_dict[region_key]['p1_succ_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,75] = all_regions_dict[region_key]['p0_succ_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,76] = all_regions_dict[region_key]['r0_fail_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,77] = all_regions_dict[region_key]['r1_fail_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,78] = all_regions_dict[region_key]['r2_fail_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,79] = all_regions_dict[region_key]['r3_fail_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,80] = all_regions_dict[region_key]['p3_catch_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,81] = all_regions_dict[region_key]['p2_catch_cuet%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,82] = all_regions_dict[region_key]['p1_catch_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,83] = all_regions_dict[region_key]['p0_catch_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,84] = all_regions_dict[region_key]['r0_catch_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,85] = all_regions_dict[region_key]['r1_catch_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,86] = all_regions_dict[region_key]['r2_catch_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,87] = all_regions_dict[region_key]['r3_catch_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,88] = all_regions_dict[region_key]['p3_catch_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,89] = all_regions_dict[region_key]['p2_catch_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,90] = all_regions_dict[region_key]['p1_catch_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,91] = all_regions_dict[region_key]['p0_catch_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,92] = all_regions_dict[region_key]['r0_catch_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,93] = all_regions_dict[region_key]['r1_catch_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,94] = all_regions_dict[region_key]['r2_catch_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,95] = all_regions_dict[region_key]['r3_catch_result%s' %(suffix)][0]['gf_std']
        except:
                pass

        ###################
        try:
                temp_array[:,96] = all_regions_dict[region_key]['r_all_catch_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,97] = all_regions_dict[region_key]['p_all_catch_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,98] = all_regions_dict[region_key]['r_all_catch_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,99] = all_regions_dict[region_key]['p_all_catch_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,100] = all_regions_dict[region_key]['r_all_catch_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,101] = all_regions_dict[region_key]['p_all_catch_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,102] = all_regions_dict[region_key]['r_all_catch_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,103] = all_regions_dict[region_key]['p_all_catch_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        #########
        try:
                temp_array[:,104] = all_regions_dict[region_key]['ra_succ_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,105] = all_regions_dict[region_key]['ra_fail_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,106] = all_regions_dict[region_key]['pa_succ_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,107] = all_regions_dict[region_key]['pa_fail_cue%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,108] = all_regions_dict[region_key]['ra_succ_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,109] = all_regions_dict[region_key]['ra_fail_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,110] = all_regions_dict[region_key]['pa_succ_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,111] = all_regions_dict[region_key]['pa_fail_result%s' %(suffix)][0]['gf_avg']
        except:
                pass
        try:
                temp_array[:,112] = all_regions_dict[region_key]['ra_succ_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,113] = all_regions_dict[region_key]['ra_fail_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,114] = all_regions_dict[region_key]['pa_succ_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,115] = all_regions_dict[region_key]['pa_fail_cue%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,116] = all_regions_dict[region_key]['ra_succ_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,117] = all_regions_dict[region_key]['ra_fail_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,118] = all_regions_dict[region_key]['pa_succ_result%s' %(suffix)][0]['gf_std']
        except:
                pass
        try:
                temp_array[:,119] = all_regions_dict[region_key]['pa_fail_result%s' %(suffix)][0]['gf_std']
        except:
                pass


        names = ['p3_fail_cue_avg','p2_fail_cue_avg','p1_fail_cue_avg','p0_fail_cue_avg','r0_succ_cue_avg','r1_succ_cue_avg','r2_succ_cue_avg','r3_succ_cue_avg','p3_succ_cue_avg','p2_succ_cue_avg','p1_succ_cue_avg','p0_succ_cue_avg','r0_fail_cue_avg','r1_fail_cue_avg','r2_fail_cue_avg','r3_fail_cue_avg','p3_fail_result_avg','p2_fail_result_avg','p1_fail_result_avg','p0_fail_result_avg','r0_succ_result_avg','r1_succ_result_avg','r2_succ_result_avg','r3_succ_result_avg','p3_succ_result_avg','p2_succ_result_avg','p1_succ_result_avg','p0_succ_result_avg','r0_fail_result_avg','r1_fail_result_avg','r2_fail_result_avg','r3_fail_result_avg','p3_catch_cue_avg','p2_catch_cue_avg','p1_catch_cue_avg','p0_catch_cue_avg','r0_catch_cue_avg','r1_catch_cue_avg','r2_catch_cue_avg','r3_catch_cue_avg','p3_catch_result_avg','p2_catch_result_avg','p1_catch_result_avg','p0_catch_result_avg','r0_catch_result_avg','r1_catch_result_avg','r2_catch_result_avg','r3_catch_result_avg','p3_fail_cue_std','p2_fail_cue_std','p1_fail_cue_std','p0_fail_cue_std','r0_succ_cue_std','r1_succ_cue_std','r2_succ_cue_std','r3_succ_cue_std','p3_succ_cue_std','p2_succ_cue_std','p1_succ_cue_std','p0_succ_cue_std','r0_fail_cue_std','r1_fail_cue_std','r2_fail_cue_std','r3_fail_cue_std','p3_fail_result_std','p2_fail_result_std','p1_fail_result_std','p0_fail_result_std','r0_succ_result_std','r1_succ_result_std','r2_succ_result_std','r3_succ_result_std','p3_succ_result_std','p2_succ_result_std','p1_succ_result_std','p0_succ_result_std','r0_fail_result_std','r1_fail_result_std','r2_fail_result_std','r3_fail_result_std','p3_catch_cue_std','p2_catch_cue_std','p1_catch_cue_std','p0_catch_cue_std','r0_catch_cue_std','r1_catch_cue_std','r2_catch_cue_std','r3_catch_cue_std','p3_catch_result_std','p2_catch_result_std','p1_catch_result_std','p0_catch_result_std','r0_catch_result_std','r1_catch_result_std','r2_catch_result_std','r3_catch_result_std','r_all_catch_cue_avg','p_all_catch_cue_avg','r_all_catch_result_avg','p_all_catch_result_avg','r_all_catch_cue_std','p_all_catch_cue_std','r_all_catch_result_std','p_all_catch_result_std','ra_succ_cue_avg','ra_fail_cue_avg','pa_succ_cue_avg','pa_fail_cue_avg','ra_succ_result_avg','ra_fail_result_avg','pa_succ_result_avg','pa_fail_result_avg','ra_succ_cue_std','ra_fail_cue_std','pa_succ_cue_std','pa_fail_cue_std','ra_succ_result_std','ra_fail_result_std','pa_succ_result_std','pa_fail_result_std']
        
        gf_worksheet = gf_workbook.add_worksheet('unit_%s' %(0))
        gf_worksheet.write_row(0,0,names)
        for j in range(temp_array.shape[0]):
                gf_worksheet.write_row(j+1,0,temp_array[j,:])

