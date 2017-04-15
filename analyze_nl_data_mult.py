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
save_bool = False

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
		before_bins = event_data[i]['avg_nl'][0:num_bins_before]
		after_bins = event_data[i]['avg_nl'][num_bins_before:2*num_bins_before]

		try:
			mann_w_temp = sp.stats.mannwhitneyu(before_bins,after_bins,'two-sided')
			mann_w_sig_bool_temp = mann_w_temp[1] <= 0.05

		except:
			mann_w_temp = [99,99]   #TODO some other error?
			mann_w_sig_bool_temp = False

		mann_w.append(mann_w_temp)
		mann_w_sig_bool.append(mann_w_sig_bool_temp)

	stats_all['mann_w'] = mann_w
	stats_all['mann_w_sig_bool'] = mann_w_sig_bool

	perc_sig = sum(mann_w_sig_bool) / float(len(mann_w_sig_bool))
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
		if not 'stats' in key:
			dummy = np.asarray(all_dict[key])
			nl_array=[]
			for i in range(len(dummy)):
				nl_array.append(dummy[i]['avg_nl'])
			nl_array = np.asarray(nl_array)
			nlized_comparison_data[key] = nl_array

			d3_nl_array.append(nl_array)
			d3_index[index] = key
			index+=1


	d3_nl_array = np.asarray(d3_nl_array)

	all_before = d3_nl_array[:,:,0:num_bins_before]
	all_after = d3_nl_array[:,:,num_bins_before:2*num_bins_before]

	anovas = []

        #print 'd3_nl_array shape = %s' %(str(d3_nl_array.shape))

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

	anova_pval = anovas[:,1]
	anova_sig_pval = np.where(anova_pval <= 0.05)
	anova_sig_pval_bool = anova_pval <= 0.05

	anova_sig_perc = sum(anova_sig_pval_bool) / float(len(anova_sig_pval_bool))
	anova_sig_perc_str= np.around(anova_sig_perc *100,decimals = 2)
	print '%s%% units significantly different between events' %(anova_sig_perc_str)

	for_post_hoc = []
	for i in range(len(anova_sig_pval_bool)):
		if anova_sig_pval_bool[i]:
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
        print "plotting %s" %(region_key)
        suffix = '_%ss' %(region_key[0:-6])

        for i in range(len(all_regions_dict[region_key]['r0_succ_cue%s' %(suffix)])):
        #TESTING plot
        #for i in range(2):
		#before_bins = all_regions_dict[region_key][i]['avg_nl'][0:num_bins_before]
		#full second or half? half for now
                #after_bins = all_regions_dict[region_key][i]['avg_nl'][num_bins_before:2*num_bins_before]

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

                r0_fail_cue_pooled_std = np.sqrt(sum(np.square(r0_fail_cue_std_devs))/len(r0_succ_cue_std_devs))
                r1_fail_cue_pooled_std = np.sqrt(sum(np.square(r1_fail_cue_std_devs))/len(r1_succ_cue_std_devs))
                r2_fail_cue_pooled_std = np.sqrt(sum(np.square(r2_fail_cue_std_devs))/len(r2_succ_cue_std_devs))
                r3_fail_cue_pooled_std = np.sqrt(sum(np.square(r3_fail_cue_std_devs))/len(r3_succ_cue_std_devs))
                p0_succ_cue_pooled_std = np.sqrt(sum(np.square(p0_succ_cue_std_devs))/len(p0_fail_cue_std_devs))
                p1_succ_cue_pooled_std = np.sqrt(sum(np.square(p1_succ_cue_std_devs))/len(p1_fail_cue_std_devs))
                p2_succ_cue_pooled_std = np.sqrt(sum(np.square(p2_succ_cue_std_devs))/len(p2_fail_cue_std_devs))
                p3_succ_cue_pooled_std = np.sqrt(sum(np.square(p3_succ_cue_std_devs))/len(p3_fail_cue_std_devs))

                r0_fail_result_pooled_std = np.sqrt(sum(np.square(r0_fail_result_std_devs))/len(r0_succ_result_std_devs))
                r1_fail_result_pooled_std = np.sqrt(sum(np.square(r1_fail_result_std_devs))/len(r1_succ_result_std_devs))
                r2_fail_result_pooled_std = np.sqrt(sum(np.square(r2_fail_result_std_devs))/len(r2_succ_result_std_devs))
                r3_fail_result_pooled_std = np.sqrt(sum(np.square(r3_fail_result_std_devs))/len(r3_succ_result_std_devs))
                p0_succ_result_pooled_std = np.sqrt(sum(np.square(p0_succ_result_std_devs))/len(p0_fail_result_std_devs))
                p1_succ_result_pooled_std = np.sqrt(sum(np.square(p1_succ_result_std_devs))/len(p1_fail_result_std_devs))
                p2_succ_result_pooled_std = np.sqrt(sum(np.square(p2_succ_result_std_devs))/len(p2_fail_result_std_devs))
                p3_succ_result_pooled_std = np.sqrt(sum(np.square(p3_succ_result_std_devs))/len(p3_fail_result_std_devs))

                ax = plt.gca()
                plt.subplot(2,1,1)

                rp_levels_p = [-3,-2,-1,0]
                rp_levels_r = [0,1,2,3]
                avg_cue_total_p_fail = [p3_fail_cue_avg,p2_fail_cue_avg,p1_fail_cue_avg,p0_fail_cue_avg]
                avg_cue_total_r_succ = [r0_succ_cue_avg,r1_succ_cue_avg,r2_succ_cue_avg,r3_succ_cue_avg]
                pooled_std_cue_total_p_fail = [p3_fail_cue_pooled_std,p2_fail_cue_pooled_std,p1_fail_cue_pooled_std,p0_fail_cue_pooled_std]
                pooled_std_cue_total_r_succ = [r0_succ_cue_pooled_std,r1_succ_cue_pooled_std,r2_succ_cue_pooled_std,r3_succ_cue_pooled_std]

                avg_cue_total_p_succ = [p3_succ_cue_avg,p2_succ_cue_avg,p1_succ_cue_avg,p0_succ_cue_avg]
                avg_cue_total_r_fail = [r0_fail_cue_avg,r1_fail_cue_avg,r2_fail_cue_avg,r3_fail_cue_avg]
                pooled_std_cue_total_p_succ = [p3_succ_cue_pooled_std,p2_succ_cue_pooled_std,p1_succ_cue_pooled_std,p0_succ_cue_pooled_std]
                pooled_std_cue_total_r_fail = [r0_fail_cue_pooled_std,r1_fail_cue_pooled_std,r2_fail_cue_pooled_std,r3_fail_cue_pooled_std]

                linestyle = {'marker':'o','color':'r'}
                plt.errorbar(rp_levels_p,avg_cue_total_p_fail,yerr=pooled_std_cue_total_p_fail,**linestyle)
                linestyle = {'marker':'o','color':'g'}
                plt.errorbar(rp_levels_r,avg_cue_total_r_succ,yerr=pooled_std_cue_total_r_succ,**linestyle)
 
                if plot_both_bool:
                        linestyle = {'marker':'o','color':'pink'} #'lw':0.5}
                        plt.errorbar(rp_levels_p,avg_cue_total_p_succ,yerr=pooled_std_cue_total_p_succ,**linestyle)
                        linestyle = {'marker':'o','color':'palegreen'} #'lw':0.5}
                        plt.errorbar(rp_levels_r,avg_cue_total_r_fail,yerr=pooled_std_cue_total_r_fail,**linestyle)
                        
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

                linestyle = {'marker':'o','color':'r'}
                plt.errorbar(rp_levels_p,avg_result_total_p_fail,yerr=pooled_std_result_total_p_fail,**linestyle)
                linestyle = {'marker':'o','color':'g'}
                plt.errorbar(rp_levels_r,avg_result_total_r_succ,yerr=pooled_std_result_total_r_succ,**linestyle)

                if plot_both_bool:
                        linestyle = {'marker':'o','color':'pink','lw':0.5}
                        plt.errorbar(rp_levels_p,avg_result_total_p_succ,yerr=pooled_std_result_total_p_succ,**linestyle)
                        linestyle = {'marker':'o','color':'palegreen','lw':0.5}
                        plt.errorbar(rp_levels_r,avg_result_total_r_fail,yerr=pooled_std_result_total_r_fail,**linestyle)

                plt.xlim(xmin = -3.5, xmax = 3.5)
                #plt.ylim(ymin = 0, ymax = 1.0)
                plt.title('result')
                #does ylim obfuscate smaller changes?

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
                plt.savefig('avg_rp_%s_%s' %(region_key,str(i).zfill(2)))
                plt.clf()



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
		#	if 'stats' in event_key:
		cue_temp = all_regions_dict[region_key]['cue_dicts']
		key_temp = region_key[0:-6]
		key_temp = key_temp + 's'
		sig_cued_rewarding = []
		sig_cued_punishing = []

		print key_temp
		for i in range(len(cue_temp['stats_r0_succ_cue_%s' %(key_temp)]['mann_w_sig_bool'])):
			if cue_temp['stats_r0_succ_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_r1_succ_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_r2_succ_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_r3_succ_cue_%s' %(key_temp)]['mann_w_sig_bool'][i]:
				sig_cued_rewarding.append(i)
				print 'unit %s sig for all succ cued rewarding' %(i)
			elif cue_temp['stats_p0_fail_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_p1_fail_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_p2_fail_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_p3_fail_cue_%s' %(key_temp)]['mann_w_sig_bool'][i]:
				sig_cued_punishing.append(i)
				print 'unit %s sig for all unsucc cued punishing' %(i)
                                #TODO all cued rewarding and punishing, not succ or unsucc

                #TODO combinations here
		#result_temp = all_regions_dict[region_key]['result_dicts']
		#for i in range(len(delivery_temp['stats_r_only_s_rdelivery_%s' %(key_temp)]['mann_w_sig_bool'])):
		#	if delivery_temp['stats_r_only_s_rdelivery_%s' %(key_temp)]['mann_w_sig_bool'][i] and delivery_temp['stats_rp_s_rdelivery_%s' %(key_temp)]['mann_w_sig_bool'][i]:
		#		sig_reward_delivery.append(i)
		#		print 'unit %s sig for all reward delivery' %(i)
		#	elif delivery_temp['stats_nrnp_s_nextreset_%s' %(key_temp)]['mann_w_sig_bool'][i] and delivery_temp['stats_p_only_s_nextreset_%s' %(key_temp)]['mann_w_sig_bool'][i]:
		#		sig_succ_noreward.append(i)
		#		print 'unit %s sig for all successful without reward delivery' %(i)
		#	elif delivery_temp['stats_r_only_f_nextreset_%s' %(key_temp)]['mann_w_sig_bool'][i] and delivery_temp['stats_nrnp_f_nextreset_%s' %(key_temp)]['mann_w_sig_bool'][i]:
		#		sig_unsucc_nopunishment.append(i)
		#		print 'unit %s sig for all unsuccessful without punishment delivery' %(i)
		#	elif delivery_temp['stats_rp_f_pdelivery_%s' %(key_temp)]['mann_w_sig_bool'][i] and delivery_temp['stats_p_only_f_pdelivery_%s' %(key_temp)]['mann_w_sig_bool'][i]:
		#		sig_punishment_delivery.append(i)
		#		print 'unit %s sig for all punishment delivery' %(i)

		#because indexes in anova not the same
		for ind,region in all_regions_dict[region_key]['cue_comp_stats']['d3_index'].items():
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
		for i in range(len(post_hoc_cue)):
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




if plot_bool:
	print "begin plotting\n"

	for region_key,region_val in all_regions_dict.iteritems():		
		plot_comparison(region_key,region_val)

        #TESTING Plot
        #plot_comparison('M1_dict_total',all_regions_dict['M1_dict_total'])

if save_bool:	
	#save npy and xls files
	print 'saving data'
        dict_to_save = {'all_regions_dict':all_regions_dict,'sig_compare':sig_compare}
	np.save('data_%s' %(filename),dict_to_save)

	workbook = xlsxwriter.Workbook('data_%s.xlsx' %(filename))
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

	
	pyexcel.merge_all_to_a_book(glob.glob('*.csv'),'post_hoc_results_%s.xlsx' %(filename))

#TODO for now took out of save_bool if statement
for csv in glob.glob('*.csv'):
        os.remove(csv)
