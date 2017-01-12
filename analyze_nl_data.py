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

#TODO: on beaver install statsmodels

#######################
#params to set ########
#######################

filename = glob.glob('avg_fr_and_nlized_data*.npy')

print 'loading %s' %(filename[0])

save_dict = np.load(filename[0])[()]

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
	for i in range(len(event_key)):
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

			
		#stats['mann_w'] = mann_w
		#stats['mann_w_sig_bool'] = mann_w_sig_bool
		
		#stats_all.append(stats)

	#for i in range(len(event_key))
	stats_all['mann_w'] = mann_w
	stats_all['mann_w_sig_bool'] = mann_w_sig_bool

	perc_sig = sum(mann_w_sig_bool) / float(len(mann_w_sig_bool))
	stats_all['perc_sig'] = perc_sig
	perc_sig_str = np.around(perc_sig *100,decimals = 2)
	print '%s%% units in %s significantly different' %(perc_sig_str,event_key)
	
	return stats_all


def do_comp_stats(all_dict):

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

	#unit_stats = []
	anovas = []
	if d3_nl_array.shape[0] == 8:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(d3_nl_array[0][i],d3_nl_array[1][i],d3_nl_array[2][i],d3_nl_array[3][i],d3_nl_array[4][i],d3_nl_array[5][i],d3_nl_array[6][i],d3_nl_array[7][i]))
		anovas = np.asarray(anovas)
	elif dl_nl_array.shape[0] == 7:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(d3_nl_array[0][i],d3_nl_array[1][i],d3_nl_array[2][i],d3_nl_array[3][i],d3_nl_array[4][i],d3_nl_array[5][i],d3_nl_array[6][i]))
		anovas = np.asarray(anovas)
	elif dl_nl_array.shape[0] == 6:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(d3_nl_array[0][i],d3_nl_array[1][i],d3_nl_array[2][i],d3_nl_array[3][i],d3_nl_array[4][i],d3_nl_array[5][i]))
		anovas = np.asarray(anovas)
	elif dl_nl_array.shape[0] == 5:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(d3_nl_array[0][i],d3_nl_array[1][i],d3_nl_array[2][i],d3_nl_array[3][i],d3_nl_array[4][i]))
		anovas = np.asarray(anovas)
	elif dl_nl_array.shape[0] == 4:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(d3_nl_array[0][i],d3_nl_array[1][i],d3_nl_array[2][i],d3_nl_array[3][i]))
		anovas = np.asarray(anovas)
	elif dl_nl_array.shape[0] == 3:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(d3_nl_array[0][i],d3_nl_array[1][i],d3_nl_array[2][i]))
		anovas = np.asarray(anovas)
	elif dl_nl_array.shape[0] == 2:
		for i in range(d3_nl_array.shape[1]):
			anovas.append(sp.stats.f_oneway(d3_nl_array[0][i],d3_nl_array[1][i]))
		anovas = np.asarray(anovas)
	else:
		print 'ANOVA error'

	#for sig anovas, t-tests between each of the options (?)
		
	anova_pval = anovas[:,1]
	anova_sig_pval = np.where(anova_pval <= 0.05)
	anova_sig_pval_bool = anova_pval <= 0.05

	anova_sig_perc = sum(anova_sig_pval_bool) / float(len(anova_sig_pval_bool))

	anova_sig_perc_str= np.around(anova_sig_perc *100,decimals = 2)
	print '%s%% units significantly different between events' %(anova_sig_perc_str)


	return_dict={'d3_nl_array':d3_nl_array,'d3_index':d3_index,'nlized_comparison_data':nlized_comparison_data,'anovas':anovas,'anova_sig_pval':anova_sig_pval,'anova_sig_pval_bool':anova_sig_pval_bool,'anova_sig_perc':anova_sig_perc}
								
	return return_dict



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

for region_key,region_dict in all_regions_dict.iteritems():
	cue_dicts = {}
	delivery_dicts = {}
	for event_key,event_value in region_dict.iteritems():
		if 'cue' in event_key:
			cue_dicts[event_key] = event_value
			stats = do_stats(event_key,event_value)
			cue_dicts['stats_%s' %(event_key)] = stats
		else:
			delivery_dicts[event_key] = event_value
			stats = do_stats(event_key,event_value)
			delivery_dicts['stats_%s' %(event_key)] = stats

		#for event_key,event_value in cue_dicts.iteritems():
		#	if not 'stats' in event_key:
		#		comp_stats = do_comp_stats(event_key,event_value)
		#		#region_dict[event_key
		#for event_key,event_value in delivery_dicts.iteritems():
		#	if not 'stats' in event_key:
		#		comp_stats = do_comp_stats(event_key,event_value)

	print '%s cue:' %(region_key)
	cue_comp_stats = do_comp_stats(cue_dicts)
	print '%s delivery:' %(region_key)
	delivery_comp_stats = do_comp_stats(delivery_dicts)

	all_regions_dict[region_key]['cue_dicts'] = cue_dicts
	all_regions_dict[region_key]['delivery_dicts'] = delivery_dicts

	all_regions_dict[region_key]['cue_comp_stats'] = cue_comp_stats
	all_regions_dict[region_key]['delivery_comp_stats'] = delivery_comp_stats
								

	
