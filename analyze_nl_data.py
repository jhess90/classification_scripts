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

#TODO: on beaver install statsmodels, pyexcel, pyexcel_xlsx

filename = glob.glob('avg_fr_and_nlized_data*.npy')
filename = filename[0]

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
		#TODO did this fix it? Or is something else up??
		stats = {}
		#print i
		#print event_key
		#if len(event_data) > i:
		#	#TODO what's up here. 1015block10059
		#	mann_w_temp = [99,99]
		#	mann_w_sig_bool_temp = False
		#	break
		#print len(event_data)
		#print i
		#print len(event_key)
		#print event_data[i]['avg_nl']
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
	if d3_nl_array.shape[0] == 8:
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

	print '%s cue:' %(region_key)
	cue_comp_stats = do_comp_stats(cue_dicts,region_key,'cue')
	print '%s delivery:' %(region_key)
	delivery_comp_stats = do_comp_stats(delivery_dicts,region_key,'delivery')

	all_regions_dict[region_key]['cue_dicts'] = cue_dicts
	all_regions_dict[region_key]['delivery_dicts'] = delivery_dicts

	all_regions_dict[region_key]['cue_comp_stats'] = cue_comp_stats
	all_regions_dict[region_key]['delivery_comp_stats'] = delivery_comp_stats

	
print '\n################################################\n###Time to interpret this shit #################\n################################################\n'

#anova.reject = bool of comparisons

sig_compare = {}
for region_key,region_val in all_regions_dict.iteritems():
	#for event_key,event_val in all_regions_dict[region_key]['cue_dicts'].iteritems():
	#	if 'stats' in event_key:
	cue_temp = all_regions_dict[region_key]['cue_dicts']
	key_temp = region_key[0:-6]
	key_temp = key_temp + 's'
	sig_cued_rewarding = []
	sig_cued_punishing = []
	sig_reward_delivery = []
	sig_succ_noreward = []
	sig_punishment_delivery = []
	sig_unsucc_nopunishment = []

	
	print key_temp
	for i in range(len(cue_temp['stats_r_only_s_cue_%s' %(key_temp)]['mann_w_sig_bool'])):
		if cue_temp['stats_r_only_s_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_r_only_f_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_rp_s_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_rp_f_cue_%s' %(key_temp)]['mann_w_sig_bool'][i]:
			sig_cued_rewarding.append(i)
			print 'unit %s sig for all cued rewarding' %(i)
		elif cue_temp['stats_rp_s_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_rp_f_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_p_only_s_cue_%s' %(key_temp)]['mann_w_sig_bool'][i] and cue_temp['stats_p_only_f_cue_%s' %(key_temp)]['mann_w_sig_bool'][i]:
			sig_cued_punishing.append(i)
			print 'unit %s sig for all cued punishing' %(i)

			####TODO WHAT COMBINATIONS DO I NEED HERE?!?!
	delivery_temp = all_regions_dict[region_key]['delivery_dicts']
	for i in range(len(delivery_temp['stats_r_only_s_rdelivery_%s' %(key_temp)]['mann_w_sig_bool'])):
		if delivery_temp['stats_r_only_s_rdelivery_%s' %(key_temp)]['mann_w_sig_bool'][i] and delivery_temp['stats_rp_s_rdelivery_%s' %(key_temp)]['mann_w_sig_bool'][i]:
			sig_reward_delivery.append(i)
			print 'unit %s sig for all reward delivery' %(i)
		elif delivery_temp['stats_nrnp_s_nextreset_%s' %(key_temp)]['mann_w_sig_bool'][i] and delivery_temp['stats_p_only_s_nextreset_%s' %(key_temp)]['mann_w_sig_bool'][i]:
			sig_succ_noreward.append(i)
			print 'unit %s sig for all successful without reward delivery' %(i)
		elif delivery_temp['stats_r_only_f_nextreset_%s' %(key_temp)]['mann_w_sig_bool'][i] and delivery_temp['stats_nrnp_f_nextreset_%s' %(key_temp)]['mann_w_sig_bool'][i]:
			sig_unsucc_nopunishment.append(i)
			print 'unit %s sig for all unsuccessful without punishment delivery' %(i)
		elif delivery_temp['stats_rp_f_pdelivery_%s' %(key_temp)]['mann_w_sig_bool'][i] and delivery_temp['stats_p_only_f_pdelivery_%s' %(key_temp)]['mann_w_sig_bool'][i]:
			sig_punishment_delivery.append(i)
			print 'unit %s sig for all punishment delivery' %(i)

	#because indexes in anova not the same
	for ind,region in all_regions_dict[region_key]['cue_comp_stats']['d3_index'].items():
		if 'r_only_s' in region:
			r_only_s_cue_ind = ind
		elif 'r_only_f' in region:
			r_only_f_cue_ind = ind
		elif 'rp_s' in region:
			rp_s_cue_ind = ind
		elif 'rp_f' in region:
			rp_f_cue_ind = ind
		elif 'p_only_s' in region:
			p_only_s_cue_ind = ind
		elif 'p_only_f' in region:
			p_only_f_cue_ind = ind
		elif 'nrnp_s' in region:
			nrnp_s_cue_ind = ind
		elif 'nrnp_f' in region:
			nrnp_f_cue_ind = ind

			
	for ind,region in all_regions_dict[region_key]['delivery_comp_stats']['d3_index'].items():
		if 'r_only_s' in region:
			r_only_s_delivery_ind = ind
		elif 'r_only_f' in region:
			r_only_f_delivery_ind = ind
		elif 'rp_s' in region:
			rp_s_delivery_ind = ind
		elif 'rp_f' in region:
			rp_f_delivery_ind = ind
		elif 'p_only_s' in region:
			p_only_s_delivery_ind = ind
		elif 'p_only_f' in region:
			p_only_f_delivery_ind = ind
		elif 'nrnp_s' in region:
			nrnp_s_delivery_ind = ind
		elif 'nrnp_f' in region:
			nrnp_f_delivery_ind = ind

	# ind of tukey.reject bool array = val x vs val y in anova
	#0 = 0 vs 1     #10 = 1 vs 5     #20 = 3 vs 6
	#1 = 0 vs 2     #11 = 1 vs 6     #21 = 3 vs 7
	#2 = 0 vs 3     #12 = 1 vs 7     #22 = 4 vs 5
	#3 = 0 vs 4     #13 = 2 vs 3     #23 = 4 vs 6
	#4 = 0 vs 5     #14 = 2 vs 4     #24 = 4 vs 7
	#5 = 0 vs 6     #15 = 2 vs 5     #25 = 5 vs 6
	#6 = 0 vs 7     #16 = 2 vs 6     #26 = 5 vs 7
	#7 = 1 vs 2     #17 = 2 vs 7     #27 = 6 vs 7
	#8 = 1 vs 3     #18 = 3 vs 4
	#9 = 1 vs 4     #19 = 3 vs 5

	all_inds = np.array(((0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(2,4),(2,5),(2,6),(2,7),(3,4),(3,5),(3,6),(3,7),(4,5),(4,6),(4,7),(5,6),(5,7),(6,7)))
	
	#what to compare? if rewarding trials sig diff from nonrewarding and/or punishing, and if punishing tirals sig diff from nonpunishing and/or rewarding
	post_hoc_cue = all_regions_dict[region_key]['cue_comp_stats']['for_post_hoc']
	post_hoc_delivery = all_regions_dict[region_key]['delivery_comp_stats']['for_post_hoc']
	diffs_dict = {}
	for i in range(len(post_hoc_cue)):
		tukey_cue = post_hoc_cue[i]['tukey']
		reject_cue = tukey_cue.reject
		unit = post_hoc_cue[i]['unit_no']

		#rp_s, rp_f, r_only_s, r_only_f
		diffs = np.where(reject_cue)
		if diffs:
			for j in range(len(diffs)):
				pair_temp = all_inds[diffs[0][j]]
				temp_a = pair_temp[0] 
				temp_b = pair_temp[1] 

				print unit
				if (temp_a or temp_b) == r_only_s_cue_ind:
					print "r only s cue"
				if (temp_a or temp_b) == r_only_f_cue_ind:
					print 'r only f cue'
				if (temp_a or temp_b) == rp_s_cue_ind:
					print 'rp s cue'
				if (temp_a or temp_b) == rp_f_cue_ind:
					print 'rp f cue'
				if (temp_a or temp_b) == p_only_s_cue_ind:
					print 'p only s cue'
				if (temp_a or temp_b) == p_only_f_cue_ind:
					print 'p only f cue'
				if (temp_a or temp_b) == nrnp_s_cue_ind:
					print 'nrnp s cue'
				if (temp_a or temp_b) == nrnp_f_cue_ind:
					print 'nrnp f cue'

			

		diffs_dict['unit'] = unit

	#TODO repeat for delivery, diff for loops b/ diff lengths
	
	
			
			

	sig_compare[key_temp] = {'sig_cued_rewarding':sig_cued_rewarding,'sig_cued_punishing':sig_cued_punishing,'sig_reward_delivery':sig_reward_delivery,'sig_succ_noreward':sig_succ_noreward,'sig_unsucc_nopunishment':sig_unsucc_nopunishment,'sig_punishment_delivery':sig_punishment_delivery}






#save npy and xls files
print 'saving data'
np.save('data_%s' %(filename),all_regions_dict, sig_compare)

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
	worksheet.write(row,1,'delivery_dicts')
	for event_key,event_data in all_regions_dict[region_key]['delivery_dicts'].iteritems():
		if 'stats' in event_key:
			worksheet.write(row,2,event_key)
			worksheet.write(row,3,all_regions_dict[region_key]['delivery_dicts'][event_key]['perc_sig'])
			row +=1

row = 0
worksheet2 = workbook.add_worksheet()
for region_key,region_data in all_regions_dict.iteritems():
	worksheet2.write(row,0,region_key)
	worksheet2.write(row,1,'cue_comp_stats')
	worksheet2.write(row,2,all_regions_dict[region_key]['cue_comp_stats']['anova_sig_perc'])
	worksheet2.write(row,3,'delivery_comp_stats')
	worksheet2.write(row,4,all_regions_dict[region_key]['delivery_comp_stats']['anova_sig_perc'])
	row +=1

	
pyexcel.merge_all_to_a_book(glob.glob('*.csv'),'post_hoc_results_%s.xlsx' %(filename))

#TODO delete csvs
