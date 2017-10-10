#!/usr/bin/env python

#import packages
import numpy as np
import pdb
import sys
import xlsxwriter
import glob
import matplotlib.pyplot as plt

#######
#sd_bool = False
#multirp_bool = False
#binaryrp_bool = True

#########
#if sd_bool:
#	data = np.load('dpca_results.npy')[()]
#elif multirp_bool:
#	data = np.load('dpca_results_multirp.npy')[()]
#elif binaryrp_bool:
#	data = np.load('dpca_results_binaryrp.npy')[()]

data={}
data['M1'] = np.load('dpca_results_binaryrp_M1.npy')[()]
data['S1'] = np.load('dpca_results_binaryrp_S1.npy')[()]
data['PmD'] = np.load('dpca_results_binaryrp_PmD.npy')[()]


variance_workbook = xlsxwriter.Workbook('perc_variance.xlsx',options={'nan_inf_to_errors':True})
i=0
worksheet = variance_workbook.add_worksheet('variance')
rownames = ['t','r','rt','rp','rd','p','pt','pd','d','dt','rpd','rpt','rdt','pdt','rpdt']
colnames = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','total']
worksheet.write_column(2,0,rownames)

for region_key,region_val in data.iteritems():
	if region_key == 'sort_dict':
		continue
	data[region_key]['dpca_analysis'] = {}
	
	#dpca_results = data[region_key]['dpca_results']
	dpca_results = data[region_key]
        total_var = 0

	for cond_key,cond_value in dpca_results['explained_var'].iteritems():
		total_var += np.sum(cond_value)
		print 'cond: %s, val: %s' %(cond_key,np.sum(cond_value))
		
	data[region_key]['dpca_analysis']['total_explained_variance'] = total_var
	print 'region %s, total explained variance= %s' %(region_key,total_var)

        explained_var = dpca_results['explained_var']

        worksheet.write(0,i+1,region_key)
        worksheet.write_row(1,i+1,colnames)
        worksheet.write_row(2,i+1,explained_var['t'])
        worksheet.write_row(3,i+1,explained_var['r'])
        worksheet.write_row(4,i+1,explained_var['rt'])
        worksheet.write_row(5,i+1,explained_var['rp'])
        worksheet.write_row(6,i+1,explained_var['rd'])
        worksheet.write_row(7,i+1,explained_var['p'])
        worksheet.write_row(8,i+1,explained_var['pt'])
        worksheet.write_row(9,i+1,explained_var['pd'])
        worksheet.write_row(10,i+1,explained_var['d'])
        worksheet.write_row(11,i+1,explained_var['dt'])
        worksheet.write_row(12,i+1,explained_var['rpd'])
        worksheet.write_row(13,i+1,explained_var['rpt'])
        worksheet.write_row(14,i+1,explained_var['rdt'])
        worksheet.write_row(15,i+1,explained_var['pdt'])
        worksheet.write_row(16,i+1,explained_var['rpdt'])
	
        worksheet.write(2,i+16,sum(explained_var['t']))
        worksheet.write(3,i+16,sum(explained_var['r']))
        worksheet.write(4,i+16,sum(explained_var['rt']))
        worksheet.write(5,i+16,sum(explained_var['rp']))
        worksheet.write(6,i+16,sum(explained_var['rd']))
        worksheet.write(7,i+16,sum(explained_var['p']))
        worksheet.write(8,i+16,sum(explained_var['pt']))
        worksheet.write(9,i+16,sum(explained_var['pd']))
        worksheet.write(10,i+16,sum(explained_var['d']))
        worksheet.write(11,i+16,sum(explained_var['dt']))
        worksheet.write(12,i+16,sum(explained_var['rpd']))
        worksheet.write(13,i+16,sum(explained_var['rpt']))
        worksheet.write(14,i+16,sum(explained_var['rdt']))
        worksheet.write(15,i+16,sum(explained_var['pdt']))
        worksheet.write(16,i+16,sum(explained_var['rpdt']))
        
        worksheet.write(17,i+15,'total')
        worksheet.write(17,i+16,total_var)
        
        i += 16

variance_workbook.close()
