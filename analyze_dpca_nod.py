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
#data['M1'] = np.load('dpca_results_binaryrp_nod_M1.npy')[()]
data['S1'] = np.load('dpca_results_binaryrp_nod_S1.npy')[()]
data['PmD'] = np.load('dpca_results_binaryrp_nod_PmD.npy')[()]
try:
        data['M1'] = np.load('dpca_results_binaryrp_nod_M1.npy')[()]
except:
        explained_var = np.load('dpca_results_binaryrp_nod_M1_explained_var.npy')[()]
        sig_analysis = np.load('dpca_results_binaryrp_nod_M1_sig_analysis.npy')[()]
        data['M1'] = {}
        data['M1']['explained_var'] = explained_var
        data['M1']['sig_analysis'] = sig_analysis

variance_workbook = xlsxwriter.Workbook('perc_variance.xlsx',options={'nan_inf_to_errors':True})
i=0
worksheet = variance_workbook.add_worksheet('variance')
#rownames = ['t','r','rt','rp','rd','p','pt','pd','d','dt','rpd','rpt','rdt','pdt','rpdt']
rownames = ['t','rt','pt','rpt']
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
        #worksheet.write_row(3,i+1,explained_var['r'])
        worksheet.write_row(3,i+1,explained_var['rt'])
        #worksheet.write_row(5,i+1,explained_var['rp'])
        #worksheet.write_row(6,i+1,explained_var['rd'])
        #worksheet.write_row(7,i+1,explained_var['p'])
        worksheet.write_row(4,i+1,explained_var['pt'])
        #worksheet.write_row(9,i+1,explained_var['pd'])
        #worksheet.write_row(10,i+1,explained_var['d'])
        #worksheet.write_row(5,i+1,explained_var['dt'])
        #worksheet.write_row(12,i+1,explained_var['rpd'])
        worksheet.write_row(5,i+1,explained_var['rpt'])
        #worksheet.write_row(7,i+1,explained_var['rdt'])
        #worksheet.write_row(8,i+1,explained_var['pdt'])
        #worksheet.write_row(9,i+1,explained_var['rpdt'])
	
        worksheet.write(2,i+16,sum(explained_var['t']))
        #worksheet.write(3,i+16,sum(explained_var['r']))
        worksheet.write(3,i+16,sum(explained_var['rt']))
        #worksheet.write(5,i+16,sum(explained_var['rp']))
        #worksheet.write(6,i+16,sum(explained_var['rd']))
        #worksheet.write(7,i+16,sum(explained_var['p']))
        worksheet.write(4,i+16,sum(explained_var['pt']))
        #worksheet.write(9,i+16,sum(explained_var['pd']))
        #worksheet.write(10,i+16,sum(explained_var['d']))
        #worksheet.write(5,i+16,sum(explained_var['dt']))
        #worksheet.write(12,i+16,sum(explained_var['rpd']))
        worksheet.write(5,i+16,sum(explained_var['rpt']))
        #worksheet.write(7,i+16,sum(explained_var['rdt']))
        #worksheet.write(8,i+16,sum(explained_var['pdt']))
        #worksheet.write(9,i+16,sum(explained_var['rpdt']))
        
        worksheet.write(6,i+15,'total')
        worksheet.write(6,i+16,total_var)
        
        i += 16

variance_workbook.close()


###plot significance analysis

my_ticks = ['-0.5','0','0.5','-0.5','0','0.5','1.0']
tot_bins = int(np.shape(data['M1']['sig_analysis'][0]['rt'])[1])
bfr_bins = int(tot_bins / 6)
aft_bins = bfr_bins * 2
my_ticks_num = np.arange(0,tot_bins*7/6,tot_bins/6)


for region_key,region_value in data.iteritems():
        sig_data = region_value['sig_analysis']
        sig_binary = sig_data[0]
        bins = np.shape(sig_data[0]['rt'])[1]
        time = np.arange(bins)

        #plot 1
        for comb_key,comb_val in sig_data[0].iteritems():
                ax = plt.gca()
                for component in range(4):
                        plt.subplot(2,2,component+1)
                        #plot shuffled
                        temp = np.asarray(sig_data[2][comb_key])
                        #for i in range(np.shape(sig_data[1][comb_key])[0]):
                        for i in range(np.shape(sig_data[2][comb_key])[0]):
                                plt.plot(time,temp[i,component,:],color='grey')
                        for i in range(bins):
                                if sig_binary[comb_key][component,i]:
                                        plt.plot(i,0.95,'ko')
                        #plot actual
                        plt.plot(time,sig_data[1][comb_key][component,:])
                        plt.ylim([0,1])

                        plt.axvline(x=bfr_bins,color='g',linestyle='--')
			plt.axvline(x=bfr_bins+aft_bins,color='k')
			plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
			plt.title('Component: %s' %(component+1),fontsize='small')
			plt.xticks(my_ticks_num,my_ticks)
                        plt.xlabel('Time (s)',fontsize='small')
                        plt.ylabel('Classification Accuracy',fontsize='small')

                plt.tight_layout(w_pad=0.1)
                plt.rcParams['xtick.labelsize'] = 8
                plt.rcParams['ytick.labelsize'] = 8
                plt.subplots_adjust(top=0.9)
                plt.suptitle('Region: %s, combination: %s' %(region_key,comb_key))
                plt.savefig('sigplt_%s_%s_1' %(region_key,comb_key))
                plt.clf()
        #plot 2
        for comb_key,comb_val in sig_data[0].iteritems():
                ax = plt.gca()
                for component in range(4):
                        plt.subplot(2,2,component+1)
                        #plot shuffled
                        temp = np.asarray(sig_data[2][comb_key])
                        #for i in range(np.shape(sig_data[1][comb_key])[0]):
                        for i in range(np.shape(sig_data[2][comb_key])[0]):
                                plt.plot(time,temp[i,component+4,:],color='grey')
                        for i in range(bins):
                                if sig_binary[comb_key][component+4,i]:
                                        plt.plot(i,0.95,'ko')
                        #plot actual
                        plt.plot(time,sig_data[1][comb_key][component+4,:])
                        plt.ylim([0,1])

                        plt.axvline(x=bfr_bins,color='g',linestyle='--')
			plt.axvline(x=bfr_bins+aft_bins,color='k')
			plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
			plt.title('Component: %s' %(component+5),fontsize='small')
			plt.xticks(my_ticks_num,my_ticks)
                        plt.xlabel('Time (s)',fontsize='small')
                        plt.ylabel('Classification Accuracy',fontsize='small')

                plt.tight_layout(w_pad=0.1)
                plt.rcParams['xtick.labelsize'] = 8
                plt.rcParams['ytick.labelsize'] = 8
                plt.subplots_adjust(top=0.9)
                plt.suptitle('Region: %s, combination: %s' %(region_key,comb_key))
                plt.savefig('sigplt_%s_%s_2' %(region_key,comb_key))
                plt.clf()

        #plot 3
        for comb_key,comb_val in sig_data[0].iteritems():
                ax = plt.gca()
                for component in range(4):
                        plt.subplot(2,2,component+1)
                        #plot shuffled
                        temp = np.asarray(sig_data[2][comb_key])
                        #for i in range(np.shape(sig_data[1][comb_key])[0]):
                        for i in range(np.shape(sig_data[2][comb_key])[0]):
                                plt.plot(time,temp[i,component+8,:],color='grey')
                        for i in range(bins):
                                if sig_binary[comb_key][component+8,i]:
                                        plt.plot(i,0.95,'ko')
                        #plot actual
                        plt.plot(time,sig_data[1][comb_key][component+8,:])
                        plt.ylim([0,1])

                        plt.axvline(x=bfr_bins,color='g',linestyle='--')
			plt.axvline(x=bfr_bins+aft_bins,color='k')
			plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
			plt.title('Component: %s' %(component+9),fontsize='small')
			plt.xticks(my_ticks_num,my_ticks)
                        plt.xlabel('Time (s)',fontsize='small')
                        plt.ylabel('Classification Accuracy',fontsize='small')

                plt.tight_layout(w_pad=0.1)
                plt.rcParams['xtick.labelsize'] = 8
                plt.rcParams['ytick.labelsize'] = 8
                plt.subplots_adjust(top=0.9)
                plt.suptitle('Region: %s, combination: %s' %(region_key,comb_key))
                plt.savefig('sigplt_%s_%s_3' %(region_key,comb_key))
                plt.clf()

        #plot 4
        for comb_key,comb_val in sig_data[0].iteritems():
                ax = plt.gca()
                for component in range(3):
                        plt.subplot(2,2,component+1)
                        #plot shuffled
                        temp = np.asarray(sig_data[2][comb_key])
                        #for i in range(np.shape(sig_data[1][comb_key])[0]):
                        for i in range(np.shape(sig_data[2][comb_key])[0]):
                                plt.plot(time,temp[i,component+12,:],color='grey')
                        for i in range(bins):
                                if sig_binary[comb_key][component+12,i]:
                                        plt.plot(i,0.95,'ko')
                        #plot actual
                        plt.plot(time,sig_data[1][comb_key][component+12,:])
                        plt.ylim([0,1])

                        plt.axvline(x=bfr_bins,color='g',linestyle='--')
			plt.axvline(x=bfr_bins+aft_bins,color='k')
			plt.axvline(x=2*bfr_bins+aft_bins,color='b',linestyle='--')
			plt.title('Component: %s' %(component+13),fontsize='small')
			plt.xticks(my_ticks_num,my_ticks)
                        plt.xlabel('Time (s)',fontsize='small')
                        plt.ylabel('Classification Accuracy',fontsize='small')

                plt.tight_layout(w_pad=0.1)
                plt.rcParams['xtick.labelsize'] = 8
                plt.rcParams['ytick.labelsize'] = 8
                plt.subplots_adjust(top=0.9)
                plt.suptitle('Region: %s, combination: %s' %(region_key,comb_key))
                plt.savefig('sigplt_%s_%s_4' %(region_key,comb_key))
                plt.clf()

