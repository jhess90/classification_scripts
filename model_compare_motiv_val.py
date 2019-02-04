#!/usr/bin/env python

#import packages
import numpy as np
import pdb
#import matplotlib.colors as colors
#import matplotlib.colorbar as colorbar
#import sys
import xlsxwriter
import glob
#from scipy.optimize import curve_fit
#import matplotlib.pyplot as plt
#import statsmodels.api as sm
#import pandas as pd
#from matplotlib import cm
#import scipy.stats as stats
#from scipy import ndimage
#from math import isinf
#import os
#from scipy.stats.stats import pearsonr
#from mpl_toolkits.mplot3d import Axes3D
#import math
import os

####################

def check_sign(param1,param2):
    sign1 = np.sign(param1)
    sign2 = np.sign(param2)

    motiv_vector = sign1 == sign2

    length_sig = np.shape(motiv_vector)[0]

    if length_sig > 0:
        num_motiv = np.sum(motiv_vector)
        if num_motiv > 0:
            perc_motiv = num_motiv / float(length_sig) * 100
        else:
            perc_motiv = 0
        perc_val = 100 - perc_motiv
    
        #motiv, value, sig
        mvs = [perc_motiv,perc_val,length_sig]
    else:
        mvs = [0,0,0]
        
    vm_dict = {'motiv_vector':motiv_vector,'mvs':mvs}
    
    return(vm_dict)

####################

data = np.load('model_save.npy')[()]


for region_key,region_value in data.iteritems():

    #
    lin_dict = data[region_key]['models']['linear']

    params = lin_dict['aft_cue']['fit_params'][lin_dict['aft_cue']['p_val_total'] < 0.05]
    lin_motiv_ac = check_sign(params[:,0],params[:,1])
    params = lin_dict['bfr_res']['fit_params'][lin_dict['bfr_res']['p_val_total'] < 0.05]
    lin_motiv_br = check_sign(params[:,0],params[:,1])
    params = lin_dict['aft_res']['fit_params'][lin_dict['aft_res']['p_val_total'] < 0.05]
    lin_motiv_ar = check_sign(params[:,0],params[:,1])
    
    #motiv, val, num sig for ac, then br, then ar
    lin_mvs_all = lin_motiv_ac['mvs'] + lin_motiv_br['mvs'] + lin_motiv_ar['mvs']
    
    #
    div_nl_dict = data[region_key]['models']['div_nl']

    params = div_nl_dict['aft_cue']['fit_params'][div_nl_dict['aft_cue']['p_val_total'] < 0.05]
    div_nl_motiv_ac = check_sign(params[:,0],params[:,1])
    params = div_nl_dict['bfr_res']['fit_params'][div_nl_dict['bfr_res']['p_val_total'] < 0.05]
    div_nl_motiv_br = check_sign(params[:,0],params[:,1])
    params = div_nl_dict['aft_res']['fit_params'][div_nl_dict['aft_res']['p_val_total'] < 0.05]
    div_nl_motiv_ar = check_sign(params[:,0],params[:,1])
    
    div_nl_mvs_all = div_nl_motiv_ac['mvs'] + div_nl_motiv_br['mvs'] + div_nl_motiv_ar['mvs']

    #
    y_dict = data[region_key]['models']['div_nl_Y']

    params = y_dict['aft_cue']['fit_params'][y_dict['aft_cue']['p_val_total'] < 0.05]
    y_motiv_ac = check_sign(params[:,0],params[:,1])
    params = y_dict['bfr_res']['fit_params'][y_dict['bfr_res']['p_val_total'] < 0.05]
    y_motiv_br = check_sign(params[:,0],params[:,1])
    params = y_dict['aft_res']['fit_params'][y_dict['aft_res']['p_val_total'] < 0.05]
    y_motiv_ar = check_sign(params[:,0],params[:,1])

    y_mvs_all = y_motiv_ac['mvs'] + y_motiv_br['mvs'] + y_motiv_ar['mvs']

    #
    sep_add_dict = data[region_key]['models']['div_nl_separate_add']

    params = sep_add_dict['aft_cue']['fit_params'][sep_add_dict['aft_cue']['p_val_total'] < 0.05]
    sep_add_motiv_ac = check_sign(params[:,0],params[:,2])
    params = sep_add_dict['bfr_res']['fit_params'][sep_add_dict['bfr_res']['p_val_total'] < 0.05]
    sep_add_motiv_br = check_sign(params[:,0],params[:,2])
    params = sep_add_dict['aft_res']['fit_params'][sep_add_dict['aft_res']['p_val_total'] < 0.05]
    sep_add_motiv_ar = check_sign(params[:,0],params[:,2])

    sep_add_mvs_all = sep_add_motiv_ac['mvs'] + sep_add_motiv_br['mvs'] + sep_add_motiv_ar['mvs']

    #
    ars_dict = data[region_key]['models']['div_nl_separate_multiply']

    params = ars_dict['aft_cue']['fit_params'][ars_dict['aft_cue']['p_val_total'] < 0.05]
    ars_motiv_ac = check_sign(params[:,0],params[:,2])
    params = ars_dict['bfr_res']['fit_params'][ars_dict['bfr_res']['p_val_total'] < 0.05]
    ars_motiv_br = check_sign(params[:,0],params[:,2])
    params = ars_dict['aft_res']['fit_params'][ars_dict['aft_res']['p_val_total'] < 0.05]
    ars_motiv_ar = check_sign(params[:,0],params[:,2])
    
    ars_mvs_all = ars_motiv_ac['mvs'] + ars_motiv_br['mvs'] + ars_motiv_ar['mvs']

    data[region_key]['mvs_dict'] = {'lin':lin_mvs_all,'div_nl':div_nl_mvs_all,'y':y_mvs_all,'sep_add':sep_add_mvs_all,'ars':ars_mvs_all}
    

if os.path.isfile('vm_numbs.xlxs'):
    os.remove('vm_numbs.xlsx')
    
vm_workbook = xlsxwriter.Workbook('vm_numbs.xlsx',options={'nan_inf_to_errors':True})
worksheet = vm_workbook.add_worksheet('percs')

row_names = ['linear','div nl','Y','Sep Add','ARS']
sup_col_names = ['AC','','','BR','','','AR','','']
col_names = ['%M','%V','#Sig','%M','%V','#Sig','%M','%V','#Sig']

worksheet.write_row(0,1,sup_col_names)
worksheet.write(1,0,'PMd')
worksheet.write_row(1,1,col_names)
worksheet.write_column(2,0,row_names)
worksheet.write_row(2,1,data['PmD_dicts']['mvs_dict']['lin'])
worksheet.write_row(3,1,data['PmD_dicts']['mvs_dict']['div_nl'])
worksheet.write_row(4,1,data['PmD_dicts']['mvs_dict']['y'])
worksheet.write_row(5,1,data['PmD_dicts']['mvs_dict']['sep_add'])
worksheet.write_row(6,1,data['PmD_dicts']['mvs_dict']['ars'])

worksheet.write(7,0,'M1')
worksheet.write_column(8,0,row_names)
worksheet.write_row(8,1,data['M1_dicts']['mvs_dict']['lin'])
worksheet.write_row(9,1,data['M1_dicts']['mvs_dict']['div_nl'])
worksheet.write_row(10,1,data['M1_dicts']['mvs_dict']['y'])
worksheet.write_row(11,1,data['M1_dicts']['mvs_dict']['sep_add'])
worksheet.write_row(12,1,data['M1_dicts']['mvs_dict']['ars'])

worksheet.write(13,0,'S1')
worksheet.write_column(14,0,row_names)
worksheet.write_row(14,1,data['S1_dicts']['mvs_dict']['lin'])
worksheet.write_row(15,1,data['S1_dicts']['mvs_dict']['div_nl'])
worksheet.write_row(16,1,data['S1_dicts']['mvs_dict']['y'])
worksheet.write_row(17,1,data['S1_dicts']['mvs_dict']['sep_add'])
worksheet.write_row(18,1,data['S1_dicts']['mvs_dict']['ars'])





vm_workbook.close()
