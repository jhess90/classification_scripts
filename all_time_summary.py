#!/usr/bin/env python

#import packages
import scipy.io as sio
import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import xlsxwriter
import glob
import pandas as pd
import csv

#nhp_id = '0059'
nhp_id = '504'


if nhp_id == '0059':
    print nhp_id

    dict1 = np.load('times1.npy')[()]
    dict2 = np.load('times2.npy')[()]
    dict3 = np.load('times3.npy')[()]
    dict4 = np.load('times4.npy')[()]


    all_cue_reach = np.concatenate((dict1['cue_reach'],dict2['cue_reach'],dict3['cue_reach'],dict4['cue_reach']))
    all_cue_reach = all_cue_reach[all_cue_reach > 0]

    all_cue_grasp = np.concatenate((dict1['cue_grasp'],dict2['cue_grasp'],dict3['cue_grasp'],dict4['cue_grasp']))
    all_cue_grasp = all_cue_grasp[all_cue_grasp > 0] 

    all_cue_transport = np.concatenate((dict1['cue_transport'],dict2['cue_transport'],dict3['cue_transport'],dict4['cue_transport']))
    all_cue_transport = all_cue_transport[all_cue_transport > 0] 

    all_cue_fail = np.concatenate((dict1['cue_fail'],dict2['cue_fail'],dict3['cue_fail'],dict4['cue_fail']))
    all_cue_fail = all_cue_fail[all_cue_fail > 0] 

    all_result_next_cue = np.concatenate((dict1['result_next_cue'],dict2['result_next_cue'],dict3['result_next_cue'],dict4['result_next_cue']))
    all_result_next_cue = all_result_next_cue[all_result_next_cue > 0]

    times = [all_cue_reach,all_cue_grasp,all_cue_transport,all_cue_fail,all_result_next_cue]

    plt.figure()
    plt.boxplot(times)
    
    plt.hlines(1.0,0,6,colors='g',linestyle='dashed')
    plt.xticks([1,2,3,4,5],["Cue-Reach","Cue-Grasp","Cue-Transport","Cue-Fail","Result-Next Cue"])

    plt.savefig('0059_times')
    plt.clf()

    #
    plt.figure()
    times = [all_cue_reach,all_cue_grasp,all_cue_transport,all_cue_fail]

    plt.boxplot(times)
    
    plt.hlines(1.0,0,6,colors='g',linestyle='dashed')
    plt.xticks([1,2,3,4],["Cue-Reach","Cue-Grasp","Cue-Transport","Cue-Fail"])

    plt.savefig('0059_times_2')
    plt.clf()

    titles_print = ['','Mean','Std','Min','Max']
    cue_reach_print = ['Cue-Reach',np.mean(all_cue_reach),np.std(all_cue_reach),np.min(all_cue_reach),np.max(all_cue_reach)]
    cue_grasp_print = ['Cue-Grasp',np.mean(all_cue_grasp),np.std(all_cue_grasp),np.min(all_cue_grasp),np.max(all_cue_grasp)]
    cue_transport_print = ['Cue-Transport',np.mean(all_cue_transport),np.std(all_cue_transport),np.min(all_cue_transport),np.max(all_cue_transport)]
    cue_fail_print = ['Cue-Fail',np.mean(all_cue_fail),np.std(all_cue_fail),np.min(all_cue_fail),np.max(all_cue_fail)]
    result_next_cue_print = ['Result-Next Cue',np.mean(all_result_next_cue),np.std(all_result_next_cue),np.min(all_result_next_cue),np.max(all_result_next_cue)]

    
    with open('0059.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(titles_print)
        writer.writerow(cue_reach_print)
        writer.writerow(cue_grasp_print)
        writer.writerow(cue_transport_print)
        writer.writerow(cue_fail_print)
        writer.writerow(result_next_cue_print)



else:
    print nhp_id

    dict1 = np.load('times1.npy')[()]
    dict2 = np.load('times2.npy')[()]
    dict3 = np.load('times3.npy')[()]
    dict4 = np.load('times4.npy')[()]
    dict5 = np.load('times5.npy')[()]
    dict6 = np.load('times6.npy')[()]
    dict7 = np.load('times7.npy')[()]


    all_cue_reach = np.concatenate((dict1['cue_reach'],dict2['cue_reach'],dict3['cue_reach'],dict4['cue_reach'],dict5['cue_reach'],dict6['cue_reach'],dict7['cue_reach']))
    all_cue_reach = all_cue_reach[all_cue_reach > 0]

    all_cue_grasp = np.concatenate((dict1['cue_grasp'],dict2['cue_grasp'],dict3['cue_grasp'],dict4['cue_grasp'],dict5['cue_grasp'],dict6['cue_grasp'],dict7['cue_grasp']))
    all_cue_grasp = all_cue_grasp[all_cue_grasp > 0] 

    all_cue_transport = np.concatenate((dict1['cue_transport'],dict2['cue_transport'],dict3['cue_transport'],dict4['cue_transport'],dict5['cue_transport'],dict6['cue_transport'],dict7['cue_transport']))
    all_cue_transport = all_cue_transport[all_cue_transport > 0] 

    all_cue_fail = np.concatenate((dict1['cue_fail'],dict2['cue_fail'],dict3['cue_fail'],dict4['cue_fail'],dict5['cue_fail'],dict6['cue_fail'],dict7['cue_fail']))
    all_cue_fail = all_cue_fail[all_cue_fail > 0] 

    all_result_next_cue = np.concatenate((dict1['result_next_cue'],dict2['result_next_cue'],dict3['result_next_cue'],dict4['result_next_cue'],dict5['result_next_cue'],dict6['result_next_cue'],dict7['result_next_cue']))
    all_result_next_cue = all_result_next_cue[all_result_next_cue > 0]

    times = [all_cue_reach,all_cue_grasp,all_cue_transport,all_cue_fail,all_result_next_cue]

    plt.figure()
    plt.boxplot(times)
    
    plt.hlines(1.0,0,6,colors='g',linestyle='dashed')
    plt.xticks([1,2,3,4,5],["Cue-Reach","Cue-Grasp","Cue-Transport","Cue-Fail","Result-Next Cue"])

    plt.savefig('504_times')
    plt.clf()

    #
    plt.figure()
    times = [all_cue_reach,all_cue_grasp,all_cue_transport,all_cue_fail]

    plt.boxplot(times)
    
    plt.hlines(1.0,0,6,colors='g',linestyle='dashed')
    plt.xticks([1,2,3,4],["Cue-Reach","Cue-Grasp","Cue-Transport","Cue-Fail"])

    plt.savefig('504_times_2')
    plt.clf()

    titles_print = ['','Mean','Std','Min','Max']
    cue_reach_print = ['Cue-Reach',np.mean(all_cue_reach),np.std(all_cue_reach),np.min(all_cue_reach),np.max(all_cue_reach)]
    cue_grasp_print = ['Cue-Grasp',np.mean(all_cue_grasp),np.std(all_cue_grasp),np.min(all_cue_grasp),np.max(all_cue_grasp)]
    cue_transport_print = ['Cue-Transport',np.mean(all_cue_transport),np.std(all_cue_transport),np.min(all_cue_transport),np.max(all_cue_transport)]
    cue_fail_print = ['Cue-Fail',np.mean(all_cue_fail),np.std(all_cue_fail),np.min(all_cue_fail),np.max(all_cue_fail)]
    result_next_cue_print = ['Result-Next Cue',np.mean(all_result_next_cue),np.std(all_result_next_cue),np.min(all_result_next_cue),np.max(all_result_next_cue)]

    
    with open('504.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(titles_print)
        writer.writerow(cue_reach_print)
        writer.writerow(cue_grasp_print)
        writer.writerow(cue_transport_print)
        writer.writerow(cue_fail_print)
        writer.writerow(result_next_cue_print)

