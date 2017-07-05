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

#######################
#params to set ########
#######################

plot_bool = True
time_boundry={'-0.5-1.0':[-0.5,1.0]}
bin_size = 50 #in ms
nlize_to_b4_trial = True

#### laptop ##############
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-12-48-52.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-13-02-45.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-12-48-19.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-13-10-44.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-14-23.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-33-52.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-25-20.mat'
#filename = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-46-25.mat'

### beaver ##############
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-12-48-52.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-13-02-45.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-12-48-19.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-13-10-44.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-14-23.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-33-52.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-25-20.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-46-25.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_0059/Extracted_0059_2016-05-25-15-41-44.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_0059/Extracted_0059_2016-05-25-15-58-17.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_504/Extracted_504_2016-05-25-14-46-46.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_504/Extracted_504_2016-05-25-15-02-58.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_0059/Extracted_0059_2016-05-26-12-17-53.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_0059/Extracted_0059_2016-05-26-12-38-56.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_504/Extracted_504_2016-05-26-11-25-03.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_504/Extracted_504_2016-05-26-11-45-52.mat'

### beaver ##############
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-12-48-52.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160118_0059/Extracted_0059_2016-01-18-13-02-45.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-12-48-19.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20150929_504/Extracted_504_2015-09-29-13-10-44.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-14-23.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151001_504/Extracted_504_2015-10-01-15-33-52.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-25-20.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-46-25.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_0059/Extracted_0059_2016-05-25-15-41-44.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_0059/Extracted_0059_2016-05-25-15-58-17.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_504/Extracted_504_2016-05-25-14-46-46.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160525_504/Extracted_504_2016-05-25-15-02-58.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_0059/Extracted_0059_2016-05-26-12-17-53.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_0059/Extracted_0059_2016-05-26-12-38-56.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_504/Extracted_504_2016-05-26-11-25-03.mat'
#filename = '/home/jack/Dropbox/single_rp_files/extracted/20160526_504/Extracted_504_2016-05-26-11-45-52.mat'
#filename = '/home/jack/Dropbox/mult_rp_files/workspace/20160226_0059/block3/Extracted_0059_2016-02-26-16-28-27.mat'
#filename = '/home/jack/Dropbox/mult_rp_files/workspace/20160226_0059/block4/Extracted_0059_2016-02-26-16-38-13.mat'
#filename = '/home/jack/Dropbox/mult_rp_files/workspace/20160111_504/block1/Extracted_504_2016-01-11-13-56-44.mat'
#filename = '/home/jack/Dropbox/mult_rp_files/workspace/20160111_504/block2/Extracted_504_2016-01-11-14-10-01.mat'
#filename = '/Users/johnhessburg/dropbox/mult_rp_files/workspace/20160226_0059/block3/Extracted_0059_2016-02-26-16-28-27.mat'
#filename = '/Users/johnhessburg/dropbox/mult_rp_files/workspace/test/Extracted_504_2016-01-11-13-56-44.mat'
#filename = '/Users/johnhessburg/dropbox/mult_rp_files/workspace/test/Extracted_0059_2017-02-01-13-48-35.mat'
#filename = '/Users/johnhessburg/dropbox/mult_rp_files/workspace/20170209_0059/block1/Extracted_0059_2017-02-09-12-52-17.mat'
#filename = '/Users/johnhessburg/dropbox/mult_rp_files/workspace/20170209_0059/block3/Extracted_0059_2017-02-09-13-46-37.mat'
#filename = '/Users/johnhessburg/dropbox/mult_rp_files/workspace/20170214_504/Extracted_504_2017-02-14-12-09-21.mat'
#filename = '/Users/johnhessburg/dropbox/mult_rp_files/workspace/20170214_504/Extracted_504_2017-02-14-12-35-41.mat'
#filename = '/Users/johnhessburg/dropbox/mult_rp_files/workspace/20170214_504/Extracted_504_2017-02-14-13-01-34.mat'

###### beaver ########
#filename = '/home/jack/workspace/mult_rp/504_test/Extracted_504_2017-02-14-12-09-21.mat'
#filename = '/home/jack/workspace/mult_rp/504_test/Extracted_504_2017-02-14-12-35-41.mat'
#filename = '/home/jack/workspace/mult_rp/504_test/Extracted_504_2017-02-14-13-01-34.mat'

#filename = '/home/jack/workspace/mult_rp/0059/0208/Extracted_0059_2017-02-08-11-43-22.mat'
#filename = '/home/jack/workspace/mult_rp/0059/0208/Extracted_0059_2017-02-08-12-09-22.mat'
#filename = '/home/jack/workspace/mult_rp/0059/0209/Extracted_0059_2017-02-09-12-52-17.mat'
#filename = '/home/jack/workspace/mult_rp/0059/0209/Extracted_0059_2017-02-09-13-46-37.mat'
#filename = '/home/jack/workspace/mult_rp/504/0208/Extracted_504_2017-02-08-10-36-11.mat'
#filename = '/home/jack/workspace/mult_rp/504/0208/Extracted_504_2017-02-08-11-02-03.mat'
#filename = '/home/jack/workspace/mult_rp/504/0209/Extracted_504_2017-02-09-11-50-03.mat'
#filename = '/home/jack/workspace/mult_rp/504/0209/Extracted_504_2017-02-09-12-15-57.mat' #ERROR loading file maybe?!?
##filename = '/home/jack/workspace/mult_rp/504/0214/Extracted_504_2017-02-14-12-09-21.mat'
#filename = '/home/jack/workspace/mult_rp/504/0214/Extracted_504_2017-02-14-12-35-41.mat'
#filename = '/home/jack/workspace/mult_rp/504/0214/Extracted_504_2017-02-14-13-01-34.mat'

#filename = '/home/jack/workspace/mult_rp/504_test/Extracted_504_2017-02-08-10-36-11.mat'

#filename = '/home/jack/Dropbox/to_sort/20170208_0059/Extracted_0059_2017-02-08-11-43-22.mat'
#filename = '/home/jack/Dropbox/to_sort/20170208_0059/Extracted_0059_2017-02-08-12-09-22.mat'
#filename = '/home/jack/Dropbox/to_sort/20170209_0059/Extracted_0059_2017-02-09-12-52-17.mat'
#filename = '/home/jack/Dropbox/to_sort/20170209_0059/Extracted_0059_2017-02-09-13-46-37.mat'
filename = '/home/jack/Dropbox/to_sort/20170208_504/Extracted_504_2017-02-08-10-36-11.mat'
#filename = '/home/jack/Dropbox/to_sort/20170208_504/Extracted_504_2017-02-08-11-02-03.mat'
#filename = '/home/jack/Dropbox/to_sort/20170209_504/Extracted_504_2017-02-09-11-50-03.mat'
#filename = '/home/jack/Dropbox/to_sort/20170209_504/Extracted_504_2017-02-09-12-15-57.mat'
#filename = '/home/jack/Dropbox/to_sort/20170214_504/Extracted_504_2017-02-14-12-09-21.mat'
#filename = '/home/jack/Dropbox/to_sort/20170214_504/Extracted_504_2017-02-14-12-35-41.mat'
#filename = '/home/jack/Dropbox/to_sort/20170214_504/Extracted_504_2017-02-14-13-01-34.mat'


######################
print filename
a = sio.loadmat(filename);

#Pull reward timestamps. Separate timespamp file must exist in folder containing neural data, with same name+_timestamps
print filename[:-4]+"_timestamps"+filename[-4:]
Timestamps = sio.loadmat(filename[:-4]+"_timestamps"+filename[-4:]);

#catch trials- sometimes empty, so that's why there's the rcatch/pcatch bools
#reward cue, successfully completed, but no reward delivered
r_s_catch_cue = Timestamps['r_s_catch_cue_ts'][0];
r_s_catch_nextreset = Timestamps['r_s_catch_nextreset'][0];
if not r_s_catch_cue.size:
	print ('No r_s catch trials')
	rcatch_bool = False
else:
	print ('r_s catch trials: %s' %(len(r_s_catch_cue)))
	rcatch_bool = True

#punishment cue, successfully completed, but no reward delivered
p_f_catch_cue = Timestamps['p_f_catch_cue_ts'][0];
p_f_catch_nextreset = Timestamps['p_f_catch_nextreset'][0];
if not p_f_catch_cue.size:
	print ('No p_f catch trials')
	pcatch_bool = False
else:
	print ('p_f catch trials: %s' %(len(p_f_catch_cue)))
	pcatch_bool = True


######################################################################
### NEW ##############################################################
######################################################################
#maybe simpler way to do this
#TODO remove earlier stuff that's now redundant
#TODO make sure catch trials accounted for

trial_breakdown = Timestamps['trial_breakdown']

condensed = np.zeros((np.shape(trial_breakdown)[0],6))

#col 0 = disp_rp, 1 = succ scene, 2 = failure scene, 3 = rnum, 4 = pnum, 5 = catch_num

condensed[:,0] = trial_breakdown[:,1]
condensed[:,1] = trial_breakdown[:,2]
condensed[:,2] = trial_breakdown[:,3]
condensed[:,3] = trial_breakdown[:,5]
condensed[:,4] = trial_breakdown[:,7]
condensed[:,5] = trial_breakdown[:,10]

if condensed[-1,1] == condensed[-1,2] == 0:
	new_condensed = condensed[0:-1,:]

        condensed = new_condensed

condensed = condensed[condensed[:,0] != 0]

#both_bool_succ = [x and y for x,y in zip(condensed[:,1] !=0, condensed[:,5] == 0)]
#both_bool_fail = [x and y for x,y in zip(condensed[:,2] !=0, condensed[:,5] == 0)]
#catch_succ_bool = [x and y for x,y in zip(condensed[:,1] !=0, condensed[:,5] ==1)]
#catch_fail_bool = [x and y for x,y in zip(condensed[:,2] !=0, condensed[:,5] ==2)]

#all_succ = []
#all_fail = []
#catch_succ = []
#catch_fail = []

#out_of_bounds_bool = False
#for i in range(np.shape(condensed)[0]):
#        if both_bool_succ[i]:
#                all_succ.append(condensed[i,:])
#        if both_bool_fail[i]:
#                all_fail.append(condensed[i,:])
#        if catch_succ_bool[i]:
#                catch_succ.append(condensed[i,:])
#        if catch_fail_bool[i]:
#                catch_fail.append(condensed[i,:])

        #if not out_of_bounds_bool and condensed[i,0]*20.0 > 


#all_succ = np.asarray(all_succ)
#all_fail = np.asarray(all_fail)
#catch_succ = np.asarray(catch_succ)
#catch_fail = np.asarray(catch_fail)

##############################################################################
	
#Pull neural spikes
neural_data=a['neural_data']

#print neural_data
Spikes = a['neural_data']['spikeTimes'];

#Break spikes into M1 S1 pmd 
M1_spikes = Spikes[0,0][0,1];
PmD_spikes = Spikes[0,0][0,0];
S1_spikes = Spikes[0,0][0,2];
#Find first channel count for pmv on map1
#PmV is requires extra processing to generate as the data is the last 32 channels of each MAP system
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

#generate time intervals of interest for PSTH generation
#time_boundry={'-0.5-0.0':[-0.5,0.0],'0-0.5':[0.0,0.5], '0.5-1.0':[0.5,1.0], '1.0-1.5':[1.0,1.5], '1.5-2.0':[1.5,2.0],'all':[-0.5,2.0]}
#time_boundry={'-0.5-0.0':[-0.5,0.0],'0-0.5':[0.0,0.5],'0.5-1.0':[0.5,1.0],'all_incpre':[-0.5,2.0]}
#time_boundry={'-0.5-0.0':[-0.5,0.0],'0-0.5':[0.0,0.5],'0.5-1.0':[0.5,1.0],'all':[0.0,2.0]}

time_before = -0.5 #make sure if change to pos time before it still works
time_after = 1.0


no_bins = int((time_after - time_before) * 1000 / bin_size)

print 'bin size = %s' %(bin_size)

param_dict = {'time_before':time_before,'time_after':time_after,'time_boundry':time_boundry,'bin_size':bin_size,'no_bins':no_bins}

gripforce_all = np.asarray(Timestamps['gripforce'])

min_hist_size = 1000000

def normalize_data(spike_data):
	hist = []
	nl_spike_data = []
	hist_all = []
	nl_dict = {}
	min_bin = np.zeros(len(spike_data))
	max_bin = np.zeros(len(spike_data))
	global min_hist_size

	for i in range(len(spike_data)):
                                
                total_bin_range = np.arange(0,int(np.ceil(spike_data[i].max())),bin_size/1000.0)
                hist = np.histogram(spike_data[i], bins=total_bin_range,range=(0,int(np.ceil(spike_data[i].max()))),normed=False, density=False)
                
                if len(hist[0]) < min_hist_size:
                        min_hist_size = len(hist[0])
                

                min_bin[i] = min(hist[0])
                max_bin[i] = max(hist[0])
		
                num = np.subtract(hist[0],min_bin[i])
                denom = max_bin[i]-min_bin[i]
		
                nl_temp = np.true_divide(num,denom)
                nl_spike_data.append(nl_temp)
                hist_all.append(hist)

                if np.max(nl_temp) > 1.0:
                        print 'error normalizing unit %s' %(i)
 
	nl_dict['hist_all'] = hist_all
	nl_dict['nl_spike_data'] = nl_spike_data
	nl_dict['min_bin'] = min_bin
	nl_dict['max_bin'] = max_bin
	
	
	return(nl_dict)

def make_nl_raster_new(ts_value,ts_key,nl_dict,data_key,unit_num,spikes):
        hist_all = nl_dict['hist_all'][unit_num][0]
        
        #print len(hist_all)
        
        nlized_hist_data = nl_dict['nl_spike_data'][unit_num]
        bin_size_sec = bin_size / 1000.0
        no_bins_before = int(time_before / bin_size_sec) #neg, keep neg
        no_bins_after = int(time_after / bin_size_sec)

        event_hist = []
        event_hist_nl = []

        if 'result' in ts_key and 'all' in ts_key and not 'catch' in ts_key:
                result_bool = True
                prefix = ts_key[0:1]
                cue_key = prefix + 'all_cue'
                cue_ts = ts_dict[cue_key]
        elif 'result' in ts_key and 'all' in ts_key and 'catch' in ts_key:
                result_bool = True
                prefix = ts_key[0:1]
                cue_key = prefix + '_all_catch_cue'
                cue_ts = ts_dict[cue_key]
        elif 'result' in ts_key and 'catch' in ts_key:
                result_bool = True
                prefix = ts_key[0:3]
                cue_key= prefix + 'catch_cue'
                cue_ts = ts_dict[cue_key]
        elif 'result' in ts_key:
                prefix = ts_key[0:8]
                cue_key = prefix + 'cue'
                result_bool = True
                cue_ts = ts_dict[cue_key]
        else:
                result_bool = False


        for i in range(len(ts_value)):
                #only need to do it once, arbitrarily set it for the first unit
                if unit_num == 0:
                        gf_event = np.zeros((len(ts_value),300))
                        gf_start_ind = np.argmin(np.abs(gripforce_all[:,0] - ts_value[i]))
                        #sampling rate looks like 200 Hz
                        if gf_start_ind - 100 > gf_event[i,0]:
                                gf_event[i,:] = gripforce_all[gf_start_ind - 100 : gf_start_ind + 200, 1]
                        else:
                                #for some reason it doens't look like gripforce logged immediately, so might be after first trial or two. This accounts for that
                                print 'error with gf, event before gf started recording'
                                gf_event[i,:] = np.zeros((300))

                closest_start_time = round(ts_value[i] /bin_size_sec) * bin_size_sec
                closest_start_time = np.around(closest_start_time,decimals=2)
                start_bin = int(closest_start_time / bin_size_sec)

                if result_bool and not i > len(cue_ts):
                        closest_cue_start_time = round(cue_ts[i] / bin_size_sec) * bin_size_sec
                        closest_cue_start_time = np.around(closest_cue_start_time,decimals=2)
                        cue_start_bin = int(closest_cue_start_time / bin_size_sec)
                
                if not start_bin + no_bins_after > len(hist_all):
                        hist_temp = hist_all[start_bin + no_bins_before:start_bin + no_bins_after]
                        if i == 0:
                                binned = hist_temp
                        else:
                                binned = np.append(binned,hist_temp,axis=0)
        
                        
                        if not result_bool:
                                baseline = hist_all[start_bin - (1.0/bin_size_sec) :start_bin]
                        else:
                                baseline = hist_all[cue_start_bin - (1.0/bin_size_sec) : cue_start_bin]
                    
                        baseline_max = np.max(baseline)
                        baseline_min = np.min(baseline)
                        
                        if (hist_temp ==0).all() and (baseline == 0).all():
                                nl_temp = np.zeros((1,-1*no_bins_before + no_bins_after))
                 
                        elif (hist_temp !=0).any() and (baseline == 0).all():
                                nl_temp = hist_temp
                        else:

                                num = np.subtract(hist_temp,baseline_min)
                                denom = baseline_max - baseline_min
                                nl_temp = np.true_divide(num,denom)

                        if len(baseline) == 0:
                                #usually if too close to beginning of block (ie w/in 1 sec)
                                pdb.set_trace()
                                nl_temp = np.zeros((1,-1*no_bins_before + no_bins_after))
                        nl_temp = np.reshape(nl_temp,[1,-1*no_bins_before + no_bins_after])
                        if i == 0:
                                event_hist_nl = nl_temp
                        else:
                                event_hist_nl = np.append(event_hist_nl,nl_temp,axis=0)
                elif len(ts_value) == 1:
                        print 'ts too late, start %s, length %s' %(start_bin,len(hist_all))
                        #pdb.set_trace()
                        return
                else:
                        print 'start bin %s, length %s' %(start_bin,len(hist_all))

        binned_nl = np.asarray(event_hist_nl)

        try:
                firing_rate = np.zeros(binned.shape)
        except:
                pdb.set_trace()
	for i in range(len(binned)):
 		firing_rate[i] = binned[i] / float(bin_size) * 1000 #because bin_size in ms, want to         
        if unit_num == 0:
                gf_avg = np.mean(gf_event,axis=0)
                gf_std = np.std(gf_event,axis=0)

        if np.isnan(binned_nl).any():
                pdb.set_trace()
        if np.isinf(binned_nl).any():
                pdb.set_trace()

        avg_firing_rate = np.mean(firing_rate,axis=0)
        std_dev_firing_rate = np.std(firing_rate,axis=0)
        
        avg_nl = np.mean(binned_nl,axis=0)
        std_dev_nl = np.std(binned_nl,axis=0)

        if unit_num == 0:
            return_dict={'avg_firing_rate':avg_firing_rate,'std_dev_firing_rate':std_dev_firing_rate,'binned':binned,'firing_rate':firing_rate,'binned_nl':binned_nl,'avg_nl':avg_nl,'std_dev_nl':std_dev_nl,'gf_event':gf_event,'gf_avg':gf_avg,'gf_std':gf_std}
        else:
            return_dict={'avg_firing_rate':avg_firing_rate,'std_dev_firing_rate':std_dev_firing_rate,'binned':binned,'firing_rate':firing_rate,'binned_nl':binned_nl,'avg_nl':avg_nl,'std_dev_nl':std_dev_nl}            

	return(return_dict)
                
def make_nl_raster_old(ts_value,ts_key,nl_dict,data_key, unit_num,spikes):
        hist_all = nl_dict['hist_all'][unit_num][0]
        nlized_hist_data = nl_dict['nl_spike_data'][unit_num]
        bin_size_sec = bin_size / 1000.0
        no_bins_before = int(time_before / bin_size_sec) #neg, keep neg
        no_bins_after = int(time_after / bin_size_sec)

        event_hist = []
        event_hist_nl = []

        if unit_num == 0:
            gf_event = np.zeros((len(ts_value),300))
        
        #TODO figure out why some ts_value = 0
        #ts_value = ts_value[ts_value != 0]

        for i in range(len(ts_value)):
                try:
                        closest_start_time = round(ts_value[i] /bin_size_sec) * bin_size_sec
                        closest_start_time = np.around(closest_start_time,decimals=2)
                        start_bin = int(closest_start_time / bin_size_sec)
                
                        if not start_bin + no_bins_after > len(nlized_hist_data):
                                hist_nl_temp = nlized_hist_data[start_bin + no_bins_before:start_bin + no_bins_after]
                                if len(hist_nl_temp) == no_bins_before*-1 + no_bins_after:
                                        event_hist_nl.append(hist_nl_temp)
                                        event_hist.append(hist_all[start_bin + no_bins_before : start_bin + no_bins_after])

                        if unit_num == 0:
                            gf_start_ind = np.argmin(np.abs(gripforce_all[:,0] - ts_value[i]))
                            #sampling rate looks like 200 Hz
                            gf_event[i,:] = gripforce_all[gf_start_ind - 100 : gf_start_ind + 200, 1]

                except:
                        continue

        if unit_num == 0:
            gf_avg = np.mean(gf_event,axis=0)
            gf_std = np.std(gf_event,axis=0)

        binned_nl = np.asarray(event_hist_nl)
        binned = np.asarray(event_hist)
        
        if nlize_to_b4_trial:
                if 'result' in ts_key and 'all' in ts_key and not 'catch' in ts_key:
                        #pdb.set_trace()
                        result_bool = True
                        prefix = ts_key[0:1]
                        cue_key = prefix + 'all_cue'
                        cue_ts = ts_dict[cue_key]
                elif 'result' in ts_key and 'all' in ts_key and 'catch' in ts_key:
                        result_bool = True
                        prefix = ts_key[0:1]
                        cue_key = prefix + '_all_catch_cue'
                        cue_ts = ts_dict[cue_key]
                elif 'result' in ts_key and 'catch' in ts_key:
                        result_bool = True
                        prefix = ts_key[0:3]
                        cue_key= prefix + 'catch_cue'
                        cue_ts = ts_dict[cue_key]
                elif 'result' in ts_key:
                        prefix = ts_key[0:8]
                        #TODO bug: fixed below commented out 'result' to 'cue
                        #cue_key = prefix + 'result'
                        cue_key = prefix + 'cue'
                        result_bool = True
                        cue_ts = ts_dict[cue_key]
                else:
                        result_bool = False

                if result_bool and len(ts_value) != len(cue_ts):
                        #pdb.set_trace()
                        print 'error: %s and %s mismatch in length. Skipping for now.' %(ts_key, cue_key)
                        
                        return_dict={'avg_firing_rate':0,'std_dev_firing_rate':0,'binned':0,'firing_rate':0,'binned_nl':0,'avg_nl':0,'std_dev_nl':0,'gf_event':0,'gf_avg':0,'gf_std':0}
	
                        return(return_dict)
        

                        
                for i in range(len(ts_value)):
                        if ts_value[i] == 0:
                                print 'error with a ts value %s' %(i)
                                continue

                        closest_start_time = round(ts_value[i] /bin_size_sec) * bin_size_sec
                        closest_start_time = np.around(closest_start_time,decimals=2)
                        start_bin = int(closest_start_time / bin_size_sec)

                        try:
                                if result_bool and not i > len(cue_ts):
                                        closest_cue_start_time = round(cue_ts[i] /bin_size_sec) * bin_size_sec
                                        closest_cue_start_time = np.around(closest_cue_start_time,decimals=2)
                                        cue_start_bin = int(closest_cue_start_time / bin_size_sec)
                                elif result_bool and i > len(cue_ts):
                                        pdb.set_trace()
                        except:
                                pdb.set_trace()

                        #pdb.set_trace()
                        if not start_bin + no_bins_after > len(hist_all):
                                hist_temp = hist_all[start_bin + no_bins_before:start_bin + no_bins_after]
                                
                                if not result_bool:
                                        baseline = hist_all[start_bin - (1.0/bin_size_sec) :start_bin]
                                else:
                                        baseline = hist_all[cue_start_bin - (1.0/bin_size_sec) : cue_start_bin]
                    
                                try:
                                        baseline_max = np.max(baseline)
                                        baseline_min = np.min(baseline)

                                        num = np.subtract(hist_temp,baseline_min)
                                        denom = baseline_max - baseline_min
                                        nl_temp = np.true_divide(num,denom)
                                except:
                                        if len(baseline) == 0:
                                                #usually if too close to beginning of block (ie w/in 1 sec)
                                                nl_temp = np.zeros((1,-1*no_bins_before + no_bins_after))
                                        else:
                                                pdb.set_trace()

                                event_hist_nl.append(nl_temp)
                binned_nl = np.asarray(event_hist_nl)

        try:
                if np.amax(binned_nl) > 1.0:
                        test=[]
                        #print 'error: binned nl > 1.0 for unit %s, %s, %s' %(unit_num, ts_key, data_key)
        except:
                binned_new = np.zeros((binned_nl.shape[0],-1*no_bins_before+no_bins_after))
                for j in range(binned_nl.shape[0]):
                        binned_new[j] = binned_nl[j]

                binned_nl = binned_new
                try: 
                        if np.amax(binned_nl) > 1.0: 
                                print 'error:binned nl > 1.0 for unit %s, %s, %s' %(unit_num, ts_key, data_key)
                except:
                        binned_nl = np.zeros((binned_new))
                        print 'double exception with unit %s, %s, %s' %(unit_num, ts_key, data_key)

        ax = plt.gca()
        plt.subplot(2,2,1)

        dummy = []
	for i in range(len(ts_value)):
		dummy1=[]
		b = ts_value[i]
		for j in range(len(spikes)):
			a = spikes[j]
			c = a[np.where(np.logical_and(a >= b+time_before, a<= b+time_after))]
			dummy1.append(c)
		dummy.append(dummy1)
	dummy = np.asarray(dummy)

        for i in range(dummy.shape[0]):
                adjusted_times = dummy[i][unit_num] - ts_value[i]
                plt.vlines(adjusted_times, i + .5, i + 1.5)
                plt.ylim(.5, dummy.shape[0] + .5)

        firing_rate = np.zeros(binned.shape)
	for i in range(len(binned)):
 		firing_rate[i] = binned[i] / float(bin_size) * 1000 #because bin_size in ms, want to get Hz

	#avg_firing_rate = np.mean(firing_rate,axis=0)
	#std_dev_firing_rate = np.std(firing_rate,axis=0)
  	try:

                avg_firing_rate = np.mean(firing_rate,axis=0)
                std_dev_firing_rate = np.std(firing_rate,axis=0)

                avg_nl = np.mean(binned_nl,axis=0)
                std_dev_nl = np.std(binned_nl,axis=0)
                
                if np.isnan(avg_nl).any() or np.isnan(std_dev_nl).any():
                        print 'isnan'
                        pdb.set_trace()
                
                if np.isinf(avg_nl).any() or np.isinf(std_dev_nl).any():
                        print 'isinf'
                        pdb.set_trace()

        except:
                pdb.set_trace()
                #'avg = 0'
                #avg_nl = np.zeros((no_bins))
                #std_dev_nl = np.zeros((no_bins))
                #avg_firing_rate = np.zeros((no_bins))
                #std_dev_firing_rate = np.zeros((no_bins))
                
	plt.axvline(0.0,color='r')
	plt.xlim(time_before,time_after)
	
	plt.title('Raster plot')
	plt.xlabel('Time from event')
	plt.ylabel('Trial')

	#Average firing rate
	plt.subplot(2,2,2)
	plt.title('average firing rate')
	plt.xlabel('Time from event')
	plt.ylabel('Firing rate (Hz)')
	
	ymax = avg_firing_rate + std_dev_firing_rate
	ymin = avg_firing_rate - std_dev_firing_rate
        
        plt.errorbar(np.linspace(time_before,time_after,no_bins,endpoint=True),avg_firing_rate,yerr=std_dev_firing_rate)
        plt.ylim(ymin=0)
        plt.xlim(time_before,time_after)
        plt.axvline(0.0,color='r')

        #normalized firing rate
        plt.subplot(2,2,3)

        plt.errorbar(np.linspace(time_before,time_after,no_bins,endpoint=True),avg_nl,yerr=std_dev_nl)
        plt.ylim(0,1.0)
	plt.xlim(time_before,time_after)
	plt.axvline(0.0,color='r')
	
	plt.title('average normalized firing rate')
	plt.xlabel('Time from event')
	plt.ylabel('Normalized firing rate')

	plt.suptitle('%s, %s, unit: %s' %(ts_key,data_key,str(unit_num).zfill(2)), fontsize = 20)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	
	plt.savefig('raster_%s_%s_unit%s' %(ts_key, data_key, str(unit_num).zfill(2)))
	plt.clf()

        if unit_num == 0:
            return_dict={'avg_firing_rate':avg_firing_rate,'std_dev_firing_rate':std_dev_firing_rate,'binned':binned,'firing_rate':firing_rate,'binned_nl':binned_nl,'avg_nl':avg_nl,'std_dev_nl':std_dev_nl,'gf_event':gf_event,'gf_avg':gf_avg,'gf_std':gf_std}
        else:
            return_dict={'avg_firing_rate':avg_firing_rate,'std_dev_firing_rate':std_dev_firing_rate,'binned':binned,'firing_rate':firing_rate,'binned_nl':binned_nl,'avg_nl':avg_nl,'std_dev_nl':std_dev_nl}            

	return(return_dict)
             

def make_nl_raster(data,spikes,nl,key,key2,unit_no,time_before,time_after,min_bin,max_bin):

	dummy = []
	for i in range(len(data)):
		dummy1=[]
		b = data[i]
		for j in range(len(spikes)):
			a = spikes[j]
			c = a[np.where(np.logical_and(a >= b+time_before, a<= b+time_after))]
			dummy1.append(c)
		dummy.append(dummy1)
	dummy = np.asarray(dummy)
	ax = plt.gca()
	adjusted_times = []
	binned = []
	binned_nl = []
	plt.subplot(2,2,1)

	for event_num in range(dummy.shape[0]):
                #print 'unit %s, %s, %s, event %s' %(unit_no,key,key2,str(event_num))
                adjusted_times = dummy[event_num][unit_no] - data[event_num]
		plt.vlines(adjusted_times, event_num+.5, event_num+1.5)
		plt.ylim(.5, dummy.shape[0] + .5)

		binned_temp = np.histogram(adjusted_times,bins=no_bins,normed=False,density=False)
		binned.append(np.nan_to_num(binned_temp[0]))

		if event_num >= len(min_bin):
			print 'potential indexing error: breaking, event_num = %s, key = %s, key2 = %s, unit = %s' %(event_num, key, key2, unit_no)
			break
                
                if max(binned_temp[0]) > max_bin[event_num]:
                        print 'max hist > max_bin event %s unit %s, diff = %s' %(event_num,unit_no,max(binned_temp[0])-max_bin[event_num])
                        max_bin[event_num] = max(binned_temp[0])
                        


		binned_nl_temp_num = binned_temp[0] - min_bin[event_num]
		binned_nl_temp_denom = max_bin[event_num] - min_bin[event_num]

		binned_nl_temp = np.true_divide(binned_nl_temp_num,binned_nl_temp_denom)
                if binned_nl_temp.max() > 1.0:
                        print 'error: binned nl > 1.0 for event %s, unit %s, %s, %s' %(str(event_num),unit_no, key, key2)
		binned_nl.append(binned_nl_temp)
	
	binned = np.asarray(binned)
	binned_nl = np.asarray(binned_nl)
        
        firing_rate = np.zeros(len(binned),dtype=object)
	for i in range(len(binned)):
 		firing_rate[i] = binned[i] / float(bin_size) * 1000 #because bin_size in ms, want to get Hz

	avg_firing_rate = np.mean(firing_rate)
	std_dev_firing_rate = np.std(firing_rate)

	avg_nl = np.mean(binned_nl,axis=0)
	std_dev_nl = np.std(binned_nl,axis=0)
	
	plt.axvline(0.0,color='r')
	plt.xlim(time_before,time_after)
	
	plt.title('Raster plot')
	plt.xlabel('Time from event')
	plt.ylabel('Trial')

	#Average firing rate
	plt.subplot(2,2,2)
	plt.title('average firing rate')
	plt.xlabel('Time from event')
	plt.ylabel('Firing rate (Hz)')
	
	ymax = avg_firing_rate + std_dev_firing_rate
	ymin = avg_firing_rate - std_dev_firing_rate

        plt.errorbar(np.linspace(time_before,time_after,no_bins,endpoint=True),avg_firing_rate,yerr=std_dev_firing_rate)
        plt.ylim(ymin=0)
        plt.xlim(time_before,time_after)
        plt.axvline(0.0,color='r')

        #normalized firing rate
        plt.subplot(2,2,3)

        plt.errorbar(np.linspace(time_before,time_after,no_bins,endpoint=True),avg_nl,yerr=std_dev_nl)
        plt.ylim(0,1.0)
	plt.xlim(time_before,time_after)
	plt.axvline(0.0,color='r')
	
	plt.title('average normalized firing rate')
	plt.xlabel('Time from event')
	plt.ylabel('Normalized firing rate')

	plt.suptitle('%s, %s, unit: %s' %(key,key2,str(unit_no)), fontsize = 20)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	
	plt.savefig('raster_%s_%s_unit%s' %(key, key2, str(unit_no)))
	plt.clf()

	return_dict={'avg_firing_rate':avg_firing_rate,'std_dev_firing_rate':std_dev_firing_rate,'binned':binned,'firing_rate':firing_rate,'adjusted_times':adjusted_times,'dummy':dummy,'spikes':spikes,'dummy1':dummy1,'a':a,'b':b,'c':c,'binned_temp':binned_temp,'binned_nl':binned_nl,'avg_nl':avg_nl,'std_dev_nl':std_dev_nl}
	
	return(return_dict)


####################
# NEW TS Dict
####################
#col 0 = disp_rp, 1 = succ scene, 2 = failure scene, 3 = rnum, 4 = pnum, 5 = catch_num

r0_succ = all_succ[all_succ[:,3] == 0]
r1_succ = all_succ[all_succ[:,3] == 1]
r2_succ = all_succ[all_succ[:,3] == 2]
r3_succ = all_succ[all_succ[:,3] == 3]
r0_fail = all_fail[all_fail[:,3] == 0]
r1_fail = all_fail[all_fail[:,3] == 1]
r2_fail = all_fail[all_fail[:,3] == 2]
r3_fail = all_fail[all_fail[:,3] == 3]
ra_succ = all_succ[all_succ[:,3] != 0]
ra_fail = all_fail[all_fail[:,3] != 0]

p0_fail = all_fail[all_fail[:,4] == 0]
p1_fail = all_fail[all_fail[:,4] == 1]
p2_fail = all_fail[all_fail[:,4] == 2]
p3_fail = all_fail[all_fail[:,4] == 3]
p0_succ = all_succ[all_succ[:,4] == 0]
p1_succ = all_succ[all_succ[:,4] == 1]
p2_succ = all_succ[all_succ[:,4] == 2]
p3_succ = all_succ[all_succ[:,4] == 3]
pa_fail = all_fail[all_fail[:,4] != 0]
pa_succ = all_succ[all_succ[:,4] != 0]

#catch
r0_catch_succ = catch_succ[catch_succ[:,3] == 0]
r1_catch_succ = catch_succ[catch_succ[:,3] == 1]
r2_catch_succ = catch_succ[catch_succ[:,3] == 2]
r3_catch_succ = catch_succ[catch_succ[:,3] == 3]
p0_catch_fail = catch_fail[catch_fail[:,4] == 0]
p1_catch_fail = catch_fail[catch_fail[:,4] == 1]
p2_catch_fail = catch_fail[catch_fail[:,4] == 2]
p3_catch_fail = catch_fail[catch_fail[:,4] == 3]
#TODO for when there's not an occasion

r0_succ_cue = r0_succ[:,0]
r1_succ_cue = r1_succ[:,0]
r2_succ_cue = r2_succ[:,0]
r3_succ_cue = r3_succ[:,0]
p0_fail_cue = p0_fail[:,0]
p1_fail_cue = p1_fail[:,0]
p2_fail_cue = p2_fail[:,0]
p3_fail_cue = p3_fail[:,0]
ra_succ_cue = ra_succ[:,0]
pa_fail_cue = pa_fail[:,0]

r0_catch_cue = r0_catch_succ[:,0]
r1_catch_cue = r1_catch_succ[:,0]
r2_catch_cue = r2_catch_succ[:,0]
r3_catch_cue = r3_catch_succ[:,0]
p0_catch_cue = p0_catch_fail[:,0]
p1_catch_cue = p1_catch_fail[:,0]
p2_catch_cue = p2_catch_fail[:,0]
p3_catch_cue = p3_catch_fail[:,0]
r_all_catch_cue = catch_succ[:,0]
p_all_catch_cue = catch_fail[:,0]

r0_succ_result = r0_succ[:,1]
r1_succ_result = r1_succ[:,1]
r2_succ_result = r2_succ[:,1]
r3_succ_result = r3_succ[:,1]
p0_fail_result = p0_fail[:,2]
p1_fail_result = p1_fail[:,2]
p2_fail_result = p2_fail[:,2]
p3_fail_result = p3_fail[:,2]
ra_succ_result = ra_succ[:,1]
pa_fail_result = pa_fail[:,2]

r0_catch_result = r0_catch_succ[:,1]
r1_catch_result = r1_catch_succ[:,1]
r2_catch_result = r2_catch_succ[:,1]
r3_catch_result = r3_catch_succ[:,1]
p0_catch_result = p0_catch_fail[:,2]
p1_catch_result = p1_catch_fail[:,2]
p2_catch_result = p2_catch_fail[:,2]
p3_catch_result = p3_catch_fail[:,2]
r_all_catch_result = catch_succ[:,1]
p_all_catch_result = catch_fail[:,2]

r0_fail_cue = r0_fail[:,0]
r1_fail_cue = r1_fail[:,0]
r2_fail_cue = r2_fail[:,0]
r3_fail_cue = r3_fail[:,0]
p0_succ_cue = p0_succ[:,0]
p1_succ_cue = p1_succ[:,0]
p2_succ_cue = p2_succ[:,0]
p3_succ_cue = p3_succ[:,0]
ra_fail_cue = ra_fail[:,0]
pa_succ_cue = pa_succ[:,0]

r0_fail_result = r0_fail[:,2]
r1_fail_result = r1_fail[:,2]
r2_fail_result = r2_fail[:,2]
r3_fail_result = r3_fail[:,2]
p0_succ_result = p0_succ[:,1]
p1_succ_result = p1_succ[:,1]
p2_succ_result = p2_succ[:,1]
p3_succ_result = p3_succ[:,1]
ra_fail_result = ra_fail[:,2]
pa_succ_result = pa_succ[:,1]

if rcatch_bool:
        ts_dict = {'r0_succ_cue':r0_succ_cue,'r1_succ_cue':r1_succ_cue,'r2_succ_cue':r2_succ_cue,'r3_succ_cue':r3_succ_cue,'p0_fail_cue':p0_fail_cue,'p1_fail_cue':p1_fail_cue,'p2_fail_cue':p2_fail_cue,'p3_fail_cue':p3_fail_cue,'r0_succ_result':r0_succ_result,'r1_succ_result':r1_succ_result,'r2_succ_result':r2_succ_result,'r3_succ_result':r3_succ_result,'p0_fail_result':p0_fail_result,'p1_fail_result':p1_fail_result,'p2_fail_result':p2_fail_result,'p3_fail_result':p3_fail_result,'r0_fail_cue':r0_fail_cue,'r1_fail_cue':r1_fail_cue,'r2_fail_cue':r2_fail_cue,'r3_fail_cue':r3_fail_cue,'p0_succ_cue':p0_succ_cue,'p1_succ_cue':p1_succ_cue,'p2_succ_cue':p2_succ_cue,'p3_succ_cue':p3_succ_cue,'r0_fail_result':r0_fail_result,'r1_fail_result':r1_fail_result,'r2_fail_result':r2_fail_result,'r3_fail_result':r3_fail_result,'p0_succ_result':p0_succ_result,'p1_succ_result':p1_succ_result,'p2_succ_result':p2_succ_result,'p3_succ_result':p3_succ_result,'r0_catch_cue':r0_catch_cue,'r1_catch_cue':r1_catch_cue,'r2_catch_cue':r2_catch_cue,'r3_catch_cue':r3_catch_cue,'p0_catch_cue':p0_catch_cue,'p1_catch_cue':p1_catch_cue,'p2_catch_cue':p2_catch_cue,'p3_catch_cue':p3_catch_cue,'r_all_catch_cue':r_all_catch_cue,'p_all_catch_cue':p_all_catch_cue,'r0_catch_result':r0_catch_result,'r1_catch_result':r1_catch_result,'r2_catch_result':r2_catch_result,'r3_catch_result':r3_catch_result,'p0_catch_result':p0_catch_result,'p1_catch_result':p1_catch_result,'p2_catch_result':p2_catch_result,'p3_catch_result':p3_catch_result,'r_all_catch_result':r_all_catch_result,'p_all_catch_result':p_all_catch_result,'ra_succ_cue':ra_succ_cue,'ra_fail_cue':ra_fail_cue,'ra_succ_result':ra_succ_result,'ra_fail_result':ra_fail_result,'pa_succ_cue':pa_succ_cue,'pa_fail_cue':pa_fail_cue,'pa_succ_result':pa_succ_result,'pa_fail_result':pa_fail_result}        
else:
        ts_dict = {'r0_succ_cue':r0_succ_cue,'r1_succ_cue':r1_succ_cue,'r2_succ_cue':r2_succ_cue,'r3_succ_cue':r3_succ_cue,'p0_fail_cue':p0_fail_cue,'p1_fail_cue':p1_fail_cue,'p2_fail_cue':p2_fail_cue,'p3_fail_cue':p3_fail_cue,'r0_succ_result':r0_succ_result,'r1_succ_result':r1_succ_result,'r2_succ_result':r2_succ_result,'r3_succ_result':r3_succ_result,'p0_fail_result':p0_fail_result,'p1_fail_result':p1_fail_result,'p2_fail_result':p2_fail_result,'p3_fail_result':p3_fail_result,'r0_fail_cue':r0_fail_cue,'r1_fail_cue':r1_fail_cue,'r2_fail_cue':r2_fail_cue,'r3_fail_cue':r3_fail_cue,'p0_succ_cue':p0_succ_cue,'p1_succ_cue':p1_succ_cue,'p2_succ_cue':p2_succ_cue,'p3_succ_cue':p3_succ_cue,'r0_fail_result':r0_fail_result,'r1_fail_result':r1_fail_result,'r2_fail_result':r2_fail_result,'r3_fail_result':r3_fail_result,'p0_succ_result':p0_succ_result,'p1_succ_result':p1_succ_result,'p2_succ_result':p2_succ_result,'p3_succ_result':p3_succ_result,'ra_succ_cue':ra_succ_cue,'ra_fail_cue':ra_fail_cue,'ra_succ_result':ra_succ_result,'ra_fail_result':ra_fail_result,'pa_succ_cue':pa_succ_cue,'pa_fail_cue':pa_fail_cue,'pa_succ_result':pa_succ_result,'pa_fail_result':pa_fail_result}

#remove if no instances of a ts
ts_new = {}
for ts_key,ts_data in ts_dict.iteritems():
        if np.shape(ts_dict[ts_key])[0] == 0:
                print 'no instances of %s' %(ts_key)
        else:
                ts_new[ts_key] = ts_dict[ts_key]
ts_dict = ts_new

data_dict={'M1_spikes':M1_spikes,'S1_spikes':S1_spikes,'PmD_spikes':PmD_spikes,'PmV_spikes':PmV_spikes}

data_dict_nl = {}
for key, value in data_dict.iteritems():
	spike_data = []
	if key == 'PmV_spikes':
		for i in range(len(value)):
			spike_data.append(value[i])
	else:
		for i in range(len(value)):
			spike_data.append(value[i]['ts'][0,0][0])

	print 'normalizing %s' %(key)		
	normalized_data = normalize_data(spike_data)
	data_dict_nl['%s_nl' %(key)] = normalized_data

print "min hist len = %s" %(min_hist_size)

condensed = condensed[condensed[:,0] < min_hist_size * 50 / 1000.0]

if condensed[-1,1] != 0:
        if condensed[-1,1] > min_hist_len * 50 / 1000.0:
                condensed = condensed[0:-1,:]

if condensed[-1,2] != 0:
        if condensed[-1,2] > min_hist_len * 50 / 1000.0:
                condensed = condensed[0:-1,:]

both_bool_succ = [x and y for x,y in zip(condensed[:,1] !=0, condensed[:,5] == 0)]
both_bool_fail = [x and y for x,y in zip(condensed[:,2] !=0, condensed[:,5] == 0)]
catch_succ_bool = [x and y for x,y in zip(condensed[:,1] !=0, condensed[:,5] ==1)]
catch_fail_bool = [x and y for x,y in zip(condensed[:,2] !=0, condensed[:,5] ==2)]

all_succ = []
all_fail = []
catch_succ = []
catch_fail = []

out_of_bounds_bool = False
for i in range(np.shape(condensed)[0]):
        if both_bool_succ[i]:
                all_succ.append(condensed[i,:])
        if both_bool_fail[i]:
                all_fail.append(condensed[i,:])
        if catch_succ_bool[i]:
                catch_succ.append(condensed[i,:])
        if catch_fail_bool[i]:
                catch_fail.append(condensed[i,:])

        #if not out_of_bounds_bool and condensed[i,0]*20.0 > 


all_succ = np.asarray(all_succ)
all_fail = np.asarray(all_fail)
catch_succ = np.asarray(catch_succ)
catch_fail = np.asarray(catch_fail)

M1_dicts = {'spikes':data_dict['M1_spikes'],'nl':data_dict_nl['M1_spikes_nl']} 
S1_dicts = {'spikes':data_dict['S1_spikes'],'nl':data_dict_nl['S1_spikes_nl']}
PmD_dicts = {'spikes':data_dict['PmD_spikes'],'nl':data_dict_nl['PmD_spikes_nl']}
PmV_dicts = {'spikes':data_dict['PmV_spikes'],'nl':data_dict_nl['PmV_spikes_nl']}

data_dict_all = {'M1_dicts':M1_dicts,'S1_dicts':S1_dicts,'PmD_dicts':PmD_dicts,'PmV_dicts':PmV_dicts}

##### TESTING #####
#ts_dict = {'r0_succ_cue':r0_succ_cue}
#ts_dict = {'p2_succ_result':p2_succ_result,'p2_succ_cue':p2_succ_cue}
#data_dict_all = {'M1_dicts':M1_dicts}
#ts_dict = {'r0_catch_result':r0_catch_result,'r0_catch_cue':r0_catch_cue}



save_dict = {}
save_dict['data_dict_all'] = data_dict_all
save_dict['param_dict'] = param_dict


total_rasters = []
for ts_key, ts_value in ts_dict.iteritems():
	if not (len(ts_value) == 1 and ts_value[0] == 0):
		for data_key, data_value in data_dict_all.iteritems():
			print 'making rasters %s, %s' %(ts_key, data_key)

			spikes=[]
			if data_key == 'PmV_dicts':
				for i in range(len(data_dict_all[data_key]['spikes'])):
					spikes.append(data_dict_all[data_key]['spikes'][i])
			else:
				for i in range(len(data_dict_all[data_key]['spikes'])):
					spikes.append(data_dict_all[data_key]['spikes'][i]['ts'][0,0][0])			

			for i in range(len(data_dict_all[data_key]['spikes'])):
                                if not value.size == 0:				
                                        rasters = make_nl_raster_new(ts_value,ts_key,data_dict_all[data_key]['nl'],data_key,i,spikes)
                                        if rasters:
                                                total_rasters.append(rasters)
                                else:
                                        print 'no instances of %s' %(ts_key)

                        save_dict['%s_%s' %(ts_key, data_key)] = total_rasters
                        total_rasters = []
	else:
		print 'no instances of event %s, no plot generation' %(ts_key)


short_filename = filename[-27:-4]
if short_filename.startswith('0'):
	short_filename = '0%s' %(short_filename)

np.save('avg_fr_and_nlized_data_%s_bin%s' %(short_filename,bin_size),save_dict)
