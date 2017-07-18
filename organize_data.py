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

bin_size = 50 #in ms
time_before = -0.5 #negative value
time_after = 1.0
vm_bool = True

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

###############################################
#functions #####################################
###############################################

#create histogram of spike data, normalize (if want to use spike data normalized throughout the trial. Saves as normalized and not-normalized hist data)
def normalize_data(spike_data):
	hist = []
	nl_spike_data = []
	hist_all = []
	nl_dict = {}
	min_bin = np.zeros(len(spike_data))
	max_bin = np.zeros(len(spike_data))
	global min_hist_len

	for i in range(len(spike_data)):
                total_bin_range = np.arange(0,int(np.ceil(spike_data[i].max())),bin_size/1000.0)
                hist = np.histogram(spike_data[i], bins=total_bin_range,range=(0,int(np.ceil(spike_data[i].max()))),normed=False, density=False)
                
                if len(hist[0]) < min_hist_len:
                        min_hist_len = len(hist[0])
                
                min_bin[i] = min(hist[0])
                max_bin[i] = max(hist[0])
		
                num = np.subtract(hist[0],min_bin[i])
                denom = max_bin[i]-min_bin[i]		
                nl_temp = np.true_divide(num,denom)
                nl_spike_data.append(nl_temp)
                hist_all.append(hist)

 	nl_dict['hist_all'] = hist_all
	nl_dict['nl_spike_data'] = nl_spike_data
	nl_dict['min_bin'] = min_bin
	nl_dict['max_bin'] = max_bin

	return(nl_dict)

def make_nl_raster(ts_value,ts_key,nl_dict,data_key,unit_num,spikes):
        hist_all = nl_dict['hist_all'][unit_num][0]
        
        nlized_hist_data = nl_dict['nl_spike_data'][unit_num]
        bin_size_sec = bin_size / 1000.0
        no_bins_before = int(time_before / bin_size_sec) #neg, keep neg
        no_bins_after = int(time_after / bin_size_sec)

        event_hist = []
        event_hist_nl = []

        if 'result' in ts_key and 'all' in ts_key and not 'catch' in ts_key and not vm_bool:
                result_bool = True
                prefix = ts_key[0:1]
                cue_key = prefix + 'all_cue'
                cue_ts = ts_dict[cue_key]
        elif 'result' in ts_key and 'all' in ts_key and 'catch' in ts_key and not vm_bool:
                result_bool = True
                prefix = ts_key[0:1]
                cue_key = prefix + '_all_catch_cue'
                cue_ts = ts_dict[cue_key]
        elif 'result' in ts_key and 'catch' in ts_key and not vm_bool:
                result_bool = True
                prefix = ts_key[0:3]
                cue_key= prefix + 'catch_cue'
                cue_ts = ts_dict[cue_key]
        elif 'result' in ts_key and not vm_bool:
                prefix = ts_key[0:8]
                cue_key = prefix + 'cue'
                result_bool = True
                cue_ts = ts_dict[cue_key]

        elif 'result' in ts_key and vm_bool:
                #pdb.set_trace()
                prefix = ts_key[0:9]
                
                if ts_key[-2] == '-':
                        suffix = ts_key[-3:]
                else:
                        suffix = ts_key[-2:]
                cue_key = prefix + 'cue' + suffix
                cue_ts = ts_dict[cue_key]
                result_bool = True
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


###############################################
#start ########################################
###############################################

#load files (from Extracted and timestamp files)
print filename
a = sio.loadmat(filename);
#Pull reward timestamps. Separate timestamp file must exist in folder containing neural data, with same name+_timestamps
print filename[:-4]+"_timestamps"+filename[-4:]
Timestamps = sio.loadmat(filename[:-4]+"_timestamps"+filename[-4:]);

#create matrix of trial-by-trial info
trial_breakdown = Timestamps['trial_breakdown']
condensed = np.zeros((np.shape(trial_breakdown)[0],6))

#col 0 = disp_rp, 1 = succ scene, 2 = failure scene, 3 = rnum, 4 = pnum, 5 = catch_num
condensed[:,0] = trial_breakdown[:,1]
condensed[:,1] = trial_breakdown[:,2]
condensed[:,2] = trial_breakdown[:,3]
condensed[:,3] = trial_breakdown[:,5]
condensed[:,4] = trial_breakdown[:,7]
condensed[:,5] = trial_breakdown[:,10]

#delete end trials if not fully finished
if condensed[-1,1] == condensed[-1,2] == 0:
	new_condensed = condensed[0:-1,:]
        condensed = new_condensed
condensed = condensed[condensed[:,0] != 0]

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

#take care of some misc params, save param dict
no_bins = int((time_after - time_before) * 1000 / bin_size)
print 'bin size = %s' %(bin_size)
param_dict = {'time_before':time_before,'time_after':time_after,'bin_size':bin_size,'no_bins':no_bins}
gripforce_all = np.asarray(Timestamps['gripforce'])
min_hist_len = 1000000 #set arbitrarily high number, because want to determin min length later
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

#make sure not trying to index trials that go beyond smallest recorded hist size 
#TODO ADD length to hist below
print "min hist len = %s" %(min_hist_len)
condensed = condensed[condensed[:,0] < (min_hist_len * 50 / 1000.0 + time_after)]
if condensed[-1,1] != 0:
        if condensed[-1,1] > (min_hist_len * 50 / 1000.0 + time_after):
                condensed = condensed[0:-1,:]
if condensed[-1,2] != 0:
        if condensed[-1,2] > (min_hist_len * 50 / 1000.0 + time_after):
                condensed = condensed[0:-1,:]

condensed_mv = np.zeros((np.shape(condensed)[0],8))
condensed_mv[:,0:6] = condensed
condensed_mv[:,6] = condensed[:,3] - condensed[:,4] #value vector
condensed_mv[:,7] = condensed[:,3] + condensed[:,4] #motivation vector

val_sorted = condensed_mv[np.argsort(condensed_mv[:,6])]
mtv_sorted = condensed_mv[np.argsort(condensed_mv[:,7])]

#break uo into trial types
both_bool_succ = [x and y for x,y in zip(condensed[:,1] !=0, condensed[:,5] == 0)]
both_bool_fail = [x and y for x,y in zip(condensed[:,2] !=0, condensed[:,5] == 0)]
catch_succ_bool = [x and y for x,y in zip(condensed[:,1] !=0, condensed[:,5] ==1)]
catch_fail_bool = [x and y for x,y in zip(condensed[:,2] !=0, condensed[:,5] ==2)]

#bool for use later, for blocks with and without catch trials
if catch_succ_bool.count(True) != 0 or catch_fail_bool.count(True) !=0:
    catch_bool = True

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

all_succ = np.asarray(all_succ)
all_fail = np.asarray(all_fail)
catch_succ = np.asarray(catch_succ)
catch_fail = np.asarray(catch_fail)

M1_dicts = {'spikes':data_dict['M1_spikes'],'nl':data_dict_nl['M1_spikes_nl']} 
S1_dicts = {'spikes':data_dict['S1_spikes'],'nl':data_dict_nl['S1_spikes_nl']}
PmD_dicts = {'spikes':data_dict['PmD_spikes'],'nl':data_dict_nl['PmD_spikes_nl']}
PmV_dicts = {'spikes':data_dict['PmV_spikes'],'nl':data_dict_nl['PmV_spikes_nl']}

data_dict_all = {'M1_dicts':M1_dicts,'S1_dicts':S1_dicts,'PmD_dicts':PmD_dicts,'PmV_dicts':PmV_dicts}

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


###########################################3
vm_ts_dicts = {}
for i in range(7):

        mtv_succ_cue_key = 'mtv_succ_cue_' + str(i)
        mtv_fail_cue_key = 'mtv_fail_cue_' + str(i)
        mtv_all_cue_key = 'mtv_all_cue_' + str(i)
        mtv_succ_result_key = 'mtv_succ_result_' + str(i)
        mtv_fail_result_key = 'mtv_fail_result_' + str(i)
        val_succ_cue_key = 'val_succ_cue_' + str(i-3)
        val_fail_cue_key = 'val_fail_cue_' + str(i-3)
        val_all_cue_key = 'val_all_cue_' + str(i-3)
        val_succ_result_key = 'val_succ_result_' + str(i-3)
        val_fail_result_key = 'val_fail_result_' + str(i-3)
        
        temp_mtv = condensed_mv[condensed_mv[:,7] == i]
        temp_val = condensed_mv[condensed_mv[:,6] == i-3]

        mtv_all_cue = temp_mtv[:,0]
        mtv_succ_bool = [x and y for x,y in zip(temp_mtv[:,1] != 0, temp_mtv[:,5] == 0)]
        mtv_fail_bool = [x and y for x,y in zip(temp_mtv[:,2] != 0, temp_mtv[:,5] == 0)]

        val_all_cue = temp_val[:,0]
        val_succ_bool = [x and y for x,y in zip(temp_val[:,1] != 0, temp_val[:,5] == 0)]
        val_fail_bool = [x and y for x,y in zip(temp_val[:,2] != 0, temp_val[:,5] == 0)]

        mtv_succ_cue = []
        mtv_fail_cue = []
        mtv_succ_result = []
        mtv_fail_result = []
        val_succ_cue = []
        val_fail_cue = []
        val_succ_result = []
        val_fail_result = []

        for i in range(np.shape(temp_mtv)[0]):
                if mtv_succ_bool[i]:
                        mtv_succ_cue.append(temp_mtv[i,0])
                        mtv_succ_result.append(temp_mtv[i,1])
                if mtv_fail_bool[i]:
                        mtv_fail_cue.append(temp_mtv[i,0])
                        mtv_fail_result.append(temp_mtv[i,2])
        for i in range(np.shape(temp_val)[0]):
                if val_succ_bool[i]:
                        val_succ_cue.append(temp_val[i,0])
                        val_succ_result.append(temp_val[i,1])
                if val_fail_bool[i]:
                        val_fail_cue.append(temp_val[i,0])
                        val_fail_result.append(temp_val[i,2])
        
        vm_ts_dicts[mtv_succ_cue_key] = mtv_succ_cue
        vm_ts_dicts[mtv_succ_result_key] = mtv_succ_result
        vm_ts_dicts[mtv_fail_cue_key] = mtv_fail_cue
        vm_ts_dicts[mtv_fail_result_key] = mtv_fail_result
        vm_ts_dicts[val_succ_cue_key] = val_succ_cue
        vm_ts_dicts[val_succ_result_key] = val_succ_result
        vm_ts_dicts[val_fail_cue_key] = val_fail_cue
        vm_ts_dicts[val_fail_result_key] = val_fail_result
        vm_ts_dicts[mtv_all_cue_key] = mtv_all_cue
        vm_ts_dicts[val_all_cue_key] = val_all_cue
                

if catch_bool:
        ts_dict = {'r0_succ_cue':r0_succ_cue,'r1_succ_cue':r1_succ_cue,'r2_succ_cue':r2_succ_cue,'r3_succ_cue':r3_succ_cue,'p0_fail_cue':p0_fail_cue,'p1_fail_cue':p1_fail_cue,'p2_fail_cue':p2_fail_cue,'p3_fail_cue':p3_fail_cue,'r0_succ_result':r0_succ_result,'r1_succ_result':r1_succ_result,'r2_succ_result':r2_succ_result,'r3_succ_result':r3_succ_result,'p0_fail_result':p0_fail_result,'p1_fail_result':p1_fail_result,'p2_fail_result':p2_fail_result,'p3_fail_result':p3_fail_result,'r0_fail_cue':r0_fail_cue,'r1_fail_cue':r1_fail_cue,'r2_fail_cue':r2_fail_cue,'r3_fail_cue':r3_fail_cue,'p0_succ_cue':p0_succ_cue,'p1_succ_cue':p1_succ_cue,'p2_succ_cue':p2_succ_cue,'p3_succ_cue':p3_succ_cue,'r0_fail_result':r0_fail_result,'r1_fail_result':r1_fail_result,'r2_fail_result':r2_fail_result,'r3_fail_result':r3_fail_result,'p0_succ_result':p0_succ_result,'p1_succ_result':p1_succ_result,'p2_succ_result':p2_succ_result,'p3_succ_result':p3_succ_result,'r0_catch_cue':r0_catch_cue,'r1_catch_cue':r1_catch_cue,'r2_catch_cue':r2_catch_cue,'r3_catch_cue':r3_catch_cue,'p0_catch_cue':p0_catch_cue,'p1_catch_cue':p1_catch_cue,'p2_catch_cue':p2_catch_cue,'p3_catch_cue':p3_catch_cue,'r_all_catch_cue':r_all_catch_cue,'p_all_catch_cue':p_all_catch_cue,'r0_catch_result':r0_catch_result,'r1_catch_result':r1_catch_result,'r2_catch_result':r2_catch_result,'r3_catch_result':r3_catch_result,'p0_catch_result':p0_catch_result,'p1_catch_result':p1_catch_result,'p2_catch_result':p2_catch_result,'p3_catch_result':p3_catch_result,'r_all_catch_result':r_all_catch_result,'p_all_catch_result':p_all_catch_result,'ra_succ_cue':ra_succ_cue,'ra_fail_cue':ra_fail_cue,'ra_succ_result':ra_succ_result,'ra_fail_result':ra_fail_result,'pa_succ_cue':pa_succ_cue,'pa_fail_cue':pa_fail_cue,'pa_succ_result':pa_succ_result,'pa_fail_result':pa_fail_result}        
else:
        ts_dict = {'r0_succ_cue':r0_succ_cue,'r1_succ_cue':r1_succ_cue,'r2_succ_cue':r2_succ_cue,'r3_succ_cue':r3_succ_cue,'p0_fail_cue':p0_fail_cue,'p1_fail_cue':p1_fail_cue,'p2_fail_cue':p2_fail_cue,'p3_fail_cue':p3_fail_cue,'r0_succ_result':r0_succ_result,'r1_succ_result':r1_succ_result,'r2_succ_result':r2_succ_result,'r3_succ_result':r3_succ_result,'p0_fail_result':p0_fail_result,'p1_fail_result':p1_fail_result,'p2_fail_result':p2_fail_result,'p3_fail_result':p3_fail_result,'r0_fail_cue':r0_fail_cue,'r1_fail_cue':r1_fail_cue,'r2_fail_cue':r2_fail_cue,'r3_fail_cue':r3_fail_cue,'p0_succ_cue':p0_succ_cue,'p1_succ_cue':p1_succ_cue,'p2_succ_cue':p2_succ_cue,'p3_succ_cue':p3_succ_cue,'r0_fail_result':r0_fail_result,'r1_fail_result':r1_fail_result,'r2_fail_result':r2_fail_result,'r3_fail_result':r3_fail_result,'p0_succ_result':p0_succ_result,'p1_succ_result':p1_succ_result,'p2_succ_result':p2_succ_result,'p3_succ_result':p3_succ_result,'ra_succ_cue':ra_succ_cue,'ra_fail_cue':ra_fail_cue,'ra_succ_result':ra_succ_result,'ra_fail_result':ra_fail_result,'pa_succ_cue':pa_succ_cue,'pa_fail_cue':pa_fail_cue,'pa_succ_result':pa_succ_result,'pa_fail_result':pa_fail_result}

save_dict = {}
save_dict['data_dict_all'] = data_dict_all
save_dict['param_dict'] = param_dict


total_rasters = []
if vm_bool:
        ts_dict = vm_ts_dicts

for ts_key, ts_value in ts_dict.iteritems():
	#if not (len(ts_value) == 1 and ts_value[0] == 0):
	if len(ts_value) != 0:
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
                                        rasters = make_nl_raster(ts_value,ts_key,data_dict_all[data_key]['nl'],data_key,i,spikes)
                                        if rasters:
                                                total_rasters.append(rasters)
                                else:
                                        print 'no instances of %s' %(ts_key)

                        save_dict['%s_%s' %(ts_key, data_key)] = total_rasters
                        total_rasters = []
	else:
		print 'no instances of event %s' %(ts_key)


short_filename = filename[-27:-4]
if short_filename.startswith('0'):
	short_filename = '0%s' %(short_filename)

if not vm_bool:
        np.save('avg_fr_and_nlized_data_%s_bin%s' %(short_filename,bin_size),save_dict)
else:
        np.save('avg_fr_and_nlized_data_mv_%s_bin%s' %(short_filename,bin_size),save_dict)

        

#for data_key,data_value in data_dict_all.iteritems():
#        for i in range(7):
#                print i
