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

bin_size = 500 #in ms

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
filename = '/home/jack/Dropbox/single_rp_files/extracted/20151019_0059/Extracted_0059_2015-10-19-16-25-20.mat'
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


######################
print filename
a = sio.loadmat(filename);

#Pull reward timestamps. Separate timespamp file must exist in folder containing neural data, with same name+_timestamps
print filename[:-4]+"_timestamps"+filename[-4:]
Timestamps = sio.loadmat(filename[:-4]+"_timestamps"+filename[-4:]);

#Timestamp generation: generates arrays of timestamps depending on reward level, including offset for cue presentation and reward delivery
#one reward cue, one punishment cue, successful (rewarding)
rp_s_rdelivery = Timestamps['rp_s_rdelivery_ts'][0];
rp_s_cue = Timestamps['rp_s_cue_ts'][0];
print ('rp_s trials: %s'%(len(rp_s_rdelivery)))
#in case there happened to not be any particular trial during a block, this makes the arrays dimensions equal. TODO look into better way instead of creating a timepoint
if not rp_s_rdelivery.size:
	rp_s_rdelivery = np.array([0])
	rp_s_cue = np.array([0])

#one reward cue, one punishment cue, unsuccessful (punishing)
rp_f_pdelivery = Timestamps['rp_f_pdelivery_ts'][0];
rp_f_cue = Timestamps['rp_f_cue_ts'][0];
print ('rp_f trials: %s'%(len(rp_f_pdelivery)))
if not rp_f_pdelivery.size:
	rp_f_pdelivery = np.array([0])
	rp_f_cue = np.array([0])

#no reward cue, no punishment cue, successful (non-rewarding)
nrnp_s_nextreset = Timestamps['nrnp_s_nextreset'][0];
nrnp_s_cue = Timestamps['nrnp_s_cue_ts'][0];
print ('nrnp_s trials: %s'%(len(nrnp_s_cue)))
if not nrnp_s_nextreset.size:
	nrnp_s_nextreset = np.array([0])
	nrnp_s_cue = np.array([0])

#no reward cue, no punishment cue, unsuccessful (non-punishing)
nrnp_f_nextreset = Timestamps['nrnp_f_nextreset'][0];
nrnp_f_cue = Timestamps['nrnp_f_cue_ts'][0];
print ('nrnp_f trials: %s'%(len(nrnp_f_cue)))   
if not nrnp_f_nextreset.size:
	nrnp_f_nextreset = np.array([0])
	nrnp_f_cue = np.array([0])

#reward cue only (no punishment cue), successful (rewarding)
r_only_s_rdelivery = Timestamps['r_only_s_rdelivery_ts'][0];
r_only_s_cue = Timestamps['r_only_s_cue_ts'][0];
print ('r_only_s trials: %s'%(len(r_only_s_cue)))
if not r_only_s_rdelivery.size:
	r_only_s_rdelivery = np.array([0])
	r_only_s_cue = np.array([0])

#reward cue only (no punishment cue), unsuccessful (non-rewarding)
r_only_f_nextreset = Timestamps['r_only_f_nextreset'][0];
r_only_f_cue = Timestamps['r_only_f_cue_ts'][0];
print ('r_only_f trials: %s'%(len(r_only_f_cue)))
if not r_only_f_nextreset.size:
	r_only_f_nextreset = np.array([0])
	r_only_f_cue = np.array([0])

#punishment cue only (no reward cue), successful (non-punishing)
p_only_s_nextreset = Timestamps['p_only_s_nextreset'][0];
p_only_s_cue = Timestamps['p_only_s_cue_ts'][0];
print ('p_only_s trials: %s'%(len(p_only_s_cue)))
if not p_only_s_nextreset.size:
	p_only_s_nextreset = np.array([0])
	p_only_s_cue = np.array([0])

#punishment cue only (no reward cue), unsuccessful (punishing)
p_only_f_pdelivery = Timestamps['p_only_f_pdelivery_ts'][0];
p_only_f_cue = Timestamps['p_only_f_cue_ts'][0];
print ('p_only_f trials: %s'%(len(p_only_f_cue)))
if not  p_only_f_pdelivery.size:
	p_only_f_pdelivery = np.array([0])
	p_only_f_cue = np.array([0])

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

#Pull neural spikes

neural_data=a['neural_data']
#print neural_data

Spikes = a['neural_data']['spikeTimes'];
#Break spikes into M1 S1 pmd 
#Different for jack/dave- jack has MAP1=PMd, MAP2=M1, MAP3=S1, ALL=PMv
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
dummy = [];
#M1_limit not defined for 0526_0059, blocks 1 and 2
for i in range(M1_limit,M1_spikes.shape[1]):
    dummy.append(M1_spikes[0,i]['ts'][0,0][0])
unit_names['M1_unit_names']=M1_unit_names
#Find first channel count for pmv on map3
S1_unit_names = []  
for i in range(0,S1_spikes.shape[1]):
    if int(S1_spikes[0,i]['signame'][0][0][0][3:-1]) > 96:
        S1_limit =i;
        print ("The number of units in S1 is: %s"%(S1_limit))
        break
    S1_unit_names.append(S1_spikes[0,i]['signame'][0][0][0])
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


#Begin main loop of analysis
for name_of_bin,time_of_bin in time_boundry.iteritems():
    before_time = -time_of_bin[0]
    after_time = time_of_bin[1]
    print( 'bin %s' %(name_of_bin))        
    
    #Assemble a final dictionary of the final dataset for saving
    def Make_final_data(data,start_position):
        dummy = []
        for i in range(0,data[start_position].shape[0]):
            dummy.append(np.hstack(data[start_position][i]))
        rp_s_data = np.array(dummy)
        rp_s_targets = np.ones(rp_s_data.shape[0])*0
        dummy = []
        for i in range(0,data[start_position+8].shape[0]):
            dummy.append(np.hstack(data[start_position+8][i]))
        rp_f_data = np.array(dummy)
        rp_f_targets = np.ones(rp_f_data.shape[0])*1
        dummy = []
        for i in range(0,data[start_position+16].shape[0]):
            dummy.append(np.hstack(data[start_position+16][i]))
        nrnp_s_data = np.array(dummy)
        nrnp_s_targets = np.ones(nrnp_s_data.shape[0])*2
        dummy = []
        for i in range(0,data[start_position+24].shape[0]):
            dummy.append(np.hstack(data[start_position+24][i]))
        nrnp_f_data = np.array(dummy)
        nrnp_f_targets = np.ones(nrnp_f_data.shape[0])*3
        dummy = []
        for i in range(0,data[start_position+32].shape[0]):
            dummy.append(np.hstack(data[start_position+32][i]))
        r_only_s_data = np.array(dummy)
        r_only_s_targets = np.ones(r_only_s_data.shape[0])*4
        dummy = []
        for i in range(0,data[start_position+40].shape[0]):
            dummy.append(np.hstack(data[start_position+40][i]))
        r_only_f_data = np.array(dummy)
        r_only_f_targets = np.ones(r_only_f_data.shape[0])*5
        dummy = []
        for i in range(0,data[start_position+48].shape[0]):
            dummy.append(np.hstack(data[start_position+48][i]))
        p_only_s_data = np.array(dummy)
        p_only_s_targets = np.ones(p_only_s_data.shape[0])*6
        dummy = []
        for i in range(0,data[start_position+56].shape[0]):
            dummy.append(np.hstack(data[start_position+56][i]))
        p_only_f_data = np.array(dummy)
        p_only_f_targets = np.ones(p_only_f_data.shape[0])*7
		
        return(np.vstack([rp_s_data,rp_f_data,nrnp_s_data,nrnp_f_data,r_only_s_data,r_only_f_data,p_only_s_data,p_only_f_data]),np.hstack([rp_s_targets,rp_f_targets,nrnp_s_targets,nrnp_f_targets,r_only_s_targets,r_only_f_targets,p_only_s_targets,p_only_f_targets]))


time_before = -0.5 #make sure if change to pos time before it still works
time_after = 1.0

no_bins = (time_after - time_before) * 1000 / bin_size

print 'bin size = %s' %(bin_size)

param_dict = {'time_before':time_before,'time_after':time_after,'time_boundry':time_boundry,'bin_size':bin_size,'no_bins':no_bins}


def normalize_data(spike_data):
	hist = []
	nl_spike_data = []
	hist_all = []
	nl_dict = {}
	min_bin = np.zeros(len(spike_data))
	max_bin = np.zeros(len(spike_data))
	
	for i in range(len(spike_data)):
		total_bin_range = np.arange(0,int(np.ceil(spike_data[i].max())),bin_size/1000.0)
		hist = np.histogram(spike_data[i], bins=total_bin_range,range=(0,int(np.ceil(spike_data[i].max()))),normed=False, density=False)

		min_bin[i] = min(hist[0])
		max_bin[i] = max(hist[0])
		
		#num = np.subtract(hist[0],min(hist[0]))
		#denom = max(hist[0]) - min(hist[0])
		num = np.subtract(hist[0],min_bin[i])
		denom = max_bin[i]-min_bin[i]
		
		nl_spike_data.append(np.true_divide(num,denom))
		hist_all.append(hist)

	nl_dict['hist_all'] = hist_all
	nl_dict['nl_spike_data'] = nl_spike_data
	nl_dict['min_bin'] = min_bin
	nl_dict['max_bin'] = max_bin
	
	
	return(nl_dict)
	

def make_raster(data,spikes,key,key2,unit_no,time_before,time_after):

	dummy = []
	
	for i in range(0,data.shape[0]):
		dummy1=[]
		b = data[i]
		for j in range(len(spikes)):
			a = spikes[j]
			c = a[np.where(np.logical_and(a>=b+time_before,a<=b+time_after))]
			dummy1.append(c)
		dummy.append(dummy1)

	dummy = np.asarray(dummy)
	ax = plt.gca()
	adjusted_times = []
	binned = []

	#rasters
	plt.subplot(2,1,1)
	
	for i in range(dummy.shape[0]):
		adjusted_times=dummy[i][unit_no]-value[i]
		plt.vlines(adjusted_times, i+.5, i+1.5)
		plt.ylim(.5, dummy.shape[0] + .5)

		binned_temp = np.histogram(adjusted_times,bins=no_bins,normed=False,density=False)
		binned.append(np.nan_to_num(binned_temp[0]))

	binned = np.asarray(binned)
	firing_rate = np.zeros(len(binned),dtype=object)
	for i in range(len(binned)):
 		firing_rate[i] = binned[i] / float(bin_size) * 1000 #because bin_size in ms, want to get Hz

	avg_firing_rate = np.mean(firing_rate)
	std_dev_firing_rate = np.std(firing_rate)
	
	plt.axvline(0.0)
	
	plt.title('Raster plot')
	plt.xlabel('Time from event')
	plt.ylabel('Trial')

	#average firing rate
	plt.subplot(2,1,2)
	plt.title('average firing rate')

	#TODO make time instead of bin
	plt.xlabel('Bin num')
	plt.ylabel('Firing rate (Hz)')
	
	ymax = avg_firing_rate + std_dev_firing_rate
	ymin = avg_firing_rate - std_dev_firing_rate

	#plt.plot(avg_firing_rate)
	#plt.fill_between(ymin,ymax,alpha=1,edgecolor='#3F7F4C',facecolor='#7EFF99',linewidth=0)
	plt.errorbar(np.arange(no_bins),avg_firing_rate,yerr=std_dev_firing_rate)
	plt.xlim(0,no_bins)
	plt.ylim(ymin=0)

	plt.axvline(0.0)
	
	plt.suptitle('%s, %s, unit: %s' %(key,key2,str(unit_no)), fontsize = 20)

	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	
	plt.savefig('raster_%s_%s_unit%s' %(key, key2, str(unit_no)))
	plt.clf()

	return_dict={'avg_firing_rate':avg_firing_rate,'std_dev_firing_rate':std_dev_firing_rate,'binned':binned,'firing_rate':firing_rate,'adjusted_times':adjusted_times,'dummy':dummy}
	
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

	for i in range(dummy.shape[0]):
		adjusted_times = dummy[i][unit_no] - data[i]
		plt.vlines(adjusted_times, i+.5, i+1.5)
		plt.ylim(.5, dummy.shape[0] + .5)

		binned_temp = np.histogram(adjusted_times,bins=no_bins,normed=False,density=False)
		binned.append(np.nan_to_num(binned_temp[0]))

		binned_nl_temp_num = binned_temp[0] - min_bin[i]
		binned_nl_temp_denom = max_bin[i] - min_bin[i]

		binned_nl_temp = np.divide(binned_nl_temp_num,binned_nl_temp_denom)
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

	#plt.plot(avg_firing_rate)
	#plt.fill_between(ymin,ymax,alpha=1,edgecolor='#3F7F4C',facecolor='#7EFF99',linewidth=0)
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

	#return_dict={'avg_firing_rate':avg_firing_rate,'std_dev_firing_rate':std_dev_firing_rate,'binned':binned,'firing_rate':firing_rate,'adjusted_times':adjusted_times,'dummy':dummy,'spikes':spikes,'dummy1':dummy1,'a':a,'b':b,'c':c,'nlized':nlized,'binned_temp':binned_temp,'binned_nl':binned_nl} #'a_nl':a_nl,'b_nl':b_nl,'c_nl':c_nl,'nlized':nlized,'dummy_nl':dummy_nl}
	return_dict={'avg_firing_rate':avg_firing_rate,'std_dev_firing_rate':std_dev_firing_rate,'adjusted_times':adjusted_times,'binned':binned,'binned_nl':binned_nl,'firing_rate':firing_rate,'avg_nl':avg_nl,'std_dev_nl':std_dev_nl}
	
	return(return_dict)


ts_dict={'rp_s_cue':rp_s_cue, 'rp_s_rdelivery':rp_s_rdelivery, 'rp_f_cue':rp_f_cue, 'rp_f_pdelivery':rp_f_pdelivery, 'nrnp_s_cue':nrnp_s_cue, 'nrnp_s_nextreset':nrnp_s_nextreset, 'nrnp_f_cue':nrnp_f_cue, 'nrnp_f_nextreset':nrnp_f_nextreset, 'r_only_s_cue':r_only_s_cue, 'r_only_s_rdelivery':r_only_s_rdelivery, 'r_only_f_cue':r_only_f_cue, 'r_only_f_nextreset':r_only_f_nextreset, 'p_only_s_cue':p_only_s_cue, 'p_only_s_nextreset':p_only_s_nextreset, 'p_only_f_cue':p_only_f_cue, 'p_only_f_pdelivery':p_only_f_pdelivery}

#testing dict
#ts_dict={'rp_s_cue':rp_s_cue, 'r_only_s':r_only_s_rdelivery,'nrnp_s_cue':nrnp_s_cue}
#ts_dict = {'rp_s_rdelivery':rp_s_rdelivery}

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

	
M1_dicts = {'spikes':data_dict['M1_spikes'],'nl':data_dict_nl['M1_spikes_nl']} 
S1_dicts = {'spikes':data_dict['S1_spikes'],'nl':data_dict_nl['S1_spikes_nl']}
PmD_dicts = {'spikes':data_dict['PmD_spikes'],'nl':data_dict_nl['PmD_spikes_nl']}
PmV_dicts = {'spikes':data_dict['PmV_spikes'],'nl':data_dict_nl['PmV_spikes_nl']}

data_dict_all = {'M1_dicts':M1_dicts,'S1_dicts':S1_dicts,'PmD_dicts':PmD_dicts,'PmV_dicts':PmV_dicts}

save_dict = {}
save_dict['data_dict_all'] = data_dict_all
save_dict['param_dict'] = param_dict

#total_rasters = []
#for key, value in ts_dict.iteritems():
#	if not (len(value) == 1 and value[0] == 0):
#		for key2, value2 in data_dict.iteritems():
#		#for key2, value2 in data_dict_nl.iteritems():
#			print 'making rasters %s, %s' %(key, key2)
#			spikes=[]
#
#			if key2 == 'PmV_spikes':
#				for i in range(len(value2)):
#					spikes.append(value2[i])
#			else:
#				for i in range(len(value2)):
#					spikes.append(value2[i]['ts'][0,0][0])			
#				for i in range(len(spikes)):
#					rasters = make_raster(value,spikes,key,key2,i,time_before,time_after)
#
#			#for i in range(len(value2['nl_spike_data'])):
#				#rasters = make_nl_raster(value,value2['nl_spike_data'],key,key2,i,time_before,time_after)
#				#total_rasters.append(rasters)
#			#save_dict['%s_%s' %(key, key2)] = total_rasters
#	else:
#		print 'no instances of event %s, no plot generation' %(key)



total_rasters = []
for key, value in ts_dict.iteritems():
	if not (len(value) == 1 and value[0] == 0):
		for key2, value2 in data_dict_all.iteritems():
			print 'making rasters %s, %s' %(key, key2)

			spikes=[]
			if key2 == 'PmV_dicts':
				for i in range(len(data_dict_all[key2]['spikes'])):
					spikes.append(data_dict_all[key2]['spikes'][i])
			else:
				for i in range(len(data_dict_all[key2]['spikes'])):
					spikes.append(data_dict_all[key2]['spikes'][i]['ts'][0,0][0])			

			for i in range(len(data_dict_all[key2]['spikes'])):
				rasters = make_nl_raster(value,spikes,data_dict_all[key2]['nl']['nl_spike_data'],key,key2,i,time_before,time_after,data_dict_all[key2]['nl']['min_bin'],data_dict_all[key2]['nl']['max_bin'])

				total_rasters.append(rasters)
			save_dict['%s_%s' %(key, key2)] = total_rasters
			total_rasters = []
	else:
		print 'no instances of event %s, no plot generation' %(key)

short_filename = filename[-27:-4]
if short_filename.startswith('0'):
	short_filename = '0%s' %(short_filename)

np.save('avg_fr_and_nlized_data_%s_bin%s' %(short_filename,bin_size),save_dict)
