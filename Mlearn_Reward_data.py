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

#-----------Pull neural data in Extracted_ Form----------------

#root = tk.Tk()
#root.withdraw()
#filename = tkFileDialog.askopenfilename();

filename = '/Volumes/ADATA_SH93/nhp_data/20151026_504/sorted_files/matfiles/Extracted_504_2015-10-26-14-03-03.mat'
print filename
a = sio.loadmat(filename);

#Pull reward timestamps. Separate timespamp file must exist in folder containing neural data, with same name+_timestamps
print filename[:-4]+"_timestamps"+filename[-4:]
Timestamps = sio.loadmat(filename[:-4]+"_timestamps"+filename[-4:]);
#Timestamp generation: generates arrays of timestamps depending on reward level, including offset for cue presentation and reward delivery
zero_reward_delivery = Timestamps['zero_reward_Reward'][0];
zero_reward_cue = Timestamps['zero_reward_cue'][0];
one_reward_delivery = Timestamps['one_reward_Reward'][0]+.50;
one_reward_cue = Timestamps['one_reward_cue'][0]+1.0;
two_reward_delivery = Timestamps['two_reward_Reward'][0]+1.0;
two_reward_cue = Timestamps['two_reward_cue'][0]+2.0;
three_reward_delivery = Timestamps['three_reward_Reward'][0]+1.5;
three_reward_cue = Timestamps['three_reward_cue'][0]+3.0;

#Pull neural spikes

Spikes = a['neural_data']['spikeTimes'];
#Break spikes into M1 S1 pmd 
M1_spikes = Spikes[0,0][0,0];
PmD_spikes = Spikes[0,0][0,1];
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
    M1_unit_names.append(M1_spikes[0,i]['signame'][0][0][0])
dummy = [];
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
    pmd_unit_names.append(PmD_spikes[0,i]['signame'][0][0][0])
for i in range(PmD_limit,PmD_spikes.shape[1]):
    dummy.append(PmD_spikes[0,i]['ts'][0,0][0])
#Generate Pmv Spiking data
PmV_spikes = np.array(dummy);
unit_names['pmd_unit_names']=pmd_unit_names
#PSTH input parameters
no_bins = 15;

#trim M1,S1,and PmD spikes to remove PmV spikes trailing at end; nerual data pruning now complete
M1_spikes = M1_spikes[0,0:M1_limit];
S1_spikes = S1_spikes[0,0:S1_limit];
PmD_spikes = PmD_spikes[0,0:PmD_limit];
#generate time intervals of interest for PSTH generation
time_boundry={'0-2.5':[0.0,2.5],'2.5-5.0':[2.5,5.0],'0-1.5':[0.0,1.5],'1.5-3.5':[1.5,3.5],}

#Begin main loop of analysis
for name_of_bin,time_of_bin in time_boundry.iteritems():
 before_time = -time_of_bin[0]
 after_time = time_of_bin[1]
 #Build and save M1 PSTH

 #Function that builds histograms of M1,S1,PmD spikes based on parameters entered above
 def Build_hist(data,timestamps,no_bins,time_before,time_after):
     dummy = []
     for i in range(0,timestamps.shape[0]):
         dummy1=[]
         b = timestamps[i]
         for ii in range(data.shape[0]):
             a = data[ii]['ts'][0,0][0]
             c = (np.histogram(a[np.where(np.logical_and(a>=b-time_before,a<=b+time_after))],bins=no_bins,normed=False,density=False))
             dummy1.append(np.nan_to_num(c[0]))
         dummy.append(MaxAbsScaler().fit_transform(dummy1))
     return np.array(dummy)

 #Build PmV histograms
 def Build_hist_PmV(data,timestamps,no_bins,time_before,time_after):
     dummy = []
     for i in range(0,timestamps.shape[0]):
         dummy1=[]
         b = timestamps[i]
         for ii in range(data.shape[0]):
             a = data[ii]
             c = (np.histogram(a[np.where(np.logical_and(a>=b-time_before,a<=b+time_after))],bins=no_bins,normed=False,density=False))
             dummy1.append(np.nan_to_num(c[0]))
         dummy.append(MaxAbsScaler().fit_transform(dummy1))
     return np.array(dummy)

 #Assemble a final dictionary of the final dataset for saving
 def Make_final_data(data,start_position):
         dummy = []
         for i in range(0,data[start_position].shape[0]):
             dummy.append(np.hstack(data[start_position][i]))
         Zr_data = np.array(dummy)
         Zr_targets = np.ones(Zr_data.shape[0])*0
         dummy = []
         for i in range(0,data[start_position+8].shape[0]):
             dummy.append(np.hstack(data[start_position+8][i]))
         Or_data = np.array(dummy)
         Or_targets = np.ones(Or_data.shape[0])*1
         dummy = []
         for i in range(0,data[start_position+16].shape[0]):
             dummy.append(np.hstack(data[start_position+16][i]))
         Tr_data = np.array(dummy)
         Tr_targets = np.ones(Tr_data.shape[0])*2
         dummy = []
         for i in range(0,data[start_position+24].shape[0]):
             dummy.append(np.hstack(data[start_position+24][i]))
         Thr_data = np.array(dummy)
         Thr_targets = np.ones(Thr_data.shape[0])*3
         return (np.vstack([Zr_data,Or_data,Tr_data,Thr_data]),np.hstack([Zr_targets,Or_targets,Tr_targets,Thr_targets]))

 
 #Zero Reward 
 M1_zero_reward_hists = Build_hist(M1_spikes,zero_reward_delivery,no_bins,before_time,after_time)
 S1_zero_reward_hists = Build_hist(S1_spikes,zero_reward_delivery,no_bins,before_time,after_time)
 PmD_zero_reward_hists = Build_hist(PmD_spikes,zero_reward_delivery,no_bins,before_time,after_time)
 PmV_zero_reward_hists = Build_hist_PmV(PmV_spikes,zero_reward_delivery,no_bins,before_time,after_time)
 M1_zero_reward_cue_hists = Build_hist(M1_spikes,zero_reward_cue,no_bins,before_time,after_time)
 S1_zero_reward_cue_hists = Build_hist(S1_spikes,zero_reward_cue,no_bins,before_time,after_time)
 PmD_zero_reward_cue_hists = Build_hist(PmD_spikes,zero_reward_cue,no_bins,before_time,after_time)
 PmV_zero_reward_cue_hists = Build_hist_PmV(PmV_spikes,zero_reward_cue,no_bins,before_time,after_time)

 #One Reward 
 M1_one_reward_hists = Build_hist(M1_spikes,one_reward_delivery,no_bins,before_time,after_time)
 S1_one_reward_hists = Build_hist(S1_spikes,one_reward_delivery,no_bins,before_time,after_time)
 PmD_one_reward_hists = Build_hist(PmD_spikes,one_reward_delivery,no_bins,before_time,after_time)
 PmV_one_reward_hists = Build_hist_PmV(PmV_spikes,one_reward_delivery,no_bins,before_time,after_time)
 M1_one_reward_cue_hists = Build_hist(M1_spikes,one_reward_cue,no_bins,before_time,after_time)
 S1_one_reward_cue_hists = Build_hist(S1_spikes,one_reward_cue,no_bins,before_time,after_time)
 PmD_one_reward_cue_hists = Build_hist(PmD_spikes,one_reward_cue,no_bins,before_time,after_time)
 PmV_one_reward_cue_hists = Build_hist_PmV(PmV_spikes,one_reward_cue,no_bins,before_time,after_time)

 #Two Reward 
 M1_two_reward_hists = Build_hist(M1_spikes,two_reward_delivery,no_bins,before_time,after_time)
 S1_two_reward_hists = Build_hist(S1_spikes,two_reward_delivery,no_bins,before_time,after_time)
 PmD_two_reward_hists = Build_hist(PmD_spikes,two_reward_delivery,no_bins,before_time,after_time)
 PmV_two_reward_hists = Build_hist_PmV(PmV_spikes,two_reward_delivery,no_bins,before_time,after_time)
 M1_two_reward_cue_hists = Build_hist(M1_spikes,two_reward_cue,no_bins,before_time,after_time)
 S1_two_reward_cue_hists = Build_hist(S1_spikes,two_reward_cue,no_bins,before_time,after_time)
 PmD_two_reward_cue_hists = Build_hist(PmD_spikes,two_reward_cue,no_bins,before_time,after_time)
 PmV_two_reward_cue_hists = Build_hist_PmV(PmV_spikes,two_reward_cue,no_bins,before_time,after_time)

 #Three Reward 
 M1_three_reward_hists = Build_hist(M1_spikes,three_reward_delivery,no_bins,before_time,after_time)
 S1_three_reward_hists = Build_hist(S1_spikes,three_reward_delivery,no_bins,before_time,after_time)
 PmD_three_reward_hists = Build_hist(PmD_spikes,three_reward_delivery,no_bins,before_time,after_time)
 PmV_three_reward_hists = Build_hist_PmV(PmV_spikes,three_reward_delivery,no_bins,before_time,after_time)
 M1_three_reward_cue_hists = Build_hist(M1_spikes,three_reward_cue,no_bins,before_time,after_time)
 S1_three_reward_cue_hists = Build_hist(S1_spikes,three_reward_cue,no_bins,before_time,after_time)
 PmD_three_reward_cue_hists = Build_hist(PmD_spikes,three_reward_cue,no_bins,before_time,after_time)
 PmV_three_reward_cue_hists = Build_hist_PmV(PmV_spikes,three_reward_cue,no_bins,before_time,after_time)
 
 #This list is a consortium of all of the reward histograms, both cue and reward for each region at each level of reward
 pnt_data = [M1_zero_reward_hists,S1_zero_reward_hists,PmD_zero_reward_hists,PmV_zero_reward_hists,M1_zero_reward_cue_hists,S1_zero_reward_cue_hists,PmD_zero_reward_cue_hists,PmV_zero_reward_cue_hists,M1_one_reward_hists,S1_one_reward_hists,PmD_one_reward_hists,PmV_one_reward_hists,M1_one_reward_cue_hists,S1_one_reward_cue_hists,PmD_one_reward_cue_hists,PmV_one_reward_cue_hists,M1_two_reward_hists,S1_two_reward_hists,PmD_two_reward_hists,PmV_two_reward_hists,M1_two_reward_cue_hists,S1_two_reward_cue_hists,PmD_two_reward_cue_hists,PmV_two_reward_cue_hists,M1_three_reward_hists,S1_three_reward_hists,PmD_three_reward_hists,PmV_three_reward_hists,M1_three_reward_cue_hists,S1_three_reward_cue_hists,PmD_three_reward_cue_hists,PmV_three_reward_cue_hists]

 #Generates final data dictionary for each region based on pnt_data; _d_ is following reward delivery and _c_ is following cue
 M1_d_data,targets=Make_final_data(pnt_data,0)
 S1_d_data,targets=Make_final_data(pnt_data,1)
 pmd_d_data,targets=Make_final_data(pnt_data,2)
 pmv_d_data,targets = Make_final_data(pnt_data,3)

 M1_c_data,targets=Make_final_data(pnt_data,4)
 S1_c_data,targets=Make_final_data(pnt_data,5)
 pmd_c_data,targets=Make_final_data(pnt_data,6)
 pmv_c_data,targets = Make_final_data(pnt_data,7) 
 
 #Construct dictionary for saving array
 final_data = {'M1_d_data':M1_d_data,'S1_d_data':S1_d_data,'pmd_d_data':pmd_d_data,'pmv_d_data':pmv_d_data,'M1_c_data':M1_c_data,'S1_c_data':S1_c_data,'pmd_c_data':pmd_c_data,'pmv_c_data':pmv_c_data,'targets':targets}

 #Construct temparary dictionary for figure generation
 final_data_no_targets = {'M1 at Delivery':M1_d_data,'S1 at Delivery':S1_d_data,'PmD at Delivery':pmd_d_data,'PmV at Delivery':pmv_d_data,'M1 at Cue':M1_c_data,'S1 at Cue':S1_c_data,'PmD at Cue':pmd_c_data,'PmV at Cue':pmv_c_data}

 np.save("multi_reward"+filename[-15:-4]+"_hists_"+name_of_bin,(final_data,unit_names))

 #Perform PCA on PSTH followed By LDA on PCA transform of PSTH data and save figure showing results for each bin
 for key,value in final_data_no_targets.iteritems():
     print key
     lda = LDA(n_components=2)
     pca = RandomizedPCA(n_components=20)
     proj = pca.fit_transform(value)
     proj = lda.fit_transform(proj,targets)
     print proj.shape
     plt.clf()
     plt.scatter(proj[:, 0], proj[:, 1], c=targets)
     plt.title(key+" from "+name_of_bin)
     plt.xlabel("LD1")
     plt.ylabel("LD2")
     plt.colorbar()
     plt.savefig(key+" from "+name_of_bin+"s.png")
     plt.clf()
     



                                                                                
