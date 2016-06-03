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
from os import listdir
from os.path import isfile, join
from os import getcwd
from fnmatch import fnmatch, fnmatchcase
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split


##########################################################
#date_str = '18-12-48-52'
date_str = '18-13-02-45'
#date_str = '29-12-48-19'
#date_str = '29-13-10-44'
#date_str = '01-15-14-23'
#date_str = '01-15-33-52'
#date_str = '19-16-25-20'
#date_str = '19-16-46-25'

bin_size=10
condensed_targs = False
##########################################################

#Load hists
#mypath = getcwd()
#Data=np.load('single_rp_19-16-46-25_hists_0-0.5.npy')

def make_avg_hist(Data):
    S1_c_hists = Data[0]['S1_c_data']
    S1_d_hists = Data[0]['S1_d_data']
    M1_c_hists = Data[0]['M1_c_data']
    M1_d_hists = Data[0]['M1_d_data']
    pmd_c_hists = Data[0]['pmd_c_data']
    pmd_d_hists = Data[0]['pmd_d_data']
    pmv_c_hists = Data[0]['pmv_c_data']
    pmv_d_hists = Data[0]['pmv_d_data']
    targets = Data[0]['targets']
    
    hist_dict = {'S1_c_hists':S1_c_hists,'S1_d_hists':S1_d_hists,'M1_c_hists':M1_c_hists,'M1_d_hists':M1_d_hists,'pmd_c_hists':pmd_c_hists,'pmd_d_hists':pmd_d_hists,'pmv_c_hists':pmv_c_hists,'pmv_d_hists':pmv_d_hists}
    avg_hist_all = {}
    unit_avg_dict = {}  
    classifier_dict = {}

    for key,value in hist_dict.iteritems():
        print key

        #TODO change back if not just looking at all (if:50, else:10)
        bin_size = 50
        
        #reshape into proper format: hist_dict = (timestamp, channel, #bins)
        dummy=np.zeros((value.shape[0],value.shape[1]/bin_size,bin_size))
        for j in range(value.shape[0]):
            for i in range(value.shape[1]/bin_size):
                dummy[j][i]=value[j][i*bin_size:(i+1)*bin_size]
        print dummy.shape
        hist_dict[key]=dummy
        
        #calc average
        hist_sum=np.zeros((dummy.shape[0],1,bin_size))
        unit_avg=np.zeros((dummy.shape[0], dummy.shape[1]))
        for i in range(dummy.shape[0]):
            for j in range(dummy.shape[1]):
                hist_sum[i] = dummy[i][j][:] + hist_sum[i]
                unit_avg[i][j]= np.sum(dummy[i][j][0:bin_size])/bin_size
        avg_hist = hist_sum/j
        #this yields a [timestamp #, 1, 10] ndarray
        avg_hist=avg_hist.reshape(dummy.shape[0],bin_size)  #45=event_num
        avg_hist_all[key]=avg_hist

        #pdb.set_trace()
        #if fnmatch(filename,'*all.npy'):
        bin_size = 10
        unit_all_avg = np.zeros((dummy.shape[0],dummy.shape[1],5))
        for i in range(dummy.shape[0]):
            for j in range(dummy.shape[1]):
                hist_sum[i] = dummy[i][j][:] + hist_sum[i]
                unit_all_avg[i][j][0] = np.sum(dummy[i][j][0:bin_size])/bin_size
                unit_all_avg[i][j][1] = np.sum(dummy[i][j][bin_size+1:bin_size*2])/bin_size
                unit_all_avg[i][j][2] = np.sum(dummy[i][j][bin_size*2+1:bin_size*3])/bin_size
                unit_all_avg[i][j][3] = np.sum(dummy[i][j][bin_size*3+1:bin_size*4])/bin_size
                unit_all_avg[i][j][4] = np.sum(dummy[i][j][bin_size*4+1:bin_size*5])/bin_size
        unit_avg_dict[key] = unit_all_avg
        unit_avg = unit_all_avg

        bin_size = 50
        avg_hist = hist_sum/j
        #this yields a [timestamp #, 1, 10] ndarray
        avg_hist=avg_hist.reshape(dummy.shape[0],bin_size)
        avg_hist_all[key]=avg_hist
        #unit_avg_dict[key] = unit_all_avg
        #unit_avg = unit_all_avg
            
        #############################
        #classsifier shit
        #make sure in proper loop
        ############################
        ##X = value
        ##y = Data[0]['targets']
        ##bdt.fit(X,y)
        ##plot_colors = "br"
        ##plot_step = 0.02
        ##class_names = "AB" #TODO ?
        ##plt.figure(figsize = (10,5))
        ##x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        ##y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        ##xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
         
        ##Z = bdt.predict(np.c_[xx.ravel(),yy.ravel()])
        ##Z = Z.reshape(xx.shape)
        ##cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        ##plt.axis("tight")
            
        x = hist_dict[key]
        #x = value
        y = targets
            
        ##training data (not first/second half because ordered by target). Or look into sorting by ts instead?
        ##odds = x[::2]
        ##print odds.shape
        ##odd_targ = y[::2]
        ###print odd_targ.shape
        ##evens = x[1::2]
        ##even_targ = y[1::2]
           
        #make rewarding targs and punishing targs (ie condensed)
        #rewarding = 0, 4
        #punishing = 1, 7
        #nrnp_s and _f = 2, 3
        #get opposite = 5, 6
        #lump 2, 3, 5, 6 as get nothing  [also, try w/ lumping nrnp_s w/ rewarding, nrnp_f as punishing, etc]

        condensed_targ = np.copy(targets)
        condensed_targ[condensed_targ == 4] = 0  #setting two rewarding types to 0
        condensed_targ[condensed_targ == 7] = 1 #setting two punishing types to 1
        condensed_targ[condensed_targ == 3] = 2
        condensed_targ[condensed_targ == 5] = 2
        condensed_targ[condensed_targ == 6] = 2 #set all rest to 2

        #split into training and testing samples. test_size = proportion of data used for test
        x_train, x_test, y_train, y_test = train_test_split(x, condensed_targ, test_size = .4) 


        #########################
        #ADABoost Classifier
        #########################
        #bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1)

        #bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1.5,algorithm="SAMME")
        #bdt_real.fit(x_train, y_train)
        #bdt_discrete.fit(x_train, y_train)

        #real_test_errors = []
        #discrete_test_errors = []

        #for real_test_predict, discrete_train_predict in zip(bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
            #real_test_errors.append(1. - accuracy_score(real_test_predict, y_test))
            #discrete_test_errors.append(1. - accuracy_score(discrete_train_predict, y_test))

        #n_trees_discrete = len(bdt_discrete)
        #n_trees_real = len(bdt_real)

        # Boosting might terminate early, but the following arrays are always
        # n_estimators long. We crop them to the actual number of trees here:
        #discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
        #real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
        #discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

        #plt.figure(figsize=(15, 5))

        #plt.subplot(131)
        #plt.plot(range(1, n_trees_discrete + 1),discrete_test_errors, c='black', label='SAMME')
        #plt.plot(range(1, n_trees_real + 1),real_test_errors, c='black',linestyle='dashed', label='SAMME.R')
        #plt.legend()
        #plt.ylim(0.18, 0.62)
        #plt.ylabel('Test Error')
        #plt.xlabel('Number of Trees')

        #plt.subplot(132)
        #plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,"b", label='SAMME', alpha=.5)
        #plt.plot(range(1, n_trees_real + 1), real_estimator_errors,"r", label='SAMME.R', alpha=.5)
        #plt.legend()
        #plt.ylabel('Error')
        #plt.xlabel('Number of Trees')
        #plt.ylim((.2,max(real_estimator_errors.max(),discrete_estimator_errors.max()) * 1.2))
        #plt.xlim((-20, len(bdt_discrete) + 20))

        #plt.subplot(133)
        #plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,"b", label='SAMME')
        #plt.legend()
        #plt.ylabel('Weight')
        #plt.xlabel('Number of Trees')
        #plt.ylim((0, discrete_estimator_weights.max() * 1.2))
        #plt.xlim((-20, n_trees_discrete + 20))

        ## prevent overlapping y-axis labels
        #plt.subplots_adjust(wspace=0.25)
        #plt.show()

        ###########################
        
        ##other_targ_odd = other_targ[::2]
        ##other_targ_even = other_targ[1::2]
                
        #pca first?
        #pca = RandomizedPCA(n_components = 100)
        #lda = LDA(n_components = 2)
        #proj = pca.fit_transform(odds)
        #even_proj = pca.fit_transform(evens)
        #z  = lda.fit(proj, odd_targ)
        ##z=lda.transform(evens)
        #z_labels = lda.predict(even_proj)
        #z_prob = lda.predict_proba(even_proj)
         
        #lda2 = LDA(n_components = 2)
        #z2 = lda2.fit(odds, other_targ_odd)
        #z2_labels = lda2.predict(evens)
        #z_prob2 = lda2.predict_proba(evens)

        #temp_dict = {'z_labels':z_labels,'z_prob':z_prob,'odds':odds,'evens':evens,'odd_targ':odd_targ,'even_targ':even_targ,'other_targ_odd':other_targ_odd,'other_targ_even':other_targ_even,'z2_labels':z2_labels,'z_prob2':z_prob2}

        temp_dict = {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test, 'x':x}
        classifier_dict[key] = temp_dict

        ##############################

    return_dict={'avg_hist_all':avg_hist_all,'unit_avg':unit_avg, 'unit_avg_dict':unit_avg_dict, 'classifier_dict':classifier_dict, 'hist_dict_binned':hist_dict}
    #return_dict={'avg_hist_all':avg_hist_all,'unit_avg':unit_avg, 'unit_avg_dict':unit_avg_dict,'hist_dict_binned':hist_dict}
    return return_dict


file_dict = {}

filename = 'single_rp_'+date_str+'_hists_-0.5-0.0.npy'
Data = np.load(filename)
print filename
return_dict = make_avg_hist(Data)
file_dict[filename] = return_dict
#return_dict=make_avg_hist(Data)
#avg_05_00 = return_dict['avg_hist_all']
#unit_avg_05_00 = return_dict['unit_avg']

filename = 'single_rp_'+date_str+'_hists_0-0.5.npy'
Data = np.load(filename)
print filename
return_dict = make_avg_hist(Data)
file_dict[filename] = return_dict
#return_dict=make_avg_hist(Data)
#avg_00_05 = return_dict['avg_hist_all']
#unit_avg_00_05 = return_dict['unit_avg']

#filename = 'single_rp_19-16-46-25_hists_0.5-1.0.npy'
#Data = np.load(filename)
#print filename
#return_dict=make_avg_hist(Data)
#avg_05_10 = return_dict['avg_hist_all']
#unit_avg_05_10 = return_dict['unit_avg']

#filename = 'single_rp_19-16-46-25_hists_1.0-1.5.npy'
#Data = np.load(filename)
#print filename
#return_dict=make_avg_hist(Data)
#avg_10_15 = return_dict['avg_hist_all']
#unit_avg_10_15 = return_dict['unit_avg']

#filename = 'single_rp_19-16-46-25_hists_1.5-2.0.npy'
#Data = np.load(filename)
#print filename
#return_dict=make_avg_hist(Data)
#avg_15_20 = return_dict['avg_hist_all']
#unit_avg_15_20 = return_dict['unit_avg']

filename = 'single_rp_'+date_str+'_hists_all.npy'
Data = np.load(filename)
print filename
return_dict = make_avg_hist(Data)
file_dict[filename] = return_dict


for file_key,file_value in file_dict.iteritems():
    print file_key
    avg_all = file_dict[file_key]['avg_hist_all']
    unit_avg_all = file_dict[file_key]['unit_avg']
    unit_avg_all_dict = file_dict[file_key]['unit_avg_dict']
    classifier_dict = file_dict[file_key]['classifier_dict']

    #avg_all = return_dict['avg_hist_all']
    #unit_avg_all = return_dict['unit_avg']
    #unit_avg_all_dict = return_dict['unit_avg_dict']
    #classifier_dict = return_dict['classifier_dict']
                     
    print 'generating targets'
    targets = Data[0]['targets']
    #print targets
    index=[]
    count=0
    for i in range(len(targets)):
        if targets[i] == count:
            continue
        elif targets[i] == count+1:
            index.append(i)
            count = count + 1
    index.append(len(targets))

    M1_c_total=avg_all['M1_c_hists']
    M1_d_total=avg_all['M1_d_hists']
    S1_c_total=avg_all['S1_c_hists']
    S1_d_total=avg_all['S1_d_hists']
    pmd_c_total=avg_all['pmd_c_hists']
    pmd_d_total=avg_all['pmd_d_hists']
    pmv_c_total=avg_all['pmv_c_hists']
    pmv_d_total=avg_all['pmv_d_hists']

    totals={'M1_c_total':M1_c_total,'M1_d_total':M1_d_total,'S1_c_total':S1_c_total,'S1_d_total':S1_d_total,'pmd_c_total':pmd_c_total,'pmd_d_total':pmd_d_total,'pmv_c_total':pmv_c_total,'pmv_d_total':pmv_d_total}

    if (condensed_targs):
        print "plotting hists"
        for key,value in totals.iteritems():
            print key

            plt.clf()
            fig=plt.figure()
            fig,ax = plt.subplots(4,2,figsize=(8,8))
            plt.title(key+"_hists")
    
            ###### target key #####
            #0: rp_s        rewarding
            #1: rp_f        punishing
            #2: nrnp_s
            #3: nrnp_f
            #4: r_only_s    rewarding
            #5: r_only_f   
            #6: p_only_s   
            #7: p_only f    punishing
            
            plt.subplot(4,2,1)
            plt.imshow(value[0:index[0]-1],interpolation='nearest')
            plt.title("rp_s")
    
            plt.subplot(4,2,2)
            plt.imshow(value[index[0]:index[1]-1],interpolation='nearest')
            plt.title("rp_f")
        
            plt.subplot(4,2,3)
            plt.imshow(value[index[1]:index[2]-1],interpolation='nearest')
            plt.title("nrnp_s")

            plt.subplot(4,2,4)
            plt.imshow(value[index[2]:index[3]-1],interpolation='nearest')
            plt.title("nrnp_f")
        
            plt.subplot(4,2,5)
            plt.imshow(value[index[3]:index[4]-1],interpolation='nearest')
            plt.title("r_only_s")
        
            plt.subplot(4,2,6)
            plt.imshow(value[index[4]:index[5]-1],interpolation='nearest')
            plt.title("r_only_f")
            
            plt.subplot(4,2,7)
            plt.imshow(value[index[5]:index[6]-1],interpolation='nearest')
            plt.title("p_only_s")
        
            plt.subplot(4,2,8)
            plt.imshow(value[index[6]:index[7]-1],interpolation='nearest')
            plt.title("p_only_f")
    
            plt.savefig(filename[0:-4]+"_"+key+".png")
            #plt.show()
    
            plt.clf()

    #####################################################
    #####################################################
    
    unit_rank_dict = {}
    for key2,unit_avg_all in unit_avg_all_dict.iteritems():

        ####### reward and punishment indices #####
        print "calculating reward and punishment indices for %s" %(key2)
        pre_post_fr_dif = unit_avg_all[:,:,1] - unit_avg_all[:,:,0]
        #TODO arbitrary cutoff for inc/dec, l/u
        cutoff = 0.1

        inc_fr = np.where(pre_post_fr_dif > cutoff)[0]
        dec_fr = np.where(pre_post_fr_dif < -1*cutoff)[0]
        fr_index = np.zeros((pre_post_fr_dif.shape))
        fr_index[pre_post_fr_dif > cutoff] = 1
        fr_index[pre_post_fr_dif < -1*cutoff] = -1
        
        inc_perc_all = float(sum(sum(pre_post_fr_dif > cutoff)))/(pre_post_fr_dif.shape[0] * pre_post_fr_dif.shape[1])
        dec_perc_all = float(sum(sum(pre_post_fr_dif < -1*cutoff)))/(pre_post_fr_dif.shape[0] * pre_post_fr_dif.shape[1])
        no_change_perc_all = 1-inc_perc_all-dec_perc_all

        if condensed_targs:
            rewarding_index1 = unit_avg_all[0:index[0]-1,:,:]
            rewarding_index2 = unit_avg_all[index[3]:index[4]-1,:,:]
            all_rewarding_index = np.vstack((rewarding_index1, rewarding_index2))
        
            punishing_index1 = unit_avg_all[index[0]:index[1]-1,:,:]
            punishing_index2 = unit_avg_all[index[6]:index[7]-1,:,:]
            all_punishing_index = np.vstack((punishing_index1, punishing_index2))
        
            nrnp_index1 = unit_avg_all[index[1]:index[2]-1,:,:]
            nrnp_index2 = unit_avg_all[index[2]:index[3]-1,:,:]
            nrnp_index3 = unit_avg_all[index[4]:index[5]-1,:,:]
            nrnp_index4 = unit_avg_all[index[5]:index[6]-1,:,:]
            all_nrnp_index = np.vstack((nrnp_index1, nrnp_index2, nrnp_index3, nrnp_index4))
            seperated_fr_dict = {'all_rewarding_index':all_rewarding_index,'all_punishing_index':all_punishing_index,'all_nrnp_index':all_nrnp_index}
        
            print 'condensed targets'

        else:
            #TODO make into dict then dictionary? more efficient than below
            rp_s_unit_avg = unit_avg_all[0:index[0]-1,:,:]
            rp_s_fr_index = fr_index[0:index[0]-1]
            rp_f_unit_avg = unit_avg_all[index[0]:index[1]-1,:,:]
            rp_f_fr_index = fr_index[index[0]:index[1]-1]
            nrnp_s_unit_avg = unit_avg_all[index[1]:index[2]-1,:,:]
            nrnp_s_fr_index = fr_index[index[1]:index[2]-1]
            nrnp_f_unit_avg = unit_avg_all[index[2]:index[3]-1,:,:]
            nrnp_f_fr_index = fr_index[index[2]:index[3]-1]
            r_only_s_unit_avg = unit_avg_all[index[3]:index[4]-1,:,:]
            r_only_s_fr_index = fr_index[index[3]:index[4]-1]
            r_only_f_unit_avg = unit_avg_all[index[4]:index[5]-1,:,:]
            r_only_f_fr_index = fr_index[index[4]:index[5]-1]
            p_only_s_unit_avg = unit_avg_all[index[5]:index[6]-1,:,:]
            p_only_s_fr_index = fr_index[index[5]:index[6]-1]
            p_only_f_unit_avg = unit_avg_all[index[6]:index[7]-1,:,:]
            p_only_f_fr_index = fr_index[index[6]:index[7]-1]
            seperated_fr_dict = {'rp_s_unit_avg':rp_s_unit_avg,'rp_f_unit_avg':rp_f_unit_avg,'nrnp_s_unit_avg':nrnp_s_unit_avg,'nrnp_f_unit_avg':nrnp_f_unit_avg,'r_only_s_unit_avg':r_only_s_unit_avg,'r_only_f_unit_avg':r_only_f_unit_avg,'p_only_s_unit_avg':p_only_s_unit_avg,'p_only_f_unit_avg':p_only_f_unit_avg}

        avgs_over_ts = {}
        fr_indices = {}
        for key3,value in seperated_fr_dict.iteritems():
            shape = value.shape
            #for each individual unit, averaging over ts's
            #print 'averaging %s' %(key3)
            avg_over_ts = np.zeros((shape[1],5))
            for i in range(shape[1]):
                for j in range(shape[0]):
                    avg_over_ts[i][:] = avg_over_ts[i][:] + value[j][i][:]
            avg_over_ts = avg_over_ts / (i + 1)
            fr_ind = np.zeros((avg_over_ts.shape[0]))
            fr_diff = avg_over_ts[:,1] - avg_over_ts[:,0]
            fr_ind[fr_diff > cutoff] = 1
            fr_ind[fr_diff < -1*cutoff] = -1
            #print fr_ind
    
            avgs_over_ts[key3+'s'] = avg_over_ts
            fr_indices[key3+'_fr_indices'] = fr_ind

        #### TODO look into when to calc r/p indices. Now first 500ms post-cue/delivery. "neutral" is nrnp_s/f based on if rew or punishment index
        #TODO also doesn't work if not enough p_only_f trials
    
        if not condensed_targs:
            #r index = (r-n)/(r+n)
            numer = avgs_over_ts['r_only_s_unit_avgs'][:,1] - avgs_over_ts['nrnp_s_unit_avgs'][:,1]
            denom = avgs_over_ts['r_only_s_unit_avgs'][:,1] + avgs_over_ts['nrnp_s_unit_avgs'][:,1]
            reward_index = np.divide(numer, denom)

            #p index = (p-n)/(p+n)
            numer2 = avgs_over_ts['p_only_f_unit_avgs'][:,1] - avgs_over_ts['nrnp_f_unit_avgs'][:,1]
            denom2 = avgs_over_ts['p_only_f_unit_avgs'][:,1] + avgs_over_ts['nrnp_f_unit_avgs'][:,1]
            punishment_index = np.divide(numer2, denom2)

            #modified r index
            numer3 = (avgs_over_ts['rp_s_unit_avgs'][:,1] + avgs_over_ts['r_only_s_unit_avgs'][:,1]) - (avgs_over_ts['nrnp_s_unit_avgs'][:,1] + avgs_over_ts['p_only_s_unit_avgs'][:,1]) 
            denom3 = (avgs_over_ts['rp_s_unit_avgs'][:,1] + avgs_over_ts['r_only_s_unit_avgs'][:,1]) + (avgs_over_ts['nrnp_s_unit_avgs'][:,1] + avgs_over_ts['p_only_s_unit_avgs'][:,1]) 
            alt_reward_index = np.divide(numer3, denom3)
        
            #modified p index
            numer4 = (avgs_over_ts['rp_f_unit_avgs'][:,1] + avgs_over_ts['p_only_f_unit_avgs'][:,1]) - (avgs_over_ts['nrnp_f_unit_avgs'][:,1] + avgs_over_ts['r_only_f_unit_avgs'][:,1]) 
            denom4 = (avgs_over_ts['rp_f_unit_avgs'][:,1] + avgs_over_ts['p_only_f_unit_avgs'][:,1]) + (avgs_over_ts['nrnp_f_unit_avgs'][:,1] + avgs_over_ts['r_only_f_unit_avgs'][:,1]) 
            alt_punishment_index = np.divide(numer4, denom4)

        else:
            numer = avgs_over_ts['all_rewarding_indexs'][:,1] - avgs_over_ts['all_nrnp_indexs'][:,1] 
            denom = avgs_over_ts['all_rewarding_indexs'][:,1] + avgs_over_ts['all_nrnp_indexs'][:,1] 
            reward_index = np.divide(numer,denom)
            
            numer2 = avgs_over_ts['all_punishing_indexs'][:,1] - avgs_over_ts['all_nrnp_indexs'][:,1] 
            denom2 = avgs_over_ts['all_punishing_indexs'][:,1] + avgs_over_ts['all_nrnp_indexs'][:,1] 
            punishment_index = np.divide(numer2, denom2)

    
        #TODO look into rp_s/f to compare if indices are chnaging if both cues are present, or just look at all rewarding and all punishing trials
        #TODO why nans? removed below
        bottom_10_rewarding_ch = np.zeros((10))
        top_10_rewarding_ch = np.zeros((10))
        bottom_10_punishing_ch = np.zeros((10))
        top_10_punishing_ch = np.zeros((10))

        reward_index = reward_index[~np.isnan(reward_index)]
        sorted_reward_arg = np.argsort(np.argsort(reward_index))
        punishment_index = punishment_index[~np.isnan(punishment_index)]
        sorted_punishment_arg = np.argsort(np.argsort(punishment_index))

        alt_reward_index = alt_reward_index[~np.isnan(alt_reward_index)]
        alt_punishment_index = alt_punishment_index[~np.isnan(alt_punishment_index)]
        
        try:
            perc_r_pos = len(np.where(reward_index > cutoff)[0])/float(len(reward_index))
            perc_r_neg = len(np.where(reward_index < -1*cutoff)[0])/float(len(reward_index))
            perc_p_pos = len(np.where(punishment_index > cutoff)[0])/float(len(punishment_index))
            perc_p_neg = len(np.where(punishment_index < -1*cutoff)[0])/float(len(punishment_index))
            perc_r_null = 1 - perc_r_pos - perc_r_neg
            perc_p_null = 1 - perc_p_pos - perc_p_neg
        
            perc_posneg = {'perc_r_pos':perc_r_pos,'perc_r_neg':perc_r_neg,'perc_p_pos':perc_p_pos,'perc_p_neg':perc_p_neg,'perc_r_null':perc_r_null,'perc_p_null':perc_p_null}
        except:
            print 'error computing percentages, dividing by zero'
            perc_posneg = {}
			
        #should give channels of units that respond most and least to rewarding stimulus
        try:
            for i in range (0,10):
                bottom_10_rewarding_ch[i] = np.where(sorted_reward_arg == i)[0][0]
                bottom_10_punishing_ch[i] = np.where(sorted_punishment_arg == i)[0][0]
                top_10_rewarding_ch[i] = np.where(sorted_reward_arg == (len(sorted_reward_arg) - (i+1)))[0][0]
                top_10_punishing_ch[i] = np.where(sorted_punishment_arg == (len(sorted_punishment_arg) -(i+1)))[0][0]
        except:
            print 'error calc top/bottom 10 rewarding ch'
                
        temp_dict = {}
        temp_dict = {'top_10_rewarding_ch':top_10_rewarding_ch,'bottom_10_rewarding_ch':bottom_10_rewarding_ch,'top_10_punishing_ch':top_10_punishing_ch,'bottom_10_punishing_ch':bottom_10_punishing_ch,'reward_index':reward_index,'sorted_reward_arg':sorted_reward_arg,'punishment_index':punishment_index,'sorted_punishment_arg':sorted_punishment_arg,'seperated_fr_dict':seperated_fr_dict,'classifier_dict':classifier_dict,'perc_posneg':perc_posneg,'alt_reward_index':alt_reward_index,'alt_punishment_index':alt_punishment_index}
        #temp_dict = {'top_10_rewarding_ch':top_10_rewarding_ch,'bottom_10_rewarding_ch':bottom_10_rewarding_ch,'top_10_punishing_ch':top_10_punishing_ch,'bottom_10_punishing_ch':bottom_10_punishing_ch,'reward_index':reward_index,'sorted_reward_arg':sorted_reward_arg,'punishment_index':punishment_index,'sorted_punishment_arg':sorted_punishment_arg,'seperated_fr_dict':seperated_fr_dict,'perc_posneg':perc_posneg}

        unit_rank_dict[key2] = temp_dict

    for key,value in unit_avg_all_dict.iteritems():          #totals.iteritems():
        print 'calculating unit ranks for %s' %(key)
        #key = 'S1_c_hists'

        hist_binned = return_dict['hist_dict_binned'][key]

        total_unit_avg = unit_avg_all_dict[key]
        top_r = unit_rank_dict[key]['top_10_rewarding_ch']
        btm_r = unit_rank_dict[key]['bottom_10_rewarding_ch']
        top_p = unit_rank_dict[key]['top_10_punishing_ch']
        btm_p = unit_rank_dict[key]['bottom_10_punishing_ch']

        top_r_hists = np.zeros((total_unit_avg.shape[0],10,50))
        btm_r_hists = np.zeros((total_unit_avg.shape[0],10,50))
        top_p_hists = np.zeros((total_unit_avg.shape[0],10,50))
        btm_p_hists = np.zeros((total_unit_avg.shape[0],10,50))

        for i in range(10):
            top_r_hists[:,i,:] = hist_binned[:,top_r[i],:]
            btm_r_hists[:,i,:] = hist_binned[:,btm_r[i],:]
            top_p_hists[:,i,:] = hist_binned[:,top_p[i],:]
            btm_p_hists[:,i,:] = hist_binned[:,btm_p[i],:]

            top_btm_dict = {'top_r_hists':top_r_hists,'btm_r_hists':btm_r_hists,'top_p_hists':top_p_hists,'btm_p_hists':btm_p_hists}

        tb_dict = {}
        collapsed_dict={}
        for tb_key,rank_hist in top_btm_dict.iteritems():
            tb_dummy = np.zeros((10,50))
            temp = 0
    
            #for i in range(rank_hist.shape[0]):
            for i in range(10):
                for j in range(50):
                    for k in range(rank_hist.shape[0]):
                        temp = temp + rank_hist[k][i][j] 
                
                    tb_dummy[i][j] = temp         #rank_hist[k][i][j] + tb_dummy[i][j]
                    #print temp
                    temp = 0

            tb_dummy = tb_dummy / rank_hist.shape[0]

            collapsed = np.zeros((50))
            for i in range(50):
                collapsed[i] = np.sum(tb_dummy[:,i])

            tb_dict[tb_key] = tb_dummy
            collapsed_dict[tb_key] = collapsed
    
        plt.clf()
        fig=plt.figure()
        fig,ax = plt.subplots(4,2,figsize=(8,8))
        plt.title(key+"_top_bottom_hists")

        plt.subplot(4,2,1)
        plt.imshow(tb_dict['top_r_hists'],interpolation='nearest')
        plt.title("top 10 rewarding")
    
        plt.subplot(4,2,2)
        plt.imshow(tb_dict['btm_r_hists'],interpolation='nearest')
        plt.title("bottom 10 rewarding")
        
        plt.subplot(4,2,3)
        plt.imshow(tb_dict['top_p_hists'],interpolation='nearest')
        plt.title("top 10 punishing")
    
        plt.subplot(4,2,4)
        plt.imshow(tb_dict['btm_p_hists'],interpolation='nearest')
        plt.title("bottom 10 punishing")
        
        plt.subplot(4,2,5)
        plt.plot(collapsed_dict['top_r_hists'])
        plt.plot(collapsed_dict['btm_r_hists'])
        plt.title("top and bottom r hists")

        plt.subplot(4,2,6)
        plt.plot(collapsed_dict['top_p_hists'])
        plt.plot(collapsed_dict['btm_p_hists'])
        plt.title("top and bottom p hists")

        try:
            plt.subplot(4,2,7)
            #plt.hist(unit_rank_dict[key]['reward_index'])
            #plt.title('reward index histogram')
            plt.hist(unit_rank_dict[key]['alt_reward_index'])
            plt.title('alt reward index histogram')
        except:
            print 'unable to plot reward index, likely zero'

        try:
            plt.subplot(4,2,8)
            #plt.hist(unit_rank_dict[key]['punishment_index'])
            #plt.title('punishment index histogram')
            plt.hist(unit_rank_dict[key]['alt_punishment_index'])
            plt.title('alt punishment index histogram')
        except:
            print 'unable to plot punishment index, likely zero'
            
        if condensed_targs:
            plt.savefig(file_key[0:-4]+"_condensed_"+key+"_mixed.png")
        else:
            plt.savefig(file_key[0:-4]+"_not_condensed_"+key+"_mixed.png")
        plt.clf()

    if condensed_targs:
        total = {'unit_rank_dict':unit_rank_dict,'top_btm_dict':top_btm_dict,'tb_dict':tb_dict,'collapsed_dict':collapsed_dict}
        np.save('hist_analysis_condensed_'+file_key[0:-4],total)
    else:
        total = {'unit_rank_dict':unit_rank_dict,'top_btm_dict':top_btm_dict,'tb_dict':tb_dict}
        np.save('hist_analysis_not_condensed_'+file_key[0:-4],total)
