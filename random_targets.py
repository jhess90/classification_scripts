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
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import manifold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


#mypath = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20150928_504/figs/block1/200_pca/test_random'

mypath = getcwd()

#set params
randomize = False
plotting = False
compare = True
#TODO randomize_vals = True

pca_num = 200
lda_num = 4


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for file in onlyfiles:
	if file[-3:]!='npy':
		continue
	print file
	Data = np.load(file)
	name_of_bin = file[-10:-4]
	print name_of_bin
 
	for key,data in Data[0].iteritems():
		print key
		if key == 'targets':
			targets = targets
			continue
		len_targets=len(data)

		#value=data
		
		if randomize:
			targets=np.random.randint(0,7,len_targets)
		else:
			#TODO prob not going to work
			targets = np.hstack(np.array(Data[0]['targets']))
         
		if compare:

			#TODO would it be better to do odds and evens? First and second half might split up targets into 0-3 and 4-7, not evenly
			#also, why this first/second half and then next one? only need second one then redo datavstack?
			#first_half = np.hstack(data[0:len(data)/2])
			#first_half = np.vstack(first_half)
			#second_half = np.hstack(data[len(data)/2:-1])
			#second_half = np.vstack(second_half)
			#data = np.vstack([first_half,second_half])

			
			
			#data=data.reshape(len(data),1)
			#targets=targets.reshape(len(targets),1)
			#targets=np.ones((len(data),1))

			#TODO ok so targets has to be length of data...whats up now
			#targets=np.random.randint(0,7,len(data))

			targets=Data[0]['targets']
			#targets=targets.reshape(len(targets),1)

			#targets=np.random.randint(0,7,len(targets))
			
			proj=[]
			lda = LDA(n_components=lda_num)
			pca = RandomizedPCA(n_components=pca_num)
			classifier = KNeighborsClassifier(8)
			proj = pca.fit_transform(data)
			proj = lda.fit_transform(proj,targets)

			first_half = proj[0:proj.shape[0]/2]
			first_targets = targets[0:targets.shape[0]/2]
			second_half = proj[proj.shape[0]:proj.shape[0]]
			second_targets = targets[targets.shape[0]/2:targets.shape[0]]

			odds = proj[::2]
			odds_targets = targets[::2]
			evens = proj[1::2]
			evens_targets = targets[1::2]


			#x_train, x_test, y_train, y_test=train_test_split(first_half,first_targets)
			x_train,x_test,y_train,y_test=train_test_split(odds,odds_targets)
			classifier.fit(x_train,y_train)
			expected = y_test
			predicted = classifier.predict(x_test)
			print(file[:-4])
			print("Classification report for %s classifier %s, odds:\n%s" %(key,classifier,classification_report(expected,predicted)))

			with open("classifier_log.txt","a") as myfile:
				myfile.write(file+"\n")
				myfile.write(key+"\n")
				myfile.write("Classification report for %s classifier, odds %s:\n%s" %(key,classifier,classification_report(expected,predicted)))

			alpha = 0.01
			lasso=Lasso(alpha=alpha)
			y_pred_lasso = lasso.fit(x_train,y_train).predict(x_test)
			r2_score_lasso = r2_score(y_test,y_pred_lasso)
			print(lasso)
			print("r^2 on test data: %f" % r2_score_lasso)
			with open("classifier_log.txt","a") as myfile:
				myfile.write(file+"\n")
				myfile.write("r^2 on test data: %f" %r2_score_lasso)

			x_train,x_test,y_train,y_test = train_test_split(evens,evens_targets)
			classifier.fit(x_train,y_train)
			expected = y_test
			predicted = classifier.predict(x_test)
			print(file[:-4])
			print("Classification report for %s classifier %s, evens:\n%s" %(key,classifier,classification_report(expected,predicted)))

			with open("classifier_log.txt","a") as myfile:
				myfile.write(file+"\n")
				myfile.write(key+"\n")
				myfile.write("Classification report for %s classifier %s, evens:\n%s" %(key,classifier,classification_report(expected,predicted)))

			classifier.fit(odds,odds_targets)
			expected=evens_targets
			predicted=classifier.predict(evens)
			print(file[:-4])
			print("Classification report for %s odd classifier %s, using odd data as training:\n%s" %(key,classifier,classification_report(expected,predicted)))

			with open("classifier_log.txt","a") as myfile:
				myfile.write(file+"\n")
				myfile.write(key+"\n")
				myfile.write("Classification report for %s classifier %s, using odd data as training:\n%s" %(key,classifier,classification_report(expected,predicted)))





				
		if plotting:
			lda = LDA(n_components=lda_num)
			pca = RandomizedPCA(n_components=pca_num)
			#pca_num = RandomizedPCA(n_components=100)
			proj = pca.fit_transform(data)
			proj = lda.fit_transform(proj,targets)
			print proj.shape

			plt.clf()
			figs=plt.figure()
			fig, ax = plt.subplots(1,1,figsize=(8,8))
			plt.title(key+" from "+name_of_bin)
			plt.xlabel("LD1")
			plt.ylabel("LD2")
        
			cmap=plt.cm.nipy_spectral
			cmaplist=[cmap(i) for i in range(cmap.N)]
			cmap=cmap.from_list('Custom cmap',cmaplist,cmap.N)
			bounds=np.linspace(0,8,9)
			ticks=bounds + 0.5
			norm=colors.BoundaryNorm(bounds, cmap.N)

			scatter=ax.scatter(proj[:,0],proj[:,1],c=targets,cmap=cmap,norm=norm)
			ax2=fig.add_axes([0.7,0.1,0.03,0.7])       
			cb=colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,spacing='proportional',ticks=bounds,boundaries=bounds,format='%1i')
			cb.set_ticks(ticks)
			ticklabels=np.arange(0,8,1)
			cb.set_ticklabels(ticklabels)
			ax2.set_title=('category')
			plt.figtext(0.8,0.4,'0: rp_s\n1: rp_f\n2: nrnp_s\n3: nrnp_f\n4: r_only_s\n5: r_only_f\n6: p_only_s\n7: p_only_f', bbox=dict(facecolor='white'))
			fig.subplots_adjust(right=0.7)
		 
			if randomize:
				plt.savefig(key+"_from_"+name_of_bin+"s_random.png")
			else:
				plt.savefig(key+"_from_"+name_of_bin+"s.png")

			plt.clf()
     

         
