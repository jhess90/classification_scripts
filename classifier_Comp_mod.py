#!/usr/bin/python                                                                                   
# -*- coding: utf-8 -*-   
# Code source: Gaël Varoquaux                                                                                       
#              Andreas Müller                                                                                       
# Modified for documentation by Jaques Grobler                                                                      
# License: BSD 3 clause                                                                                             

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import preprocessing
from sklearn.decomposition import RandomizedPCA
from sklearn import metrics
from sklearn.metrics import recall_score, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
import matplotlib.lines as mlines
import Tkinter as tk
import tkFileDialog

#----------------Import Data For Remainder of Analysis------------------
#Datasets = [[np.load('Pre_cue_Data.npy'),np.load('Pre_cue_targets.npy')],[np.load('Post_cue_Data.npy'),np.load('Post_cue_targets.npy')],[np.load('Pre_reward_Data.npy'),np.load('Pre_reward_targets.npy')],[np.load('Post_reward_Data.npy'),np.load('Post_reward_targets.npy')]]

#root = tk.Tk()
#root.withdraw()
#filename = tkFileDialog.askopenfilename();
filename='/Volumes/ADATA_SH93/nhp_data/20150925_0059/sorted_files/matfiles/50_percent_cue/multi_reward25-15-30-26_hists_1.0-1.5.npy'
print filename
a = np.load(filename);
#---------------------------------------------------------------------

#---------------------------------------------------------------------
TARGETS = a[0]['targets']
Datasets=[a[0]['S1_c_data'],a[0]['S1_d_data']]
figure = plt.figure(figsize=(32, 10))
#plt.suptitle('Right S1 Manual Normal Color Cue .5s Window blue = Reward trials red = No Reward Trials',fontsize=30)
i = 1
name_index = 0
#---------------------------------------------------------------------


#--------------------------Scale and Transform-------------------------------------------
for ds in Datasets:
 pca = RandomizedPCA(n_components=2)
 DATA = ds
 DATA = pca.fit_transform(DATA)
 


 h = .05  # step size in the mesh                                                                                    
#---------------------Specify Classifier Names and Type------------------------------------------------
 names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
          "Random Forest", "AdaBoost", "Naive Bayes", "LDA","QDA"]
 Labels = ['S1 Post_Cue','S1 Post_Reward']

 classifiers = [
     KNeighborsClassifier(3),
     SVC(kernel="linear", C=0.025),
     SVC(gamma=2, C=1),
     DecisionTreeClassifier(max_depth=5),
     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
     AdaBoostClassifier(),
     GaussianNB(),
     LDA(),QDA()]

 
 # iterate over datasets
 # preprocess dataset, split into training and test part                                                      
 X_train, X_test, y_train, y_test = train_test_split(DATA, TARGETS, test_size=.4)

 x_min, x_max = DATA[:, 0].min() - .5, DATA[:, 0].max() + .5
 y_min, y_max = DATA[:, 1].min() - .5, DATA[:, 1].max() + .5
 xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

#---------------------Build Subplots for Final Analysis Figure-----------------------
 cm = plt.cm.RdBu
 cm_bright = ListedColormap(['#FF0000', '#0000FF'])
 ax = plt.subplot(len(Datasets), len(classifiers) + 1, i)
 # Plot the training points                                           
 #print X_train.shape
 #print X_test.shape
 ax.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = cm_bright)
 # and testing points                                          
 ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
 ax.set_xlim(xx.min(), xx.max())
 ax.set_ylim(yy.min(), yy.max())
 ax.set_xticks(())
 ax.set_yticks(())
 ax.set_title(Labels[name_index])
 ax.set_xlabel('PC 1')
 ax.set_ylabel('PC 2')
 blue_line =mlines.Line2D([],[],color='blue',marker = 'o',markersize=5,label='No Reward')
 red_line =mlines.Line2D([],[],color='red',marker = 'o',markersize=5,label='Reward')
 ax.legend(handles=[blue_line,red_line],loc = 1, prop={'size':6})
 i += 1

  # iterate over classifiers                                                                                      


 #Classifier shape
 QQ=np.array([xx.ravel(),yy.ravel()])

 print QQ.shape
 #print DATA.shape

#--------------------Build Classification Results-------------------------------------------------
 for name, clf in zip(names, classifiers):
     ax = plt.subplot(len(Datasets), len(classifiers) + 1, i)
     clf.fit(X_train, y_train)

     predicted = clf.predict(X_test)
     expected = y_test

     score = roc_auc_score(expected,predicted)

     print metrics.classification_report(expected, predicted)
     with open("text.txt","a") as myfile:
         myfile.write("Classification report for %s:\n%s\n"
       % (clf, metrics.classification_report(expected, predicted)))
         myfile.write(Labels[name_index]+"\n")

     # Plot the decision boundary. For that, we will assign a color to each                                      
     # point in the mesh [x_min, m_max]x[y_min, y_max].                                                          
     if hasattr(clf, "decision_function"):
         Z = clf.decision_function(np.c_[QQ.T])

     else:
         print np.c_[QQ.T].shape
         Z = clf.predict_proba(np.c_[QQ.T])[:,1]

     # Put the result into a color plot                                                                     
     Z = Z.reshape(xx.shape)
     ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)


     ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
     # and testing points                                                                                        
     ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                    alpha=0.6)

     ax.set_xlim(xx.min(), xx.max())
     ax.set_ylim(yy.min(), yy.max())
     ax.set_xticks(())
     ax.set_yticks(())
     ax.set_title(name)
     ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                 size=15, horizontalalignment='right')
     i += 1
 name_index += 1

#------------------Plot Final Figure---------------------------------------------------
figure.subplots_adjust(left=.02, right=.98)
plt.tight_layout()
plt.savefig('Results.png')
plt.show()




