import scipy.io as sio
import numpy as np
import Tkinter as tk
import tkFileDialog
import matplotlib.patches as mpatches
from sklearn.decomposition import RandomizedPCA, KernelPCA
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler
from os import listdir
from os.path import isfile, join
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

mypath = '/Users/johnhessburg/dropbox/single_rp_files/extracted/20150928_504/figs/block1/200_pca/test_random'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for file in onlyfiles:
if file[-3:]!='npy':
     continue
 #if file[4:8]!='4_18':
     #continue
 Data = np.load(file)

 no_mappings=Data.shape[0]
 no_locations=Data.shape[1]
 no_thwacks=Data.shape[2]

 targets = []
 for i in range(1,no_locations+1): targets.append(np.ones(no_thwacks*no_mappings/2)*i)
 targets = np.hstack(np.array(targets))
 
 mapping_targets = []
 for i in range(1,no_mappings+1): mapping_targets.append(np.ones(no_locations*no_thwacks)*i)
 mapping_targets = np.hstack(np.array(mapping_targets))

 first_half = np.hstack(Data[0:no_mappings/2,:,:,:])
 first_half = np.vstack(first_half)
 second_half = np.hstack(Data[no_mappings/2:no_mappings,:,:,:])
 second_half = np.vstack(second_half)
 Data = np.vstack([first_half,second_half])
 # for true targets uncomment next line
 targets = np.hstack([targets,targets])

 #for random targets uncomment next line
 #targets = np.random.randint(1,no_locations+1,no_mappings*no_locations*no_thwacks)


 lda = LDA(n_components=14)
 pca = RandomizedPCA(n_components = 125)
 classifier =  KNeighborsClassifier(8)
 proj = pca.fit_transform(Data)
 proj = lda.fit_transform(proj,targets)
 proj1 = pca.fit_transform(Data)
 proj1 = lda.fit_transform(proj1,mapping_targets)
 print(file)
 plt.clf()
 plt.scatter(proj[0:proj.shape[0]/2,0],proj[0:proj.shape[0]/2,1],c=targets[0:targets.shape[0]/2])
 plt.title(file.rsplit('_')[0]+'_'+file.rsplit('_')[1]+" Before "+file.rsplit('_')[2]+" injection")
 plt.colorbar()
 plt.ylabel("LD1")
 plt.xlabel("LD2")
 plt.savefig(file.rsplit('_')[0]+'_'+file.rsplit('_')[1]+" Before "+file.rsplit('_')[2]+file[-11:-4]+" injection.svg")
 plt.show()
 plt.clf()

 
 plt.scatter(proj[proj.shape[0]/2:proj.shape[0],0],proj[proj.shape[0]/2:proj.shape[0],1],c=targets[targets.shape[0]/2:targets.shape[0]])
 plt.title(file.rsplit('_')[0]+'_'+file.rsplit('_')[1]+" After "+file.rsplit('_')[2]+" injection")
 plt.colorbar()
 plt.ylabel("LD1")
 plt.xlabel("LD2")
 plt.savefig(file.rsplit('_')[0]+'_'+file.rsplit('_')[1]+" After "+file.rsplit('_')[2]+file[-11:-4]+" injection.svg")
 plt.show()
 plt.clf()

 plt.scatter(proj1[:,0],proj1[:,1],c=mapping_targets)
 plt.title(file.rsplit('_')[0]+'_'+file.rsplit('_')[1]+" mapping number "+file.rsplit('_')[2]+" injection")
 plt.colorbar()
 plt.ylabel("LD1")
 plt.xlabel("LD2")
 plt.savefig(file.rsplit('_')[0]+'_'+file.rsplit('_')[1]+" mapping number "+file.rsplit('_')[2]+file[-11:-4]+" injection.svg")
 plt.show()
 plt.clf

 first_half = proj[0:proj.shape[0]/2]
 first_targets = targets[0:targets.shape[0]/2]
 second_half = proj[proj.shape[0]/2:proj.shape[0]]
 second_targets = targets[targets.shape[0]/2:targets.shape[0]]
 X_train, X_test, y_train, y_test = train_test_split(first_half,first_targets)
 classifier.fit(X_train,y_train)
 expected = y_test
 predicted = classifier.predict(X_test)
 print(file[:-4])
 print("Classification report for pre injection classifier %s:\n%s\n"
       % (classifier, classification_report(expected, predicted)))
 
 with open("text.txt","a") as myfile:
     myfile.write("Classification report for pre injection classifier\n"+file.rsplit('_')[0]+'_'+file.rsplit('_')[1]+" "+file.rsplit('_')[2]+" injection %s:\n%s\n"
       % (classifier, classification_report(expected, predicted)))
     myfile.write(file+"\n")


 alpha = 0.01
 lasso = Lasso(alpha=alpha)
 y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
 r2_score_lasso = r2_score(y_test, y_pred_lasso)
 print(lasso)
 print("r^2 on test data : %f" % r2_score_lasso)

 X_train, X_test, y_train, y_test = train_test_split(second_half,second_targets)
 classifier.fit(X_train,y_train)
 expected = y_test
 predicted = classifier.predict(X_test)
 print(file[:-4])
 print("Classification report for post injection classifier %s:\n%s\n"
       % (classifier, classification_report(expected, predicted)))

 with open("text.txt","a") as myfile:
     myfile.write("Classification report for post injection classifier\n"+file.rsplit('_')[0]+'_'+file.rsplit('_')[1]+" "+file.rsplit('_')[2]+" injection %s:\n%s\n"
       % (classifier, classification_report(expected, predicted)))
     myfile.write(file+"\n")

 alpha = 0.01
 lasso = Lasso(alpha=alpha)
 y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
 r2_score_lasso = r2_score(y_test, y_pred_lasso)
 print(lasso)
 print("r^2 on test data : %f" % r2_score_lasso)


 classifier.fit(first_half,first_targets)
 expected = second_targets
 predicted = classifier.predict(second_half)
 print("Classification report for post injection classifier\n using preinjection data as training%s:\n%s\n"
       % (classifier, classification_report(expected, predicted)))

 with open("text.txt","a") as myfile:
     myfile.write("Classification report for post injection classifier\ntrained on pre injection data\n"+file.rsplit('_')[0]+'_'+file.rsplit('_')[1]+" "+file.rsplit('_')[2]+" injection %s:\n%s\n"
       % (classifier, classification_report(expected, predicted)))
     myfile.write(file+"\n")




