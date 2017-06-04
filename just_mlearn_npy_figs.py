import scipy.io as sio
import numpy as np
import Tkinter as tk
import tkFileDialog
import matplotlib.patches as mpatches
from sklearn.decomposition import RandomizedPCA
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

mypath = '/Users/McNiel/Documents/workspace/nhp_data/20150902/sorted_files/matfiles/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for file in onlyfiles:
 if file[-3:]!='npy':
        continue
 #uncomment to use random targets for sanity check
 #targets = np.random.randint(0,4,proj.shape[0])
 Data =np.load(file)[0]
 targets = Data['targets']
 #for shuffled target values uncomment the next line
 #targets = np.random.randint(0,5,targets.shape[0])

 for key,value in Data.iteritems():
      if key=='targets':
          continue
      print key
      name_of_bin = file[-11:-4]
      lda = LDA(n_components=3)
      pca = RandomizedPCA(n_components=20)
      proj = pca.fit_transform(value)
      proj = lda.fit_transform(proj,targets)
      plt.clf()
      plt.scatter(proj[:, 0], proj[:, 2], c=targets)
      plt.title(key+" from "+name_of_bin)
      plt.xlabel("LD1")
      plt.ylabel("LD2")
      plt.colorbar()
      plt.savefig(key+" from "+name_of_bin+"s.png")
      plt.savefig(key+" from "+name_of_bin+"s.svg")
      plt.clf()
      classifier = naive_bayes.GaussianNB()
      X_train, X_test, y_train, y_test = train_test_split(proj,targets)
      classifier.fit(X_train,y_train)
      expected = y_test
      predicted = classifier.predict(X_test)

      with open("Classification_Report_"+file[:-4]+".txt","a") as myfile:
             myfile.write("Classification report for\n"+key+' at '+name_of_bin+" post "+key.rsplit('_')[1]+"%s:\n%s\n"
                          % (classifier, classification_report(expected, predicted)))
      print(file[:-4])
      print("Classification report for post injection classifier %s:\n%s\n"
            % (classifier, classification_report(expected, predicted)))
      
