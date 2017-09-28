#!/usr/bin/env python

#import packages
import scipy.io as sio
#import h5py
import numpy as np
import pdb
#import matplotlib.colors as colors
#import matplotlib.colorbar as colorbar
import sys
import xlsxwriter
import glob
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#import statsmodels.api as sm
#import pandas as pd
#from matplotlib import cm
import xlsxwriter
import scipy.stats as stats
#from scipy.ndimage.filters import gaussian_filter
#from sklearn.decomposition import PCA
#from dPCA import dPCA
#import dPCA_new as dPCA

#######

data = np.load('dpca_results.npy')[()]

for region_key,region_val in data.iteritems():
	if region_key == 'sort_dict':
		continue
	data[region_key]['dpca_analysis'] = {}
	
	dpca_results = data[region_key]['dpca_results']
	total_var = 0

	for cond_key,cond_value in dpca_results['explained_var'].iteritems():
		total_var += np.sum(cond_value)
		print 'cond: %s, val: %s' %(cond_key,np.sum(cond_value))
		
	data[region_key]['dpca_analysis']['total_explained_variance'] = total_var
	print 'region %s, total explained variance= %s' %(region_key,total_var)




	
