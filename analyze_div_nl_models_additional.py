#!/usr/bin/env python

#import packages
import scipy.io as sio
import h5py
import numpy as np
import pdb
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import sys
import xlsxwriter
import glob
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from matplotlib import cm
import xlsxwriter
import scipy.stats as stats
from scipy import ndimage
from math import isinf
import os
from scipy.stats.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
import math




###############
nhp_id = '0059'
#nhp_id = '504'


if nhp_id == '0059':
    data_dict_1 = np.load('model_save_0059_1.npy')[()]
    data_dict_2 = np.load('model_save_0059_2.npy')[()]
    data_dict_3 = np.load('model_save_0059_3.npy')[()]
else:
    data_dict_1 = np.load('model_save_504_1.npy')[()]
    data_dict_2 = np.load('model_save_504_2.npy')[()]
    data_dict_3 = np.load('model_save_504_3.npy')[()]


