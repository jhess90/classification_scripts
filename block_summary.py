#!/usr/bin/env python

#import packages
import scipy.io as sio
import numpy as np
import pdb
import xlsxwriter
import glob
#import matplotlib.pyplot as plt
#import statsmodels.api as sm
import pandas as pd
from matplotlib import cm
import xlsxwriter


#########

M1_import = sio.loadmat('corr_output_M1.mat')
S1_import = sio.loadmat('corr_output_S1.mat')
PmD_import = sio.loadmat('corr_output_PmD.mat')

condensed = M1_import['condensed']
num_trials = np.shape(condensed)[0]

r0 = len(np.where(condensed[:,3] == 0)[0].astype(int))
r1 = len(np.where(condensed[:,3] == 1)[0].astype(int))
r2 = len(np.where(condensed[:,3] == 2)[0].astype(int))
r3 = len(np.where(condensed[:,3] == 3)[0].astype(int))
rx = len(np.where(condensed[:,3] >= 1)[0].astype(int))

p0 = len(np.where(condensed[:,4] == 0)[0].astype(int))
p1 = len(np.where(condensed[:,4] == 1)[0].astype(int))
p2 = len(np.where(condensed[:,4] == 2)[0].astype(int))
p3 = len(np.where(condensed[:,4] == 3)[0].astype(int))
px = len(np.where(condensed[:,4] >= 1)[0].astype(int))

v_3 = len(np.where(condensed[:,6] == -3)[0].astype(int))
v_2 = len(np.where(condensed[:,6] == -2)[0].astype(int))
v_1 = len(np.where(condensed[:,6] == -1)[0].astype(int))
v0 = len(np.where(condensed[:,6] == 0)[0].astype(int))
v1 = len(np.where(condensed[:,6] == 1)[0].astype(int))
v2 = len(np.where(condensed[:,6] == 2)[0].astype(int))
v3 = len(np.where(condensed[:,6] == 3)[0].astype(int))

m0 = len(np.where(condensed[:,7] == 0)[0].astype(int))
m1 = len(np.where(condensed[:,7] == 1)[0].astype(int))
m2 = len(np.where(condensed[:,7] == 2)[0].astype(int))
m3 = len(np.where(condensed[:,7] == 3)[0].astype(int))
m4 = len(np.where(condensed[:,7] == 4)[0].astype(int))
m5 = len(np.where(condensed[:,7] == 5)[0].astype(int))
m6 = len(np.where(condensed[:,7] == 6)[0].astype(int))

res0 = len(np.where(condensed[:,5] == 0)[0].astype(int))
res1 = len(np.where(condensed[:,5] == 1)[0].astype(int))

try:
    catch_x = len(np.where(condensed[:,6] <= -1)[0].astype(int))
    catchx = len(np.where(condensed[:,6] >= 1)[0].astype(int))

    catch_3 = len(np.where(condensed[:,6] == -3)[0].astype(int))
    catch_2 = len(np.where(condensed[:,6] == -2)[0].astype(int))
    catch_1 = len(np.where(condensed[:,6] == -1)[0].astype(int))
    catch1 = len(np.where(condensed[:,6] == 1)[0].astype(int))
    catch2 = len(np.where(condensed[:,6] == 2)[0].astype(int))
    catch3 = len(np.where(condensed[:,6] == 3)[0].astype(int))
except:
    pass

r0_p0 = sum(np.logical_and(condensed[:,3] == 0, condensed[:,4] == 0))
rx_p0 = sum(np.logical_and(condensed[:,3] >= 1, condensed[:,4] == 0))
r0_px = sum(np.logical_and(condensed[:,3] == 0, condensed[:,4] >= 1))
rx_px = sum(np.logical_and(condensed[:,3] >= 1, condensed[:,4] >= 1))

r0_p0_f = sum(np.logical_and(np.logical_and(condensed[:,3] == 0, condensed[:,4] == 0),condensed[:,5]==0))
rx_p0_f = sum(np.logical_and(np.logical_and(condensed[:,3] >= 1, condensed[:,4] == 0),condensed[:,5]==0))
r0_px_f = sum(np.logical_and(np.logical_and(condensed[:,3] == 0, condensed[:,4] >= 1),condensed[:,5]==0))
rx_px_f = sum(np.logical_and(np.logical_and(condensed[:,3] >= 1, condensed[:,4] >= 1),condensed[:,5]==0))
r0_p0_s = sum(np.logical_and(np.logical_and(condensed[:,3] == 0, condensed[:,4] == 0),condensed[:,5]==0))
rx_p0_s = sum(np.logical_and(np.logical_and(condensed[:,3] >= 1, condensed[:,4] == 0),condensed[:,5]==0))
r0_px_s = sum(np.logical_and(np.logical_and(condensed[:,3] == 0, condensed[:,4] >= 1),condensed[:,5]==0))
rx_px_s = sum(np.logical_and(np.logical_and(condensed[:,3] >= 1, condensed[:,4] >= 1),condensed[:,5]==0))

M1_unit_num = np.shape(M1_import['corr_output'])[0]
S1_unit_num = np.shape(S1_import['corr_output'])[0]
PmD_unit_num = np.shape(PmD_import['corr_output'])[0]

print ('r0: %s' %(r0))
print ('r1: %s' %(r1))
print ('r2: %s' %(r2))
print ('r3: %s' %(r3))
print ('rx: %s\n' %(rx))

print ('p0: %s' %(p0))
print ('p1: %s' %(p1))
print ('p2: %s' %(p2))
print ('p3: %s' %(p3))
print ('px: %s\n' %(px))

print ('v_3: %s' %(v_3))
print ('v_2: %s' %(v_2))
print ('v_1: %s' %(v1))
print ('v0: %s' %(v0))
print ('v1: %s' %(v1))
print ('v2: %s' %(v2))
print ('v3: %s\n' %(v3))

print ('m0: %s' %(m0))
print ('m1: %s' %(m1))
print ('m2: %s' %(m2))
print ('m3: %s' %(m3))
print ('m4: %s' %(m4))
print ('m5: %s' %(m5))
print ('m6: %s\n' %(m6))

print ('succ: %s' %(res1))
print ('fail: %s\n' %(res0))

try:
    print('catch_x: %s' %(catch_x))
    print('catchx: %s\n' %(catchx))

    print('catch_3: %s' %(catch_3))
    print('catch_2: %s' %(catch_2))
    print('catch_1: %s' %(catch_1))
    print('catch1: %s' %(catch1))
    print('catch2: %s' %(catch2))
    print('catch3: %s\n' %(catch3))
except:
    pass

print('r0_p0: %s' %(r0_p0))
print('rx_p0: %s' %(rx_p0))
print('r0_px: %s' %(r0_px))
print('rx_px: %s\n' %(rx_px))

print('r0_p0_f: %s' %(r0_p0_f))
print('rx_p0_f: %s' %(rx_p0_f))
print('r0_px_f: %s' %(r0_px_f))
print('rx_px_f: %s' %(rx_px_f))
print('r0_p0_s: %s' %(r0_p0_s))
print('rx_p0_s: %s' %(rx_p0_s))
print('r0_px_s: %s' %(r0_px_s))
print('rx_px_s: %s\n' %(rx_px_s))

print('M1 units: %s' %(M1_unit_num))
print('S1 units: %s' %(S1_unit_num))
print('PmD units: %s' %(PmD_unit_num))



output = open("block_summary.txt","w")
output.write ('r0: %s\n' %(r0))
output.write ('r1: %s\n' %(r1))
output.write ('r2: %s\n' %(r2))
output.write ('r3: %s\n' %(r3))
output.write ('rx: %s\n\n' %(rx))

output.write ('p0: %s\n' %(p0))
output.write ('p1: %s\n' %(p1))
output.write ('p2: %s\n' %(p2))
output.write ('p3: %s\n' %(p3))
output.write ('px: %s\n\n' %(px))

output.write ('v_3: %s\n' %(v_3))
output.write ('v_2: %s\n' %(v_2))
output.write ('v_1: %s\n' %(v1))
output.write ('v0: %s\n' %(v0))
output.write ('v1: %s\n' %(v1))
output.write ('v2: %s\n' %(v2))
output.write ('v3: %s\n\n' %(v3))

output.write ('m0: %s\n' %(m0))
output.write ('m1: %s\n' %(m1))
output.write ('m2: %s\n' %(m2))
output.write ('m3: %s\n' %(m3))
output.write ('m4: %s\n' %(m4))
output.write ('m5: %s\n' %(m5))
output.write ('m6: %s\n\n' %(m6))

output.write ('succ: %s\n' %(res1))
output.write ('fail: %s\n\n' %(res0))

try:
    output.write('catch_x: %s\n' %(catch_x))
    output.write('catchx: %s\n\n' %(catchx))

    output.write('catch_3: %s\n' %(catch_3))
    output.write('catch_2: %s\n' %(catch_2))
    output.write('catch_1: %s\n' %(catch_1))
    output.write('catch1: %s\n' %(catch1))
    output.write('catch2: %s\n' %(catch2))
    output.write('catch3: %s\n\n' %(catch3))
except:
    pass

output.write('r0_p0: %s\n' %(r0_p0))
output.write('rx_p0: %s\n' %(rx_p0))
output.write('r0_px: %s\n' %(r0_px))
output.write('rx_px: %s\n\n' %(rx_px))

output.write('r0_p0_f: %s\n' %(r0_p0_f))
output.write('rx_p0_f: %s\n' %(rx_p0_f))
output.write('r0_px_f: %s\n' %(r0_px_f))
output.write('rx_px_f: %s\n' %(rx_px_f))
output.write('r0_p0_s: %s\n' %(r0_p0_s))
output.write('rx_p0_s: %s\n' %(rx_p0_s))
output.write('r0_px_s: %s\n' %(r0_px_s))
output.write('rx_px_s: %s\n\n' %(rx_px_s))

output.write('M1 units: %s\n' %(M1_unit_num))
output.write('S1 units: %s\n' %(S1_unit_num))
output.write('PmD units: %s\n' %(PmD_unit_num))


output.close()
