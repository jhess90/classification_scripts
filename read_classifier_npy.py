#!/usr/bin/env python

#import packages
import scipy.io as sio
import numpy as np
import xlsxwriter

data = np.load('accuracy_report_all.npy')[()]

workbook = xlsxwriter.Workbook('data.xlsx')
worksheet = workbook.add_worksheet()

row = 0
col = 0

worksheet.write(row,col,'Time')
worksheet.write(row,col+1,'Region')
worksheet.write(row,col+2,'Classification type')
worksheet.write(row,col+3,'SAMME avg')
worksheet.write(row,col+4,'SAMME std')
worksheet.write(row,col+5,'SAMME.R avg')
worksheet.write(row,col+6,'SAMME.R std')

for key in data.keys():
    row += 1
    worksheet.write(row,col,key)
    for item in data[key].keys():
        worksheet.write(row,col + 1,item)
        for item2 in data[key][item].keys():
            worksheet.write(row,col+2,item2)
            #worksheet.write(row,col+3,data[key][item][item2]['accuracy_samme'])
            #worksheet.write(row,col+4,data[key][item][item2]['accuracy_sammer'])
            worksheet.write(row,col+3,data[key][item][item2]['accuracy_samme_avg'])
            worksheet.write(row,col+4,data[key][item][item2]['accuracy_samme_std'])
            worksheet.write(row,col+5,data[key][item][item2]['accuracy_sammer_avg'])
            worksheet.write(row,col+6,data[key][item][item2]['accuracy_sammer_std'])
            row += 1
            
workbook.close()

