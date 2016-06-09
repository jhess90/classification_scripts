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
worksheet.write(row,col+3,'SAMME')
worksheet.write(row,col+4,'SAMME.R')

for key in data.keys():
    row += 1
    worksheet.write(row,col,key)
    #row += 1
    for item in data[key].keys():
        #row +=1
        worksheet.write(row,col + 1,item)
        #row += 1
        for item2 in data[key][item].keys():
            #row +=1
            worksheet.write(row,col+2,item2)
            #row += 1
            #for item3 in data[key][item][item2].keys():
			#row +=1
            worksheet.write(row,col+3,data[key][item][item2]['accuracy_samme'])
            worksheet.write(row,col+4,data[key][item][item2]['accuracy_sammer'])
            row += 1
            
workbook.close()

