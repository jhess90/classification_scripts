#!/usr/bin/env python

#import packages
import scipy.io as sio
import numpy as np
import xlsxwriter

data = np.load('accuracy_report_all.npy')[()]

workbook = xlsxwriter.Workbook('data.xlsx')
worksheet = workbook.add_worksheet()

#d = {'a':['e1','e2','e3'],'b':['e1','e2'],'c':['e1']}
row = 0
col = 0

for key in data.keys():
    row += 1
    worksheet.write(row,col,key)
    #row += 1
    for item in data[key].keys():
        row +=1
        worksheet.write(row,col + 1,item)
        #row += 1
        for item2 in data[key].keys():
            row +=1
            worksheet.write(row,col+2,item2)
            #row += 1
            for item3 in data[key][item2].keys():
                row +=1
                worksheet.write(row,col+3,item3)
                #row += 1
			
workbook.close()

