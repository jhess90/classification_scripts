library(openxlsx)
library(ggplot2)
library(reshape2)
source("~/dropbox/mult_rp_files/r_test/multiplot.R")
#source("~/Dropbox/mult_rp_files/r_test/multiplot.R")
library(zoo)
library(gplots)
library(RColorBrewer)
library(abind)
library(gridGraphics)
library(grid)
library(gridExtra)

master_data <- read.xlsx('all_binned_nl_data.xlsx',sheet=1,colNames=F)



