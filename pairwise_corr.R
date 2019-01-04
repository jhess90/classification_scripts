library(openxlsx)
library(ggplot2)
library(reshape2)
library(RColorBrewer)
library(abind)
library(gridGraphics)
library(grid)
library(gridExtra)
library(R.matlab)


source("~/workspace/classification_scripts/multiplot.R")

saveAsPng <- T
region_list <- c('M1','S1','PmD')


#uses simple_output_[region].mat, one from each region

readin <- readMat('simple_output_M1.mat')

all_cue_fr_M1 <- readin$return.dict[,,1]$all.cue.fr
all_res_fr_M1 <- readin$return.dict[,,1]$all.res.fr
total_unit_num_M1 <- dim(all_cue_fr)[1]

condensed <- readin$return.dict[,,1]$condensed
bin_size <- readin$return.dict[,,1]$params[,,1]$bin.size[,]
total_unit_num <- dim(all_cue_fr)[1]

readin <- readMat('simple_output_S1.mat')
all_cue_fr_S1 <- readin$return.dict[,,1]$all.cue.fr
all_res_fr_S1 <- readin$return.dict[,,1]$all.res.fr
total_unit_num_S1 <- dim(all_cue_fr)[1]

readin <- readMat('simple_output_PmD.mat')
all_cue_fr_PmD <- readin$return.dict[,,1]$all.cue.fr
all_res_fr_PmD <- readin$return.dict[,,1]$all.res.fr
total_unit_num_PmD <- dim(all_cue_fr)[1]


