library(openxlsx)
library(ggplot2)
library(reshape2)
source("~/dropbox/mult_rp_files/r_test/multiplot.R")
library(zoo)
library(gplots)
library(RColorBrewer)
library(car)

load('rearranged_data.RData')

#TODO remove rollmean??
r0 <- M1_r0_succ_cue_all[1,10:28]
r1 <- M1_r1_succ_cue_all[1,10:28]
r2 <- M1_r2_succ_cue_all[1,10:28]
r3 <- M1_r3_succ_cue_all[1,10:28]

after_time = time[10:28]

r_vals <- c(r0,r1,r2,r3)
level <- as.factor(c(rep(0,length(r0)),rep(1,length(r1)),rep(2,length(r2)),rep(3,length(r3))))
r_levels <- data.frame(value=c(r0,r1,r2,r3),after_time,level)

levene_output <- leveneTest(r_vals,level)

ancova_model <- aov(r0,r1,r2,r3 ~ r_levels$r_value + r_levels$level)
