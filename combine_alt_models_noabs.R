library(openxlsx)
library(ggplot2)
library(reshape2)
source("~/documents/lab/workspace/Classification_scripts/multiplot.R")
#source("~/workspace/classification_scripts/multiplot.R")
library(zoo)
library(gplots)
library(RColorBrewer)
library(abind)
library(gridGraphics)
library(grid)
library(gridExtra)
library(R.matlab)
library(broom)
library(plyr)
library(reshape)


nhp_id <- '0059'


##################
#### 0059 ########
##################
if (nhp_id == '0059'){
  load('~/Dropbox/model_alt/0_3_13_1/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- bfr_cue_nums
  S1_aft_cue_nums_sum <- aft_cue_nums
  S1_bfr_result_nums_sum <- bfr_result_nums
  S1_aft_result_nums_sum <- aft_result_nums
  load('~/Dropbox/model_alt/0_3_13_1/alphabeta_S1_percsig_all.RData')
  #percsig nums: no sig, both sig, beta sig, alpha sig
  S1_bfr_cue_nums_sum <- percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- percsig_bfr_res_nums
  S1_aft_res_nums_sum <- percsig_aft_res_nums
  S1_unit_total <- percsig_total_units
  S1_bfr_cue_all_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/S1_bfr_cue_all_slopes_all.rds')
  S1_aft_cue_all_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/S1_aft_cue_all_slopes_all.rds')
  S1_bfr_result_all_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/S1_bfr_result_all_slopes_all.rds')
  S1_aft_result_all_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/S1_aft_result_all_slopes_all.rds')
  S1_bfr_cue_sig_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/S1_bfr_cue_sig_slopes_all.rds')
  S1_aft_cue_sig_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/S1_aft_cue_sig_slopes_all.rds')
  S1_bfr_result_sig_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/S1_bfr_result_sig_slopes_all.rds')
  S1_aft_result_sig_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/S1_aft_result_sig_slopes_all.rds')
  
  load('~/Dropbox/model_alt/0_3_13_1/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- bfr_cue_nums
  M1_aft_cue_nums_sum <- aft_cue_nums
  M1_bfr_result_nums_sum <- bfr_result_nums
  M1_aft_result_nums_sum <- aft_result_nums
  load('~/Dropbox/model_alt/0_3_13_1/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- percsig_bfr_res_nums
  M1_aft_res_nums_sum <- percsig_aft_res_nums
  M1_unit_total <- percsig_total_units
  M1_bfr_cue_all_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/M1_bfr_cue_all_slopes_all.rds')
  M1_aft_cue_all_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/M1_aft_cue_all_slopes_all.rds')
  M1_bfr_result_all_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/M1_bfr_result_all_slopes_all.rds')
  M1_aft_result_all_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/M1_aft_result_all_slopes_all.rds')
  M1_bfr_cue_sig_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/M1_bfr_cue_sig_slopes_all.rds')
  M1_aft_cue_sig_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/M1_aft_cue_sig_slopes_all.rds')
  M1_bfr_result_sig_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/M1_bfr_result_sig_slopes_all.rds')
  M1_aft_result_sig_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/M1_aft_result_sig_slopes_all.rds')
  
  load('~/Dropbox/model_alt/0_3_13_1/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- bfr_cue_nums
  PmD_aft_cue_nums_sum <- aft_cue_nums
  PmD_bfr_result_nums_sum <- bfr_result_nums
  PmD_aft_result_nums_sum <- aft_result_nums
  load('~/Dropbox/model_alt/0_3_13_1/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- percsig_aft_res_nums
  PmD_unit_total <- percsig_total_units
  PmD_bfr_cue_all_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/PmD_bfr_cue_all_slopes_all.rds')
  PmD_aft_cue_all_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/PmD_aft_cue_all_slopes_all.rds')
  PmD_bfr_result_all_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/PmD_bfr_result_all_slopes_all.rds')
  PmD_aft_result_all_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/PmD_aft_result_all_slopes_all.rds')
  PmD_bfr_cue_sig_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/PmD_bfr_cue_sig_slopes_all.rds')
  PmD_aft_cue_sig_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/PmD_aft_cue_sig_slopes_all.rds')
  PmD_bfr_result_sig_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/PmD_bfr_result_sig_slopes_all.rds')
  PmD_aft_result_sig_slopes <- readRDS('~/Dropbox/model_alt/0_3_13_1/PmD_aft_result_sig_slopes_all.rds')

  cat('M1:',sum(M1_bfr_cue_nums_sum),'\n',sum(M1_aft_cue_nums_sum),'\n',sum(M1_bfr_res_nums_sum),'\n',sum(M1_aft_res_nums_sum),'\n\n')
  cat('S1:',sum(S1_bfr_cue_nums_sum),'\n',sum(S1_aft_cue_nums_sum),'\n',sum(S1_bfr_res_nums_sum),'\n',sum(S1_aft_res_nums_sum),'\n\n')
  cat('PmD:',sum(PmD_bfr_cue_nums_sum),'\n',sum(PmD_aft_cue_nums_sum),'\n',sum(PmD_bfr_res_nums_sum),'\n',sum(PmD_aft_res_nums_sum),'\n\n')
  
  #
  load('~/Dropbox/model_alt/0_3_13_2/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_13_2/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_13_2/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_13_2/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_13_2/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_13_2/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_2/PmD_aft_result_sig_slopes_all.rds'))
  
  cat('M1:',sum(M1_bfr_cue_nums_sum),'\n',sum(M1_aft_cue_nums_sum),'\n',sum(M1_bfr_res_nums_sum),'\n',sum(M1_aft_res_nums_sum),'\n\n')
  cat('S1:',sum(S1_bfr_cue_nums_sum),'\n',sum(S1_aft_cue_nums_sum),'\n',sum(S1_bfr_res_nums_sum),'\n',sum(S1_aft_res_nums_sum),'\n\n')
  cat('PmD:',sum(PmD_bfr_cue_nums_sum),'\n',sum(PmD_aft_cue_nums_sum),'\n',sum(PmD_bfr_res_nums_sum),'\n',sum(PmD_aft_res_nums_sum),'\n\n')

  #
  load('~/Dropbox/model_alt/0_3_13_3/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_13_3/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_13_3/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_13_3/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_13_3/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_13_3/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_13_3/PmD_aft_result_sig_slopes_all.rds'))

  cat('M1:',sum(M1_bfr_cue_nums_sum),'\n',sum(M1_aft_cue_nums_sum),'\n',sum(M1_bfr_res_nums_sum),'\n',sum(M1_aft_res_nums_sum),'\n\n')
  cat('S1:',sum(S1_bfr_cue_nums_sum),'\n',sum(S1_aft_cue_nums_sum),'\n',sum(S1_bfr_res_nums_sum),'\n',sum(S1_aft_res_nums_sum),'\n\n')
  cat('PmD:',sum(PmD_bfr_cue_nums_sum),'\n',sum(PmD_aft_cue_nums_sum),'\n',sum(PmD_bfr_res_nums_sum),'\n',sum(PmD_aft_res_nums_sum),'\n\n')
  
  #
  load('~/Dropbox/model_alt/0_3_14_1/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_14_1/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_14_1/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_14_1/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/S1_aft_result_sig_slopes_all.rds'))

  load('~/Dropbox/model_alt/0_3_14_1/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_14_1/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_1/PmD_aft_result_sig_slopes_all.rds'))

  cat('M1:',sum(M1_bfr_cue_nums_sum),'\n',sum(M1_aft_cue_nums_sum),'\n',sum(M1_bfr_res_nums_sum),'\n',sum(M1_aft_res_nums_sum),'\n\n')
  cat('S1:',sum(S1_bfr_cue_nums_sum),'\n',sum(S1_aft_cue_nums_sum),'\n',sum(S1_bfr_res_nums_sum),'\n',sum(S1_aft_res_nums_sum),'\n\n')
  cat('PmD:',sum(PmD_bfr_cue_nums_sum),'\n',sum(PmD_aft_cue_nums_sum),'\n',sum(PmD_bfr_res_nums_sum),'\n',sum(PmD_aft_res_nums_sum),'\n\n')
  
  #
  load('~/Dropbox/model_alt/0_3_14_2/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_14_2/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_14_2/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_14_2/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_14_2/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_14_2/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_2/PmD_aft_result_sig_slopes_all.rds'))

  cat('M1:',sum(M1_bfr_cue_nums_sum),'\n',sum(M1_aft_cue_nums_sum),'\n',sum(M1_bfr_res_nums_sum),'\n',sum(M1_aft_res_nums_sum),'\n\n')
  cat('S1:',sum(S1_bfr_cue_nums_sum),'\n',sum(S1_aft_cue_nums_sum),'\n',sum(S1_bfr_res_nums_sum),'\n',sum(S1_aft_res_nums_sum),'\n\n')
  cat('PmD:',sum(PmD_bfr_cue_nums_sum),'\n',sum(PmD_aft_cue_nums_sum),'\n',sum(PmD_bfr_res_nums_sum),'\n',sum(PmD_aft_res_nums_sum),'\n\n')
  
  #
  load('~/Dropbox/model_alt/0_3_14_3/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_14_3/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_14_3/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_14_3/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_14_3/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_14_3/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_14_3/PmD_aft_result_sig_slopes_all.rds'))

  cat('M1:',sum(M1_bfr_cue_nums_sum),'\n',sum(M1_aft_cue_nums_sum),'\n',sum(M1_bfr_res_nums_sum),'\n',sum(M1_aft_res_nums_sum),'\n\n')
  cat('S1:',sum(S1_bfr_cue_nums_sum),'\n',sum(S1_aft_cue_nums_sum),'\n',sum(S1_bfr_res_nums_sum),'\n',sum(S1_aft_res_nums_sum),'\n\n')
  cat('PmD:',sum(PmD_bfr_cue_nums_sum),'\n',sum(PmD_aft_cue_nums_sum),'\n',sum(PmD_bfr_res_nums_sum),'\n',sum(PmD_aft_res_nums_sum),'\n\n')
  
  #
  load('~/Dropbox/model_alt/0_3_27_1/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_27_1/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_27_1/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_27_1/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_27_1/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_27_1/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_1/PmD_aft_result_sig_slopes_all.rds'))

  cat('M1:',sum(M1_bfr_cue_nums_sum),'\n',sum(M1_aft_cue_nums_sum),'\n',sum(M1_bfr_res_nums_sum),'\n',sum(M1_aft_res_nums_sum),'\n\n')
  cat('S1:',sum(S1_bfr_cue_nums_sum),'\n',sum(S1_aft_cue_nums_sum),'\n',sum(S1_bfr_res_nums_sum),'\n',sum(S1_aft_res_nums_sum),'\n\n')
  cat('PmD:',sum(PmD_bfr_cue_nums_sum),'\n',sum(PmD_aft_cue_nums_sum),'\n',sum(PmD_bfr_res_nums_sum),'\n',sum(PmD_aft_res_nums_sum),'\n\n')
  
  #
  load('~/Dropbox/model_alt/0_3_27_2/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_27_2/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_27_2/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_27_2/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_27_2/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_27_2/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_27_2/PmD_aft_result_sig_slopes_all.rds'))

  cat('M1:',sum(M1_bfr_cue_nums_sum),'\n',sum(M1_aft_cue_nums_sum),'\n',sum(M1_bfr_res_nums_sum),'\n',sum(M1_aft_res_nums_sum),'\n\n')
  cat('S1:',sum(S1_bfr_cue_nums_sum),'\n',sum(S1_aft_cue_nums_sum),'\n',sum(S1_bfr_res_nums_sum),'\n',sum(S1_aft_res_nums_sum),'\n\n')
  cat('PmD:',sum(PmD_bfr_cue_nums_sum),'\n',sum(PmD_aft_cue_nums_sum),'\n',sum(PmD_bfr_res_nums_sum),'\n',sum(PmD_aft_res_nums_sum),'\n\n')

  
  #
  load('~/Dropbox/model_alt/0_3_28_1/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_28_1/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_28_1/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_28_1/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_28_1/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_28_1/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_1/PmD_aft_result_sig_slopes_all.rds'))

  cat('M1:',sum(M1_bfr_cue_nums_sum),'\n',sum(M1_aft_cue_nums_sum),'\n',sum(M1_bfr_res_nums_sum),'\n',sum(M1_aft_res_nums_sum),'\n\n')
  cat('S1:',sum(S1_bfr_cue_nums_sum),'\n',sum(S1_aft_cue_nums_sum),'\n',sum(S1_bfr_res_nums_sum),'\n',sum(S1_aft_res_nums_sum),'\n\n')
  cat('PmD:',sum(PmD_bfr_cue_nums_sum),'\n',sum(PmD_aft_cue_nums_sum),'\n',sum(PmD_bfr_res_nums_sum),'\n',sum(PmD_aft_res_nums_sum),'\n\n')
  
  #
  load('~/Dropbox/model_alt/0_3_28_2/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_28_2/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_28_2/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_28_2/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_28_2/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_28_2/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_2/PmD_aft_result_sig_slopes_all.rds'))

  cat('M1:',sum(M1_bfr_cue_nums_sum),'\n',sum(M1_aft_cue_nums_sum),'\n',sum(M1_bfr_res_nums_sum),'\n',sum(M1_aft_res_nums_sum),'\n\n')
  cat('S1:',sum(S1_bfr_cue_nums_sum),'\n',sum(S1_aft_cue_nums_sum),'\n',sum(S1_bfr_res_nums_sum),'\n',sum(S1_aft_res_nums_sum),'\n\n')
  cat('PmD:',sum(PmD_bfr_cue_nums_sum),'\n',sum(PmD_aft_cue_nums_sum),'\n',sum(PmD_bfr_res_nums_sum),'\n',sum(PmD_aft_res_nums_sum),'\n\n')

  
  #
  load('~/Dropbox/model_alt/0_3_28_3/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_28_3/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_28_3/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_28_3/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/0_3_28_3/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/0_3_28_3/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/0_3_28_3/PmD_aft_result_sig_slopes_all.rds'))

  cat('M1:',sum(M1_bfr_cue_nums_sum),'\n',sum(M1_aft_cue_nums_sum),'\n',sum(M1_bfr_res_nums_sum),'\n',sum(M1_aft_res_nums_sum),'\n\n')
  cat('S1:',sum(S1_bfr_cue_nums_sum),'\n',sum(S1_aft_cue_nums_sum),'\n',sum(S1_bfr_res_nums_sum),'\n',sum(S1_aft_res_nums_sum),'\n\n')
  cat('PmD:',sum(PmD_bfr_cue_nums_sum),'\n',sum(PmD_aft_cue_nums_sum),'\n',sum(PmD_bfr_res_nums_sum),'\n',sum(PmD_aft_res_nums_sum),'\n\n')


}else if (nhp_id == '504'){
    
  ##################
  #### 504 #########
  ##################
  
  
  load('~/Dropbox/model_alt/5_3_13_1/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- bfr_cue_nums
  S1_aft_cue_nums_sum <- aft_cue_nums
  S1_bfr_result_nums_sum <- bfr_result_nums
  S1_aft_result_nums_sum <- aft_result_nums
  load('~/Dropbox/model_alt/5_3_13_1/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- percsig_bfr_res_nums
  S1_aft_res_nums_sum <- percsig_aft_res_nums
  S1_unit_total <- percsig_total_units
  S1_bfr_cue_all_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/S1_bfr_cue_all_slopes_all.rds')
  S1_aft_cue_all_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/S1_aft_cue_all_slopes_all.rds')
  S1_bfr_result_all_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/S1_bfr_result_all_slopes_all.rds')
  S1_aft_result_all_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/S1_aft_result_all_slopes_all.rds')
  S1_bfr_cue_sig_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/S1_bfr_cue_sig_slopes_all.rds')
  S1_aft_cue_sig_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/S1_aft_cue_sig_slopes_all.rds')
  S1_bfr_result_sig_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/S1_bfr_result_sig_slopes_all.rds')
  S1_aft_result_sig_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/S1_aft_result_sig_slopes_all.rds')
  
  load('~/Dropbox/model_alt/5_3_13_1/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- bfr_cue_nums
  M1_aft_cue_nums_sum <- aft_cue_nums
  M1_bfr_result_nums_sum <- bfr_result_nums
  M1_aft_result_nums_sum <- aft_result_nums
  load('~/Dropbox/model_alt/5_3_13_1/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- percsig_bfr_res_nums
  M1_aft_res_nums_sum <- percsig_aft_res_nums
  M1_unit_total <- percsig_total_units
  M1_bfr_cue_all_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/M1_bfr_cue_all_slopes_all.rds')
  M1_aft_cue_all_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/M1_aft_cue_all_slopes_all.rds')
  M1_bfr_result_all_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/M1_bfr_result_all_slopes_all.rds')
  M1_aft_result_all_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/M1_aft_result_all_slopes_all.rds')
  M1_bfr_cue_sig_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/M1_bfr_cue_sig_slopes_all.rds')
  M1_aft_cue_sig_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/M1_aft_cue_sig_slopes_all.rds')
  M1_bfr_result_sig_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/M1_bfr_result_sig_slopes_all.rds')
  M1_aft_result_sig_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/M1_aft_result_sig_slopes_all.rds')
  
  load('~/Dropbox/model_alt/5_3_13_1/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- bfr_cue_nums
  PmD_aft_cue_nums_sum <- aft_cue_nums
  PmD_bfr_result_nums_sum <- bfr_result_nums
  PmD_aft_result_nums_sum <- aft_result_nums
  load('~/Dropbox/model_alt/5_3_13_1/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- percsig_aft_res_nums
  PmD_unit_total <- percsig_total_units
  PmD_bfr_cue_all_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/PmD_bfr_cue_all_slopes_all.rds')
  PmD_aft_cue_all_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/PmD_aft_cue_all_slopes_all.rds')
  PmD_bfr_result_all_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/PmD_bfr_result_all_slopes_all.rds')
  PmD_aft_result_all_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/PmD_aft_result_all_slopes_all.rds')
  PmD_bfr_cue_sig_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/PmD_bfr_cue_sig_slopes_all.rds')
  PmD_aft_cue_sig_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/PmD_aft_cue_sig_slopes_all.rds')
  PmD_bfr_result_sig_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/PmD_bfr_result_sig_slopes_all.rds')
  PmD_aft_result_sig_slopes <- readRDS('~/Dropbox/model_alt/5_3_13_1/PmD_aft_result_sig_slopes_all.rds')
  
  #
  load('~/Dropbox/model_alt/5_3_13_2/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_13_2/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_13_2/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_13_2/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_13_2/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_13_2/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_2/PmD_aft_result_sig_slopes_all.rds'))
  
  #
  load('~/Dropbox/model_alt/5_3_13_3/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_13_3/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_13_3/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_13_3/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_13_3/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_13_3/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_13_3/PmD_aft_result_sig_slopes_all.rds'))
  
  #
  load('~/Dropbox/model_alt/5_3_14_1/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_14_1/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_14_1/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_14_1/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_14_1/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_14_1/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_1/PmD_aft_result_sig_slopes_all.rds'))
  
  #
  load('~/Dropbox/model_alt/5_3_14_2/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_14_2/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_14_2/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_14_2/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_14_2/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_14_2/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_2/PmD_aft_result_sig_slopes_all.rds'))
  
  #
  load('~/Dropbox/model_alt/5_3_14_3/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_14_3/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_14_3/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_14_3/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_14_3/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_14_3/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_14_3/PmD_aft_result_sig_slopes_all.rds'))
  
  #
  load('~/Dropbox/model_alt/5_3_28_2/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_28_2/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_28_2/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_28_2/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_28_2/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_28_2/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_2/PmD_aft_result_sig_slopes_all.rds'))
  
  #
  load('~/Dropbox/model_alt/5_3_28_3/alphabeta_M1_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
  M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
  M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_28_3/alphabeta_M1_percsig_all.RData')
  M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + percsig_aft_cue_nums
  M1_bfr_res_nums_sum <- M1_bfr_res_nums_sum + percsig_bfr_res_nums
  M1_aft_res_nums_sum <- M1_aft_res_nums_sum + percsig_aft_res_nums
  M1_unit_total <- M1_unit_total + percsig_total_units
  M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/M1_bfr_cue_all_slopes_all.rds'))
  M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/M1_aft_cue_all_slopes_all.rds'))
  M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/M1_bfr_result_all_slopes_all.rds'))
  M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/M1_aft_result_all_slopes_all.rds'))
  M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/M1_bfr_cue_sig_slopes_all.rds'))
  M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/M1_aft_cue_sig_slopes_all.rds'))
  M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/M1_bfr_result_sig_slopes_all.rds'))
  M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/M1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_28_3/alphabeta_S1_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
  S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
  S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_28_3/alphabeta_S1_percsig_all.RData')
  S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + percsig_bfr_cue_nums
  S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + percsig_aft_cue_nums
  S1_bfr_res_nums_sum <- S1_bfr_res_nums_sum + percsig_bfr_res_nums
  S1_aft_res_nums_sum <- S1_aft_res_nums_sum + percsig_aft_res_nums
  S1_unit_total <- S1_unit_total + percsig_total_units
  S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/S1_bfr_cue_all_slopes_all.rds'))
  S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/S1_aft_cue_all_slopes_all.rds'))
  S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/S1_bfr_result_all_slopes_all.rds'))
  S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/S1_aft_result_all_slopes_all.rds'))
  S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/S1_bfr_cue_sig_slopes_all.rds'))
  S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/S1_aft_cue_sig_slopes_all.rds'))
  S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/S1_bfr_result_sig_slopes_all.rds'))
  S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/S1_aft_result_sig_slopes_all.rds'))
  
  load('~/Dropbox/model_alt/5_3_28_3/alphabeta_PmD_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
  PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
  PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
  load('~/Dropbox/model_alt/5_3_28_3/alphabeta_PmD_percsig_all.RData')
  PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + percsig_bfr_cue_nums
  PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + percsig_aft_cue_nums
  PmD_bfr_res_nums_sum <- PmD_bfr_res_nums_sum + percsig_bfr_res_nums
  PmD_aft_res_nums_sum <- PmD_aft_res_nums_sum + percsig_aft_res_nums
  PmD_unit_total <- PmD_unit_total + percsig_total_units
  PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/PmD_bfr_cue_all_slopes_all.rds'))
  PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/PmD_aft_cue_all_slopes_all.rds'))
  PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/PmD_bfr_result_all_slopes_all.rds'))
  PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/PmD_aft_result_all_slopes_all.rds'))
  PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/PmD_bfr_cue_sig_slopes_all.rds'))
  PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/PmD_aft_cue_sig_slopes_all.rds'))
  PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/PmD_bfr_result_sig_slopes_all.rds'))
  PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/Dropbox/model_alt/5_3_28_3/PmD_aft_result_sig_slopes_all.rds'))

}

save.image(file='alt_combined_info.RData')
rm(list=ls())

#################


region_list <- c('M1','S1','PmD')

for (region_index in 1:length(region_list)){

  file_list <- Sys.glob(paste('sig_slopes*',region_list[region_index],'*.xlsx',sep=""))
  
  for (i in 1:length(file_list)){
    cat(file_list[i],'\n')
    if (i==1){
      slopes_bfr_cue <- read.xlsx(file_list[i],sheet='slopes_bfr_cue_model',colNames=T)
      slopes_aft_cue <- read.xlsx(file_list[i],sheet='slopes_aft_cue_model',colNames=T)
      slopes_bfr_result <- read.xlsx(file_list[i],sheet='slopes_bfr_result_model',colNames=T)
      slopes_aft_result <- read.xlsx(file_list[i],sheet='slopes_aft_result_model',colNames=T)
      
      slopes_bfr_cue_sigall <- read.xlsx(file_list[i],sheet='sig_all_bfr_cue_model',colNames=T)
      slopes_aft_cue_sigall <- read.xlsx(file_list[i],sheet='sig_all_aft_cue_model',colNames=T)
      slopes_bfr_result_sigall <- read.xlsx(file_list[i],sheet='sig_all_bfr_result_model',colNames=T)
      slopes_aft_result_sigall <- read.xlsx(file_list[i],sheet='sig_all_aft_result_model',colNames=T)
      
    }else{
      temp_bfr_cue <- read.xlsx(file_list[i],sheet='slopes_bfr_cue_model',colNames=T)
      temp_aft_cue <- read.xlsx(file_list[i],sheet='slopes_aft_cue_model',colNames=T)
      temp_bfr_result <- read.xlsx(file_list[i],sheet='slopes_bfr_result_model',colNames=T)
      temp_aft_result <- read.xlsx(file_list[i],sheet='slopes_aft_result_model',colNames=T)
      
      temp_bfr_cue_sigall <- read.xlsx(file_list[i],sheet='sig_all_bfr_cue_model',colNames=T)
      temp_aft_cue_sigall <- read.xlsx(file_list[i],sheet='sig_all_aft_cue_model',colNames=T)
      temp_bfr_result_sigall <- read.xlsx(file_list[i],sheet='sig_all_bfr_result_model',colNames=T)
      temp_aft_result_sigall <- read.xlsx(file_list[i],sheet='sig_all_aft_result_model',colNames=T)
      
      slopes_bfr_cue <- rbind(slopes_bfr_cue,temp_bfr_cue)
      slopes_aft_cue <- rbind(slopes_aft_cue,temp_aft_cue)
      slopes_bfr_result <- rbind(slopes_bfr_result,temp_bfr_result)
      slopes_aft_result <- rbind(slopes_aft_result,temp_aft_result)
      
      slopes_bfr_cue_sigall <- rbind(slopes_bfr_cue_sigall,temp_bfr_cue_sigall)
      slopes_aft_cue_sigall <- rbind(slopes_aft_cue_sigall,temp_aft_cue_sigall)
      slopes_bfr_result_sigall <- rbind(slopes_bfr_result_sigall,temp_bfr_result_sigall)
      slopes_aft_result_sigall <- rbind(slopes_aft_result_sigall,temp_aft_result_sigall)
    }
    
    assign(paste(region_list[region_index],"_slopes_bfr_cue",sep=""),slopes_bfr_cue)
    assign(paste(region_list[region_index],"_slopes_aft_cue",sep=""),slopes_aft_cue)
    assign(paste(region_list[region_index],"_slopes_bfr_result",sep=""),slopes_bfr_result)
    assign(paste(region_list[region_index],"_slopes_aft_result",sep=""),slopes_aft_result)
    
    assign(paste(region_list[region_index],"_slopes_bfr_cue_sigall",sep=""),slopes_bfr_cue_sigall)
    assign(paste(region_list[region_index],"_slopes_aft_cue_sigall",sep=""),slopes_aft_cue_sigall)
    assign(paste(region_list[region_index],"_slopes_bfr_result_sigall",sep=""),slopes_bfr_result_sigall)
    assign(paste(region_list[region_index],"_slopes_aft_result_sigall",sep=""),slopes_aft_result_sigall)
    
    
  }
  
  
  
}

save.image(file='alt_combined_xlsx_info.RData')
rm(list=ls())



