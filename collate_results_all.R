library(openxlsx)
library(ggplot2)
library(reshape2)
#source("~/documents/lab/workspace/Classification_scripts/multiplot.R")
source("~/workspace/classification_scripts/multiplot.R")
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


nhp_id <- '504'

if (nhp_id == '0059'){

   ##################
   #### 0059 ########
   ##################

   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/alphabeta_S1_all.RData')
   S1_bfr_cue_nums_sum <- bfr_cue_nums
   S1_aft_cue_nums_sum <- aft_cue_nums
   S1_bfr_result_nums_sum <- bfr_result_nums
   S1_aft_result_nums_sum <- aft_result_nums
   S1_bfr_cue_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/S1_bfr_cue_all_slopes_all.rds')
   S1_aft_cue_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/S1_aft_cue_all_slopes_all.rds')
   S1_bfr_result_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/S1_bfr_result_all_slopes_all.rds')
   S1_aft_result_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/S1_aft_result_all_slopes_all.rds')
   S1_bfr_cue_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/S1_bfr_cue_sig_slopes_all.rds')
   S1_aft_cue_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/S1_aft_cue_sig_slopes_all.rds')
   S1_bfr_result_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/S1_bfr_result_sig_slopes_all.rds')
   S1_aft_result_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/S1_aft_result_sig_slopes_all.rds')

   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/alphabeta_PmD_all.RData')
   PmD_bfr_cue_nums_sum <- bfr_cue_nums
   PmD_aft_cue_nums_sum <- aft_cue_nums
   PmD_bfr_result_nums_sum <- bfr_result_nums
   PmD_aft_result_nums_sum <- aft_result_nums
   PmD_bfr_cue_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/PmD_bfr_cue_all_slopes_all.rds')
   PmD_aft_cue_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/PmD_aft_cue_all_slopes_all.rds')
   PmD_bfr_result_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/PmD_bfr_result_all_slopes_all.rds')
   PmD_aft_result_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/PmD_aft_result_all_slopes_all.rds')
   PmD_bfr_cue_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/PmD_bfr_cue_sig_slopes_all.rds')
   PmD_aft_cue_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/PmD_aft_cue_sig_slopes_all.rds')
   PmD_bfr_result_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/PmD_bfr_result_sig_slopes_all.rds')
   PmD_aft_result_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/PmD_aft_result_sig_slopes_all.rds')
   
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/alphabeta_M1_all.RData')
   M1_bfr_cue_nums_sum <- bfr_cue_nums
   M1_aft_cue_nums_sum <- aft_cue_nums
   M1_bfr_result_nums_sum <- bfr_result_nums
   M1_aft_result_nums_sum <- aft_result_nums
   M1_bfr_cue_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/M1_bfr_cue_all_slopes_all.rds')
   M1_aft_cue_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/M1_aft_cue_all_slopes_all.rds')
   M1_bfr_result_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/M1_bfr_result_all_slopes_all.rds')
   M1_aft_result_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/M1_aft_result_all_slopes_all.rds')
   M1_bfr_cue_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/M1_bfr_cue_sig_slopes_all.rds')
   M1_aft_cue_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/M1_aft_cue_sig_slopes_all.rds')
   M1_bfr_result_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/M1_bfr_result_sig_slopes_all.rds')
   M1_aft_result_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_1/M1_aft_result_sig_slopes_all.rds')

   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/alphabeta_M1_all.RData')
   M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
   M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
   M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
   M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
   M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/M1_bfr_cue_all_slopes_all.rds'))
   M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/M1_aft_cue_all_slopes_all.rds'))
   M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/M1_bfr_result_all_slopes_all.rds'))
   M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/M1_aft_result_all_slopes_all.rds'))
   M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/M1_bfr_cue_sig_slopes_all.rds'))
   M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/M1_aft_cue_sig_slopes_all.rds'))
   M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/M1_bfr_result_sig_slopes_all.rds'))
   M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/M1_aft_result_sig_slopes_all.rds'))

   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/alphabeta_S1_all.RData')
   S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
   S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
   S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
   S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
   S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/S1_bfr_cue_all_slopes_all.rds'))
   S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/S1_aft_cue_all_slopes_all.rds'))
   S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/S1_bfr_result_all_slopes_all.rds'))
   S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/S1_aft_result_all_slopes_all.rds'))
   S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/S1_bfr_cue_sig_slopes_all.rds'))
   S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/S1_aft_cue_sig_slopes_all.rds'))
   S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/S1_bfr_result_sig_slopes_all.rds'))
   S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/S1_aft_result_sig_slopes_all.rds'))

   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/alphabeta_PmD_all.RData')
   PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
   PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
   PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
   PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
   PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/PmD_bfr_cue_all_slopes_all.rds'))
   PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/PmD_aft_cue_all_slopes_all.rds'))
   PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/PmD_bfr_result_all_slopes_all.rds'))
   PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/PmD_aft_result_all_slopes_all.rds')) 
   PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/PmD_bfr_cue_sig_slopes_all.rds'))
   PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/PmD_aft_cue_sig_slopes_all.rds')) 
   PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/PmD_bfr_result_sig_slopes_all.rds'))
   PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_8_2/PmD_aft_result_sig_slopes_all.rds'))

   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/alphabeta_M1_all.RData')
   M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
   M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
   M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
   M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
   M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/M1_bfr_cue_all_slopes_all.rds'))
   M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/M1_aft_cue_all_slopes_all.rds'))
   M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/M1_bfr_result_all_slopes_all.rds'))
   M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/M1_aft_result_all_slopes_all.rds'))
   M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/M1_bfr_cue_sig_slopes_all.rds'))
   M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/M1_aft_cue_sig_slopes_all.rds'))
   M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/M1_bfr_result_sig_slopes_all.rds'))
   M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/M1_aft_result_sig_slopes_all.rds'))
   
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/alphabeta_S1_all.RData')
   S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
   S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
   S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
   S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
   S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/S1_bfr_cue_all_slopes_all.rds'))
   S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/S1_aft_cue_all_slopes_all.rds'))
   S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/S1_bfr_result_all_slopes_all.rds'))
   S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/S1_aft_result_all_slopes_all.rds'))
   S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/S1_bfr_cue_sig_slopes_all.rds'))
   S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/S1_aft_cue_sig_slopes_all.rds'))
   S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/S1_bfr_result_sig_slopes_all.rds'))
   S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/S1_aft_result_sig_slopes_all.rds'))

   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/alphabeta_PmD_all.RData')
   PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
   PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
   PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
   PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
   PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/PmD_bfr_cue_all_slopes_all.rds'))
   PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/PmD_aft_cue_all_slopes_all.rds'))
   PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/PmD_bfr_result_all_slopes_all.rds'))
   PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/PmD_aft_result_all_slopes_all.rds'))
   PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/PmD_bfr_cue_sig_slopes_all.rds'))
   PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/PmD_aft_cue_sig_slopes_all.rds'))
   PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/PmD_bfr_result_sig_slopes_all.rds'))
   PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_1/PmD_aft_result_sig_slopes_all.rds'))

   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/alphabeta_M1_all.RData')
   M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
   M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
   M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
   M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
   M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/M1_bfr_cue_all_slopes_all.rds'))
   M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/M1_aft_cue_all_slopes_all.rds'))
   M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/M1_bfr_result_all_slopes_all.rds'))
   M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/M1_aft_result_all_slopes_all.rds'))
   M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/M1_bfr_cue_sig_slopes_all.rds'))
   M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/M1_aft_cue_sig_slopes_all.rds'))
   M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/M1_bfr_result_sig_slopes_all.rds'))
   M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/M1_aft_result_sig_slopes_all.rds'))

   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/alphabeta_S1_all.RData')
   S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
   S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
   S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
   S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
   S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/S1_bfr_cue_all_slopes_all.rds'))
   S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/S1_aft_cue_all_slopes_all.rds'))
   S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/S1_bfr_result_all_slopes_all.rds'))
   S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/S1_aft_result_all_slopes_all.rds'))
   S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/S1_bfr_cue_sig_slopes_all.rds'))
   S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/S1_aft_cue_sig_slopes_all.rds'))
   S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/S1_bfr_result_sig_slopes_all.rds'))
   S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/S1_aft_result_sig_slopes_all.rds'))

   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/alphabeta_PmD_all.RData')
   PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
   PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
   PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
   PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
   PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/PmD_bfr_cue_all_slopes_all.rds'))
   PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/PmD_aft_cue_all_slopes_all.rds'))
   PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/PmD_bfr_result_all_slopes_all.rds'))
   PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/PmD_aft_result_all_slopes_all.rds'))
   PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/PmD_bfr_cue_sig_slopes_all.rds'))
   PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/PmD_aft_cue_sig_slopes_all.rds'))
   PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/PmD_bfr_result_sig_slopes_all.rds'))
   PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/0_9_2/PmD_aft_result_sig_slopes_all.rds'))
}else if (nhp_id == '504'){
   ##################
   #### 504 #########
   ##################

   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/alphabeta_S1_all.RData')
   S1_bfr_cue_nums_sum <- bfr_cue_nums
   S1_aft_cue_nums_sum <- aft_cue_nums
   S1_bfr_result_nums_sum <- bfr_result_nums
   S1_aft_result_nums_sum <- aft_result_nums
   S1_bfr_cue_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/S1_bfr_cue_all_slopes_all.rds')
   S1_aft_cue_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/S1_aft_cue_all_slopes_all.rds')
   S1_bfr_result_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/S1_bfr_result_all_slopes_all.rds')
   S1_aft_result_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/S1_aft_result_all_slopes_all.rds')
   S1_bfr_cue_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/S1_bfr_cue_sig_slopes_all.rds')
   S1_aft_cue_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/S1_aft_cue_sig_slopes_all.rds')
   S1_bfr_result_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/S1_bfr_result_sig_slopes_all.rds')
   S1_aft_result_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/S1_aft_result_sig_slopes_all.rds')
    
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/alphabeta_PmD_all.RData')
   PmD_bfr_cue_nums_sum <- bfr_cue_nums
   PmD_aft_cue_nums_sum <- aft_cue_nums
   PmD_bfr_result_nums_sum <- bfr_result_nums
   PmD_aft_result_nums_sum <- aft_result_nums
   PmD_bfr_cue_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/PmD_bfr_cue_all_slopes_all.rds')
   PmD_aft_cue_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/PmD_aft_cue_all_slopes_all.rds')
   PmD_bfr_result_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/PmD_bfr_result_all_slopes_all.rds')
   PmD_aft_result_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/PmD_aft_result_all_slopes_all.rds')
   PmD_bfr_cue_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/PmD_bfr_cue_sig_slopes_all.rds')
   PmD_aft_cue_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/PmD_aft_cue_sig_slopes_all.rds')
   PmD_bfr_result_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/PmD_bfr_result_sig_slopes_all.rds')
   PmD_aft_result_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/PmD_aft_result_sig_slopes_all.rds')
   
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/alphabeta_M1_all.RData')
   M1_bfr_cue_nums_sum <- bfr_cue_nums
   M1_aft_cue_nums_sum <- aft_cue_nums
   M1_bfr_result_nums_sum <- bfr_result_nums
   M1_aft_result_nums_sum <- aft_result_nums
   M1_bfr_cue_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/M1_bfr_cue_all_slopes_all.rds')
   M1_aft_cue_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/M1_aft_cue_all_slopes_all.rds')
   M1_bfr_result_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/M1_bfr_result_all_slopes_all.rds')
   M1_aft_result_all_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/M1_aft_result_all_slopes_all.rds')
   M1_bfr_cue_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/M1_bfr_cue_sig_slopes_all.rds')
   M1_aft_cue_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/M1_aft_cue_sig_slopes_all.rds')
   M1_bfr_result_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/M1_bfr_result_sig_slopes_all.rds')
   M1_aft_result_sig_slopes <- readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_1/M1_aft_result_sig_slopes_all.rds')
 
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/alphabeta_M1_all.RData')
   M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
   M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
   M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
   M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
   M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/M1_bfr_cue_all_slopes_all.rds'))
   M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/M1_aft_cue_all_slopes_all.rds'))
   M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/M1_bfr_result_all_slopes_all.rds'))
   M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/M1_aft_result_all_slopes_all.rds'))
   M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/M1_bfr_cue_sig_slopes_all.rds'))
   M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/M1_aft_cue_sig_slopes_all.rds'))
   M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/M1_bfr_result_sig_slopes_all.rds'))
   M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/M1_aft_result_sig_slopes_all.rds'))
    
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/alphabeta_S1_all.RData')
   S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
   S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
   S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
   S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
   S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/S1_bfr_cue_all_slopes_all.rds'))
   S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/S1_aft_cue_all_slopes_all.rds'))
   S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/S1_bfr_result_all_slopes_all.rds'))
   S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/S1_aft_result_all_slopes_all.rds'))
   S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/S1_bfr_cue_sig_slopes_all.rds'))
   S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/S1_aft_cue_sig_slopes_all.rds'))
   S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/S1_bfr_result_sig_slopes_all.rds'))
   S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/S1_aft_result_sig_slopes_all.rds'))
 
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/alphabeta_PmD_all.RData')
   PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
   PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
   PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
   PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
   PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/PmD_bfr_cue_all_slopes_all.rds'))
   PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/PmD_aft_cue_all_slopes_all.rds'))
   PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/PmD_bfr_result_all_slopes_all.rds'))
   PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/PmD_aft_result_all_slopes_all.rds'))
   PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/PmD_bfr_cue_sig_slopes_all.rds'))
   PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/PmD_aft_cue_sig_slopes_all.rds'))
   PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/PmD_bfr_result_sig_slopes_all.rds'))
   PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_8_2/PmD_aft_result_sig_slopes_all.rds'))
   
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/alphabeta_M1_all.RData')
   M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
   M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
   M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
   M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
   M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/M1_bfr_cue_all_slopes_all.rds'))
   M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/M1_aft_cue_all_slopes_all.rds'))
   M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/M1_bfr_result_all_slopes_all.rds'))
   M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/M1_aft_result_all_slopes_all.rds'))
   M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/M1_bfr_cue_sig_slopes_all.rds'))
   M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/M1_aft_cue_sig_slopes_all.rds'))
   M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/M1_bfr_result_sig_slopes_all.rds'))
   M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/M1_aft_result_sig_slopes_all.rds'))

   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/alphabeta_S1_all.RData')
   S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
   S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
   S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
   S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
   S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/S1_bfr_cue_all_slopes_all.rds'))
   S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/S1_aft_cue_all_slopes_all.rds'))
   S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/S1_bfr_result_all_slopes_all.rds'))
   S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/S1_aft_result_all_slopes_all.rds'))
   S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/S1_bfr_cue_sig_slopes_all.rds'))
   S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/S1_aft_cue_sig_slopes_all.rds'))
   S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/S1_bfr_result_sig_slopes_all.rds'))
   S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/S1_aft_result_sig_slopes_all.rds'))
 
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/alphabeta_PmD_all.RData')
   PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
   PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
   PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
   PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
   PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/PmD_bfr_cue_all_slopes_all.rds'))
   PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/PmD_aft_cue_all_slopes_all.rds'))
   PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/PmD_bfr_result_all_slopes_all.rds'))
   PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/PmD_aft_result_all_slopes_all.rds'))
   PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/PmD_bfr_cue_sig_slopes_all.rds'))
   PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/PmD_aft_cue_sig_slopes_all.rds'))
   PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/PmD_bfr_result_sig_slopes_all.rds'))
   PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_1/PmD_aft_result_sig_slopes_all.rds'))
 
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/alphabeta_M1_all.RData')
   M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
   M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
   M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
   M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
   M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/M1_bfr_cue_all_slopes_all.rds'))
   M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/M1_aft_cue_all_slopes_all.rds'))
   M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/M1_bfr_result_all_slopes_all.rds'))
   M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/M1_aft_result_all_slopes_all.rds'))
   M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/M1_bfr_cue_sig_slopes_all.rds'))
   M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/M1_aft_cue_sig_slopes_all.rds'))
   M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/M1_bfr_result_sig_slopes_all.rds'))
   M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/M1_aft_result_sig_slopes_all.rds'))
 
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/alphabeta_S1_all.RData')
   S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
   S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
   S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
   S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
   S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/S1_bfr_cue_all_slopes_all.rds'))
   S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/S1_aft_cue_all_slopes_all.rds'))
   S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/S1_bfr_result_all_slopes_all.rds'))
   S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/S1_aft_result_all_slopes_all.rds'))
   S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/S1_bfr_cue_sig_slopes_all.rds'))
   S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/S1_aft_cue_sig_slopes_all.rds'))
   S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/S1_bfr_result_sig_slopes_all.rds'))
   S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/S1_aft_result_sig_slopes_all.rds'))
 
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/alphabeta_PmD_all.RData')
   PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
   PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
   PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
   PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
   PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/PmD_bfr_cue_all_slopes_all.rds'))
   PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/PmD_aft_cue_all_slopes_all.rds'))
   PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/PmD_bfr_result_all_slopes_all.rds'))
   PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/PmD_aft_result_all_slopes_all.rds'))
   PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/PmD_bfr_cue_sig_slopes_all.rds'))
   PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/PmD_aft_cue_sig_slopes_all.rds'))
   PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/PmD_bfr_result_sig_slopes_all.rds'))
   PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_9_2/PmD_aft_result_sig_slopes_all.rds'))
 
   #load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/alphabeta_M1_all.RData')
   #M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
   #M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
   #M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
   #M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
   #M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/M1_bfr_cue_all_slopes_all.rds'))
   #M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/M1_aft_cue_all_slopes_all.rds'))
   #M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/M1_bfr_result_all_slopes_all.rds'))
   #M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/M1_aft_result_all_slopes_all.rds'))
   #M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/M1_bfr_cue_sig_slopes_all.rds'))
   #M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/M1_aft_cue_sig_slopes_all.rds'))
   #M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/M1_bfr_result_sig_slopes_all.rds'))
   #M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/M1_aft_result_sig_slopes_all.rds'))
 
   #load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/alphabeta_S1_all.RData')
   #S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
   #S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
   #S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
   #S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
   #S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/S1_bfr_cue_all_slopes_all.rds'))
   #S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/S1_aft_cue_all_slopes_all.rds'))
   #S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/S1_bfr_result_all_slopes_all.rds'))
   #S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/S1_aft_result_all_slopes_all.rds'))
   #S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/S1_bfr_cue_sig_slopes_all.rds'))
   #S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/S1_aft_cue_sig_slopes_all.rds'))
   #S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/S1_bfr_result_sig_slopes_all.rds'))
   #S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/S1_aft_result_sig_slopes_all.rds'))
 
   #load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/alphabeta_PmD_all.RData')
   #PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
   #PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
   #PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
   #PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
   #PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/PmD_bfr_cue_all_slopes_all.rds'))
   #PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/PmD_aft_cue_all_slopes_all.rds'))
   #PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/PmD_bfr_result_all_slopes_all.rds'))
   #PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/PmD_aft_result_all_slopes_all.rds'))
   #PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/PmD_bfr_cue_sig_slopes_all.rds'))
   #PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/PmD_aft_cue_sig_slopes_all.rds'))
   #PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/PmD_bfr_result_sig_slopes_all.rds'))
   #PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_1/PmD_aft_result_sig_slopes_all.rds'))
 
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/alphabeta_M1_all.RData')
   M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
   M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
   M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
   M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
   M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/M1_bfr_cue_all_slopes_all.rds'))
   M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/M1_aft_cue_all_slopes_all.rds'))
   M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/M1_bfr_result_all_slopes_all.rds'))
   M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/M1_aft_result_all_slopes_all.rds'))
   M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/M1_bfr_cue_sig_slopes_all.rds'))
   M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/M1_aft_cue_sig_slopes_all.rds'))
   M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/M1_bfr_result_sig_slopes_all.rds'))
   M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/M1_aft_result_sig_slopes_all.rds'))
 
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/alphabeta_S1_all.RData')
   S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
   S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
   S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
   S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
   S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/S1_bfr_cue_all_slopes_all.rds'))
   S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/S1_aft_cue_all_slopes_all.rds'))
   S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/S1_bfr_result_all_slopes_all.rds'))
   S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/S1_aft_result_all_slopes_all.rds'))
   S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/S1_bfr_cue_sig_slopes_all.rds'))
   S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/S1_aft_cue_sig_slopes_all.rds'))
   S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/S1_bfr_result_sig_slopes_all.rds'))
   S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/S1_aft_result_sig_slopes_all.rds'))
    
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/alphabeta_PmD_all.RData')
   PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
   PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
   PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
   PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
   PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/PmD_bfr_cue_all_slopes_all.rds'))
   PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/PmD_aft_cue_all_slopes_all.rds'))
   PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/PmD_bfr_result_all_slopes_all.rds'))
   PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/PmD_aft_result_all_slopes_all.rds'))
   PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/PmD_bfr_cue_sig_slopes_all.rds'))
   PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/PmD_aft_cue_sig_slopes_all.rds'))
   PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/PmD_bfr_result_sig_slopes_all.rds'))
   PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_2/PmD_aft_result_sig_slopes_all.rds'))
    
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/alphabeta_M1_all.RData')
   M1_bfr_cue_nums_sum <- M1_bfr_cue_nums_sum + bfr_cue_nums
   M1_aft_cue_nums_sum <- M1_aft_cue_nums_sum + aft_cue_nums
   M1_bfr_result_nums_sum <- M1_bfr_result_nums_sum + bfr_result_nums
   M1_aft_result_nums_sum <- M1_aft_result_nums_sum + aft_result_nums
   M1_bfr_cue_all_slopes <- rbind(M1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/M1_bfr_cue_all_slopes_all.rds'))
   M1_aft_cue_all_slopes <- rbind(M1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/M1_aft_cue_all_slopes_all.rds'))
   M1_bfr_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/M1_bfr_result_all_slopes_all.rds'))
   M1_aft_result_all_slopes <- rbind(M1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/M1_aft_result_all_slopes_all.rds'))
   M1_bfr_cue_sig_slopes <- rbind(M1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/M1_bfr_cue_sig_slopes_all.rds'))
   M1_aft_cue_sig_slopes <- rbind(M1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/M1_aft_cue_sig_slopes_all.rds'))
   M1_bfr_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/M1_bfr_result_sig_slopes_all.rds'))
   M1_aft_result_sig_slopes <- rbind(M1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/M1_aft_result_sig_slopes_all.rds'))
 
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/alphabeta_S1_all.RData')
   S1_bfr_cue_nums_sum <- S1_bfr_cue_nums_sum + bfr_cue_nums
   S1_aft_cue_nums_sum <- S1_aft_cue_nums_sum + aft_cue_nums
   S1_bfr_result_nums_sum <- S1_bfr_result_nums_sum + bfr_result_nums
   S1_aft_result_nums_sum <- S1_aft_result_nums_sum + aft_result_nums
   S1_bfr_cue_all_slopes <- rbind(S1_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/S1_bfr_cue_all_slopes_all.rds'))
   S1_aft_cue_all_slopes <- rbind(S1_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/S1_aft_cue_all_slopes_all.rds'))
   S1_bfr_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/S1_bfr_result_all_slopes_all.rds'))
   S1_aft_result_all_slopes <- rbind(S1_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/S1_aft_result_all_slopes_all.rds'))
   S1_bfr_cue_sig_slopes <- rbind(S1_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/S1_bfr_cue_sig_slopes_all.rds'))
   S1_aft_cue_sig_slopes <- rbind(S1_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/S1_aft_cue_sig_slopes_all.rds'))
   S1_bfr_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/S1_bfr_result_sig_slopes_all.rds'))
   S1_aft_result_sig_slopes <- rbind(S1_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/S1_aft_result_sig_slopes_all.rds'))
 
   load('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/alphabeta_PmD_all.RData')
   PmD_bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum + bfr_cue_nums
   PmD_aft_cue_nums_sum <- PmD_aft_cue_nums_sum + aft_cue_nums
   PmD_bfr_result_nums_sum <- PmD_bfr_result_nums_sum + bfr_result_nums
   PmD_aft_result_nums_sum <- PmD_aft_result_nums_sum + aft_result_nums
   PmD_bfr_cue_all_slopes <- rbind(PmD_bfr_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/PmD_bfr_cue_all_slopes_all.rds'))
   PmD_aft_cue_all_slopes <- rbind(PmD_aft_cue_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/PmD_aft_cue_all_slopes_all.rds'))
   PmD_bfr_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/PmD_bfr_result_all_slopes_all.rds'))
   PmD_aft_result_all_slopes <- rbind(PmD_bfr_result_all_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/PmD_aft_result_all_slopes_all.rds'))
   PmD_bfr_cue_sig_slopes <- rbind(PmD_bfr_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/PmD_bfr_cue_sig_slopes_all.rds'))
   PmD_aft_cue_sig_slopes <- rbind(PmD_aft_cue_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/PmD_aft_cue_sig_slopes_all.rds'))
   PmD_bfr_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/PmD_bfr_result_sig_slopes_all.rds'))
   PmD_aft_result_sig_slopes <- rbind(PmD_bfr_result_sig_slopes,readRDS('~/moved_from_dropbox/model_backup/avg_ab_z_nl/5_14_3/PmD_aft_result_sig_slopes_all.rds'))
 
}

#########
#labs <- c("both pos","both neg","alpha pos","beta pos")
labs <- c("both pos","both neg","beta pos","alpha pos")
region_list <- c('M1','S1','PmD')

for (region_index in 1:length(region_list)){
    if (region_list[region_index] == 'M1'){
      bfr_cue_nums_sum <- M1_bfr_cue_nums_sum
      aft_cue_nums_sum <- M1_aft_cue_nums_sum
      bfr_result_nums_sum <- M1_bfr_result_nums_sum
      aft_result_nums_sum <- M1_aft_result_nums_sum
      bfr_cue_all_slopes <- M1_bfr_cue_all_slopes
      bfr_cue_sig_slopes <- M1_bfr_cue_sig_slopes
      aft_cue_all_slopes <- M1_aft_cue_all_slopes
      aft_cue_sig_slopes <- M1_aft_cue_sig_slopes
      bfr_result_all_slopes <- M1_bfr_result_all_slopes
      bfr_result_sig_slopes <- M1_bfr_result_sig_slopes
      aft_result_all_slopes <- M1_aft_result_all_slopes
      aft_result_sig_slopes <- M1_aft_result_sig_slopes
    }else if(region_list[region_index] == 'S1'){
      bfr_cue_nums_sum <- S1_bfr_cue_nums_sum
      aft_cue_nums_sum <- S1_aft_cue_nums_sum
      bfr_result_nums_sum <- S1_bfr_result_nums_sum
      aft_result_nums_sum <- S1_aft_result_nums_sum     
      bfr_cue_all_slopes <- S1_bfr_cue_all_slopes
      bfr_cue_sig_slopes <- S1_bfr_cue_sig_slopes
      aft_cue_all_slopes <- S1_aft_cue_all_slopes
      aft_cue_sig_slopes <- S1_aft_cue_sig_slopes
      bfr_result_all_slopes <- S1_bfr_result_all_slopes
      bfr_result_sig_slopes <- S1_bfr_result_sig_slopes
      aft_result_all_slopes <- S1_aft_result_all_slopes
      aft_result_sig_slopes <- S1_aft_result_sig_slopes
    }else if(region_list[region_index] == 'PmD'){
      bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum
      aft_cue_nums_sum <- PmD_aft_cue_nums_sum
      bfr_result_nums_sum <- PmD_bfr_result_nums_sum
      aft_result_nums_sum <- PmD_aft_result_nums_sum
      bfr_cue_all_slopes <- PmD_bfr_cue_all_slopes
      bfr_cue_sig_slopes <- PmD_bfr_cue_sig_slopes
      aft_cue_all_slopes <- PmD_aft_cue_all_slopes
      aft_cue_sig_slopes <- PmD_aft_cue_sig_slopes
      bfr_result_all_slopes <- PmD_bfr_result_all_slopes
      bfr_result_sig_slopes <- PmD_bfr_result_sig_slopes
      aft_result_all_slopes <- PmD_aft_result_all_slopes
      aft_result_sig_slopes <- PmD_aft_result_sig_slopes
    }

  bfr_cue_df <-data.frame(perc=bfr_cue_nums_sum/sum(bfr_cue_nums_sum),labs,type='bfr_cue')
  aft_cue_df <-data.frame(perc=aft_cue_nums_sum/sum(aft_cue_nums_sum),labs,type='aft_cue')
  bfr_result_df <-data.frame(perc=bfr_result_nums_sum/sum(bfr_result_nums_sum),labs,type='bfr_result')
  aft_result_df <-data.frame(perc=aft_result_nums_sum/sum(aft_result_nums_sum),labs,type='aft_result')
  
  x <- rev(order(bfr_cue_df$labs))
  
  bfr_cue_df <- bfr_cue_df[rev(order(bfr_cue_df$labs)),]
  aft_cue_df <- aft_cue_df[rev(order(aft_cue_df$labs)),]
  bfr_result_df <- bfr_result_df[rev(order(bfr_result_df$labs)),]
  aft_result_df <- aft_result_df[rev(order(aft_result_df$labs)),]
  
  bfr_cue_df <- ddply(bfr_cue_df,.(type),transform,position=(cumsum(bfr_cue_df$perc)-0.5*bfr_cue_df$perc))
  aft_cue_df <- ddply(aft_cue_df,.(type),transform,position=(cumsum(aft_cue_df$perc)-0.5*aft_cue_df$perc))
  bfr_result_df <- ddply(bfr_result_df,.(type),transform,position=(cumsum(bfr_result_df$perc)-0.5*bfr_result_df$perc))
  aft_result_df <- ddply(aft_result_df,.(type),transform,position=(cumsum(aft_result_df$perc)-0.5*aft_result_df$perc))
  
  bfr_cue_df <- ddply(bfr_cue_df,.(type),transform,label=paste(scales::percent(bfr_cue_df$perc),' n=',bfr_cue_nums_sum,sep=""))
  aft_cue_df <- ddply(aft_cue_df,.(type),transform,label=paste(scales::percent(aft_cue_df$perc),' n=',aft_cue_nums_sum,sep=""))
  bfr_result_df <- ddply(bfr_result_df,.(type),transform,label=paste(scales::percent(bfr_result_df$perc),' n=',bfr_result_nums_sum,sep=""))
  aft_result_df <- ddply(aft_result_df,.(type),transform,label=paste(scales::percent(aft_result_df$perc),' n=',aft_result_nums_sum,sep=""))
  
  png(paste('ALL_all_signs_bar_plotted_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  
  df_all <- rbind(bfr_cue_df,aft_cue_df,bfr_result_df,aft_result_df)
  df_all <- df_all[which(df_all$perc > 0),]
  
  bar_plt <- ggplot() + geom_bar(aes(x=df_all$type,y=df_all$perc,fill=df_all$labs),data=df_all,stat="identity") 
  bar_plt <- bar_plt + labs(title=region_list[region_index],fill="",x="Time Window",y="Percentage") + scale_fill_manual(values=c("plum2","turquoise4","lightsalmon","royalblue"))
  bar_plt <- bar_plt + geom_text(aes(x=df_all$type,y=df_all$position,label=df_all$label),size=4,stat="identity")
  
  plot(bar_plt)
  graphics.off()
  
  #########################
  # png(paste('ALL_slope_collated_hist_',region_list[region_index],'_bfr_cue.png',sep=""),width=8,height=6,units="in",res=500)
  # 
  # sig_plt <- ggplot(bfr_cue_sig_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # sig_plt <- sig_plt + labs(title=paste(region_list[region_index],'bfr cue','significant')) 
  # all_plt <- ggplot(bfr_cue_all_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # all_plt <- all_plt + labs(title=paste(region_list[region_index],'bfr cue','all'))
  # 
  # 
  # 
  # multiplot(sig_plt,all_plt,cols=1)
  # graphics.off()
  # 
  # png(paste('ALL_slope_collated_hist_',region_list[region_index],'_aft_cue.png',sep=""),width=8,height=6,units="in",res=500)
  # 
  # sig_plt <- ggplot(aft_cue_sig_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # sig_plt <- sig_plt + labs(title=paste(region_list[region_index],'aft cue','significant')) 
  # all_plt <- ggplot(aft_cue_all_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # all_plt <- all_plt + labs(title=paste(region_list[region_index],'aft cue','all'))
  # 
  # multiplot(sig_plt,all_plt,cols=1)
  # graphics.off()
  # 
  # png(paste('ALL_slope_collated_hist_',region_list[region_index],'_bfr_result.png',sep=""),width=8,height=6,units="in",res=500)
  # 
  # sig_plt <- ggplot(bfr_result_sig_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # sig_plt <- sig_plt + labs(title=paste(region_list[region_index],'bfr result','significant')) 
  # all_plt <- ggplot(bfr_result_all_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # all_plt <- all_plt + labs(title=paste(region_list[region_index],'bfr result','all'))
  # 
  # multiplot(sig_plt,all_plt,cols=1)
  # graphics.off()
  # 
  # png(paste('ALL_slope_collated_hist_',region_list[region_index],'_aft_result.png',sep=""),width=8,height=6,units="in",res=500)
  # 
  # sig_plt <- ggplot(aft_result_sig_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2) 
  # sig_plt <- sig_plt + labs(title=paste(region_list[region_index],'aft result','significant')) 
  # all_plt <- ggplot(aft_result_all_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # all_plt <- all_plt + labs(title=paste(region_list[region_index],'aft result','all'))
  # 
  # multiplot(sig_plt,all_plt,cols=1)
  # graphics.off()
  # 
  
  png(paste('ALL_slope_collated_hist_',region_list[region_index],'_bfr_cue.png',sep=""),width=8,height=6,units="in",res=500)
  
  sig_plt <- ggplot(bfr_cue_sig_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(bfr_cue_sig_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(bfr_cue_sig_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'bfr cue','significant')) 

  all_plt <- ggplot(bfr_cue_all_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(bfr_cue_all_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(bfr_cue_all_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'bfr cue','all')) 
  
  multiplot(sig_plt,all_plt,cols=1)
  graphics.off()
  
  png(paste('ALL_slope_collated_hist_',region_list[region_index],'_aft_cue.png',sep=""),width=8,height=6,units="in",res=500)
  
  sig_plt <- ggplot(aft_cue_sig_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(aft_cue_sig_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(aft_cue_sig_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'aft cue','significant')) 
  
  all_plt <- ggplot(aft_cue_all_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(aft_cue_all_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(aft_cue_all_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'aft cue','all')) 
  
  multiplot(sig_plt,all_plt,cols=1)
  graphics.off()  
  
  png(paste('ALL_slope_collated_hist_',region_list[region_index],'_bfr_result.png',sep=""),width=8,height=6,units="in",res=500)
  
  sig_plt <- ggplot(bfr_result_sig_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(bfr_result_sig_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(bfr_result_sig_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'bfr result','significant')) 
  
  all_plt <- ggplot(bfr_result_all_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(bfr_result_all_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(bfr_result_all_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'bfr result','all')) 
  
  multiplot(sig_plt,all_plt,cols=1)
  graphics.off()  
  
  png(paste('ALL_slope_collated_hist_',region_list[region_index],'_aft_result.png',sep=""),width=8,height=6,units="in",res=500)
  
  sig_plt <- ggplot(aft_result_sig_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(aft_result_sig_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(aft_result_sig_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'aft result','significant')) 
  
  all_plt <- ggplot(aft_result_all_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(aft_result_all_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(aft_result_all_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'aft result','all')) 
  
  multiplot(sig_plt,all_plt,cols=1)
  graphics.off()
  
  
}


#save.image(file='rp_collated_info.RData')
#rm(list=ls())

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



save.image(file='rp_collated_info.RData')
#save.image(file='alt_combined_xlsx_info.RData')
rm(list=ls())

