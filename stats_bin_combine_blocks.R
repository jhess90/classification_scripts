rm(list=ls())

library(openxlsx)
library(ggplot2)
library(reshape2)
library(zoo)
library(gplots)
library(RColorBrewer)
library(abind)
library(gridGraphics)
library(grid)
library(gridExtra)
library(R.matlab)
library(plyr)
library(dunn.test)
library(PMCMRplus)
library(xlsx)

tryCatch({
  source("~/documents/lab/workspace/Classification_scripts/multiplot.R")
  source("~/documents/lab/workspace/Classification_scripts/Ski.Mack.JH.R")
  print('on laptop')
},warning=function(war){(print('on beaver'))
  source("~/workspace/classification_scripts/multiplot.R")
  source("~/workspace/classification_scripts/Ski.Mack.JH.R")
},finally={print('sourced multiplot and ski mack')})

saveAsPng <- T
region_list <- c('M1','S1','PmD')
#ph_list_names <- c('comb','comb_outcome','m','p_catch','p','p_outcome','r_catch','r','r_outcome','v')
ph_list_names <- c('comb','comb_outcome','m','p_catch','p_outcome','r_catch','r_outcome','v')
time_windows <- c('ac','br','ar','rw')

nhp_id <- '504'
#nhp_id <- '0059'

if(nhp_id == '0059'){
  attach('0_3_10_1.RData')
  cat('0059\n')
  
  M1_sig_sign_percs_total <- M1_sig_sign_percs
  M1_total_unit_num <- length(M1_p_val_list$r0_p_vals[,1])
  S1_sig_sign_percs_total <- S1_sig_sign_percs
  S1_total_unit_num <- length(S1_p_val_list$r0_p_vals[,1])
  PmD_sig_sign_percs_total <- PmD_sig_sign_percs
  PmD_total_unit_num <- length(PmD_p_val_list$r0_p_vals[,1])
  
  M1_diffs_length_list_ac_total <- M1_diffs_length_list_ac
  M1_diffs_length_list_br_total <- M1_diffs_length_list_br
  M1_diffs_length_list_ar_total <- M1_diffs_length_list_ar
  M1_diffs_length_list_rw_total <- M1_diffs_length_list_rw
  S1_diffs_length_list_ac_total <- S1_diffs_length_list_ac
  S1_diffs_length_list_br_total <- S1_diffs_length_list_br
  S1_diffs_length_list_ar_total <- S1_diffs_length_list_ar
  S1_diffs_length_list_rw_total <- S1_diffs_length_list_rw
  PmD_diffs_length_list_ac_total <- PmD_diffs_length_list_ac
  PmD_diffs_length_list_br_total <- PmD_diffs_length_list_br
  PmD_diffs_length_list_ar_total <- PmD_diffs_length_list_ar
  PmD_diffs_length_list_rw_total <- PmD_diffs_length_list_rw
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        assign(paste(name,'_totals',sep=""),list())
        if(length(temp) > 0){
          #cat(name,'\n')
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  attach('0_3_10_2.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  #NOTE just using raw number of units, not percs. So perc sections of these arrays can be ignored
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_10_3.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_13_1.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_13_2.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_13_3.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_14_1.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_14_2.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_14_3.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_27_1.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_27_2.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_28_1.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_28_2.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_9_1.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_9_2.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('0_3_9_3.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()

  
}else if(nhp_id == '504'){
  attach('5_3_10_1.RData')
  cat('504\n')
  
  M1_sig_sign_percs_total <- M1_sig_sign_percs
  M1_total_unit_num <- length(M1_p_val_list$r0_p_vals[,1])
  S1_sig_sign_percs_total <- S1_sig_sign_percs
  S1_total_unit_num <- length(S1_p_val_list$r0_p_vals[,1])
  PmD_sig_sign_percs_total <- PmD_sig_sign_percs
  PmD_total_unit_num <- length(PmD_p_val_list$r0_p_vals[,1])
  
  M1_diffs_length_list_ac_total <- M1_diffs_length_list_ac
  M1_diffs_length_list_br_total <- M1_diffs_length_list_br
  M1_diffs_length_list_ar_total <- M1_diffs_length_list_ar
  M1_diffs_length_list_rw_total <- M1_diffs_length_list_rw
  S1_diffs_length_list_ac_total <- S1_diffs_length_list_ac
  S1_diffs_length_list_br_total <- S1_diffs_length_list_br
  S1_diffs_length_list_ar_total <- S1_diffs_length_list_ar
  S1_diffs_length_list_rw_total <- S1_diffs_length_list_rw
  PmD_diffs_length_list_ac_total <- PmD_diffs_length_list_ac
  PmD_diffs_length_list_br_total <- PmD_diffs_length_list_br
  PmD_diffs_length_list_ar_total <- PmD_diffs_length_list_ar
  PmD_diffs_length_list_rw_total <- PmD_diffs_length_list_rw
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        assign(paste(name,'_totals',sep=""),list())
        if(length(temp) > 0){
          #cat(name,'\n')
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  attach('5_3_10_2.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  #NOTE just using raw number of units, not percs. So perc sections of these arrays can be ignored
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  attach('5_3_10_3.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('5_3_13_1.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('5_3_13_2.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('5_3_13_3.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('5_3_14_1.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('5_3_14_2.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('5_3_14_3.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('5_3_28_2.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('5_3_28_3.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('5_3_9_1.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  attach('5_3_9_2.RData')
  M1_temp <- M1_sig_sign_percs
  M1_total_unit_num <- M1_total_unit_num + length(M1_p_val_list$r0_p_vals[,1])
  S1_temp <- S1_sig_sign_percs
  S1_total_unit_num <- S1_total_unit_num + length(S1_p_val_list$r0_p_vals[,1])
  PmD_temp <- PmD_sig_sign_percs
  PmD_total_unit_num <- PmD_total_unit_num + length(PmD_p_val_list$r0_p_vals[,1])
  
  for(name in names(M1_sig_sign_percs_total)){
    M1_sig_sign_percs_total[[name]] <- M1_sig_sign_percs_total[[name]] + M1_temp[[name]]
    S1_sig_sign_percs_total[[name]] <- S1_sig_sign_percs_total[[name]] + S1_temp[[name]]
    PmD_sig_sign_percs_total[[name]] <- PmD_sig_sign_percs_total[[name]] + PmD_temp[[name]]
  }
  for(name in names(M1_diffs_length_list_ac_total)){
    M1_diffs_length_list_ac_total[[name]] <- M1_diffs_length_list_ac_total[[name]] + M1_diffs_length_list_ac[[name]]
    M1_diffs_length_list_br_total[[name]] <- M1_diffs_length_list_br_total[[name]] + M1_diffs_length_list_br[[name]]
    M1_diffs_length_list_ar_total[[name]] <- M1_diffs_length_list_ar_total[[name]] + M1_diffs_length_list_ar[[name]]
    M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
    S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
    S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
    S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
    S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
    PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
    PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
    PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
    PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
  }
  for(region_name in region_list){
    for(ph_name in ph_list_names){
      for(window_name in time_windows){
        name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
        temp <- get(name)
        
        if(length(temp) > 0){
          temp2 <- c(get(paste(name,'_totals',sep="")),temp)
          assign(paste(name,'_totals',sep=""),temp2)
          
        }}}}
  detach()
  
  
}


#######################
###plot ###############
########################

for(region_index in 1:length(region_list)){
  total_unit_num <- get(paste(region_list[region_index],'_total_unit_num',sep=""))
  out_sig_sign_percs <- get(paste(region_list[region_index],'_sig_sign_percs_total',sep=""))
  
  window_names <- c('baseline vs \nafter cue','baseline vs \nbefore result','baseline vs \nafter result','baseline vs \nresult window','before result vs \nafter result','before result vs \nresult window')
  
  #reward
  png(paste(region_list[region_index],'_r_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$r0_sig_sign_percs[2,],out_sig_sign_percs$rx_sig_sign_percs[2,])
  rownames(num_inc) <- c(0,'x')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$r0_sig_sign_percs[3,],out_sig_sign_percs$rx_sig_sign_percs[3,])
  rownames(num_dec) <- c(0,'x')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Reward Level',y='Number of units')
  
  plot(plt)
  graphics.off()
  
  #punishment
  png(paste(region_list[region_index],'_p_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$p0_sig_sign_percs[2,],out_sig_sign_percs$px_sig_sign_percs[2,])
  rownames(num_inc) <- c(0,'x')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$p0_sig_sign_percs[3,],out_sig_sign_percs$px_sig_sign_percs[3,])
  rownames(num_dec) <- c(0,'x')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Punishment Level',y='Number of units')
  
  plot(plt)
  graphics.off()
  
  #rx outcome
  png(paste(region_list[region_index],'_rx_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$r0_s_sig_sign_percs[2,],out_sig_sign_percs$r0_f_sig_sign_percs[2,],out_sig_sign_percs$rx_s_sig_sign_percs[2,],out_sig_sign_percs$rx_f_sig_sign_percs[2,])
  rownames(num_inc) <- c('r0s','r0f','rxs','rxf')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$r0_f_sig_sign_percs[3,],out_sig_sign_percs$r0_f_sig_sign_percs[3,],out_sig_sign_percs$rx_s_sig_sign_percs[3,],out_sig_sign_percs$rx_f_sig_sign_percs[3,])
  rownames(num_dec) <- c('r0s','r0f','rxs','rxf')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Reward and Outcome',y='Number of units')
  
  plot(plt)
  graphics.off()
  
  #px outcome
  png(paste(region_list[region_index],'_px_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$p0_s_sig_sign_percs[2,],out_sig_sign_percs$p0_f_sig_sign_percs[2,],out_sig_sign_percs$px_s_sig_sign_percs[2,],out_sig_sign_percs$px_f_sig_sign_percs[2,])
  rownames(num_inc) <- c('p0s','p0f','pxs','pxf')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$p0_f_sig_sign_percs[3,],out_sig_sign_percs$p0_f_sig_sign_percs[3,],out_sig_sign_percs$px_s_sig_sign_percs[3,],out_sig_sign_percs$px_f_sig_sign_percs[3,])
  rownames(num_dec) <- c('p0s','p0f','pxs','pxf')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Punishment and Outcome',y='Number of units')
  
  plot(plt)
  graphics.off()
  
  #res outcome
  png(paste(region_list[region_index],'_res_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$res0_sig_sign_percs[2,],out_sig_sign_percs$res1_sig_sign_percs[2,])
  rownames(num_inc) <- c('fail','succ')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$res0_sig_sign_percs[3,],out_sig_sign_percs$res1_sig_sign_percs[3,])
  rownames(num_dec) <- c('fail','succ')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Result',y='Number of units')
  
  plot(plt)
  graphics.off()
  
  #comb
  png(paste(region_list[region_index],'_comb_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$r0_p0_sig_sign_percs[2,],out_sig_sign_percs$rx_p0_sig_sign_percs[2,],out_sig_sign_percs$r0_px_sig_sign_percs[2,],out_sig_sign_percs$rx_px_sig_sign_percs[2,])
  rownames(num_inc) <- c('r0p0','rxp0','r0px','rxpx')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$r0_p0_sig_sign_percs[3,],out_sig_sign_percs$rx_p0_sig_sign_percs[3,],out_sig_sign_percs$r0_px_sig_sign_percs[3,],out_sig_sign_percs$rx_px_sig_sign_percs[3,])
  rownames(num_dec) <- c('r0p0','rxp0','r0px','rxpx')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Combination',y='Number of units')
  plt <- plt + theme(axis.text.x = element_text(size=rel(0.8)))
  
  plot(plt)
  graphics.off()
  
  #comb outcome
  png(paste(region_list[region_index],'_comb_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$r0_p0_s_sig_sign_percs[2,],out_sig_sign_percs$rx_p0_s_sig_sign_percs[2,],out_sig_sign_percs$r0_px_s_sig_sign_percs[2,],out_sig_sign_percs$rx_px_s_sig_sign_percs[2,],out_sig_sign_percs$r0_p0_f_sig_sign_percs[2,],out_sig_sign_percs$rx_p0_f_sig_sign_percs[2,],out_sig_sign_percs$r0_px_f_sig_sign_percs[2,],out_sig_sign_percs$rx_px_f_sig_sign_percs[2,])
  rownames(num_inc) <- c('r0p0s','rxp0s','r0pxs','rxpxs','r0p0f','rxp0f','r0pxf','rxpxf')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$r0_p0_s_sig_sign_percs[3,],out_sig_sign_percs$rx_p0_s_sig_sign_percs[3,],out_sig_sign_percs$r0_px_s_sig_sign_percs[3,],out_sig_sign_percs$rx_px_s_sig_sign_percs[3,],out_sig_sign_percs$r0_p0_f_sig_sign_percs[3,],out_sig_sign_percs$rx_p0_f_sig_sign_percs[3,],out_sig_sign_percs$r0_px_f_sig_sign_percs[3,],out_sig_sign_percs$rx_px_f_sig_sign_percs[3,])
  rownames(num_dec) <- c('r0p0s','rxp0s','r0pxs','rxpxs','r0p0f','rxp0f','r0pxf','rxpxf')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Combination and Outcome',y='Number of units')
  plt <- plt + theme(axis.text.x = element_text(angle=45,hjust=1))
  
  plot(plt)
  graphics.off()
  
  #motivation
  png(paste(region_list[region_index],'_m_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$m0_sig_sign_percs[2,],out_sig_sign_percs$mx_sig_sign_percs[2,],out_sig_sign_percs$m2x_sig_sign_percs[2,])
  rownames(num_inc) <- c('m0','mx','m2x')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$m0_sig_sign_percs[3,],out_sig_sign_percs$mx_sig_sign_percs[3,],out_sig_sign_percs$m2x_sig_sign_percs[3,])
  rownames(num_dec) <- c('m0','mx','m2x')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Motivation',y='Number of units')
  plt <- plt + theme(axis.text.x = element_text(size=rel(0.8)))
  
  plot(plt)
  graphics.off()
  
  #value
  png(paste(region_list[region_index],'_v_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$v_x_sig_sign_percs[2,],out_sig_sign_percs$v0_sig_sign_percs[2,],out_sig_sign_percs$vx_sig_sign_percs[2,])
  rownames(num_inc) <- c('v_x','v0','vx')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$v_x_sig_sign_percs[3,],out_sig_sign_percs$v0_sig_sign_percs[3,],out_sig_sign_percs$vx_sig_sign_percs[3,])
  rownames(num_dec) <-c('v_x','v0','vx')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Value',y='Number of units')
  plt <- plt + theme(axis.text.x = element_text(size=rel(0.8)))
  
  plot(plt)
  graphics.off()
  
  #catch
  png(paste(region_list[region_index],'_catch_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$catch_x_sig_sign_percs[2,],out_sig_sign_percs$catchx_sig_sign_percs[2,])
  rownames(num_inc) <- c('P catch trial','R catch trial')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$res0_sig_sign_percs[3,],out_sig_sign_percs$res1_sig_sign_percs[3,])
  rownames(num_dec) <- c('P catch trial','R catch trial')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Catch trials',y='Number of units')
  plt <- plt + theme(axis.text.x = element_text(angle=45,hjust=1))
  
  plot(plt)
  graphics.off()
  
  
  
  #####
  #####
  diffs_ac <- get(paste(region_list[region_index],'_diffs_length_list_ac_total',sep=""))
  diffs_br <- get(paste(region_list[region_index],'_diffs_length_list_br_total',sep=""))
  diffs_ar <- get(paste(region_list[region_index],'_diffs_length_list_ar_total',sep=""))
  diffs_rw <- get(paste(region_list[region_index],'_diffs_length_list_rw_total',sep=""))
  
  all_diffs_length <- rbind(aft_cue=diffs_ac,bfr_res=diffs_br,aft_res=diffs_ar,res_wind=diffs_rw)
  
  assign(paste(region_list[region_index],'_all_diffs_length',sep=""),all_diffs_length)
  write.table(all_diffs_length,file=paste(region_list[region_index],'_all_diffs_length.csv',sep=""),sep=",",col.names=NA)
}

for(region_name in region_list){
  for(ph_name in ph_list_names){
    perc_list_windows <- list()
    #perc_list_windows_cbind <- c()
    for(window_name in time_windows){
      name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,'_totals',sep="")
      temp <- get(name)
      if(length(temp) > 0){
        
        sig_unit_num <- length(temp)
        cat(name,':',sig_unit_num,'total sig units\n')
        ph_sig_num <- 0
        
        ph_list <- list()
        for(i in 1:sig_unit_num){
          sig_comparisons <- temp[[i]]$comparisons[temp[[i]]$P < 0.05]
          if(length(sig_comparisons) > 0){
            cat('unit:',i,sig_comparisons,'\n')
            ph_list[[ph_sig_num + 1]] <- sig_comparisons
            ph_sig_num <- ph_sig_num + 1
            
          }
        }
        assign(paste(name,'_ph_list',sep=""),ph_list)
        
        perc_ph_sig <- ph_sig_num / sig_unit_num
        cat('perc ph pairwise sig:',perc_ph_sig*100,'\n')
        
        if(length(ph_list) > 0){
          comp_perc_list <- list()
          for(comp_name in temp[[1]]$comparisons){
            ct <- 0
            for(i in 1:ph_sig_num){
              if(comp_name %in% ph_list[[i]]){
                ct <- ct + 1
              }
              comp_perc_list[[comp_name]] <- ct / ph_sig_num
            }
            comp_perc_list[['total']] <- ph_sig_num
          }
          perc_list_windows[[window_name]] <- comp_perc_list
        }
      }
    }
    if(length(perc_list_windows) > 0){
      suppressWarnings(list_length <- do.call(rbind, lapply(perc_list_windows, length)))
      
      suppressWarnings(perc_list_windows_cbind <- as.data.frame(do.call(cbind,perc_list_windows)))
      assign(paste(region_name,'_',ph_name,'_cpl',sep=""),perc_list_windows_cbind)
      #cat(ph_name,'\n\n')
      write.xlsx(perc_list_windows_cbind,file=paste(region_name,'_ph_percs.xlsx',sep=""),sheetName=ph_name,append=T)
    }
  }
}


save.image(paste(nhp_id,"_summary.RData",sep=""))




