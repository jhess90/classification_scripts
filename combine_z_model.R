library(openxlsx)
library(ggplot2)
library(reshape2)
#source("~/documents/lab/workspace/Classification_scripts/multiplot.R")
#source("~/workspace/classification_scripts/multiplot.R")
library(zoo)
library(gplots)
library(RColorBrewer)
library(abind)
library(gridGraphics)
library(grid)
library(gridExtra)
library(R.matlab)
library(ggpmisc)

region_list <- c('M1','S1','PmD')
labs <- c("both pos","both neg","beta pos","alpha pos")

for (region_ind in 1:length(region_list)){
  cat('collating ',region_list[region_ind],'\n')
  
  ###############
  
  block_list <- Sys.glob(paste('model*',region_list[region_ind],'*dicts.mat',sep=""))
  
  bfr_cue_col_total_unit_num <- 0
  bfr_cue_col_num_slopes <- 0
  bfr_cue_col_num_sig_slopes <- 0
  bfr_cue_col_num_sig_alpha <- 0
  bfr_cue_col_num_sig_beta <- 0
  bfr_cue_col_num_sig_alpha_only <- 0
  bfr_cue_col_num_sig_beta_only <- 0
  bfr_cue_col_num_both_pos <- 0
  bfr_cue_col_num_both_neg <- 0
  bfr_cue_col_num_alpha_pos <- 0
  bfr_cue_col_num_beta_pos <- 0
  
  aft_cue_col_total_unit_num <- 0
  aft_cue_col_num_slopes <- 0
  aft_cue_col_num_sig_slopes <- 0
  aft_cue_col_num_sig_alpha <- 0
  aft_cue_col_num_sig_beta <- 0
  aft_cue_col_num_sig_alpha_only <- 0
  aft_cue_col_num_sig_beta_only <- 0
  aft_cue_col_num_both_pos <- 0
  aft_cue_col_num_both_neg <- 0
  aft_cue_col_num_alpha_pos <- 0
  aft_cue_col_num_beta_pos <- 0
  
  bfr_res_col_total_unit_num <- 0
  bfr_res_col_num_slopes <- 0
  bfr_res_col_num_sig_slopes <- 0
  bfr_res_col_num_sig_alpha <- 0
  bfr_res_col_num_sig_beta <- 0
  bfr_res_col_num_sig_alpha_only <- 0
  bfr_res_col_num_sig_beta_only <- 0
  bfr_res_col_num_both_pos <- 0
  bfr_res_col_num_both_neg <- 0
  bfr_res_col_num_alpha_pos <- 0
  bfr_res_col_num_beta_pos <- 0
  
  aft_res_col_total_unit_num <- 0
  aft_res_col_num_slopes <- 0
  aft_res_col_num_sig_slopes <- 0
  aft_res_col_num_sig_alpha <- 0
  aft_res_col_num_sig_beta <- 0
  aft_res_col_num_sig_alpha_only <- 0
  aft_res_col_num_sig_beta_only <- 0
  aft_res_col_num_both_pos <- 0
  aft_res_col_num_both_neg <- 0
  aft_res_col_num_alpha_pos <- 0
  aft_res_col_num_beta_pos <- 0
  
  for (block_ind in 1:length(block_list)){
  
    summary <- readMat(block_list[block_ind])
   
    total_unit_num <- summary$perc.summary[,,1]$bfr.cue.model[,,1]$total.unit.num
    num_slopes <- summary$perc.summary[,,1]$bfr.cue.model[,,1]$num.slopes
    num_sig_slopes <- summary$perc.summary[,,1]$bfr.cue.model[,,1]$num.sig.slopes
    perc_alpha_pos <- summary$perc.summary[,,1]$bfr.cue.model[,,1]$perc.alpha.pos
    perc_beta_pos <- summary$perc.summary[,,1]$bfr.cue.model[,,1]$perc.beta.pos
    perc_both_pos <- summary$perc.summary[,,1]$bfr.cue.model[,,1]$perc.both.pos
    perc_both_neg <- summary$perc.summary[,,1]$bfr.cue.model[,,1]$perc.both.neg
  
    num_sig_alpha <- summary$perc.summary[,,1]$bfr.cue.model[,,1]$num.sig.alpha
    num_sig_beta <- summary$perc.summary[,,1]$bfr.cue.model[,,1]$num.sig.beta
    num_sig_alpha_only <- summary$perc.summary[,,1]$bfr.cue.model[,,1]$num.sig.alpha.only
    num_sig_beta_only <- summary$perc.summary[,,1]$bfr.cue.model[,,1]$num.sig.beta.only
    num_both_pos <- perc_both_pos*num_slopes
    num_both_neg <- perc_both_neg*num_slopes
    num_alpha_pos <- perc_alpha_pos*num_slopes
    num_beta_pos <- perc_beta_pos*num_slopes
    
    bfr_cue_col_total_unit_num <- bfr_cue_col_total_unit_num + total_unit_num
    bfr_cue_col_num_slopes <- bfr_cue_col_num_slopes + num_slopes
    bfr_cue_col_num_sig_slopes <- bfr_cue_col_num_sig_slopes + num_sig_slopes
    bfr_cue_col_num_sig_alpha <- bfr_cue_col_num_sig_alpha + num_sig_alpha
    bfr_cue_col_num_sig_beta <- bfr_cue_col_num_sig_beta + num_sig_beta
    bfr_cue_col_num_sig_alpha_only <- bfr_cue_col_num_sig_alpha_only + num_sig_alpha_only
    bfr_cue_col_num_sig_beta_only <- bfr_cue_col_num_sig_beta_only + num_sig_beta_only
    
    bfr_cue_col_num_both_pos <- bfr_cue_col_num_both_pos + num_both_pos
    bfr_cue_col_num_both_neg <- bfr_cue_col_num_both_neg + num_both_neg
    bfr_cue_col_num_alpha_pos <- bfr_cue_col_num_alpha_pos + num_alpha_pos
    bfr_cue_col_num_beta_pos <- bfr_cue_col_num_beta_pos + num_beta_pos
    #
    total_unit_num <- summary$perc.summary[,,1]$aft.cue.model[,,1]$total.unit.num
    num_slopes <- summary$perc.summary[,,1]$aft.cue.model[,,1]$num.slopes
    num_sig_slopes <- summary$perc.summary[,,1]$aft.cue.model[,,1]$num.sig.slopes
    perc_alpha_pos <- summary$perc.summary[,,1]$aft.cue.model[,,1]$perc.alpha.pos
    perc_beta_pos <- summary$perc.summary[,,1]$aft.cue.model[,,1]$perc.beta.pos
    perc_both_pos <- summary$perc.summary[,,1]$aft.cue.model[,,1]$perc.both.pos
    perc_both_neg <- summary$perc.summary[,,1]$aft.cue.model[,,1]$perc.both.neg
    
    num_sig_alpha <- summary$perc.summary[,,1]$aft.cue.model[,,1]$num.sig.alpha
    num_sig_beta <- summary$perc.summary[,,1]$aft.cue.model[,,1]$num.sig.beta
    num_sig_alpha_only <- summary$perc.summary[,,1]$aft.cue.model[,,1]$num.sig.alpha.only
    num_sig_beta_only <- summary$perc.summary[,,1]$aft.cue.model[,,1]$num.sig.beta.only
    num_both_pos <- perc_both_pos*num_slopes
    num_both_neg <- perc_both_neg*num_slopes
    num_alpha_pos <- perc_alpha_pos*num_slopes
    num_beta_pos <- perc_beta_pos*num_slopes
    
    aft_cue_col_total_unit_num <- aft_cue_col_total_unit_num + total_unit_num
    aft_cue_col_num_slopes <- aft_cue_col_num_slopes + num_slopes
    aft_cue_col_num_sig_slopes <- aft_cue_col_num_sig_slopes + num_sig_slopes
    aft_cue_col_num_sig_alpha <- aft_cue_col_num_sig_alpha + num_sig_alpha
    aft_cue_col_num_sig_beta <- aft_cue_col_num_sig_beta + num_sig_beta
    aft_cue_col_num_sig_alpha_only <- aft_cue_col_num_sig_alpha_only + num_sig_alpha_only
    aft_cue_col_num_sig_beta_only <- aft_cue_col_num_sig_beta_only + num_sig_beta_only
    
    aft_cue_col_num_both_pos <- aft_cue_col_num_both_pos + num_both_pos
    aft_cue_col_num_both_neg <- aft_cue_col_num_both_neg + num_both_neg
    aft_cue_col_num_alpha_pos <- aft_cue_col_num_alpha_pos + num_alpha_pos
    aft_cue_col_num_beta_pos <- aft_cue_col_num_beta_pos + num_beta_pos
    #   
    total_unit_num <- summary$perc.summary[,,1]$bfr.result.model[,,1]$total.unit.num
    num_slopes <- summary$perc.summary[,,1]$bfr.result.model[,,1]$num.slopes
    num_sig_slopes <- summary$perc.summary[,,1]$bfr.result.model[,,1]$num.sig.slopes
    perc_alpha_pos <- summary$perc.summary[,,1]$bfr.result.model[,,1]$perc.alpha.pos
    perc_beta_pos <- summary$perc.summary[,,1]$bfr.result.model[,,1]$perc.beta.pos
    perc_both_pos <- summary$perc.summary[,,1]$bfr.result.model[,,1]$perc.both.pos
    perc_both_neg <- summary$perc.summary[,,1]$bfr.result.model[,,1]$perc.both.neg
    
    num_sig_alpha <- summary$perc.summary[,,1]$bfr.result.model[,,1]$num.sig.alpha
    num_sig_beta <- summary$perc.summary[,,1]$bfr.result.model[,,1]$num.sig.beta
    num_sig_alpha_only <- summary$perc.summary[,,1]$bfr.result.model[,,1]$num.sig.alpha.only
    num_sig_beta_only <- summary$perc.summary[,,1]$bfr.result.model[,,1]$num.sig.beta.only
    num_both_pos <- perc_both_pos*num_slopes
    num_both_neg <- perc_both_neg*num_slopes
    num_alpha_pos <- perc_alpha_pos*num_slopes
    num_beta_pos <- perc_beta_pos*num_slopes
    
    bfr_res_col_total_unit_num <- bfr_res_col_total_unit_num + total_unit_num
    bfr_res_col_num_slopes <- bfr_res_col_num_slopes + num_slopes
    bfr_res_col_num_sig_slopes <- bfr_res_col_num_sig_slopes + num_sig_slopes
    bfr_res_col_num_sig_alpha <- bfr_res_col_num_sig_alpha + num_sig_alpha
    bfr_res_col_num_sig_beta <- bfr_res_col_num_sig_beta + num_sig_beta
    bfr_res_col_num_sig_alpha_only <- bfr_res_col_num_sig_alpha_only + num_sig_alpha_only
    bfr_res_col_num_sig_beta_only <- bfr_res_col_num_sig_beta_only + num_sig_beta_only
    
    bfr_res_col_num_both_pos <- bfr_res_col_num_both_pos + num_both_pos
    bfr_res_col_num_both_neg <- bfr_res_col_num_both_neg + num_both_neg
    bfr_res_col_num_alpha_pos <- bfr_res_col_num_alpha_pos + num_alpha_pos
    bfr_res_col_num_beta_pos <- bfr_res_col_num_beta_pos + num_beta_pos
    #
    total_unit_num <- summary$perc.summary[,,1]$aft.result.model[,,1]$total.unit.num
    num_slopes <- summary$perc.summary[,,1]$aft.result.model[,,1]$num.slopes
    num_sig_slopes <- summary$perc.summary[,,1]$aft.result.model[,,1]$num.sig.slopes
    perc_alpha_pos <- summary$perc.summary[,,1]$aft.result.model[,,1]$perc.alpha.pos
    perc_beta_pos <- summary$perc.summary[,,1]$aft.result.model[,,1]$perc.beta.pos
    perc_both_pos <- summary$perc.summary[,,1]$aft.result.model[,,1]$perc.both.pos
    perc_both_neg <- summary$perc.summary[,,1]$aft.result.model[,,1]$perc.both.neg
    
    num_sig_alpha <- summary$perc.summary[,,1]$aft.result.model[,,1]$num.sig.alpha
    num_sig_beta <- summary$perc.summary[,,1]$aft.result.model[,,1]$num.sig.beta
    num_sig_alpha_only <- summary$perc.summary[,,1]$aft.result.model[,,1]$num.sig.alpha.only
    num_sig_beta_only <- summary$perc.summary[,,1]$aft.result.model[,,1]$num.sig.beta.only
    num_both_pos <- perc_both_pos*num_slopes
    num_both_neg <- perc_both_neg*num_slopes
    num_alpha_pos <- perc_alpha_pos*num_slopes
    num_beta_pos <- perc_beta_pos*num_slopes
    
    aft_res_col_total_unit_num <- aft_res_col_total_unit_num + total_unit_num
    aft_res_col_num_slopes <- aft_res_col_num_slopes + num_slopes
    aft_res_col_num_sig_slopes <- aft_res_col_num_sig_slopes + num_sig_slopes
    aft_res_col_num_sig_alpha <- aft_res_col_num_sig_alpha + num_sig_alpha
    aft_res_col_num_sig_beta <- aft_res_col_num_sig_beta + num_sig_beta
    aft_res_col_num_sig_alpha_only <- aft_res_col_num_sig_alpha_only + num_sig_alpha_only
    aft_res_col_num_sig_beta_only <- aft_res_col_num_sig_beta_only + num_sig_beta_only
    
    aft_res_col_num_both_pos <- aft_res_col_num_both_pos + num_both_pos
    aft_res_col_num_both_neg <- aft_res_col_num_both_neg + num_both_neg
    aft_res_col_num_alpha_pos <- aft_res_col_num_alpha_pos + num_alpha_pos
    aft_res_col_num_beta_pos <- aft_res_col_num_beta_pos + num_beta_pos
     
    
  }
  bfr_cue_col_perc_slopes <- bfr_cue_col_num_slopes/bfr_cue_col_total_unit_num
  bfr_cue_col_perc_sig_slopes <- bfr_cue_col_num_sig_slopes/bfr_cue_col_total_unit_num
  bfr_cue_col_perc_alpha_pos <- bfr_cue_col_num_alpha_pos/bfr_cue_col_total_unit_num
  bfr_cue_col_perc_beta_pos <- bfr_cue_col_num_beta_pos/bfr_cue_col_total_unit_num
  bfr_cue_col_perc_both_pos <- bfr_cue_col_num_both_pos/bfr_cue_col_total_unit_num
  bfr_cue_col_perc_both_neg <- bfr_cue_col_num_both_neg/bfr_cue_col_total_unit_num
  bfr_cue_col_perc_sig_alpha <- bfr_cue_col_num_sig_alpha/bfr_cue_col_total_unit_num
  bfr_cue_col_perc_sig_beta <- bfr_cue_col_num_sig_beta/bfr_cue_col_total_unit_num
  bfr_cue_col_perc_sig_alpha_only <- bfr_cue_col_num_sig_alpha_only/bfr_cue_col_total_unit_num
  bfr_cue_col_perc_sig_beta_only <- bfr_cue_col_num_sig_beta_only/bfr_cue_col_total_unit_num
  
  aft_cue_col_perc_slopes <- aft_cue_col_num_slopes/aft_cue_col_total_unit_num
  aft_cue_col_perc_sig_slopes <- aft_cue_col_num_sig_slopes/aft_cue_col_total_unit_num
  aft_cue_col_perc_alpha_pos <- aft_cue_col_num_alpha_pos/aft_cue_col_total_unit_num
  aft_cue_col_perc_beta_pos <- aft_cue_col_num_beta_pos/aft_cue_col_total_unit_num
  aft_cue_col_perc_both_pos <- aft_cue_col_num_both_pos/aft_cue_col_total_unit_num
  aft_cue_col_perc_both_neg <- aft_cue_col_num_both_neg/aft_cue_col_total_unit_num
  aft_cue_col_perc_sig_alpha <- aft_cue_col_num_sig_alpha/aft_cue_col_total_unit_num
  aft_cue_col_perc_sig_beta <- aft_cue_col_num_sig_beta/aft_cue_col_total_unit_num
  aft_cue_col_perc_sig_alpha_only <- aft_cue_col_num_sig_alpha_only/aft_cue_col_total_unit_num
  aft_cue_col_perc_sig_beta_only <- aft_cue_col_num_sig_beta_only/aft_cue_col_total_unit_num

  bfr_res_col_perc_slopes <- bfr_res_col_num_slopes/bfr_res_col_total_unit_num
  bfr_res_col_perc_sig_slopes <- bfr_res_col_num_sig_slopes/bfr_res_col_total_unit_num
  bfr_res_col_perc_alpha_pos <- bfr_res_col_num_alpha_pos/bfr_res_col_total_unit_num
  bfr_res_col_perc_beta_pos <- bfr_res_col_num_beta_pos/bfr_res_col_total_unit_num
  bfr_res_col_perc_both_pos <- bfr_res_col_num_both_pos/bfr_res_col_total_unit_num
  bfr_res_col_perc_both_neg <- bfr_res_col_num_both_neg/bfr_res_col_total_unit_num
  bfr_res_col_perc_sig_alpha <- bfr_res_col_num_sig_alpha/bfr_res_col_total_unit_num
  bfr_res_col_perc_sig_beta <- bfr_res_col_num_sig_beta/bfr_res_col_total_unit_num
  bfr_res_col_perc_sig_alpha_only <- bfr_res_col_num_sig_alpha_only/bfr_res_col_total_unit_num
  bfr_res_col_perc_sig_beta_only <- bfr_res_col_num_sig_beta_only/bfr_res_col_total_unit_num
  
  aft_res_col_perc_slopes <- aft_res_col_num_slopes/aft_res_col_total_unit_num
  aft_res_col_perc_sig_slopes <- aft_res_col_num_sig_slopes/aft_res_col_total_unit_num
  aft_res_col_perc_alpha_pos <- aft_res_col_num_alpha_pos/aft_res_col_total_unit_num
  aft_res_col_perc_beta_pos <- aft_res_col_num_beta_pos/aft_res_col_total_unit_num
  aft_res_col_perc_both_pos <- aft_res_col_num_both_pos/aft_res_col_total_unit_num
  aft_res_col_perc_both_neg <- aft_res_col_num_both_neg/aft_res_col_total_unit_num
  aft_res_col_perc_sig_alpha <- aft_res_col_num_sig_alpha/aft_res_col_total_unit_num
  aft_res_col_perc_sig_beta <- aft_res_col_num_sig_beta/aft_res_col_total_unit_num
  aft_res_col_perc_sig_alpha_only <- aft_res_col_num_sig_alpha_only/aft_res_col_total_unit_num
  aft_res_col_perc_sig_beta_only <- aft_res_col_num_sig_beta_only/aft_res_col_total_unit_num

  

  # if (region_list[region_index] == 'M1'){
  #   bfr_cue_nums_sum <- M1_bfr_cue_nums_sum
  #   aft_cue_nums_sum <- M1_aft_cue_nums_sum
  #   bfr_result_nums_sum <- M1_bfr_result_nums_sum
  #   aft_result_nums_sum <- M1_aft_result_nums_sum
  #   bfr_cue_all_slopes <- M1_bfr_cue_all_slopes
  #   bfr_cue_sig_slopes <- M1_bfr_cue_sig_slopes
  #   aft_cue_all_slopes <- M1_aft_cue_all_slopes
  #   aft_cue_sig_slopes <- M1_aft_cue_sig_slopes
  #   bfr_result_all_slopes <- M1_bfr_result_all_slopes
  #   bfr_result_sig_slopes <- M1_bfr_result_sig_slopes
  #   aft_result_all_slopes <- M1_aft_result_all_slopes
  #   aft_result_sig_slopes <- M1_aft_result_sig_slopes
  # }else if(region_list[region_index] == 'S1'){
  #   bfr_cue_nums_sum <- S1_bfr_cue_nums_sum
  #   aft_cue_nums_sum <- S1_aft_cue_nums_sum
  #   bfr_result_nums_sum <- S1_bfr_result_nums_sum
  #   aft_result_nums_sum <- S1_aft_result_nums_sum     
  #   bfr_cue_all_slopes <- S1_bfr_cue_all_slopes
  #   bfr_cue_sig_slopes <- S1_bfr_cue_sig_slopes
  #   aft_cue_all_slopes <- S1_aft_cue_all_slopes
  #   aft_cue_sig_slopes <- S1_aft_cue_sig_slopes
  #   bfr_result_all_slopes <- S1_bfr_result_all_slopes
  #   bfr_result_sig_slopes <- S1_bfr_result_sig_slopes
  #   aft_result_all_slopes <- S1_aft_result_all_slopes
  #   aft_result_sig_slopes <- S1_aft_result_sig_slopes
  # }else if(region_list[region_index] == 'PmD'){
  #   bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum
  #   aft_cue_nums_sum <- PmD_aft_cue_nums_sum
  #   bfr_result_nums_sum <- PmD_bfr_result_nums_sum
  #   aft_result_nums_sum <- PmD_aft_result_nums_sum
  #   bfr_cue_all_slopes <- PmD_bfr_cue_all_slopes
  #   bfr_cue_sig_slopes <- PmD_bfr_cue_sig_slopes
  #   aft_cue_all_slopes <- PmD_aft_cue_all_slopes
  #   aft_cue_sig_slopes <- PmD_aft_cue_sig_slopes
  #   bfr_result_all_slopes <- PmD_bfr_result_all_slopes
  #   bfr_result_sig_slopes <- PmD_bfr_result_sig_slopes
  #   aft_result_all_slopes <- PmD_aft_result_all_slopes
  #   aft_result_sig_slopes <- PmD_aft_result_sig_slopes
  # }
  
  
  
  #bfr_cue_df <-data.frame(perc=bfr_cue_nums_sum/sum(bfr_cue_nums_sum),labs,type='bfr_cue')
  #aft_cue_df <-data.frame(perc=aft_cue_nums_sum/sum(aft_cue_nums_sum),labs,type='aft_cue')
  #bfr_result_df <-data.frame(perc=bfr_result_nums_sum/sum(bfr_result_nums_sum),labs,type='bfr_result')
  #aft_result_df <-data.frame(perc=aft_result_nums_sum/sum(aft_result_nums_sum),labs,type='aft_result')
  
  bfr_cue_df <- data.frame()
  
  
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

  
