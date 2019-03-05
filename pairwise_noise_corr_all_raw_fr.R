library(openxlsx)
library(ggplot2)
library(reshape2)
library(RColorBrewer)
library(abind)
library(gridGraphics)
library(grid)
library(gridExtra)
library(R.matlab)
library(corrplot)
library(ggcorrplot)

source("~/workspace/classification_scripts/multiplot.R")

saveAsPng <- T
region_list <- c('M1','S1','PMd')

##############################

compute_plot_corr <- function(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,type_name,trial_inds){
  
  if (length(trial_inds) == 0){return()}
  
  #look at only trials for this particular index  
  cue_fr_M1 <- all_cue_fr_M1[,trial_inds,]
  cue_fr_S1 <- all_cue_fr_S1[,trial_inds,]
  cue_fr_PmD <- all_cue_fr_PmD[,trial_inds,]

  res_fr_M1 <- all_res_fr_M1[,trial_inds,]
  res_fr_S1 <- all_res_fr_S1[,trial_inds,]
  res_fr_PmD <- all_res_fr_PmD[,trial_inds,]
  
  #average over all trials of this particular trial index for each unit
  
  avg_psth_cue_M1 <- apply(cue_fr_M1,c(1,3),mean)
  avg_psth_res_M1 <- apply(res_fr_M1,c(1,3),mean)
  
  demeaned_cue_fr_M1 <- cue_fr_M1
  demeaned_res_fr_M1 <- res_fr_M1
  for(unit_num in 1:total_unit_num_M1){
    if (length(trial_inds) > 1){demeaned_cue_fr_M1[unit_num,,] <- sweep(cue_fr_M1[unit_num,,],2,avg_psth_cue_M1[unit_num,],'-')}else{demeaned_cue_fr_M1[unit_num,] < - cue_fr_M1[unit_num,] - avg_psth_cue_M1[unit_num,]}
    if (length(trial_inds) > 1){demeaned_res_fr_M1[unit_num,,] <- sweep(res_fr_M1[unit_num,,],2,avg_psth_res_M1[unit_num,],'-')}else{demeaned_res_fr_M1[unit_num,] < - res_fr_M1[unit_num,] - avg_psth_res_M1[unit_num,]}
  }
  demeaned_cue_avgs_M1 <- c()
  demeaned_res_avgs_M1 <- c()
  for(unit_num in 1:total_unit_num_M1){
    if (length(trial_inds) > 1){demeaned_cue_avgs_M1 <- rbind(demeaned_cue_avgs_M1,colMeans(demeaned_cue_fr_M1[unit_num,,]))}else  if (length(trial_inds) == 1){demeaned_cue_avgs_M1 <- rbind(demeaned_cue_avgs_M1,demeaned_cue_fr_M1[unit_num,])}
    if (length(trial_inds) > 1){demeaned_res_avgs_M1 <- rbind(demeaned_res_avgs_M1,colMeans(demeaned_res_fr_M1[unit_num,,]))}else if (length(trial_inds) == 1){demeaned_res_avgs_M1 <- rbind(demeaned_res_avgs_M1,demeaned_res_fr_M1[unit_num,])}
  }
  
  # #
  # cue_avgs_S1 <- c()
  # res_avgs_S1 <- c()
  # for(unit_num in 1:total_unit_num_S1){
  #   if (length(trial_inds) > 1){cue_avgs_S1 <- rbind(cue_avgs_S1,colMeans(cue_fr_S1[unit_num,,]))}else if (length(trial_inds) == 1){cue_avgs_S1 <- rbind(cue_avgs_S1,cue_fr_S1[unit_num,])}
  #   if (length(trial_inds) > 1){res_avgs_S1 <- rbind(res_avgs_S1,colMeans(res_fr_S1[unit_num,,]))}else if (length(trial_inds) == 1){res_avgs_S1 <- rbind(res_avgs_S1,res_fr_S1[unit_num,])}
  # }
  # demeaned_cue_fr_S1 <- cue_fr_S1
  # demeaned_res_fr_S1 <- res_fr_S1
  # for(unit_num in 1:total_unit_num_S1){
  #   if (length(trial_inds) > 1){demeaned_cue_fr_S1[unit_num,,] <- sweep(cue_fr_S1[unit_num,,],2,cue_avgs_S1[unit_num,],'-')}else{demeaned_cue_fr_S1[unit_num,] < - cue_fr_S1[unit_num,] - cue_fr_S1[unit_num,]}
  #   if (length(trial_inds) > 1){demeaned_res_fr_S1[unit_num,,] <- sweep(res_fr_S1[unit_num,,],2,res_avgs_S1[unit_num,],'-')}else{demeaned_res_fr_S1[unit_num,] < - res_fr_S1[unit_num,] - res_fr_S1[unit_num,]}
  # }
  # demeaned_cue_avgs_S1 <- c()
  # demeaned_res_avgs_S1 <- c()
  # for(unit_num in 1:total_unit_num_S1){
  #   if (length(trial_inds) > 1){demeaned_cue_avgs_S1 <- rbind(demeaned_cue_avgs_S1,colMeans(demeaned_cue_fr_S1[unit_num,,]))}else if (length(trial_inds) == 1){demeaned_cue_avgs_S1 <- rbind(demeaned_cue_avgs_S1,demeaned_cue_fr_S1[unit_num,])}
  #   if (length(trial_inds) > 1){demeaned_res_avgs_S1 <- rbind(demeaned_res_avgs_S1,colMeans(demeaned_res_fr_S1[unit_num,,]))}else if (length(trial_inds) == 1){demeaned_res_avgs_S1 <- rbind(demeaned_res_avgs_S1,demeaned_res_fr_S1[unit_num,])}
  # }
  # 
  # #
  # cue_avgs_PmD <- c()
  # res_avgs_PmD <- c()
  # for(unit_num in 1:total_unit_num_PmD){
  #   if (length(trial_inds) > 1){cue_avgs_PmD <- rbind(cue_avgs_PmD,colMeans(cue_fr_PmD[unit_num,,]))}else if (length(trial_inds) == 1){cue_avgs_PmD <- rbind(cue_avgs_PmD,cue_fr_PmD[unit_num,])}
  #   if (length(trial_inds) > 1){res_avgs_PmD <- rbind(res_avgs_PmD,colMeans(res_fr_PmD[unit_num,,]))}else if (length(trial_inds) == 1){res_avgs_PmD <- rbind(res_avgs_PmD,res_fr_PmD[unit_num,])}
  # }
  # demeaned_cue_fr_PmD <- cue_fr_PmD
  # demeaned_res_fr_PmD <- res_fr_PmD
  # for(unit_num in 1:total_unit_num_PmD){
  #   if (length(trial_inds) > 1){demeaned_cue_fr_PmD[unit_num,,] <- sweep(cue_fr_PmD[unit_num,,],2,cue_avgs_PmD[unit_num,],'-')}else{demeaned_cue_fr_PmD[unit_num,] < - cue_fr_PmD[unit_num,] - cue_fr_PmD[unit_num,]}
  #   if (length(trial_inds) > 1){demeaned_res_fr_PmD[unit_num,,] <- sweep(res_fr_PmD[unit_num,,],2,res_avgs_PmD[unit_num,],'-')}else{demeaned_res_fr_PmD[unit_num,] < - res_fr_PmD[unit_num,] - res_fr_PmD[unit_num,]}
  # }
  # demeaned_cue_avgs_PmD <- c()
  # demeaned_res_avgs_PmD <- c()
  # for(unit_num in 1:total_unit_num_PmD){
  #   if (length(trial_inds) > 1){demeaned_cue_avgs_PmD <- rbind(demeaned_cue_avgs_PmD,colMeans(demeaned_cue_fr_PmD[unit_num,,]))}else if (length(trial_inds) == 1){demeaned_cue_avgs_PmD <- rbind(demeaned_cue_avgs_PmD,demeaned_cue_fr_PmD[unit_num,])}
  #   if (length(trial_inds) > 1){demeaned_res_avgs_PmD <- rbind(demeaned_res_avgs_PmD,colMeans(demeaned_res_fr_PmD[unit_num,,]))}else if (length(trial_inds) == 1){demeaned_res_avgs_PmD <- rbind(demeaned_res_avgs_PmD,demeaned_res_fr_PmD[unit_num,])}
  # }
  
  #
  avg_psth_cue_S1 <- apply(cue_fr_S1,c(1,3),mean)
  avg_psth_res_S1 <- apply(res_fr_S1,c(1,3),mean)
  
  demeaned_cue_fr_S1 <- cue_fr_S1
  demeaned_res_fr_S1 <- res_fr_S1
  for(unit_num in 1:total_unit_num_S1){
    if (length(trial_inds) > 1){demeaned_cue_fr_S1[unit_num,,] <- sweep(cue_fr_S1[unit_num,,],2,avg_psth_cue_S1[unit_num,],'-')}else{demeaned_cue_fr_S1[unit_num,] < - cue_fr_S1[unit_num,] - avg_psth_cue_S1[unit_num,]}
    if (length(trial_inds) > 1){demeaned_res_fr_S1[unit_num,,] <- sweep(res_fr_S1[unit_num,,],2,avg_psth_res_S1[unit_num,],'-')}else{demeaned_res_fr_S1[unit_num,] < - res_fr_S1[unit_num,] - avg_psth_res_S1[unit_num,]}
  }
  demeaned_cue_avgs_S1 <- c()
  demeaned_res_avgs_S1 <- c()
  for(unit_num in 1:total_unit_num_S1){
    if (length(trial_inds) > 1){demeaned_cue_avgs_S1 <- rbind(demeaned_cue_avgs_S1,colMeans(demeaned_cue_fr_S1[unit_num,,]))}else  if (length(trial_inds) == 1){demeaned_cue_avgs_S1 <- rbind(demeaned_cue_avgs_S1,demeaned_cue_fr_S1[unit_num,])}
    if (length(trial_inds) > 1){demeaned_res_avgs_S1 <- rbind(demeaned_res_avgs_S1,colMeans(demeaned_res_fr_S1[unit_num,,]))}else if (length(trial_inds) == 1){demeaned_res_avgs_S1 <- rbind(demeaned_res_avgs_S1,demeaned_res_fr_S1[unit_num,])}
  }
  
  #
  avg_psth_cue_PmD <- apply(cue_fr_PmD,c(1,3),mean)
  avg_psth_res_PmD <- apply(res_fr_PmD,c(1,3),mean)
  
  demeaned_cue_fr_PmD <- cue_fr_PmD
  demeaned_res_fr_PmD <- res_fr_PmD
  for(unit_num in 1:total_unit_num_PmD){
    if (length(trial_inds) > 1){demeaned_cue_fr_PmD[unit_num,,] <- sweep(cue_fr_PmD[unit_num,,],2,avg_psth_cue_PmD[unit_num,],'-')}else{demeaned_cue_fr_PmD[unit_num,] < - cue_fr_PmD[unit_num,] - avg_psth_cue_PmD[unit_num,]}
    if (length(trial_inds) > 1){demeaned_res_fr_PmD[unit_num,,] <- sweep(res_fr_PmD[unit_num,,],2,avg_psth_res_PmD[unit_num,],'-')}else{demeaned_res_fr_PmD[unit_num,] < - res_fr_PmD[unit_num,] - avg_psth_res_PmD[unit_num,]}
  }
  demeaned_cue_avgs_PmD <- c()
  demeaned_res_avgs_PmD <- c()
  for(unit_num in 1:total_unit_num_PmD){
    if (length(trial_inds) > 1){demeaned_cue_avgs_PmD <- rbind(demeaned_cue_avgs_PmD,colMeans(demeaned_cue_fr_PmD[unit_num,,]))}else  if (length(trial_inds) == 1){demeaned_cue_avgs_PmD <- rbind(demeaned_cue_avgs_PmD,demeaned_cue_fr_PmD[unit_num,])}
    if (length(trial_inds) > 1){demeaned_res_avgs_PmD <- rbind(demeaned_res_avgs_PmD,colMeans(demeaned_res_fr_PmD[unit_num,,]))}else if (length(trial_inds) == 1){demeaned_res_avgs_PmD <- rbind(demeaned_res_avgs_PmD,demeaned_res_fr_PmD[unit_num,])}
  }
  
  
  #combine into larger matrix with all regions: M1 units, then S1 units, then PMd units
  combined_cue_avgs <- rbind(demeaned_cue_avgs_M1,demeaned_cue_avgs_S1,demeaned_cue_avgs_PmD)
  combined_res_avgs <- rbind(demeaned_res_avgs_M1,demeaned_res_avgs_S1,demeaned_res_avgs_PmD)
  
  #compute rate correlation coeffs and p values. Spearman non-parametric
  cue_corr <- cor(t(combined_cue_avgs),method=c("spearman"))
  res_corr <- cor(t(combined_res_avgs),method=c("spearman"))
  
  cue_pmat <- cor_pmat(t(combined_cue_avgs),method=c("spearman"))
  res_pmat <- cor_pmat(t(combined_res_avgs),method=c("spearman"))
  
  #region label data frame
  label_df <- data.frame(
    x = c(total_unit_num_M1/2,total_unit_num_M1+total_unit_num_S1/2,total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD/2,5,5,5),
    y = c(-8,-8,-8,total_unit_num_M1/2,total_unit_num_M1+total_unit_num_S1/2,total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD/2),
    text = c('M1','S1','PMd','M1','S1','PMd')
  )
  
  #calculate sum of correlations, put in data frame for plotting
  cue_corr_sum <- cue_corr
  res_corr_sum <- res_corr
  
  #cue_corr_sum[cue_pmat >= 0.05] <- 0
  #res_corr_sum[res_pmat >= 0.05] <- 0
  
  cue_corr_sum[lower.tri(cue_corr_sum,diag=T)] <- 0 #removing half of matrix, including the diagonal, so can sum
  res_corr_sum[lower.tri(res_corr_sum,diag=T)] <- 0
  
  abs_cue_corr_sum <- abs(cue_corr_sum)
  abs_res_corr_sum <- abs(res_corr_sum)
  
  m1_m1_cue <- round(sum(cue_corr_sum[1:total_unit_num_M1,1:total_unit_num_M1],na.rm=T),2)
  m1_s1_cue <- round(sum(cue_corr_sum[1:total_unit_num_M1,(1+total_unit_num_M1):(total_unit_num_M1 + total_unit_num_S1)],na.rm=T),2)
  m1_pmd_cue <- round(sum(cue_corr_sum[1:total_unit_num_M1,(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD)],na.rm=T),2)
  s1_s1_cue <- round(sum(cue_corr_sum[(1+total_unit_num_M1):(total_unit_num_M1+total_unit_num_S1),(1+total_unit_num_M1):(total_unit_num_M1+total_unit_num_S1)],na.rm=T),2)
  s1_pmd_cue <- round(sum(cue_corr_sum[(1+total_unit_num_M1):(total_unit_num_M1+total_unit_num_S1),(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD)],na.rm=T),2)
  pmd_pmd_cue <- round(sum(cue_corr_sum[(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD),(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD)],na.rm=T),2)
  
  abs_m1_m1_cue <- round(sum(abs_cue_corr_sum[1:total_unit_num_M1,1:total_unit_num_M1],na.rm=T),2)
  abs_m1_s1_cue <- round(sum(abs_cue_corr_sum[1:total_unit_num_M1,(1+total_unit_num_M1):(total_unit_num_M1 + total_unit_num_S1)],na.rm=T),2)
  abs_m1_pmd_cue <- round(sum(abs_cue_corr_sum[1:total_unit_num_M1,(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD)],na.rm=T),2)
  abs_s1_s1_cue <- round(sum(abs_cue_corr_sum[(1+total_unit_num_M1):(total_unit_num_M1+total_unit_num_S1),(1+total_unit_num_M1):(total_unit_num_M1+total_unit_num_S1)],na.rm=T),2)
  abs_s1_pmd_cue <- round(sum(abs_cue_corr_sum[(1+total_unit_num_M1):(total_unit_num_M1+total_unit_num_S1),(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD)],na.rm=T),2)
  abs_pmd_pmd_cue <- round(sum(abs_cue_corr_sum[(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD),(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD)],na.rm=T),2)
  
  m1_m1_res <- round(sum(res_corr_sum[1:total_unit_num_M1,1:total_unit_num_M1],na.rm=T),2)
  m1_s1_res <- round(sum(res_corr_sum[1:total_unit_num_M1,(1+total_unit_num_M1):(total_unit_num_M1 + total_unit_num_S1)],na.rm=T),2)
  m1_pmd_res <- round(sum(res_corr_sum[1:total_unit_num_M1,(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD)],na.rm=T),2)
  s1_s1_res <- round(sum(res_corr_sum[(1+total_unit_num_M1):(total_unit_num_M1+total_unit_num_S1),(1+total_unit_num_M1):(total_unit_num_M1+total_unit_num_S1)],na.rm=T),2)
  s1_pmd_res <- round(sum(res_corr_sum[(1+total_unit_num_M1):(total_unit_num_M1+total_unit_num_S1),(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD)],na.rm=T),2)
  pmd_pmd_res <- round(sum(res_corr_sum[(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD),(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD)],na.rm=T),2)
  
  abs_m1_m1_res <- round(sum(abs_res_corr_sum[1:total_unit_num_M1,1:total_unit_num_M1],na.rm=T),2)
  abs_m1_s1_res <- round(sum(abs_res_corr_sum[1:total_unit_num_M1,(1+total_unit_num_M1):(total_unit_num_M1 + total_unit_num_S1)],na.rm=T),2)
  abs_m1_pmd_res <- round(sum(abs_res_corr_sum[1:total_unit_num_M1,(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD)],na.rm=T),2)
  abs_s1_s1_res <- round(sum(abs_res_corr_sum[(1+total_unit_num_M1):(total_unit_num_M1+total_unit_num_S1),(1+total_unit_num_M1):(total_unit_num_M1+total_unit_num_S1)],na.rm=T),2)
  abs_s1_pmd_res <- round(sum(abs_res_corr_sum[(1+total_unit_num_M1):(total_unit_num_M1+total_unit_num_S1),(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD)],na.rm=T),2)
  abs_pmd_pmd_res <- round(sum(abs_res_corr_sum[(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD),(1+total_unit_num_M1+total_unit_num_S1):(total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD)],na.rm=T),2)
  
  cue_sum_label_df <- data.frame(
    x = c(total_unit_num_M1/2,total_unit_num_M1/2,total_unit_num_M1/2,total_unit_num_M1+total_unit_num_S1/2,total_unit_num_M1+total_unit_num_S1/2,total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD/2),
    y = c(total_unit_num_M1-12,total_unit_num_M1+total_unit_num_S1-12,total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD-12,total_unit_num_M1+total_unit_num_S1-12,total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD-12,total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD-12),
    text = c(paste(m1_m1_cue,abs_m1_m1_cue,sep="\n"),paste(m1_s1_cue,abs_m1_s1_cue,sep="\n"),paste(m1_pmd_cue,abs_m1_pmd_cue,sep="\n"),paste(s1_s1_cue,abs_s1_s1_cue,sep="\n"),paste(s1_pmd_cue,abs_s1_pmd_cue,sep="\n"),paste(pmd_pmd_cue,abs_pmd_pmd_cue,sep="\n"))
  )
  
  res_sum_label_df <- data.frame(
    x = c(total_unit_num_M1/2,total_unit_num_M1/2,total_unit_num_M1/2,total_unit_num_M1+total_unit_num_S1/2,total_unit_num_M1+total_unit_num_S1/2,total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD/2),
    y = c(total_unit_num_M1-12,total_unit_num_M1+total_unit_num_S1-12,total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD-12,total_unit_num_M1+total_unit_num_S1-12,total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD-12,total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD-12),
    text = c(paste(m1_m1_res,abs_m1_m1_res,sep="\n"),paste(m1_s1_res,abs_m1_s1_res,sep="\n"),paste(m1_pmd_res,abs_m1_pmd_res,sep="\n"),paste(s1_s1_res,abs_s1_s1_res,sep="\n"),paste(s1_pmd_res,abs_s1_pmd_res,sep="\n"),paste(pmd_pmd_res,abs_pmd_pmd_res,sep="\n"))
  )
  
  #write.table(cue_sum_label_df,paste('raw_noise_',type_name,'_cue.txt',sep=""),sep="\t",row.names=FALSE)
  #write.table(res_sum_label_df,paste('raw_noise_',type_name,'_res.txt',sep=""),sep="\t",row.names=FALSE)
  
  #num same-region pairs = x(x-1)/2
  m1_m1_num <- total_unit_num_M1*(total_unit_num_M1 - 1)/2
  s1_s1_num <- total_unit_num_S1*(total_unit_num_S1 - 1)/2
  pmd_pmd_num <- total_unit_num_PmD*(total_unit_num_PmD - 1)/2
  
  #cue, abs cue, res, abs res
  m1_m1_avgs <- c(m1_m1_cue,abs_m1_m1_cue,m1_m1_res,abs_m1_m1_res)/m1_m1_num
  s1_s1_avgs <- c(s1_s1_cue,abs_s1_s1_cue,s1_s1_res,abs_s1_s1_res)/s1_s1_num
  pmd_pmd_avgs <- c(pmd_pmd_cue,abs_pmd_pmd_cue,pmd_pmd_res,abs_pmd_pmd_res)/pmd_pmd_num
  m1_s1_avgs <- c(m1_s1_cue,abs_m1_s1_cue,m1_s1_res,abs_m1_s1_res)/(total_unit_num_M1*total_unit_num_S1)
  m1_pmd_avgs <- c(m1_pmd_cue,abs_m1_pmd_cue,m1_pmd_res,abs_m1_pmd_res)/(total_unit_num_M1*total_unit_num_PmD)
  s1_pmd_avgs <- c(s1_pmd_cue,abs_s1_pmd_cue,s1_pmd_res,abs_s1_pmd_res)/(total_unit_num_S1*total_unit_num_PmD)
  
  m1_m1_sums <- c(m1_m1_cue,abs_m1_m1_cue,m1_m1_res,abs_m1_m1_res)
  s1_s1_sums <- c(s1_s1_cue,abs_s1_s1_cue,s1_s1_res,abs_s1_s1_res)
  pmd_pmd_sums <- c(pmd_pmd_cue,abs_pmd_pmd_cue,pmd_pmd_res,abs_pmd_pmd_res)
  m1_s1_sums <- c(m1_s1_cue,abs_m1_s1_cue,m1_s1_res,abs_m1_s1_res)
  m1_pmd_sums <- c(m1_pmd_cue,abs_m1_pmd_cue,m1_pmd_res,abs_m1_pmd_res)
  s1_pmd_sums <- c(s1_pmd_cue,abs_s1_pmd_cue,s1_pmd_res,abs_s1_pmd_res)
  
  output <- data.frame(m1_m1_avgs,m1_s1_avgs,m1_pmd_avgs,s1_s1_avgs,s1_pmd_avgs,pmd_pmd_avgs,m1_m1_sums,m1_s1_sums,m1_pmd_sums,s1_s1_sums,s1_pmd_sums,pmd_pmd_sums)
  rownames(output) <- c('cue','abs cue','res','abs res')
  
  write.table(output,paste('avg_sum_noise_',type_name,'.txt',sep=""),sep=',')
  
  #plot correlation matrix
  png(paste("raw_noise_corr_spear_all_",type_name,".png",sep=""),width=8,height=6,units="in",res=500)
  
  cue_plt <- ggcorrplot(cue_corr,type="lower",outline.col='white')
  cue_plt <- cue_plt + geom_vline(xintercept=total_unit_num_M1,color="grey") + geom_vline(xintercept=total_unit_num_M1+total_unit_num_S1,color="grey")
  cue_plt <- cue_plt + geom_hline(yintercept=total_unit_num_M1,color="grey") + geom_hline(yintercept=total_unit_num_M1+total_unit_num_S1,color="grey")
  cue_plt <- cue_plt + labs(title=paste(type_name,":\nCue",sep="")) + theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank(),legend.position="none",axis.text.y=element_blank(),plot.title=element_text(size=16))
  cue_plt <- cue_plt + geom_text(data=label_df,aes(x=x,y=y,label=text),inherit.aes=FALSE,size=5)
  cue_plt <- cue_plt + geom_text(data=cue_sum_label_df,aes(x=x,y=y,label=text),inherit.aes=FALSE,size=2)

  res_plt <- ggcorrplot(res_corr,type="lower",outline.col='white')
  res_plt <- res_plt + geom_vline(xintercept=total_unit_num_M1,color="grey") + geom_vline(xintercept=total_unit_num_M1+total_unit_num_S1,color="grey")
  res_plt <- res_plt + geom_hline(yintercept=total_unit_num_M1,color="grey") + geom_hline(yintercept=total_unit_num_M1+total_unit_num_S1,color="grey")
  res_plt <- res_plt + labs(title="\nResult") + theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank(),legend.position="none",axis.text.y=element_blank(),plot.title=element_text(size=16))
  res_plt <- res_plt + geom_text(data=label_df,aes(x=x,y=y,label=text),inherit.aes=FALSE,size=5)
  res_plt <- res_plt + geom_text(data=res_sum_label_df,aes(x=x,y=y,label=text),inherit.aes=FALSE,size=2)
  
  multiplot(cue_plt,res_plt,cols=2)
  graphics.off()
  
  #plot correlation matrix
  png(paste("raw_noise_corr_spear_all_no_txt_",type_name,".png",sep=""),width=8,height=6,units="in",res=500)
  
  cue_plt <- ggcorrplot(cue_corr,type="lower",outline.col='white')
  cue_plt <- cue_plt + geom_vline(xintercept=total_unit_num_M1,color="grey") + geom_vline(xintercept=total_unit_num_M1+total_unit_num_S1,color="grey")
  cue_plt <- cue_plt + geom_hline(yintercept=total_unit_num_M1,color="grey") + geom_hline(yintercept=total_unit_num_M1+total_unit_num_S1,color="grey")
  cue_plt <- cue_plt + labs(title=paste(type_name,":\nCue",sep="")) + theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank(),legend.position="none",axis.text.y=element_blank(),plot.title=element_text(size=16))
  cue_plt <- cue_plt + geom_text(data=label_df,aes(x=x,y=y,label=text),inherit.aes=FALSE,size=5)
  #cue_plt <- cue_plt + geom_text(data=cue_sum_label_df,aes(x=x,y=y,label=text),inherit.aes=FALSE,size=2)
  
  res_plt <- ggcorrplot(res_corr,type="lower",outline.col='white')
  res_plt <- res_plt + geom_vline(xintercept=total_unit_num_M1,color="grey") + geom_vline(xintercept=total_unit_num_M1+total_unit_num_S1,color="grey")
  res_plt <- res_plt + geom_hline(yintercept=total_unit_num_M1,color="grey") + geom_hline(yintercept=total_unit_num_M1+total_unit_num_S1,color="grey")
  res_plt <- res_plt + labs(title="\nResult") + theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank(),legend.position="none",axis.text.y=element_blank(),plot.title=element_text(size=16))
  res_plt <- res_plt + geom_text(data=label_df,aes(x=x,y=y,label=text),inherit.aes=FALSE,size=5)
  #res_plt <- res_plt + geom_text(data=res_sum_label_df,aes(x=x,y=y,label=text),inherit.aes=FALSE,size=2)
  
  multiplot(cue_plt,res_plt,cols=2)
  graphics.off()
  
  return()
  
}


##############################

#uses simple_output_[region].mat, one from each region
readin <- readMat('non_z_simple_output_M1.mat')

all_cue_fr_M1 <- readin$return.dict[,,1]$all.cue.fr
all_res_fr_M1 <- readin$return.dict[,,1]$all.res.fr

condensed <- readin$return.dict[,,1]$condensed
bin_size <- readin$return.dict[,,1]$params[,,1]$bin.size[,]
total_unit_num_M1 <- dim(all_cue_fr_M1)[1]

readin <- readMat('non_z_simple_output_S1.mat')
all_cue_fr_S1 <- readin$return.dict[,,1]$all.cue.fr
all_res_fr_S1 <- readin$return.dict[,,1]$all.res.fr
total_unit_num_S1 <- dim(all_cue_fr_S1)[1]

readin <- readMat('non_z_simple_output_PmD.mat')
all_cue_fr_PmD <- readin$return.dict[,,1]$all.cue.fr
all_res_fr_PmD <- readin$return.dict[,,1]$all.res.fr
total_unit_num_PmD <- dim(all_cue_fr_PmD)[1]


#TODO unhardcode. Right now just looking at after-cue and after-result windows (AC and AR), instead of BC + AC for cue, and BR + AR for result
all_cue_fr_M1 <- all_cue_fr_M1[,,51:150]
all_res_fr_M1 <- all_res_fr_M1[,,51:150]

all_cue_fr_S1 <- all_cue_fr_S1[,,51:150]
all_res_fr_S1 <- all_res_fr_S1[,,51:150]

all_cue_fr_PmD <- all_cue_fr_PmD[,,51:150]
all_res_fr_PmD <- all_res_fr_PmD[,,51:150]

#get indices for different trial types
r0 <- which(condensed[,4] == 0)
rx <- which(condensed[,4] >= 1)

p0 <- which(condensed[,5] == 0)
px <- which(condensed[,5] >= 1)

res0 <- which(condensed[,6] == 0)
res1 <- which(condensed[,6] == 1)

r0_f <- res0[which(res0 %in% r0)]
rx_f <- res0[which(res0 %in% rx)]
r0_s <- res1[which(res1 %in% r0)]
rx_s <- res1[which(res1 %in% rx)]

p0_f <- res0[which(res0 %in% p0)]
px_f <- res0[which(res0 %in% px)]
p0_s <- res1[which(res1 %in% p0)]
px_s <- res1[which(res1 %in% px)]

r0_p0 <- r0[which(r0 %in% p0)]
rx_p0 <- rx[which(rx %in% p0)]
r0_px <- r0[which(r0 %in% px)]
rx_px <- rx[which(rx %in% px)]

r0_p0_f <- res0[which(res0 %in% r0_p0)]
rx_p0_f <- res0[which(res0 %in% rx_p0)]
r0_px_f <- res0[which(res0 %in% r0_px)]
rx_px_f <- res0[which(res0 %in% rx_px)]

r0_p0_s <- res1[which(res1 %in% r0_p0)]
rx_p0_s <- res1[which(res1 %in% rx_p0)]
r0_px_s <- res1[which(res1 %in% r0_px)]
rx_px_s <- res1[which(res1 %in% rx_px)]

#compute correlations and plot figs
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'All',1:dim(all_cue_fr_M1)[2])

compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0',r0)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RX',rx)

compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'P0',p0)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'PX',px)

compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'Succ',res1)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'Fail',res0)

#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0 Succ',r0_s)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0 Fail',r0_f)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RX Succ',rx_s)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RX Fail',rx_f)

#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'P0 Succ',p0_s)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'P0 Fail',p0_f)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'PX Succ',px_s)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'PX Fail',px_f)

#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0P0',r0_p0)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0PX',r0_px)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RXP0',rx_p0)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RXPX',rx_px)

#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0P0 Fail',r0_p0_f)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0PX Fail',r0_px_f)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RXP0 Fail',rx_p0_f)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RXPX Fail',rx_px_f)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0P0 Succ',r0_p0_s)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0PX Succ',r0_px_s)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RXP0 Succ',rx_p0_s)
#compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RXPX Succ',rx_px_s)
