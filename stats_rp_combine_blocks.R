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

nhp_id <- '504'
#nhp_id <- '0059'

if(nhp_id == '0059'){
  attach('0_8_1.RData')
  cat('0059\n')
  
  M1_sig_sign_percs_total <- M1_sig_sign_percs
  M1_total_unit_num <- length(M1_p_val_list$r0_p_vals[,1])
  S1_sig_sign_percs_total <- S1_sig_sign_percs
  S1_total_unit_num <- length(S1_p_val_list$r0_p_vals[,1])
  PmD_sig_sign_percs_total <- PmD_sig_sign_percs
  PmD_total_unit_num <- length(PmD_p_val_list$r0_p_vals[,1])
  
  detach()
  attach('0_8_2.RData')
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
  
  detach()
  attach('0_9_1.RData')
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
  
  detach()
  attach('0_9_2.RData')
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
  detach()
}else if(nhp_id == '504'){
  attach('5_8_1.RData')
  cat('504\n')
  
  M1_sig_sign_percs_total <- M1_sig_sign_percs
  M1_total_unit_num <- length(M1_p_val_list$r0_p_vals[,1])
  S1_sig_sign_percs_total <- S1_sig_sign_percs
  S1_total_unit_num <- length(S1_p_val_list$r0_p_vals[,1])
  PmD_sig_sign_percs_total <- PmD_sig_sign_percs
  PmD_total_unit_num <- length(PmD_p_val_list$r0_p_vals[,1])
  
  detach()
  attach('5_8_2.RData')
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
  
  detach()
  attach('5_9_1.RData')
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
  
  detach()
  attach('5_9_2.RData')
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
  
  detach()
  attach('5_14_1.RData')
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
  
  detach()
  attach('5_14_2.RData')
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
  
  detach()
  attach('5_14_3.RData')
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
  detach()
}

  
#######################
###plot ###############
########################

for(region_index in 1:length(region_list)){
  total_unit_num <- get(paste(region_list[region_index],'_total_unit_num',sep=""))
  out_sig_sign_percs <- get(paste(region_list[region_index],'_sig_sign_percs_total',sep=""))

  cat('plotting',region_list[region_index],'\n')
  window_names <- c('baseline vs \nafter cue','baseline vs \nbefore result','baseline vs \nafter result','baseline vs \nresult window','before result vs \nafter result','before result vs \nresult window')
  
  #reward
  png(paste(region_list[region_index],'_r_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$r0_sig_sign_percs[2,],out_sig_sign_percs$r1_sig_sign_percs[2,],out_sig_sign_percs$r2_sig_sign_percs[2,],out_sig_sign_percs$r3_sig_sign_percs[2,])
  rownames(num_inc) <- c(0,1,2,3)
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$r0_sig_sign_percs[3,],out_sig_sign_percs$r1_sig_sign_percs[3,],out_sig_sign_percs$r2_sig_sign_percs[3,],out_sig_sign_percs$r3_sig_sign_percs[3,])
  rownames(num_dec) <- c(0,1,2,3)
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
  
  num_inc <- rbind(out_sig_sign_percs$p0_sig_sign_percs[2,],out_sig_sign_percs$p1_sig_sign_percs[2,],out_sig_sign_percs$p2_sig_sign_percs[2,],out_sig_sign_percs$p3_sig_sign_percs[2,])
  rownames(num_inc) <- c(0,1,2,3)
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$p0_sig_sign_percs[3,],out_sig_sign_percs$p1_sig_sign_percs[3,],out_sig_sign_percs$p2_sig_sign_percs[3,],out_sig_sign_percs$p3_sig_sign_percs[3,])
  rownames(num_dec) <- c(0,1,2,3)
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
  
  #reward outcome
  png(paste(region_list[region_index],'_r_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$r0_succ_sig_sign_percs[2,],out_sig_sign_percs$r1_succ_sig_sign_percs[2,],out_sig_sign_percs$r2_succ_sig_sign_percs[2,],out_sig_sign_percs$r3_succ_sig_sign_percs[2,],out_sig_sign_percs$r0_fail_sig_sign_percs[2,],out_sig_sign_percs$r1_fail_sig_sign_percs[2,],out_sig_sign_percs$r2_fail_sig_sign_percs[2,],out_sig_sign_percs$r3_fail_sig_sign_percs[2,])
  rownames(num_inc) <- c('r0s','r1s','r2s','r3s','r0f','r1f','r2f','r3f')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$r0_succ_sig_sign_percs[3,],out_sig_sign_percs$r1_succ_sig_sign_percs[3,],out_sig_sign_percs$r2_succ_sig_sign_percs[3,],out_sig_sign_percs$r3_succ_sig_sign_percs[3,],out_sig_sign_percs$r0_fail_sig_sign_percs[3,],out_sig_sign_percs$r1_fail_sig_sign_percs[3,],out_sig_sign_percs$r2_fail_sig_sign_percs[3,],out_sig_sign_percs$r3_fail_sig_sign_percs[3,])
  rownames(num_dec) <- c('r0s','r1s','r2s','r3s','r0f','r1f','r2f','r3f')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Reward Level and Outcome',y='Number of units')
  plt <- plt + theme(axis.text.x = element_text(angle=45,hjust=1))
  
  plot(plt)
  graphics.off()
  
  #punishment outcome
  png(paste(region_list[region_index],'_p_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$p0_succ_sig_sign_percs[2,],out_sig_sign_percs$p1_succ_sig_sign_percs[2,],out_sig_sign_percs$p2_succ_sig_sign_percs[2,],out_sig_sign_percs$p3_succ_sig_sign_percs[2,],out_sig_sign_percs$p0_fail_sig_sign_percs[2,],out_sig_sign_percs$p1_fail_sig_sign_percs[2,],out_sig_sign_percs$p2_fail_sig_sign_percs[2,],out_sig_sign_percs$p3_fail_sig_sign_percs[2,])
  rownames(num_inc) <- c('p0s','p1s','p2s','p3s','p0f','p1f','p2f','p3f')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$p0_succ_sig_sign_percs[3,],out_sig_sign_percs$p1_succ_sig_sign_percs[3,],out_sig_sign_percs$p2_succ_sig_sign_percs[3,],out_sig_sign_percs$p3_succ_sig_sign_percs[3,],out_sig_sign_percs$p0_fail_sig_sign_percs[3,],out_sig_sign_percs$p1_fail_sig_sign_percs[3,],out_sig_sign_percs$p2_fail_sig_sign_percs[3,],out_sig_sign_percs$p3_fail_sig_sign_percs[3,])
  rownames(num_dec) <- c('p0s','p1s','p2s','p3s','p0f','p1f','p2f','p3f')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Punishment Level and Outcome',y='Number of units')
  plt <- plt + theme(axis.text.x = element_text(angle=45,hjust=1))
  
  plot(plt)
  graphics.off()
  
  #motivation
  png(paste(region_list[region_index],'_m_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$m0_sig_sign_percs[2,],out_sig_sign_percs$m1_sig_sign_percs[2,],out_sig_sign_percs$m2_sig_sign_percs[2,],out_sig_sign_percs$m3_sig_sign_percs[2,],out_sig_sign_percs$m4_sig_sign_percs[2,],out_sig_sign_percs$m5_sig_sign_percs[2,],out_sig_sign_percs$m6_sig_sign_percs[2,])
  rownames(num_inc) <- c('m0','m1','m2','m3','m4','m5','m6')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$m0_sig_sign_percs[3,],out_sig_sign_percs$m1_sig_sign_percs[3,],out_sig_sign_percs$m2_sig_sign_percs[3,],out_sig_sign_percs$m3_sig_sign_percs[3,],out_sig_sign_percs$m4_sig_sign_percs[3,],out_sig_sign_percs$m5_sig_sign_percs[3,],out_sig_sign_percs$m6_sig_sign_percs[3,])
  rownames(num_dec) <- c('m0','m1','m2','m3','m4','m5','m6')
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
  
  num_inc <- rbind(out_sig_sign_percs$v_3_sig_sign_percs[2,],out_sig_sign_percs$v_2_sig_sign_percs[2,],out_sig_sign_percs$v_1_sig_sign_percs[2,],out_sig_sign_percs$v0_sig_sign_percs[2,],out_sig_sign_percs$v1_sig_sign_percs[2,],out_sig_sign_percs$v2_sig_sign_percs[2,],out_sig_sign_percs$v3_sig_sign_percs[2,])
  rownames(num_inc) <- c('v_3','v_2','v_1','v0','v1','v2','v3')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$v_3_sig_sign_percs[3,],out_sig_sign_percs$v_2_sig_sign_percs[3,],out_sig_sign_percs$v_1_sig_sign_percs[3,],out_sig_sign_percs$v0_sig_sign_percs[3,],out_sig_sign_percs$v1_sig_sign_percs[3,],out_sig_sign_percs$v2_sig_sign_percs[3,],out_sig_sign_percs$v3_sig_sign_percs[3,])
  rownames(num_dec) <- c('v_3','v_2','v_1','v0','v1','v2','v3')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Motivation',y='Number of units')
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
  
  
  
}







