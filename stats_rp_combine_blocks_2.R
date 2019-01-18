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
#library(PMCMRplus)
#library(xlsx)

tryCatch({
  source("~/workspace/classification_scripts/multiplot.R")
  source("~/workspace/classification_scripts/Ski.Mack.JH.R")
  print('on laptop')
},warning=function(war){(print('on beaver'))
  source("~/workspace/classification_scripts/multiplot.R")
  source("~/workspace/classification_scripts/Ski.Mack.JH.R")
},finally={print('sourced multiplot and ski mack')})

####To generate:
#see run_stat_plot.sh
#from individual blocks, run python simple_fr_smoothing_gen.py then stats_rp_nocatch.R. Copy block.RDATA into a combination directory for each animal, rename block1.RData, block2.RData, etc
#then run this

#######Set these params #######
pval_adjusted_bool <- TRUE
#pval_adjusted_bool <- FALSE
##############################

saveAsPng <- T
region_list <- c('M1','S1','PmD')
binary_bool <- FALSE

ph_list_names <- c('comb','comb_outcome','m','p_catch','p','p_outcome','r_catch','r','r_outcome','v','r_bin','p_bin')
#ph_list_names <- c('comb','comb_outcome','m','p','p_outcome','r','r_outcome','v')
time_windows <- c('ac','br','ar','rw')

file_list <- Sys.glob("block*.RData")

ind = 1
for(block_name in file_list){
  cat(block_name)
  

  if(ind == 1){
    attach(block_name)

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
          tryCatch({
          
            temp <- get(name)
          
            assign(paste(name,'_totals',sep=""),list())
            if(length(temp) > 0){
              #cat(name,'\n')
              temp2 <- c(get(paste(name,'_totals',sep="")),temp)
              assign(paste(name,'_totals',sep=""),temp2)
            }
          },error=function(e){
            #cat(name)
          },finally={})
              
          }}}
    
    detach()
    
  }else{
    attach(block_name)
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
      #M1_diffs_length_list_rw_total[[name]] <- M1_diffs_length_list_rw_total[[name]] + M1_diffs_length_list_rw[[name]]
      S1_diffs_length_list_ac_total[[name]] <- S1_diffs_length_list_ac_total[[name]] + S1_diffs_length_list_ac[[name]]
      S1_diffs_length_list_br_total[[name]] <- S1_diffs_length_list_br_total[[name]] + S1_diffs_length_list_br[[name]]
      S1_diffs_length_list_ar_total[[name]] <- S1_diffs_length_list_ar_total[[name]] + S1_diffs_length_list_ar[[name]]
      #S1_diffs_length_list_rw_total[[name]] <- S1_diffs_length_list_rw_total[[name]] + S1_diffs_length_list_rw[[name]]
      PmD_diffs_length_list_ac_total[[name]] <- PmD_diffs_length_list_ac_total[[name]] + PmD_diffs_length_list_ac[[name]]
      PmD_diffs_length_list_br_total[[name]] <- PmD_diffs_length_list_br_total[[name]] + PmD_diffs_length_list_br[[name]]
      PmD_diffs_length_list_ar_total[[name]] <- PmD_diffs_length_list_ar_total[[name]] + PmD_diffs_length_list_ar[[name]]
      #PmD_diffs_length_list_rw_total[[name]] <- PmD_diffs_length_list_rw_total[[name]] + PmD_diffs_length_list_rw[[name]]
    }

    
    for(region_name in region_list){
      for(ph_name in ph_list_names){
        for(window_name in time_windows){
          name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,sep="")
          tryCatch({
            
            temp <- get(name)
            
            assign(paste(name,'_totals',sep=""),list())
            if(length(temp) > 0){
              #cat(name,'\n')
              temp2 <- c(get(paste(name,'_totals',sep="")),temp)
              assign(paste(name,'_totals',sep=""),temp2)
            }
          },error=function(e){
            #cat(name)
          },finally={})
          
        }}}
    
    
    detach()
  }
  ind <- ind + 1
}


  
#######################
###plot ###############
########################


for(region_index in 1:length(region_list)){
  total_unit_num <- get(paste(region_list[region_index],'_total_unit_num',sep=""))
  out_sig_sign_percs <- get(paste(region_list[region_index],'_sig_sign_percs_total',sep=""))

  cat('plotting',region_list[region_index],'\n')
  #window_names <- c('baseline vs \nafter cue','baseline vs \nbefore result','baseline vs \nafter result','baseline vs \nresult window','before result vs \nafter result','before result vs \nresult window')
  window_names <- c('BC vs AC','BR vs AR')
  
  
  #reward
  png(paste(region_list[region_index],'_r_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$r0_sig_sign_percs[2,],out_sig_sign_percs$r1_sig_sign_percs[2,],out_sig_sign_percs$r2_sig_sign_percs[2,],out_sig_sign_percs$r3_sig_sign_percs[2,])
  num_inc <- cbind(num_inc[,1],num_inc[,5])
  num_inc <- num_inc / total_unit_num
  
  if(dim(num_inc)[1] != 4){
    cat('binary\n')
  }else{
    rownames(num_inc) <- c(0,1,2,3)
    colnames(num_inc) <- window_names
    num_inc_melt <- melt(num_inc,varnames=c('level','window'))
    num_inc_melt$direction <- 'inc'
    
    num_dec <- rbind(out_sig_sign_percs$r0_sig_sign_percs[3,],out_sig_sign_percs$r1_sig_sign_percs[3,],out_sig_sign_percs$r2_sig_sign_percs[3,],out_sig_sign_percs$r3_sig_sign_percs[3,])
    num_dec <- cbind(num_dec[,1],num_dec[,5])
    num_dec <- num_dec / total_unit_num
    
    rownames(num_dec) <- c(0,1,2,3)
    colnames(num_dec) <- window_names
    num_dec_melt <- melt(num_dec,varnames=c('level','window'))
    num_dec_melt$direction <- 'dec'
    
    both_num <- rbind(num_inc_melt,num_dec_melt)
    
    plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
    plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Reward Level',y='Proportion Significant')
    plt <- plt + theme_classic() + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25) ,strip.text=element_text(size=16),plot.title=element_text(size=16),axis.title=element_text(size=16),axis.text=element_text(size=12))
    
    plot(plt)
  }
  graphics.off()
  
  #punishment
  png(paste(region_list[region_index],'_p_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  if(dim(num_inc)[1] != 4){
    cat('binary\n')
  }else{
    num_inc <- rbind(out_sig_sign_percs$p0_sig_sign_percs[2,],out_sig_sign_percs$p1_sig_sign_percs[2,],out_sig_sign_percs$p2_sig_sign_percs[2,],out_sig_sign_percs$p3_sig_sign_percs[2,])
    num_inc <- cbind(num_inc[,1],num_inc[,5])
    num_inc <- num_inc / total_unit_num
    rownames(num_inc) <- c(0,1,2,3)
    colnames(num_inc) <- window_names
    num_inc_melt <- melt(num_inc,varnames=c('level','window'))
    num_inc_melt$direction <- 'inc'
    
    num_dec <- rbind(out_sig_sign_percs$p0_sig_sign_percs[3,],out_sig_sign_percs$p1_sig_sign_percs[3,],out_sig_sign_percs$p2_sig_sign_percs[3,],out_sig_sign_percs$p3_sig_sign_percs[3,])
    num_dec <- cbind(num_dec[,1],num_dec[,5])
    num_dec <- num_dec / total_unit_num
    rownames(num_dec) <- c(0,1,2,3)
    colnames(num_dec) <- window_names
    num_dec_melt <- melt(num_dec,varnames=c('level','window'))
    num_dec_melt$direction <- 'dec'
    
    both_num <- rbind(num_inc_melt,num_dec_melt)
    
    plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
    plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Punishment Level',y='Proportion Significant')
    plt <- plt + theme_classic() + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25) ,strip.text=element_text(size=16),plot.title=element_text(size=16),axis.title=element_text(size=16),axis.text=element_text(size=12))
    
    plot(plt)
  }
  graphics.off()
  
  #rx outcome
  png(paste(region_list[region_index],'_rx_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$r0_s_sig_sign_percs[2,],out_sig_sign_percs$r0_f_sig_sign_percs[2,],out_sig_sign_percs$rx_s_sig_sign_percs[2,],out_sig_sign_percs$rx_f_sig_sign_percs[2,])
  num_inc <- cbind(num_inc[,1],num_inc[,5])
  num_inc <- num_inc / total_unit_num
  rownames(num_inc) <- c('r0s','r0f','rxs','rxf')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$r0_f_sig_sign_percs[3,],out_sig_sign_percs$r0_f_sig_sign_percs[3,],out_sig_sign_percs$rx_s_sig_sign_percs[3,],out_sig_sign_percs$rx_f_sig_sign_percs[3,])
  num_dec <- cbind(num_dec[,1],num_dec[,5])
  num_dec <- num_dec / total_unit_num
  rownames(num_dec) <- c('r0s','r0f','rxs','rxf')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Reward and Outcome',y='Proportion Significant')
  plt <- plt + theme_classic() + theme(strip.text=element_text(size=16),plot.title=element_text(size=16),axis.title=element_text(size=16),axis.text=element_text(size=12))
  
  plot(plt)
  graphics.off()
  
  #px outcome
  png(paste(region_list[region_index],'_px_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$p0_s_sig_sign_percs[2,],out_sig_sign_percs$p0_f_sig_sign_percs[2,],out_sig_sign_percs$px_s_sig_sign_percs[2,],out_sig_sign_percs$px_f_sig_sign_percs[2,])
  num_inc <- cbind(num_inc[,1],num_inc[,5])
  num_inc <- num_inc / total_unit_num
  rownames(num_inc) <- c('p0s','p0f','pxs','pxf')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$p0_f_sig_sign_percs[3,],out_sig_sign_percs$p0_f_sig_sign_percs[3,],out_sig_sign_percs$px_s_sig_sign_percs[3,],out_sig_sign_percs$px_f_sig_sign_percs[3,])
  num_dec <- cbind(num_dec[,1],num_dec[,5])
  num_dec <- num_dec / total_unit_num
  rownames(num_dec) <- c('p0s','p0f','pxs','pxf')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Punishment and Outcome',y='Proportion Significant')
  plt <- plt + theme_classic() + theme(strip.text=element_text(size=16),plot.title=element_text(size=16),axis.title=element_text(size=16),axis.text=element_text(size=12))
  
  plot(plt)
  graphics.off()
  
  #r binary
  png(paste(region_list[region_index],'_r_binary_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$r0_sig_sign_percs[2,],out_sig_sign_percs$rx_sig_sign_percs[2,])
  num_inc <- cbind(num_inc[,1],num_inc[,5])
  num_inc <- num_inc / total_unit_num
  rownames(num_inc) <- c('r0','rx')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$r0_sig_sign_percs[3,],out_sig_sign_percs$rx_sig_sign_percs[3,])
  num_dec <- cbind(num_dec[,1],num_dec[,5])
  num_dec <- num_dec / total_unit_num
  rownames(num_dec) <- c('r0','rx')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Reward and Outcome',y='Proportion Significant')
  plt <- plt + theme_classic() + theme(strip.text=element_text(size=16),plot.title=element_text(size=16),axis.title=element_text(size=16),axis.text=element_text(size=12))
  
  plot(plt)
  graphics.off()
  
  #px outcome
  png(paste(region_list[region_index],'_p_binary_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$p0_sig_sign_percs[2,],out_sig_sign_percs$px_sig_sign_percs[2,])
  num_inc <- cbind(num_inc[,1],num_inc[,5])
  num_inc <- num_inc / total_unit_num
  rownames(num_inc) <- c('p0','px')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$p0_sig_sign_percs[3,],out_sig_sign_percs$px_sig_sign_percs[3,])
  num_dec <- cbind(num_dec[,1],num_dec[,5])
  num_dec <- num_dec / total_unit_num
  rownames(num_dec) <- c('p0','px')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Punishment and Outcome',y='Proportion Significant')
  plt <- plt + theme_classic() + theme(strip.text=element_text(size=16),plot.title=element_text(size=16),axis.title=element_text(size=16),axis.text=element_text(size=12))
  
  plot(plt)
  graphics.off()
  
  
  #res outcome
  png(paste(region_list[region_index],'_res_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$res0_sig_sign_percs[2,],out_sig_sign_percs$res1_sig_sign_percs[2,])
  num_inc <- cbind(num_inc[,1],num_inc[,5])
  num_inc <- num_inc / total_unit_num
  rownames(num_inc) <- c('fail','succ')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$res0_sig_sign_percs[3,],out_sig_sign_percs$res1_sig_sign_percs[3,])
  num_dec <- cbind(num_dec[,1],num_dec[,5])
  num_dec <- num_dec / total_unit_num
  rownames(num_dec) <- c('fail','succ')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Result',y='Proportion Significant')
  plt <- plt + theme_classic() + theme(strip.text=element_text(size=16),plot.title=element_text(size=16),axis.title=element_text(size=16),axis.text=element_text(size=12))
  
  plot(plt)
  graphics.off()
  
  #comb
  png(paste(region_list[region_index],'_comb_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$r0_p0_sig_sign_percs[2,],out_sig_sign_percs$rx_p0_sig_sign_percs[2,],out_sig_sign_percs$r0_px_sig_sign_percs[2,],out_sig_sign_percs$rx_px_sig_sign_percs[2,])
  num_inc <- cbind(num_inc[,1],num_inc[,5])
  num_inc <- num_inc / total_unit_num
  rownames(num_inc) <- c('r0p0','rxp0','r0px','rxpx')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$r0_p0_sig_sign_percs[3,],out_sig_sign_percs$rx_p0_sig_sign_percs[3,],out_sig_sign_percs$r0_px_sig_sign_percs[3,],out_sig_sign_percs$rx_px_sig_sign_percs[3,])
  num_dec <- cbind(num_dec[,1],num_dec[,5])
  num_dec <- num_dec / total_unit_num
  rownames(num_dec) <- c('r0p0','rxp0','r0px','rxpx')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Combination',y='Proportion Significant')
  plt <- plt + theme(axis.text.x = element_text(size=rel(0.8)))
  plt <- plt + theme_classic() + theme(strip.text=element_text(size=16),plot.title=element_text(size=16),axis.title=element_text(size=16),axis.text=element_text(size=12))
  
  plot(plt)
  graphics.off()
  
  #comb outcome
  png(paste(region_list[region_index],'_comb_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$r0_p0_s_sig_sign_percs[2,],out_sig_sign_percs$rx_p0_s_sig_sign_percs[2,],out_sig_sign_percs$r0_px_s_sig_sign_percs[2,],out_sig_sign_percs$rx_px_s_sig_sign_percs[2,],out_sig_sign_percs$r0_p0_f_sig_sign_percs[2,],out_sig_sign_percs$rx_p0_f_sig_sign_percs[2,],out_sig_sign_percs$r0_px_f_sig_sign_percs[2,],out_sig_sign_percs$rx_px_f_sig_sign_percs[2,])
  num_inc <- cbind(num_inc[,1],num_inc[,5])
  num_inc <- num_inc / total_unit_num
  rownames(num_inc) <- c('r0p0s','rxp0s','r0pxs','rxpxs','r0p0f','rxp0f','r0pxf','rxpxf')
  colnames(num_inc) <- window_names
  num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  num_inc_melt$direction <- 'inc'
  
  num_dec <- rbind(out_sig_sign_percs$r0_p0_s_sig_sign_percs[3,],out_sig_sign_percs$rx_p0_s_sig_sign_percs[3,],out_sig_sign_percs$r0_px_s_sig_sign_percs[3,],out_sig_sign_percs$rx_px_s_sig_sign_percs[3,],out_sig_sign_percs$r0_p0_f_sig_sign_percs[3,],out_sig_sign_percs$rx_p0_f_sig_sign_percs[3,],out_sig_sign_percs$r0_px_f_sig_sign_percs[3,],out_sig_sign_percs$rx_px_f_sig_sign_percs[3,])
  num_dec <- cbind(num_dec[,1],num_dec[,5])
  num_dec <- num_dec / total_unit_num
  rownames(num_dec) <- c('r0p0s','rxp0s','r0pxs','rxpxs','r0p0f','rxp0f','r0pxf','rxpxf')
  colnames(num_dec) <- window_names
  num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  num_dec_melt$direction <- 'dec'
  
  both_num <- rbind(num_inc_melt,num_dec_melt)
  
  plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Combination and Outcome',y='Proportion Significant')
  plt <- plt + theme(axis.text.x = element_text(angle=45,hjust=1))
  plt <- plt + theme_classic() + theme(strip.text=element_text(size=16),plot.title=element_text(size=16),axis.title=element_text(size=16),axis.text=element_text(size=12))
  
  plot(plt)
  graphics.off()
  
  #reward outcome
  png(paste(region_list[region_index],'_r_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$r0_succ_sig_sign_percs[2,],out_sig_sign_percs$r1_succ_sig_sign_percs[2,],out_sig_sign_percs$r2_succ_sig_sign_percs[2,],out_sig_sign_percs$r3_succ_sig_sign_percs[2,],out_sig_sign_percs$r0_fail_sig_sign_percs[2,],out_sig_sign_percs$r1_fail_sig_sign_percs[2,],out_sig_sign_percs$r2_fail_sig_sign_percs[2,],out_sig_sign_percs$r3_fail_sig_sign_percs[2,])
  num_inc <- cbind(num_inc[,1],num_inc[,5])
  num_inc <- num_inc / total_unit_num
  if(length(num_inc) > 0){
    if(dim(num_inc)[1] != 8){
      cat('binary\n')
    }else{
      rownames(num_inc) <- c('r0s','r1s','r2s','r3s','r0f','r1f','r2f','r3f')
      colnames(num_inc) <- window_names
      num_inc_melt <- melt(num_inc,varnames=c('level','window'))
      num_inc_melt$direction <- 'inc'
      
      num_dec <- rbind(out_sig_sign_percs$r0_succ_sig_sign_percs[3,],out_sig_sign_percs$r1_succ_sig_sign_percs[3,],out_sig_sign_percs$r2_succ_sig_sign_percs[3,],out_sig_sign_percs$r3_succ_sig_sign_percs[3,],out_sig_sign_percs$r0_fail_sig_sign_percs[3,],out_sig_sign_percs$r1_fail_sig_sign_percs[3,],out_sig_sign_percs$r2_fail_sig_sign_percs[3,],out_sig_sign_percs$r3_fail_sig_sign_percs[3,])
      num_dec <- cbind(num_dec[,1],num_dec[,5])
      num_dec <- num_dec / total_unit_num
      rownames(num_dec) <- c('r0s','r1s','r2s','r3s','r0f','r1f','r2f','r3f')
      colnames(num_dec) <- window_names
      num_dec_melt <- melt(num_dec,varnames=c('level','window'))
      num_dec_melt$direction <- 'dec'
      
      both_num <- rbind(num_inc_melt,num_dec_melt)
      
      plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
      plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Reward Level and Outcome',y='Proportion Significant')
      plt <- plt + theme(axis.text.x = element_text(angle=45,hjust=1))
      plt <- plt + theme_classic() + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25) ,strip.text=element_text(size=16),plot.title=element_text(size=16),axis.title=element_text(size=16),axis.text=element_text(size=12))
      
      plot(plt)
  }}
  graphics.off()
  
  #punishment outcome
  png(paste(region_list[region_index],'_p_outcome_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  num_inc <- rbind(out_sig_sign_percs$p0_succ_sig_sign_percs[2,],out_sig_sign_percs$p1_succ_sig_sign_percs[2,],out_sig_sign_percs$p2_succ_sig_sign_percs[2,],out_sig_sign_percs$p3_succ_sig_sign_percs[2,],out_sig_sign_percs$p0_fail_sig_sign_percs[2,],out_sig_sign_percs$p1_fail_sig_sign_percs[2,],out_sig_sign_percs$p2_fail_sig_sign_percs[2,],out_sig_sign_percs$p3_fail_sig_sign_percs[2,])
  num_inc <- cbind(num_inc[,1],num_inc[,5])
  num_inc <- num_inc / total_unit_num
  if(length(num_inc) > 0){
    if(dim(num_inc)[1] != 8){
      cat('binary\n')
    }else{
      rownames(num_inc) <- c('p0s','p1s','p2s','p3s','p0f','p1f','p2f','p3f')
      colnames(num_inc) <- window_names
      num_inc_melt <- melt(num_inc,varnames=c('level','window'))
      num_inc_melt$direction <- 'inc'
      
      num_dec <- rbind(out_sig_sign_percs$p0_succ_sig_sign_percs[3,],out_sig_sign_percs$p1_succ_sig_sign_percs[3,],out_sig_sign_percs$p2_succ_sig_sign_percs[3,],out_sig_sign_percs$p3_succ_sig_sign_percs[3,],out_sig_sign_percs$p0_fail_sig_sign_percs[3,],out_sig_sign_percs$p1_fail_sig_sign_percs[3,],out_sig_sign_percs$p2_fail_sig_sign_percs[3,],out_sig_sign_percs$p3_fail_sig_sign_percs[3,])
      num_dec <- cbind(num_dec[,1],num_dec[,5])
      num_dec <- num_dec / total_unit_num
      rownames(num_dec) <- c('p0s','p1s','p2s','p3s','p0f','p1f','p2f','p3f')
      colnames(num_dec) <- window_names
      num_dec_melt <- melt(num_dec,varnames=c('level','window'))
      num_dec_melt$direction <- 'dec'
      
      both_num <- rbind(num_inc_melt,num_dec_melt)
      
      plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
      plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Punishment Level and Outcome',y='Proportion Significant')
      plt <- plt + theme(axis.text.x = element_text(angle=45,hjust=1))
      plt <- plt + theme_classic() + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25) ,strip.text=element_text(size=16),plot.title=element_text(size=16),axis.title=element_text(size=16),axis.text=element_text(size=12))
      
      plot(plt)
  }}
  graphics.off()
  # 
  # #motivation
  # num_inc <- rbind(out_sig_sign_percs$m0_sig_sign_percs[2,],out_sig_sign_percs$m1_sig_sign_percs[2,],out_sig_sign_percs$m2_sig_sign_percs[2,],out_sig_sign_percs$m3_sig_sign_percs[2,],out_sig_sign_percs$m4_sig_sign_percs[2,],out_sig_sign_percs$m5_sig_sign_percs[2,],out_sig_sign_percs$m6_sig_sign_percs[2,])
  # if(dim(num_inc)[1] == 7){
  #   png(paste(region_list[region_index],'_m_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  #   
  #   num_inc <- cbind(num_inc[,1],num_inc[,5])
  #   num_inc <- num_inc / total_unit_num
  #   rownames(num_inc) <- c('m0','m1','m2','m3','m4','m5','m6')
  #   colnames(num_inc) <- window_names
  #   num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  #   num_inc_melt$direction <- 'inc'
  # 
  #   num_dec <- rbind(out_sig_sign_percs$m0_sig_sign_percs[3,],out_sig_sign_percs$m1_sig_sign_percs[3,],out_sig_sign_percs$m2_sig_sign_percs[3,],out_sig_sign_percs$m3_sig_sign_percs[3,],out_sig_sign_percs$m4_sig_sign_percs[3,],out_sig_sign_percs$m5_sig_sign_percs[3,],out_sig_sign_percs$m6_sig_sign_percs[3,])
  # 
  #   num_dec <- cbind(num_dec[,1],num_dec[,5])
  #   num_dec <- num_dec / total_unit_num
  #   rownames(num_dec) <- c('m0','m1','m2','m3','m4','m5','m6')
  #   colnames(num_dec) <- window_names
  #   num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  #   num_dec_melt$direction <- 'dec'
  # 
  #   both_num <- rbind(num_inc_melt,num_dec_melt)
  # 
  #   plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  #   plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Motivation',y='Proportion Significant')
  #   plt <- plt + theme(axis.text.x = element_text(size=rel(0.8)))
  # 
  #   plot(plt)
  #   graphics.off()
  # }else{
  #   
  #   #motivation binary
  #   png(paste(region_list[region_index],'_m_binary_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  #   
  #   num_inc <- rbind(out_sig_sign_percs$m0_sig_sign_percs[2,],out_sig_sign_percs$mx_sig_sign_percs[2,],out_sig_sign_percs$m2x_sig_sign_percs[2,])
  #   num_inc <- cbind(num_inc[,1],num_inc[,5])
  #   num_inc <- num_inc / total_unit_num
  #   rownames(num_inc) <- c('m0','mx','m2x')
  #   colnames(num_inc) <- window_names
  #   num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  #   num_inc_melt$direction <- 'inc'
  #     
  #   num_dec <- rbind(out_sig_sign_percs$m0_sig_sign_percs[3,],out_sig_sign_percs$mx_sig_sign_percs[3,],out_sig_sign_percs$m2x_sig_sign_percs[3,])
  #     
  #   num_dec <- cbind(num_dec[,1],num_dec[,5])
  #   num_dec <- num_dec / total_unit_num
  #   rownames(num_dec) <- c('m0','mx','m2x')
  #   colnames(num_dec) <- window_names
  #   num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  #   num_dec_melt$direction <- 'dec'
  #     
  #   both_num <- rbind(num_inc_melt,num_dec_melt)
  #     
  #   plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  #   plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Motivation',y='Proportion Significant')
  #   plt <- plt + theme(axis.text.x = element_text(size=rel(0.8)))
  #     
  #   plot(plt)
  #   graphics.off()
  # }
  #   
  # #value
  # 
  # num_inc <- rbind(out_sig_sign_percs$v_3_sig_sign_percs[2,],out_sig_sign_percs$v_2_sig_sign_percs[2,],out_sig_sign_percs$v_1_sig_sign_percs[2,],out_sig_sign_percs$v0_sig_sign_percs[2,],out_sig_sign_percs$v1_sig_sign_percs[2,],out_sig_sign_percs$v2_sig_sign_percs[2,],out_sig_sign_percs$v3_sig_sign_percs[2,])
  # if(dim(num_inc)[1] == 7){
  #   png(paste(region_list[region_index],'_v_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  #   
  #   num_inc <- cbind(num_inc[,1],num_inc[,5])
  #   num_inc <- num_inc / total_unit_num
  #   rownames(num_inc) <- c('v_3','v_2','v_1','v0','v1','v2','v3')
  #   colnames(num_inc) <- window_names
  #   num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  #   num_inc_melt$direction <- 'inc'
  # 
  #   num_dec <- rbind(out_sig_sign_percs$v_3_sig_sign_percs[3,],out_sig_sign_percs$v_2_sig_sign_percs[3,],out_sig_sign_percs$v_1_sig_sign_percs[3,],out_sig_sign_percs$v0_sig_sign_percs[3,],out_sig_sign_percs$v1_sig_sign_percs[3,],out_sig_sign_percs$v2_sig_sign_percs[3,],out_sig_sign_percs$v3_sig_sign_percs[3,])
  # 
  #   num_dec <- cbind(num_dec[,1],num_dec[,5])
  #   num_dec <- num_dec / total_unit_num
  #   rownames(num_dec) <- c('v_3','v_2','v_1','v0','v1','v2','v3')
  #   colnames(num_dec) <- window_names
  #   num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  #   num_dec_melt$direction <- 'dec'
  # 
  #   both_num <- rbind(num_inc_melt,num_dec_melt)
  # 
  #   plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  #   plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Value',y='Proportion Significant')
  #   plt <- plt + theme(axis.text.x = element_text(size=rel(0.8)))
  # 
  #   plot(plt)
  #   graphics.off()
  # }else{
  # 
  #   #value binary
  #   png(paste(region_list[region_index],'_v_binary_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  #   
  #   num_inc <- rbind(out_sig_sign_percs$v_x_sig_sign_percs[2,],out_sig_sign_percs$v0_sig_sign_percs[2,],out_sig_sign_percs$vx_sig_sign_percs[2,])
  #   num_inc <- cbind(num_inc[,1],num_inc[,5])
  #   num_inc <- num_inc / total_unit_num
  #   rownames(num_inc) <- c('v_x','v0','vx')
  #   colnames(num_inc) <- window_names
  #   num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  #   num_inc_melt$direction <- 'inc'
  #     
  #   num_dec <- rbind(out_sig_sign_percs$v_x_sig_sign_percs[3,],out_sig_sign_percs$v0_sig_sign_percs[3,],out_sig_sign_percs$vx_sig_sign_percs[3,])
  #     
  #   num_dec <- cbind(num_dec[,1],num_dec[,5])
  #   num_dec <- num_dec / total_unit_num
  #   rownames(num_dec) <- c('v_x','v0','vx')
  #   colnames(num_dec) <- window_names
  #   num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  #   num_dec_melt$direction <- 'dec'
  #     
  #   both_num <- rbind(num_inc_melt,num_dec_melt)
  #     
  #   plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  #   plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Value',y='Proportion Significant')
  #   plt <- plt + theme(axis.text.x = element_text(size=rel(0.8)))
  #     
  #   plot(plt)
  #   graphics.off()
  # }
  #   
    
  #catch
  #png(paste(region_list[region_index],'_catch_sig_diffs_total.png',sep=""),width=8,height=6,units="in",res=500)
  
  #num_inc <- rbind(out_sig_sign_percs$catch_x_sig_sign_percs[2,],out_sig_sign_percs$catchx_sig_sign_percs[2,])
  #
  #num_inc <- cbind(num_inc[,1],num_inc[,5])
  #num_inc <- num_inc / total_unit_num
  #rownames(num_inc) <- c('P catch trial','R catch trial')
  #colnames(num_inc) <- window_names
  #num_inc_melt <- melt(num_inc,varnames=c('level','window'))
  #num_inc_melt$direction <- 'inc'
  
  #num_dec <- rbind(out_sig_sign_percs$res0_sig_sign_percs[3,],out_sig_sign_percs$res1_sig_sign_percs[3,])
  #
  #num_dec <- cbind(num_dec[,1],num_dec[,5])
  #num_dec <- num_dec / total_unit_num
  #rownames(num_dec) <- c('P catch trial','R catch trial')
  #colnames(num_dec) <- window_names
  #num_dec_melt <- melt(num_dec,varnames=c('level','window'))
  #num_dec_melt$direction <- 'dec'
  
  #both_num <- rbind(num_inc_melt,num_dec_melt)
  
  #plt <- ggplot() + geom_bar(data=both_num,aes(y=value,x=level,fill=direction),stat="identity",position="stack",show.legend=F) + facet_grid(~window)
  #plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Catch trials',y='Proportion Significant')
  #plt <- plt + theme(axis.text.x = element_text(angle=45,hjust=1))
  #
  #plot(plt)
  #graphics.off()
  
  #####
  #####
  diffs_ac <- get(paste(region_list[region_index],'_diffs_length_list_ac_total',sep=""))
  diffs_br <- get(paste(region_list[region_index],'_diffs_length_list_br_total',sep=""))
  diffs_ar <- get(paste(region_list[region_index],'_diffs_length_list_ar_total',sep=""))
  diffs_rw <- get(paste(region_list[region_index],'_diffs_length_list_rw_total',sep=""))
  
  all_diffs_length <- rbind(aft_cue=diffs_ac,bfr_res=diffs_br,aft_res=diffs_ar,res_wind=diffs_rw)

  assign(paste(region_list[region_index],'_all_diffs_length',sep=""),all_diffs_length)
  #write.table(all_diffs_length,file=paste(region_list[region_index],'_all_diffs_length.csv',sep=""),sep=",",col.names=NA)
  if(pval_adjusted_bool){
    write.table(all_diffs_length,file=paste(region_list[region_index],'_all_diffs_length_padj.csv',sep=""),sep=",",col.names=NA)
  }else{  
    write.table(all_diffs_length,file=paste(region_list[region_index],'_all_diffs_length.csv',sep=""),sep=",",col.names=NA)
  }
}


for(region_name in region_list){
  for(ph_name in ph_list_names){
    cat(ph_name,'\n')
    perc_list_windows <- list()
    total_by_window.l <- list()
    for(window_name in time_windows){
      cat(window_name,'\n')
      name <- paste(region_name,'_ph_',ph_name,'_levels_',window_name,'_totals',sep="")
      
      tryCatch({
      temp <- get(name)
      
      
      if(length(temp) > 0){
        
        sig_unit_num <- length(temp)
        cat(name,':',sig_unit_num,'total sig units\n')
        ph_sig_num <- 0
        
        ph_list <- list()
        for(i in 1:sig_unit_num){
          if(pval_adjusted_bool){
            sig_comparisons <- temp[[i]]$comparisons[temp[[i]]$P.adjusted < 0.05]
            if(!identical(temp[[i]]$P < 0.05,temp[[i]]$P.adjusted < 0.05)){
              cat('not equal\n')
            }
          }else{
            sig_comparisons <- temp[[i]]$comparisons[temp[[i]]$P < 0.05]
          }
          
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
          
          if(ph_name == 'comb'){
            comp_name_list <- c('r0_p0 - r0_px','r0_p0 - rx_p0','r0_p0 - rx_px','r0_px - rx_p0','r0_px - rx_px','rx_p0 - rx_px')
          }else if(ph_name == 'comb_outcome'){
            comp_name_list <- c('r0_p0_f - r0_p0_s','r0_p0_f - r0_px_f','r0_p0_f - r0_px_s','r0_p0_f - rx_p0_f','r0_p0_f - rx_p0_s','r0_p0_f - rx_px_f','r0_p0_f - rx_px_s','r0_p0_s - r0_px_f','r0_p0_s - r0_px_s','r0_p0_s - rx_p0_f','r0_p0_s - rx_p0_s','r0_p0_s - rx_px_f','r0_p0_s - rx_px_s','r0_px_f - r0_px_s','r0_px_f - rx_p0_f',
                                'r0_px_f - rx_p0_s','r0_px_f - rx_px_f','r0_px_f - rx_px_s','r0_px_s - rx_p0_f','r0_px_s - rx_p0_s','r0_px_s - rx_px_f','r0_px_s - rx_px_s','rx_p0_f - rx_p0_s','rx_p0_f - rx_px_f','rx_p0_f - rx_px_s','rx_p0_s - rx_px_f','rx_p0_s - rx_px_s','rx_px_f - rx_px_s')
          }else if(ph_name == 'm'){
            comp_name_list <- c('m0 - m1','m0 - m2','m0 - m3','m0 - m4','m0 - m5','m0 - m6','m1 - m2','m1 - m3','m1 - m4','m1 - m5','m1 - m6','m2 - m3','m2 - m4','m2 - m5','m2 - m6','m3 - m4','m3 - m5','m3 - m6','m4 - m5','m4 - m6','m5 - m6')
          }else if(ph_name == 'p_catch'){
            comp_name_list <- c('catch_x - p0_f','catch_x - px_f','p0_f - px_f')
          }else if(ph_name == 'p_outcome'){
            comp_name_list <- c('p0_f - p0_s','p0_f - px_f','p0_f - px_s','p0_s - px_f','p0_s - px_s','px_f - px_s')
          }else if(ph_name == 'r_catch'){
            comp_name_list <- c('catchx - r0_s','catchx - rx_s','r0_s - rx_s')
          }else if(ph_name == 'r_outcome'){
            comp_name_list <- c('r0_f - r0_s','r0_f - rx_f','r0_f - rx_s','r0_s - rx_f','r0_s - rx_s','rx_f - rx_s')
          }else if(ph_name == 'v'){
            comp_name_list <- c('v_1 - v_2','v_1 - v_3','v_1 - v2','v_1 - v3','v_2 - v_3','v_2 - v3','v0 - v_1','v0 - v_2','v0 - v_3','v0 - v1','v0 - v2','v0 - v3','v1 - v_1','v1 - v_2','v1 - v_3','v1 - v2','v1 - v3','v2 - v_2','v2 - v_3','v2 - v3','v3 - v_3')
          }else if(ph_name == 'p'){
            comp_name_list <- c('p0 - p1','p0 - p2','p0 - p3','p1 - p2','p1 - p3','p2 - p3')
          }else if (ph_name == 'r'){
            comp_name_list <- c('r0 - r1','r0 - r2','r0 - r3','r1 - r2','r1 - r3','r2 - r3')
          }
        
          for(comp_name in comp_name_list){
            ct <- 0
            
            for(i in 1:ph_sig_num){
              if(comp_name %in% ph_list[[i]]){
                ct <- ct+1
              }
            }
            comp_perc_list[[comp_name]] <- ct / ph_sig_num
          }
          perc_list_windows[[window_name]] <- comp_perc_list
          total_by_window.l[[window_name]] <- ph_sig_num
          
        }
      }
      },error=function(e){},finally={})
      
    }
    
    if(length(perc_list_windows) > 0){
      suppressWarnings(perc_list_windows_cbind <- as.data.frame(do.call(cbind,perc_list_windows)))
      
      perc_list_windows_cbind <- rbind(perc_list_windows_cbind,total=total_by_window.l,make.row.names=T)
      assign(paste(region_name,'_',ph_name,'_cpl',sep=""),perc_list_windows_cbind)
      if(pval_adjusted_bool){
        write.xlsx(perc_list_windows_cbind,file=paste(region_name,'_ph_percs_padj.xlsx',sep=""),sheetName=ph_name,append=T)
      }else{  
        write.xlsx(perc_list_windows_cbind,file=paste(region_name,'_ph_percs.xlsx',sep=""),sheetName=ph_name,append=T)
      }

    }  }
}

save.image("summary.RData")

#rm(list=ls())



