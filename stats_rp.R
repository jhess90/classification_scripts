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

tryCatch({
  source("~/documents/lab/workspace/Classification_scripts/multiplot.R")
  print('on laptop')
  },warning=function(war){(print('on beaver'))
    source("~/workspace/classification_scripts/multiplot.R")
  },finally={print('sourced multiplot')})

saveAsPng <- T
region_list <- c('M1','S1','PmD')



##### func

calc_wsr <- function(baseline,window){
  
  if(identical(baseline,window)){return(NA)}
  
  suppressWarnings(wsr <- wilcox.test(baseline,window,paired=T))
  p_val <- wsr$p.v

  if (mean(baseline) > mean(window)){change = -1}else if(mean(baseline) < mean(window)){change = 1}else{change=0}
  
  output <- c(p_val,change)
  
  return(output)
}

##########


for(region_index in 1:length(region_list)){
  cat("region:",region_list[region_index],'\n')
  
  readin <- readMat(paste('simple_output_',region_list[region_index],'.mat',sep=""))
  
  all_cue_fr <- readin$return.dict[,,1]$all.cue.fr
  all_res_fr <- readin$return.dict[,,1]$all.res.fr
  condensed <- readin$return.dict[,,1]$condensed
  bin_size <- readin$return.dict[,,1]$params[,,1]$bin.size[,]
  total_unit_num <- dim(all_cue_fr)[1]

  old_time <- seq(from=-0.5,to=(1.0-bin_size/1000),by=bin_size/1000)

  ###########
  r0 <- which(condensed[,4] == 0)
  r1 <- which(condensed[,4] == 1)
  r2 <- which(condensed[,4] == 2)
  r3 <- which(condensed[,4] == 3)
  rx <- which(condensed[,4] >= 1)
  
  p0 <- which(condensed[,5] == 0)
  p1 <- which(condensed[,5] == 1)
  p2 <- which(condensed[,5] == 2)
  p3 <- which(condensed[,5] == 3)
  px <- which(condensed[,5] >= 1)
  
  v_3 <- which(condensed[,7] == -3)
  v_2 <- which(condensed[,7] == -2)
  v_1 <- which(condensed[,7] == -1)
  v0 <- which(condensed[,7] == 0)
  v1 <- which(condensed[,7] == 1)
  v2 <- which(condensed[,7] == 2)
  v3 <- which(condensed[,7] == 3)
  
  m0 <- which(condensed[,8] == 0)
  m1 <- which(condensed[,8] == 1)
  m2 <- which(condensed[,8] == 2)
  m3 <- which(condensed[,8] == 3)
  m4 <- which(condensed[,8] == 4)
  m5 <- which(condensed[,8] == 5)
  m6 <- which(condensed[,8] == 6)
  
  res0 <- which(condensed[,6] == 0)
  res1 <- which(condensed[,6] == 1)
  
  r0_fail <- res0[which(res0 %in% r0)]
  r1_fail <- res0[which(res0 %in% r1)]
  r2_fail <- res0[which(res0 %in% r2)]
  r3_fail <- res0[which(res0 %in% r3)]
  r0_succ <- res1[which(res1 %in% r0)]
  r1_succ <- res1[which(res1 %in% r1)]
  r2_succ <- res1[which(res1 %in% r2)]
  r3_succ <- res1[which(res1 %in% r3)]
  
  p0_fail <- res0[which(res0 %in% p0)]
  p1_fail <- res0[which(res0 %in% p1)]
  p2_fail <- res0[which(res0 %in% p2)]
  p3_fail <- res0[which(res0 %in% p3)]
  p0_succ <- res1[which(res1 %in% p0)]
  p1_succ <- res1[which(res1 %in% p1)]
  p2_succ <- res1[which(res1 %in% p2)]
  p3_succ <- res1[which(res1 %in% p3)]
  
  catch_x <- which(condensed[,12] <= -1)
  catch0 <- which(condensed[,12] == 0)
  catchx <- which(condensed[,12] >= 1)
  
  catch_3 <- which(condensed[,12] == -3)
  catch_2 <- which(condensed[,12] == -2)
  catch_1 <- which(condensed[,12] == -1)
  catch1 <- which(condensed[,12] == 1)
  catch2 <- which(condensed[,12] == 2)
  catch3 <- which(condensed[,12] == 3)
  
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
  
  r0_p0_s <- res1[which(res1 %in% r0_p0)]
  rx_p0_s <- res1[which(res1 %in% rx_p0)]
  r0_px_s <- res1[which(res1 %in% r0_px)]
  rx_px_s <- res1[which(res1 %in% rx_px)]
  r0_p0_f <- res0[which(res0 %in% r0_p0)]
  rx_p0_f <- res0[which(res0 %in% rx_p0)]
  r0_px_f <- res0[which(res0 %in% r0_px)]
  rx_px_f <- res0[which(res0 %in% rx_px)]
  
  #########################
  
  comb_list <- c('r0','r1','r2','r3','p0','p1','p2','p3','res0','res1','r0_fail','r1_fail','r2_fail','r3_fail','r0_succ','r1_succ','r2_succ','r3_succ','p0_fail','p1_fail','p2_fail','p3_fail','p0_succ','p1_succ','p2_succ','p3_succ','catch_x','catch0','catchx','rx','px','r0_f','rx_f','r0_s','rx_s','p0_f','px_f','p0_s','px_s','r0_p0','rx_p0','r0_px','rx_px','r0_p0_s','rx_p0_s','r0_px_s','rx_px_s','r0_p0_f','rx_p0_f','r0_px_f','rx_px_f','v_3','v_2','v_1','v0','v1','v2','v3','m0','m1','m2','m3','m4','m5','m6')
  
  out_p_list <- c()
  out_perc_sig_list <- c()
  out_mean_diff_array <- c()
  out_sig_sign_percs <- c()
  
  for (i in comb_list){
    comb_inds <- get(i)
    
    if(length(comb_inds) == 0){
      cat('no instances of',i,'\n')
      sig_sign_percs <- cbind(bl_ac = rep(0,5),bl_br = rep(0,5),bl_ar = rep(0,5),bl_rw = rep(0,5),br_ar = rep(0,5),br_rw = rep(0,5))
      out_sig_sign_percs[[paste(i,'_sig_sign_percs',sep="")]] <- sig_sign_percs
      next
    }else if(length(comb_inds) == 1){
      cat('only one instance of',i,'\n')
      sig_sign_percs <- cbind(bl_ac = rep(0,5),bl_br = rep(0,5),bl_ar = rep(0,5),bl_rw = rep(0,5),br_ar = rep(0,5),br_rw = rep(0,5))
      out_sig_sign_percs[[paste(i,'_sig_sign_percs',sep="")]] <- sig_sign_percs
      next
    }
    
    #row 1 = baseline vs aft_cue, 2 = baseline vs bfr_res, 3 = baseline vs aft_res, 4 = basline vs res_wind, 5 = bfr_res vs after_res, 6 = bfr_res vs res_wind
    p_val_array <- array(data=NA,dim=c(total_unit_num,6))
    mean_diff_array <- array(data=NA,dim=c(total_unit_num,6))
    for (unit_num in 1:total_unit_num){

      #TODO unhardcode time
      baseline <- rowMeans(all_cue_fr[unit_num,comb_inds,1:50])
    
      aft_cue <- rowMeans(all_cue_fr[unit_num,comb_inds,51:150])
      bfr_res <- rowMeans(all_res_fr[unit_num,comb_inds,1:50])
      aft_res <- rowMeans(all_res_fr[unit_num,comb_inds,51:150])
      res_wind <- rowMeans(all_res_fr[unit_num,comb_inds,81:130])
      
      temp <- calc_wsr(baseline,aft_cue)
      p_val_array[unit_num,1] <- temp[1]
      mean_diff_array[unit_num,1] <- temp[2]
      
      temp <- calc_wsr(baseline,bfr_res)
      p_val_array[unit_num,2] <- temp[1]
      mean_diff_array[unit_num,2] <- temp[2]
      
      temp <- calc_wsr(baseline,aft_res)
      p_val_array[unit_num,3] <- temp[1]
      mean_diff_array[unit_num,3] <- temp[2]
      
      temp <- calc_wsr(baseline,res_wind)
      p_val_array[unit_num,4] <- temp[1]
      mean_diff_array[unit_num,4] <- temp[2]
      
      temp <- calc_wsr(bfr_res,aft_res)
      p_val_array[unit_num,5] <- temp[1]
      mean_diff_array[unit_num,5] <- temp[2]
      
      temp <- calc_wsr(bfr_res,res_wind)
      p_val_array[unit_num,6] <- temp[1]
      mean_diff_array[unit_num,6] <- temp[2]
      
    }
    perc_sig <- c(length(which(p_val_array[,1] < 0.05)),length(which(p_val_array[,2] < 0.05)),length(which(p_val_array[,3] < 0.05)),length(which(p_val_array[,4] < 0.05)),length(which(p_val_array[,5] < 0.05)),length(which(p_val_array[,6] < 0.05))) / dim(p_val_array)[1]
    
    sig_signs_1 <- mean_diff_array[,1][p_val_array[,1] < 0.05 & !is.na(p_val_array[,1])]
    sig_signs_2 <- mean_diff_array[,2][p_val_array[,2] < 0.05 & !is.na(p_val_array[,2])]
    sig_signs_3 <- mean_diff_array[,3][p_val_array[,3] < 0.05 & !is.na(p_val_array[,3])]
    sig_signs_4 <- mean_diff_array[,4][p_val_array[,4] < 0.05 & !is.na(p_val_array[,4])]
    sig_signs_5 <- mean_diff_array[,5][p_val_array[,5] < 0.05 & !is.na(p_val_array[,5])]
    sig_signs_6 <- mean_diff_array[,6][p_val_array[,6] < 0.05 & !is.na(p_val_array[,6])]
    
    #[num sig units, num sig units w/ increase mean fr, num sig units w/ decrease fr, % inc, %dec]
    out_1 <- c(length(sig_signs_1),length(sig_signs_1[sig_signs_1 > 0]),length(sig_signs_1[sig_signs_1 < 0]),length(sig_signs_1[sig_signs_1 > 0])/length(sig_signs_1),length(sig_signs_1[sig_signs_1 < 0])/length(sig_signs_1))
    out_2 <- c(length(sig_signs_2),length(sig_signs_2[sig_signs_2 > 0]),length(sig_signs_2[sig_signs_2 < 0]),length(sig_signs_2[sig_signs_2 > 0])/length(sig_signs_2),length(sig_signs_2[sig_signs_2 < 0])/length(sig_signs_2))
    out_3 <- c(length(sig_signs_3),length(sig_signs_3[sig_signs_3 > 0]),length(sig_signs_3[sig_signs_3 < 0]),length(sig_signs_3[sig_signs_3 > 0])/length(sig_signs_3),length(sig_signs_3[sig_signs_3 < 0])/length(sig_signs_3))
    out_4 <- c(length(sig_signs_4),length(sig_signs_4[sig_signs_4 > 0]),length(sig_signs_4[sig_signs_4 < 0]),length(sig_signs_4[sig_signs_4 > 0])/length(sig_signs_4),length(sig_signs_4[sig_signs_4 < 0])/length(sig_signs_4))
    out_5 <- c(length(sig_signs_5),length(sig_signs_5[sig_signs_5 > 0]),length(sig_signs_5[sig_signs_5 < 0]),length(sig_signs_5[sig_signs_5 > 0])/length(sig_signs_5),length(sig_signs_5[sig_signs_5 < 0])/length(sig_signs_5))
    out_6 <- c(length(sig_signs_6),length(sig_signs_6[sig_signs_6 > 0]),length(sig_signs_6[sig_signs_6 < 0]),length(sig_signs_6[sig_signs_6 > 0])/length(sig_signs_6),length(sig_signs_6[sig_signs_6 < 0])/length(sig_signs_6))
    
    sig_sign_percs <- cbind(bl_ac = out_1,bl_br = out_2,bl_ar = out_3,bl_rw = out_4,br_ar = out_5,br_rw =out_6)
    
    out_p_list[[paste(i,'_p_vals',sep="")]] <- p_val_array
    out_perc_sig_list[[paste(i,'_p_vals',sep="")]] <- perc_sig
    out_mean_diff_array[[paste(i,'_mean_diffs',sep="")]] <- mean_diff_array
    out_sig_sign_percs[[paste(i,'_sig_sign_percs',sep="")]] <- sig_sign_percs
  }
  
  assign(paste(region_list[region_index],'_p_val_list',sep=""),out_p_list)
  assign(paste(region_list[region_index],'_perc_sig_list',sep=""),out_perc_sig_list)
  assign(paste(region_list[region_index],'_mean_diffs_list',sep=""),out_mean_diff_array)
  assign(paste(region_list[region_index],'_sig_sign_percs',sep=""),out_sig_sign_percs)
  
  #######################
  ###plot ###############
  ########################
  
  cat('plotting\n')
  window_names <- c('baseline vs \nafter cue','baseline vs \nbefore result','baseline vs \nafter result','baseline vs \nresult window','before result vs \nafter result','before result vs \nresult window')
  
  #reward
  png(paste(region_list[region_index],'_r_sig_diffs.png',sep=""),width=8,height=6,units="in",res=500)
  
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
  png(paste(region_list[region_index],'_p_sig_diffs.png',sep=""),width=8,height=6,units="in",res=500)
  
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
  png(paste(region_list[region_index],'_rx_outcome_sig_diffs.png',sep=""),width=8,height=6,units="in",res=500)
  
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
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Reward and Outcome Combination',y='Number of units')
  
  plot(plt)
  graphics.off()
  
  #px outcome
  png(paste(region_list[region_index],'_px_outcome_sig_diffs.png',sep=""),width=8,height=6,units="in",res=500)
  
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
  plt <- plt + theme_bw() + scale_fill_manual(values=c("lightcoral","royalblue")) + labs(title=paste("Region: ",region_list[region_index],'\nTotal units: ',total_unit_num,sep=""),x='Punishment and Outcome Combination',y='Number of units')
  
  plot(plt)
  graphics.off()
  
  #res outcome
  png(paste(region_list[region_index],'_res_outcome_sig_diffs.png',sep=""),width=8,height=6,units="in",res=500)
  
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
  png(paste(region_list[region_index],'_comb_sig_diffs.png',sep=""),width=8,height=6,units="in",res=500)
  
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
  png(paste(region_list[region_index],'_comb_outcome_sig_diffs.png',sep=""),width=8,height=6,units="in",res=500)
  
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
  plt <- plt + theme(axis.text.x = element_text(size=rel(0.8)))
  
  plot(plt)
  graphics.off()
  
  #reward outcome
  png(paste(region_list[region_index],'_r_outcome_sig_diffs.png',sep=""),width=8,height=6,units="in",res=500)
  
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
  plt <- plt + theme(axis.text.x = element_text(size=rel(0.8)))
  
  plot(plt)
  graphics.off()
  
  #punishment outcome
  png(paste(region_list[region_index],'_p_outcome_sig_diffs.png',sep=""),width=8,height=6,units="in",res=500)
  
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
  plt <- plt + theme(axis.text.x = element_text(size=rel(0.8)))
  
  plot(plt)
  graphics.off()
  
  #motivation
  png(paste(region_list[region_index],'_m_outcome_sig_diffs.png',sep=""),width=8,height=6,units="in",res=500)
  
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
  png(paste(region_list[region_index],'_v_outcome_sig_diffs.png',sep=""),width=8,height=6,units="in",res=500)
  
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
  
  
}

###############

####friedman test

#save vals

