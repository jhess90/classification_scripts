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

saveAsPng <- T

region_list <- c('M1','S1','PmD')


#########


for(region_index in 1:length(region_list)){
  cat("\nplotting region:",region_list[region_index])
  
  readin <- readMat(paste('simple_output_',region_list[region_index],'.mat',sep=""))
  
  all_cue_fr <- readin$return.dict[,,1]$all.cue.fr
  all_res_fr <- readin$return.dict[,,1]$all.res.fr
  condensed_old <- readin$return.dict[,,1]$condensed
  bin_size <- readin$return.dict[,,1]$params[,,1]$bin.size[,]
  rp_vals <- c(0,1,2,3)
  
  readin <- readMat(paste('corr_output_',region_list[region_index],'.mat',sep=""))
  condensed <- readin$condensed
  
  condensed <- condensed[condensed[,1] %in% condensed_old[,1],]
  
  
  old_time <- seq(from=-0.5,to=(1.0-bin_size/1000),by=bin_size/1000)
  time <- seq(from=-0.5+2*bin_size/1000,to=(1.0-3*bin_size/1000),by=bin_size/1000)
  
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
  
  
  for (unit_num in 1:dim(all_cue_fr)[1]){
    
    ## catch bin both
    png(paste(region_list[region_index],"_catch_bin_both_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    if (length(catch_x) == 1){catch_x_fr_cue <- rollmean(all_cue_fr[unit_num,catch_x,],5)}else if(length(catch_x) == 0){catch_x_fr_cue <- rep(0,146)}else{catch_x_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,catch_x,]),5)}
    if (length(catchx) == 1){catchx_fr_cue <- rollmean(all_cue_fr[unit_num,catchx,],5)}else if(length(catchx) == 0){catchx_fr_cue <- rep(0,146)}else{catchx_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,catchx,]),5)}
    if (length(rx_s) == 1){rx_s_fr_cue <- rollmean(all_cue_fr[unit_num,rx_s,],5)}else if(length(rx_s) == 0){rx_s_fr_cue <- rep(0,146)}else{rx_s_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,rx_s,]),5)}
    if (length(px_f) == 1){px_f_fr_cue <- rollmean(all_cue_fr[unit_num,px_f,],5)}else if(length(px_f) == 0){px_f_fr_cue <- rep(0,146)}else{px_f_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,px_f,]),5)}
    
    if (length(catch_x) == 1){catch_x_fr_res <- rollmean(all_res_fr[unit_num,catch_x,],5)}else if(length(catch_x) == 0){catch_x_fr_res <- rep(0,146)}else{catch_x_fr_res <- rollmean(colMeans(all_res_fr[unit_num,catch_x,]),5)}
    if (length(catchx) == 1){catchx_fr_res <- rollmean(all_res_fr[unit_num,catchx,],5)}else if(length(catchx) == 0){catchx_fr_res <- rep(0,146)}else{catchx_fr_res <- rollmean(colMeans(all_res_fr[unit_num,catchx,]),5)}
    if (length(rx_s) == 1){rx_s_fr_res <- rollmean(all_res_fr[unit_num,rx_s,],5)}else if(length(rx_s) == 0){rx_s_fr_res <- rep(0,146)}else{rx_s_fr_res <- rollmean(colMeans(all_res_fr[unit_num,rx_s,]),5)}
    if (length(px_f) == 1){px_f_fr_res <- rollmean(all_res_fr[unit_num,px_f,],5)}else if(length(px_f) == 0){px_f_fr_res <- rep(0,146)}else{px_f_fr_res <- rollmean(colMeans(all_res_fr[unit_num,px_f,]),5)}
    
    catch_bin_both_cue_avgs <- data.frame(time=time,catch_x_cue=catch_x_fr_cue,catchx_cue=catchx_fr_cue,rx_s_cue=rx_s_fr_cue,px_f_cue=px_f_fr_cue)
    catch_bin_both_res_avgs <- data.frame(time=time,catch_x_res=catch_x_fr_res,catchx_res=catchx_fr_res,rx_s_res=rx_s_fr_res,px_f_res=px_f_fr_res)
    
    catch_bin_both_cue_avgs.m <- melt(catch_bin_both_cue_avgs,id.vars="time",variable="comb")
    plt_cue <- ggplot(catch_bin_both_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb,linetype=comb),show.legend=F,size=0.5) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("lightpink2","darkseagreen3","forestgreen","firebrick")) + labs(title=paste("Catch combination: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0) + scale_linetype_manual(values=c("longdash","longdash","solid","solid"))
    
    catch_bin_both_res_avgs.m <- melt(catch_bin_both_res_avgs,id.vars="time",variable="comb")
    plt_res <- ggplot(catch_bin_both_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb,linetype=comb),show.legend=F,size=0.5) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("lightpink2","darkseagreen3","forestgreen","firebrick")) + labs(title="Result",y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0) + scale_linetype_manual(values=c("longdash","longdash","solid","solid"))
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    ## catch bin reward
    png(paste(region_list[region_index],"_catch_bin_r_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    if (length(catchx) == 1){catchx_fr_cue <- rollmean(all_cue_fr[unit_num,catchx,],5)}else if(length(catchx) == 0){catchx_fr_cue <- rep(0,146)}else{catchx_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,catchx,]),5)}
    if (length(rx_s) == 1){rx_s_fr_cue <- rollmean(all_cue_fr[unit_num,rx_s,],5)}else if(length(rx_s) == 0){rx_s_fr_cue <- rep(0,146)}else{rx_s_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,rx_s,]),5)}
    if (length(r0_s) == 1){r0_s_fr_cue <- rollmean(all_cue_fr[unit_num,r0_s,],5)}else if(length(r0_s) == 0){r0_s_fr_cue <- rep(0,146)}else{r0_s_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,r0_s,]),5)}
    
    if (length(catchx) == 1){catchx_fr_res <- rollmean(all_res_fr[unit_num,catchx,],5)}else if(length(catchx) == 0){catchx_fr_res <- rep(0,146)}else{catchx_fr_res <- rollmean(colMeans(all_res_fr[unit_num,catchx,]),5)}
    if (length(rx_s) == 1){rx_s_fr_res <- rollmean(all_res_fr[unit_num,rx_s,],5)}else if(length(rx_s) == 0){rx_s_fr_res <- rep(0,146)}else{rx_s_fr_res <- rollmean(colMeans(all_res_fr[unit_num,rx_s,]),5)}
    if (length(r0_s) == 1){r0_s_fr_res <- rollmean(all_res_fr[unit_num,r0_s,],5)}else if(length(r0_s) == 0){r0_s_fr_res <- rep(0,146)}else{r0_s_fr_res <- rollmean(colMeans(all_res_fr[unit_num,r0_s,]),5)}
    
    catch_bin_r_cue_avgs <- data.frame(time=time,catchx_cue=catchx_fr_cue,rx_s_cue=rx_s_fr_cue,r0_s_cue=r0_s_fr_cue)
    catch_bin_r_res_avgs <- data.frame(time=time,catchx_res=catchx_fr_res,rx_s_res=rx_s_fr_res,r0_s_res=r0_s_fr_res)
    
    catch_bin_r_cue_avgs.m <- melt(catch_bin_r_cue_avgs,id.vars="time",variable="comb")
    catch_bin_r_res_avgs.m <- melt(catch_bin_r_res_avgs,id.vars="time",variable="comb")
    
    plt_cue <- ggplot(catch_bin_r_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb,linetype=comb),show.legend=F,size=0.5) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("darkseagreen3","forestgreen","goldenrod")) + labs(title=paste("Catch reward: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0) + scale_linetype_manual(values=c("longdash","solid","solid"))
    
    plt_res <- ggplot(catch_bin_r_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb,linetype=comb),show.legend=F,size=0.5) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("darkseagreen3","forestgreen","goldenrod")) + labs(title="Result",y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0) + scale_linetype_manual(values=c("longdash","solid","solid"))
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    
    ## catch bin punishment
    png(paste(region_list[region_index],"_catch_bin_p_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    if (length(catch_x) == 1){catch_x_fr_cue <- rollmean(all_cue_fr[unit_num,catch_x,],5)}else if(length(catch_x) == 0){catch_x_fr_cue <- rep(0,146)}else{catch_x_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,catch_x,]),5)}
    if (length(px_f) == 1){px_f_fr_cue <- rollmean(all_cue_fr[unit_num,px_f,],5)}else if(length(px_f) == 0){px_f_fr_cue <- rep(0,146)}else{px_f_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,px_f,]),5)}
    if (length(p0_f) == 1){p0_f_fr_cue <- rollmean(all_cue_fr[unit_num,p0_f,],5)}else if(length(p0_f) == 0){p0_f_fr_cue <- rep(0,146)}else{p0_f_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,p0_f,]),5)}
    
    if (length(catch_x) == 1){catch_x_fr_res <- rollmean(all_res_fr[unit_num,catch_x,],5)}else if(length(catch_x) == 0){catch_x_fr_res <- rep(0,146)}else{catch_x_fr_res <- rollmean(colMeans(all_res_fr[unit_num,catch_x,]),5)}
    if (length(px_f) == 1){px_f_fr_res <- rollmean(all_res_fr[unit_num,px_f,],5)}else if(length(px_f) == 0){px_f_fr_res <- rep(0,146)}else{px_f_fr_res <- rollmean(colMeans(all_res_fr[unit_num,px_f,]),5)}
    if (length(p0_f) == 1){p0_f_fr_res <- rollmean(all_res_fr[unit_num,p0_f,],5)}else if(length(p0_f) == 0){p0_f_fr_res <- rep(0,146)}else{p0_f_fr_res <- rollmean(colMeans(all_res_fr[unit_num,p0_f,]),5)}
    
    catch_bin_p_cue_avgs <- data.frame(time=time,catch_x_cue=catch_x_fr_cue,px_f_cue=px_f_fr_cue,p0_f_cue=p0_f_fr_cue)
    catch_bin_p_res_avgs <- data.frame(time=time,catch_x_res=catch_x_fr_res,px_f_res=px_f_fr_res,p0_f_res=p0_f_fr_res)
    
    catch_bin_p_cue_avgs.m <- melt(catch_bin_p_cue_avgs,id.vars="time",variable="comb")
    catch_bin_p_res_avgs.m <- melt(catch_bin_p_res_avgs,id.vars="time",variable="comb")
    
    plt_cue <- ggplot(catch_bin_p_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb,linetype=comb),show.legend=F,size=0.5) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("lightpink2","firebrick","goldenrod")) + labs(title=paste("Catch punishment: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0) + scale_linetype_manual(values=c("longdash","solid","solid"))
    
    plt_res <- ggplot(catch_bin_p_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb,linetype=comb),show.legend=F,size=0.5) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("lightpink2","firebrick","goldenrod")) + labs(title="Result",y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0) + scale_linetype_manual(values=c("longdash","solid","solid"))
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    

    
  }
}


# save.image(file="rearranged_data.RData")
rm(list=ls())

