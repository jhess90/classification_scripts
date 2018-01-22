library(openxlsx)
library(ggplot2)
library(reshape2)
#source("~/dropbox/mult_rp_files/r_test/multiplot.R")
#source("~/Dropbox/mult_rp_files/r_test/multiplot.R")
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
  condensed <- readin$return.dict[,,1]$condensed
  bin_size <- readin$return.dict[,,1]$params[,,1]$bin.size[,]
  rp_vals <- c(0,1)
  
  
  #TODO make for diff bfr and aft times
  #TODO unhardcode rollmean (time and rollmean val)
  old_time <- seq(from=-0.5,to=(1.0-bin_size/1000),by=bin_size/1000)
  time <- seq(from=-0.5+2*bin_size/1000,to=(1.0-3*bin_size/1000),by=bin_size/1000)
  
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
  
  r0_p0_s <- res1[which(res1 %in% r0_p0)]
  rx_p0_s <- res1[which(res1 %in% rx_p0)]
  r0_px_s <- res1[which(res1 %in% r0_px)]
  rx_px_s <- res1[which(res1 %in% rx_px)]
  r0_p0_f <- res0[which(res0 %in% r0_p0)]
  rx_p0_f <- res0[which(res0 %in% rx_p0)]
  r0_px_f <- res0[which(res0 %in% r0_px)]
  rx_px_f <- res0[which(res0 %in% rx_px)]
  

  for (unit_num in 1:dim(all_cue_fr)[1]){
    
    ## comb
    png(paste(region_list[region_index],"_comb_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    comb_cue_avgs <- data.frame(time=time,r0_p0=rollmean(colMeans(all_cue_fr[unit_num,r0_p0,]),5),rx_p0=rollmean(colMeans(all_cue_fr[unit_num,rx_p0,]),5),r0_px=rollmean(colMeans(all_cue_fr[unit_num,r0_px,]),5),rx_px=rollmean(colMeans(all_cue_fr[unit_num,rx_px,]),5))
    comb_res_avgs <- data.frame(time=time,r0_p0=rollmean(colMeans(all_res_fr[unit_num,r0_p0,]),5),rx_p0=rollmean(colMeans(all_res_fr[unit_num,rx_p0,]),5),r0_px=rollmean(colMeans(all_res_fr[unit_num,r0_px,]),5),rx_px=rollmean(colMeans(all_res_fr[unit_num,rx_px,]),5))
    
    comb_cue_avgs.m <- melt(comb_cue_avgs,id.vars="time",variable="comb")
    plt_cue <- ggplot(comb_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("black","forestgreen","firebrick","mediumblue")) + labs(title=paste("Combination: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0)
    
    comb_res_avgs.m <- melt(comb_res_avgs,id.vars="time",variable="comb")
    plt_res <- ggplot(comb_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("black","forestgreen","firebrick","mediumblue")) + labs(title="Result",y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0)
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    
    ## comb res
    png(paste(region_list[region_index],"_comb_res_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    if (length(r0_p0_f) == 1){r0_p0_f_fr <- rollmean(all_cue_fr[unit_num,r0_p0_f,],5)}else if(length(r0_p0_f) == 0){r0_p0_f_fr <- rep(0,146)}else{r0_p0_f_fr <- rollmean(colMeans(all_cue_fr[unit_num,r0_p0_f,]),5)}
    if (length(r0_px_f) == 1){r0_px_f_fr <- rollmean(all_cue_fr[unit_num,r0_px_f,],5)}else if(length(r0_px_f) == 0){r0_px_f_fr <- rep(0,146)}else{r0_px_f_fr <- rollmean(colMeans(all_cue_fr[unit_num,r0_px_f,]),5)}
    if (length(rx_p0_f) == 1){rx_p0_f_fr <- rollmean(all_cue_fr[unit_num,rx_p0_f,],5)}else if(length(rx_p0_f) == 0){rx_p0_f_fr <- rep(0,146)}else{rx_p0_f_fr <- rollmean(colMeans(all_cue_fr[unit_num,rx_p0_f,]),5)}
    if (length(rx_px_f) == 1){rx_px_f_fr <- rollmean(all_cue_fr[unit_num,rx_px_f,],5)}else if(length(rx_px_f) == 0){rx_px_f_fr <- rep(0,146)}else{rx_px_f_fr <- rollmean(colMeans(all_cue_fr[unit_num,rx_px_f,]),5)}
    
    if (length(r0_p0_s) == 1){r0_p0_s_fr <- rollmean(all_cue_fr[unit_num,r0_p0_s,],5)}else if(length(r0_p0_s) == 0){r0_p0_s_fr <- rep(0,146)}else{r0_p0_s_fr <- rollmean(colMeans(all_cue_fr[unit_num,r0_p0_s,]),5)}
    if (length(r0_px_s) == 1){r0_px_s_fr <- rollmean(all_cue_fr[unit_num,r0_px_s,],5)}else if(length(r0_px_s) == 0){r0_px_s_fr <- rep(0,146)}else{r0_px_s_fr <- rollmean(colMeans(all_cue_fr[unit_num,r0_px_s,]),5)}
    if (length(rx_p0_s) == 1){rx_p0_s_fr <- rollmean(all_cue_fr[unit_num,rx_p0_s,],5)}else if(length(rx_p0_s) == 0){rx_p0_s_fr <- rep(0,146)}else{rx_p0_s_fr <- rollmean(colMeans(all_cue_fr[unit_num,rx_p0_s,]),5)}
    if (length(rx_px_s) == 1){rx_px_s_fr <- rollmean(all_cue_fr[unit_num,rx_px_s,],5)}else if(length(rx_px_s) == 0){rx_px_s_fr <- rep(0,146)}else{rx_px_s_fr <- rollmean(colMeans(all_cue_fr[unit_num,rx_px_s,]),5)}
    
    comb_cue_avgs <- data.frame(time=time,r0_p0_s=r0_p0_s_fr,rx_p0_s=rx_p0_s_fr,r0_px_s=r0_px_s_fr,rx_px_s=rx_px_s_fr,r0_p0_f=r0_p0_f_fr,rx_p0_f=rx_p0_f_fr,r0_px_f=r0_px_f_fr,rx_px_f=rx_px_f_fr)
           
    if (length(r0_px_f) == 1){r0_px_f_fr <- rollmean(all_res_fr[unit_num,r0_px_f,],5)}else if(length(r0_px_f) == 0){r0_px_f_fr <- rep(0,146)}else{r0_px_f_fr <- rollmean(colMeans(all_res_fr[unit_num,r0_px_f,]),5)}
    if (length(rx_p0_f) == 1){rx_p0_f_fr <- rollmean(all_res_fr[unit_num,rx_p0_f,],5)}else if(length(rx_p0_f) == 0){rx_p0_f_fr <- rep(0,146)}else{rx_p0_f_fr <- rollmean(colMeans(all_res_fr[unit_num,rx_p0_f,]),5)}
    if (length(rx_px_f) == 1){rx_px_f_fr <- rollmean(all_res_fr[unit_num,rx_px_f,],5)}else if(length(rx_px_f) == 0){rx_px_f_fr <- rep(0,146)}else{rx_px_f_fr <- rollmean(colMeans(all_res_fr[unit_num,rx_px_f,]),5)}
    
    if (length(r0_p0_s) == 1){r0_p0_s_fr <- rollmean(all_res_fr[unit_num,r0_p0_s,],5)}else if(length(r0_p0_s) == 0){r0_p0_s_fr <- rep(0,146)}else{r0_p0_s_fr <- rollmean(colMeans(all_res_fr[unit_num,r0_p0_s,]),5)}
    if (length(r0_px_s) == 1){r0_px_s_fr <- rollmean(all_res_fr[unit_num,r0_px_s,],5)}else if(length(r0_px_s) == 0){r0_px_s_fr <- rep(0,146)}else{r0_px_s_fr <- rollmean(colMeans(all_res_fr[unit_num,r0_px_s,]),5)}
    if (length(rx_p0_s) == 1){rx_p0_s_fr <- rollmean(all_res_fr[unit_num,rx_p0_s,],5)}else if(length(rx_p0_s) == 0){rx_p0_s_fr <- rep(0,146)}else{rx_p0_s_fr <- rollmean(colMeans(all_res_fr[unit_num,rx_p0_s,]),5)}
    if (length(rx_px_s) == 1){rx_px_s_fr <- rollmean(all_res_fr[unit_num,rx_px_s,],5)}else if(length(rx_px_s) == 0){rx_px_s_fr <- rep(0,146)}else{rx_px_s_fr <- rollmean(colMeans(all_res_fr[unit_num,rx_px_s,]),5)}
    
    comb_res_avgs <- data.frame(time=time,r0_p0_s=r0_p0_s_fr,rx_p0_s=rx_p0_s_fr,r0_px_s=r0_px_s_fr,rx_px_s=rx_px_s_fr,r0_p0_f=r0_p0_f_fr,rx_p0_f=rx_p0_f_fr,r0_px_f=r0_px_f_fr,rx_px_f=rx_px_f_fr)
    
    comb_cue_avgs.m <- melt(comb_cue_avgs,id.vars="time",variable="comb")
    plt_cue <- ggplot(comb_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=0.5) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("black","forestgreen","firebrick","mediumblue","gray69","darkseagreen2","lightpink2","darkslategray1")) + labs(title=paste("Combination: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Combination") #+ geom_vline(xintercept=0) +  scale_linetype_manual(values=c("solid","solid","solid","solid","dashed","dashed","dashed","dashed"))

    
    comb_res_avgs.m <- melt(comb_res_avgs,id.vars="time",variable="comb")
    plt_res <- ggplot(comb_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=0.5) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("black","forestgreen","firebrick","mediumblue","gray69","darkseagreen2","lightpink2","darkslategray1")) + labs(title="Result",y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0)
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    ## result sf
    png(paste(region_list[region_index],"_r_sf_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    if (length(r0_s) == 1){r0_s_fr_cue <- rollmean(all_cue_fr[unit_num,r0_s,],5)}else if(length(r0_s) == 0){r0_s_fr_cue <- rep(0,146)}else{r0_s_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,r0_s,]),5)}
    if (length(r0_f) == 1){r0_f_fr_cue <- rollmean(all_cue_fr[unit_num,r0_f,],5)}else if(length(r0_f) == 0){r0_f_fr_cue <- rep(0,146)}else{r0_f_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,r0_f,]),5)}
    if (length(rx_s) == 1){rx_s_fr_cue <- rollmean(all_cue_fr[unit_num,rx_s,],5)}else if(length(rx_s) == 0){rx_s_fr_cue <- rep(0,146)}else{rx_s_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,rx_s,]),5)}
    if (length(rx_f) == 1){rx_f_fr_cue <- rollmean(all_cue_fr[unit_num,rx_f,],5)}else if(length(rx_f) == 0){rx_f_fr_cue <- rep(0,146)}else{rx_f_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,rx_f,]),5)}
    
    if (length(r0_s) == 1){r0_s_fr_res <- rollmean(all_res_fr[unit_num,r0_s,],5)}else if(length(r0_s) == 0){r0_s_fr_res <- rep(0,146)}else{r0_s_fr_res <- rollmean(colMeans(all_res_fr[unit_num,r0_s,]),5)}
    if (length(r0_f) == 1){r0_f_fr_res <- rollmean(all_res_fr[unit_num,r0_f,],5)}else if(length(r0_f) == 0){r0_f_fr_res <- rep(0,146)}else{r0_f_fr_res <- rollmean(colMeans(all_res_fr[unit_num,r0_f,]),5)}
    if (length(rx_s) == 1){rx_s_fr_res <- rollmean(all_res_fr[unit_num,rx_s,],5)}else if(length(rx_s) == 0){rx_s_fr_res <- rep(0,146)}else{rx_s_fr_res <- rollmean(colMeans(all_res_fr[unit_num,rx_s,]),5)}
    if (length(rx_f) == 1){rx_f_fr_res <- rollmean(all_res_fr[unit_num,rx_f,],5)}else if(length(rx_f) == 0){rx_f_fr_res <- rep(0,146)}else{rx_f_fr_res <- rollmean(colMeans(all_res_fr[unit_num,rx_f,]),5)}
    
    r_sf_cue_avgs <- data.frame(time=time,r0_s_cue=r0_s_fr_cue,r0_f_cue=r0_f_fr_cue,rx_s_cue=rx_s_fr_cue,rx_f_cue=rx_f_fr_cue)
    r_sf_res_avgs <- data.frame(time=time,r0_s_res=r0_s_fr_res,r0_f_res=r0_f_fr_res,rx_s_res=rx_s_fr_res,rx_f_res=rx_f_fr_res)
    
    #r_sf_cue_avgs <- data.frame(time=time,r0_s=rollmean(colMeans(all_cue_fr[unit_num,r0_s,]),5),r0_f=rollmean(colMeans(all_cue_fr[unit_num,r0_f,]),5),rx_s=rollmean(colMeans(all_cue_fr[unit_num,rx_s,]),5),rx_f=rollmean(colMeans(all_cue_fr[unit_num,rx_f,]),5))
    #r_sf_res_avgs <- data.frame(time=time,r0_s=rollmean(colMeans(all_res_fr[unit_num,r0_s,]),5),r0_f=rollmean(colMeans(all_res_fr[unit_num,r0_f,]),5),rx_s=rollmean(colMeans(all_res_fr[unit_num,rx_s,]),5),rx_f=rollmean(colMeans(all_res_fr[unit_num,rx_f,]),5))
    
    r_sf_cue_avgs.m <- melt(r_sf_cue_avgs,id.vars="time",variable="comb")
    plt_cue <- ggplot(r_sf_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("black","gray69","forestgreen","darkseagreen2")) + labs(title=paste("Reward combination: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0)
    
    r_sf_res_avgs.m <- melt(r_sf_res_avgs,id.vars="time",variable="comb")
    plt_res <- ggplot(r_sf_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("black","gray69","forestgreen","darkseagreen2")) + labs(title="Result",y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0)
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    ## punishment sf
    png(paste(region_list[region_index],"_p_sf_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    if (length(p0_s) == 1){p0_s_fr_cue <- rollmean(all_cue_fr[unit_num,p0_s,],5)}else if(length(p0_s) == 0){p0_s_fr_cue <- rep(0,146)}else{p0_s_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,p0_s,]),5)}
    if (length(p0_f) == 1){p0_f_fr_cue <- rollmean(all_cue_fr[unit_num,p0_f,],5)}else if(length(p0_f) == 0){p0_f_fr_cue <- rep(0,146)}else{p0_f_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,p0_f,]),5)}
    if (length(px_s) == 1){px_s_fr_cue <- rollmean(all_cue_fr[unit_num,px_s,],5)}else if(length(px_s) == 0){px_s_fr_cue <- rep(0,146)}else{px_s_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,px_s,]),5)}
    if (length(px_f) == 1){px_f_fr_cue <- rollmean(all_cue_fr[unit_num,px_f,],5)}else if(length(px_f) == 0){px_f_fr_cue <- rep(0,146)}else{px_f_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,px_f,]),5)}

    if (length(p0_s) == 1){p0_s_fr_res <- rollmean(all_res_fr[unit_num,p0_s,],5)}else if(length(p0_s) == 0){p0_s_fr_res <- rep(0,146)}else{p0_s_fr_res <- rollmean(colMeans(all_res_fr[unit_num,p0_s,]),5)}
    if (length(p0_f) == 1){p0_f_fr_res <- rollmean(all_res_fr[unit_num,p0_f,],5)}else if(length(p0_f) == 0){p0_f_fr_res <- rep(0,146)}else{p0_f_fr_res <- rollmean(colMeans(all_res_fr[unit_num,p0_f,]),5)}
    if (length(px_s) == 1){px_s_fr_res <- rollmean(all_res_fr[unit_num,px_s,],5)}else if(length(px_s) == 0){px_s_fr_res <- rep(0,146)}else{px_s_fr_res <- rollmean(colMeans(all_res_fr[unit_num,px_s,]),5)}
    if (length(px_f) == 1){px_f_fr_res <- rollmean(all_res_fr[unit_num,px_f,],5)}else if(length(px_f) == 0){px_f_fr_res <- rep(0,146)}else{px_f_fr_res <- rollmean(colMeans(all_res_fr[unit_num,px_f,]),5)}
    
    p_sf_cue_avgs <- data.frame(time=time,p0_s_cue=p0_s_fr_cue,p0_f_cue=p0_f_fr_cue,px_s_cue=px_s_fr_cue,px_f_cue=px_f_fr_cue)
    p_sf_res_avgs <- data.frame(time=time,p0_s_res=p0_s_fr_res,p0_f_res=p0_f_fr_res,px_s_res=px_s_fr_res,px_f_res=px_f_fr_res)
    
    #p_sf_cue_avgs <- data.frame(time=time,p0_s=rollmean(colMeans(all_cue_fr[unit_num,p0_s,]),5),p0_f=rollmean(colMeans(all_cue_fr[unit_num,p0_f,]),5),px_s=rollmean(colMeans(all_cue_fr[unit_num,px_s,]),5),px_f=rollmean(colMeans(all_cue_fr[unit_num,px_f,]),5))
    #p_sf_res_avgs <- data.frame(time=time,p0_s=rollmean(colMeans(all_res_fr[unit_num,p0_s,]),5),p0_f=rollmean(colMeans(all_res_fr[unit_num,p0_f,]),5),px_s=rollmean(colMeans(all_res_fr[unit_num,px_s,]),5),px_f=rollmean(colMeans(all_res_fr[unit_num,px_f,]),5))
    
    p_sf_cue_avgs.m <- melt(p_sf_cue_avgs,id.vars="time",variable="comb")
    plt_cue <- ggplot(p_sf_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("black","gray69","firebrick","lightpink2")) + labs(title=paste("Punishment combination: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0)
    
    p_sf_res_avgs.m <- melt(p_sf_res_avgs,id.vars="time",variable="comb")
    plt_res <- ggplot(p_sf_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("black","gray69","firebrick","lightpink2")) + labs(title="Result",y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0)
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    
  }
}

# save.image(file="rearranged_data.RData")
rm(list=ls())

