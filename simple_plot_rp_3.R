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

#file_list <- c('simple_output')
region_list <- c('M1','S1','PmD')







#########


for(region_index in 1:length(region_list)){
  cat("\nplotting region:",region_list[region_index])
  
  readin <- readMat(paste('simple_output_',region_list[region_index],'.mat',sep=""))
  
  all_cue_fr <- readin$return.dict[,,1]$all.cue.fr
  all_res_fr <- readin$return.dict[,,1]$all.res.fr
  condensed <- readin$return.dict[,,1]$condensed
  bin_size <- readin$return.dict[,,1]$params[,,1]$bin.size[,]
  rp_vals <- c(0,3)
  
  
  #TODO make for diff bfr and aft times
  #TODO unhardcode rollmean (time and rollmean val)
  old_time <- seq(from=-0.5,to=(1.0-bin_size/1000),by=bin_size/1000)
  time <- seq(from=-0.5+2*bin_size/1000,to=(1.0-3*bin_size/1000),by=bin_size/1000)
  
  r0 <- which(condensed[,4] == 0)
  r3 <- which(condensed[,4] == 3)
  
  p0 <- which(condensed[,5] == 0)
  p3 <- which(condensed[,5] == 3)
  
  v_3 <- which(condensed[,7] == -3)
  v0 <- which(condensed[,7] == 0)
  v3 <- which(condensed[,7] == 3)
  
  m0 <- which(condensed[,8] == 0)
  m3 <- which(condensed[,8] == 3)
  m6 <- which(condensed[,8] == 6)
  
  res0 <- which(condensed[,6] == 0)
  res1 <- which(condensed[,6] == 1)
  
  #r0_fail <- which(res0 %in% r0)
  #r3_fail <- which(res0 %in% r3)
  #r0_succ <- which(res1 %in% r0)
  #r3_succ <- which(res1 %in% r3)
  
  for (unit_num in 1:dim(all_cue_fr)[1]){
    
    ## reward
    png(paste(region_list[region_index],"_r_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    r_cue_avgs <- data.frame(time=time,r0=rollmean(colMeans(all_cue_fr[unit_num,r0,]),5),r3=rollmean(colMeans(all_cue_fr[unit_num,r3,]),5))
    r_res_avgs <- data.frame(time=time,r0=rollmean(colMeans(all_res_fr[unit_num,r0,]),5),r3=rollmean(colMeans(all_res_fr[unit_num,r3,]),5))
    
    r_cue_avgs.m <- melt(r_cue_avgs,id.vars="time",variable="r_level")
    plt_cue <- ggplot(r_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=r_level),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("goldenrod","darkgreen")) + labs(title=paste("Reward: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Reward Level") + geom_vline(xintercept=0)
    
    r_res_avgs.m <- melt(r_res_avgs,id.vars="time",variable="r_level")
    plt_res <- ggplot(r_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=r_level),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("goldenrod","darkgreen")) + labs(title="Result",y="z-score", x="Time(s)",colour="Reward Level") + geom_vline(xintercept=0)
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    ## punishment
    png(paste(region_list[region_index],"_p_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    p_cue_avgs <- data.frame(time=time,p0=rollmean(colMeans(all_cue_fr[unit_num,p0,]),5),p3=rollmean(colMeans(all_cue_fr[unit_num,p3,]),5))
    p_res_avgs <- data.frame(time=time,p0=rollmean(colMeans(all_res_fr[unit_num,p0,]),5),p3=rollmean(colMeans(all_res_fr[unit_num,p3,]),5))
    
    p_cue_avgs.m <- melt(p_cue_avgs,id.vars="time",variable="p_level")
    plt_cue <- ggplot(p_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=p_level),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("goldenrod","red2")) + labs(title=paste("Punishment: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Punishment Level") + geom_vline(xintercept=0)
    
    p_res_avgs.m <- melt(p_res_avgs,id.vars="time",variable="p_level")
    plt_res <- ggplot(p_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=p_level),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("goldenrod","red2")) + labs(title="Result",y="z-score", x="Time(s)",colour="Punishment Level") + geom_vline(xintercept=0)
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    ## value
    png(paste(region_list[region_index],"_v_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    v_cue_avgs <- data.frame(time=time,v_3=rollmean(colMeans(all_cue_fr[unit_num,v_3,]),5),v0=rollmean(colMeans(all_cue_fr[unit_num,v0,]),5),v3=rollmean(colMeans(all_cue_fr[unit_num,v3,]),5))
    v_res_avgs <- data.frame(time=time,v_3=rollmean(colMeans(all_res_fr[unit_num,v_3,]),5),v0=rollmean(colMeans(all_res_fr[unit_num,v0,]),5),v3=rollmean(colMeans(all_res_fr[unit_num,v3,]),5))
    
    v_cue_avgs.m <- melt(v_cue_avgs,id.vars="time",variable="v_level")
    plt_cue <- ggplot(v_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=v_level),size=1) + theme_classic()
    plt_cue <- plt_cue +  scale_colour_manual(values=c("saddlebrown","goldenrod","turquoise4")) + labs(title=paste("Value: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0)
    
    v_res_avgs.m <- melt(v_res_avgs,id.vars="time",variable="v_level")
    plt_res <- ggplot(v_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=v_level),size=1) + theme_classic()
    plt_res <- plt_res +  scale_colour_manual(values=c("saddlebrown","goldenrod","turquoise4")) + labs(title="Result",y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0)
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    ## motivation
    png(paste(region_list[region_index],"_m_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    m_cue_avgs <- data.frame(time=time,m0=rollmean(colMeans(all_cue_fr[unit_num,m0,]),5),m3=rollmean(colMeans(all_cue_fr[unit_num,m3,]),5),m6=rollmean(colMeans(all_cue_fr[unit_num,m6,]),5))
    m_res_avgs <- data.frame(time=time,m0=rollmean(colMeans(all_res_fr[unit_num,m0,]),5),m3=rollmean(colMeans(all_res_fr[unit_num,m3,]),5),m6=rollmean(colMeans(all_res_fr[unit_num,m6,]),5))
    
    m_cue_avgs.m <- melt(m_cue_avgs,id.vars="time",variable="m_level")
    plt_cue <- ggplot(m_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=m_level),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("darkslategray2","aquamarine4","darkslateblue")) + labs(title=paste("Motivation: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0)
    
    m_res_avgs.m <- melt(m_res_avgs,id.vars="time",variable="m_level")
    plt_res <- ggplot(m_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=m_level),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("darkslategray2","aquamarine4","darkslateblue")) + labs(title="Result",y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0)
    
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    ## result
    png(paste(region_list[region_index],"_res_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    res_cue_avgs <- data.frame(time=time,fail=rollmean(colMeans(all_cue_fr[unit_num,res0,]),5),succ=rollmean(colMeans(all_cue_fr[unit_num,res1,]),5))
    res_res_avgs <- data.frame(time=time,fail=rollmean(colMeans(all_res_fr[unit_num,res0,]),5),succ=rollmean(colMeans(all_res_fr[unit_num,res1,]),5))
    
    res_cue_avgs.m <- melt(res_cue_avgs,id.vars="time",variable="res_level")
    plt_cue <- ggplot(res_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=res_level),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("skyblue4","purple")) + labs(title=paste("Result: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Result") + geom_vline(xintercept=0)
    
    res_res_avgs.m <- melt(res_res_avgs,id.vars="time",variable="res_level")
    plt_res <- ggplot(res_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=res_level),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("skyblue4","purple")) + labs(title="Result",y="z-score", x="Time(s)",colour="Result") + geom_vline(xintercept=0)
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    
    
    
    #if(length(r0_fail)==1){r0_fail_avg <- rollmean(all_res_fr[unit_num,r0_fail,],5)
    #}else{r0_fail_avg <- rollmean(colMeans(all_res_fr[unit_num,r0_fail,]),5)}
    #if(length(r1_fail)==1){r1_fail_avg <- rollmean(all_res_fr[unit_num,r1_fail,],5)
    #}else{r1_fail_avg <- rollmean(colMeans(all_res_fr[unit_num,r1_fail,]),5)}
    #if(length(r2_fail)==1){r2_fail_avg <- rollmean(all_res_fr[unit_num,r2_fail,],5)
    #}else{r2_fail_avg <- rollmean(colMeans(all_res_fr[unit_num,r2_fail,]),5)}
    #if(length(r3_fail)==1){r3_fail_avg <- rollmean(all_res_fr[unit_num,r3_fail,],5)
    #}else{r3_fail_avg <- rollmean(colMeans(all_res_fr[unit_num,r3_fail,]),5)}
    
    #r_res_succ_avgs <- data.frame(time=time,r0_succ=rollmean(colMeans(all_res_fr[unit_num,r0_succ,]),5),r1_succ=rollmean(colMeans(all_res_fr[unit_num,r1_succ,]),5),r2_succ=rollmean(colMeans(all_res_fr[unit_num,r2_succ,]),5),r3_succ=rollmean(colMeans(all_res_fr[unit_num,r3_succ,]),5))
    #r_res_fail_avgs <- data.frame(time=time,r0_fail=r0_fail_avg,r1_fail=r1_fail_avg,r2_fail=r2_fail_avg,r3_fail=r3_fail_avg)
    
    #r_res_succ_avgs <- data.frame(time=time,r0=rollmean(colMeans(all_res_fr[unit_num,r0_succ,]),5),r1=rollmean(colMeans(all_res_fr[unit_num,r1_succ,]),5),r2=rollmean(colMeans(all_res_fr[unit_num,r2_succ,]),5),r3=rollmean(colMeans(all_res_fr[unit_num,r3_succ,]),5))
    #r_res_fail_avgs <- data.frame(time=time,r0=r0_fail_avg,r1=r1_fail_avg,r2=r2_fail_avg,r3=r3_fail_avg)
    
    #r_res_succ_avgs.m <- melt(r_res_succ_avgs,id.vars="time",variable="r_level")
    #r_res_fail_avgs.m <- melt(r_res_fail_avgs,id.vars="time",variable="r_level")
    
    #r_res_avgs.m <- melt(r_res_succ_avgs,r_res_fail_avgs,id.vars="time",variable="r_level",value=c("succ","fail"))
    #r_res_avgs.m <- merge(r_res_succ_avgs.m,r_res_fail_avgs.m)
    #plt_res <- ggplot(r_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=r_level),size=1)
    #plt_res <- plt_res + scale_color_brewer(palette="RdYlGn") +labs(title="Result",y="z-score", x="Time(s)",colour="Reward Level") + geom_vline(xintercept=0)
    #plot(plt_res)
    #multiplot(plt_cue,plt_res)
    
    
  }
}




# save.image(file="rearranged_data.RData")
#rm(list=ls())

