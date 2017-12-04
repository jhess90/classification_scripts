library(openxlsx)
library(ggplot2)
library(reshape2)
#source("~/dropbox/mult_rp_files/r_test/multiplot.R")
#source("~/Dropbox/mult_rp_files/r_test/multiplot.R")
source("~/documents/lab/workspace/Classification_scripts/multiplot.R")
library(zoo)
library(gplots)
library(RColorBrewer)
library(abind)
library(gridGraphics)
library(grid)
library(gridExtra)
library(R.matlab)

#################
#### Params #####
#################

units_to_plot <- 3
region_list <- c('M1','S1','PmD')


for (region_index in 1:length(region_list)){
  
  #corr hists
  readin <- readMat(paste('corr_output_',region_list[region_index],'.mat',sep=""))
  corr_output <- readin$corr.output
  corr_output_df <- data.frame(r_corr=corr_output[,1],p_corr=corr_output[,2],v_corr=corr_output[,3],m_corr=corr_output[,4],res_corr=corr_output[,5])
  
  r_corr <- ggplot(corr_output_df,aes(x=corr_output_df$r_corr))+geom_histogram(position='identity',binwidth=0.05)
  r_corr <- r_corr + labs(title="Reward Correlation",y="Number of Units",x="Correlation Coefficient")
  
  p_corr <- ggplot(corr_output_df,aes(x=corr_output_df$p_corr))+geom_histogram(position='identity',binwidth=0.05)
  p_corr <- p_corr + labs(title="Punishment Correlation",y="Number of Units",x="Correlation Coefficient")
  
  v_corr <- ggplot(corr_output_df,aes(x=corr_output_df$v_corr))+geom_histogram(position='identity',binwidth=0.05)
  v_corr <- v_corr + labs(title="Value Correlation",y="Number of Units",x="Correlation Coefficient")
  
  m_corr <- ggplot(corr_output_df,aes(x=corr_output_df$m_corr))+geom_histogram(position='identity',binwidth=0.05)
  m_corr <- m_corr + labs(title="Motivation Correlation",y="Number of Units",x="Correlation Coefficient")
  
  res_corr <- ggplot(corr_output_df,aes(x=corr_output_df$res_corr))+geom_histogram(position='identity',binwidth=0.05)
  res_corr <- res_corr + labs(title="Result Correlation",y="Number of Units",x="Correlation Coefficient")
  
  png(paste('corr_hist_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  multiplot(r_corr,p_corr,v_corr,m_corr,res_corr,cols=2)
  graphics.off()
  
  
  ##################
  
  #
  readin <- readMat(paste('r_avgs_',region_list[region_index],'.mat',sep=""))
  r_avgs <- lapply(readin$r.avgs,'[[',1)
  
  all_avgs <- lapply(r_avgs,mean)
  all_stds <- lapply(r_avgs,sd)
  
  r_values <- c(0,1,2,3)
  top_avg_cue <- c(all_avgs[[1]],all_avgs[[3]],all_avgs[[5]],all_avgs[[7]])
  btm_avg_cue <- c(all_avgs[[9]],all_avgs[[11]],all_avgs[[13]],all_avgs[[15]])
  top_avg_res <- c(all_avgs[[2]],all_avgs[[4]],all_avgs[[6]],all_avgs[[8]])
  btm_avg_res <- c(all_avgs[[10]],all_avgs[[12]],all_avgs[[14]],all_avgs[[16]])
  
  r_avgs <- data.frame(r_values=r_values,top_cue=top_avg_cue,btm_cue=btm_avg_cue,top_res=top_avg_res,btm_res=btm_avg_res)
  
  top_std_cue <- c(all_stds[[1]],all_stds[[3]],all_stds[[5]],all_stds[[7]])
  btm_std_cue <- c(all_stds[[9]],all_stds[[11]],all_stds[[13]],all_stds[[15]])
  top_std_res <- c(all_stds[[2]],all_stds[[4]],all_stds[[6]],all_stds[[8]])
  btm_std_res <- c(all_stds[[10]],all_stds[[12]],all_stds[[14]],all_stds[[16]])
  
  r_stds <- data.frame(r_values=r_values,top_cue=top_std_cue,btm_cue=btm_std_cue,top_res=top_std_res,btm_res=btm_std_res)
  
  avg_melt <- melt(r_avgs,id="r_values",variable.name='type',value.name='avg')
  std_melt <- melt(r_stds,id="r_values",variable.name='type',value.name='std')
  test <- merge(std_melt,avg_melt,row.names='r_values')
  
  png(paste('corr_tb_r_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  
  plt <- ggplot(test,aes(x=r_values,y=avg,color=type)) #+ geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1,position=pd) 
  plt <- plt +  geom_line(aes(linetype=type)) + geom_point() + scale_linetype_manual(name="Unit Key",values=c("top_cue"="solid","btm_cue"="dotted","top_res"="solid","btm_res"="dotted")) + scale_colour_manual(name="Unit Key",values=c("top_cue"="maroon","btm_cue"="maroon","top_res"="slateblue","btm_res"="slateblue"))
  plt <- plt + labs(title="Reward",x="Reward",y="Firing rate") 
  plot(plt)
  graphics.off()

  #
  readin <- readMat(paste('p_avgs_',region_list[region_index],'.mat',sep=""))
  p_avgs <- lapply(readin$p.avgs,'[[',1)
  
  all_avgs <- lapply(p_avgs,mean)
  all_stds <- lapply(p_avgs,sd)
  
  p_values <- c(0,1,2,3)
  top_avg_cue <- c(all_avgs[[1]],all_avgs[[3]],all_avgs[[5]],all_avgs[[7]])
  btm_avg_cue <- c(all_avgs[[9]],all_avgs[[11]],all_avgs[[13]],all_avgs[[15]])
  top_avg_res <- c(all_avgs[[2]],all_avgs[[4]],all_avgs[[6]],all_avgs[[8]])
  btm_avg_res <- c(all_avgs[[10]],all_avgs[[12]],all_avgs[[14]],all_avgs[[16]])

  p_avgs <- data.frame(p_values=p_values,top_cue=top_avg_cue,btm_cue=btm_avg_cue,top_res=top_avg_res,btm_res=btm_avg_res)

  top_std_cue <- c(all_stds[[1]],all_stds[[3]],all_stds[[5]],all_stds[[7]])
  btm_std_cue <- c(all_stds[[9]],all_stds[[11]],all_stds[[13]],all_stds[[15]])
  top_std_res <- c(all_stds[[2]],all_stds[[4]],all_stds[[6]],all_stds[[8]])
  btm_std_res <- c(all_stds[[10]],all_stds[[12]],all_stds[[14]],all_stds[[16]])
  
  p_stds <- data.frame(p_values=p_values,top_cue=top_std_cue,btm_cue=btm_std_cue,top_res=top_std_res,btm_res=btm_std_res)
  
  avg_melt <- melt(p_avgs,id="p_values",variable.name='type',value.name='avg')
  std_melt <- melt(p_stds,id="p_values",variable.name='type',value.name='std')
  test <- merge(std_melt,avg_melt,row.names='p_values')
  
  png(paste('corr_tb_p_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  
  plt <- ggplot(test,aes(x=p_values,y=avg,color=type)) #+ geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1,position=pd) 
  plt <- plt +  geom_line(aes(linetype=type)) + geom_point() + scale_linetype_manual(name="Unit Key",values=c("top_cue"="solid","btm_cue"="dotted","top_res"="solid","btm_res"="dotted")) + scale_colour_manual(name="Unit Key",values=c("top_cue"="maroon","btm_cue"="maroon","top_res"="slateblue","btm_res"="slateblue"))
  plt <- plt + labs(title="Punishment",x="Punishment",y="Firing rate") 
  plot(plt)
  graphics.off()

  #
  readin <- readMat(paste('res_avgs_',region_list[region_index],'.mat',sep=""))
  res_avgs <- lapply(readin$res.avgs,'[[',1)
  
  all_avgs <- lapply(res_avgs,mean)
  all_stds <- lapply(res_avgs,sd)
  
  res_values <- c(0,1)
  top_avg_cue <- c(all_avgs[[1]],all_avgs[[3]])
  btm_avg_cue <- c(all_avgs[[5]],all_avgs[[7]])
  top_avg_res <- c(all_avgs[[2]],all_avgs[[4]])
  btm_avg_res <- c(all_avgs[[6]],all_avgs[[8]])
  
  res_avgs <- data.frame(res_values=res_values,top_cue=top_avg_cue,btm_cue=btm_avg_cue,top_res=top_avg_res,btm_res=btm_avg_res)
  
  top_std_cue <- c(all_stds[[1]],all_stds[[3]])
  btm_std_cue <- c(all_stds[[5]],all_stds[[7]])
  top_std_res <- c(all_stds[[2]],all_stds[[4]])
  btm_std_res <- c(all_stds[[6]],all_stds[[8]])
  
  res_stds <- data.frame(res_values=res_values,top_cue=top_std_cue,btm_cue=btm_std_cue,top_res=top_std_res,btm_res=btm_std_res)
  
  avg_melt <- melt(res_avgs,id="res_values",variable.name='type',value.name='avg')
  std_melt <- melt(res_stds,id="res_values",variable.name='type',value.name='std')
  test <- merge(std_melt,avg_melt,row.names='res_values')
  
  png(paste('corr_tb_res_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  
  plt <- ggplot(test,aes(x=res_values,y=avg,color=type)) #+ geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1,position=pd) 
  plt <- plt +  geom_line(aes(linetype=type)) + geom_point() + scale_linetype_manual(name="Unit Key",values=c("top_cue"="solid","btm_cue"="dotted","top_res"="solid","btm_res"="dotted")) + scale_colour_manual(name="Unit Key",values=c("top_cue"="maroon","btm_cue"="maroon","top_res"="slateblue","btm_res"="slateblue"))
  plt <- plt + labs(title="Result",x="0: Failure, 1: Success",y="Firing rate") 
  plot(plt)
  graphics.off()
  
  #
  readin <- readMat(paste('v_avgs_',region_list[region_index],'.mat',sep=""))
  v_avgs <- lapply(readin$v.avgs,'[[',1)
  
  all_avgs <- lapply(v_avgs,mean)
  all_stds <- lapply(v_avgs,sd)
    
  v_values <- c(-3,-2,-1,0,1,2,3)
  top_avg_cue <- c(all_avgs[[1]],all_avgs[[3]],all_avgs[[5]],all_avgs[[7]],all_avgs[[9]],all_avgs[[11]],all_avgs[[13]])
  btm_avg_cue <- c(all_avgs[[15]],all_avgs[[17]],all_avgs[[19]],all_avgs[[21]],all_avgs[[23]],all_avgs[[25]],all_avgs[[27]])
  top_avg_res <- c(all_avgs[[2]],all_avgs[[4]],all_avgs[[6]],all_avgs[[8]],all_avgs[[10]],all_avgs[[12]],all_avgs[[14]])
  btm_avg_res <- c(all_avgs[[16]],all_avgs[[18]],all_avgs[[20]],all_avgs[[22]],all_avgs[[24]],all_avgs[[26]],all_avgs[[28]])
  
  v_avgs <- data.frame(v_values=v_values,top_cue=top_avg_cue,btm_cue=btm_avg_cue,top_res=top_avg_res,btm_res=btm_avg_res)
  #rownames(v_avgs) <- v_values
  
  top_std_cue <- c(all_stds[[1]],all_stds[[3]],all_stds[[5]],all_stds[[7]],all_stds[[9]],all_stds[[11]],all_stds[[13]])
  btm_std_cue <- c(all_stds[[15]],all_stds[[17]],all_stds[[19]],all_stds[[21]],all_stds[[23]],all_stds[[25]],all_stds[[27]])
  top_std_res <- c(all_stds[[2]],all_stds[[4]],all_stds[[6]],all_stds[[8]],all_stds[[10]],all_stds[[12]],all_stds[[14]])
  btm_std_res <- c(all_stds[[16]],all_stds[[18]],all_stds[[20]],all_stds[[22]],all_stds[[24]],all_stds[[26]],all_stds[[28]])
  
  v_stds <- data.frame(v_values=v_values,top_cue=top_std_cue,btm_cue=btm_std_cue,top_res=top_std_res,btm_res=btm_std_res)

  #TODO SE instead of std?
  avg_melt <- melt(v_avgs,id="v_values",variable.name='type',value.name='avg')
  std_melt <- melt(v_stds,id="v_values",variable.name='type',value.name='std')
  test <- merge(std_melt,avg_melt,row.names='v_values')
  
  png(paste('corr_tb_v_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)

  plt <- ggplot(test,aes(x=v_values,y=avg,color=type)) #+ geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1,position=pd) 
  plt <- plt +  geom_line(aes(linetype=type)) + geom_point() + scale_linetype_manual(name="Unit Key",values=c("top_cue"="solid","btm_cue"="dotted","top_res"="solid","btm_res"="dotted")) + scale_colour_manual(name="Unit Key",values=c("top_cue"="maroon","btm_cue"="maroon","top_res"="slateblue","btm_res"="slateblue"))
  plt <- plt + labs(title="Value",x="Value",y="Firing rate") 
  plot(plt)
  graphics.off()
  
  #
  readin <- readMat(paste('m_avgs_',region_list[region_index],'.mat',sep=""))
  m_avgs <- lapply(readin$m.avgs,'[[',1)
  
  all_avgs <- lapply(m_avgs,mean)
  all_stds <- lapply(m_avgs,sd)
  
  v_values <- c(0,1,2,3,4,5,6)
  top_avg_cue <- c(all_avgs[[1]],all_avgs[[3]],all_avgs[[5]],all_avgs[[7]],all_avgs[[9]],all_avgs[[11]],all_avgs[[13]])
  btm_avg_cue <- c(all_avgs[[15]],all_avgs[[17]],all_avgs[[19]],all_avgs[[21]],all_avgs[[23]],all_avgs[[25]],all_avgs[[27]])
  top_avg_res <- c(all_avgs[[2]],all_avgs[[4]],all_avgs[[6]],all_avgs[[8]],all_avgs[[10]],all_avgs[[12]],all_avgs[[14]])
  btm_avg_res <- c(all_avgs[[16]],all_avgs[[18]],all_avgs[[20]],all_avgs[[22]],all_avgs[[24]],all_avgs[[26]],all_avgs[[28]])
  
  m_avgs <- data.frame(m_values=m_values,top_cue=top_avg_cue,btm_cue=btm_avg_cue,top_res=top_avg_res,btm_res=btm_avg_res)
  
  top_std_cue <- c(all_stds[[1]],all_stds[[3]],all_stds[[5]],all_stds[[7]],all_stds[[9]],all_stds[[11]],all_stds[[13]])
  btm_std_cue <- c(all_stds[[15]],all_stds[[17]],all_stds[[19]],all_stds[[21]],all_stds[[23]],all_stds[[25]],all_stds[[27]])
  top_std_res <- c(all_stds[[2]],all_stds[[4]],all_stds[[6]],all_stds[[8]],all_stds[[10]],all_stds[[12]],all_stds[[14]])
  btm_std_res <- c(all_stds[[16]],all_stds[[18]],all_stds[[20]],all_stds[[22]],all_stds[[24]],all_stds[[26]],all_stds[[28]])
  
  m_stds <- data.frame(m_values=v_values,top_cue=top_std_cue,btm_cue=btm_std_cue,top_res=top_std_res,btm_res=btm_std_res)
  
  avg_melt <- melt(m_avgs,id="m_values",variable.name='type',value.name='avg')
  std_melt <- melt(m_stds,id="m_values",variable.name='type',value.name='std')
  test <- merge(std_melt,avg_melt,row.names='m_values')
  
  png(paste('corr_tb_m_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  
  plt <- ggplot(test,aes(x=m_values,y=avg,color=type)) #+ geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1,position=pd) 
  plt <- plt +  geom_line(aes(linetype=type)) + geom_point() + scale_linetype_manual(name="Unit Key",values=c("top_cue"="solid","btm_cue"="dotted","top_res"="solid","btm_res"="dotted")) + scale_colour_manual(name="Unit Key",values=c("top_cue"="maroon","btm_cue"="maroon","top_res"="slateblue","btm_res"="slateblue"))
  plt <- plt + labs(title="Motivation",x="Motivation",y="Firing rate") 
  plot(plt)
  graphics.off()
  
  
  
  ##############
  
  readin <- readMat(paste('fr_all_dict_',region_list[region_index],'.mat',sep=""))
  cue_avg <- readin$fr.all.dict[,,1]$cue.avg
  res_avg <- readin$fr.all.dict[,,1]$res.avg
  cue_peaks <- readin$fr.all.dict[,,1]$cue.peaks
  res_peaks <- readin$fr.all.dict[,,1]$res.peaks
  
  readin <- readMat(paste('order_dict_',region_list[region_index],'.mat',sep=""))
  r_corr_order <- readin$order.dict[,,1]$r.corr.order
  p_corr_order <- readin$order.dict[,,1]$p.corr.order
  val_corr_order <- readin$order.dict[,,1]$val.corr.order
  mtv_corr_order <- readin$order.dict[,,1]$mtv.corr.order
  res_corr_order <- readin$order.dict[,,1]$res.corr.order
  condensed <- readin$order.dict[,,1]$condensed
  
  #plot top units
  #i-1 b/ python indexing
  for (i in 1:units_to_plot){
    r_unit_ind <- which((i-1)==r_corr_order)
    p_unit_ind <- which((i-1)==p_corr_order)
    val_unit_ind <- which((i-1)==val_corr_order)
    mtv_unit_ind <- which((i-1)==mtv_corr_order)
    res_unit_ind <- which((i-1)==res_corr_order)
    
    #cue_avg_frs <- cue_avg[unit_ind,]
    #res_avg_frs <- res_avg[unit_ind,]
    #cue_peak_frs <- cue_peaks[unit_ind,]
    #res_peak_frs <- res_peaks[unit_ind,]
    
    r0_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==0)]
    r1_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==1)]
    r2_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==2)]
    r3_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==3)]
    r0_res <- res_avg[r_unit_ind,][which(condensed[,4]==0)]
    r1_res <- res_avg[r_unit_ind,][which(condensed[,4]==1)]
    r2_res <- res_avg[r_unit_ind,][which(condensed[,4]==2)]
    r3_res <- res_avg[r_unit_ind,][which(condensed[,4]==3)]
    
    r0_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==0) & which(condensed[,6]==1)]
    r1_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==1) & which(condensed[,6]==1)]
    r2_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==2) & which(condensed[,6]==1)]
    r3_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==3) & which(condensed[,6]==1)]
    r0_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==0) & which(condensed[,6]==0)]
    r1_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==1) & which(condensed[,6]==0)]
    r2_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==2) & which(condensed[,6]==0)]
    r3_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==3) & which(condensed[,6]==0)]
        
    p0_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==0)]
    p1_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==1)]
    p2_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==2)]
    p3_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==3)]
    p0_res <- res_avg[p_unit_ind,][which(condensed[,5]==0)]
    p1_res <- res_avg[p_unit_ind,][which(condensed[,5]==1)]
    p2_res <- res_avg[p_unit_ind,][which(condensed[,5]==2)]
    p3_res <- res_avg[p_unit_ind,][which(condensed[,5]==3)]
    
    res0_cue <- cue_avg[res_unit_ind,][which(condensed[,6]==0)]
    res1_cue <- cue_avg[res_unit_ind,][which(condensed[,6]==1)]
    res0_res <- res_avg[res_unit_ind,][which(condensed[,6]==0)]
    res1_res <- res_avg[res_unit_ind,][which(condensed[,6]==1)]
    
    v_3_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==-3)]
    v_2_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==-2)]
    v_1_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==-1)]
    v0_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==0)]
    v1_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==1)]
    v2_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==2)]
    v3_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==3)]
    v_3_res <- res_avg[val_unit_ind,][which(condensed[,7]==-3)]
    v_2_res <- res_avg[val_unit_ind,][which(condensed[,7]==-2)]
    v_1_res <- res_avg[val_unit_ind,][which(condensed[,7]==-1)]
    v0_res <- res_avg[val_unit_ind,][which(condensed[,7]==0)]
    v1_res <- res_avg[val_unit_ind,][which(condensed[,7]==1)]
    v2_res <- res_avg[val_unit_ind,][which(condensed[,7]==2)]
    v3_res <- res_avg[val_unit_ind,][which(condensed[,7]==3)]    
    
    m0_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==0)]
    m1_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==1)]
    m2_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==2)]
    m3_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==3)]
    m4_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==4)]
    m5_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==5)]
    m6_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==6)]
    m0_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==0)]
    m1_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==1)]
    m2_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==2)]
    m3_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==3)]
    m4_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==4)]
    m5_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==5)]
    m6_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==6)] 
    
    ##########
    png(paste('corr_r_t',i,'_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
    
    ravgs <- data.frame(r_values=c(0,1,2,3),r_cue = c(mean(r0_cue),mean(r1_cue),mean(r2_cue),mean(r3_cue)),r_res = c(mean(r0_res),mean(r1_res),mean(r2_res),mean(r3_res)))
    rstds <- data.frame(r_values=c(0,1,2,3),r_cue=c(sd(r0_cue),sd(r1_cue),sd(r2_cue),sd(r3_cue)),r_res=c(sd(r0_res),sd(r1_res),sd(r2_res),sd(r3_res)))
    
    avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
    std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='r_values')
    
    plt <- ggplot(test,aes(x=r_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1) 
    plt <- plt +  geom_line() + geom_point() + scale_colour_manual(name="Unit Key",values=c("r_cue"="maroon","r_res"="slateblue"))
    plt <- plt + labs(title=paste('Reward: T',i,' corr, Unit ',r_unit_ind,sep=""),x="Reward Number",y="Firing rate") 
    plot(plt)
    graphics.off()
    
    #############
    png(paste('corr_p_t',i,'_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
    
    pavgs <- data.frame(p_values=c(0,1,2,3),p_cue=c(mean(p0_cue),mean(p1_cue),mean(p2_cue),mean(p3_cue)),p_res = c(mean(p0_res),mean(p1_res),mean(p2_res),mean(p3_res)))
    pstds <- data.frame(p_values=c(0,1,2,3),p_cue=c(sd(p0_cue),sd(p1_cue),sd(p2_cue),sd(p3_cue)),p_res=c(sd(p0_res),sd(p1_res),sd(p2_res),sd(p3_res)))
    
    avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
    std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='p_values')
    
    plt <- ggplot(test,aes(x=p_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1) 
    plt <- plt +  geom_line() + geom_point() + scale_colour_manual(name="Unit Key",values=c("p_cue"="maroon","p_res"="slateblue"))
    plt <- plt + labs(title=paste('Punishment: T',i,' corr, Unit ',p_unit_ind,sep=""),x="Punishment Number",y="Firing rate") 
    plot(plt)
    graphics.off()
    
    ##########
    png(paste('corr_res_t',i,'_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
    
    resavgs <- data.frame(res_values=c(0,1),res_cue = c(mean(res0_cue),mean(res1_cue)),res_res = c(mean(res0_res),mean(res1_res)))
    resstds <- data.frame(res_values=c(0,1),res_cue=c(sd(res0_cue),sd(res1_cue)),res_res=c(sd(res0_res),sd(res1_res)))
    
    avg_melt <- melt(resavgs,id="res_values",variable.name='type',value.name='avg')
    std_melt <- melt(resstds,id="res_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='res_values')
    
    plt <- ggplot(test,aes(x=res_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1) 
    plt <- plt +  geom_line() + geom_point() + scale_colour_manual(name="Unit Key",values=c("res_cue"="maroon","res_res"="slateblue"))
    plt <- plt + labs(title=paste('Result: T',i,' corr, Unit ',res_unit_ind,sep=""),x="0: fail, 1: Succ",y="Firing rate") 
    plot(plt)
    graphics.off()
    
    ##########
    png(paste('corr_v_t',i,'_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
    
    vavgs <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue = c(mean(v_3_cue),mean(v_2_cue),mean(v_1_cue),mean(v0_cue),mean(v1_cue),mean(v2_cue),mean(v3_cue)),v_res = c(mean(v_3_res),mean(v_2_res),mean(v_1_res),mean(v0_res),mean(v1_res),mean(v2_res),mean(v3_res)))
    vstds <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue=c(sd(v_3_cue),sd(v_2_cue),sd(v_1_cue),sd(v0_cue),sd(v1_cue),sd(v2_cue),sd(v3_cue)),v_res=c(sd(v_3_res),sd(v_2_res),sd(v_1_res),sd(v0_res),sd(v1_res),sd(v2_res),sd(v3_res)))
    
    avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
    std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='v_values')
    
    plt <- ggplot(test,aes(x=v_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1) 
    plt <- plt +  geom_line() + geom_point() + scale_colour_manual(name="Unit Key",values=c("v_cue"="maroon","v_res"="slateblue"))
    plt <- plt + labs(title=paste('Value: T',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="Firing rate") 
    plot(plt)
    graphics.off()
    
    ##########
    png(paste('corr_m_t',i,'_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
    
    mavgs <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue = c(mean(m0_cue),mean(m1_cue),mean(m2_cue),mean(m3_cue),mean(m4_cue),mean(m5_cue),mean(v3_cue)),m_res = c(mean(m0_res),mean(m1_res),mean(m2_res),mean(m3_res),mean(m4_res),mean(m5_res),mean(v3_res)))
    mstds <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue=c(sd(m0_cue),sd(m1_cue),sd(m2_cue),sd(m3_cue),sd(m4_cue),sd(m5_cue),sd(v3_cue)),m_res=c(sd(m0_res),sd(m1_res),sd(m2_res),sd(m3_res),sd(m4_res),sd(m5_res),sd(v3_res)))
    
    avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
    std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='m_values')
    
    plt <- ggplot(test,aes(x=m_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1) 
    plt <- plt +  geom_line() + geom_point() + scale_colour_manual(name="Unit Key",values=c("m_cue"="maroon","m_res"="slateblue"))
    plt <- plt + labs(title=paste('Motivation: T',i,' corr, Unit ',mtv_unit_ind,sep=""),x="Motivation",y="Firing rate") 
    plot(plt)
    graphics.off()  
  }
  
  #plot bottom units
  for (i in 1:units_to_plot){
    r_unit_ind <- which((length(r_corr_order)-i)==r_corr_order)
    p_unit_ind <- which((length(p_corr_order)-i)==p_corr_order)
    val_unit_ind <- which((length(val_corr_order)-i)==val_corr_order)
    mtv_unit_ind <- which((length(mtv_corr_order)-i)==mtv_corr_order)
    res_unit_ind <- which((length(res_corr_order)-i)==res_corr_order)
    
    #cue_avg_frs <- cue_avg[unit_ind,]
    #res_avg_frs <- res_avg[unit_ind,]
    #cue_peak_frs <- cue_peaks[unit_ind,]
    #res_peak_frs <- res_peaks[unit_ind,]
    
    r0_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==0)]
    r1_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==1)]
    r2_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==2)]
    r3_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==3)]
    r0_res <- res_avg[r_unit_ind,][which(condensed[,4]==0)]
    r1_res <- res_avg[r_unit_ind,][which(condensed[,4]==1)]
    r2_res <- res_avg[r_unit_ind,][which(condensed[,4]==2)]
    r3_res <- res_avg[r_unit_ind,][which(condensed[,4]==3)]
    
    p0_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==0)]
    p1_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==1)]
    p2_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==2)]
    p3_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==3)]
    p0_res <- res_avg[p_unit_ind,][which(condensed[,5]==0)]
    p1_res <- res_avg[p_unit_ind,][which(condensed[,5]==1)]
    p2_res <- res_avg[p_unit_ind,][which(condensed[,5]==2)]
    p3_res <- res_avg[p_unit_ind,][which(condensed[,5]==3)]
    
    res0_cue <- cue_avg[res_unit_ind,][which(condensed[,6]==0)]
    res1_cue <- cue_avg[res_unit_ind,][which(condensed[,6]==1)]
    res0_res <- res_avg[res_unit_ind,][which(condensed[,6]==0)]
    res1_res <- res_avg[res_unit_ind,][which(condensed[,6]==1)]
    
    v_3_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==-3)]
    v_2_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==-2)]
    v_1_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==-1)]
    v0_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==0)]
    v1_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==1)]
    v2_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==2)]
    v3_cue <- cue_avg[val_unit_ind,][which(condensed[,7]==3)]
    v_3_res <- res_avg[val_unit_ind,][which(condensed[,7]==-3)]
    v_2_res <- res_avg[val_unit_ind,][which(condensed[,7]==-2)]
    v_1_res <- res_avg[val_unit_ind,][which(condensed[,7]==-1)]
    v0_res <- res_avg[val_unit_ind,][which(condensed[,7]==0)]
    v1_res <- res_avg[val_unit_ind,][which(condensed[,7]==1)]
    v2_res <- res_avg[val_unit_ind,][which(condensed[,7]==2)]
    v3_res <- res_avg[val_unit_ind,][which(condensed[,7]==3)]    
    
    m0_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==0)]
    m1_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==1)]
    m2_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==2)]
    m3_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==3)]
    m4_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==4)]
    m5_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==5)]
    m6_cue <- cue_avg[mtv_unit_ind,][which(condensed[,8]==6)]
    m0_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==0)]
    m1_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==1)]
    m2_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==2)]
    m3_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==3)]
    m4_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==4)]
    m5_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==5)]
    m6_res <- res_avg[mtv_unit_ind,][which(condensed[,8]==6)] 
    
    ##########
    png(paste('corr_r_b',i,'_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
    
    ravgs <- data.frame(r_values=c(0,1,2,3),r_cue = c(mean(r0_cue),mean(r1_cue),mean(r2_cue),mean(r3_cue)),r_res = c(mean(r0_res),mean(r1_res),mean(r2_res),mean(r3_res)))
    rstds <- data.frame(r_values=c(0,1,2,3),r_cue=c(sd(r0_cue),sd(r1_cue),sd(r2_cue),sd(r3_cue)),r_res=c(sd(r0_res),sd(r1_res),sd(r2_res),sd(r3_res)))
    
    avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
    std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='r_values')
    
    plt <- ggplot(test,aes(x=r_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1) 
    plt <- plt +  geom_line() + geom_point() + scale_colour_manual(name="Unit Key",values=c("r_cue"="maroon","r_res"="slateblue"))
    plt <- plt + labs(title=paste('Reward: B',i,' corr, Unit ',r_unit_ind,sep=""),x="Reward Number",y="Firing rate") 
    plot(plt)
    graphics.off()
    
    #############
    png(paste('corr_p_b',i,'_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
    
    pavgs <- data.frame(p_values=c(0,1,2,3),p_cue=c(mean(p0_cue),mean(p1_cue),mean(p2_cue),mean(p3_cue)),p_res = c(mean(p0_res),mean(p1_res),mean(p2_res),mean(p3_res)))
    pstds <- data.frame(p_values=c(0,1,2,3),p_cue=c(sd(p0_cue),sd(p1_cue),sd(p2_cue),sd(p3_cue)),p_res=c(sd(p0_res),sd(p1_res),sd(p2_res),sd(p3_res)))
    
    avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
    std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='p_values')
    
    plt <- ggplot(test,aes(x=p_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1) 
    plt <- plt +  geom_line() + geom_point() + scale_colour_manual(name="Unit Key",values=c("p_cue"="maroon","p_res"="slateblue"))
    plt <- plt + labs(title=paste('Punishment: B',i,' corr, Unit ',p_unit_ind,sep=""),x="Punishment Number",y="Firing rate") 
    plot(plt)
    graphics.off()
    
    ##########
    png(paste('corr_res_b',i,'_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
    
    resavgs <- data.frame(res_values=c(0,1),res_cue = c(mean(res0_cue),mean(res1_cue)),res_res = c(mean(res0_res),mean(res1_res)))
    resstds <- data.frame(res_values=c(0,1),res_cue=c(sd(res0_cue),sd(res1_cue)),res_res=c(sd(res0_res),sd(res1_res)))
    
    avg_melt <- melt(resavgs,id="res_values",variable.name='type',value.name='avg')
    std_melt <- melt(resstds,id="res_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='res_values')
    
    plt <- ggplot(test,aes(x=res_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1) 
    plt <- plt +  geom_line() + geom_point() + scale_colour_manual(name="Unit Key",values=c("res_cue"="maroon","res_res"="slateblue"))
    plt <- plt + labs(title=paste('Result: B',i,' corr, Unit ',res_unit_ind,sep=""),x="0: fail, 1: Succ",y="Firing rate") 
    plot(plt)
    graphics.off()
    
    ##########
    png(paste('corr_v_b',i,'_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
    
    vavgs <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue = c(mean(v_3_cue),mean(v_2_cue),mean(v_1_cue),mean(v0_cue),mean(v1_cue),mean(v2_cue),mean(v3_cue)),v_res = c(mean(v_3_res),mean(v_2_res),mean(v_1_res),mean(v0_res),mean(v1_res),mean(v2_res),mean(v3_res)))
    vstds <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue=c(sd(v_3_cue),sd(v_2_cue),sd(v_1_cue),sd(v0_cue),sd(v1_cue),sd(v2_cue),sd(v3_cue)),v_res=c(sd(v_3_res),sd(v_2_res),sd(v_1_res),sd(v0_res),sd(v1_res),sd(v2_res),sd(v3_res)))
    
    avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
    std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='v_values')
    
    plt <- ggplot(test,aes(x=v_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1) 
    plt <- plt +  geom_line() + geom_point() + scale_colour_manual(name="Unit Key",values=c("v_cue"="maroon","v_res"="slateblue"))
    plt <- plt + labs(title=paste('Value: B',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="Firing rate") 
    plot(plt)
    graphics.off()
    
    ##########
    png(paste('corr_m_b',i,'_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
    
    mavgs <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue = c(mean(m0_cue),mean(m1_cue),mean(m2_cue),mean(m3_cue),mean(m4_cue),mean(m5_cue),mean(v3_cue)),m_res = c(mean(m0_res),mean(m1_res),mean(m2_res),mean(m3_res),mean(m4_res),mean(m5_res),mean(v3_res)))
    mstds <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue=c(sd(m0_cue),sd(m1_cue),sd(m2_cue),sd(m3_cue),sd(m4_cue),sd(m5_cue),sd(v3_cue)),m_res=c(sd(m0_res),sd(m1_res),sd(m2_res),sd(m3_res),sd(m4_res),sd(m5_res),sd(v3_res)))
    
    avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
    std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='m_values')
    
    plt <- ggplot(test,aes(x=m_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1) 
    plt <- plt +  geom_line() + geom_point() + scale_colour_manual(name="Unit Key",values=c("m_cue"="maroon","m_res"="slateblue"))
    plt <- plt + labs(title=paste('Motivation: B',i,' corr, Unit ',mtv_unit_ind,sep=""),x="Motivation",y="Firing rate") 
    plot(plt)
    graphics.off()  
  }
  
  
  
}