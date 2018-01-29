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
library(ggpmisc)

#################
#### Params #####
#################

units_to_plot <- 5

plot_p_bool <- TRUE


region_list <- c('M1','S1','PmD')


#############
formula <- y ~ x

for (region_index in 1:length(region_list)){
  cat('plotting ',region_list[region_index],'\n')
  
  #corr hists
  readin <- readMat(paste('corr_output_',region_list[region_index],'.mat',sep=""))
  corr_output <- readin$corr.output
  corr_output_df <- data.frame(catch_bin_corr=corr_output[,6],catch_mult_corr=corr_output[,7])
  
  catch_bin_corr <- ggplot(corr_output_df,aes(x=corr_output_df$catch_bin_corr))+geom_histogram(position='identity',binwidth=0.05)
  catch_bin_corr <- catch_bin_corr + labs(title="Catch Binary Correlation",y="Number of Units",x="Correlation Coefficient")
  
  catch_mult_corr <- ggplot(corr_output_df,aes(x=corr_output_df$catch_mult_corr))+geom_histogram(position='identity',binwidth=0.05)
  catch_mult_corr <- catch_mult_corr + labs(title="Catch Multiple Correlation",y="Number of Units",x="Correlation Coefficient")
  

  png(paste('corr_hist_catch_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  multiplot(catch_bin_corr,catch_mult_corr,cols=1)
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
  
  png(paste('corr_tb_r_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  plt <- ggplot(test,aes(x=r_values,y=avg,color=type))  + theme_classic() #+ geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1,position=pd)   ######geom_line(aes(linetype=type))
  plt <- plt + geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_linetype_manual(name="Unit Key",values=c("top_cue"="solid","btm_cue"="dotted","top_res"="solid","btm_res"="dotted")) + scale_colour_manual(name="Unit Key",values=c("top_cue"="coral4","btm_cue"="coral1","top_res"="slateblue","btm_res"="slategray3"))
  plt <- plt + labs(title="Reward",x="Reward",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 1.0, size = 3,na.rm=T)}
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
  
  png(paste('corr_tb_p_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  plt <- ggplot(test,aes(x=p_values,y=avg,color=type))  + theme_classic()#+ geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1,position=pd) 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_linetype_manual(name="Unit Key",values=c("top_cue"="solid","btm_cue"="dotted","top_res"="solid","btm_res"="dotted")) + scale_colour_manual(name="Unit Key",values=c("top_cue"="coral4","btm_cue"="coral1","top_res"="slateblue","btm_res"="slategray3"))
  plt <- plt + labs(title="Punishment",x="Punishment",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 1.0, size = 3,na.rm=T)}
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
  
  png(paste('corr_tb_res_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  plt <- ggplot(test,aes(x=res_values,y=avg,color=type)) + theme_classic() #+ geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1,position=pd) 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_linetype_manual(name="Unit Key",values=c("top_cue"="solid","btm_cue"="dotted","top_res"="solid","btm_res"="dotted")) + scale_colour_manual(name="Unit Key",values=c("top_cue"="coral4","btm_cue"="coral1","top_res"="slateblue","btm_res"="slategray3"))
  plt <- plt + labs(title="Result",x="0: Failure, 1: Success",y="z-score") 
  #if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 1.0, size = 3,na.rm=T)}
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
  
  png(paste('corr_tb_v_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)

  plt <- ggplot(test,aes(x=v_values,y=avg,color=type))  + theme_classic()#+ geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1,position=pd) 
  plt <- plt + geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_linetype_manual(name="Unit Key",values=c("top_cue"="solid","btm_cue"="dotted","top_res"="solid","btm_res"="dotted")) + scale_colour_manual(name="Unit Key",values=c("top_cue"="coral4","btm_cue"="coral1","top_res"="slateblue","btm_res"="slategray3"))
  plt <- plt + labs(title="Value",x="Value",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 1.0, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  #
  readin <- readMat(paste('m_avgs_',region_list[region_index],'.mat',sep=""))
  m_avgs <- lapply(readin$m.avgs,'[[',1)
  
  all_avgs <- lapply(m_avgs,mean)
  all_stds <- lapply(m_avgs,sd)
  
  m_values <- c(0,1,2,3,4,5,6)
  top_avg_cue <- c(all_avgs[[1]],all_avgs[[3]],all_avgs[[5]],all_avgs[[7]],all_avgs[[9]],all_avgs[[11]],all_avgs[[13]])
  btm_avg_cue <- c(all_avgs[[15]],all_avgs[[17]],all_avgs[[19]],all_avgs[[21]],all_avgs[[23]],all_avgs[[25]],all_avgs[[27]])
  top_avg_res <- c(all_avgs[[2]],all_avgs[[4]],all_avgs[[6]],all_avgs[[8]],all_avgs[[10]],all_avgs[[12]],all_avgs[[14]])
  btm_avg_res <- c(all_avgs[[16]],all_avgs[[18]],all_avgs[[20]],all_avgs[[22]],all_avgs[[24]],all_avgs[[26]],all_avgs[[28]])
  
  m_avgs <- data.frame(m_values=m_values,top_cue=top_avg_cue,btm_cue=btm_avg_cue,top_res=top_avg_res,btm_res=btm_avg_res)
  
  top_std_cue <- c(all_stds[[1]],all_stds[[3]],all_stds[[5]],all_stds[[7]],all_stds[[9]],all_stds[[11]],all_stds[[13]])
  btm_std_cue <- c(all_stds[[15]],all_stds[[17]],all_stds[[19]],all_stds[[21]],all_stds[[23]],all_stds[[25]],all_stds[[27]])
  top_std_res <- c(all_stds[[2]],all_stds[[4]],all_stds[[6]],all_stds[[8]],all_stds[[10]],all_stds[[12]],all_stds[[14]])
  btm_std_res <- c(all_stds[[16]],all_stds[[18]],all_stds[[20]],all_stds[[22]],all_stds[[24]],all_stds[[26]],all_stds[[28]])
  
  m_stds <- data.frame(m_values=m_values,top_cue=top_std_cue,btm_cue=btm_std_cue,top_res=top_std_res,btm_res=btm_std_res)
  
  avg_melt <- melt(m_avgs,id="m_values",variable.name='type',value.name='avg')
  std_melt <- melt(m_stds,id="m_values",variable.name='type',value.name='std')
  test <- merge(std_melt,avg_melt,row.names='m_values')
  
  png(paste('corr_tb_m_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  plt <- ggplot(test,aes(x=m_values,y=avg,color=type))  + theme_classic()#+ geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1,position=pd) 
  plt <- plt + geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_linetype_manual(name="Unit Key",values=c("top_cue"="solid","btm_cue"="dotted","top_res"="solid","btm_res"="dotted")) + scale_colour_manual(name="Unit Key",values=c("top_cue"="coral4","btm_cue"="coral1","top_res"="slateblue","btm_res"="slategray3"))
  plt <- plt + labs(title="Motivation",x="Motivation",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 1.0, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  readin <- readMat(paste('catch_mult_avgs_',region_list[region_index],'.mat',sep=""))
  catch_mult_avgs <- lapply(readin$catch.mult.avgs,'[[',1)
  
  readin <- readMat(paste('catch_bin_avgs_',region_list[region_index],'.mat',sep=""))
  catch_bin_avgs <- lapply(readin$catch.bin.avgs,'[[',1)
  
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
  catch_mult_corr_order <- readin$order.dict[,,1]$catch.mult.order
  catch_bin_corr_order <- readin$order.dict[,,1]$catch.bin.order
  condensed <- readin$order.dict[,,1]$condensed
  
  r0_cue_total_t <- c()
  r1_cue_total_t <- c()
  r2_cue_total_t <- c()
  r3_cue_total_t <- c()
  r0_res_total_t <- c()
  r1_res_total_t <- c()
  r2_res_total_t <- c()
  r3_res_total_t <- c()
  r0_res_succ_total_t <- c()
  r1_res_succ_total_t <- c()
  r2_res_succ_total_t <- c()
  r3_res_succ_total_t <- c()
  r0_res_fail_total_t <- c()
  r1_res_fail_total_t <- c()
  r2_res_fail_total_t <- c()
  r3_res_fail_total_t <- c()
  
  p0_cue_total_t <- c()
  p1_cue_total_t <- c()
  p2_cue_total_t <- c()
  p3_cue_total_t <- c()
  p0_res_total_t <- c()
  p1_res_total_t <- c()
  p2_res_total_t <- c()
  p3_res_total_t <- c()
  p0_res_succ_total_t <- c()
  p1_res_succ_total_t <- c()
  p2_res_succ_total_t <- c()
  p3_res_succ_total_t <- c()
  p0_res_fail_total_t <- c()
  p1_res_fail_total_t <- c()
  p2_res_fail_total_t <- c()
  p3_res_fail_total_t <- c()
  
  res0_cue_total_t <- c()
  res1_cue_total_t <- c()
  res0_res_total_t <- c()
  res1_res_total_t <- c()

  v_3_cue_total_t <- c()
  v_2_cue_total_t <- c()
  v_1_cue_total_t <- c()
  v0_cue_total_t <- c()
  v1_cue_total_t <- c()
  v2_cue_total_t <- c()
  v3_cue_total_t <- c()
  v_3_res_total_t <- c()
  v_2_res_total_t <- c()
  v_1_res_total_t <- c()
  v0_res_total_t <- c()
  v1_res_total_t <- c()
  v2_res_total_t <- c()
  v3_res_total_t <- c()
  v_3_res_succ_total_t <- c()
  v_2_res_succ_total_t <- c()
  v_1_res_succ_total_t <- c()
  v0_res_succ_total_t <- c()
  v1_res_succ_total_t <- c()
  v2_res_succ_total_t <- c()
  v3_res_succ_total_t <- c()
  v_3_res_fail_total_t <- c()
  v_2_res_fail_total_t <- c()
  v_1_res_fail_total_t <- c()
  v0_res_fail_total_t <- c()
  v1_res_fail_total_t <- c()
  v2_res_fail_total_t <- c()
  v3_res_fail_total_t <- c()
  
  m0_cue_total_t <- c()
  m1_cue_total_t <- c()
  m2_cue_total_t <- c()
  m3_cue_total_t <- c()
  m4_cue_total_t <- c()
  m5_cue_total_t <- c()
  m6_cue_total_t <- c()
  m0_res_total_t <- c()
  m1_res_total_t <- c()
  m2_res_total_t <- c()
  m3_res_total_t <- c()
  m4_res_total_t <- c()
  m5_res_total_t <- c()
  m6_res_total_t <- c()
  m0_res_succ_total_t <- c()
  m1_res_succ_total_t <- c()
  m2_res_succ_total_t <- c()
  m3_res_succ_total_t <- c()
  m4_res_succ_total_t <- c()
  m5_res_succ_total_t <- c()
  m6_res_succ_total_t <- c()
  m0_res_fail_total_t <- c()
  m1_res_fail_total_t <- c()
  m2_res_fail_total_t <- c()
  m3_res_fail_total_t <- c()
  m4_res_fail_total_t <- c()
  m5_res_fail_total_t <- c()
  m6_res_fail_total_t <- c()
  
  r0_cue_total_b <- c()
  r1_cue_total_b <- c()
  r2_cue_total_b <- c()
  r3_cue_total_b <- c()
  r0_res_total_b <- c()
  r1_res_total_b <- c()
  r2_res_total_b <- c()
  r3_res_total_b <- c()
  r0_res_succ_total_b <- c()
  r1_res_succ_total_b <- c()
  r2_res_succ_total_b <- c()
  r3_res_succ_total_b <- c()
  r0_res_fail_total_b <- c()
  r1_res_fail_total_b <- c()
  r2_res_fail_total_b <- c()
  r3_res_fail_total_b <- c()
  
  p0_cue_total_b <- c()
  p1_cue_total_b <- c()
  p2_cue_total_b <- c()
  p3_cue_total_b <- c()
  p0_res_total_b <- c()
  p1_res_total_b <- c()
  p2_res_total_b <- c()
  p3_res_total_b <- c()
  p0_res_succ_total_b <- c()
  p1_res_succ_total_b <- c()
  p2_res_succ_total_b <- c()
  p3_res_succ_total_b <- c()
  p0_res_fail_total_b <- c()
  p1_res_fail_total_b <- c()
  p2_res_fail_total_b <- c()
  p3_res_fail_total_b <- c()
  
  res0_cue_total_b <- c()
  res1_cue_total_b <- c()
  res0_res_total_b <- c()
  res1_res_total_b <- c()

  v_3_cue_total_b <- c()
  v_2_cue_total_b <- c()
  v_1_cue_total_b <- c()
  v0_cue_total_b <- c()
  v1_cue_total_b <- c()
  v2_cue_total_b <- c()
  v3_cue_total_b <- c()
  v_3_res_total_b <- c()
  v_2_res_total_b <- c()
  v_1_res_total_b <- c()
  v0_res_total_b <- c()
  v1_res_total_b <- c()
  v2_res_total_b <- c()
  v3_res_total_b <- c()
  v_3_res_succ_total_b <- c()
  v_2_res_succ_total_b <- c()
  v_1_res_succ_total_b <- c()
  v0_res_succ_total_b <- c()
  v1_res_succ_total_b <- c()
  v2_res_succ_total_b <- c()
  v3_res_succ_total_b <- c()
  v_3_res_fail_total_b <- c()
  v_2_res_fail_total_b <- c()
  v_1_res_fail_total_b <- c()
  v0_res_fail_total_b <- c()
  v1_res_fail_total_b <- c()
  v2_res_fail_total_b <- c()
  v3_res_fail_total_b <- c()
  
  m0_cue_total_b <- c()
  m1_cue_total_b <- c()
  m2_cue_total_b <- c()
  m3_cue_total_b <- c()
  m4_cue_total_b <- c()
  m5_cue_total_b <- c()
  m6_cue_total_b <- c()
  m0_res_total_b <- c()
  m1_res_total_b <- c()
  m2_res_total_b <- c()
  m3_res_total_b <- c()
  m4_res_total_b <- c()
  m5_res_total_b <- c()
  m6_res_total_b <- c()
  m0_res_succ_total_b <- c()
  m1_res_succ_total_b <- c()
  m2_res_succ_total_b <- c()
  m3_res_succ_total_b <- c()
  m4_res_succ_total_b <- c()
  m5_res_succ_total_b <- c()
  m6_res_succ_total_b <- c()
  m0_res_fail_total_b <- c()
  m1_res_fail_total_b <- c()
  m2_res_fail_total_b <- c()
  m3_res_fail_total_b <- c()
  m4_res_fail_total_b <- c()
  m5_res_fail_total_b <- c()
  m6_res_fail_total_b <- c()
  
  catch_3_cue_total_t <- c()
  catch_2_cue_total_t <- c()
  catch_1_cue_total_t <- c()
  catch1_cue_total_t <- c()
  catch2_cue_total_t <- c()
  catch3_cue_total_t <- c()
  catch_3_res_total_t <- c()
  catch_2_res_total_t <- c()
  catch_1_res_total_t <- c()
  catch1_res_total_t <- c()
  catch2_res_total_t <- c()
  catch3_res_total_t <- c()

  catch_x_cue_total_t <- c()
  catchx_cue_total_t <- c()
  catch_x_res_total_t <- c()
  catchx_res_total_t <- c()
  
  catch_3_cue_total_b <- c()
  catch_2_cue_total_b <- c()
  catch_1_cue_total_b <- c()
  catch1_cue_total_b <- c()
  catch2_cue_total_b <- c()
  catch3_cue_total_b <- c()
  catch_3_res_total_b <- c()
  catch_2_res_total_b <- c()
  catch_1_res_total_b <- c()
  catch1_res_total_b <- c()
  catch2_res_total_b <- c()
  catch3_res_total_b <- c()
  
  catch_x_cue_total_b <- c()
  catchx_cue_total_b <- c()
  catch_x_res_total_b <- c()
  catchx_res_total_b <- c()
  
  #plot top units
  #i-1 b/ python indexing
  for (i in 1:units_to_plot){
    r_unit_ind <- which((i-1)==r_corr_order)
    p_unit_ind <- which((i-1)==p_corr_order)
    val_unit_ind <- which((i-1)==val_corr_order)
    mtv_unit_ind <- which((i-1)==mtv_corr_order)
    res_unit_ind <- which((i-1)==res_corr_order)
    catch_mult_unit_ind <- which((i-1)==catch_mult_corr_order)
    catch_bin_unit_ind <- which((i-1)==catch_bin_corr_order)
    
    r0_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==0)]
    r1_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==1)]
    r2_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==2)]
    r3_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==3)]
    r0_res <- res_avg[r_unit_ind,][which(condensed[,4]==0)]
    r1_res <- res_avg[r_unit_ind,][which(condensed[,4]==1)]
    r2_res <- res_avg[r_unit_ind,][which(condensed[,4]==2)]
    r3_res <- res_avg[r_unit_ind,][which(condensed[,4]==3)]
    
    r0_res_succ <- res_avg[r_unit_ind,][which(condensed[,4]==0 & condensed[,6]==1)]
    r1_res_succ <- res_avg[r_unit_ind,][which(condensed[,4]==1 & condensed[,6]==1)]
    r2_res_succ <- res_avg[r_unit_ind,][which(condensed[,4]==2 & condensed[,6]==1)]
    r3_res_succ <- res_avg[r_unit_ind,][which(condensed[,4]==3 & condensed[,6]==1)]
    r0_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==0 & condensed[,6]==0)]
    r1_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==1 & condensed[,6]==0)]
    r2_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==2 & condensed[,6]==0)]
    r3_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==3 & condensed[,6]==0)]
        
    p0_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==0)]
    p1_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==1)]
    p2_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==2)]
    p3_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==3)]
    p0_res <- res_avg[p_unit_ind,][which(condensed[,5]==0)]
    p1_res <- res_avg[p_unit_ind,][which(condensed[,5]==1)]
    p2_res <- res_avg[p_unit_ind,][which(condensed[,5]==2)]
    p3_res <- res_avg[p_unit_ind,][which(condensed[,5]==3)]
    
    p0_res_succ <- res_avg[p_unit_ind,][which(condensed[,5]==0 & condensed[,6]==1)]
    p1_res_succ <- res_avg[p_unit_ind,][which(condensed[,5]==1 & condensed[,6]==1)]
    p2_res_succ <- res_avg[p_unit_ind,][which(condensed[,5]==2 & condensed[,6]==1)]
    p3_res_succ <- res_avg[p_unit_ind,][which(condensed[,5]==3 & condensed[,6]==1)]
    p0_res_fail <- res_avg[p_unit_ind,][which(condensed[,5]==0 & condensed[,6]==0)]
    p1_res_fail <- res_avg[p_unit_ind,][which(condensed[,5]==1 & condensed[,6]==0)]
    p2_res_fail <- res_avg[p_unit_ind,][which(condensed[,5]==2 & condensed[,6]==0)]
    p3_res_fail <- res_avg[p_unit_ind,][which(condensed[,5]==3 & condensed[,6]==0)]
    
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
    
    v_3_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]==-3 & condensed[,6]==1)]
    v_2_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]==-2 & condensed[,6]==1)]
    v_1_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]== 1 & condensed[,6]==1)]
    v0_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]==0 & condensed[,6]==1)]
    v1_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]==1 & condensed[,6]==1)]
    v2_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]==2 & condensed[,6]==1)]
    v3_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]==3 & condensed[,6]==1)] 
    v_3_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]==-3 & condensed[,6]==0)]
    v_2_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]==-2 & condensed[,6]==0)]
    v_1_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]== 1 & condensed[,6]==0)]
    v0_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]==0 & condensed[,6]==0)]
    v1_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]==1 & condensed[,6]==0)]
    v2_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]==2 & condensed[,6]==0)]
    v3_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]==3 & condensed[,6]==0)] 
    
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
    
    m0_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==0 & condensed[,6]==1)]
    m1_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==1 & condensed[,6]==1)]
    m2_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==2 & condensed[,6]==1)]
    m3_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==3 & condensed[,6]==1)]
    m4_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==4 & condensed[,6]==1)]
    m5_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==5 & condensed[,6]==1)]
    m6_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==6 & condensed[,6]==1)] 
    m0_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==0 & condensed[,6]==0)]
    m1_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==1 & condensed[,6]==0)]
    m2_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==2 & condensed[,6]==0)]
    m3_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==3 & condensed[,6]==0)]
    m4_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==4 & condensed[,6]==0)]
    m5_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==5 & condensed[,6]==0)]
    m6_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==6 & condensed[,6]==0)] 
    
    catch_3_cue <- cue_avg[catch_mult_unit_ind,][which(condensed[,12]==-3)]
    catch_2_cue <- cue_avg[catch_mult_unit_ind,][which(condensed[,12]==-2)]
    catch_1_cue <- cue_avg[catch_mult_unit_ind,][which(condensed[,12]==-1)]
    catch1_cue <- cue_avg[catch_mult_unit_ind,][which(condensed[,12]==1)]
    catch2_cue <- cue_avg[catch_mult_unit_ind,][which(condensed[,12]==2)]
    catch3_cue <- cue_avg[catch_mult_unit_ind,][which(condensed[,12]==3)]
    catch_3_res <- res_avg[catch_mult_unit_ind,][which(condensed[,12]==-3)]
    catch_2_res <- res_avg[catch_mult_unit_ind,][which(condensed[,12]==-2)]
    catch_1_res <- res_avg[catch_mult_unit_ind,][which(condensed[,12]==-1)]
    catch1_res <- res_avg[catch_mult_unit_ind,][which(condensed[,12]==1)]
    catch2_res <- res_avg[catch_mult_unit_ind,][which(condensed[,12]==2)]
    catch3_res <- res_avg[catch_mult_unit_ind,][which(condensed[,12]==3)]    
    
    catch_x_cue <- cue_avg[catch_bin_unit_ind,][which(condensed[,12]<=-1)]
    catchx_cue <- cue_avg[catch_bin_unit_ind,][which(condensed[,12]>=1)]
    
    catch_x_res <- res_avg[catch_bin_unit_ind,][which(condensed[,12]<=-1)]
    catchx_res <- res_avg[catch_bin_unit_ind,][which(condensed[,12]>=1)]
    

    ##########
    r0_cue_total_t <- c(r0_cue_total_t,r0_cue)
    r1_cue_total_t <- c(r1_cue_total_t,r1_cue)
    r2_cue_total_t <- c(r2_cue_total_t,r2_cue)
    r3_cue_total_t <- c(r3_cue_total_t,r3_cue)
    r0_res_total_t <- c(r0_res_total_t,r0_res)
    r1_res_total_t <- c(r1_res_total_t,r1_res)
    r2_res_total_t <- c(r2_res_total_t,r2_res)
    r3_res_total_t <- c(r3_res_total_t,r3_res)
    r0_res_succ_total_t <- c(r0_res_succ_total_t,r0_res_succ)
    r1_res_succ_total_t <- c(r1_res_succ_total_t,r1_res_succ)
    r2_res_succ_total_t <- c(r2_res_succ_total_t,r2_res_succ)
    r3_res_succ_total_t <- c(r3_res_succ_total_t,r3_res_succ)
    r0_res_fail_total_t <- c(r0_res_fail_total_t,r0_res_fail)
    r1_res_fail_total_t <- c(r1_res_fail_total_t,r1_res_fail)
    r2_res_fail_total_t <- c(r2_res_fail_total_t,r2_res_fail)
    r3_res_fail_total_t <- c(r3_res_fail_total_t,r3_res_fail)
    
    p0_cue_total_t <- c(p0_cue_total_t,p0_cue)
    p1_cue_total_t <- c(p1_cue_total_t,p1_cue)
    p2_cue_total_t <- c(p2_cue_total_t,p2_cue)
    p3_cue_total_t <- c(p3_cue_total_t,p3_cue)
    p0_res_total_t <- c(p0_res_total_t,p0_res)
    p1_res_total_t <- c(p1_res_total_t,p1_res)
    p2_res_total_t <- c(p2_res_total_t,p2_res)
    p3_res_total_t <- c(p3_res_total_t,p3_res)
    p0_res_succ_total_t <- c(p0_res_succ_total_t,p0_res_succ)
    p1_res_succ_total_t <- c(p1_res_succ_total_t,p1_res_succ)
    p2_res_succ_total_t <- c(p2_res_succ_total_t,p2_res_succ)
    p3_res_succ_total_t <- c(p3_res_succ_total_t,p3_res_succ)
    p0_res_fail_total_t <- c(p0_res_fail_total_t,p0_res_fail)
    p1_res_fail_total_t <- c(p1_res_fail_total_t,p1_res_fail)
    p2_res_fail_total_t <- c(p2_res_fail_total_t,p2_res_fail)
    p3_res_fail_total_t <- c(p3_res_fail_total_t,p3_res_fail)
    
    res0_cue_total_t <- c(res0_cue_total_t,res0_cue)
    res1_cue_total_t <- c(res1_cue_total_t,res1_cue)
    res0_res_total_t <- c(res0_res_total_t,res0_res)
    res1_res_total_t <- c(res1_res_total_t,res1_res)

    v_3_cue_total_t <- c(v_3_cue_total_t,v_3_cue)
    v_2_cue_total_t <- c(v_2_cue_total_t,v_2_cue)
    v_1_cue_total_t <- c(v_1_cue_total_t,v_1_cue)
    v0_cue_total_t <- c(v0_cue_total_t,v0_cue)
    v1_cue_total_t <- c(v1_cue_total_t,v1_cue)
    v2_cue_total_t <- c(v2_cue_total_t,v2_cue)
    v3_cue_total_t <- c(v3_cue_total_t,v3_cue)
    v_3_res_total_t <- c(v_3_res_total_t,v_3_res)
    v_2_res_total_t <- c(v_2_res_total_t,v_2_res)
    v_1_res_total_t <- c(v_1_res_total_t,v_1_res)
    v0_res_total_t <- c(v0_res_total_t,v0_res)
    v1_res_total_t <- c(v1_res_total_t,v1_res)
    v2_res_total_t <- c(v2_res_total_t,v2_res)
    v3_res_total_t <- c(v3_res_total_t,v3_res)
    v_3_res_succ_total_t <- c(v_3_res_succ_total_t,v_3_res_succ)
    v_2_res_succ_total_t <- c(v_2_res_succ_total_t,v_2_res_succ)
    v_1_res_succ_total_t <- c(v_1_res_succ_total_t,v_1_res_succ)
    v0_res_succ_total_t <- c(v0_res_succ_total_t,v0_res_succ)
    v1_res_succ_total_t <- c(v1_res_succ_total_t,v1_res_succ)
    v2_res_succ_total_t <- c(v2_res_succ_total_t,v2_res_succ)
    v3_res_succ_total_t <- c(v3_res_succ_total_t,v3_res_succ)
    v_3_res_fail_total_t <- c(v_3_res_fail_total_t,v_3_res_fail)
    v_2_res_fail_total_t <- c(v_2_res_fail_total_t,v_2_res_fail)
    v_1_res_fail_total_t <- c(v_1_res_fail_total_t,v_1_res_fail)
    v0_res_fail_total_t <- c(v0_res_fail_total_t,v0_res_fail)
    v1_res_fail_total_t <- c(v1_res_fail_total_t,v1_res_fail)
    v2_res_fail_total_t <- c(v2_res_fail_total_t,v2_res_fail)
    v3_res_fail_total_t <- c(v3_res_fail_total_t,v3_res_fail)

    m0_cue_total_t <- c(m0_cue_total_t,m0_cue)
    m1_cue_total_t <- c(m1_cue_total_t,m1_cue)
    m2_cue_total_t <- c(m2_cue_total_t,m2_cue)
    m3_cue_total_t <- c(m3_cue_total_t,m3_cue)
    m4_cue_total_t <- c(m4_cue_total_t,m4_cue)
    m5_cue_total_t <- c(m5_cue_total_t,m5_cue)
    m6_cue_total_t <- c(m6_cue_total_t,m6_cue)
    m0_res_total_t <- c(m0_res_total_t,m0_res)
    m1_res_total_t <- c(m1_res_total_t,m1_res)
    m2_res_total_t <- c(m2_res_total_t,m2_res)
    m3_res_total_t <- c(m3_res_total_t,m3_res)
    m4_res_total_t <- c(m4_res_total_t,m4_res)
    m5_res_total_t <- c(m5_res_total_t,m5_res)
    m6_res_total_t <- c(m6_res_total_t,m6_res)
    m0_res_succ_total_t <- c(m0_res_succ_total_t,m0_res_succ)
    m1_res_succ_total_t <- c(m1_res_succ_total_t,m1_res_succ)
    m2_res_succ_total_t <- c(m2_res_succ_total_t,m2_res_succ)
    m3_res_succ_total_t <- c(m3_res_succ_total_t,m3_res_succ)
    m4_res_succ_total_t <- c(m4_res_succ_total_t,m4_res_succ)
    m5_res_succ_total_t <- c(m5_res_succ_total_t,m5_res_succ)
    m6_res_succ_total_t <- c(m6_res_succ_total_t,m6_res_succ)
    m0_res_fail_total_t <- c(m0_res_fail_total_t,m0_res_fail)
    m1_res_fail_total_t <- c(m1_res_fail_total_t,m1_res_fail)
    m2_res_fail_total_t <- c(m2_res_fail_total_t,m2_res_fail)
    m3_res_fail_total_t <- c(m3_res_fail_total_t,m3_res_fail)
    m4_res_fail_total_t <- c(m4_res_fail_total_t,m4_res_fail)
    m5_res_fail_total_t <- c(m5_res_fail_total_t,m5_res_fail)
    m6_res_fail_total_t <- c(m6_res_fail_total_t,m6_res_fail)

    catch_3_cue_total_t <- c(catch_3_cue_total_t,catch_3_cue)
    catch_2_cue_total_t <- c(catch_2_cue_total_t,catch_2_cue)
    catch_1_cue_total_t <- c(catch_1_cue_total_t,catch_1_cue)
    catch1_cue_total_t <- c(catch1_cue_total_t,catch1_cue)
    catch2_cue_total_t <- c(catch2_cue_total_t,catch2_cue)
    catch3_cue_total_t <- c(catch3_cue_total_t,catch3_cue)
    catch_3_res_total_t <- c(catch_3_res_total_t,catch_3_res)
    catch_2_res_total_t <- c(catch_2_res_total_t,catch_2_res)
    catch_1_res_total_t <- c(catch_1_res_total_t,catch_1_res)
    catch1_res_total_t <- c(catch1_res_total_t,catch1_res)
    catch2_res_total_t <- c(catch2_res_total_t,catch2_res)
    catch3_res_total_t <- c(catch3_res_total_t,catch3_res)
    
    catch_x_cue_total_t <- c(catch_x_cue_total_t,catch_x_cue)
    catchx_cue_total_t <- c(catchx_cue_total_t,catchx_cue)
    
    catch_x_res_total_t <- c(catch_x_res_total_t,catch_x_res)
    catchx_res_total_t <- c(catchx_res_total_t,catchx_res)
    
    
    ##########

    png(paste('corr_r_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    ravgs <- data.frame(r_values=c(0,1,2,3),r_cue = c(mean(r0_cue),mean(r1_cue),mean(r2_cue),mean(r3_cue)),r_res = c(mean(r0_res),mean(r1_res),mean(r2_res),mean(r3_res)))
    rstds <- data.frame(r_values=c(0,1,2,3),r_cue=c(sd(r0_cue),sd(r1_cue),sd(r2_cue),sd(r3_cue)),r_res=c(sd(r0_res),sd(r1_res),sd(r2_res),sd(r3_res)))
    
    avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
    std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='r_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=r_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic()
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("r_cue"="coral4","r_res"="slateblue"))
    plt <- plt + labs(title=paste('Reward: T',i,' corr, Unit ',r_unit_ind,sep=""),x="Reward Number",y="z-score")
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    ##
    png(paste('corr_r_sf_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    ravgs <- data.frame(r_values=c(0,1,2,3),r_cue = c(mean(r0_cue),mean(r1_cue),mean(r2_cue),mean(r3_cue)),r_res_succ = c(mean(r0_res_succ),mean(r1_res_succ),mean(r2_res_succ),mean(r3_res_succ)),r_res_fail = c(mean(r0_res_fail),mean(r1_res_fail),mean(r2_res_fail),mean(r3_res_fail)))
    rstds <- data.frame(r_values=c(0,1,2,3),r_cue=c(sd(r0_cue),sd(r1_cue),sd(r2_cue),sd(r3_cue)),r_res_succ=c(sd(r0_res_succ),sd(r1_res_succ),sd(r2_res_succ),sd(r3_res_succ)),r_res_fail=c(sd(r0_res_fail),sd(r1_res_fail),sd(r2_res_fail),sd(r3_res_fail)))
    
    avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
    std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='r_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=r_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("r_cue"="darkorchid4","r_res_succ"="forestgreen","r_res_fail"="darkred"))
    plt <- plt + labs(title=paste('Reward: T',i,' corr, Unit ',r_unit_ind,sep=""),x="Reward Number",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    #############
    png(paste('corr_p_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    pavgs <- data.frame(p_values=c(0,1,2,3),p_cue=c(mean(p0_cue),mean(p1_cue),mean(p2_cue),mean(p3_cue)),p_res = c(mean(p0_res),mean(p1_res),mean(p2_res),mean(p3_res)))
    pstds <- data.frame(p_values=c(0,1,2,3),p_cue=c(sd(p0_cue),sd(p1_cue),sd(p2_cue),sd(p3_cue)),p_res=c(sd(p0_res),sd(p1_res),sd(p2_res),sd(p3_res)))
    
    avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
    std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='p_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=p_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("p_cue"="coral4","p_res"="slateblue"))
    plt <- plt + labs(title=paste('Punishment: T',i,' corr, Unit ',p_unit_ind,sep=""),x="Punishment Number",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    #
    png(paste('corr_p_sf_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    pavgs <- data.frame(p_values=c(0,1,2,3),p_cue=c(mean(p0_cue),mean(p1_cue),mean(p2_cue),mean(p3_cue)),p_res_succ = c(mean(p0_res_succ),mean(p1_res_succ),mean(p2_res_succ),mean(p3_res_succ)),p_res_fail = c(mean(p0_res_fail),mean(p1_res_fail),mean(p2_res_fail),mean(p3_res_fail)))
    pstds <- data.frame(p_values=c(0,1,2,3),p_cue=c(sd(p0_cue),sd(p1_cue),sd(p2_cue),sd(p3_cue)),p_res_succ=c(sd(p0_res_succ),sd(p1_res_succ),sd(p2_res_succ),sd(p3_res_succ)),p_res_fail=c(sd(p0_res_fail),sd(p1_res_fail),sd(p2_res_fail),sd(p3_res_fail)))
    
    avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
    std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='p_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=p_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("p_cue"="darkorchid4","p_res_succ"="forestgreen","p_res_fail"="darkred"))
    plt <- plt + labs(title=paste('Punishment: T',i,' corr, Unit ',p_unit_ind,sep=""),x="Punishment Number",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    ##########
    png(paste('corr_res_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    resavgs <- data.frame(res_values=c(0,1),res_cue = c(mean(res0_cue),mean(res1_cue)),res_res = c(mean(res0_res),mean(res1_res)))
    resstds <- data.frame(res_values=c(0,1),res_cue=c(sd(res0_cue),sd(res1_cue)),res_res=c(sd(res0_res),sd(res1_res)))
    
    avg_melt <- melt(resavgs,id="res_values",variable.name='type',value.name='avg')
    std_melt <- melt(resstds,id="res_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='res_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=res_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("res_cue"="coral4","res_res"="slateblue"))
    plt <- plt + labs(title=paste('Result: T',i,' corr, Unit ',res_unit_ind,sep=""),x="0: fail, 1: Succ",y="z-score") 
    #if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    ##########
    png(paste('corr_v_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    vavgs <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue = c(mean(v_3_cue),mean(v_2_cue),mean(v_1_cue),mean(v0_cue),mean(v1_cue),mean(v2_cue),mean(v3_cue)),v_res = c(mean(v_3_res),mean(v_2_res),mean(v_1_res),mean(v0_res),mean(v1_res),mean(v2_res),mean(v3_res)))
    vstds <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue=c(sd(v_3_cue),sd(v_2_cue),sd(v_1_cue),sd(v0_cue),sd(v1_cue),sd(v2_cue),sd(v3_cue)),v_res=c(sd(v_3_res),sd(v_2_res),sd(v_1_res),sd(v0_res),sd(v1_res),sd(v2_res),sd(v3_res)))
    
    avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
    std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='v_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=v_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("v_cue"="coral4","v_res"="slateblue"))
    plt <- plt + labs(title=paste('Value: T',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    #
    png(paste('corr_v_sf_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    vavgs <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue = c(mean(v_3_cue),mean(v_2_cue),mean(v_1_cue),mean(v0_cue),mean(v1_cue),mean(v2_cue),mean(v3_cue)),v_res_succ = c(mean(v_3_res_succ),mean(v_2_res_succ),mean(v_1_res_succ),mean(v0_res_succ),mean(v1_res_succ),mean(v2_res_succ),mean(v3_res_succ)),v_res_fail = c(mean(v_3_res_fail),mean(v_2_res_fail),mean(v_1_res_fail),mean(v0_res_fail),mean(v1_res_fail),mean(v2_res_fail),mean(v3_res_fail)))
    vstds <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue=c(sd(v_3_cue),sd(v_2_cue),sd(v_1_cue),sd(v0_cue),sd(v1_cue),sd(v2_cue),sd(v3_cue)),v_res_succ=c(sd(v_3_res_succ),sd(v_2_res_succ),sd(v_1_res_succ),sd(v0_res_succ),sd(v1_res_succ),sd(v2_res_succ),sd(v3_res_succ)),v_res_fail=c(sd(v_3_res_fail),sd(v_2_res_fail),sd(v_1_res_fail),sd(v0_res_fail),sd(v1_res_fail),sd(v2_res_fail),sd(v3_res_fail)))
    
    avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
    std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='v_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=v_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("v_cue"="darkorchid4","v_res_succ"="forestgreen","v_res_fail"="darkred"))
    plt <- plt + labs(title=paste('Value: T',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    ##########
    png(paste('corr_m_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    mavgs <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue = c(mean(m0_cue),mean(m1_cue),mean(m2_cue),mean(m3_cue),mean(m4_cue),mean(m5_cue),mean(m6_cue)),m_res = c(mean(m0_res),mean(m1_res),mean(m2_res),mean(m3_res),mean(m4_res),mean(m5_res),mean(m6_res)))
    mstds <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue=c(sd(m0_cue),sd(m1_cue),sd(m2_cue),sd(m3_cue),sd(m4_cue),sd(m5_cue),sd(m6_cue)),m_res=c(sd(m0_res),sd(m1_res),sd(m2_res),sd(m3_res),sd(m4_res),sd(m5_res),sd(m6_res)))
    
    avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
    std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='m_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=m_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("m_cue"="coral4","m_res"="slateblue"))
    plt <- plt + labs(title=paste('Motivation: T',i,' corr, Unit ',mtv_unit_ind,sep=""),x="Motivation",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()  
    
    #
    png(paste('corr_m_sf_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    mavgs <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue = c(mean(m0_cue),mean(m1_cue),mean(m2_cue),mean(m3_cue),mean(m4_cue),mean(m5_cue),mean(m6_cue)),m_res_succ = c(mean(m0_res_succ),mean(m1_res_succ),mean(m2_res_succ),mean(m3_res_succ),mean(m4_res_succ),mean(m5_res_succ),mean(m6_res_succ)),m_res_fail = c(mean(m0_res_fail),mean(m1_res_fail),mean(m2_res_fail),mean(m3_res_fail),mean(m4_res_fail),mean(m5_res_fail),mean(m6_res_fail)))
    mstds <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue=c(sd(m0_cue),sd(m1_cue),sd(m2_cue),sd(m3_cue),sd(m4_cue),sd(m5_cue),sd(m6_cue)),m_res_succ=c(sd(m0_res_succ),sd(m1_res_succ),sd(m2_res_succ),sd(m3_res_succ),sd(m4_res),sd(m5_res_succ),sd(m6_res_succ)),m_res_fail=c(sd(m0_res_fail),sd(m1_res_fail),sd(m2_res_fail),sd(m3_res_fail),sd(m4_res),sd(m5_res_fail),sd(m6_res_fail)))
    
    avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
    std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='m_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=m_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("m_cue"="darkorchid4","m_res_succ"="forestgreen","m_res_fail"="darkred"))
    plt <- plt + labs(title=paste('Motivation: T',i,' corr, Unit ',mtv_unit_ind,sep=""),x="Motivation",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()  
   
    ##############
    ##
    png(paste('corr_catch_mult_r_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    cm_r_avgs <- data.frame(cm_values=c(0,1,2,3),catch_r_res = c(NA,mean(catch1_res),mean(catch2_res),mean(catch3_res)),r_res_succ = c(mean(r0_res_succ),mean(r1_res_succ),mean(r2_res_succ),mean(r3_res_succ)),r_res_fail = c(mean(r0_res_fail),mean(r1_res_fail),mean(r2_res_fail),mean(r3_res_fail)))
    cm_r_stds <- data.frame(cm_values=c(0,1,2,3),catch_r_res = c(NA,sd(catch1_res),sd(catch2_res),sd(catch3_res)),r_res_succ = c(sd(r0_res_succ),sd(r1_res_succ),sd(r2_res_succ),sd(r3_res_succ)),r_res_fail = c(sd(r0_res_fail),sd(r1_res_fail),sd(r2_res_fail),sd(r3_res_fail)))

    avg_melt <- melt(cm_r_avgs,id="cm_values",variable.name='type',value.name='avg')
    std_melt <- melt(cm_r_stds,id="cm_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='cm_values')
    
    plt <- ggplot(test,aes(x=cm_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("catch_r_res"="slateblue4","r_res_succ"="forestgreen","r_res_fail"="gray50"))
    plt <- plt + labs(title=paste('Catch mult: T',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    #p levels catch
    png(paste('corr_catch_mult_p_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    cm_p_avgs <- data.frame(cm_values=c(0,1,2,3),catch_p_res = c(NA,mean(catch_1_res),mean(catch_2_res),mean(catch_3_res)),p_res_succ = c(mean(p0_res_succ),mean(p1_res_succ),mean(p2_res_succ),mean(p3_res_succ)),p_res_fail = c(mean(p0_res_fail),mean(p1_res_fail),mean(p2_res_fail),mean(p3_res_fail)))
    cm_p_stds <- data.frame(cm_values=c(0,1,2,3),catch_p_res = c(NA,sd(catch_1_res),sd(catch_2_res),sd(catch_3_res)),p_res_succ = c(sd(p0_res_succ),sd(p1_res_succ),sd(p2_res_succ),sd(p3_res_succ)),p_res_fail = c(sd(p0_res_fail),sd(p1_res_fail),sd(p2_res_fail),sd(p3_res_fail)))
    
    avg_melt <- melt(cm_p_avgs,id="cm_values",variable.name='type',value.name='avg')
    std_melt <- melt(cm_p_stds,id="cm_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='cm_values')
    
    plt <- ggplot(test,aes(x=cm_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("catch_p_res"="slateblue4","p_res_succ"="gray50","p_res_fail"="firebrick"))
    plt <- plt + labs(title=paste('Catch mult: T',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    ## catch bin r
    png(paste('corr_catch_bin_r_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    cm_r_avgs <- data.frame(cm_values=c(0,1),catch_r_res = c(NA,mean(catchx_res)),r_res_succ = c(mean(r0_res_succ),mean(c(r1_res_succ,r2_res_succ,r3_res_succ))),r_res_fail = c(mean(r0_res_fail),mean(c(r1_res_fail,r2_res_fail,r3_res_fail))))
    cm_r_stds <- data.frame(cm_values=c(0,1),catch_r_res = c(NA,sd(catchx_res)),r_res_succ = c(sd(r0_res_succ),sd(c(r1_res_succ,r2_res_succ,r3_res_succ))),r_res_fail = c(sd(r0_res_fail),sd(c(r1_res_fail,r2_res_fail,r3_res_fail))))
    
    avg_melt <- melt(cm_r_avgs,id="cm_values",variable.name='type',value.name='avg')
    std_melt <- melt(cm_r_stds,id="cm_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='cm_values')
    
    plt <- ggplot(test,aes(x=cm_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("catch_r_res"="slateblue4","r_res_succ"="forestgreen","r_res_fail"="gray50"))
    plt <- plt + labs(title=paste('Catch binary: T',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="z-score") 
    plot(plt)
    graphics.off()
    
    ## catch bin p
    png(paste('corr_catch_bin_p_t',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    cm_p_avgs <- data.frame(cm_values=c(0,1),catch_p_res = c(NA,mean(catch_x_res)),p_res_succ = c(mean(p0_res_succ),mean(c(p1_res_succ,p2_res_succ,p3_res_succ))),p_res_fail = c(mean(p0_res_fail),mean(c(p1_res_fail,p2_res_fail,p3_res_fail))))
    cm_p_stds <- data.frame(cm_values=c(0,1),catch_p_res = c(NA,sd(catch_x_res)),p_res_succ = c(sd(p0_res_succ),sd(c(p1_res_succ,p2_res_succ,p3_res_succ))),p_res_fail = c(sd(p0_res_fail),sd(c(p1_res_fail,p2_res_fail,p3_res_fail))))
    
    avg_melt <- melt(cm_p_avgs,id="cm_values",variable.name='type',value.name='avg')
    std_melt <- melt(cm_p_stds,id="cm_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='cm_values')
    
    plt <- ggplot(test,aes(x=cm_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("catch_p_res"="slateblue4","p_res_succ"="gray50","p_res_fail"="firebrick"))
    plt <- plt + labs(title=paste('Catch binary: T',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="z-score") 
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
    catch_mult_unit_ind <- which((length(catch_mult_corr_order)-i)==catch_mult_corr_order)
    catch_bin_unit_ind <- which((length(catch_bin_corr_order)-i)==catch_bin_corr_order)
    
    r0_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==0)]
    r1_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==1)]
    r2_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==2)]
    r3_cue <- cue_avg[r_unit_ind,][which(condensed[,4]==3)]
    r0_res <- res_avg[r_unit_ind,][which(condensed[,4]==0)]
    r1_res <- res_avg[r_unit_ind,][which(condensed[,4]==1)]
    r2_res <- res_avg[r_unit_ind,][which(condensed[,4]==2)]
    r3_res <- res_avg[r_unit_ind,][which(condensed[,4]==3)]
    
    r0_res_succ <- res_avg[r_unit_ind,][which(condensed[,4]==0 & condensed[,6]==1)]
    r1_res_succ <- res_avg[r_unit_ind,][which(condensed[,4]==1 & condensed[,6]==1)]
    r2_res_succ <- res_avg[r_unit_ind,][which(condensed[,4]==2 & condensed[,6]==1)]
    r3_res_succ <- res_avg[r_unit_ind,][which(condensed[,4]==3 & condensed[,6]==1)]
    r0_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==0 & condensed[,6]==0)]
    r1_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==1 & condensed[,6]==0)]
    r2_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==2 & condensed[,6]==0)]
    r3_res_fail <- res_avg[r_unit_ind,][which(condensed[,4]==3 & condensed[,6]==0)]
    
    p0_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==0)]
    p1_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==1)]
    p2_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==2)]
    p3_cue <- cue_avg[p_unit_ind,][which(condensed[,5]==3)]
    p0_res <- res_avg[p_unit_ind,][which(condensed[,5]==0)]
    p1_res <- res_avg[p_unit_ind,][which(condensed[,5]==1)]
    p2_res <- res_avg[p_unit_ind,][which(condensed[,5]==2)]
    p3_res <- res_avg[p_unit_ind,][which(condensed[,5]==3)]
    
    p0_res_succ <- res_avg[p_unit_ind,][which(condensed[,5]==0 & condensed[,6]==1)]
    p1_res_succ <- res_avg[p_unit_ind,][which(condensed[,5]==1 & condensed[,6]==1)]
    p2_res_succ <- res_avg[p_unit_ind,][which(condensed[,5]==2 & condensed[,6]==1)]
    p3_res_succ <- res_avg[p_unit_ind,][which(condensed[,5]==3 & condensed[,6]==1)]
    p0_res_fail <- res_avg[p_unit_ind,][which(condensed[,5]==0 & condensed[,6]==0)]
    p1_res_fail <- res_avg[p_unit_ind,][which(condensed[,5]==1 & condensed[,6]==0)]
    p2_res_fail <- res_avg[p_unit_ind,][which(condensed[,5]==2 & condensed[,6]==0)]
    p3_res_fail <- res_avg[p_unit_ind,][which(condensed[,5]==3 & condensed[,6]==0)]
    
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
    
    v_3_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]==-3 & condensed[,6]==1)]
    v_2_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]==-2 & condensed[,6]==1)]
    v_1_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]== 1 & condensed[,6]==1)]
    v0_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]==0 & condensed[,6]==1)]
    v1_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]==1 & condensed[,6]==1)]
    v2_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]==2 & condensed[,6]==1)]
    v3_res_succ <- res_avg[val_unit_ind,][which(condensed[,7]==3 & condensed[,6]==1)] 
    v_3_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]==-3 & condensed[,6]==0)]
    v_2_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]==-2 & condensed[,6]==0)]
    v_1_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]== 1 & condensed[,6]==0)]
    v0_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]==0 & condensed[,6]==0)]
    v1_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]==1 & condensed[,6]==0)]
    v2_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]==2 & condensed[,6]==0)]
    v3_res_fail <- res_avg[val_unit_ind,][which(condensed[,7]==3 & condensed[,6]==0)] 
    
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
    
    m0_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==0 & condensed[,6]==1)]
    m1_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==1 & condensed[,6]==1)]
    m2_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==2 & condensed[,6]==1)]
    m3_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==3 & condensed[,6]==1)]
    m4_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==4 & condensed[,6]==1)]
    m5_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==5 & condensed[,6]==1)]
    m6_res_succ <- res_avg[mtv_unit_ind,][which(condensed[,8]==6 & condensed[,6]==1)] 
    m0_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==0 & condensed[,6]==0)]
    m1_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==1 & condensed[,6]==0)]
    m2_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==2 & condensed[,6]==0)]
    m3_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==3 & condensed[,6]==0)]
    m4_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==4 & condensed[,6]==0)]
    m5_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==5 & condensed[,6]==0)]
    m6_res_fail <- res_avg[mtv_unit_ind,][which(condensed[,8]==6 & condensed[,6]==0)] 
    
    
    catch_3_cue <- cue_avg[catch_mult_unit_ind,][which(condensed[,12]==-3)]
    catch_2_cue <- cue_avg[catch_mult_unit_ind,][which(condensed[,12]==-2)]
    catch_1_cue <- cue_avg[catch_mult_unit_ind,][which(condensed[,12]==-1)]
    catch1_cue <- cue_avg[catch_mult_unit_ind,][which(condensed[,12]==1)]
    catch2_cue <- cue_avg[catch_mult_unit_ind,][which(condensed[,12]==2)]
    catch3_cue <- cue_avg[catch_mult_unit_ind,][which(condensed[,12]==3)]
    catch_3_res <- res_avg[catch_mult_unit_ind,][which(condensed[,12]==-3)]
    catch_2_res <- res_avg[catch_mult_unit_ind,][which(condensed[,12]==-2)]
    catch_1_res <- res_avg[catch_mult_unit_ind,][which(condensed[,12]==-1)]
    catch1_res <- res_avg[catch_mult_unit_ind,][which(condensed[,12]==1)]
    catch2_res <- res_avg[catch_mult_unit_ind,][which(condensed[,12]==2)]
    catch3_res <- res_avg[catch_mult_unit_ind,][which(condensed[,12]==3)]    
    
    catch_x_cue <- cue_avg[catch_bin_unit_ind,][which(condensed[,12]<=-1)]
    catchx_cue <- cue_avg[catch_bin_unit_ind,][which(condensed[,12]>=1)]
    catch_x_res <- res_avg[catch_bin_unit_ind,][which(condensed[,12]<=-1)]
    catchx_res <- res_avg[catch_bin_unit_ind,][which(condensed[,12]>=1)]
    
    
    #######
    r0_cue_total_b <- c(r0_cue_total_b,r0_cue)
    r1_cue_total_b <- c(r1_cue_total_b,r1_cue)
    r2_cue_total_b <- c(r2_cue_total_b,r2_cue)
    r3_cue_total_b <- c(r3_cue_total_b,r3_cue)
    r0_res_total_b <- c(r0_res_total_b,r0_res)
    r1_res_total_b <- c(r1_res_total_b,r1_res)
    r2_res_total_b <- c(r2_res_total_b,r2_res)
    r3_res_total_b <- c(r3_res_total_b,r3_res)
    r0_res_succ_total_b <- c(r0_res_succ_total_b,r0_res_succ)
    r1_res_succ_total_b <- c(r1_res_succ_total_b,r1_res_succ)
    r2_res_succ_total_b <- c(r2_res_succ_total_b,r2_res_succ)
    r3_res_succ_total_b <- c(r3_res_succ_total_b,r3_res_succ)
    r0_res_fail_total_b <- c(r0_res_fail_total_b,r0_res_fail)
    r1_res_fail_total_b <- c(r1_res_fail_total_b,r1_res_fail)
    r2_res_fail_total_b <- c(r2_res_fail_total_b,r2_res_fail)
    r3_res_fail_total_b <- c(r3_res_fail_total_b,r3_res_fail)
    
    p0_cue_total_b <- c(p0_cue_total_b,p0_cue)
    p1_cue_total_b <- c(p1_cue_total_b,p1_cue)
    p2_cue_total_b <- c(p2_cue_total_b,p2_cue)
    p3_cue_total_b <- c(p3_cue_total_b,p3_cue)
    p0_res_total_b <- c(p0_res_total_b,p0_res)
    p1_res_total_b <- c(p1_res_total_b,p1_res)
    p2_res_total_b <- c(p2_res_total_b,p2_res)
    p3_res_total_b <- c(p3_res_total_b,p3_res)
    p0_res_succ_total_b <- c(p0_res_succ_total_b,p0_res_succ)
    p1_res_succ_total_b <- c(p1_res_succ_total_b,p1_res_succ)
    p2_res_succ_total_b <- c(p2_res_succ_total_b,p2_res_succ)
    p3_res_succ_total_b <- c(p3_res_succ_total_b,p3_res_succ)
    p0_res_fail_total_b <- c(p0_res_fail_total_b,p0_res_fail)
    p1_res_fail_total_b <- c(p1_res_fail_total_b,p1_res_fail)
    p2_res_fail_total_b <- c(p2_res_fail_total_b,p2_res_fail)
    p3_res_fail_total_b <- c(p3_res_fail_total_b,p3_res_fail)
    
    res0_cue_total_b <- c(res0_cue_total_b,res0_cue)
    res1_cue_total_b <- c(res1_cue_total_b,res1_cue)
    res0_res_total_b <- c(res0_res_total_b,res0_res)
    res1_res_total_b <- c(res1_res_total_b,res1_res)

    v_3_cue_total_b <- c(v_3_cue_total_b,v_3_cue)
    v_2_cue_total_b <- c(v_2_cue_total_b,v_2_cue)
    v_1_cue_total_b <- c(v_1_cue_total_b,v_1_cue)
    v0_cue_total_b <- c(v0_cue_total_b,v0_cue)
    v1_cue_total_b <- c(v1_cue_total_b,v1_cue)
    v2_cue_total_b <- c(v2_cue_total_b,v2_cue)
    v3_cue_total_b <- c(v3_cue_total_b,v3_cue)
    v_3_res_total_b <- c(v_3_res_total_b,v_3_res)
    v_2_res_total_b <- c(v_2_res_total_b,v_2_res)
    v_1_res_total_b <- c(v_1_res_total_b,v_1_res)
    v0_res_total_b <- c(v0_res_total_b,v0_res)
    v1_res_total_b <- c(v1_res_total_b,v1_res)
    v2_res_total_b <- c(v2_res_total_b,v2_res)
    v3_res_total_b <- c(v3_res_total_b,v3_res)
    v_3_res_succ_total_b <- c(v_3_res_succ_total_b,v_3_res_succ)
    v_2_res_succ_total_b <- c(v_2_res_succ_total_b,v_2_res_succ)
    v_1_res_succ_total_b <- c(v_1_res_succ_total_b,v_1_res_succ)
    v0_res_succ_total_b <- c(v0_res_succ_total_b,v0_res_succ)
    v1_res_succ_total_b <- c(v1_res_succ_total_b,v1_res_succ)
    v2_res_succ_total_b <- c(v2_res_succ_total_b,v2_res_succ)
    v3_res_succ_total_b <- c(v3_res_succ_total_b,v3_res_succ)
    v_3_res_fail_total_b <- c(v_3_res_fail_total_b,v_3_res_fail)
    v_2_res_fail_total_b <- c(v_2_res_fail_total_b,v_2_res_fail)
    v_1_res_fail_total_b <- c(v_1_res_fail_total_b,v_1_res_fail)
    v0_res_fail_total_b <- c(v0_res_fail_total_b,v0_res_fail)
    v1_res_fail_total_b <- c(v1_res_fail_total_b,v1_res_fail)
    v2_res_fail_total_b <- c(v2_res_fail_total_b,v2_res_fail)
    v3_res_fail_total_b <- c(v3_res_fail_total_b,v3_res_fail)
    
    m0_cue_total_b <- c(m0_cue_total_b,m0_cue)
    m1_cue_total_b <- c(m1_cue_total_b,m1_cue)
    m2_cue_total_b <- c(m2_cue_total_b,m2_cue)
    m3_cue_total_b <- c(m3_cue_total_b,m3_cue)
    m4_cue_total_b <- c(m4_cue_total_b,m4_cue)
    m5_cue_total_b <- c(m5_cue_total_b,m5_cue)
    m6_cue_total_b <- c(m6_cue_total_b,m6_cue)
    m0_res_total_b <- c(m0_res_total_b,m0_res)
    m1_res_total_b <- c(m1_res_total_b,m1_res)
    m2_res_total_b <- c(m2_res_total_b,m2_res)
    m3_res_total_b <- c(m3_res_total_b,m3_res)
    m4_res_total_b <- c(m4_res_total_b,m4_res)
    m5_res_total_b <- c(m5_res_total_b,m5_res)
    m6_res_total_b <- c(m6_res_total_b,m6_res)
    m0_res_succ_total_b <- c(m0_res_succ_total_b,m0_res_succ)
    m1_res_succ_total_b <- c(m1_res_succ_total_b,m1_res_succ)
    m2_res_succ_total_b <- c(m2_res_succ_total_b,m2_res_succ)
    m3_res_succ_total_b <- c(m3_res_succ_total_b,m3_res_succ)
    m4_res_succ_total_b <- c(m4_res_succ_total_b,m4_res_succ)
    m5_res_succ_total_b <- c(m5_res_succ_total_b,m5_res_succ)
    m6_res_succ_total_b <- c(m6_res_succ_total_b,m6_res_succ)
    m0_res_fail_total_b <- c(m0_res_fail_total_b,m0_res_fail)
    m1_res_fail_total_b <- c(m1_res_fail_total_b,m1_res_fail)
    m2_res_fail_total_b <- c(m2_res_fail_total_b,m2_res_fail)
    m3_res_fail_total_b <- c(m3_res_fail_total_b,m3_res_fail)
    m4_res_fail_total_b <- c(m4_res_fail_total_b,m4_res_fail)
    m5_res_fail_total_b <- c(m5_res_fail_total_b,m5_res_fail)
    m6_res_fail_total_b <- c(m6_res_fail_total_b,m6_res_fail)
    
    ##########
    png(paste('corr_r_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    ravgs <- data.frame(r_values=c(0,1,2,3),r_cue = c(mean(r0_cue),mean(r1_cue),mean(r2_cue),mean(r3_cue)),r_res = c(mean(r0_res),mean(r1_res),mean(r2_res),mean(r3_res)))
    rstds <- data.frame(r_values=c(0,1,2,3),r_cue=c(sd(r0_cue),sd(r1_cue),sd(r2_cue),sd(r3_cue)),r_res=c(sd(r0_res),sd(r1_res),sd(r2_res),sd(r3_res)))
    
    avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
    std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='r_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=r_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("r_cue"="coral4","r_res"="slateblue"))
    plt <- plt + labs(title=paste('Reward: B',i,' corr, Unit ',r_unit_ind,sep=""),x="Reward Number",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    ##
    png(paste('corr_r_sf_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    ravgs <- data.frame(r_values=c(0,1,2,3),r_cue = c(mean(r0_cue),mean(r1_cue),mean(r2_cue),mean(r3_cue)),r_res_succ = c(mean(r0_res_succ),mean(r1_res_succ),mean(r2_res_succ),mean(r3_res_succ)),r_res_fail = c(mean(r0_res_fail),mean(r1_res_fail),mean(r2_res_fail),mean(r3_res_fail)))
    rstds <- data.frame(r_values=c(0,1,2,3),r_cue=c(sd(r0_cue),sd(r1_cue),sd(r2_cue),sd(r3_cue)),r_res_succ=c(sd(r0_res_succ),sd(r1_res_succ),sd(r2_res_succ),sd(r3_res_succ)),r_res_fail=c(sd(r0_res_fail),sd(r1_res_fail),sd(r2_res_fail),sd(r3_res_fail)))
    
    avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
    std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='r_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=r_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("r_cue"="darkorchid4","r_res_succ"="forestgreen","r_res_fail"="darkred"))
    plt <- plt + labs(title=paste('Reward: B',i,' corr, Unit ',r_unit_ind,sep=""),x="Reward Number",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    #############
    png(paste('corr_p_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    pavgs <- data.frame(p_values=c(0,1,2,3),p_cue=c(mean(p0_cue),mean(p1_cue),mean(p2_cue),mean(p3_cue)),p_res = c(mean(p0_res),mean(p1_res),mean(p2_res),mean(p3_res)))
    pstds <- data.frame(p_values=c(0,1,2,3),p_cue=c(sd(p0_cue),sd(p1_cue),sd(p2_cue),sd(p3_cue)),p_res=c(sd(p0_res),sd(p1_res),sd(p2_res),sd(p3_res)))
    
    avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
    std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='p_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=p_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("p_cue"="coral4","p_res"="slateblue"))
    plt <- plt + labs(title=paste('Punishment: B',i,' corr, Unit ',p_unit_ind,sep=""),x="Punishment Number",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    #
    png(paste('corr_p_sf_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    pavgs <- data.frame(p_values=c(0,1,2,3),p_cue=c(mean(p0_cue),mean(p1_cue),mean(p2_cue),mean(p3_cue)),p_res_succ = c(mean(p0_res_succ),mean(p1_res_succ),mean(p2_res_succ),mean(p3_res_succ)),p_res_fail = c(mean(p0_res_fail),mean(p1_res_fail),mean(p2_res_fail),mean(p3_res_fail)))
    pstds <- data.frame(p_values=c(0,1,2,3),p_cue=c(sd(p0_cue),sd(p1_cue),sd(p2_cue),sd(p3_cue)),p_res_succ=c(sd(p0_res_succ),sd(p1_res_succ),sd(p2_res_succ),sd(p3_res_succ)),p_res_fail=c(sd(p0_res_fail),sd(p1_res_fail),sd(p2_res_fail),sd(p3_res_fail)))
    
    avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
    std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='p_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=p_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("p_cue"="darkorchid4","p_res_succ"="forestgreen","p_res_fail"="darkred"))
    plt <- plt + labs(title=paste('Punishment: B',i,' corr, Unit ',p_unit_ind,sep=""),x="Punishment Number",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    ##########
    png(paste('corr_res_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    resavgs <- data.frame(res_values=c(0,1),res_cue = c(mean(res0_cue),mean(res1_cue)),res_res = c(mean(res0_res),mean(res1_res)))
    resstds <- data.frame(res_values=c(0,1),res_cue=c(sd(res0_cue),sd(res1_cue)),res_res=c(sd(res0_res),sd(res1_res)))
    
    avg_melt <- melt(resavgs,id="res_values",variable.name='type',value.name='avg')
    std_melt <- melt(resstds,id="res_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='res_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=res_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("res_cue"="coral4","res_res"="slateblue"))
    plt <- plt + labs(title=paste('Result: B',i,' corr, Unit ',res_unit_ind,sep=""),x="0: fail, 1: Succ",y="z-score") 
    #if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    ##########
    png(paste('corr_v_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    vavgs <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue = c(mean(v_3_cue),mean(v_2_cue),mean(v_1_cue),mean(v0_cue),mean(v1_cue),mean(v2_cue),mean(v3_cue)),v_res = c(mean(v_3_res),mean(v_2_res),mean(v_1_res),mean(v0_res),mean(v1_res),mean(v2_res),mean(v3_res)))
    vstds <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue=c(sd(v_3_cue),sd(v_2_cue),sd(v_1_cue),sd(v0_cue),sd(v1_cue),sd(v2_cue),sd(v3_cue)),v_res=c(sd(v_3_res),sd(v_2_res),sd(v_1_res),sd(v0_res),sd(v1_res),sd(v2_res),sd(v3_res)))
    
    avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
    std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='v_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=v_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("v_cue"="coral4","v_res"="slateblue"))
    plt <- plt + labs(title=paste('Value: B',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    #
    png(paste('corr_v_sf_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    vavgs <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue = c(mean(v_3_cue),mean(v_2_cue),mean(v_1_cue),mean(v0_cue),mean(v1_cue),mean(v2_cue),mean(v3_cue)),v_res_succ = c(mean(v_3_res_succ),mean(v_2_res_succ),mean(v_1_res_succ),mean(v0_res_succ),mean(v1_res_succ),mean(v2_res_succ),mean(v3_res_succ)),v_res_fail = c(mean(v_3_res_fail),mean(v_2_res_fail),mean(v_1_res_fail),mean(v0_res_fail),mean(v1_res_fail),mean(v2_res_fail),mean(v3_res_fail)))
    vstds <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue=c(sd(v_3_cue),sd(v_2_cue),sd(v_1_cue),sd(v0_cue),sd(v1_cue),sd(v2_cue),sd(v3_cue)),v_res_succ=c(sd(v_3_res_succ),sd(v_2_res_succ),sd(v_1_res_succ),sd(v0_res_succ),sd(v1_res_succ),sd(v2_res_succ),sd(v3_res_succ)),v_res_fail=c(sd(v_3_res_fail),sd(v_2_res_fail),sd(v_1_res_fail),sd(v0_res_fail),sd(v1_res_fail),sd(v2_res_fail),sd(v3_res_fail)))
    
    avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
    std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='v_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=v_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("v_cue"="darkorchid4","v_res_succ"="forestgreen","v_res_fail"="darkred"))
    plt <- plt + labs(title=paste('Value: B',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    
    ##########
    png(paste('corr_m_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    mavgs <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue = c(mean(m0_cue),mean(m1_cue),mean(m2_cue),mean(m3_cue),mean(m4_cue),mean(m5_cue),mean(m6_cue)),m_res = c(mean(m0_res),mean(m1_res),mean(m2_res),mean(m3_res),mean(m4_res),mean(m5_res),mean(m6_res)))
    mstds <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue=c(sd(m0_cue),sd(m1_cue),sd(m2_cue),sd(m3_cue),sd(m4_cue),sd(m5_cue),sd(m6_cue)),m_res=c(sd(m0_res),sd(m1_res),sd(m2_res),sd(m3_res),sd(m4_res),sd(m5_res),sd(m6_res)))
    
    avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
    std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='m_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=m_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("m_cue"="coral4","m_res"="slateblue"))
    plt <- plt + labs(title=paste('Motivation: B',i,' corr, Unit ',mtv_unit_ind,sep=""),x="Motivation",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()  
    
    #
    png(paste('corr_m_sf_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    mavgs <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue = c(mean(m0_cue),mean(m1_cue),mean(m2_cue),mean(m3_cue),mean(m4_cue),mean(m5_cue),mean(m6_cue)),m_res_succ = c(mean(m0_res_succ),mean(m1_res_succ),mean(m2_res_succ),mean(m3_res_succ),mean(m4_res_succ),mean(m5_res_succ),mean(m6_res_succ)),m_res_fail = c(mean(m0_res_fail),mean(m1_res_fail),mean(m2_res_fail),mean(m3_res_fail),mean(m4_res_fail),mean(m5_res_fail),mean(m6_res_fail)))
    mstds <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue=c(sd(m0_cue),sd(m1_cue),sd(m2_cue),sd(m3_cue),sd(m4_cue),sd(m5_cue),sd(m6_cue)),m_res_succ=c(sd(m0_res_succ),sd(m1_res_succ),sd(m2_res_succ),sd(m3_res_succ),sd(m4_res),sd(m5_res_succ),sd(m6_res_succ)),m_res_fail=c(sd(m0_res_fail),sd(m1_res_fail),sd(m2_res_fail),sd(m3_res_fail),sd(m4_res),sd(m5_res_fail),sd(m6_res_fail)))
    
    avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
    std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='m_values')
    #test[is.na(test)] <- 0
    
    plt <- ggplot(test,aes(x=m_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("m_cue"="darkorchid4","m_res_succ"="forestgreen","m_res_fail"="darkred"))
    plt <- plt + labs(title=paste('Motivation: B',i,' corr, Unit ',mtv_unit_ind,sep=""),x="Motivation",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()  
    
    
    ##
    png(paste('corr_catch_mult_r_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    cm_r_avgs <- data.frame(cm_values=c(0,1,2,3),catch_r_res = c(NA,mean(catch1_res),mean(catch2_res),mean(catch3_res)),r_res_succ = c(mean(r0_res_succ),mean(r1_res_succ),mean(r2_res_succ),mean(r3_res_succ)),r_res_fail = c(mean(r0_res_fail),mean(r1_res_fail),mean(r2_res_fail),mean(r3_res_fail)))
    cm_r_stds <- data.frame(cm_values=c(0,1,2,3),catch_r_res = c(NA,sd(catch1_res),sd(catch2_res),sd(catch3_res)),r_res_succ = c(sd(r0_res_succ),sd(r1_res_succ),sd(r2_res_succ),sd(r3_res_succ)),r_res_fail = c(sd(r0_res_fail),sd(r1_res_fail),sd(r2_res_fail),sd(r3_res_fail)))
    
    avg_melt <- melt(cm_r_avgs,id="cm_values",variable.name='type',value.name='avg')
    std_melt <- melt(cm_r_stds,id="cm_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='cm_values')
    
    plt <- ggplot(test,aes(x=cm_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("catch_r_res"="slateblue4","r_res_succ"="forestgreen","r_res_fail"="gray50"))
    plt <- plt + labs(title=paste('Catch mult: B',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    #p levels catch
    png(paste('corr_catch_mult_p_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    cm_p_avgs <- data.frame(cm_values=c(0,1,2,3),catch_p_res = c(NA,mean(catch_1_res),mean(catch_2_res),mean(catch_3_res)),p_res_succ = c(mean(p0_res_succ),mean(p1_res_succ),mean(p2_res_succ),mean(p3_res_succ)),p_res_fail = c(mean(p0_res_fail),mean(p1_res_fail),mean(p2_res_fail),mean(p3_res_fail)))
    cm_p_stds <- data.frame(cm_values=c(0,1,2,3),catch_p_res = c(NA,sd(catch_1_res),sd(catch_2_res),sd(catch_3_res)),p_res_succ = c(sd(p0_res_succ),sd(p1_res_succ),sd(p2_res_succ),sd(p3_res_succ)),p_res_fail = c(sd(p0_res_fail),sd(p1_res_fail),sd(p2_res_fail),sd(p3_res_fail)))
    
    avg_melt <- melt(cm_p_avgs,id="cm_values",variable.name='type',value.name='avg')
    std_melt <- melt(cm_p_stds,id="cm_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='cm_values')
    
    plt <- ggplot(test,aes(x=cm_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("catch_p_res"="slateblue4","p_res_succ"="gray50","p_res_fail"="firebrick"))
    plt <- plt + labs(title=paste('Catch mult: B',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="z-score") 
    if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
    plot(plt)
    graphics.off()
    
    ## catch bin r
    png(paste('corr_catch_bin_r_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    cm_r_avgs <- data.frame(cm_values=c(0,1),catch_r_res = c(NA,mean(catchx_res)),r_res_succ = c(mean(r0_res_succ),mean(c(r1_res_succ,r2_res_succ,r3_res_succ))),r_res_fail = c(mean(r0_res_fail),mean(c(r1_res_fail,r2_res_fail,r3_res_fail))))
    cm_r_stds <- data.frame(cm_values=c(0,1),catch_r_res = c(NA,sd(catchx_res)),r_res_succ = c(sd(r0_res_succ),sd(c(r1_res_succ,r2_res_succ,r3_res_succ))),r_res_fail = c(sd(r0_res_fail),sd(c(r1_res_fail,r2_res_fail,r3_res_fail))))
    
    avg_melt <- melt(cm_r_avgs,id="cm_values",variable.name='type',value.name='avg')
    std_melt <- melt(cm_r_stds,id="cm_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='cm_values')
    
    plt <- ggplot(test,aes(x=cm_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("catch_r_res"="slateblue4","r_res_succ"="forestgreen","r_res_fail"="gray50"))
    plt <- plt + labs(title=paste('Catch binary: B',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="z-score") 
    plot(plt)
    graphics.off()
    
    ## catch bin p
    png(paste('corr_catch_bin_p_b',i,'_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
    
    cm_p_avgs <- data.frame(cm_values=c(0,1),catch_p_res = c(NA,mean(catch_x_res)),p_res_succ = c(mean(p0_res_succ),mean(c(p1_res_succ,p2_res_succ,p3_res_succ))),p_res_fail = c(mean(p0_res_fail),mean(c(p1_res_fail,p2_res_fail,p3_res_fail))))
    cm_p_stds <- data.frame(cm_values=c(0,1),catch_p_res = c(NA,sd(catch_x_res)),p_res_succ = c(sd(p0_res_succ),sd(c(p1_res_succ,p2_res_succ,p3_res_succ))),p_res_fail = c(sd(p0_res_fail),sd(c(p1_res_fail,p2_res_fail,p3_res_fail))))
    
    avg_melt <- melt(cm_p_avgs,id="cm_values",variable.name='type',value.name='avg')
    std_melt <- melt(cm_p_stds,id="cm_values",variable.name='type',value.name='std')
    
    test <- merge(std_melt,avg_melt,row.names='cm_values')
    
    plt <- ggplot(test,aes(x=cm_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
    plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("catch_p_res"="slateblue4","p_res_succ"="gray50","p_res_fail"="firebrick"))
    plt <- plt + labs(title=paste('Catch binary: B',i,' corr, Unit ',val_unit_ind,sep=""),x="Value",y="z-score") 
    plot(plt)
    graphics.off()
    
    
    
    
  }
  
  ######################
  ######################  
  ######################
  
  
  #plot avg over these units
  png(paste('corr_r_t_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  ravgs <- data.frame(r_values=c(0,1,2,3),r_cue_total_t = c(mean(r0_cue_total_t),mean(r1_cue_total_t),mean(r2_cue_total_t),mean(r3_cue_total_t)),r_res_total_t = c(mean(r0_res_total_t),mean(r1_res_total_t),mean(r2_res_total_t),mean(r3_res_total_t)))
  rstds <- data.frame(r_values=c(0,1,2,3),r_cue_total_t=c(sd(r0_cue_total_t),sd(r1_cue_total_t),sd(r2_cue_total_t),sd(r3_cue_total_t)),r_res_total_t=c(sd(r0_res_total_t),sd(r1_res_total_t),sd(r2_res_total_t),sd(r3_res_total_t)))
  
  avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
  std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='r_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=r_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("r_cue_total_t"="coral4","r_res_total_t"="slateblue"))
  plt <- plt + labs(title=paste(region_list[region_index],'Reward Top Corr Total'),x="Reward Number",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  ##
  png(paste('corr_r_sf_t_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  ravgs <- data.frame(r_values=c(0,1,2,3),r_cue_total_t = c(mean(r0_cue_total_t),mean(r1_cue_total_t),mean(r2_cue_total_t),mean(r3_cue_total_t)),r_res_succ_total_t = c(mean(r0_res_succ_total_t),mean(r1_res_succ_total_t),mean(r2_res_succ_total_t),mean(r3_res_succ_total_t)),r_res_fail_total_t = c(mean(r0_res_fail_total_t),mean(r1_res_fail_total_t),mean(r2_res_fail_total_t),mean(r3_res_fail_total_t)))
  rstds <- data.frame(r_values=c(0,1,2,3),r_cue_total_t=c(sd(r0_cue_total_t),sd(r1_cue_total_t),sd(r2_cue_total_t),sd(r3_cue_total_t)),r_res_succ_total_t=c(sd(r0_res_succ_total_t),sd(r1_res_succ_total_t),sd(r2_res_succ_total_t),sd(r3_res_succ_total_t)),r_res_fail_total_t=c(sd(r0_res_fail_total_t),sd(r1_res_fail_total_t),sd(r2_res_fail_total_t),sd(r3_res_fail_total_t)))
  
  avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
  std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='r_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=r_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("r_cue_total_t"="darkorchid4","r_res_succ_total_t"="forestgreen","r_res_fail_total_t"="darkred"))
  plt <- plt + labs(title=paste(region_list[region_index],'Reward Top Corr Total'),x="Reward Number",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  #############
  png(paste('corr_p_t_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  pavgs <- data.frame(p_values=c(0,1,2,3),p_cue_total_t=c(mean(p0_cue_total_t),mean(p1_cue_total_t),mean(p2_cue_total_t),mean(p3_cue_total_t)),p_res = c(mean(p0_res_total_t),mean(p1_res_total_t),mean(p2_res_total_t),mean(p3_res_total_t)))
  pstds <- data.frame(p_values=c(0,1,2,3),p_cue_total_t=c(sd(p0_cue_total_t),sd(p1_cue_total_t),sd(p2_cue_total_t),sd(p3_cue_total_t)),p_res=c(sd(p0_res_total_t),sd(p1_res_total_t),sd(p2_res_total_t),sd(p3_res_total_t)))
  
  avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
  std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='p_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=p_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("p_cue_total_t"="coral4","p_res"="slateblue"))
  plt <- plt + labs(title=paste(region_list[region_index],'Punishment Top Corr Total'),x="Punishment Number",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  #
  png(paste('corr_p_sf_t_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  pavgs <- data.frame(p_values=c(0,1,2,3),p_cue_total_t=c(mean(p0_cue_total_t),mean(p1_cue_total_t),mean(p2_cue_total_t),mean(p3_cue_total_t)),p_res_succ_total_t = c(mean(p0_res_succ_total_t),mean(p1_res_succ_total_t),mean(p2_res_succ_total_t),mean(p3_res_succ_total_t)),p_res_fail_total_t = c(mean(p0_res_fail_total_t),mean(p1_res_fail_total_t),mean(p2_res_fail_total_t),mean(p3_res_fail_total_t)))
  pstds <- data.frame(p_values=c(0,1,2,3),p_cue_total_t=c(sd(p0_cue_total_t),sd(p1_cue_total_t),sd(p2_cue_total_t),sd(p3_cue_total_t)),p_res_succ_total_t=c(sd(p0_res_succ_total_t),sd(p1_res_succ_total_t),sd(p2_res_succ_total_t),sd(p3_res_succ_total_t)),p_res_fail_total_t=c(sd(p0_res_fail_total_t),sd(p1_res_fail_total_t),sd(p2_res_fail_total_t),sd(p3_res_fail_total_t)))
  
  avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
  std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='p_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=p_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("p_cue_total_t"="darkorchid4","p_res_succ_total_t"="forestgreen","p_res_fail_total_t"="darkred"))
  plt <- plt + labs(title=paste(region_list[region_index],'Punishment Top Corr Total'),x="Punishment Number",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  ##########
  png(paste('corr_res_t_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  resavgs <- data.frame(res_values=c(0,1),res_cue_total_t = c(mean(res0_cue_total_t),mean(res1_cue_total_t)),res_res = c(mean(res0_res_total_t),mean(res1_res_total_t)))
  resstds <- data.frame(res_values=c(0,1),res_cue_total_t=c(sd(res0_cue_total_t),sd(res1_cue_total_t)),res_res=c(sd(res0_res_total_t),sd(res1_res_total_t)))
  
  avg_melt <- melt(resavgs,id="res_values",variable.name='type',value.name='avg')
  std_melt <- melt(resstds,id="res_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='res_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=res_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("res_cue_total_t"="coral4","res_res"="slateblue"))
  plt <- plt + labs(title=paste(region_list[region_index],'Result Top Corr Total'),x="0: fail, 1: Succ",y="z-score") 
  #if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  ##########
  png(paste('corr_v_t_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  vavgs <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue_total_t = c(mean(v_3_cue_total_t),mean(v_2_cue_total_t),mean(v_1_cue_total_t),mean(v0_cue_total_t),mean(v1_cue_total_t),mean(v2_cue_total_t),mean(v3_cue_total_t)),v_res = c(mean(v_3_res_total_t),mean(v_2_res_total_t),mean(v_1_res_total_t),mean(v0_res_total_t),mean(v1_res_total_t),mean(v2_res_total_t),mean(v3_res_total_t)))
  vstds <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue_total_t=c(sd(v_3_cue_total_t),sd(v_2_cue_total_t),sd(v_1_cue_total_t),sd(v0_cue_total_t),sd(v1_cue_total_t),sd(v2_cue_total_t),sd(v3_cue_total_t)),v_res=c(sd(v_3_res_total_t),sd(v_2_res_total_t),sd(v_1_res_total_t),sd(v0_res_total_t),sd(v1_res_total_t),sd(v2_res_total_t),sd(v3_res_total_t)))
  
  avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
  std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='v_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=v_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("v_cue_total_t"="coral4","v_res"="slateblue"))
  plt <- plt + labs(title=paste(region_list[region_index],'Value Top Corr Total'),x="Value",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  #
  png(paste('corr_v_sf_t_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  vavgs <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue_total_t = c(mean(v_3_cue_total_t),mean(v_2_cue_total_t),mean(v_1_cue_total_t),mean(v0_cue_total_t),mean(v1_cue_total_t),mean(v2_cue_total_t),mean(v3_cue_total_t)),v_res_succ_total_t = c(mean(v_3_res_succ_total_t),mean(v_2_res_succ_total_t),mean(v_1_res_succ_total_t),mean(v0_res_succ_total_t),mean(v1_res_succ_total_t),mean(v2_res_succ_total_t),mean(v3_res_succ_total_t)),v_res_fail_total_t = c(mean(v_3_res_fail_total_t),mean(v_2_res_fail_total_t),mean(v_1_res_fail_total_t),mean(v0_res_fail_total_t),mean(v1_res_fail_total_t),mean(v2_res_fail_total_t),mean(v3_res_fail_total_t)))
  vstds <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue_total_t=c(sd(v_3_cue_total_t),sd(v_2_cue_total_t),sd(v_1_cue_total_t),sd(v0_cue_total_t),sd(v1_cue_total_t),sd(v2_cue_total_t),sd(v3_cue_total_t)),v_res_succ_total_t=c(sd(v_3_res_succ_total_t),sd(v_2_res_succ_total_t),sd(v_1_res_succ_total_t),sd(v0_res_succ_total_t),sd(v1_res_succ_total_t),sd(v2_res_succ_total_t),sd(v3_res_succ_total_t)),v_res_fail_total_t=c(sd(v_3_res_fail_total_t),sd(v_2_res_fail_total_t),sd(v_1_res_fail_total_t),sd(v0_res_fail_total_t),sd(v1_res_fail_total_t),sd(v2_res_fail_total_t),sd(v3_res_fail_total_t)))
  
  avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
  std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='v_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=v_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("v_cue_total_t"="darkorchid4","v_res_succ_total_t"="forestgreen","v_res_fail_total_t"="darkred"))
  plt <- plt + labs(title=paste(region_list[region_index],'Value Top Corr Total'),x="Value",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  ##########
  png(paste('corr_m_t_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  mavgs <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue_total_t = c(mean(m0_cue_total_t),mean(m1_cue_total_t),mean(m2_cue_total_t),mean(m3_cue_total_t),mean(m4_cue_total_t),mean(m5_cue_total_t),mean(m6_cue_total_t)),m_res = c(mean(m0_res_total_t),mean(m1_res_total_t),mean(m2_res_total_t),mean(m3_res_total_t),mean(m4_res_total_t),mean(m5_res_total_t),mean(m6_res_total_t)))
  mstds <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue_total_t=c(sd(m0_cue_total_t),sd(m1_cue_total_t),sd(m2_cue_total_t),sd(m3_cue_total_t),sd(m4_cue_total_t),sd(m5_cue_total_t),sd(m6_cue_total_t)),m_res=c(sd(m0_res_total_t),sd(m1_res_total_t),sd(m2_res_total_t),sd(m3_res_total_t),sd(m4_res_total_t),sd(m5_res_total_t),sd(m6_res_total_t)))
  
  avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
  std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='m_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=m_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("m_cue_total_t"="coral4","m_res"="slateblue"))
  plt <- plt + labs(title=paste(region_list[region_index],'Motivation Top Corr Total'),x="Motivation",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()  
  
  #
  png(paste('corr_m_sf_t_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  mavgs <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue_total_t = c(mean(m0_cue_total_t),mean(m1_cue_total_t),mean(m2_cue_total_t),mean(m3_cue_total_t),mean(m4_cue_total_t),mean(m5_cue_total_t),mean(m6_cue_total_t)),m_res_succ_total_t = c(mean(m0_res_succ_total_t),mean(m1_res_succ_total_t),mean(m2_res_succ_total_t),mean(m3_res_succ_total_t),mean(m4_res_succ_total_t),mean(m5_res_succ_total_t),mean(m6_res_succ_total_t)),m_res_fail_total_t = c(mean(m0_res_fail_total_t),mean(m1_res_fail_total_t),mean(m2_res_fail_total_t),mean(m3_res_fail_total_t),mean(m4_res_fail_total_t),mean(m5_res_fail_total_t),mean(m6_res_fail_total_t)))
  mstds <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue_total_t=c(sd(m0_cue_total_t),sd(m1_cue_total_t),sd(m2_cue_total_t),sd(m3_cue_total_t),sd(m4_cue_total_t),sd(m5_cue_total_t),sd(m6_cue_total_t)),m_res_succ_total_t=c(sd(m0_res_succ_total_t),sd(m1_res_succ_total_t),sd(m2_res_succ_total_t),sd(m3_res_succ_total_t),sd(m4_res_total_t),sd(m5_res_succ_total_t),sd(m6_res_succ_total_t)),m_res_fail_total_t=c(sd(m0_res_fail_total_t),sd(m1_res_fail_total_t),sd(m2_res_fail_total_t),sd(m3_res_fail_total_t),sd(m4_res_total_t),sd(m5_res_fail_total_t),sd(m6_res_fail_total_t)))
  
  avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
  std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='m_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=m_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("m_cue_total_t"="darkorchid4","m_res_succ_total_t"="forestgreen","m_res_fail_total_t"="darkred"))
  plt <- plt + labs(paste(region_list[region_index],'Motivation Top Corr Total'),x="Motivation",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  ##### btm
  png(paste('corr_r_b_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  ravgs <- data.frame(r_values=c(0,1,2,3),r_cue_total_b = c(mean(r0_cue_total_b),mean(r1_cue_total_b),mean(r2_cue_total_b),mean(r3_cue_total_b)),r_res = c(mean(r0_res_total_b),mean(r1_res_total_b),mean(r2_res_total_b),mean(r3_res_total_b)))
  rstds <- data.frame(r_values=c(0,1,2,3),r_cue_total_b=c(sd(r0_cue_total_b),sd(r1_cue_total_b),sd(r2_cue_total_b),sd(r3_cue_total_b)),r_res=c(sd(r0_res_total_b),sd(r1_res_total_b),sd(r2_res_total_b),sd(r3_res_total_b)))
  
  avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
  std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='r_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=r_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("r_cue_total_b"="coral4","r_res"="slateblue"))
  plt <- plt + labs(title=paste(region_list[region_index],'Reward Bottom Corr Total'),x="Reward Number",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  ##
  png(paste('corr_r_sf_b_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  ravgs <- data.frame(r_values=c(0,1,2,3),r_cue_total_b = c(mean(r0_cue_total_b),mean(r1_cue_total_b),mean(r2_cue_total_b),mean(r3_cue_total_b)),r_res_succ_total_b = c(mean(r0_res_succ_total_b),mean(r1_res_succ_total_b),mean(r2_res_succ_total_b),mean(r3_res_succ_total_b)),r_res_fail_total_b = c(mean(r0_res_fail_total_b),mean(r1_res_fail_total_b),mean(r2_res_fail_total_b),mean(r3_res_fail_total_b)))
  rstds <- data.frame(r_values=c(0,1,2,3),r_cue_total_b=c(sd(r0_cue_total_b),sd(r1_cue_total_b),sd(r2_cue_total_b),sd(r3_cue_total_b)),r_res_succ_total_b=c(sd(r0_res_succ_total_b),sd(r1_res_succ_total_b),sd(r2_res_succ_total_b),sd(r3_res_succ_total_b)),r_res_fail_total_b=c(sd(r0_res_fail_total_b),sd(r1_res_fail_total_b),sd(r2_res_fail_total_b),sd(r3_res_fail_total_b)))
  
  avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
  std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='r_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=r_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("r_cue_total_b"="darkorchid4","r_res_succ_total_b"="forestgreen","r_res_fail_total_b"="darkred"))
  plt <- plt + labs(title=paste(region_list[region_index],'Punishment Bottom Corr Total'),x="Reward Number",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  #############
  png(paste('corr_p_b_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  pavgs <- data.frame(p_values=c(0,1,2,3),p_cue_total_b=c(mean(p0_cue_total_b),mean(p1_cue_total_b),mean(p2_cue_total_b),mean(p3_cue_total_b)),p_res = c(mean(p0_res_total_b),mean(p1_res_total_b),mean(p2_res_total_b),mean(p3_res_total_b)))
  pstds <- data.frame(p_values=c(0,1,2,3),p_cue_total_b=c(sd(p0_cue_total_b),sd(p1_cue_total_b),sd(p2_cue_total_b),sd(p3_cue_total_b)),p_res=c(sd(p0_res_total_b),sd(p1_res_total_b),sd(p2_res_total_b),sd(p3_res_total_b)))
  
  avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
  std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='p_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=p_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("p_cue_total_b"="coral4","p_res"="slateblue"))
  plt <- plt + labs(title=paste(region_list[region_index],'Punishment Bottom Corr Total'),x="Punishment Number",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  #
  png(paste('corr_p_sf_b_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  pavgs <- data.frame(p_values=c(0,1,2,3),p_cue_total_b=c(mean(p0_cue_total_b),mean(p1_cue_total_b),mean(p2_cue_total_b),mean(p3_cue_total_b)),p_res_succ_total_b = c(mean(p0_res_succ_total_b),mean(p1_res_succ_total_b),mean(p2_res_succ_total_b),mean(p3_res_succ_total_b)),p_res_fail_total_b = c(mean(p0_res_fail_total_b),mean(p1_res_fail_total_b),mean(p2_res_fail_total_b),mean(p3_res_fail_total_b)))
  pstds <- data.frame(p_values=c(0,1,2,3),p_cue_total_b=c(sd(p0_cue_total_b),sd(p1_cue_total_b),sd(p2_cue_total_b),sd(p3_cue_total_b)),p_res_succ_total_b=c(sd(p0_res_succ_total_b),sd(p1_res_succ_total_b),sd(p2_res_succ_total_b),sd(p3_res_succ_total_b)),p_res_fail_total_b=c(sd(p0_res_fail_total_b),sd(p1_res_fail_total_b),sd(p2_res_fail_total_b),sd(p3_res_fail_total_b)))
  
  avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
  std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='p_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=p_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("p_cue_total_b"="darkorchid4","p_res_succ_total_b"="forestgreen","p_res_fail_total_b"="darkred"))
  plt <- plt + labs(title=paste(region_list[region_index],'Punishment Bottom Corr Total'),x="Punishment Number",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  ##########
  png(paste('corr_res_b_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  resavgs <- data.frame(res_values=c(0,1),res_cue_total_b = c(mean(res0_cue_total_b),mean(res1_cue_total_b)),res_res = c(mean(res0_res_total_b),mean(res1_res_total_b)))
  resstds <- data.frame(res_values=c(0,1),res_cue_total_b=c(sd(res0_cue_total_b),sd(res1_cue_total_b)),res_res=c(sd(res0_res_total_b),sd(res1_res_total_b)))
  
  avg_melt <- melt(resavgs,id="res_values",variable.name='type',value.name='avg')
  std_melt <- melt(resstds,id="res_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='res_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=res_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("res_cue_total_b"="coral4","res_res"="slateblue"))
  plt <- plt + labs(title=paste(region_list[region_index],'Result Bottom Corr Total'),x="0: fail, 1: Succ",y="z-score") 
  #if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
  ##########
  png(paste('corr_v_b_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  vavgs <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue_total_b = c(mean(v_3_cue_total_b),mean(v_2_cue_total_b),mean(v_1_cue_total_b),mean(v0_cue_total_b),mean(v1_cue_total_b),mean(v2_cue_total_b),mean(v3_cue_total_b)),v_res = c(mean(v_3_res_total_b),mean(v_2_res_total_b),mean(v_1_res_total_b),mean(v0_res_total_b),mean(v1_res_total_b),mean(v2_res_total_b),mean(v3_res_total_b)))
  vstds <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue_total_b=c(sd(v_3_cue_total_b),sd(v_2_cue_total_b),sd(v_1_cue_total_b),sd(v0_cue_total_b),sd(v1_cue_total_b),sd(v2_cue_total_b),sd(v3_cue_total_b)),v_res=c(sd(v_3_res_total_b),sd(v_2_res_total_b),sd(v_1_res_total_b),sd(v0_res_total_b),sd(v1_res_total_b),sd(v2_res_total_b),sd(v3_res_total_b)))
  
  avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
  std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='v_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=v_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("v_cue_total_b"="coral4","v_res"="slateblue"))
  plt <- plt + labs(title=paste(region_list[region_index],'Value Bottom Corr Total'),x="Value",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()

  #
  png(paste('corr_v_sf_b_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  vavgs <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue_total_b = c(mean(v_3_cue_total_b),mean(v_2_cue_total_b),mean(v_1_cue_total_b),mean(v0_cue_total_b),mean(v1_cue_total_b),mean(v2_cue_total_b),mean(v3_cue_total_b)),v_res_succ_total_b = c(mean(v_3_res_succ_total_b),mean(v_2_res_succ_total_b),mean(v_1_res_succ_total_b),mean(v0_res_succ_total_b),mean(v1_res_succ_total_b),mean(v2_res_succ_total_b),mean(v3_res_succ_total_b)),v_res_fail_total_b = c(mean(v_3_res_fail_total_b),mean(v_2_res_fail_total_b),mean(v_1_res_fail_total_b),mean(v0_res_fail_total_b),mean(v1_res_fail_total_b),mean(v2_res_fail_total_b),mean(v3_res_fail_total_b)))
  vstds <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),v_cue_total_b=c(sd(v_3_cue_total_b),sd(v_2_cue_total_b),sd(v_1_cue_total_b),sd(v0_cue_total_b),sd(v1_cue_total_b),sd(v2_cue_total_b),sd(v3_cue_total_b)),v_res_succ_total_b=c(sd(v_3_res_succ_total_b),sd(v_2_res_succ_total_b),sd(v_1_res_succ_total_b),sd(v0_res_succ_total_b),sd(v1_res_succ_total_b),sd(v2_res_succ_total_b),sd(v3_res_succ_total_b)),v_res_fail_total_b=c(sd(v_3_res_fail_total_b),sd(v_2_res_fail_total_b),sd(v_1_res_fail_total_b),sd(v0_res_fail_total_b),sd(v1_res_fail_total_b),sd(v2_res_fail_total_b),sd(v3_res_fail_total_b)))
  
  avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
  std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='v_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=v_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("v_cue_total_b"="darkorchid4","v_res_succ_total_b"="forestgreen","v_res_fail_total_b"="darkred"))
  plt <- plt + labs(title=paste(region_list[region_index],'Value Bottom Corr Total'),x="Value",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()

  ##########
  png(paste('corr_m_b_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  mavgs <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue_total_b = c(mean(m0_cue_total_b),mean(m1_cue_total_b),mean(m2_cue_total_b),mean(m3_cue_total_b),mean(m4_cue_total_b),mean(m5_cue_total_b),mean(m6_cue_total_b)),m_res = c(mean(m0_res_total_b),mean(m1_res_total_b),mean(m2_res_total_b),mean(m3_res_total_b),mean(m4_res_total_b),mean(m5_res_total_b),mean(m6_res_total_b)))
  mstds <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue_total_b=c(sd(m0_cue_total_b),sd(m1_cue_total_b),sd(m2_cue_total_b),sd(m3_cue_total_b),sd(m4_cue_total_b),sd(m5_cue_total_b),sd(m6_cue_total_b)),m_res=c(sd(m0_res_total_b),sd(m1_res_total_b),sd(m2_res_total_b),sd(m3_res_total_b),sd(m4_res_total_b),sd(m5_res_total_b),sd(m6_res_total_b)))
  
  avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
  std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='m_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=m_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("m_cue_total_b"="coral4","m_res"="slateblue"))
  plt <- plt + labs(title=paste(region_list[region_index],'Motivation Bottom Corr Total'),x="Motivation",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()  
  
  #
  png(paste('corr_m_sf_b_total_',region_list[region_index],'_lr.png',sep=""),width=8,height=6,units="in",res=500)
  
  mavgs <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue_total_b = c(mean(m0_cue_total_b),mean(m1_cue_total_b),mean(m2_cue_total_b),mean(m3_cue_total_b),mean(m4_cue_total_b),mean(m5_cue_total_b),mean(m6_cue_total_b)),m_res_succ_total_b = c(mean(m0_res_succ_total_b),mean(m1_res_succ_total_b),mean(m2_res_succ_total_b),mean(m3_res_succ_total_b),mean(m4_res_succ_total_b),mean(m5_res_succ_total_b),mean(m6_res_succ_total_b)),m_res_fail_total_b = c(mean(m0_res_fail_total_b),mean(m1_res_fail_total_b),mean(m2_res_fail_total_b),mean(m3_res_fail_total_b),mean(m4_res_fail_total_b),mean(m5_res_fail_total_b),mean(m6_res_fail_total_b)))
  mstds <- data.frame(m_values=c(0,1,2,3,4,5,6),m_cue_total_b=c(sd(m0_cue_total_b),sd(m1_cue_total_b),sd(m2_cue_total_b),sd(m3_cue_total_b),sd(m4_cue_total_b),sd(m5_cue_total_b),sd(m6_cue_total_b)),m_res_succ_total_b=c(sd(m0_res_succ_total_b),sd(m1_res_succ_total_b),sd(m2_res_succ_total_b),sd(m3_res_succ_total_b),sd(m4_res),sd(m5_res_succ_total_b),sd(m6_res_succ_total_b)),m_res_fail_total_b=c(sd(m0_res_fail_total_b),sd(m1_res_fail_total_b),sd(m2_res_fail_total_b),sd(m3_res_fail_total_b),sd(m4_res),sd(m5_res_fail_total_b),sd(m6_res_fail_total_b)))
  
  avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
  std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')
  
  test <- merge(std_melt,avg_melt,row.names='m_values')
  #test[is.na(test)] <- 0
  
  plt <- ggplot(test,aes(x=m_values,y=avg,color=type)) + geom_errorbar(aes(ymin=avg-std,ymax=avg+std),width=0.1,linetype=1.0,alpha=0.75,na.rm=T) + theme_classic() 
  plt <- plt +  geom_smooth(method=lm,se=F,size=0.5,na.rm=T) + geom_point(na.rm=T) + scale_colour_manual(name="Unit Key",values=c("m_cue_total_b"="darkorchid4","m_res_succ_total_b"="forestgreen","m_res_fail_total_b"="darkred"))
  plt <- plt + labs(title=paste(region_list[region_index],'Motivation Bottom Corr Total'),x="Motivation",y="z-score") 
  if(plot_p_bool){plt <- plt + stat_fit_glance(method = 'lm',method.args=list(formula=formula),geom='text',aes(label = paste("P-value = ", signif(..p.value.., digits = 3), sep = "")),label.x.npc = 'right', label.y.npc = 0.05, size = 3,na.rm=T)}
  plot(plt)
  graphics.off()
  
}

#rm(list=ls())
