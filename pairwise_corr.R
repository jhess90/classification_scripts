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
region_list <- c('M1','S1','PmD')

##############################

compute_plot_corr <- function(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,type_name,trial_inds){
  cue_fr_M1 <- all_cue_fr_M1[,trial_inds,]
  cue_fr_S1 <- all_cue_fr_S1[,trial_inds,]
  cue_fr_PmD <- all_cue_fr_PmD[,trial_inds,]

  res_fr_M1 <- all_res_fr_M1[,trial_inds,]
  res_fr_S1 <- all_res_fr_S1[,trial_inds,]
  res_fr_PmD <- all_res_fr_PmD[,trial_inds,]
  
  
  cue_avgs_M1 <- c()
  res_avgs_M1 <- c()
  for(unit_num in 1:total_unit_num_M1){
    cue_avgs_M1 <- rbind(cue_avgs_M1,colMeans(cue_fr_M1[unit_num,,]))
    res_avgs_M1 <- rbind(res_avgs_M1,colMeans(res_fr_M1[unit_num,,]))
  }
  
  cue_avgs_S1 <- c()
  res_avgs_S1 <- c()
  for(unit_num in 1:total_unit_num_S1){
    cue_avgs_S1 <- rbind(cue_avgs_S1,colMeans(cue_fr_S1[unit_num,,]))
    res_avgs_S1 <- rbind(res_avgs_S1,colMeans(res_fr_S1[unit_num,,]))
  }
  
  cue_avgs_PmD <- c()
  res_avgs_PmD <- c()
  for(unit_num in 1:total_unit_num_PmD){
    cue_avgs_PmD <- rbind(cue_avgs_PmD,colMeans(cue_fr_PmD[unit_num,,]))
    res_avgs_PmD <- rbind(res_avgs_PmD,colMeans(res_fr_PmD[unit_num,,]))
  }
  
  combined_cue_avgs <- rbind(cue_avgs_M1,cue_avgs_S1,cue_avgs_PmD)
  combined_res_avgs <- rbind(res_avgs_M1,res_avgs_S1,res_avgs_PmD)
  
  cue_corr <- round(cor(t(combined_cue_avgs)),2)
  res_corr <- round(cor(t(combined_res_avgs)),2)
  
  cue_pmat <- cor_pmat(t(combined_cue_avgs))
  res_pmat <- cor_pmat(t(combined_res_avgs))
  
  label_df <- data.frame(
    x = c(total_unit_num_M1/2,total_unit_num_M1+total_unit_num_S1/2,total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD/2,5,5,5),
    y = c(-8,-8,-8,total_unit_num_M1/2,total_unit_num_M1+total_unit_num_S1/2,total_unit_num_M1+total_unit_num_S1+total_unit_num_PmD/2),
    text = c('M1','S1','PmD','M1','S1','PmD')
  )
  
  
  png(paste("corr_",type_name,".png",sep=""),width=8,height=6,units="in",res=500)
  
  cue_plt <- ggcorrplot(cue_corr,p.mat=cue_pmat,type="lower",insig="blank",outline.col='white')
  cue_plt <- cue_plt + geom_vline(xintercept=total_unit_num_M1,color="grey") + geom_vline(xintercept=total_unit_num_M1+total_unit_num_S1,color="grey")
  cue_plt <- cue_plt + geom_hline(yintercept=total_unit_num_M1,color="grey") + geom_hline(yintercept=total_unit_num_M1+total_unit_num_S1,color="grey")
  cue_plt <- cue_plt + labs(title=paste(type_name,":\nCue",sep="")) + theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank(),legend.position="none",axis.text.y=element_blank(),plot.title=element_text(size=16))
  cue_plt <- cue_plt + geom_text(data=label_df,aes(x=x,y=y,label=text),inherit.aes=FALSE,size=5)
  
  res_plt <- ggcorrplot(res_corr,p.mat=res_pmat,type="lower",insig="blank",outline.col='white')
  res_plt <- res_plt + geom_vline(xintercept=total_unit_num_M1,color="grey") + geom_vline(xintercept=total_unit_num_M1+total_unit_num_S1,color="grey")
  res_plt <- res_plt + geom_hline(yintercept=total_unit_num_M1,color="grey") + geom_hline(yintercept=total_unit_num_M1+total_unit_num_S1,color="grey")
  res_plt <- res_plt + labs(title="\nResult") + theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank(),legend.position="none",axis.text.y=element_blank(),plot.title=element_text(size=16))
  res_plt <- res_plt + geom_text(data=label_df,aes(x=x,y=y,label=text),inherit.aes=FALSE,size=5)
  
  multiplot(cue_plt,res_plt,cols=2)
  graphics.off()
  
}



##############################




#uses simple_output_[region].mat, one from each region
readin <- readMat('simple_output_M1.mat')

all_cue_fr_M1 <- readin$return.dict[,,1]$all.cue.fr
all_res_fr_M1 <- readin$return.dict[,,1]$all.res.fr

condensed <- readin$return.dict[,,1]$condensed
bin_size <- readin$return.dict[,,1]$params[,,1]$bin.size[,]
total_unit_num_M1 <- dim(all_cue_fr_M1)[1]

readin <- readMat('simple_output_S1.mat')
all_cue_fr_S1 <- readin$return.dict[,,1]$all.cue.fr
all_res_fr_S1 <- readin$return.dict[,,1]$all.res.fr
total_unit_num_S1 <- dim(all_cue_fr_S1)[1]

readin <- readMat('simple_output_PmD.mat')
all_cue_fr_PmD <- readin$return.dict[,,1]$all.cue.fr
all_res_fr_PmD <- readin$return.dict[,,1]$all.res.fr
total_unit_num_PmD <- dim(all_cue_fr_PmD)[1]

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

#

compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'All',1:dim(all_cue_fr_M1)[2])

compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0',r0)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RX',rx)

compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'P0',p0)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'PX',px)

compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'Succ',res1)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'Fail',res0)

compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0 Succ',r0_s)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0 Fail',r0_f)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RX Succ',rx_s)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RX Fail',rx_f)

compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'P0 Succ',p0_s)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'P0 Fail',p0_f)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'PX Succ',px_s)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'PX Fail',px_f)

compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0P0',r0_p0)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'R0PX',r0_px)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RXP0',rx_p0)
compute_plot_corr(all_cue_fr_M1,all_cue_fr_S1,all_cue_fr_PmD,all_res_fr_M1,all_res_fr_S1,all_res_fr_PmD,total_unit_num_M1,total_unit_num_S1,total_unit_num_PmD,'RXPX',rx_px)




