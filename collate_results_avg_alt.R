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
library(broom)
library(plyr)
library(reshape)


load('alt_combined_info.RData')
load('alt_combined_xlsx_info.RData')

#########
#labs <- c("both pos","both neg","alpha pos","beta pos")
labs <- c("both pos","both neg","beta pos","alpha pos")
region_list <- c('M1','S1','PmD')

for (region_index in 1:length(region_list)){
  #nums: [num both pos, num both neg, num beta pos, num alpha pos] 
   if (region_list[region_index] == 'M1'){
      bfr_cue_nums_sum <- M1_bfr_cue_nums_sum
      aft_cue_nums_sum <- M1_aft_cue_nums_sum
      bfr_result_nums_sum <- M1_bfr_result_nums_sum
      aft_result_nums_sum <- M1_aft_result_nums_sum
      bfr_cue_all_slopes <- M1_bfr_cue_all_slopes
      bfr_cue_sig_slopes <- M1_bfr_cue_sig_slopes
      aft_cue_all_slopes <- M1_aft_cue_all_slopes
      aft_cue_sig_slopes <- M1_aft_cue_sig_slopes
      bfr_result_all_slopes <- M1_bfr_result_all_slopes
      bfr_result_sig_slopes <- M1_bfr_result_sig_slopes
      aft_result_all_slopes <- M1_aft_result_all_slopes
      aft_result_sig_slopes <- M1_aft_result_sig_slopes
      percsig_bfr_cue_nums <- M1_ps_bfr_cue_nums_sum
      percsig_aft_cue_nums <- M1_ps_aft_cue_nums_sum
      percsig_bfr_res_nums <- M1_ps_bfr_res_nums_sum
      percsig_aft_res_nums <- M1_ps_aft_res_nums_sum
      percsig_unit_total <- M1_unit_total

    }else if(region_list[region_index] == 'S1'){
      bfr_cue_nums_sum <- S1_bfr_cue_nums_sum
      aft_cue_nums_sum <- S1_aft_cue_nums_sum
      bfr_result_nums_sum <- S1_bfr_result_nums_sum
      aft_result_nums_sum <- S1_aft_result_nums_sum     
      bfr_cue_all_slopes <- S1_bfr_cue_all_slopes
      bfr_cue_sig_slopes <- S1_bfr_cue_sig_slopes
      aft_cue_all_slopes <- S1_aft_cue_all_slopes
      aft_cue_sig_slopes <- S1_aft_cue_sig_slopes
      bfr_result_all_slopes <- S1_bfr_result_all_slopes
      bfr_result_sig_slopes <- S1_bfr_result_sig_slopes
      aft_result_all_slopes <- S1_aft_result_all_slopes
      aft_result_sig_slopes <- S1_aft_result_sig_slopes
      percsig_bfr_cue_nums <- S1_ps_bfr_cue_nums_sum
      percsig_aft_cue_nums <- S1_ps_aft_cue_nums_sum
      percsig_bfr_res_nums <- S1_ps_bfr_res_nums_sum
      percsig_aft_res_nums <- S1_ps_aft_res_nums_sum
      percsig_unit_total <- S1_unit_total
      
    }else if(region_list[region_index] == 'PmD'){
      bfr_cue_nums_sum <- PmD_bfr_cue_nums_sum
      aft_cue_nums_sum <- PmD_aft_cue_nums_sum
      bfr_result_nums_sum <- PmD_bfr_result_nums_sum
      aft_result_nums_sum <- PmD_aft_result_nums_sum
      bfr_cue_all_slopes <- PmD_bfr_cue_all_slopes
      bfr_cue_sig_slopes <- PmD_bfr_cue_sig_slopes
      aft_cue_all_slopes <- PmD_aft_cue_all_slopes
      aft_cue_sig_slopes <- PmD_aft_cue_sig_slopes
      bfr_result_all_slopes <- PmD_bfr_result_all_slopes
      bfr_result_sig_slopes <- PmD_bfr_result_sig_slopes
      aft_result_all_slopes <- PmD_aft_result_all_slopes
      aft_result_sig_slopes <- PmD_aft_result_sig_slopes
      percsig_bfr_cue_nums <- PmD_ps_bfr_cue_nums_sum
      percsig_aft_cue_nums <- PmD_ps_aft_cue_nums_sum
      percsig_bfr_res_nums <- PmD_ps_bfr_res_nums_sum
      percsig_aft_res_nums <- PmD_ps_aft_res_nums_sum
      percsig_unit_total <- PmD_unit_total
      
    }

  bfr_cue_df <-data.frame(perc=bfr_cue_nums_sum/sum(bfr_cue_nums_sum),labs,type='bfr_cue')
  aft_cue_df <-data.frame(perc=aft_cue_nums_sum/sum(aft_cue_nums_sum),labs,type='aft_cue')
  bfr_result_df <-data.frame(perc=bfr_result_nums_sum/sum(bfr_result_nums_sum),labs,type='bfr_result')
  aft_result_df <-data.frame(perc=aft_result_nums_sum/sum(aft_result_nums_sum),labs,type='aft_result')
  
  x <- rev(order(bfr_cue_df$labs))
  
  bfr_cue_df <- bfr_cue_df[rev(order(bfr_cue_df$labs)),]
  aft_cue_df <- aft_cue_df[rev(order(aft_cue_df$labs)),]
  bfr_result_df <- bfr_result_df[rev(order(bfr_result_df$labs)),]
  aft_result_df <- aft_result_df[rev(order(aft_result_df$labs)),]
  
  bfr_cue_df <- ddply(bfr_cue_df,.(type),transform,position=(cumsum(bfr_cue_df$perc)-0.5*bfr_cue_df$perc))
  aft_cue_df <- ddply(aft_cue_df,.(type),transform,position=(cumsum(aft_cue_df$perc)-0.5*aft_cue_df$perc))
  bfr_result_df <- ddply(bfr_result_df,.(type),transform,position=(cumsum(bfr_result_df$perc)-0.5*bfr_result_df$perc))
  aft_result_df <- ddply(aft_result_df,.(type),transform,position=(cumsum(aft_result_df$perc)-0.5*aft_result_df$perc))
  
  bfr_cue_df <- ddply(bfr_cue_df,.(type),transform,label=paste(scales::percent(bfr_cue_df$perc),' n=',bfr_cue_nums_sum,sep=""))
  aft_cue_df <- ddply(aft_cue_df,.(type),transform,label=paste(scales::percent(aft_cue_df$perc),' n=',aft_cue_nums_sum,sep=""))
  bfr_result_df <- ddply(bfr_result_df,.(type),transform,label=paste(scales::percent(bfr_result_df$perc),' n=',bfr_result_nums_sum,sep=""))
  aft_result_df <- ddply(aft_result_df,.(type),transform,label=paste(scales::percent(aft_result_df$perc),' n=',aft_result_nums_sum,sep=""))
  
  png(paste('AVG_all_signs_bar_plotted_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  
  df_all <- rbind(bfr_cue_df,aft_cue_df,bfr_result_df,aft_result_df)
  df_all <- df_all[which(df_all$perc > 0),]
  
  bar_plt <- ggplot() + geom_bar(aes(x=df_all$type,y=df_all$perc,fill=df_all$labs),data=df_all,stat="identity") 
  bar_plt <- bar_plt + labs(title=region_list[region_index],fill="",x="Time Window",y="Percentage") + scale_fill_manual(values=c("plum2","turquoise4","lightsalmon","royalblue"))
  bar_plt <- bar_plt + geom_text(aes(x=df_all$type,y=df_all$position,label=df_all$label),size=4,stat="identity") + theme_classic()
  
  plot(bar_plt)
  graphics.off()
  
  # #########################
  # png(paste('AVG_slope_collated_hist_',region_list[region_index],'_bfr_cue.png',sep=""),width=8,height=6,units="in",res=500)
  #  
  # sig_plt <- ggplot(bfr_cue_sig_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # sig_plt <- sig_plt + labs(title=paste(region_list[region_index],'bfr cue','significant')) 
  # all_plt <- ggplot(bfr_cue_all_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # all_plt <- all_plt + labs(title=paste(region_list[region_index],'bfr cue','all'))
  #  
  #  
  #  
  # multiplot(sig_plt,all_plt,cols=1)
  # graphics.off()
  #  
  # png(paste('AVG_slope_collated_hist_',region_list[region_index],'_aft_cue.png',sep=""),width=8,height=6,units="in",res=500)
  #  
  # sig_plt <- ggplot(aft_cue_sig_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # sig_plt <- sig_plt + labs(title=paste(region_list[region_index],'aft cue','significant')) 
  # all_plt <- ggplot(aft_cue_all_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # all_plt <- all_plt + labs(title=paste(region_list[region_index],'aft cue','all'))
  #  
  # multiplot(sig_plt,all_plt,cols=1)
  # graphics.off()
  #  
  # png(paste('AVG_slope_collated_hist_',region_list[region_index],'_bfr_result.png',sep=""),width=8,height=6,units="in",res=500)
  #  
  # sig_plt <- ggplot(bfr_result_sig_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # sig_plt <- sig_plt + labs(title=paste(region_list[region_index],'bfr result','significant')) 
  # all_plt <- ggplot(bfr_result_all_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # all_plt <- all_plt + labs(title=paste(region_list[region_index],'bfr result','all'))
  # 
  # multiplot(sig_plt,all_plt,cols=1)
  # graphics.off()
  #  
  # png(paste('AVG_slope_collated_hist_',region_list[region_index],'_aft_result.png',sep=""),width=8,height=6,units="in",res=500)
  #  
  # sig_plt <- ggplot(aft_result_sig_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2) 
  # sig_plt <- sig_plt + labs(title=paste(region_list[region_index],'aft result','significant')) 
  # all_plt <- ggplot(aft_result_all_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  # all_plt <- all_plt + labs(title=paste(region_list[region_index],'aft result','all'))
  # 
  # multiplot(sig_plt,all_plt,cols=1)
  # graphics.off()
  ###########
  
  png(paste('AVG_2_slope_collated_hist_',region_list[region_index],'_bfr_cue.png',sep=""),width=8,height=6,units="in",res=500)
  
  #only plot those w/in 3 sd of mean for now
  bc_sig_bounds <- 3*sd(bfr_cue_sig_slopes$slopes)
  bc_all_bounds <- 3*sd(bfr_cue_all_slopes$slopes)
  ac_sig_bounds <- 3*sd(aft_cue_sig_slopes$slopes)
  ac_all_bounds <- 3*sd(aft_cue_all_slopes$slopes)
  br_sig_bounds <- 3*sd(bfr_result_sig_slopes$slopes)
  br_all_bounds <- 3*sd(bfr_result_all_slopes$slopes)
  ar_sig_bounds <- 3*sd(aft_result_sig_slopes$slopes)
  ar_all_bounds <- 3*sd(aft_result_all_slopes$slopes)
  
  
  sig_plt <- ggplot(bfr_cue_sig_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(bfr_cue_sig_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(bfr_cue_sig_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'bfr cue','significant')) + theme_classic() + xlim(-1*bc_sig_bounds,bc_sig_bounds)

  all_plt <- ggplot(bfr_cue_all_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(bfr_cue_all_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(bfr_cue_all_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'bfr cue','all')) + theme_classic() + xlim(-1*bc_all_bounds,bc_all_bounds)
  
  multiplot(sig_plt,all_plt,cols=1)
  graphics.off()
  
  png(paste('AVG_2_slope_collated_hist_',region_list[region_index],'_aft_cue.png',sep=""),width=8,height=6,units="in",res=500)
  
  sig_plt <- ggplot(aft_cue_sig_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(aft_cue_sig_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(aft_cue_sig_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'aft cue','significant')) + theme_classic() + xlim(-1*ac_sig_bounds,ac_sig_bounds)
  
  all_plt <- ggplot(aft_cue_all_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(aft_cue_all_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(aft_cue_all_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'aft cue','all')) + theme_classic() + xlim(-1*ac_all_bounds,ac_all_bounds)
  
  multiplot(sig_plt,all_plt,cols=1)
  graphics.off()  
  
  png(paste('AVG_2_slope_collated_hist_',region_list[region_index],'_bfr_result.png',sep=""),width=8,height=6,units="in",res=500)
  
  sig_plt <- ggplot(bfr_result_sig_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(bfr_result_sig_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(bfr_result_sig_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'bfr result','significant')) + theme_classic() + xlim(-1*br_sig_bounds,br_sig_bounds)
  
  all_plt <- ggplot(bfr_result_all_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(bfr_result_all_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(bfr_result_all_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'bfr result','all')) + theme_classic() + xlim(-1*br_all_bounds,br_all_bounds)
  
  multiplot(sig_plt,all_plt,cols=1)
  graphics.off()  
  
  png(paste('AVG_2_slope_collated_hist_',region_list[region_index],'_aft_result.png',sep=""),width=8,height=6,units="in",res=500)
  
  sig_plt <- ggplot(aft_result_sig_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(aft_result_sig_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(aft_result_sig_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'aft result','significant')) + theme_classic() + xlim(-1*ar_sig_bounds,ar_sig_bounds)
  
  all_plt <- ggplot(aft_result_all_slopes,aes(x=slopes)) +
    geom_histogram(data=subset(aft_result_all_slopes,type=='val'),fill='maroon',alpha=0.75,position='identity',binwidth=0.2) + 
    geom_histogram(data=subset(aft_result_all_slopes,type=='mtv'),fill='slateblue',alpha=0.6,position='identity',binwidth=0.2) + 
    scale_fill_manual(name="type",values=c("maroon","slateblue")) + labs(title=paste(region_list[region_index],'aft result','all')) + theme_classic() + xlim(-1*ar_all_bounds,ar_all_bounds)
  
  multiplot(sig_plt,all_plt,cols=1)
  graphics.off()
  
  
  
  ####################################

  labs <- c('alpha only sig','beta only sig','both sig','no sig')
  
  #percsig nums: no sig, both sig, beta sig, alpha sig
  #bfr_cue_vals <- c(percsig_bfr_cue_nums[4]/percsig_total_units,percsig_bfr_cue_nums[3]/percsig_total_units,percsig_bfr_cue_nums[2]/percsig_total_units,percsig_bfr_cue_nums[1]/percsig_total_units)
  #aft_cue_vals <- c(percsig_aft_cue_nums[4]/percsig_total_units,percsig_aft_cue_nums[3]/percsig_total_units,percsig_aft_cue_nums[2]/percsig_total_units,percsig_aft_cue_nums[1]/percsig_total_units)
  #bfr_res_vals <- c(percsig_bfr_res_nums[4]/percsig_total_units,percsig_bfr_res_nums[3]/percsig_total_units,percsig_bfr_res_nums[2]/percsig_total_units,percsig_bfr_res_nums[1]/percsig_total_units)
  #aft_res_vals <- c(percsig_aft_res_nums[4]/percsig_total_units,percsig_aft_res_nums[3]/percsig_total_units,percsig_aft_res_nums[2]/percsig_total_units,percsig_aft_res_nums[1]/percsig_total_units)
  
  bfr_cue_vals <- percsig_bfr_cue_nums / percsig_unit_total
  aft_cue_vals <- percsig_aft_cue_nums / percsig_unit_total
  bfr_res_vals <- percsig_bfr_res_nums / percsig_unit_total
  aft_res_vals <- percsig_aft_res_nums / percsig_unit_total
  
  # bfr_cue_vals <- c(bfr_cue_alpha_sig,bfr_cue_beta_sig,bfr_cue_both_sig,bfr_cue_non)
  # aft_cue_vals <- c(aft_cue_alpha_sig,aft_cue_beta_sig,aft_cue_both_sig,aft_cue_non)
  # bfr_result_vals <- c(bfr_result_alpha_sig,bfr_result_beta_sig,bfr_result_both_sig,bfr_result_non)
  # aft_result_vals <- c(aft_result_alpha_sig,aft_result_beta_sig,aft_result_both_sig,aft_result_non)
  
  bfr_cue_df <- data.frame(perc=bfr_cue_vals,labs,type='bfr_cue')
  aft_cue_df <- data.frame(perc=aft_cue_vals,labs,type='aft_cue')
  bfr_res_df <- data.frame(perc=bfr_res_vals,labs,type='bfr_result')
  aft_res_df <- data.frame(perc=aft_res_vals,labs,type='aft_result')
  
  bfr_cue_df <- bfr_cue_df[rev(order(bfr_cue_df$labs)),]
  aft_cue_df <- aft_cue_df[rev(order(aft_cue_df$labs)),]
  bfr_res_df <- bfr_res_df[rev(order(bfr_res_df$labs)),]
  aft_res_df <- aft_res_df[rev(order(aft_res_df$labs)),]
  
  bfr_cue_df <- ddply(bfr_cue_df,.(type),transform,position=(cumsum(bfr_cue_df$perc)-0.5*bfr_cue_df$perc))
  aft_cue_df <- ddply(aft_cue_df,.(type),transform,position=(cumsum(aft_cue_df$perc)-0.5*aft_cue_df$perc))
  bfr_res_df <- ddply(bfr_res_df,.(type),transform,position=(cumsum(bfr_res_df$perc)-0.5*bfr_res_df$perc))
  aft_res_df <- ddply(aft_res_df,.(type),transform,position=(cumsum(aft_res_df$perc)-0.5*aft_res_df$perc))
  
  #percsig nums: no sig, both sig, beta sig, alpha sig
  #ACTUAL percsig nums: alpha sig, both sig, beta sig, no sig
  bfr_cue_nums <- c(percsig_bfr_cue_nums[4],percsig_bfr_cue_nums[3],percsig_bfr_cue_nums[2],percsig_bfr_cue_nums[1])
  aft_cue_nums <- c(percsig_aft_cue_nums[4],percsig_aft_cue_nums[3],percsig_aft_cue_nums[2],percsig_aft_cue_nums[1])
  bfr_res_nums <- c(percsig_bfr_res_nums[4],percsig_bfr_res_nums[3],percsig_bfr_res_nums[2],percsig_bfr_res_nums[1])
  aft_res_nums <- c(percsig_aft_res_nums[4],percsig_aft_res_nums[3],percsig_aft_res_nums[2],percsig_aft_res_nums[1])
  
  # 
  # bfr_cue_nums <- percsig_bfr_cue_nums
  # aft_cue_nums <- percsig_aft_cue_nums
  # bfr_res_nums <- percsig_bfr_res_nums
  # aft_res_nums <- percsig_aft_res_nums
  # 
  # bfr_cue_nums <- c(bfr_cue[5]-bfr_cue[4]-bfr_cue[13]-bfr_cue[12],bfr_cue[4],bfr_cue[13],bfr_cue[12])
  # aft_cue_nums <- c(aft_cue[5]-aft_cue[4]-aft_cue[13]-aft_cue[12],aft_cue[4],aft_cue[13],aft_cue[12])
  # bfr_result_nums <- c(bfr_result[5]-bfr_result[4]-bfr_result[13]-bfr_result[12],bfr_result[4],bfr_result[13],bfr_result[12])
  # aft_result_nums <- c(aft_result[5]-aft_result[4]-aft_result[13]-aft_result[12],aft_result[4],aft_result[13],aft_result[12])
  
  bfr_cue_df <- ddply(bfr_cue_df,.(type),transform,label=paste(scales::percent(bfr_cue_df$perc),' n=',bfr_cue_nums,sep=""))
  aft_cue_df <- ddply(aft_cue_df,.(type),transform,label=paste(scales::percent(aft_cue_df$perc),' n=',aft_cue_nums,sep=""))
  bfr_res_df <- ddply(bfr_res_df,.(type),transform,label=paste(scales::percent(bfr_res_df$perc),' n=',bfr_res_nums,sep=""))
  aft_res_df <- ddply(aft_res_df,.(type),transform,label=paste(scales::percent(aft_res_df$perc),' n=',aft_res_nums,sep=""))
  
  png(paste('all_bar_plotted_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  
  df_all <- rbind(bfr_cue_df,aft_cue_df,bfr_res_df,aft_res_df)
  df_all <- df_all[which(df_all$perc > 0),]
  
  bar_plt <- ggplot() + geom_bar(aes(x=df_all$type,y=df_all$perc,fill=df_all$labs),data=df_all,stat="identity") 
  bar_plt <- bar_plt + labs(title=region_list[region_index],fill="",x="Time Window",y="Percentage") + scale_fill_manual(values=c("lightblue","seagreen","grey","slateblue"))
  bar_plt <- bar_plt + geom_text(aes(x=df_all$type,y=df_all$position,label=df_all$label),size=3,stat="identity") + theme_classic()
  
  plot(bar_plt)
  graphics.off()
  
  
}

#rm(list=ls())
