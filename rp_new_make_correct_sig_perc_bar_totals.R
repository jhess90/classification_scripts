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
library(broom)
library(plyr)
library(reshape)


attach('~/dropbox/model_rp/0_8_1/no_abs/alphabeta_M1_percsig_avg.RData')
M1_percsig_bfr_cue_nums_total <- percsig_bfr_cue_nums
M1_percsig_aft_cue_nums_total <- percsig_aft_cue_nums
M1_percsig_bfr_res_nums_total <- percsig_bfr_res_nums
M1_percsig_aft_res_nums_total <- percsig_aft_res_nums
M1_percsig_total_units <- percsig_total_units
detach()
attach('~/dropbox/model_rp/0_8_1/no_abs/alphabeta_S1_percsig_avg.RData')
S1_percsig_bfr_cue_nums_total <- percsig_bfr_cue_nums
S1_percsig_aft_cue_nums_total <- percsig_aft_cue_nums
S1_percsig_bfr_res_nums_total <- percsig_bfr_res_nums
S1_percsig_aft_res_nums_total <- percsig_aft_res_nums
S1_percsig_total_units <- percsig_total_units
detach()
attach('~/dropbox/model_rp/0_8_1/no_abs/alphabeta_PmD_percsig_avg.RData')
PmD_percsig_bfr_cue_nums_total <- percsig_bfr_cue_nums
PmD_percsig_aft_cue_nums_total <- percsig_aft_cue_nums
PmD_percsig_bfr_res_nums_total <- percsig_bfr_res_nums
PmD_percsig_aft_res_nums_total <- percsig_aft_res_nums
PmD_percsig_total_units <- percsig_total_units
detach()

attach('~/dropbox/model_rp/0_8_2/no_abs/alphabeta_M1_percsig_avg.RData')
M1_percsig_bfr_cue_nums_total <- M1_percsig_bfr_cue_nums_total + percsig_bfr_cue_nums
M1_percsig_aft_cue_nums_total <- M1_percsig_aft_cue_nums_total + percsig_aft_cue_nums
M1_percsig_bfr_res_nums_total <- M1_percsig_bfr_res_nums_total + percsig_bfr_res_nums
M1_percsig_aft_res_nums_total <- M1_percsig_aft_res_nums_total + percsig_aft_res_nums
M1_percsig_total_units <- M1_percsig_total_units + percsig_total_units
detach()
attach('~/dropbox/model_rp/0_8_2/no_abs/alphabeta_S1_percsig_avg.RData')
S1_percsig_bfr_cue_nums_total <- S1_percsig_bfr_cue_nums_total + percsig_bfr_cue_nums
S1_percsig_aft_cue_nums_total <- S1_percsig_aft_cue_nums_total + percsig_aft_cue_nums
S1_percsig_bfr_res_nums_total <- S1_percsig_bfr_res_nums_total + percsig_bfr_res_nums
S1_percsig_aft_res_nums_total <- S1_percsig_aft_res_nums_total + percsig_aft_res_nums
S1_percsig_total_units <- S1_percsig_total_units + percsig_total_units
detach()
attach('~/dropbox/model_rp/0_8_2/no_abs/alphabeta_PmD_percsig_avg.RData')
PmD_percsig_bfr_cue_nums_total <- PmD_percsig_bfr_cue_nums_total + percsig_bfr_cue_nums
PmD_percsig_aft_cue_nums_total <- PmD_percsig_aft_cue_nums_total + percsig_aft_cue_nums
PmD_percsig_bfr_res_nums_total <- PmD_percsig_bfr_res_nums_total + percsig_bfr_res_nums
PmD_percsig_aft_res_nums_total <- PmD_percsig_aft_res_nums_total + percsig_aft_res_nums
PmD_percsig_total_units <- PmD_percsig_total_units + percsig_total_units
detach()

attach('~/dropbox/model_rp/0_9_1/no_abs/alphabeta_M1_percsig_avg.RData')
M1_percsig_bfr_cue_nums_total <- M1_percsig_bfr_cue_nums_total + percsig_bfr_cue_nums
M1_percsig_aft_cue_nums_total <- M1_percsig_aft_cue_nums_total + percsig_aft_cue_nums
M1_percsig_bfr_res_nums_total <- M1_percsig_bfr_res_nums_total + percsig_bfr_res_nums
M1_percsig_aft_res_nums_total <- M1_percsig_aft_res_nums_total + percsig_aft_res_nums
M1_percsig_total_units <- M1_percsig_total_units + percsig_total_units
detach()
attach('~/dropbox/model_rp/0_9_1/no_abs/alphabeta_S1_percsig_avg.RData')
S1_percsig_bfr_cue_nums_total <- S1_percsig_bfr_cue_nums_total + percsig_bfr_cue_nums
S1_percsig_aft_cue_nums_total <- S1_percsig_aft_cue_nums_total + percsig_aft_cue_nums
S1_percsig_bfr_res_nums_total <- S1_percsig_bfr_res_nums_total + percsig_bfr_res_nums
S1_percsig_aft_res_nums_total <- S1_percsig_aft_res_nums_total + percsig_aft_res_nums
S1_percsig_total_units <- S1_percsig_total_units + percsig_total_units
detach()
attach('~/dropbox/model_rp/0_9_1/no_abs/alphabeta_PmD_percsig_avg.RData')
PmD_percsig_bfr_cue_nums_total <- PmD_percsig_bfr_cue_nums_total + percsig_bfr_cue_nums
PmD_percsig_aft_cue_nums_total <- PmD_percsig_aft_cue_nums_total + percsig_aft_cue_nums
PmD_percsig_bfr_res_nums_total <- PmD_percsig_bfr_res_nums_total + percsig_bfr_res_nums
PmD_percsig_aft_res_nums_total <- PmD_percsig_aft_res_nums_total + percsig_aft_res_nums
PmD_percsig_total_units <- PmD_percsig_total_units + percsig_total_units
detach()

attach('~/dropbox/model_rp/0_9_2/no_abs/alphabeta_M1_percsig_avg.RData')
M1_percsig_bfr_cue_nums_total <- M1_percsig_bfr_cue_nums_total + percsig_bfr_cue_nums
M1_percsig_aft_cue_nums_total <- M1_percsig_aft_cue_nums_total + percsig_aft_cue_nums
M1_percsig_bfr_res_nums_total <- M1_percsig_bfr_res_nums_total + percsig_bfr_res_nums
M1_percsig_aft_res_nums_total <- M1_percsig_aft_res_nums_total + percsig_aft_res_nums
M1_percsig_total_units <- M1_percsig_total_units + percsig_total_units
detach()
attach('~/dropbox/model_rp/0_9_2/no_abs/alphabeta_S1_percsig_avg.RData')
S1_percsig_bfr_cue_nums_total <- S1_percsig_bfr_cue_nums_total + percsig_bfr_cue_nums
S1_percsig_aft_cue_nums_total <- S1_percsig_aft_cue_nums_total + percsig_aft_cue_nums
S1_percsig_bfr_res_nums_total <- S1_percsig_bfr_res_nums_total + percsig_bfr_res_nums
S1_percsig_aft_res_nums_total <- S1_percsig_aft_res_nums_total + percsig_aft_res_nums
S1_percsig_total_units <- S1_percsig_total_units + percsig_total_units
detach()
attach('~/dropbox/model_rp/0_9_2/no_abs/alphabeta_PmD_percsig_avg.RData')
PmD_percsig_bfr_cue_nums_total <- PmD_percsig_bfr_cue_nums_total + percsig_bfr_cue_nums
PmD_percsig_aft_cue_nums_total <- PmD_percsig_aft_cue_nums_total + percsig_aft_cue_nums
PmD_percsig_bfr_res_nums_total <- PmD_percsig_bfr_res_nums_total + percsig_bfr_res_nums
PmD_percsig_aft_res_nums_total <- PmD_percsig_aft_res_nums_total + percsig_aft_res_nums
PmD_percsig_total_units <- PmD_percsig_total_units + percsig_total_units
detach()


for (region_index in 1:length(region_list)){
  
  ########
  if (region_index==1){
    
    percsig_bfr_cue_nums <- M1_percsig_bfr_cue_nums_total
    percsig_aft_cue_nums <- M1_percsig_aft_cue_nums_total
    percsig_bfr_res_nums <- M1_percsig_bfr_res_nums_total
    percsig_aft_res_nums <- M1_percsig_aft_res_nums_total
    percsig_unit_total <- M1_percsig_total_units
    
    #slopes_bfr_cue <- M1_slopes_bfr_cue
    #slopes_aft_cue <- M1_slopes_aft_cue
    #slopes_bfr_result <- M1_slopes_bfr_result
    #slopes_aft_result <- M1_slopes_aft_result
    #slopes_bfr_cue_sigall <- M1_slopes_bfr_cue_sigall
    #slopes_aft_cue_sigall <- M1_slopes_aft_cue_sigall
    #slopes_bfr_result_sigall <- M1_slopes_bfr_result_sigall
    #slopes_aft_result_sigall <- M1_slopes_aft_result_sigall
  }else if(region_index == 2){
    percsig_bfr_cue_nums <- S1_percsig_bfr_cue_nums_total
    percsig_aft_cue_nums <- S1_percsig_aft_cue_nums_total
    percsig_bfr_res_nums <- S1_percsig_bfr_res_nums_total
    percsig_aft_res_nums <- S1_percsig_aft_res_nums_total
    percsig_unit_total <- S1_percsig_total_units
    #slopes_bfr_cue <- S1_slopes_bfr_cue
    #slopes_aft_cue <- S1_slopes_aft_cue
    #slopes_bfr_result <- S1_slopes_bfr_result
    #slopes_aft_result <- S1_slopes_aft_result
    #slopes_bfr_cue_sigall <- S1_slopes_bfr_cue_sigall
    #slopes_aft_cue_sigall <- S1_slopes_aft_cue_sigall
    #slopes_bfr_result_sigall <- S1_slopes_bfr_result_sigall
    #slopes_aft_result_sigall <- S1_slopes_aft_result_sigall
  }else{
    percsig_bfr_cue_nums <- PmD_percsig_bfr_cue_nums_total
    percsig_aft_cue_nums <- PmD_percsig_aft_cue_nums_total
    percsig_bfr_res_nums <- PmD_percsig_bfr_res_nums_total
    percsig_aft_res_nums <- PmD_percsig_aft_res_nums_total
    percsig_unit_total <- PmD_percsig_total_units
    #slopes_bfr_cue <- PmD_slopes_bfr_cue
    #slopes_aft_cue <- PmD_slopes_aft_cue
    #slopes_bfr_result <- PmD_slopes_bfr_result
    #slopes_aft_result <- PmD_slopes_aft_result
    #slopes_aft_cue_sigall <- PmD_slopes_aft_cue_sigall
    #slopes_bfr_result_sigall <- PmD_slopes_bfr_result_sigall
    #slopes_aft_result_sigall <- PmD_slopes_aft_result_sigall
  }
  
  #########
  
  
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

  png(paste('NEW_all_bar_plotted_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)

  df_all <- rbind(bfr_cue_df,aft_cue_df,bfr_res_df,aft_res_df)
  df_all <- df_all[which(df_all$perc > 0),]

  bar_plt <- ggplot() + geom_bar(aes(x=df_all$type,y=df_all$perc,fill=df_all$labs),data=df_all,stat="identity")
  bar_plt <- bar_plt + labs(title=region_list[region_index],fill="",x="Time Window",y="Percentage") + scale_fill_manual(values=c("lightblue","seagreen","grey","slateblue"))
  bar_plt <- bar_plt + geom_text(aes(x=df_all$type,y=df_all$position,label=df_all$label),size=3,stat="identity") + theme_classic()

  plot(bar_plt)
  graphics.off()
  
  
}