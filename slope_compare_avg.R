library(openxlsx)
library(ggplot2)
library(reshape2)
#source("~/dropbox/mult_rp_files/r_test/multiplot.R")
#source("~/Dropbox/mult_rp_files/r_test/multiplot.R")
#source("~/documents/lab/workspace/Classification_scripts/multiplot.R")l
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

#######

avg_alphabeta_bool = TRUE
all_alphabeta_bool = FALSE

if (avg_alphabeta_bool & all_alphabeta_bool){cat('ERROR both cant be true')}

#######

#file_list <- c('sig_slopes_M1_dicts.xlsx','sig_slopes_S1_dicts.xlsx','sig_slopes_PmD_dicts.xlsx')
file_list <- c(Sys.glob('sig_slopes*M1*.xlsx'),Sys.glob('sig_slopes*S1*.xlsx'),Sys.glob('sig_slopes*PmD*.xlsx'))
if (length(file_list) != 3){cat('FILE LIST TOO LONG\n')}

region_list <- c('M1','S1','PmD')


for (region_index in 1:length(file_list)){
    
  slopes_bfr_cue <- read.xlsx(file_list[region_index],sheet='slopes_bfr_cue_model',colNames=T)
  slopes_aft_cue <- read.xlsx(file_list[region_index],sheet='slopes_aft_cue_model',colNames=T)
  slopes_bfr_result <- read.xlsx(file_list[region_index],sheet='slopes_bfr_result_model',colNames=T)
  slopes_aft_result <- read.xlsx(file_list[region_index],sheet='slopes_aft_result_model',colNames=T)

  slopes_bfr_cue_sigall <- read.xlsx(file_list[region_index],sheet='sig_all_bfr_cue_model',colNames=T)
  slopes_aft_cue_sigall <- read.xlsx(file_list[region_index],sheet='sig_all_aft_cue_model',colNames=T)
  slopes_bfr_result_sigall <- read.xlsx(file_list[region_index],sheet='sig_all_bfr_result_model',colNames=T)
  slopes_aft_result_sigall <- read.xlsx(file_list[region_index],sheet='sig_all_aft_result_model',colNames=T)
  
  avg_bfr_cue <- c(mean(abs(slopes_bfr_cue$alpha)),mean(abs(slopes_bfr_cue$beta)),mean(abs(slopes_bfr_cue_sigall$alpha)),mean(abs(slopes_bfr_cue_sigall$beta)))
  avg_aft_cue <- c(mean(abs(slopes_aft_cue$alpha)),mean(abs(slopes_aft_cue$beta)),mean(abs(slopes_aft_cue_sigall$alpha)),mean(abs(slopes_aft_cue_sigall$beta)))
  avg_bfr_result <- c(mean(abs(slopes_bfr_result$alpha)),mean(abs(slopes_bfr_result$beta)),mean(abs(slopes_bfr_result_sigall$alpha)),mean(abs(slopes_bfr_result_sigall$beta)))
  avg_aft_result <- c(mean(abs(slopes_aft_result$alpha)),mean(abs(slopes_aft_result$beta)),mean(abs(slopes_aft_result_sigall$alpha)),mean(abs(slopes_aft_result_sigall$beta)))
  
  std_bfr_cue <- c(sd(abs(slopes_bfr_cue$alpha)),sd(abs(slopes_bfr_cue$beta)),sd(abs(slopes_bfr_cue_sigall$alpha)),sd(abs(slopes_bfr_cue_sigall$beta)))
  std_aft_cue <- c(sd(abs(slopes_aft_cue$alpha)),sd(abs(slopes_aft_cue$beta)),sd(abs(slopes_aft_cue_sigall$alpha)),sd(abs(slopes_aft_cue_sigall$beta)))
  std_bfr_result <- c(sd(abs(slopes_bfr_result$alpha)),sd(abs(slopes_bfr_result$beta)),sd(abs(slopes_bfr_result_sigall$alpha)),sd(abs(slopes_bfr_result_sigall$beta)))
  std_aft_result <- c(sd(abs(slopes_aft_result$alpha)),sd(abs(slopes_aft_result$beta)),sd(abs(slopes_aft_result_sigall$alpha)),sd(abs(slopes_aft_result_sigall$beta)))
  
  key <- c('alpha','beta','alpha','beta')
  cat <- c('one sig','one sig','both sig','both sig')
  
  df_bfr_cue <- data.frame(value= avg_bfr_cue,cat,key,std_bfr_cue)
  df_aft_cue <- data.frame(value = avg_aft_cue,cat,key,std_aft_cue)
  df_bfr_result <- data.frame(value = avg_bfr_result,cat,key,std_bfr_result)
  df_aft_result <- data.frame(value = avg_aft_result,cat,key,std_aft_result)
  
  bfr_cue_plt <- ggplot(df_bfr_cue) + aes(x=cat,y=value,fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="before cue") + geom_errorbar(aes(ymin=df_bfr_cue$value + df_bfr_cue$std_bfr_cue,ymax=df_bfr_cue$value - df_bfr_cue$std_bfr_cue),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
  aft_cue_plt <- ggplot(df_aft_cue) + aes(x=cat,y=value, fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="after cue") + geom_errorbar(aes(ymin=df_aft_cue$value + df_aft_cue$std_aft_cue,ymax=df_aft_cue$value - df_aft_cue$std_aft_cue),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
  bfr_result_plt <- ggplot(df_bfr_result) + aes(x=cat,y=value, fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="before result") + geom_errorbar(aes(ymin=df_bfr_result$value + df_bfr_result$std_bfr_result,ymax=df_bfr_result$value - df_bfr_result$std_bfr_result),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
  aft_result_plt <- ggplot(df_aft_result) + aes(x=cat,y=value, fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="after result") + geom_errorbar(aes(ymin=df_aft_result$value + df_aft_result$std_aft_result,ymax=df_aft_result$value - df_aft_result$std_aft_result),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
  
  png(paste(region_list[region_index],'_avg_abs_slopes.png',sep=""),width=8,height=6,units="in",res=500)
  multiplot(bfr_cue_plt,aft_cue_plt,bfr_result_plt,aft_result_plt,cols=2)
  
  #dev.off()
  graphics.off()
  
  ####################
  avg_bfr_cue <- c(mean(slopes_bfr_cue$alpha),mean(slopes_bfr_cue$beta),mean(slopes_bfr_cue_sigall$alpha),mean(slopes_bfr_cue_sigall$beta))
  avg_aft_cue <- c(mean(slopes_aft_cue$alpha),mean(slopes_aft_cue$beta),mean(slopes_aft_cue_sigall$alpha),mean(slopes_aft_cue_sigall$beta))
  avg_bfr_result <- c(mean(slopes_bfr_result$alpha),mean(slopes_bfr_result$beta),mean(slopes_bfr_result_sigall$alpha),mean(slopes_bfr_result_sigall$beta))
  avg_aft_result <- c(mean(slopes_aft_result$alpha),mean(slopes_aft_result$beta),mean(slopes_aft_result_sigall$alpha),mean(slopes_aft_result_sigall$beta))
  
  std_bfr_cue <- c(sd(slopes_bfr_cue$alpha),sd(slopes_bfr_cue$beta),sd(slopes_bfr_cue_sigall$alpha),sd(slopes_bfr_cue_sigall$beta))
  std_aft_cue <- c(sd(slopes_aft_cue$alpha),sd(slopes_aft_cue$beta),sd(slopes_aft_cue_sigall$alpha),sd(slopes_aft_cue_sigall$beta))
  std_bfr_result <- c(sd(slopes_bfr_result$alpha),sd(slopes_bfr_result$beta),sd(slopes_bfr_result_sigall$alpha),sd(slopes_bfr_result_sigall$beta))
  std_aft_result <- c(sd(slopes_aft_result$alpha),sd(slopes_aft_result$beta),sd(slopes_aft_result_sigall$alpha),sd(slopes_aft_result_sigall$beta))
  
  df_bfr_cue <- data.frame(value= avg_bfr_cue,cat,key,std_bfr_cue)
  df_aft_cue <- data.frame(value = avg_aft_cue,cat,key,std_aft_cue)
  df_bfr_result <- data.frame(value = avg_bfr_result,cat,key,std_bfr_result)
  df_aft_result <- data.frame(value = avg_aft_result,cat,key,std_aft_result)
  
  bfr_cue_plt <- ggplot(df_bfr_cue) + aes(x=cat,y=value,fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="before cue") + geom_errorbar(aes(ymin=df_bfr_cue$value + df_bfr_cue$std_bfr_cue,ymax=df_bfr_cue$value - df_bfr_cue$std_bfr_cue),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
  aft_cue_plt <- ggplot(df_aft_cue) + aes(x=cat,y=value, fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="after cue") + geom_errorbar(aes(ymin=df_aft_cue$value + df_aft_cue$std_aft_cue,ymax=df_aft_cue$value - df_aft_cue$std_aft_cue),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
  bfr_result_plt <- ggplot(df_bfr_result) + aes(x=cat,y=value, fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="before result") + geom_errorbar(aes(ymin=df_bfr_result$value + df_bfr_result$std_bfr_result,ymax=df_bfr_result$value - df_bfr_result$std_bfr_result),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
  aft_result_plt <- ggplot(df_aft_result) + aes(x=cat,y=value, fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="after result") + geom_errorbar(aes(ymin=df_aft_result$value + df_aft_result$std_aft_result,ymax=df_aft_result$value - df_aft_result$std_aft_result),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
  
  png(paste(region_list[region_index],'_avg_slopes.png',sep=""),width=8,height=6,units="in",res=500)
  multiplot(bfr_cue_plt,aft_cue_plt,bfr_result_plt,aft_result_plt,cols=2)
  
  #dev.off()
  graphics.off()
  
}


for (region_index in 1:length(region_list)){
  sheet_name = paste(region_list[region_index],"_dicts",sep="")
  percs <- read.xlsx('percs_workbook.xlsx',sheet=sheet_name,colNames=T)
  
  bfr_cue <- percs$bfr_cue_model
  aft_cue <- percs$aft_cue_model
  bfr_result <- percs$bfr_result_model
  aft_result <- percs$aft_result_model
  
  num_units = bfr_cue[5]
  
  bfr_cue_perc_slopes <- bfr_cue[1]
  bfr_cue_perc_slopes_sigall <- bfr_cue[2]
  bfr_cue_num_slopes <- bfr_cue[3]
  bfr_cue_num_slopes_sigall <- bfr_cue[4]
  aft_cue_perc_slopes <- aft_cue[1]
  aft_cue_perc_slopes_sigall <- aft_cue[2]
  aft_cue_num_slopes <- aft_cue[3]
  aft_cue_num_slopes_sigall <- aft_cue[4]  
  bfr_result_perc_slopes <- bfr_result[1]
  bfr_result_perc_slopes_sigall <- bfr_result[2]
  bfr_result_num_slopes <- bfr_result[3]
  bfr_result_num_slopes_sigall <- bfr_result[4]  
  aft_result_perc_slopes <- aft_result[1]
  aft_result_perc_slopes_sigall <- aft_result[2]
  aft_result_num_slopes <- aft_result[3]
  aft_result_num_slopes_sigall <- aft_result[4]
  
  bfr_cue_non <- 1- bfr_cue_perc_slopes 
  bfr_cue_non_sigall <- 1 - bfr_cue_perc_slopes_sigall
  aft_cue_non <- 1- aft_cue_perc_slopes 
  aft_cue_non_sigall <- 1 - aft_cue_perc_slopes_sigall
  bfr_result_non <- 1- bfr_result_perc_slopes 
  bfr_result_non_sigall <- 1 - bfr_result_perc_slopes_sigall
  aft_result_non <- 1- aft_result_perc_slopes 
  aft_result_non_sigall <- 1 - aft_result_perc_slopes_sigall
  
  labs <- c('perc sig','perc not sig')
  
  bfr_cue_vals <- c(bfr_cue_perc_slopes,bfr_cue_non)
  bfr_cue_sigall_vals <- c(bfr_cue_perc_slopes_sigall,bfr_cue_non_sigall)
  aft_cue_vals <- c(aft_cue_perc_slopes,aft_cue_non)
  aft_cue_sigall_vals <- c(aft_cue_perc_slopes_sigall,aft_cue_non_sigall)
  bfr_result_vals <- c(bfr_result_perc_slopes,bfr_result_non)
  bfr_result_sigall_vals <- c(bfr_result_perc_slopes_sigall,bfr_result_non_sigall)
  aft_result_vals <- c(aft_result_perc_slopes,aft_result_non)
  aft_result_sigall_vals <- c(aft_result_perc_slopes_sigall,aft_result_non_sigall)
  
  #can't make NA, get plotting error later
  #if (identical(bfr_result_vals,numeric(0))){bfr_result_vals=c(0,0,0,0)}
  #if (identical(bfr_result_sigall_vals,numeric(0))){bfr_result_sigall_vals=c(0,0,0,0)}
    
  bfr_cue_df <- data.frame(bfr_cue_vals,labs)
  bfr_cue_sigall_df <- data.frame(bfr_cue_sigall_vals,labs)
  aft_cue_df <- data.frame(aft_cue_vals,labs)
  aft_cue_sigall_df <- data.frame(aft_cue_sigall_vals,labs)
  bfr_result_df <- data.frame(bfr_result_vals,labs)
  bfr_result_sigall_df <- data.frame(bfr_result_sigall_vals,labs)
  aft_result_df <- data.frame(aft_result_vals,labs)
  aft_result_sigall_df <- data.frame(aft_result_sigall_vals,labs)
  
  png(paste('sig_plots',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  bfr_cue_plt <- ggplot(bfr_cue_df,aes(x="",y=bfr_cue_vals,fill=labs)) + geom_bar(width=1,stat="identity") + coord_polar('y',start=0)
  bfr_cue_plt <- bfr_cue_plt + theme_minimal() +  theme(axis.title=element_blank(),axis.text=element_blank(),panel.grid=element_blank(),axis.ticks=element_blank())
  bfr_cue_plt <- bfr_cue_plt + geom_text(aes(y = bfr_cue_vals/2 + c(0, cumsum(bfr_cue_vals)[-length(bfr_cue_vals)]),label = scales::percent(bfr_cue_vals)))
  bfr_cue_plt <- bfr_cue_plt + labs(title="before cue")
  
  aft_cue_plt <- ggplot(aft_cue_df,aes(x="",y=aft_cue_vals,fill=labs)) + geom_bar(width=1,stat="identity") + coord_polar('y',start=0)
  aft_cue_plt <- aft_cue_plt + theme_minimal() +  theme(axis.title=element_blank(),axis.text=element_blank(),panel.grid=element_blank(),axis.ticks=element_blank())
  aft_cue_plt <- aft_cue_plt + geom_text(aes(y = aft_cue_vals/2 + c(0, cumsum(aft_cue_vals)[-length(aft_cue_vals)]),label = scales::percent(aft_cue_vals)))
  aft_cue_plt <- aft_cue_plt + labs(title="after cue")
  
  bfr_result_plt <- ggplot(bfr_result_df,aes(x="",y=bfr_result_vals,fill=labs)) + geom_bar(width=1,stat="identity") + coord_polar('y',start=0)
  bfr_result_plt <- bfr_result_plt + theme_minimal() +  theme(axis.title=element_blank(),axis.text=element_blank(),panel.grid=element_blank(),axis.ticks=element_blank())
  bfr_result_plt <- bfr_result_plt + geom_text(aes(y = bfr_result_vals/2 + c(0, cumsum(bfr_result_vals)[-length(bfr_result_vals)]),label = scales::percent(bfr_result_vals)))
  bfr_result_plt <- bfr_result_plt + labs(title="before result")
  
  aft_result_plt <- ggplot(aft_result_df,aes(x="",y=aft_result_vals,fill=labs)) + geom_bar(width=1,stat="identity") + coord_polar('y',start=0)
  aft_result_plt <- aft_result_plt + theme_minimal() +  theme(axis.title=element_blank(),axis.text=element_blank(),panel.grid=element_blank(),axis.ticks=element_blank())
  aft_result_plt <- aft_result_plt + geom_text(aes(y = aft_result_vals/2 + c(0, cumsum(aft_result_vals)[-length(aft_result_vals)]),label = scales::percent(aft_result_vals)))
  aft_result_plt <- aft_result_plt + labs(title="after result")
  
  multiplot(bfr_cue_plt,aft_cue_plt,bfr_result_plt,aft_result_plt,cols=2)
  graphics.off()

  png(paste('both_sig_plots',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  bfr_cue_plt_sigall <- ggplot(bfr_cue_sigall_df,aes(x="",y=bfr_cue_sigall_vals,fill=labs)) + geom_bar(width=1,stat="identity") + coord_polar('y',start=0)
  bfr_cue_plt_sigall <- bfr_cue_plt_sigall + theme_minimal() +  theme(axis.title=element_blank(),axis.text=element_blank(),panel.grid=element_blank(),axis.ticks=element_blank())
  bfr_cue_plt_sigall <- bfr_cue_plt_sigall + geom_text(aes(y = bfr_cue_sigall_vals/2 + c(0, cumsum(bfr_cue_sigall_vals)[-length(bfr_cue_sigall_vals)]),label = scales::percent(bfr_cue_sigall_vals)))
  bfr_cue_plt_sigall <- bfr_cue_plt_sigall + labs(title="before cue (both sig)")
  
  aft_cue_plt_sigall <- ggplot(aft_cue_sigall_df,aes(x="",y=aft_cue_sigall_vals,fill=labs)) + geom_bar(width=1,stat="identity") + coord_polar('y',start=0)
  aft_cue_plt_sigall <- aft_cue_plt_sigall + theme_minimal() +  theme(axis.title=element_blank(),axis.text=element_blank(),panel.grid=element_blank(),axis.ticks=element_blank())
  aft_cue_plt_sigall <- aft_cue_plt_sigall + geom_text(aes(y = aft_cue_sigall_vals/2 + c(0, cumsum(aft_cue_sigall_vals)[-length(aft_cue_sigall_vals)]),label = scales::percent(aft_cue_sigall_vals)))
  aft_cue_plt_sigall <- aft_cue_plt_sigall + labs(title="after cue (both sig)")
  
  bfr_result_plt_sigall <- ggplot(bfr_result_sigall_df,aes(x="",y=bfr_result_sigall_vals,fill=labs)) + geom_bar(width=1,stat="identity") + coord_polar('y',start=0)
  bfr_result_plt_sigall <- bfr_result_plt_sigall + theme_minimal() +  theme(axis.title=element_blank(),axis.text=element_blank(),panel.grid=element_blank(),axis.ticks=element_blank())
  bfr_result_plt_sigall <- bfr_result_plt_sigall + geom_text(aes(y = bfr_result_sigall_vals/2 + c(0, cumsum(bfr_result_sigall_vals)[-length(bfr_result_sigall_vals)]),label = scales::percent(bfr_result_sigall_vals)))
  bfr_result_plt_sigall <- bfr_result_plt_sigall + labs(title="before result (both sig)")
  
  aft_result_plt_sigall <- ggplot(aft_result_sigall_df,aes(x="",y=aft_result_sigall_vals,fill=labs)) + geom_bar(width=1,stat="identity") + coord_polar('y',start=0)
  aft_result_plt_sigall <- aft_result_plt_sigall + theme_minimal() +  theme(axis.title=element_blank(),axis.text=element_blank(),panel.grid=element_blank(),axis.ticks=element_blank())
  aft_result_plt_sigall <- aft_result_plt_sigall + geom_text(aes(y = aft_result_sigall_vals/2 + c(0, cumsum(aft_result_sigall_vals)[-length(aft_result_sigall_vals)]),label = scales::percent(aft_result_sigall_vals)))
  aft_result_plt_sigall <- aft_result_plt_sigall + labs(title="after result (both sig)")
  
  multiplot(bfr_cue_plt_sigall,aft_cue_plt_sigall,bfr_result_plt_sigall,aft_result_plt_sigall,cols=2)
  graphics.off()
  
  ####################################
 
  bfr_cue_alpha_sig <- bfr_cue[10]
  bfr_cue_beta_sig <- bfr_cue[11]
  bfr_cue_both_sig <- bfr_cue[2]
  aft_cue_alpha_sig <- aft_cue[10]
  aft_cue_beta_sig <- aft_cue[11]
  aft_cue_both_sig <- aft_cue[2]
  bfr_result_alpha_sig <- bfr_result[10]
  bfr_result_beta_sig <- bfr_result[11]
  bfr_result_both_sig <- bfr_result[2]
  aft_result_alpha_sig <- aft_result[10]
  aft_result_beta_sig <- aft_result[11]
  aft_result_both_sig <- aft_result[2]
  
  bfr_cue_non <- 1 - (bfr_cue_alpha_sig + bfr_cue_beta_sig + bfr_cue_both_sig)
  aft_cue_non <- 1 - (aft_cue_alpha_sig + aft_cue_beta_sig + aft_cue_both_sig)
  bfr_result_non <- 1 - (bfr_result_alpha_sig + bfr_result_beta_sig + bfr_result_both_sig)
  aft_result_non <- 1 - (aft_result_alpha_sig + aft_result_beta_sig + aft_result_both_sig)

  labs <- c('alpha only sig','beta only sig','both sig','no sig')
  
  bfr_cue_vals <- c(bfr_cue_alpha_sig,bfr_cue_beta_sig,bfr_cue_both_sig,bfr_cue_non)
  aft_cue_vals <- c(aft_cue_alpha_sig,aft_cue_beta_sig,aft_cue_both_sig,aft_cue_non)
  bfr_result_vals <- c(bfr_result_alpha_sig,bfr_result_beta_sig,bfr_result_both_sig,bfr_result_non)
  aft_result_vals <- c(aft_result_alpha_sig,aft_result_beta_sig,aft_result_both_sig,aft_result_non)

  #if (identical(bfr_result_vals,numeric(0))){bfr_result_vals=c(0,0,0,0)}
  
  bfr_cue_df <- data.frame(perc=bfr_cue_vals,labs,type='bfr_cue')
  aft_cue_df <- data.frame(perc=aft_cue_vals,labs,type='aft_cue')
  bfr_result_df <- data.frame(perc=bfr_result_vals,labs,type='bfr_result')
  aft_result_df <- data.frame(perc=aft_result_vals,labs,type='aft_result')
  
  bfr_cue_df <- bfr_cue_df[rev(order(bfr_cue_df$labs)),]
  aft_cue_df <- aft_cue_df[rev(order(aft_cue_df$labs)),]
  bfr_result_df <- bfr_result_df[rev(order(bfr_result_df$labs)),]
  aft_result_df <- aft_result_df[rev(order(aft_result_df$labs)),]
  
  bfr_cue_df <- ddply(bfr_cue_df,.(type),transform,position=(cumsum(bfr_cue_df$perc)-0.5*bfr_cue_df$perc))
  aft_cue_df <- ddply(aft_cue_df,.(type),transform,position=(cumsum(aft_cue_df$perc)-0.5*aft_cue_df$perc))
  bfr_result_df <- ddply(bfr_result_df,.(type),transform,position=(cumsum(bfr_result_df$perc)-0.5*bfr_result_df$perc))
  aft_result_df <- ddply(aft_result_df,.(type),transform,position=(cumsum(aft_result_df$perc)-0.5*aft_result_df$perc))
  
  total_units <- bfr_cue[5]
  bfr_cue_nums <- c(bfr_cue[5]-bfr_cue[4]-bfr_cue[13]-bfr_cue[12],bfr_cue[4],bfr_cue[13],bfr_cue[12])
  aft_cue_nums <- c(aft_cue[5]-aft_cue[4]-aft_cue[13]-aft_cue[12],aft_cue[4],aft_cue[13],aft_cue[12])
  bfr_result_nums <- c(bfr_result[5]-bfr_result[4]-bfr_result[13]-bfr_result[12],bfr_result[4],bfr_result[13],bfr_result[12])
  aft_result_nums <- c(aft_result[5]-aft_result[4]-aft_result[13]-aft_result[12],aft_result[4],aft_result[13],aft_result[12])
  
  bfr_cue_df <- ddply(bfr_cue_df,.(type),transform,label=paste(scales::percent(bfr_cue_df$perc),' n=',bfr_cue_nums,sep=""))
  aft_cue_df <- ddply(aft_cue_df,.(type),transform,label=paste(scales::percent(aft_cue_df$perc),' n=',aft_cue_nums,sep=""))
  bfr_result_df <- ddply(bfr_result_df,.(type),transform,label=paste(scales::percent(bfr_result_df$perc),' n=',bfr_result_nums,sep=""))
  aft_result_df <- ddply(aft_result_df,.(type),transform,label=paste(scales::percent(aft_result_df$perc),' n=',aft_result_nums,sep=""))
  
  png(paste('all_bar_plotted_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  
  df_all <- rbind(bfr_cue_df,aft_cue_df,bfr_result_df,aft_result_df)
  df_all <- df_all[which(df_all$perc > 0),]
  
  bar_plt <- ggplot() + geom_bar(aes(x=df_all$type,y=df_all$perc,fill=df_all$labs),data=df_all,stat="identity") 
  bar_plt <- bar_plt + labs(title=region_list[region_index],fill="",x="Time Window",y="Percentage") + scale_fill_manual(values=c("lightblue","seagreen","grey","slateblue"))
  bar_plt <- bar_plt + geom_text(aes(x=df_all$type,y=df_all$position,label=df_all$label),size=3.5,stat="identity")
  
  plot(bar_plt)
  graphics.off()
  
  percsig_df_all <- df_all
  percsig_bfr_cue <- bfr_cue_df    
  percsig_aft_cue <- aft_cue_df
  percsig_bfr_res <- bfr_result_df
  percsig_aft_res <- aft_result_df
  percsig_total_units <- total_units
  percsig_bfr_cue_nums <- bfr_cue_nums
  percsig_aft_cue_nums <- aft_cue_nums
  percsig_bfr_res_nums <- bfr_result_nums
  percsig_aft_res_nums <- aft_result_nums
  
  save(percsig_df_all,percsig_bfr_cue,percsig_aft_cue,percsig_bfr_res,percsig_aft_res,percsig_total_units,percsig_bfr_cue_nums,percsig_aft_cue_nums,percsig_bfr_res_nums,percsig_aft_res_nums,file=paste('alphabeta_',region_list[region_index],'_percsig_avg.RData',sep=""))
  
  
  # png(paste('all_pie_plotted_',region_list[region_ind],'.png',sep=""),width=8,height=6,units="in",res=500)
  # bfr_cue_plt <- ggplot(bfr_cue_df,aes(x="",y=bfr_cue_vals,fill=labs)) + geom_bar(width=1,stat="identity") + coord_polar('y',start=0)
  # bfr_cue_plt <- bfr_cue_plt + theme_minimal() +  theme(axis.title=element_blank(),axis.text=element_blank(),panel.grid=element_blank(),axis.ticks=element_blank())
  # #bfr_cue_plt <- bfr_cue_plt + geom_text(aes(y = bfr_cue_vals/4 + c(0, cumsum(bfr_cue_vals)[-length(bfr_cue_vals)]),label = scales::percent(bfr_cue_vals)))
  # bfr_cue_plt <- bfr_cue_plt + labs(title="before cue")
  # 
  # aft_cue_plt <- ggplot(aft_cue_df,aes(x="",y=aft_cue_vals,fill=labs)) + geom_bar(width=1,stat="identity") + coord_polar('y',start=0)
  # aft_cue_plt <- aft_cue_plt + theme_minimal() +  theme(axis.title=element_blank(),axis.text=element_blank(),panel.grid=element_blank(),axis.ticks=element_blank())
  # #aft_cue_plt <- aft_cue_plt + geom_text(aes(y = aft_cue_vals/4 + c(0, cumsum(aft_cue_vals)[-length(aft_cue_vals)]),label = scales::percent(aft_cue_vals)))
  # aft_cue_plt <- aft_cue_plt + labs(title="after cue")
  # 
  # bfr_result_plt <- ggplot(bfr_result_df,aes(x="",y=bfr_result_vals,fill=labs)) + geom_bar(width=1,stat="identity") + coord_polar('y',start=0)
  # bfr_result_plt <- bfr_result_plt + theme_minimal() +  theme(axis.title=element_blank(),axis.text=element_blank(),panel.grid=element_blank(),axis.ticks=element_blank())
  # #bfr_result_plt <- bfr_result_plt + geom_text(aes(y = bfr_result_vals/4 + c(0, cumsum(bfr_result_vals)[-length(bfr_result_vals)]),label = scales::percent(bfr_result_vals)))
  # bfr_result_plt <- bfr_result_plt + labs(title="before result")
  # 
  # aft_result_plt <- ggplot(aft_result_df,aes(x="",y=aft_result_vals,fill=labs)) + geom_bar(width=1,stat="identity") + coord_polar('y',start=0)
  # aft_result_plt <- aft_result_plt + theme_minimal() +  theme(axis.title=element_blank(),axis.text=element_blank(),panel.grid=element_blank(),axis.ticks=element_blank())
  # #aft_result_plt <- aft_result_plt + geom_text(aes(y = aft_result_vals/4 + c(0, cumsum(aft_result_vals)[-length(aft_result_vals)]),label = scales::percent(aft_result_vals)))
  # aft_result_plt <- aft_result_plt + labs(title="after result")
  # 
  # multiplot(bfr_cue_plt,aft_cue_plt,bfr_result_plt,aft_result_plt,cols=2)
  # graphics.off()
  # 
  # #plot as bar
  # # png(paste('all_bar_plotted_',region_list[region_ind],'.png',sep=""),width=8,height=6,units="in",res=500)
  # # 
  # # merge_temp <- merge(bfr_cue_df,aft_cue_df)
  # # merge_temp <- merge(merge_temp,bfr_result_df)
  # # merged <- merge(merge_temp,aft_result_df)
  # # melted <- melt(merged,id.vars = "labs")
  # # melted <- melted[which(melted$value > 0),]
  # # 
  # # bar_plt <- ggplot() + geom_bar(aes(x=melted$variable,y=melted$value,fill=melted$labs),data=melted,stat="identity") 
  # # bar_plt <- bar_plt + labs(t itle=region_list[region_ind],fill="",x="Time Window",y="Percentage") + scale_fill_manual(values=c("lightblue","seagreen","grey","slateblue"))
  # # 
  # # #bar_plt <- bar_plt + geom_text(data=melted,aes(x=melted$variable,y=melted$value,label=scales::percent(melted$value)),size=4)
  # # 
  # # plot(bar_plt)
  # # graphics.off()
  
  ########################
  bfr_cue_both_pos <- bfr_cue[14]
  bfr_cue_both_neg <- bfr_cue[15]
  bfr_cue_alpha_pos <- bfr_cue[16]
  bfr_cue_beta_pos <- bfr_cue[17]
  aft_cue_both_pos <- aft_cue[14]
  aft_cue_both_neg <- aft_cue[15]
  aft_cue_alpha_pos <- aft_cue[16]
  aft_cue_beta_pos <- aft_cue[17]
  bfr_result_both_pos <- bfr_result[14]
  bfr_result_both_neg <- bfr_result[15]
  bfr_result_alpha_pos <- bfr_result[16]
  bfr_result_beta_pos <- bfr_result[17]
  aft_result_both_pos <- aft_result[14]
  aft_result_both_neg <- aft_result[15]
  aft_result_alpha_pos <- aft_result[16]
  aft_result_beta_pos <- aft_result[17]
  
  labs <- c("both pos","both neg","alpha pos","beta pos")
  bfr_cue_vals <- c(bfr_cue_both_pos,bfr_cue_both_neg,bfr_cue_alpha_pos,bfr_cue_beta_pos)
  aft_cue_vals <- c(aft_cue_both_pos,aft_cue_both_neg,aft_cue_alpha_pos,aft_cue_beta_pos)
  bfr_result_vals <- c(bfr_result_both_pos,bfr_result_both_neg,bfr_result_alpha_pos,bfr_result_beta_pos)
  aft_result_vals <- c(aft_result_both_pos,aft_result_both_neg,aft_result_alpha_pos,aft_result_beta_pos)
  
  #if (identical(bfr_result_vals,numeric(0))){bfr_result_vals=c(0,0,0,0)}
  #if (is.null(bfr_result_vals)){bfr_result_vals=c(0,0,0,0)}
  
  bfr_cue_df <- data.frame(perc=bfr_cue_vals,labs,type='bfr_cue')
  aft_cue_df <- data.frame(perc=aft_cue_vals,labs,type='aft_cue')
  bfr_result_df <- data.frame(perc=bfr_result_vals,labs,type='bfr_result')
  aft_result_df <- data.frame(perc=aft_result_vals,labs,type='aft_result')
  
  bfr_cue_df <- bfr_cue_df[rev(order(bfr_cue_df$labs)),]
  aft_cue_df <- aft_cue_df[rev(order(aft_cue_df$labs)),]
  bfr_result_df <- bfr_result_df[rev(order(bfr_result_df$labs)),]
  aft_result_df <- aft_result_df[rev(order(aft_result_df$labs)),]
  
  bfr_cue_df <- ddply(bfr_cue_df,.(type),transform,position=(cumsum(bfr_cue_df$perc)-0.5*bfr_cue_df$perc))
  aft_cue_df <- ddply(aft_cue_df,.(type),transform,position=(cumsum(aft_cue_df$perc)-0.5*aft_cue_df$perc))
  bfr_result_df <- ddply(bfr_result_df,.(type),transform,position=(cumsum(bfr_result_df$perc)-0.5*bfr_result_df$perc))
  aft_result_df <- ddply(aft_result_df,.(type),transform,position=(cumsum(aft_result_df$perc)-0.5*aft_result_df$perc))

  total_units = bfr_cue[5]
  bfr_cue_onesig_num = bfr_cue[3]
  bfr_cue_nums= c(bfr_cue[14]*bfr_cue_onesig_num,bfr_cue[15]*bfr_cue_onesig_num,bfr_cue[17]*bfr_cue_onesig_num,bfr_cue[16]*bfr_cue_onesig_num)
  aft_cue_onesig_num = aft_cue[3]
  aft_cue_nums= c(aft_cue[14]*aft_cue_onesig_num,aft_cue[15]*aft_cue_onesig_num,aft_cue[17]*aft_cue_onesig_num,aft_cue[16]*aft_cue_onesig_num)
  bfr_result_onesig_num = bfr_result[3]
  bfr_result_nums= c(bfr_result[14]*bfr_result_onesig_num,bfr_result[15]*bfr_result_onesig_num,bfr_result[17]*bfr_result_onesig_num,bfr_result[16]*bfr_result_onesig_num)
  aft_result_onesig_num = aft_result[3]
  aft_result_nums= c(aft_result[14]*aft_result_onesig_num,aft_result[15]*aft_result_onesig_num,aft_result[17]*aft_result_onesig_num,aft_result[16]*aft_result_onesig_num)
  
  bfr_cue_df <- ddply(bfr_cue_df,.(type),transform,label=paste(scales::percent(bfr_cue_df$perc),' n=',bfr_cue_nums,sep=""))
  aft_cue_df <- ddply(aft_cue_df,.(type),transform,label=paste(scales::percent(aft_cue_df$perc),' n=',aft_cue_nums,sep=""))
  bfr_result_df <- ddply(bfr_result_df,.(type),transform,label=paste(scales::percent(bfr_result_df$perc),' n=',bfr_result_nums,sep=""))
  aft_result_df <- ddply(aft_result_df,.(type),transform,label=paste(scales::percent(aft_result_df$perc),' n=',aft_result_nums,sep=""))
  
  png(paste('signs_bar_plotted_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)

  df_all <- rbind(bfr_cue_df,aft_cue_df,bfr_result_df,aft_result_df)
  df_all <- df_all[which(df_all$perc > 0),]
  
  bar_plt <- ggplot() + geom_bar(aes(x=df_all$type,y=df_all$perc,fill=df_all$labs),data=df_all,stat="identity") 
  bar_plt <- bar_plt + labs(title=region_list[region_index],fill="",x="Time Window",y="Percentage") + scale_fill_manual(values=c("lightblue","seagreen","grey","slateblue"))
  bar_plt <- bar_plt + geom_text(aes(x=df_all$type,y=df_all$position,label=df_all$label),size=4,stat="identity")
  
  plot(bar_plt)
  graphics.off()
  
}

#####################
plot_newmv <- function(mv_array,region_key,type_key){
  val_list <- vector("list",dim(mv_array)[1])
  mtv_list <- vector("list",dim(mv_array)[1])
  
  sig_val <- 0
  sig_mtv <- 0
  sig_val_slopes <- c()
  sig_mtv_slopes <- c()
  all_val_slopes <- c()
  all_mtv_slopes <- c()
  alpha_list <- c()
  beta_list <- c()
  
  for (unit_ind in 1:dim(mv_array)[1]){
    unit_array <- mv_array[unit_ind,,]
    unit_num <- unit_array[1,8]
    
    x_val <- unit_array[,5]
    x_mtv <- unit_array[,6]
    fr <- unit_array[,7]
    
    df_val <- data.frame(x_val,fr)
    df_mtv <- data.frame(x_mtv,fr)
    
    val_plt <- ggplot(df_val,aes(x_val,fr)) + geom_point(shape=1) + geom_smooth(method=lm) + labs(title=paste(region_key,type_key),x="Value",y="Normalized Firing Rate")
    mtv_plt <- ggplot(df_mtv,aes(x_mtv,fr)) + geom_point(shape=1) + geom_smooth(method=lm)+ labs(title=paste(region_key,type_key),x="Motivation",y="Normalized Firing Rate")
    
    #TODO fix, doesn't quite work
    val_lm <- tidy(lm(fr~x_val,data=df_val))
    mtv_lm <- tidy(lm(fr~x_mtv,data=df_mtv))
    
    #cat(mtv_lm$estimate[2],'\n')
    
    #will this (or the other complete cases) affect outcomes? see what's causing
    val_lm <- val_lm[complete.cases(val_lm$p.value),]
    mtv_lm <- mtv_lm[complete.cases(mtv_lm$p.value),]
    
    #cat(region_key,' ',type_key, ' ',val_lm$p.value[2], ' ', unit_ind,'\n')
    
    if (length(val_lm$p.value) == 0){
      sig_val <- sig_val
      sig_val_slopes <- sig_val_slopes
    }else if (val_lm$p.value[2] <= 0.05){
      sig_val <- sig_val + 1
      sig_val_slopes <- c(sig_val_slopes,val_lm$estimate[2])
      }
      
    if (length(val_lm$p.value) == 0){
      sig_mtv <- sig_mtv
      sig_mtv_slopes <- sig_mtv_slopes
    }else if (mtv_lm$p.value[2] <= 0.05){
      sig_mtv <- sig_mtv + 1
      sig_mtv_slopes <- c(sig_mtv_slopes,mtv_lm$estimate[2])
      }

    all_val_slopes <- c(all_val_slopes,val_lm$estimate[2])
    all_mtv_slopes <- c(all_mtv_slopes,mtv_lm$estimate[2])
    #val_list[unit_mind] <- val_lm
    #mtv_list[unit_ind] <- mtv_lm
    
    alpha_list <- c(alpha_list,unit_array[1,3])
    beta_list <- c(beta_list,unit_array[1,4])

    #No plot for now
    # if (all_alphabeta_bool){png(paste(region_list[region_index],'_',type_key,'_unit',unit_num,'_all_mtv_val_plt.png',sep=""),width=8,height=6,units="in",res=500)
    # }else if (avg_alphabeta_bool){png(paste(region_list[region_index],'_',type_key,'_unit',unit_num,'_avg_mtv_val_plt.png',sep=""),width=8,height=6,units="in",res=500)
    # }else{png(paste(region_list[region_index],'_',type_key,'_unit',unit_num,'_mtv_val_plt.png',sep=""),width=8,height=6,units="in",res=500)}
    # 
    # multiplot(val_plt,mtv_plt,cols=2)
    # graphics.off()    
   
  }
  perc_val_sig <- sig_val / dim(mv_array)[1]
  perc_mtv_sig <- sig_mtv / dim(mv_array)[1]
  
  if (is.null(sig_val_slopes)){sig_val_slopes=NA}
  if (is.null(sig_mtv_slopes)){sig_mtv_slopes=NA}
  if (is.null(all_val_slopes)){all_val_slopes=NA}
  if (is.null(all_mtv_slopes)){all_mtv_slopes=NA}
  
  sig_val_slopes.df <- data.frame(slopes=sig_val_slopes,type='val')
  sig_mtv_slopes.df <- data.frame(slopes=sig_mtv_slopes,type='mtv')
  
  all_val_slopes.df <- data.frame(slopes=all_val_slopes,type='val')
  all_mtv_slopes.df <- data.frame(slopes=all_mtv_slopes,type='mtv')
  
  sig_slopes <- rbind(sig_val_slopes.df,sig_mtv_slopes.df)
  all_slopes <- rbind(all_val_slopes.df,all_mtv_slopes.df)
  
  sig_slopes <- sig_slopes[complete.cases(sig_slopes$slopes),]
  all_slopes <- all_slopes[complete.cases(all_slopes$slopes),]
  
  if (all_alphabeta_bool){png(paste('slope_hist_all_',region_key,'_',type_key,'.png',sep=""),width=8,height=6,units="in",res=500)
  }else if (avg_alphabeta_bool){png(paste('slope_hist_avg_',region_key,'_',type_key,'.png',sep=""),width=8,height=6,units="in",res=500)
  }else{png(paste('slope_hist_',region_key,'_',type_key,'.png',sep=""),width=8,height=6,units="in",res=500)}
  
  
  sig_plt <- ggplot(sig_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  sig_plt <- sig_plt + labs(title=paste(region_key,type_key,'significant')) #+ xlim(-1.0,1.0)

  all_plt <- ggplot(all_slopes,aes(slopes,fill=type)) + geom_histogram(alpha=0.5,position='identity',binwidth=0.2)
  all_plt <- all_plt + labs(title=paste(region_key,type_key,'all')) #+ xlim(-1.0,1.0)
  
  multiplot(sig_plt,all_plt,cols=1)
    
  graphics.off()
  
  saveRDS(sig_slopes,paste(region_key,'_',type_key,'_sig_slopes_avg.rds',sep=''))
  saveRDS(all_slopes,paste(region_key,'_',type_key,'_all_slopes_avg.rds',sep=''))
  
  return_list <- list(val = val_list,mtv = mtv_list,perc_v = perc_val_sig,perc_m = perc_mtv_sig,sig_slopes=sig_slopes,all_slopes=all_slopes,alpha_list=alpha_list,beta_list=beta_list)
  return(return_list)
}


#####################


type_list = c('bfr_cue','aft_cue','bfr_result','aft_result')
for (region_index in 1:length(region_list)){
  
  if (all_alphabeta_bool){
    npy_readin <- readMat(paste(region_list[region_index],'_dicts_bfr_cue_model_all_mv_array.mat',sep=""))
    bfr_cue_mv_array <- npy_readin$mv.array
    npy_readin <- readMat(paste(region_list[region_index],'_dicts_aft_cue_model_all_mv_array.mat',sep=""))
    aft_cue_mv_array <- npy_readin$mv.array
    npy_readin <- readMat(paste(region_list[region_index],'_dicts_bfr_result_model_all_mv_array.mat',sep=""))
    bfr_result_mv_array <- npy_readin$mv.array
    npy_readin <- readMat(paste(region_list[region_index],'_dicts_aft_result_model_all_mv_array.mat',sep=""))
    aft_result_mv_array <- npy_readin$mv.array
  }else if(avg_alphabeta_bool){
    npy_readin <- readMat(paste(region_list[region_index],'_dicts_bfr_cue_model_all_avg_mv_array.mat',sep=""))
    bfr_cue_mv_array <- npy_readin$mv.array
    npy_readin <- readMat(paste(region_list[region_index],'_dicts_aft_cue_model_all_avg_mv_array.mat',sep=""))
    aft_cue_mv_array <- npy_readin$mv.array
    npy_readin <- readMat(paste(region_list[region_index],'_dicts_bfr_result_model_all_avg_mv_array.mat',sep=""))
    bfr_result_mv_array <- npy_readin$mv.array
    npy_readin <- readMat(paste(region_list[region_index],'_dicts_aft_result_model_all_avg_mv_array.mat',sep=""))
    aft_result_mv_array <- npy_readin$mv.array
  }else{
    npy_readin <- readMat(paste(region_list[region_index],'_dicts_bfr_cue_model_mv_array.mat',sep=""))
    bfr_cue_mv_array <- npy_readin$mv.array
    npy_readin <- readMat(paste(region_list[region_index],'_dicts_aft_cue_model_mv_array.mat',sep=""))
    aft_cue_mv_array <- npy_readin$mv.array
    npy_readin <- readMat(paste(region_list[region_index],'_dicts_bfr_result_model_mv_array.mat',sep=""))
    bfr_result_mv_array <- npy_readin$mv.array
    npy_readin <- readMat(paste(region_list[region_index],'_dicts_aft_result_model_mv_array.mat',sep=""))
    aft_result_mv_array <- npy_readin$mv.array
  }
  
  bfr_cue_list <- plot_newmv(bfr_cue_mv_array,region_list[region_index],'bfr_cue')
  aft_cue_list <- plot_newmv(aft_cue_mv_array,region_list[region_index],'aft_cue')
  bfr_result_list <- plot_newmv(bfr_result_mv_array,region_list[region_index],'bfr_result')
  aft_result_list <- plot_newmv(aft_result_mv_array,region_list[region_index],'aft_result')
  
  all_list <- list(bfr_cue_list = bfr_cue_list,aft_cue_list = aft_cue_list,bfr_result_list = bfr_result_list,aft_result_list = aft_result_list)
  region_name <- paste(region_list[region_index],'_vals',sep="")
  assign(region_name,all_list)
    
  bfr_cue_sig_pos_slope <- bfr_cue_list$sig_slopes[bfr_cue_list$sig_slopes$slopes > 0.1,]
  bfr_cue_sig_neg_slope <- bfr_cue_list$sig_slopes[bfr_cue_list$sig_slopes$slopes < -0.1,]
  temp <- bfr_cue_list$sig_slopes[bfr_cue_list$sig_slopes$slopes <= 0.1,]
  bfr_cue_sig_zero_slope <- temp[temp$slopes >= -0.1,]
  aft_cue_sig_pos_slope <- aft_cue_list$sig_slopes[aft_cue_list$sig_slopes$slopes > 0.1,]
  aft_cue_sig_neg_slope <- aft_cue_list$sig_slopes[aft_cue_list$sig_slopes$slopes < -0.1,]
  temp <- aft_cue_list$sig_slopes[aft_cue_list$sig_slopes$slopes <= 0.1,]
  aft_cue_sig_zero_slope <- temp[temp$slopes >= -0.1,]
  bfr_result_sig_pos_slope <- bfr_result_list$sig_slopes[bfr_result_list$sig_slopes$slopes > 0.1,]
  bfr_result_sig_neg_slope <- bfr_result_list$sig_slopes[bfr_result_list$sig_slopes$slopes < -0.1,]
  temp <- bfr_result_list$sig_slopes[bfr_result_list$sig_slopes$slopes <= 0.1,]
  bfr_result_sig_zero_slope <- temp[temp$slopes >= -0.1,]
  aft_result_sig_pos_slope <- aft_result_list$sig_slopes[aft_result_list$sig_slopes$slopes > 0.1,]
  aft_result_sig_neg_slope <- aft_result_list$sig_slopes[aft_result_list$sig_slopes$slopes < -0.1,]
  temp <- aft_result_list$sig_slopes[aft_result_list$sig_slopes$slopes <= 0.1,]
  aft_result_sig_zero_slope <- temp[temp$slopes >= -0.1,]
  
  bfr_cue_all_pos_slope <- bfr_cue_list$all_slopes[bfr_cue_list$all_slopes$slopes > 0.1,]
  bfr_cue_all_neg_slope <- bfr_cue_list$all_slopes[bfr_cue_list$all_slopes$slopes < -0.1,]
  temp <- bfr_cue_list$all_slopes[bfr_cue_list$all_slopes$slopes <= 0.1,]
  bfr_cue_all_zero_slope <- temp[temp$slopes >= -0.1,]
  aft_cue_all_pos_slope <- aft_cue_list$all_slopes[aft_cue_list$all_slopes$slopes > 0.1,]
  aft_cue_all_neg_slope <- aft_cue_list$all_slopes[aft_cue_list$all_slopes$slopes < -0.1,]
  temp <- aft_cue_list$all_slopes[aft_cue_list$all_slopes$slopes <= 0.1,]
  aft_cue_all_zero_slope <- temp[temp$slopes >= -0.1,]
  bfr_result_all_pos_slope <- bfr_result_list$all_slopes[bfr_result_list$all_slopes$slopes > 0.1,]
  bfr_result_all_neg_slope <- bfr_result_list$all_slopes[bfr_result_list$all_slopes$slopes < -0.1,]
  temp <- bfr_result_list$all_slopes[bfr_result_list$all_slopes$slopes <= 0.1,]
  bfr_result_all_zero_slope <- temp[temp$slopes >= -0.1,]
  aft_result_all_pos_slope <- aft_result_list$all_slopes[aft_result_list$all_slopes$slopes > 0.1,]
  aft_result_all_neg_slope <- aft_result_list$all_slopes[aft_result_list$all_slopes$slopes < -0.1,]
  temp <- aft_result_list$all_slopes[aft_result_list$all_slopes$slopes <= 0.1,]
  aft_result_all_zero_slope <- temp[temp$slopes >= -0.1,]
  
  bfr_cue_pos_val_sig_num <- dim(bfr_cue_sig_pos_slope[which(bfr_cue_sig_pos_slope$type=='val'),])[1]
  bfr_cue_pos_mtv_sig_num <- dim(bfr_cue_sig_pos_slope[which(bfr_cue_sig_pos_slope$type=='mtv'),])[1]
  bfr_cue_neg_val_sig_num <- dim(bfr_cue_sig_neg_slope[which(bfr_cue_sig_neg_slope$type=='val'),])[1]
  bfr_cue_neg_mtv_sig_num <- dim(bfr_cue_sig_neg_slope[which(bfr_cue_sig_neg_slope$type=='mtv'),])[1]
  bfr_cue_zero_val_sig_num <- dim(bfr_cue_sig_zero_slope[which(bfr_cue_sig_zero_slope$type=='val'),])[1]
  bfr_cue_zero_mtv_sig_num <- dim(bfr_cue_sig_zero_slope[which(bfr_cue_sig_zero_slope$type=='mtv'),])[1]
  bfr_cue_pos_val_all_num <- dim(bfr_cue_all_pos_slope[which(bfr_cue_all_pos_slope$type=='val'),])[1]
  bfr_cue_pos_mtv_all_num <- dim(bfr_cue_all_pos_slope[which(bfr_cue_all_pos_slope$type=='mtv'),])[1]
  bfr_cue_neg_val_all_num <- dim(bfr_cue_all_neg_slope[which(bfr_cue_all_neg_slope$type=='val'),])[1]
  bfr_cue_neg_mtv_all_num <- dim(bfr_cue_all_neg_slope[which(bfr_cue_all_neg_slope$type=='mtv'),])[1]
  bfr_cue_zero_val_all_num <- dim(bfr_cue_all_zero_slope[which(bfr_cue_all_zero_slope$type=='val'),])[1]
  bfr_cue_zero_mtv_all_num <- dim(bfr_cue_all_zero_slope[which(bfr_cue_all_zero_slope$type=='mtv'),])[1]
  
  aft_cue_pos_val_sig_num <- dim(aft_cue_sig_pos_slope[which(aft_cue_sig_pos_slope$type=='val'),])[1]
  aft_cue_pos_mtv_sig_num <- dim(aft_cue_sig_pos_slope[which(aft_cue_sig_pos_slope$type=='mtv'),])[1]
  aft_cue_neg_val_sig_num <- dim(aft_cue_sig_neg_slope[which(aft_cue_sig_neg_slope$type=='val'),])[1]
  aft_cue_neg_mtv_sig_num <- dim(aft_cue_sig_neg_slope[which(aft_cue_sig_neg_slope$type=='mtv'),])[1]
  aft_cue_zero_val_sig_num <- dim(aft_cue_sig_zero_slope[which(aft_cue_sig_zero_slope$type=='val'),])[1]
  aft_cue_zero_mtv_sig_num <- dim(aft_cue_sig_zero_slope[which(aft_cue_sig_zero_slope$type=='mtv'),])[1]
  aft_cue_pos_val_all_num <- dim(aft_cue_all_pos_slope[which(aft_cue_all_pos_slope$type=='val'),])[1]
  aft_cue_pos_mtv_all_num <- dim(aft_cue_all_pos_slope[which(aft_cue_all_pos_slope$type=='mtv'),])[1]
  aft_cue_neg_val_all_num <- dim(aft_cue_all_neg_slope[which(aft_cue_all_neg_slope$type=='val'),])[1]
  aft_cue_neg_mtv_all_num <- dim(aft_cue_all_neg_slope[which(aft_cue_all_neg_slope$type=='mtv'),])[1]
  aft_cue_zero_val_all_num <- dim(aft_cue_all_zero_slope[which(aft_cue_all_zero_slope$type=='val'),])[1]
  aft_cue_zero_mtv_all_num <- dim(aft_cue_all_zero_slope[which(aft_cue_all_zero_slope$type=='mtv'),])[1]
  
  bfr_result_pos_val_sig_num <- dim(bfr_result_sig_pos_slope[which(bfr_result_sig_pos_slope$type=='val'),])[1]
  bfr_result_pos_mtv_sig_num <- dim(bfr_result_sig_pos_slope[which(bfr_result_sig_pos_slope$type=='mtv'),])[1]
  bfr_result_neg_val_sig_num <- dim(bfr_result_sig_neg_slope[which(bfr_result_sig_neg_slope$type=='val'),])[1]
  bfr_result_neg_mtv_sig_num <- dim(bfr_result_sig_neg_slope[which(bfr_result_sig_neg_slope$type=='mtv'),])[1]
  bfr_result_zero_val_sig_num <- dim(bfr_result_sig_zero_slope[which(bfr_result_sig_zero_slope$type=='val'),])[1]
  bfr_result_zero_mtv_sig_num <- dim(bfr_result_sig_zero_slope[which(bfr_result_sig_zero_slope$type=='mtv'),])[1]
  bfr_result_pos_val_all_num <- dim(bfr_result_all_pos_slope[which(bfr_result_all_pos_slope$type=='val'),])[1]
  bfr_result_pos_mtv_all_num <- dim(bfr_result_all_pos_slope[which(bfr_result_all_pos_slope$type=='mtv'),])[1]
  bfr_result_neg_val_all_num <- dim(bfr_result_all_neg_slope[which(bfr_result_all_neg_slope$type=='val'),])[1]
  bfr_result_neg_mtv_all_num <- dim(bfr_result_all_neg_slope[which(bfr_result_all_neg_slope$type=='mtv'),])[1]
  bfr_result_zero_val_all_num <- dim(bfr_result_all_zero_slope[which(bfr_result_all_zero_slope$type=='val'),])[1]
  bfr_result_zero_mtv_all_num <- dim(bfr_result_all_zero_slope[which(bfr_result_all_zero_slope$type=='mtv'),])[1]
  
  aft_result_pos_val_sig_num <- dim(aft_result_sig_pos_slope[which(aft_result_sig_pos_slope$type=='val'),])[1]
  aft_result_pos_mtv_sig_num <- dim(aft_result_sig_pos_slope[which(aft_result_sig_pos_slope$type=='mtv'),])[1]
  aft_result_neg_val_sig_num <- dim(aft_result_sig_neg_slope[which(aft_result_sig_neg_slope$type=='val'),])[1]
  aft_result_neg_mtv_sig_num <- dim(aft_result_sig_neg_slope[which(aft_result_sig_neg_slope$type=='mtv'),])[1]
  aft_result_zero_mtv_sig_num <- dim(aft_result_sig_zero_slope[which(aft_result_sig_zero_slope$type=='mtv'),])[1]
  aft_result_zero_val_sig_num <- dim(aft_result_sig_zero_slope[which(aft_result_sig_zero_slope$type=='val'),])[1]
  aft_result_pos_val_all_num <- dim(aft_result_all_pos_slope[which(aft_result_all_pos_slope$type=='val'),])[1]
  aft_result_pos_mtv_all_num <- dim(aft_result_all_pos_slope[which(aft_result_all_pos_slope$type=='mtv'),])[1]
  aft_result_neg_val_all_num <- dim(aft_result_all_neg_slope[which(aft_result_all_neg_slope$type=='val'),])[1]
  aft_result_neg_mtv_all_num <- dim(aft_result_all_neg_slope[which(aft_result_all_neg_slope$type=='mtv'),])[1]
  aft_result_zero_val_all_num <- dim(aft_result_all_zero_slope[which(aft_result_all_zero_slope$type=='val'),])[1]
  aft_result_zero_mtv_all_num <- dim(aft_result_all_zero_slope[which(aft_result_all_zero_slope$type=='mtv'),])[1]
  
  bfr_cue_sig_val_units <- dim(bfr_cue_list$sig_slopes[which(bfr_cue_list$sig_slopes$type=='val'),])[1]
  bfr_cue_sig_mtv_units <- dim(bfr_cue_list$sig_slopes[which(bfr_cue_list$sig_slopes$type=='mtv'),])[1]
  aft_cue_sig_val_units <- dim(aft_cue_list$sig_slopes[which(aft_cue_list$sig_slopes$type=='val'),])[1]
  aft_cue_sig_mtv_units <- dim(aft_cue_list$sig_slopes[which(aft_cue_list$sig_slopes$type=='mtv'),])[1]
  bfr_result_sig_val_units <- dim(bfr_result_list$sig_slopes[which(bfr_result_list$sig_slopes$type=='val'),])[1]
  bfr_result_sig_mtv_units <- dim(bfr_result_list$sig_slopes[which(bfr_result_list$sig_slopes$type=='mtv'),])[1]
  aft_result_sig_val_units <- dim(aft_result_list$sig_slopes[which(aft_result_list$sig_slopes$type=='val'),])[1]
  aft_result_sig_mtv_units <- dim(aft_result_list$sig_slopes[which(aft_result_list$sig_slopes$type=='mtv'),])[1]
  
  bfr_cue_all_val_units <- dim(bfr_cue_list$all_slopes[which(bfr_cue_list$all_slopes$type=='val'),])[1]
  bfr_cue_all_mtv_units <- dim(bfr_cue_list$all_slopes[which(bfr_cue_list$all_slopes$type=='mtv'),])[1]
  aft_cue_all_val_units <- dim(aft_cue_list$all_slopes[which(aft_cue_list$all_slopes$type=='val'),])[1]
  aft_cue_all_mtv_units <- dim(aft_cue_list$all_slopes[which(aft_cue_list$all_slopes$type=='mtv'),])[1]
  bfr_result_all_val_units <- dim(bfr_result_list$all_slopes[which(bfr_result_list$all_slopes$type=='val'),])[1]
  bfr_result_all_mtv_units <- dim(bfr_result_list$all_slopes[which(bfr_result_list$all_slopes$type=='mtv'),])[1]
  aft_result_all_val_units <- dim(aft_result_list$all_slopes[which(aft_result_list$all_slopes$type=='val'),])[1]
  aft_result_all_mtv_units <- dim(aft_result_list$all_slopes[which(aft_result_list$all_slopes$type=='mtv'),])[1]
  
  labs <- c('pos','neg','zero')
  bfr_cue_val_sig <- c(bfr_cue_pos_val_sig_num,bfr_cue_pos_val_sig_num/bfr_cue_sig_val_units,bfr_cue_neg_val_sig_num,bfr_cue_neg_val_sig_num/bfr_cue_sig_val_units,bfr_cue_zero_val_sig_num,bfr_cue_zero_val_sig_num/bfr_cue_sig_val_units)
  bfr_cue_mtv_sig <- c(bfr_cue_pos_mtv_sig_num,bfr_cue_pos_mtv_sig_num/bfr_cue_sig_mtv_units,bfr_cue_neg_mtv_sig_num,bfr_cue_neg_mtv_sig_num/bfr_cue_sig_mtv_units,bfr_cue_zero_mtv_sig_num,bfr_cue_zero_mtv_sig_num/bfr_cue_sig_mtv_units)
  aft_cue_val_sig <- c(aft_cue_pos_val_sig_num,aft_cue_pos_val_sig_num/aft_cue_sig_val_units,aft_cue_neg_val_sig_num,aft_cue_neg_val_sig_num/aft_cue_sig_val_units,aft_cue_zero_val_sig_num,aft_cue_zero_val_sig_num/aft_cue_sig_val_units)
  aft_cue_mtv_sig <- c(aft_cue_pos_mtv_sig_num,aft_cue_pos_mtv_sig_num/aft_cue_sig_mtv_units,aft_cue_neg_mtv_sig_num,aft_cue_neg_mtv_sig_num/aft_cue_sig_mtv_units,aft_cue_zero_mtv_sig_num,aft_cue_zero_mtv_sig_num/aft_cue_sig_mtv_units)
  bfr_result_val_sig <- c(bfr_result_pos_val_sig_num,bfr_result_pos_val_sig_num/bfr_result_sig_val_units,bfr_result_neg_val_sig_num,bfr_result_neg_val_sig_num/bfr_result_sig_val_units,bfr_result_zero_val_sig_num,bfr_result_zero_val_sig_num/bfr_result_sig_val_units)
  bfr_result_mtv_sig <- c(bfr_result_pos_mtv_sig_num,bfr_result_pos_mtv_sig_num/bfr_result_sig_mtv_units,bfr_result_neg_mtv_sig_num,bfr_result_neg_mtv_sig_num/bfr_result_sig_mtv_units,bfr_result_zero_mtv_sig_num,bfr_result_zero_mtv_sig_num/bfr_result_sig_mtv_units)
  aft_result_val_sig <- c(aft_result_pos_val_sig_num,aft_result_pos_val_sig_num/aft_result_sig_val_units,aft_result_neg_val_sig_num,aft_result_neg_val_sig_num/aft_result_sig_val_units,aft_result_zero_val_sig_num,aft_result_zero_val_sig_num/aft_result_sig_val_units)
  aft_result_mtv_sig <- c(aft_result_pos_mtv_sig_num,aft_result_pos_mtv_sig_num/aft_result_sig_mtv_units,aft_result_neg_mtv_sig_num,aft_result_neg_mtv_sig_num/aft_result_sig_mtv_units,aft_result_zero_mtv_sig_num,aft_result_zero_mtv_sig_num/aft_result_sig_mtv_units)
  
  bfr_cue_val_all <- c(bfr_cue_pos_val_all_num,bfr_cue_pos_val_all_num/bfr_cue_all_val_units,bfr_cue_neg_val_all_num,bfr_cue_neg_val_all_num/bfr_cue_all_val_units,bfr_cue_zero_val_all_num,bfr_cue_zero_val_all_num/bfr_cue_all_val_units)
  bfr_cue_mtv_all <- c(bfr_cue_pos_mtv_all_num,bfr_cue_pos_mtv_all_num/bfr_cue_all_mtv_units,bfr_cue_neg_mtv_all_num,bfr_cue_neg_mtv_all_num/bfr_cue_all_mtv_units,bfr_cue_zero_mtv_all_num,bfr_cue_zero_mtv_all_num/bfr_cue_all_mtv_units)
  aft_cue_val_all <- c(aft_cue_pos_val_all_num,aft_cue_pos_val_all_num/aft_cue_all_val_units,aft_cue_neg_val_all_num,aft_cue_neg_val_all_num/aft_cue_all_val_units,aft_cue_zero_val_all_num,aft_cue_zero_val_all_num/aft_cue_all_val_units)
  aft_cue_mtv_all <- c(aft_cue_pos_mtv_all_num,aft_cue_pos_mtv_all_num/aft_cue_all_mtv_units,aft_cue_neg_mtv_all_num,aft_cue_neg_mtv_all_num/aft_cue_all_mtv_units,aft_cue_zero_mtv_all_num,aft_cue_zero_mtv_all_num/aft_cue_all_mtv_units)
  bfr_result_val_all <- c(bfr_result_pos_val_all_num,bfr_result_pos_val_all_num/bfr_result_all_val_units,bfr_result_neg_val_all_num,bfr_result_neg_val_all_num/bfr_result_all_val_units,bfr_result_zero_val_all_num,bfr_result_zero_val_all_num/bfr_result_all_val_units)
  bfr_result_mtv_all <- c(bfr_result_pos_mtv_all_num,bfr_result_pos_mtv_all_num/bfr_result_all_mtv_units,bfr_result_neg_mtv_all_num,bfr_result_neg_mtv_all_num/bfr_result_all_mtv_units,bfr_result_zero_mtv_all_num,bfr_result_zero_mtv_all_num/bfr_result_all_mtv_units)
  aft_result_val_all <- c(aft_result_pos_val_all_num,aft_result_pos_val_all_num/aft_result_all_val_units,aft_result_neg_val_all_num,aft_result_neg_val_all_num/aft_result_all_val_units,aft_result_zero_val_all_num,aft_result_zero_val_all_num/aft_result_all_val_units)
  aft_result_mtv_all <- c(aft_result_pos_mtv_all_num,aft_result_pos_mtv_all_num/aft_result_all_mtv_units,aft_result_neg_mtv_all_num,aft_result_neg_mtv_all_num/aft_result_all_mtv_units,aft_result_zero_mtv_all_num,aft_result_zero_mtv_all_num/aft_result_all_mtv_units)
  
  bfr_cue_val_sig <- data.frame(perc=bfr_cue_val_sig[c(2,4,6)],num=bfr_cue_val_sig[c(1,3,5)],labs,window='bfr_cue',type='value')
  bfr_cue_mtv_sig <- data.frame(perc=bfr_cue_mtv_sig[c(2,4,6)],num=bfr_cue_mtv_sig[c(1,3,5)],labs,window='bfr_cue',type='motivation')
  aft_cue_val_sig <- data.frame(perc=aft_cue_val_sig[c(2,4,6)],num=aft_cue_val_sig[c(1,3,5)],labs,window='aft_cue',type='value')
  aft_cue_mtv_sig <- data.frame(perc=aft_cue_mtv_sig[c(2,4,6)],num=aft_cue_mtv_sig[c(1,3,5)],labs,window='aft_cue',type='motivation')
  bfr_result_val_sig <- data.frame(perc=bfr_result_val_sig[c(2,4,6)],num=bfr_result_val_sig[c(1,3,5)],labs,window='bfr_result',type='value')
  bfr_result_mtv_sig <- data.frame(perc=bfr_result_mtv_sig[c(2,4,6)],num=bfr_result_mtv_sig[c(1,3,5)],labs,window='bfr_result',type='motivation')
  aft_result_val_sig <- data.frame(perc=aft_result_val_sig[c(2,4,6)],num=aft_result_val_sig[c(1,3,5)],labs,window='aft_result',type='value')
  aft_result_mtv_sig <- data.frame(perc=aft_result_mtv_sig[c(2,4,6)],num=aft_result_mtv_sig[c(1,3,5)],labs,window='aft_result',type='motivation')
  
  bfr_cue_val_all <- data.frame(perc=bfr_cue_val_all[c(2,4,6)],num=bfr_cue_val_all[c(1,3,5)],labs,window='bfr_cue',type='value')
  bfr_cue_mtv_all <- data.frame(perc=bfr_cue_mtv_all[c(2,4,6)],num=bfr_cue_mtv_all[c(1,3,5)],labs,window='bfr_cue',type='motivation')
  aft_cue_val_all <- data.frame(perc=aft_cue_val_all[c(2,4,6)],num=aft_cue_val_all[c(1,3,5)],labs,window='aft_cue',type='value')
  aft_cue_mtv_all <- data.frame(perc=aft_cue_mtv_all[c(2,4,6)],num=aft_cue_mtv_all[c(1,3,5)],labs,window='aft_cue',type='motivation')
  bfr_result_val_all <- data.frame(perc=bfr_result_val_all[c(2,4,6)],num=bfr_result_val_all[c(1,3,5)],labs,window='bfr_result',type='value')
  bfr_result_mtv_all <- data.frame(perc=bfr_result_mtv_all[c(2,4,6)],num=bfr_result_mtv_all[c(1,3,5)],labs,window='bfr_result',type='motivation')
  aft_result_val_all <- data.frame(perc=aft_result_val_all[c(2,4,6)],num=aft_result_val_all[c(1,3,5)],labs,window='aft_result',type='value')
  aft_result_mtv_all <- data.frame(perc=aft_result_mtv_all[c(2,4,6)],num=aft_result_mtv_all[c(1,3,5)],labs,window='aft_result',type='motivation')
  
  bfr_cue_val_sig <- bfr_cue_val_sig[rev(order(bfr_cue_val_sig$labs)),]
  bfr_cue_mtv_sig <- bfr_cue_mtv_sig[rev(order(bfr_cue_mtv_sig$labs)),]
  aft_cue_val_sig <- aft_cue_val_sig[rev(order(aft_cue_val_sig$labs)),]
  aft_cue_mtv_sig <- aft_cue_mtv_sig[rev(order(aft_cue_mtv_sig$labs)),]
  bfr_result_val_sig <- bfr_result_val_sig[rev(order(bfr_result_val_sig$labs)),]
  bfr_result_mtv_sig <- bfr_result_mtv_sig[rev(order(bfr_result_mtv_sig$labs)),]
  aft_result_val_sig <- aft_result_val_sig[rev(order(aft_result_val_sig$labs)),]
  aft_result_mtv_sig <- aft_result_mtv_sig[rev(order(aft_result_mtv_sig$labs)),]
  
  bfr_cue_val_all <- bfr_cue_val_all[rev(order(bfr_cue_val_all$labs)),]
  bfr_cue_mtv_all <- bfr_cue_mtv_all[rev(order(bfr_cue_mtv_all$labs)),]
  aft_cue_val_all <- aft_cue_val_all[rev(order(aft_cue_val_all$labs)),]
  aft_cue_mtv_all <- aft_cue_mtv_all[rev(order(aft_cue_mtv_all$labs)),]
  bfr_result_val_all <- bfr_result_val_all[rev(order(bfr_result_val_all$labs)),]
  bfr_result_mtv_all <- bfr_result_mtv_all[rev(order(bfr_result_mtv_all$labs)),]
  aft_result_val_all <- aft_result_val_all[rev(order(aft_result_val_all$labs)),]
  aft_result_mtv_all <- aft_result_mtv_all[rev(order(aft_result_mtv_all$labs)),]
  
  bfr_cue_val_sig <- ddply(bfr_cue_val_sig,.(window),transform,position=(cumsum(bfr_cue_val_sig$perc)-0.5*bfr_cue_val_sig$perc))
  bfr_cue_mtv_sig <- ddply(bfr_cue_mtv_sig,.(window),transform,position=(cumsum(bfr_cue_mtv_sig$perc)-0.5*bfr_cue_mtv_sig$perc))
  aft_cue_val_sig <- ddply(aft_cue_val_sig,.(window),transform,position=(cumsum(aft_cue_val_sig$perc)-0.5*aft_cue_val_sig$perc))
  aft_cue_mtv_sig <- ddply(aft_cue_mtv_sig,.(window),transform,position=(cumsum(aft_cue_mtv_sig$perc)-0.5*aft_cue_mtv_sig$perc))
  bfr_result_val_sig <- ddply(bfr_result_val_sig,.(window),transform,position=(cumsum(bfr_result_val_sig$perc)-0.5*bfr_result_val_sig$perc))
  bfr_result_mtv_sig <- ddply(bfr_result_mtv_sig,.(window),transform,position=(cumsum(bfr_result_mtv_sig$perc)-0.5*bfr_result_mtv_sig$perc))
  aft_result_val_sig <- ddply(aft_result_val_sig,.(window),transform,position=(cumsum(aft_result_val_sig$perc)-0.5*aft_result_val_sig$perc))
  aft_result_mtv_sig <- ddply(aft_result_mtv_sig,.(window),transform,position=(cumsum(aft_result_mtv_sig$perc)-0.5*aft_result_mtv_sig$perc))
  
  bfr_cue_val_all <- ddply(bfr_cue_val_all,.(window),transform,position=(cumsum(bfr_cue_val_all$perc)-0.5*bfr_cue_val_all$perc))
  bfr_cue_mtv_all <- ddply(bfr_cue_mtv_all,.(window),transform,position=(cumsum(bfr_cue_mtv_all$perc)-0.5*bfr_cue_mtv_all$perc))
  aft_cue_val_all <- ddply(aft_cue_val_all,.(window),transform,position=(cumsum(aft_cue_val_all$perc)-0.5*aft_cue_val_all$perc))
  aft_cue_mtv_all <- ddply(aft_cue_mtv_all,.(window),transform,position=(cumsum(aft_cue_mtv_all$perc)-0.5*aft_cue_mtv_all$perc))
  bfr_result_val_all <- ddply(bfr_result_val_all,.(window),transform,position=(cumsum(bfr_result_val_all$perc)-0.5*bfr_result_val_all$perc))
  bfr_result_mtv_all <- ddply(bfr_result_mtv_all,.(window),transform,position=(cumsum(bfr_result_mtv_all$perc)-0.5*bfr_result_mtv_all$perc))
  aft_result_val_all <- ddply(aft_result_val_all,.(window),transform,position=(cumsum(aft_result_val_all$perc)-0.5*aft_result_val_all$perc))
  aft_result_mtv_all <- ddply(aft_result_mtv_all,.(window),transform,position=(cumsum(aft_result_mtv_all$perc)-0.5*aft_result_mtv_all$perc))
  
  val_sig <- rbind(bfr_cue_val_sig,aft_cue_val_sig,bfr_result_val_sig,aft_result_val_sig)
  mtv_sig <- rbind(bfr_cue_mtv_sig,aft_cue_mtv_sig,bfr_result_mtv_sig,aft_result_mtv_sig)
  val_all <- rbind(bfr_cue_val_all,aft_cue_val_all,bfr_result_val_all,aft_result_val_all)
  mtv_all <- rbind(bfr_cue_mtv_all,aft_cue_mtv_all,bfr_result_mtv_all,aft_result_mtv_all)
  
  val_sig <- val_sig[which(val_sig$perc > 0),]
  mtv_sig <- mtv_sig[which(mtv_sig$perc > 0),]
  val_all <- val_all[which(val_all$perc > 0),]
  mtv_all <- mtv_all[which(mtv_all$perc > 0),]

  val_sig <- ddply(val_sig,.(type),transform,label=paste(scales::percent(val_sig$perc),' n=',val_sig$num,sep=""))
  mtv_sig <- ddply(mtv_sig,.(type),transform,label=paste(scales::percent(mtv_sig$perc),' n=',mtv_sig$num,sep=""))
  val_all <- ddply(val_all,.(type),transform,label=paste(scales::percent(val_all$perc),' n=',val_all$num,sep=""))
  mtv_all <- ddply(mtv_all,.(type),transform,label=paste(scales::percent(mtv_all$perc),' n=',mtv_all$num,sep=""))
  
  if(all_alphabeta_bool){png(paste('linreg_all_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  }else if (avg_alphabeta_bool){png(paste('linreg_avg_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
  }else{png(paste('linreg_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)}
  
  val_sig_plt <- ggplot() + geom_bar(aes(x=val_sig$window,y=val_sig$perc,fill=val_sig$labs),data=val_sig,stat="identity")
  val_sig_plt <- val_sig_plt + labs(title='Value, sig',fill="",x="Time Window",y="percentage") + scale_fill_manual(values=c("lightblue","seagreen","slateblue"))
  val_sig_plt <- val_sig_plt + geom_text(aes(x=val_sig$window,y=val_sig$position,label=val_sig$label),size=2,stat="identity") + theme(axis.text=element_text(size=8),axis.title=element_text(size=10),title=element_text(size=10))
  val_all_plt <- ggplot() + geom_bar(aes(x=val_all$window,y=val_all$perc,fill=val_all$labs),data=val_all,stat="identity")
  val_all_plt <- val_all_plt + labs(title='Value, all',fill="",x="Time Window",y="percentage") + scale_fill_manual(values=c("lightblue","seagreen","slateblue"))
  val_all_plt <- val_all_plt + geom_text(aes(x=val_all$window,y=val_all$position,label=val_all$label),size=2,stat="identity") + theme(axis.text=element_text(size=8),axis.title=element_text(size=10),title=element_text(size=10))
  
  mtv_sig_plt <- ggplot() + geom_bar(aes(x=mtv_sig$window,y=mtv_sig$perc,fill=mtv_sig$labs),data=mtv_sig,stat="identity")
  mtv_sig_plt <- mtv_sig_plt + labs(title='Motivation, sig',fill="",x="Time Window",y="percentage") + scale_fill_manual(values=c("lightblue","seagreen","slateblue"))
  mtv_sig_plt <- mtv_sig_plt + geom_text(aes(x=mtv_sig$window,y=mtv_sig$position,label=mtv_sig$label),size=2,stat="identity") + theme(axis.text=element_text(size=8),axis.title=element_text(size=10),title=element_text(size=10))
  mtv_all_plt <- ggplot() + geom_bar(aes(x=mtv_all$window,y=mtv_all$perc,fill=mtv_all$labs),data=mtv_all,stat="identity")
  mtv_all_plt <- mtv_all_plt + labs(title='Motivation, all',fill="",x="Time Window",y="percentage") + scale_fill_manual(values=c("lightblue","seagreen","slateblue"))
  mtv_all_plt <- mtv_all_plt + geom_text(aes(x=mtv_all$window,y=mtv_all$position,label=mtv_all$label),size=2,stat="identity") + theme(axis.text=element_text(size=8),axis.title=element_text(size=10),title=element_text(size=10))
  
  multiplot(val_sig_plt,val_all_plt,mtv_sig_plt,mtv_all_plt,cols=2)
  graphics.off()
  
}
  
#all_lists <- list(M1_vals,S1_vals,PmD_vals)

det_signs <- function(alpha,beta){
  
  both_pos <- 0
  both_neg <- 0
  alpha_pos <- 0
  beta_pos <- 0
  
  num_units <- length(alpha)
  
  for(j in 1:num_units){
    if (alpha[j] > 0 & beta[j] > 0){both_pos <- both_pos + 1
    }else if (alpha[j] < 0 & beta[j] < 0){both_neg <- both_neg + 1
    }else if (alpha[j] > 0 & beta[j] < 0){alpha_pos <- alpha_pos + 1
    }else if (alpha[j] < 0 & beta[j] > 0){beta_pos <- beta_pos + 1
    }else{cat('\nab error: alpha=',alpha[j],' beta=',beta[j],' unit=',j, '\n')}
  }
    
  perc_both_pos <- both_pos / num_units
  perc_both_neg <- both_neg / num_units
  perc_alpha_pos <- alpha_pos / num_units
  perc_beta_pos <- beta_pos / num_units
  
  return_list <- list(num_both_pos = both_pos,num_both_neg=both_neg,num_alpha_pos = alpha_pos,num_beta_pos=beta_pos,perc_both_pos=perc_both_pos,perc_both_neg=perc_both_neg,perc_alpha_pos=perc_alpha_pos,perc_beta_pos=perc_beta_pos)

  return(return_list)
}



if (avg_alphabeta_bool | all_alphabeta_bool){
  #plot avg alpha and beta vals (all)
  #plot signs bar plot (all)
  
  for (region_index in 1:length(region_list)){
    if (region_list[region_index] == 'M1'){region_vals <- M1_vals
    }else if (region_list[region_index] == 'S1'){region_vals <- S1_vals
    }else if (region_list[region_index] == 'PmD'){region_vals <- PmD_vals}
    
    all_bfr_cue_alpha <- region_vals$bfr_cue_list$alpha_list
    all_bfr_cue_beta <- region_vals$bfr_cue_list$beta_list
    all_aft_cue_alpha <- region_vals$aft_cue_list$alpha_list
    all_aft_cue_beta <- region_vals$aft_cue_list$beta_list
    all_bfr_result_alpha <- region_vals$bfr_result_list$alpha_list
    all_bfr_result_beta <- region_vals$bfr_result_list$beta_list
    all_aft_result_alpha <- region_vals$aft_result_list$alpha_list
    all_aft_result_beta <- region_vals$aft_result_list$beta_list
    
    #all_avg_alpha <- (all_bfr_cue_alpha + all_aft_cue_alpha + all_bfr_result_alpha + all_aft_result_alpha) / 4
    #all_avg_beta <- (all_bfr_cue_beta + all_aft_cue_beta + all_bfr_result_beta + all_aft_result_beta) / 4
    
    #####
    #file_list <- c('sig_slopes_M1_dicts.xlsx','sig_slopes_S1_dicts.xlsx','sig_slopes_PmD_dicts.xlsx')

    slopes_bfr_cue <- read.xlsx(file_list[region_index],sheet='slopes_bfr_cue_model',colNames=T)
    slopes_aft_cue <- read.xlsx(file_list[region_index],sheet='slopes_aft_cue_model',colNames=T)
    slopes_bfr_result <- read.xlsx(file_list[region_index],sheet='slopes_bfr_result_model',colNames=T)
    slopes_aft_result <- read.xlsx(file_list[region_index],sheet='slopes_aft_result_model',colNames=T)
    
    slopes_bfr_cue_sigall <- read.xlsx(file_list[region_index],sheet='sig_all_bfr_cue_model',colNames=T)
    slopes_aft_cue_sigall <- read.xlsx(file_list[region_index],sheet='sig_all_aft_cue_model',colNames=T)
    slopes_bfr_result_sigall <- read.xlsx(file_list[region_index],sheet='sig_all_bfr_result_model',colNames=T)
    slopes_aft_result_sigall <- read.xlsx(file_list[region_index],sheet='sig_all_aft_result_model',colNames=T)
    
    avg_bfr_cue <- c(mean(abs(slopes_bfr_cue$alpha)),mean(abs(slopes_bfr_cue$beta)),mean(abs(slopes_bfr_cue_sigall$alpha)),mean(abs(slopes_bfr_cue_sigall$beta)),mean(abs(all_bfr_cue_alpha)),mean(abs(all_bfr_cue_beta)))
    avg_aft_cue <- c(mean(abs(slopes_aft_cue$alpha)),mean(abs(slopes_aft_cue$beta)),mean(abs(slopes_aft_cue_sigall$alpha)),mean(abs(slopes_aft_cue_sigall$beta)),mean(abs(all_aft_cue_alpha)),mean(abs(all_aft_cue_beta)))
    avg_bfr_result <- c(mean(abs(slopes_bfr_result$alpha)),mean(abs(slopes_bfr_result$beta)),mean(abs(slopes_bfr_result_sigall$alpha)),mean(abs(slopes_bfr_result_sigall$beta)),mean(abs(all_bfr_result_alpha)),mean(abs(all_bfr_result_beta)))
    avg_aft_result <- c(mean(abs(slopes_aft_result$alpha)),mean(abs(slopes_aft_result$beta)),mean(abs(slopes_aft_result_sigall$alpha)),mean(abs(slopes_aft_result_sigall$beta)),mean(abs(all_aft_result_alpha)),mean(abs(all_aft_result_beta)))
    
    std_bfr_cue <- c(sd(abs(slopes_bfr_cue$alpha)),sd(abs(slopes_bfr_cue$beta)),sd(abs(slopes_bfr_cue_sigall$alpha)),sd(abs(slopes_bfr_cue_sigall$beta)),sd(abs(all_bfr_cue_alpha)),sd(abs(all_bfr_cue_beta)))
    std_aft_cue <- c(sd(abs(slopes_aft_cue$alpha)),sd(abs(slopes_aft_cue$beta)),sd(abs(slopes_aft_cue_sigall$alpha)),sd(abs(slopes_aft_cue_sigall$beta)),sd(abs(all_aft_cue_alpha)),sd(abs(all_aft_cue_beta)))
    std_bfr_result <- c(sd(abs(slopes_bfr_result$alpha)),sd(abs(slopes_bfr_result$beta)),sd(abs(slopes_bfr_result_sigall$alpha)),sd(abs(slopes_bfr_result_sigall$beta)),sd(abs(all_bfr_result_alpha)),sd(abs(all_bfr_result_beta)))
    std_aft_result <- c(sd(abs(slopes_aft_result$alpha)),sd(abs(slopes_aft_result$beta)),sd(abs(slopes_aft_result_sigall$alpha)),sd(abs(slopes_aft_result_sigall$beta)),sd(abs(all_aft_result_alpha)),sd(abs(all_aft_result_beta)))
    
    key <- c('alpha','beta','alpha','beta','alpha','beta')
    cat <- c('one sig','one sig','both sig','both sig','all','all')
    
    df_bfr_cue <- data.frame(value= avg_bfr_cue,cat,key,std_bfr_cue)
    df_aft_cue <- data.frame(value = avg_aft_cue,cat,key,std_aft_cue)
    df_bfr_result <- data.frame(value = avg_bfr_result,cat,key,std_bfr_result)
    df_aft_result <- data.frame(value = avg_aft_result,cat,key,std_aft_result)
    
    bfr_cue_plt <- ggplot(df_bfr_cue) + aes(x=cat,y=value,fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="before cue") + geom_errorbar(aes(ymin=df_bfr_cue$value + df_bfr_cue$std_bfr_cue,ymax=df_bfr_cue$value - df_bfr_cue$std_bfr_cue),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
    aft_cue_plt <- ggplot(df_aft_cue) + aes(x=cat,y=value, fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="after cue") + geom_errorbar(aes(ymin=df_aft_cue$value + df_aft_cue$std_aft_cue,ymax=df_aft_cue$value - df_aft_cue$std_aft_cue),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
    bfr_result_plt <- ggplot(df_bfr_result) + aes(x=cat,y=value, fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="before result") + geom_errorbar(aes(ymin=df_bfr_result$value + df_bfr_result$std_bfr_result,ymax=df_bfr_result$value - df_bfr_result$std_bfr_result),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
    aft_result_plt <- ggplot(df_aft_result) + aes(x=cat,y=value, fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="after result") + geom_errorbar(aes(ymin=df_aft_result$value + df_aft_result$std_aft_result,ymax=df_aft_result$value - df_aft_result$std_aft_result),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
    
    if (avg_alphabeta_bool){png(paste(region_list[region_index],'_avg_avg_abs_slopes.png',sep=""),width=8,height=6,units="in",res=500)
    }else if (all_alphabeta_bool){png(paste(region_list[region_index],'_all_avg_abs_slopes.png',sep=""),width=8,height=6,units="in",res=500)}
    multiplot(bfr_cue_plt,aft_cue_plt,bfr_result_plt,aft_result_plt,cols=2)
    
    graphics.off()
    
    #####do again w/o abs
    avg_bfr_cue <- c(mean(slopes_bfr_cue$alpha),mean(slopes_bfr_cue$beta),mean(slopes_bfr_cue_sigall$alpha),mean(slopes_bfr_cue_sigall$beta),mean(all_bfr_cue_alpha),mean(all_bfr_cue_beta))
    avg_aft_cue <- c(mean(slopes_aft_cue$alpha),mean(slopes_aft_cue$beta),mean(slopes_aft_cue_sigall$alpha),mean(slopes_aft_cue_sigall$beta),mean(all_aft_cue_alpha),mean(all_aft_cue_beta))
    avg_bfr_result <- c(mean(slopes_bfr_result$alpha),mean(slopes_bfr_result$beta),mean(slopes_bfr_result_sigall$alpha),mean(slopes_bfr_result_sigall$beta),mean(all_bfr_result_alpha),mean(all_bfr_result_beta))
    avg_aft_result <- c(mean(slopes_aft_result$alpha),mean(slopes_aft_result$beta),mean(slopes_aft_result_sigall$alpha),mean(slopes_aft_result_sigall$beta),mean(all_aft_result_alpha),mean(all_aft_result_beta))
    
    std_bfr_cue <- c(sd(slopes_bfr_cue$alpha),sd(slopes_bfr_cue$beta),sd(slopes_bfr_cue_sigall$alpha),sd(slopes_bfr_cue_sigall$beta),sd(all_bfr_cue_alpha),sd(all_bfr_cue_beta))
    std_aft_cue <- c(sd(slopes_aft_cue$alpha),sd(slopes_aft_cue$beta),sd(slopes_aft_cue_sigall$alpha),sd(slopes_aft_cue_sigall$beta),sd(all_aft_cue_alpha),sd(all_aft_cue_beta))
    std_bfr_result <- c(sd(slopes_bfr_result$alpha),sd(slopes_bfr_result$beta),sd(slopes_bfr_result_sigall$alpha),sd(slopes_bfr_result_sigall$beta),sd(all_bfr_result_alpha),sd(all_bfr_result_beta))
    std_aft_result <- c(sd(slopes_aft_result$alpha),sd(slopes_aft_result$beta),sd(slopes_aft_result_sigall$alpha),sd(slopes_aft_result_sigall$beta),sd(all_aft_result_alpha),sd(all_aft_result_beta))
    
    key <- c('alpha','beta','alpha','beta','alpha','beta')
    cat <- c('one sig','one sig','both sig','both sig','all','all')
    
    df_bfr_cue <- data.frame(value= avg_bfr_cue,cat,key,std_bfr_cue)
    df_aft_cue <- data.frame(value = avg_aft_cue,cat,key,std_aft_cue)
    df_bfr_result <- data.frame(value = avg_bfr_result,cat,key,std_bfr_result)
    df_aft_result <- data.frame(value = avg_aft_result,cat,key,std_aft_result)
    
    bfr_cue_plt <- ggplot(df_bfr_cue) + aes(x=cat,y=value,fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="before cue") + geom_errorbar(aes(ymin=df_bfr_cue$value + df_bfr_cue$std_bfr_cue,ymax=df_bfr_cue$value - df_bfr_cue$std_bfr_cue),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
    aft_cue_plt <- ggplot(df_aft_cue) + aes(x=cat,y=value, fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="after cue") + geom_errorbar(aes(ymin=df_aft_cue$value + df_aft_cue$std_aft_cue,ymax=df_aft_cue$value - df_aft_cue$std_aft_cue),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
    bfr_result_plt <- ggplot(df_bfr_result) + aes(x=cat,y=value, fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="before result") + geom_errorbar(aes(ymin=df_bfr_result$value + df_bfr_result$std_bfr_result,ymax=df_bfr_result$value - df_bfr_result$std_bfr_result),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
    aft_result_plt <- ggplot(df_aft_result) + aes(x=cat,y=value, fill = key)+ geom_bar(stat="identity",position="dodge",na.rm=T) + labs(title="after result") + geom_errorbar(aes(ymin=df_aft_result$value + df_aft_result$std_aft_result,ymax=df_aft_result$value - df_aft_result$std_aft_result),stat="identity",width=0.25,position=position_dodge(0.9),color="gray40",na.rm=T)
    
    if (avg_alphabeta_bool){png(paste(region_list[region_index],'_avg_avg_slopes.png',sep=""),width=8,height=6,units="in",res=500)
    }else if (all_alphabeta_bool){png(paste(region_list[region_index],'_all_avg_slopes.png',sep=""),width=8,height=6,units="in",res=500)}
    
    multiplot(bfr_cue_plt,aft_cue_plt,bfr_result_plt,aft_result_plt,cols=2)
    
    graphics.off()
    
    #### signs all
    labs <- c("both pos","both neg","alpha pos","beta pos")
    
    bfr_cue_list <- det_signs(all_bfr_cue_alpha,all_bfr_cue_beta)
    aft_cue_list <- det_signs(all_aft_cue_alpha,all_aft_cue_beta)
    bfr_result_list <- det_signs(all_bfr_result_alpha,all_bfr_result_beta)
    aft_result_list <- det_signs(all_aft_result_alpha,all_aft_result_beta)
    
    bfr_cue_vals <- c(bfr_cue_list$perc_both_pos,bfr_cue_list$perc_both_neg,bfr_cue_list$perc_alpha_pos,bfr_cue_list$perc_beta_pos)
    aft_cue_vals <- c(aft_cue_list$perc_both_pos,aft_cue_list$perc_both_neg,aft_cue_list$perc_alpha_pos,aft_cue_list$perc_beta_pos)
    bfr_result_vals <- c(bfr_result_list$perc_both_pos,bfr_result_list$perc_both_neg,bfr_result_list$perc_alpha_pos,bfr_result_list$perc_beta_pos)
    aft_result_vals <- c(aft_result_list$perc_both_pos,aft_result_list$perc_both_neg,aft_result_list$perc_alpha_pos,aft_result_list$perc_beta_pos)
    
    #if (identical(bfr_result_vals,numeric(0))){bfr_result_vals=c(0,0,0,0)}
    #if (is.null(bfr_result_vals)){bfr_result_vals=c(0,0,0,0)}
    
    bfr_cue_df <- data.frame(perc=bfr_cue_vals,labs,type='bfr_cue')
    aft_cue_df <- data.frame(perc=aft_cue_vals,labs,type='aft_cue')
    bfr_result_df <- data.frame(perc=bfr_result_vals,labs,type='bfr_result')
    aft_result_df <- data.frame(perc=aft_result_vals,labs,type='aft_result')
    
    bfr_cue_df <- bfr_cue_df[rev(order(bfr_cue_df$labs)),]
    aft_cue_df <- aft_cue_df[rev(order(aft_cue_df$labs)),]
    bfr_result_df <- bfr_result_df[rev(order(bfr_result_df$labs)),]
    aft_result_df <- aft_result_df[rev(order(aft_result_df$labs)),]
    
    bfr_cue_df <- ddply(bfr_cue_df,.(type),transform,position=(cumsum(bfr_cue_df$perc)-0.5*bfr_cue_df$perc))
    aft_cue_df <- ddply(aft_cue_df,.(type),transform,position=(cumsum(aft_cue_df$perc)-0.5*aft_cue_df$perc))
    bfr_result_df <- ddply(bfr_result_df,.(type),transform,position=(cumsum(bfr_result_df$perc)-0.5*bfr_result_df$perc))
    aft_result_df <- ddply(aft_result_df,.(type),transform,position=(cumsum(aft_result_df$perc)-0.5*aft_result_df$perc))

    bfr_cue_nums = c(bfr_cue_list$num_both_pos,bfr_cue_list$num_both_neg,bfr_cue_list$num_beta_pos,bfr_cue_list$num_alpha_pos)
    aft_cue_nums = c(aft_cue_list$num_both_pos,aft_cue_list$num_both_neg,aft_cue_list$num_beta_pos,aft_cue_list$num_alpha_pos)
    bfr_result_nums = c(bfr_result_list$num_both_pos,bfr_result_list$num_both_neg,bfr_result_list$num_beta_pos,bfr_result_list$num_alpha_pos)
    aft_result_nums = c(aft_result_list$num_both_pos,aft_result_list$num_both_neg,aft_result_list$num_beta_pos,aft_result_list$num_alpha_pos)
    
    bfr_cue_df <- ddply(bfr_cue_df,.(type),transform,label=paste(scales::percent(bfr_cue_df$perc),' n=',bfr_cue_nums,sep=""))
    aft_cue_df <- ddply(aft_cue_df,.(type),transform,label=paste(scales::percent(aft_cue_df$perc),' n=',aft_cue_nums,sep=""))
    bfr_result_df <- ddply(bfr_result_df,.(type),transform,label=paste(scales::percent(bfr_result_df$perc),' n=',bfr_result_nums,sep=""))
    aft_result_df <- ddply(aft_result_df,.(type),transform,label=paste(scales::percent(aft_result_df$perc),' n=',aft_result_nums,sep=""))
    
    if (all_alphabeta_bool){png(paste('signs_bar_plotted_all_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)
    }else if (avg_alphabeta_bool){png(paste('signs_bar_plotted_avg_',region_list[region_index],'.png',sep=""),width=8,height=6,units="in",res=500)}
    
    df_all <- rbind(bfr_cue_df,aft_cue_df,bfr_result_df,aft_result_df)
    df_all <- df_all[which(df_all$perc > 0),]
    
    save(df_all,bfr_cue_df,aft_cue_df,bfr_result_df,aft_result_df,bfr_cue_nums,aft_cue_nums,bfr_result_nums,aft_result_nums,bfr_cue_list,aft_cue_list,bfr_result_list,aft_result_list,file=paste('alphabeta_',region_list[region_index],'_avg.RData',sep=""))
    
    bar_plt <- ggplot() + geom_bar(aes(x=df_all$type,y=df_all$perc,fill=df_all$labs),data=df_all,stat="identity") 
    bar_plt <- bar_plt + labs(title=region_list[region_index],fill="",x="Time Window",y="Percentage") + scale_fill_manual(values=c("lightblue","seagreen","grey","slateblue"))
    bar_plt <- bar_plt + geom_text(aes(x=df_all$type,y=df_all$position,label=df_all$label),size=3,stat="identity")
    
    plot(bar_plt)
    graphics.off()
    
  }
}

save.image('all_r_avg.RData')

rm(list=ls())

 