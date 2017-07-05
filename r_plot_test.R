library(openxlsx)
library(ggplot2)
library(reshape2)
source("~/dropbox/mult_rp_files/r_test/multiplot.R")
library(zoo)
library(gplots)
library(RColorBrewer)

saveAsPng <- T

file_list <- c('nl_avg_data_M1_dict_total.xlsx','nl_avg_data_S1_dict_total.xlsx','nl_avg_data_PmD_dict_total.xlsx')
catch_file_list <- c('catch_nl_avg_data_M1_dict_total.xlsx','catch_nl_avg_data_S1_dict_total.xlsx','catch_nl_avg_data_PmD_dict_total.xlsx')
region_list <- c('M1','S1','PmD')

time <- seq(from=-0.45,to=0.9,by=0.05)

for(region_index in 1:length(file_list)){
  cat("\nRegion:",region_list[region_index],"\n")
  
  filename = file_list[region_index]
  
  wb <- loadWorkbook(filename)
  num_sheets <- length(sheets(wb))
  
  r3_succ_cue_name <- paste(region_list[region_index],"_r3_succ_cue_all",sep="")
  r3_succ_result_name <- paste(region_list[region_index],"_r3_succ_result_all",sep="")
  p3_fail_cue_name <- paste(region_list[region_index],"_p3_fail_cue_all",sep="")
  p3_fail_result_name <- paste(region_list[region_index],"_p3_fail_result_all",sep="")
  
  all_r_succ_cue_name <- paste(region_list[region_index],"_all_r_succ_cue",sep="")
  all_r_fail_cue_name <- paste(region_list[region_index],"_all_r_fail_cue",sep="")
  all_r_succ_result_name <- paste(region_list[region_index],"_all_r_succ_result",sep="")
  all_r_fail_result_name <- paste(region_list[region_index],"_all_r_fail_result",sep="")
  
  all_p_succ_cue_name <- paste(region_list[region_index],"_all_p_succ_cue",sep="")
  all_p_fail_cue_name <- paste(region_list[region_index],"_all_p_fail_cue",sep="")
  all_p_succ_result_name <- paste(region_list[region_index],"_all_p_succ_result",sep="")
  all_p_fail_result_name <- paste(region_list[region_index],"_all_p_fail_result",sep="")
  
  no_r_succ_cue_name <- paste(region_list[region_index],"_no_r_succ_cue",sep="")
  no_r_fail_cue_name <- paste(region_list[region_index],"_no_r_fail_cue",sep="")
  no_r_succ_result_name <- paste(region_list[region_index],"_no_r_succ_result",sep="")
  no_r_fail_result_name <- paste(region_list[region_index],"_no_r_fail_result",sep="")
  
  no_p_succ_cue_name <- paste(region_list[region_index],"_no_p_succ_cue",sep="")
  no_p_fail_cue_name <- paste(region_list[region_index],"_no_p_fail_cue",sep="")
  no_p_succ_result_name <- paste(region_list[region_index],"_no_p_succ_result",sep="")
  no_p_fail_result_name <- paste(region_list[region_index],"_no_p_fail_result",sep="")
  
  r3_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r3_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p3_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p3_fail_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  
  all_r_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  all_r_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  all_r_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  all_r_fail_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  
  all_p_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  all_p_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  all_p_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  all_p_fail_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  
  no_r_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  no_r_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  no_r_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  no_r_fail_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  
  no_p_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  no_p_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  no_p_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  no_p_fail_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  
  for (j in 1:num_sheets){
    cat('plotting unit', j,"\n")
    tmp <- read.xlsx(filename,sheet = j, colNames=T)
  
    png(paste(region_list[region_index],"unit_",j,".png",sep=""),width=8,height=6,units="in",res=500)
    
    tmp2 <- tmp[1:28,1:32]
    tmp2$p3_fail_cue <- rollmean(tmp$p3_fail_cue,3)
    tmp2$p2_fail_cue <- rollmean(tmp$p2_fail_cue,3)
    tmp2$p1_fail_cue <- rollmean(tmp$p1_fail_cue,3)
    tmp2$p0_fail_cue <- rollmean(tmp$p0_fail_cue,3)
    tmp2$r0_succ_cue <- rollmean(tmp$r0_succ_cue,3)
    tmp2$r1_succ_cue <- rollmean(tmp$r1_succ_cue,3)
    tmp2$r2_succ_cue <- rollmean(tmp$r2_succ_cue,3)
    tmp2$r3_succ_cue <- rollmean(tmp$r3_succ_cue,3)
    tmp2$p3_fail_result <- rollmean(tmp$p3_fail_result,3)
    tmp2$p2_fail_result <- rollmean(tmp$p2_fail_result,3)
    tmp2$p1_fail_result <- rollmean(tmp$p1_fail_result,3)
    tmp2$p0_fail_result <- rollmean(tmp$p0_fail_result,3)
    tmp2$r0_succ_result <- rollmean(tmp$r0_succ_result,3)
    tmp2$r1_succ_result <- rollmean(tmp$r1_succ_result,3)
    tmp2$r2_succ_result <- rollmean(tmp$r2_succ_result,3)
    tmp2$r3_succ_result <- rollmean(tmp$r3_succ_result,3)
    
    #making array to compare between units
    r3_succ_cue_matrix[j,] <- tmp2$r3_succ_cue
    r3_succ_result_matrix[j,] <- tmp2$r3_succ_result
    p3_fail_cue_matrix[j,] <- tmp2$p3_fail_cue
    p3_fail_result_matrix[j,] <- tmp2$p3_fail_result
    
    all_r_succ_cue_matrix[j,] = (tmp2$r1_succ_cue + tmp2$r2_succ_cue + tmp2$r3_succ_cue)/3
    all_r_fail_cue_matrix[j,] = (tmp2$r1_fail_cue + tmp2$r2_fail_cue + tmp2$r3_fail_cue)/3
    all_r_succ_result_matrix[j,] = (tmp2$r1_succ_result + tmp2$r2_succ_result + tmp2$r3_succ_result)/3
    all_r_fail_result_matrix[j,] = (tmp2$r1_fail_result + tmp2$r2_fail_result + tmp2$r3_fail_result)/3
    
    all_p_succ_cue_matrix[j,] = (tmp2$p1_succ_cue + tmp2$p2_succ_cue + tmp2$p3_succ_cue)/3
    all_p_fail_cue_matrix[j,] = (tmp2$p1_fail_cue + tmp2$p2_fail_cue + tmp2$p3_fail_cue)/3
    all_p_succ_result_matrix[j,] = (tmp2$p1_succ_result + tmp2$p2_succ_result + tmp2$p3_succ_result)/3
    all_p_fail_result_matrix[j,] = (tmp2$p1_fail_result + tmp2$p2_fail_result + tmp2$p3_fail_result)/3
    
    no_r_succ_cue_matrix[j,] <- tmp2$r0_succ_cue
    no_r_fail_cue_matrix[j,] <- tmp2$r0_fail_cue
    no_r_succ_result_matrix[j,] <- tmp2$r0_succ_result
    no_r_fail_result_matrix[j,] <- tmp2$r0_fail_result
    
    no_p_succ_cue_matrix[j,] <- tmp2$p0_succ_cue
    no_p_fail_cue_matrix[j,] <- tmp2$p0_fail_cue
    no_p_succ_result_matrix[j,] <- tmp2$p0_succ_result
    no_p_fail_result_matrix[j,] <- tmp2$p0_fail_result
    
    #plotting individual unit
    df_cue <- data.frame(time,p3_fail=tmp2$p3_fail_cue,p2_fail=tmp2$p2_fail_cue,p1_fail=tmp2$p1_fail_cue,p0_fail=tmp2$p0_fail_cue,r0_succ=tmp2$r0_succ_cue,r1_succ=tmp2$r1_succ_cue,r2_succ=tmp2$r2_succ_cue,r3_succ=tmp2$r3_succ_cue)
    #TODO melt not renaming variable name
    df_cue <- melt(df_cue, id.vars="time", variable_name="level")
    plt_cue <- ggplot(df_cue, aes(time,value)) + geom_line(aes(colour=variable))
    plt_cue <- plt_cue + scale_color_brewer(palette="RdYlGn") +labs(title=paste(region_list[region_index],"unit",j,"cue"),y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)
    
    df_result <- data.frame(time,p3_fail=tmp2$p3_fail_result,p2_fail=tmp2$p2_fail_result,p1_fail=tmp2$p1_fail_result,p0_fail=tmp2$p0_fail_result,r0_succ=tmp2$r0_succ_result,r1_succ=tmp2$r1_succ_result,r2_succ=tmp2$r2_succ_result,r3_succ=tmp2$r3_succ_result)
    df_result <- melt(df_result, id.vars="time", variable_name="level")
    plt_result <- ggplot(df_result, aes(time,value)) + geom_line(aes(colour=variable))
    plt_result <- plt_result + scale_color_brewer(palette="RdYlGn") +labs(title='result',y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)
  
    multiplot(plt_cue,plt_result,cols=1)
    
    dev.off()
    rm(tmp)
    }
  assign(r3_succ_cue_name,r3_succ_cue_matrix)
  assign(r3_succ_result_name,r3_succ_result_matrix)
  assign(p3_fail_cue_name,p3_fail_cue_matrix)
  assign(p3_fail_result_name,p3_fail_result_matrix)
  assign(all_r_succ_cue_name,all_r_succ_cue_matrix)
  assign(all_r_fail_cue_name,all_r_fail_cue_matrix)
  assign(all_r_succ_result_name,all_r_succ_result_matrix)
  assign(all_r_fail_result_name,all_r_fail_result_matrix)
  assign(all_p_succ_cue_name,all_p_succ_cue_matrix)
  assign(all_p_fail_cue_name,all_p_fail_cue_matrix)
  assign(all_p_succ_result_name,all_p_succ_result_matrix)
  assign(all_p_fail_result_name,all_p_fail_result_matrix)
  assign(no_r_succ_cue_name,no_r_succ_cue_matrix)
  assign(no_r_fail_cue_name,no_r_fail_cue_matrix)
  assign(no_r_succ_result_name,no_r_succ_result_matrix)
  assign(no_r_fail_result_name,no_r_fail_result_matrix)
  assign(no_p_succ_cue_name,no_p_succ_cue_matrix)
  assign(no_p_fail_cue_name,no_p_fail_cue_matrix)
  assign(no_p_succ_result_name,no_p_succ_result_matrix)
  assign(no_p_fail_result_name,no_p_fail_result_matrix)
  
}

for(region_index in 1:length(catch_file_list)){
  cat("\nRegion:",region_list[region_index],"\n")
  
  filename = catch_file_list[region_index]
  wb <- loadWorkbook(filename)
  num_sheets <- length(sheets(wb))
  
  r_all_catch_cue_name <- paste(region_list[region_index],"_r_all_catch_cue_name",sep="")
  r_all_catch_result_name <- paste(region_list[region_index],"_r_all_catch_result_name",sep="")
  p_all_catch_cue_name <- paste(region_list[region_index],"_p_all_catch_cue_name",sep="")
  p_all_catch_result_name <- paste(region_list[region_index],"_p_all_catch_result_name",sep="")
  
  r_all_catch_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r_all_catch_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p_all_catch_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p_all_catch_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  
  
  
  for (j in 1:num_sheets){
    cat('plotting unit', j,"\n")
    tmp <- read.xlsx(filename,sheet = j, colNames=T)
    
    tmp2 <- tmp[1:28,1:20]
    tmp2$p3_catch_cue <- rollmean(tmp$p3_catch_cue,3)
    tmp2$p2_catch_cue <- rollmean(tmp$p2_catch_cue,3)
    tmp2$p1_catch_cue <- rollmean(tmp$p1_catch_cue,3)
    tmp2$p0_catch_cue <- rollmean(tmp$p0_catch_cue,3)
    tmp2$r0_catch_cue <- rollmean(tmp$r0_catch_cue,3)
    tmp2$r1_catch_cue <- rollmean(tmp$r1_catch_cue,3)
    tmp2$r2_catch_cue <- rollmean(tmp$r2_catch_cue,3)
    tmp2$r3_catch_cue <- rollmean(tmp$r3_catch_cue,3)
    tmp2$r_all_catch_cue <- rollmean(tmp$r_all_catch_cue,3)
    tmp2$p_all_catch_cue <- rollmean(tmp$p_all_catch_cue,3)
    tmp2$p3_catch_result <- rollmean(tmp$p3_catch_result,3)
    tmp2$p2_catch_result <- rollmean(tmp$p2_catch_result,3)
    tmp2$p1_catch_result <- rollmean(tmp$p1_catch_result,3)
    tmp2$p0_catch_result <- rollmean(tmp$p0_catch_result,3)
    tmp2$r0_catch_result <- rollmean(tmp$r0_catch_result,3)
    tmp2$r1_catch_result <- rollmean(tmp$r1_catch_result,3)
    tmp2$r2_catch_result <- rollmean(tmp$r2_catch_result,3)
    tmp2$r3_catch_result <- rollmean(tmp$r3_catch_result,3)
    tmp2$r_all_catch_result <- rollmean(tmp$r_all_catch_result,3)
    tmp2$p_all_catch_result <- rollmean(tmp$p_all_catch_result,3)
    
    r_all_catch_cue_matrix[j,] <- tmp2$r_all_catch_cue
    r_all_catch_result_matrix[j,] <- tmp2$r_all_catch_result
    p_all_catch_cue_matrix[j,] <- tmp2$p_all_catch_cue
    p_all_catch_result_matrix[j,] <- tmp2$p_all_catch_result
    
    df_cue <- data.frame(time,p3_catch=tmp2$p3_catch_cue,p2_catch=tmp2$p2_catch_cue,p1_catch=tmp2$p1_catch_cue,p0_catch=tmp2$p0_catch_cue,r0_catch=tmp2$r0_catch_cue,r1_catch=tmp2$r1_catch_cue,r2_catch=tmp2$r2_catch_cue,r3_catch=tmp2$r3_catch_cue)
    #TODO melt not renaming variable name
    df_cue <- melt(df_cue, id.vars="time", variable_name="level")
    plt_cue <- ggplot(df_cue, aes(time,value)) + geom_line(aes(colour=variable))
    plt_cue <- plt_cue + scale_color_brewer(palette="RdYlGn") +labs(title=paste(region_list[region_index],"unit",j,"cue"),y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)
    
    df_result <- data.frame(time,p3_catch=tmp2$p3_catch_result,p2_catch=tmp2$p2_catch_result,p1_catch=tmp2$p1_catch_result,p0_catch=tmp2$p0_catch_result,r0_catch=tmp2$r0_catch_result,r1_catch=tmp2$r1_catch_result,r2_catch=tmp2$r2_catch_result,r3_catch=tmp2$r3_catch_result)
    df_result <- melt(df_result, id.vars="time", variable_name="level")
    plt_result <- ggplot(df_result, aes(time,value)) + geom_line(aes(colour=variable))
    plt_result <- plt_result + scale_color_brewer(palette="RdYlGn") +labs(title='result',y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)
    
    df_catch_all_cue <- data.frame(time,r_all_catch=tmp2$r_all_catch_cue,p_all_catch=tmp2$p_all_catch_cue)
    df_catch_all_cue <- melt(df_catch_all_cue,id.vars="time",variable_name="level")
    plt_catch_cue <- ggplot(df_catch_all_cue,aes(time,value)) + geom_line(aes(colour=variable))
    plt_catch_cue <- plt_catch_cue + labs(title=paste(region_list[region_index],'unit',j,'cue catch all'),y='normalized firing rate',x="time(s)") + geom_vline(xintercept=0)
    
    df_catch_all_result <- data.frame(time,r_all_catch=tmp2$r_all_catch_result,p_all_catch=tmp2$p_all_catch_result)
    df_catch_all_result <- melt(df_catch_all_result,id.vars="time",variable_name="level")
    plt_catch_result <- ggplot(df_catch_all_result,aes(time,value)) + geom_line(aes(colour=variable))
    plt_catch_result <- plt_catch_result + labs(title='result catch all',y='normalized firing rate',x="time(s)") + geom_vline(xintercept=0)
    
    png(paste(region_list[region_index],"_catch_multilevels_unit_",j,".png",sep=""),width=8,height=6,units="in",res=500)
    multiplot(plt_cue,plt_result,cols=1)
    dev.off()
    
    png(paste(region_list[region_index],"_catch_all_unit_",j,".png",sep=""),width=8,height=6,units="in",res=500)
    multiplot(plt_catch_cue,plt_catch_result,cols=1)
    dev.off()
    rm(tmp)
  }
  assign(r_all_catch_cue_name,r_all_catch_cue_matrix)
  assign(r_all_catch_result_name,r_all_catch_result_matrix)
  assign(p_all_catch_cue_name,p_all_catch_cue_matrix)
  assign(p_all_catch_result_name,p_all_catch_result_matrix)
}

M1_matrices <- abind(M1_r3_succ_cue_all,M1_r3_succ_result_all,M1_p3_fail_cue_all,M1_p3_fail_result_all,M1_all_r_succ_cue,M1_all_r_fail_cue,M1_all_r_succ_result,M1_all_r_fail_result,M1_all_p_succ_cue,M1_all_p_fail_cue,M1_all_p_succ_result,M1_all_p_fail_result,M1_no_r_succ_cue,M1_no_r_fail_cue,M1_no_r_succ_result,M1_no_r_fail_result,M1_no_p_succ_cue,M1_no_p_fail_cue,M1_no_p_succ_result,M1_no_p_fail_result,M1_r_all_catch_cue,M1_r_all_catch_result,M1_p_all_catch_cue,M1_p_all_catch_result,along=3)
M1_matrix_keys <- c('M1_r3_succ_cue','M1_r3_succ_result','M1_p3_fail_cue','M1_p3_fail_result','M1_all_r_succ_cue','M1_all_r_fail_cue','M1_all_r_succ_result','M1_all_r_fail_result','M1_all_p_succ_cue','M1_all_p_fail_cue','M1_all_p_succ_result','M1_all_p_fail_result','M1_no_r_succ_cue','M1_no_r_fail_cue','M1_no_r_succ_result','M1_no_r_fail_result','M1_no_p_succ_cue','M1_no_p_fail_cue','M1_no_p_succ_result','M1_no_p_fail_result','M1_r_all_catch_cue','M1_r_all_catch_result','M1_p_all_catch_cue','M1_p_all_catch_result')

S1_matrices <- abind(S1_r3_succ_cue_all,S1_r3_succ_result_all,S1_p3_fail_cue_all,S1_p3_fail_result_all,S1_all_r_succ_cue,S1_all_r_fail_cue,S1_all_r_succ_result,S1_all_r_fail_result,S1_all_p_succ_cue,S1_all_p_fail_cue,S1_all_p_succ_result,S1_all_p_fail_result,S1_no_r_succ_cue,S1_no_r_fail_cue,S1_no_r_succ_result,S1_no_r_fail_result,S1_no_p_succ_cue,S1_no_p_fail_cue,S1_no_p_succ_result,S1_no_p_fail_result,S1_r_all_catch_cue,S1_r_all_catch_result,S1_p_all_catch_cue,S1_p_all_catch_result,along=3)
S1_matrix_keys <- c('S1_r3_succ_cue','S1_r3_succ_result','S1_p3_fail_cue','S1_p3_fail_result','S1_all_r_succ_cue','S1_all_r_fail_cue','S1_all_r_succ_result','S1_all_r_fail_result','S1_all_p_succ_cue','S1_all_p_fail_cue','S1_all_p_succ_result','S1_all_p_fail_result','S1_no_r_succ_cue','S1_no_r_fail_cue','S1_no_r_succ_result','S1_no_r_fail_result','S1_no_p_succ_cue','S1_no_p_fail_cue','S1_no_p_succ_result','S1_no_p_fail_result','S1_r_all_catch_cue','S1_r_all_catch_result','S1_p_all_catch_cue','S1_p_all_catch_result')

PmD_matrices <- abind(PmD_r3_succ_cue_all,PmD_r3_succ_result_all,PmD_p3_fail_cue_all,PmD_p3_fail_result_all,PmD_all_r_succ_cue,PmD_all_r_fail_cue,PmD_all_r_succ_result,PmD_all_r_fail_result,PmD_all_p_succ_cue,PmD_all_p_fail_cue,PmD_all_p_succ_result,PmD_all_p_fail_result,PmD_no_r_succ_cue,PmD_no_r_fail_cue,PmD_no_r_succ_result,PmD_no_r_fail_result,PmD_no_p_succ_cue,PmD_no_p_fail_cue,PmD_no_p_succ_result,PmD_no_p_fail_result,PmD_r_all_catch_cue,PmD_r_all_catch_result,PmD_p_all_catch_cue,PmD_p_all_catch_result,along=3)
PmD_matrix_keys <- c('PmD_r3_succ_cue','PmD_r3_succ_result','PmD_p3_fail_cue','PmD_p3_fail_result','PmD_all_r_succ_cue','PmD_all_r_fail_cue','PmD_all_r_succ_result','PmD_all_r_fail_result','PmD_all_p_succ_cue','PmD_all_p_fail_cue','PmD_all_p_succ_result','PmD_all_p_fail_result','PmD_no_r_succ_cue','PmD_no_r_fail_cue','PmD_no_r_succ_result','PmD_no_r_fail_result','PmD_no_p_succ_cue','PmD_no_p_fail_cue','PmD_no_p_succ_result','PmD_no_p_fail_result','PmD_r_all_catch_cue','PmD_r_all_catch_result','PmD_p_all_catch_cue','PmD_p_all_catch_result')

for (i in 1:length(M1_matrix_keys)){
  png(paste(M1_matrix_keys[i],".png",sep=""),width=8,height=6,units="in",res=500)
  heatmap.2(M1_matrices[,,i],Colv=F,dendrogram="row",scale="row",col=brewer.pal(11,"RdBu"),main=M1_matrix_keys[i],trace="none",cexRow=0.5,ylab="unit",xlab="time (s)",colsep=9)
  dev.off()
}

for (i in 1:length(S1_matrix_keys)){
  png(paste(S1_matrix_keys[i],".png",sep=""),width=8,height=6,units="in",res=500)
  heatmap.2(S1_matrices[,,i],Colv=F,dendrogram="row",scale="row",col=brewer.pal(11,"RdBu"),main=S1_matrix_keys[i],trace="none",cexRow=0.5,ylab="unit",xlab="time (s)",colsep=9)
  dev.off()
}

for (i in 1:length(PmD_matrix_keys)){
  png(paste(PmD_matrix_keys[i],".png",sep=""),width=8,height=6,units="in",res=500)
  heatmap.2(PmD_matrices[,,i],Colv=F,dendrogram="row",scale="row",col=brewer.pal(11,"RdBu"),main=PmD_matrix_keys[i],trace="none",cexRow=0.5,ylab="unit",xlab="time (s)",colsep=9)
  dev.off()
}










#sorted_M1_r3_succ_cue_all <- M1_r3_succ_cue_all[order(rowMeans(M1_r3_succ_cue_all),decreasing=F),]
