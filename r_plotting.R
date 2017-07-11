library(openxlsx)
library(ggplot2)
library(reshape2)
#source("~/dropbox/mult_rp_files/r_test/multiplot.R")
source("~/Dropbox/mult_rp_files/r_test/multiplot.R")
library(zoo)
library(gplots)
library(RColorBrewer)
library(abind)
library(gridGraphics)
library(grid)
library(gridExtra)

saveAsPng <- T

file_list <- c('nl_avg_data_M1_dict_total.xlsx','nl_avg_data_S1_dict_total.xlsx','nl_avg_data_PmD_dict_total.xlsx')
catch_file_list <- c('catch_nl_avg_data_M1_dict_total.xlsx','catch_nl_avg_data_S1_dict_total.xlsx','catch_nl_avg_data_PmD_dict_total.xlsx')
region_list <- c('M1','S1','PmD')
time <- seq(from=-0.45,to=0.9,by=0.05)

gf_total <- read.xlsx('gf_avg_data_M1_dict_total.xlsx',sheet=1,colNames=T)
gf_time <- seq(-0.45,0.90,by=0.005)

#########

plot_gf <- function(gf_value,std_value,key){
  
  png(paste(key,"_gf.png",sep=""),width=6,height=4,units="in",res=500)
  
  gf_avg <- gf_value[10:280]
  gf_std <- std_value[10:280]
  upper <- gf_avg + gf_std
  lower <- gf_avg - gf_std
  gf_df <- data.frame(gf_time,gf_avg,upper,lower)
  gf_melt <- melt(gf_df,id.vars="gf_time",variable_name="value")
  p <- ggplot(data=gf_melt$gf_avg,aes(x=gf_time,y=gf_avg)) + geom_line() + theme(plot.margin=unit(c(0.5,1.5,0.5,3.0),"cm")) #margin order: top,right,btm,left

  p <- p + geom_ribbon(aes(ymin=gf_df$lower,ymax=gf_df$upper,alpha=0.15),show.legend = F) + geom_vline(xintercept=0)
  #p <- p + labs(title=paste(region_list[region_index],"unit",j,"cue"),y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)
  p <- p + labs(title=paste("gripforce",key),y="unit", x="time(s)")
  plot(p)
  dev.off()
  
  #gf_avg <- gf_total$ra_succ_result_avg[10:280]
  #test_std <- gf_total$ra_succ_result_std[10:280]
  #test_upper <- test_avg + test_std
  #test_lower <- test_avg - test_std
  #test_df <- data.frame(gf_time,test_avg,test_upper,test_lower)
  #test_melt <- melt(test_df,id.vars="gf_time",variable_name="value")
  #p <- ggplot(data=test_melt$test_avg,aes(x=gf_time,y=test_avg)) + geom_line()
  #p <- p + geom_ribbon(aes(ymin=test_df$test_lower,ymax=test_df$test_upper,alpha=0.15),show.legend = F) 
  ##p <- p + labs(title=paste(region_list[region_index],"unit",j,"cue"),y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)
  #p <- p + labs(title="gripforce test",y="unit", x="time(s)")                                                                            
}

for(region_index in 1:length(file_list)){
  cat("\nplotting region:",region_list[region_index])
  
  filename = file_list[region_index]
  wb <- loadWorkbook(filename)
  num_sheets <- length(sheets(wb))
  
  total_array_name <- paste(region_list[region_index],"_unit_info",sep="")
  total_array <- array(NA,dim=c(28,32,num_sheets))
  
  r0_succ_cue_name <- paste(region_list[region_index],"_r0_succ_cue_all",sep="")
  r1_succ_cue_name <- paste(region_list[region_index],"_r1_succ_cue_all",sep="")
  r2_succ_cue_name <- paste(region_list[region_index],"_r2_succ_cue_all",sep="")
  r3_succ_cue_name <- paste(region_list[region_index],"_r3_succ_cue_all",sep="")
  r0_succ_result_name <- paste(region_list[region_index],"_r0_succ_result_all",sep="")
  r1_succ_result_name <- paste(region_list[region_index],"_r1_succ_result_all",sep="")
  r2_succ_result_name <- paste(region_list[region_index],"_r2_succ_result_all",sep="")
  r3_succ_result_name <- paste(region_list[region_index],"_r3_succ_result_all",sep="")
  r0_fail_cue_name <- paste(region_list[region_index],"_r0_fail_cue_all",sep="")
  r1_fail_cue_name <- paste(region_list[region_index],"_r1_fail_cue_all",sep="")
  r2_fail_cue_name <- paste(region_list[region_index],"_r2_fail_cue_all",sep="")
  r3_fail_cue_name <- paste(region_list[region_index],"_r3_fail_cue_all",sep="")
  r0_fail_result_name <- paste(region_list[region_index],"_r0_fail_result_all",sep="")
  r1_fail_result_name <- paste(region_list[region_index],"_r1_fail_result_all",sep="")
  r2_fail_result_name <- paste(region_list[region_index],"_r2_fail_result_all",sep="")
  r3_fail_result_name <- paste(region_list[region_index],"_r3_fail_result_all",sep="")
  
  p0_succ_cue_name <- paste(region_list[region_index],"_p0_succ_cue_all",sep="")
  p1_succ_cue_name <- paste(region_list[region_index],"_p1_succ_cue_all",sep="")
  p2_succ_cue_name <- paste(region_list[region_index],"_p2_succ_cue_all",sep="")
  p3_succ_cue_name <- paste(region_list[region_index],"_p3_succ_cue_all",sep="")
  p0_succ_result_name <- paste(region_list[region_index],"_p0_succ_result_all",sep="")
  p1_succ_result_name <- paste(region_list[region_index],"_p1_succ_result_all",sep="")
  p2_succ_result_name <- paste(region_list[region_index],"_p2_succ_result_all",sep="")
  p3_succ_result_name <- paste(region_list[region_index],"_p3_succ_result_all",sep="")
  p0_fail_cue_name <- paste(region_list[region_index],"_p0_fail_cue_all",sep="")
  p1_fail_cue_name <- paste(region_list[region_index],"_p1_fail_cue_all",sep="")
  p2_fail_cue_name <- paste(region_list[region_index],"_p2_fail_cue_all",sep="")
  p3_fail_cue_name <- paste(region_list[region_index],"_p3_fail_cue_all",sep="")
  p0_fail_result_name <- paste(region_list[region_index],"_p0_fail_result_all",sep="")
  p1_fail_result_name <- paste(region_list[region_index],"_p1_fail_result_all",sep="")
  p2_fail_result_name <- paste(region_list[region_index],"_p2_fail_result_all",sep="")
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
  
  r0_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r1_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r2_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r3_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r0_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r1_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r2_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r3_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r0_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r1_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r2_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r3_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r0_fail_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r1_fail_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r2_fail_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r3_fail_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  
  p0_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p1_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p2_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p3_succ_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p0_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p1_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p2_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p3_succ_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p0_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p1_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p2_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p3_fail_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p0_fail_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p1_fail_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p2_fail_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
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
    #cat('plotting unit', j,"\n")
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
    
    tmp2$p3_succ_cue <- rollmean(tmp$p3_succ_cue,3)
    tmp2$p2_succ_cue <- rollmean(tmp$p2_succ_cue,3)
    tmp2$p1_succ_cue <- rollmean(tmp$p1_succ_cue,3)
    tmp2$p0_succ_cue <- rollmean(tmp$p0_succ_cue,3)
    tmp2$r0_fail_cue <- rollmean(tmp$r0_fail_cue,3)
    tmp2$r1_fail_cue <- rollmean(tmp$r1_fail_cue,3)
    tmp2$r2_fail_cue <- rollmean(tmp$r2_fail_cue,3)
    tmp2$r3_fail_cue <- rollmean(tmp$r3_fail_cue,3)
    tmp2$p3_succ_result <- rollmean(tmp$p3_succ_result,3)
    tmp2$p2_succ_result <- rollmean(tmp$p2_succ_result,3)
    tmp2$p1_succ_result <- rollmean(tmp$p1_succ_result,3)
    tmp2$p0_succ_result <- rollmean(tmp$p0_succ_result,3)
    tmp2$r0_fail_result <- rollmean(tmp$r0_fail_result,3)
    tmp2$r1_fail_result <- rollmean(tmp$r1_fail_result,3)
    tmp2$r2_fail_result <- rollmean(tmp$r2_fail_result,3)
    tmp2$r3_fail_result <- rollmean(tmp$r3_fail_result,3)
    
    total_array[1:28,1:32,j] = data.matrix(tmp2)
    
    #making array to compare between units
    r0_succ_cue_matrix[j,] <- tmp2$r0_succ_cue
    r1_succ_cue_matrix[j,] <- tmp2$r1_succ_cue
    r2_succ_cue_matrix[j,] <- tmp2$r2_succ_cue
    r3_succ_cue_matrix[j,] <- tmp2$r3_succ_cue
    r0_succ_result_matrix[j,] <- tmp2$r0_succ_result
    r1_succ_result_matrix[j,] <- tmp2$r1_succ_result
    r2_succ_result_matrix[j,] <- tmp2$r2_succ_result
    r3_succ_result_matrix[j,] <- tmp2$r3_succ_result
    r0_fail_cue_matrix[j,] <- tmp2$r0_fail_cue
    r1_fail_cue_matrix[j,] <- tmp2$r1_fail_cue
    r2_fail_cue_matrix[j,] <- tmp2$r2_fail_cue
    r3_fail_cue_matrix[j,] <- tmp2$r3_fail_cue
    r0_fail_result_matrix[j,] <- tmp2$r0_fail_result
    r1_fail_result_matrix[j,] <- tmp2$r1_fail_result
    r2_fail_result_matrix[j,] <- tmp2$r2_fail_result
    r3_fail_result_matrix[j,] <- tmp2$r3_fail_result
    
    p0_succ_cue_matrix[j,] <- tmp2$p0_succ_cue
    p1_succ_cue_matrix[j,] <- tmp2$p1_succ_cue
    p2_succ_cue_matrix[j,] <- tmp2$p2_succ_cue
    p3_succ_cue_matrix[j,] <- tmp2$p3_succ_cue
    p0_succ_result_matrix[j,] <- tmp2$p0_succ_result
    p1_succ_result_matrix[j,] <- tmp2$p1_succ_result
    p2_succ_result_matrix[j,] <- tmp2$p2_succ_result
    p3_succ_result_matrix[j,] <- tmp2$p3_succ_result
    p0_fail_cue_matrix[j,] <- tmp2$p0_fail_cue
    p1_fail_cue_matrix[j,] <- tmp2$p1_fail_cue
    p2_fail_cue_matrix[j,] <- tmp2$p2_fail_cue
    p3_fail_cue_matrix[j,] <- tmp2$p3_fail_cue
    p0_fail_result_matrix[j,] <- tmp2$p0_fail_result
    p1_fail_result_matrix[j,] <- tmp2$p1_fail_result
    p2_fail_result_matrix[j,] <- tmp2$p2_fail_result
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
  assign(r0_succ_cue_name,r0_succ_cue_matrix)
  assign(r1_succ_cue_name,r1_succ_cue_matrix)
  assign(r2_succ_cue_name,r2_succ_cue_matrix)
  assign(r3_succ_cue_name,r3_succ_cue_matrix)
  assign(r0_succ_result_name,r0_succ_result_matrix)
  assign(r1_succ_result_name,r1_succ_result_matrix)
  assign(r2_succ_result_name,r2_succ_result_matrix)
  assign(r3_succ_result_name,r3_succ_result_matrix)
  assign(r0_fail_cue_name,r0_fail_cue_matrix)
  assign(r1_fail_cue_name,r1_fail_cue_matrix)
  assign(r2_fail_cue_name,r2_fail_cue_matrix)
  assign(r3_fail_cue_name,r3_fail_cue_matrix)
  assign(r0_fail_result_name,r0_fail_result_matrix)
  assign(r1_fail_result_name,r1_fail_result_matrix)
  assign(r2_fail_result_name,r2_fail_result_matrix)
  assign(r3_fail_result_name,r3_fail_result_matrix)
  assign(p0_succ_cue_name,p0_succ_cue_matrix)
  assign(p1_succ_cue_name,p1_succ_cue_matrix)
  assign(p2_succ_cue_name,p2_succ_cue_matrix)
  assign(p3_succ_cue_name,p3_succ_cue_matrix)
  assign(p0_succ_result_name,p0_succ_result_matrix)
  assign(p1_succ_result_name,p1_succ_result_matrix)
  assign(p2_succ_result_name,p2_succ_result_matrix)
  assign(p3_succ_result_name,p3_succ_result_matrix)
  assign(p0_fail_cue_name,p0_fail_cue_matrix)
  assign(p1_fail_cue_name,p1_fail_cue_matrix)
  assign(p2_fail_cue_name,p2_fail_cue_matrix)
  assign(p3_fail_cue_name,p3_fail_cue_matrix)
  assign(p0_fail_result_name,p0_fail_result_matrix)
  assign(p1_fail_result_name,p1_fail_result_matrix)
  assign(p2_fail_result_name,p2_fail_result_matrix)
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
  assign(total_array_name,total_array)
}

for(region_index in 1:length(catch_file_list)){
  cat("\nplotting region (catch):",region_list[region_index])
  
  filename = catch_file_list[region_index]
  wb <- loadWorkbook(filename)
  num_sheets <- length(sheets(wb))
  
  total_array_name <- paste(region_list[region_index],"_catch_unit_info",sep="")
  total_array <- array(NA,dim=c(28,20,num_sheets))
  
  r_all_catch_cue_name <- paste(region_list[region_index],"_r_all_catch_cue",sep="")
  r_all_catch_result_name <- paste(region_list[region_index],"_r_all_catch_result",sep="")
  p_all_catch_cue_name <- paste(region_list[region_index],"_p_all_catch_cue",sep="")
  p_all_catch_result_name <- paste(region_list[region_index],"_p_all_catch_result",sep="")
  
  r_all_catch_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  r_all_catch_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p_all_catch_cue_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  p_all_catch_result_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  
  for (j in 1:num_sheets){
    #cat('plotting unit', j,"catch trials\n")
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
    
    total_array[1:28,1:20,j] <- data.matrix(tmp2)
    
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
  assign(total_array_name,total_array)
}

M1_matrices <- abind(M1_r3_succ_cue_all,M1_r3_succ_result_all,M1_p3_fail_cue_all,M1_p3_fail_result_all,M1_all_r_succ_cue,M1_all_r_fail_cue,M1_all_r_succ_result,M1_all_r_fail_result,M1_all_p_succ_cue,M1_all_p_fail_cue,M1_all_p_succ_result,M1_all_p_fail_result,M1_no_r_succ_cue,M1_no_r_fail_cue,M1_no_r_succ_result,M1_no_r_fail_result,M1_no_p_succ_cue,M1_no_p_fail_cue,M1_no_p_succ_result,M1_no_p_fail_result,M1_r_all_catch_cue,M1_r_all_catch_result,M1_p_all_catch_cue,M1_p_all_catch_result,along=3)
M1_matrix_keys <- c('M1_r3_succ_cue','M1_r3_succ_result','M1_p3_fail_cue','M1_p3_fail_result','M1_all_r_succ_cue','M1_all_r_fail_cue','M1_all_r_succ_result','M1_all_r_fail_result','M1_all_p_succ_cue','M1_all_p_fail_cue','M1_all_p_succ_result','M1_all_p_fail_result','M1_no_r_succ_cue','M1_no_r_fail_cue','M1_no_r_succ_result','M1_no_r_fail_result','M1_no_p_succ_cue','M1_no_p_fail_cue','M1_no_p_succ_result','M1_no_p_fail_result','M1_r_all_catch_cue','M1_r_all_catch_result','M1_p_all_catch_cue','M1_p_all_catch_result')

S1_matrices <- abind(S1_r3_succ_cue_all,S1_r3_succ_result_all,S1_p3_fail_cue_all,S1_p3_fail_result_all,S1_all_r_succ_cue,S1_all_r_fail_cue,S1_all_r_succ_result,S1_all_r_fail_result,S1_all_p_succ_cue,S1_all_p_fail_cue,S1_all_p_succ_result,S1_all_p_fail_result,S1_no_r_succ_cue,S1_no_r_fail_cue,S1_no_r_succ_result,S1_no_r_fail_result,S1_no_p_succ_cue,S1_no_p_fail_cue,S1_no_p_succ_result,S1_no_p_fail_result,S1_r_all_catch_cue,S1_r_all_catch_result,S1_p_all_catch_cue,S1_p_all_catch_result,along=3)
S1_matrix_keys <- c('S1_r3_succ_cue','S1_r3_succ_result','S1_p3_fail_cue','S1_p3_fail_result','S1_all_r_succ_cue','S1_all_r_fail_cue','S1_all_r_succ_result','S1_all_r_fail_result','S1_all_p_succ_cue','S1_all_p_fail_cue','S1_all_p_succ_result','S1_all_p_fail_result','S1_no_r_succ_cue','S1_no_r_fail_cue','S1_no_r_succ_result','S1_no_r_fail_result','S1_no_p_succ_cue','S1_no_p_fail_cue','S1_no_p_succ_result','S1_no_p_fail_result','S1_r_all_catch_cue','S1_r_all_catch_result','S1_p_all_catch_cue','S1_p_all_catch_result')

PmD_matrices <- abind(PmD_r3_succ_cue_all,PmD_r3_succ_result_all,PmD_p3_fail_cue_all,PmD_p3_fail_result_all,PmD_all_r_succ_cue,PmD_all_r_fail_cue,PmD_all_r_succ_result,PmD_all_r_fail_result,PmD_all_p_succ_cue,PmD_all_p_fail_cue,PmD_all_p_succ_result,PmD_all_p_fail_result,PmD_no_r_succ_cue,PmD_no_r_fail_cue,PmD_no_r_succ_result,PmD_no_r_fail_result,PmD_no_p_succ_cue,PmD_no_p_fail_cue,PmD_no_p_succ_result,PmD_no_p_fail_result,PmD_r_all_catch_cue,PmD_r_all_catch_result,PmD_p_all_catch_cue,PmD_p_all_catch_result,along=3)
PmD_matrix_keys <- c('PmD_r3_succ_cue','PmD_r3_succ_result','PmD_p3_fail_cue','PmD_p3_fail_result','PmD_all_r_succ_cue','PmD_all_r_fail_cue','PmD_all_r_succ_result','PmD_all_r_fail_result','PmD_all_p_succ_cue','PmD_all_p_fail_cue','PmD_all_p_succ_result','PmD_all_p_fail_result','PmD_no_r_succ_cue','PmD_no_r_fail_cue','PmD_no_r_succ_result','PmD_no_r_fail_result','PmD_no_p_succ_cue','PmD_no_p_fail_cue','PmD_no_p_succ_result','PmD_no_p_fail_result','PmD_r_all_catch_cue','PmD_r_all_catch_result','PmD_p_all_catch_cue','PmD_p_all_catch_result')


#dev.off()
cat("\nM1 heatmaps")
for (i in 1:length(M1_matrix_keys)){
  png(paste(M1_matrix_keys[i],".png",sep=""),width=8,height=6,units="in",res=500)
  heatmap.2(M1_matrices[,,i],Colv=F,dendrogram="row",scale="row",col=rev(brewer.pal(11,"RdBu")),main=M1_matrix_keys[i],trace="none",cexRow=0.5,ylab="unit",xlab="time (s)",colsep=9)
  dev.off()
}

cat("\nS1 heatmaps")
for (i in 1:length(S1_matrix_keys)){
  png(paste(S1_matrix_keys[i],".png",sep=""),width=8,height=6,units="in",res=500)
  heatmap.2(S1_matrices[,,i],Colv=F,dendrogram="row",scale="row",col=rev(brewer.pal(11,"RdBu")),main=S1_matrix_keys[i],trace="none",cexRow=0.5,ylab="unit",xlab="time (s)",colsep=9)
  dev.off()
}

cat("\nPmD heatmaps")
for (i in 1:length(PmD_matrix_keys)){
  png(paste(PmD_matrix_keys[i],".png",sep=""),width=8,height=6,units="in",res=500)
  heatmap.2(PmD_matrices[,,i],Colv=F,dendrogram="row",scale="row",col=rev(brewer.pal(11,"RdBu")),main=PmD_matrix_keys[i],trace="none",cexRow=0.5,ylab="unit",xlab="time (s)",colsep=9)
  dev.off()
}

cat("\ngripforce plots")
gf_matrix_keys <- c('r3_succ_cue','r3_succ_result','p3_fail_cue','p3_fail_result','ra_succ_cue','ra_fail_cue','ra_succ_result','ra_fail_result','pa_succ_cue','pa_fail_cue','pa_succ_result','pa_fail_result','r0_succ_cue','r0_fail_cue','r0_succ_result','r0_fail_result','p0_succ_cue','p0_fail_cue','p0_succ_result','p0_fail_result','r_all_catch_cue','r_all_catch_result','p_all_catch_cue','p_all_catch_result')
for (i in 1:length(gf_matrix_keys)){
  avg_key <- paste(gf_matrix_keys[i],'_avg',sep="")
  std_key <- paste(gf_matrix_keys[i],'_std',sep="")
  
  gf <- gf_total[[avg_key]]
  std <- gf_total[[std_key]]
  
  plot_gf(gf,std,gf_matrix_keys[i])
  
}

cat("\nsaving")

save.image(file="rearranged_data.RData")
rm(list=ls())

