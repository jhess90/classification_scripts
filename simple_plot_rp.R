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

#file_list <- c('simple_output')
region_list <- c('M1','S1','PmD')
time <- seq(from=-0.35,to=1.0,by=0.05)


#########


for(region_index in 1:length(region_list)){
  cat("\nplotting region:",region_list[region_index])
  
  filename <- paste('simple_output_',region_list[region_index],'.mat',sep="")
  
  readin <- readMat(readMat())
  
  
  readin <- readMat(paste('simple_output_',region_list[region_index],'.mat',sep=""))
  
  
  
  
  
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
    
    tmp2 <- tmp[1:28,1:32]
    tmp2$p3_fail_cue <- rollmeanr(tmp$p3_fail_cue,3)
    tmp2$p2_fail_cue <- rollmeanr(tmp$p2_fail_cue,3)
    tmp2$p1_fail_cue <- rollmeanr(tmp$p1_fail_cue,3)
    tmp2$p0_fail_cue <- rollmeanr(tmp$p0_fail_cue,3)
    tmp2$r0_succ_cue <- rollmeanr(tmp$r0_succ_cue,3)
    tmp2$r1_succ_cue <- rollmeanr(tmp$r1_succ_cue,3)
    tmp2$r2_succ_cue <- rollmeanr(tmp$r2_succ_cue,3)
    tmp2$r3_succ_cue <- rollmeanr(tmp$r3_succ_cue,3)
    tmp2$p3_fail_result <- rollmeanr(tmp$p3_fail_result,3)
    tmp2$p2_fail_result <- rollmeanr(tmp$p2_fail_result,3)
    tmp2$p1_fail_result <- rollmeanr(tmp$p1_fail_result,3)
    tmp2$p0_fail_result <- rollmeanr(tmp$p0_fail_result,3)
    tmp2$r0_succ_result <- rollmeanr(tmp$r0_succ_result,3)
    tmp2$r1_succ_result <- rollmeanr(tmp$r1_succ_result,3)
    tmp2$r2_succ_result <- rollmeanr(tmp$r2_succ_result,3)
    tmp2$r3_succ_result <- rollmeanr(tmp$r3_succ_result,3)
    
    tmp2$p3_succ_cue <- rollmeanr(tmp$p3_succ_cue,3)
    tmp2$p2_succ_cue <- rollmeanr(tmp$p2_succ_cue,3)
    tmp2$p1_succ_cue <- rollmeanr(tmp$p1_succ_cue,3)
    tmp2$p0_succ_cue <- rollmeanr(tmp$p0_succ_cue,3)
    tmp2$r0_fail_cue <- rollmeanr(tmp$r0_fail_cue,3)
    tmp2$r1_fail_cue <- rollmeanr(tmp$r1_fail_cue,3)
    tmp2$r2_fail_cue <- rollmeanr(tmp$r2_fail_cue,3)
    tmp2$r3_fail_cue <- rollmeanr(tmp$r3_fail_cue,3)
    tmp2$p3_succ_result <- rollmeanr(tmp$p3_succ_result,3)
    tmp2$p2_succ_result <- rollmeanr(tmp$p2_succ_result,3)
    tmp2$p1_succ_result <- rollmeanr(tmp$p1_succ_result,3)
    tmp2$p0_succ_result <- rollmeanr(tmp$p0_succ_result,3)
    tmp2$r0_fail_result <- rollmeanr(tmp$r0_fail_result,3)
    tmp2$r1_fail_result <- rollmeanr(tmp$r1_fail_result,3)
    tmp2$r2_fail_result <- rollmeanr(tmp$r2_fail_result,3)
    tmp2$r3_fail_result <- rollmeanr(tmp$r3_fail_result,3)
    
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
    png(paste(region_list[region_index],"unit_",j,".png",sep=""),width=8,height=6,units="in",res=500)
    
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

}




# save.image(file="rearranged_data.RData")
rm(list=ls())

