library(openxlsx)
library(ggplot2)
library(reshape2)
#source("~/dropbox/mult_rp_files/r_test/multiplot.R")
#source("~/Dropbox/mult_rp_files/r_test/multiplot.R")
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
  
  #TODO make for diff bfr and aft times
  time <- seq(from=-0.5,to=(1.0-bin_size/1000),by=bin_size/1000)
  
  r0 <- which(condensed[,4] == 0)
  r1 <- which(condensed[,4] == 1)
  r2 <- which(condensed[,4] == 2)
  r3 <- which(condensed[,4] == 3)
  
  p0 <- which(condensed[,5] == 0)
  p1 <- which(condensed[,5] == 1)
  p2 <- which(condensed[,5] == 2)
  p3 <- which(condensed[,5] == 3)
  
  v_3 <- which(condensed[,7] == -3)
  v_2 <- which(condensed[,7] == -2)
  v_1 <- which(condensed[,7] == -1)
  v0 <- which(condensed[,7] == 0)
  v1 <- which(condensed[,7] == 1)
  v2 <- which(condensed[,7] == 2)
  v3 <- which(condensed[,7] == 3)
  
  m0 <- which(condensed[,8] == 0)
  m1 <- which(condensed[,8] == 1)
  m2 <- which(condensed[,8] == 2)
  m3 <- which(condensed[,8] == 3)
  m4 <- which(condensed[,8] == 4)
  m5 <- which(condensed[,8] == 5)
  m6 <- which(condensed[,8] == 6)
  
  res0 <- which(condensed[,6] == 0)
  res1 <- which(condensed[,6] == 1)
  
  #which(r0 %in% res0)
  
  for (unit_num in 1:length(dim(all_cue_fr[1]))){
    rp_vals <- c(0,1,2,3)
    
    r_cue_avgs <- list(colMeans(all_cue_fr[unit_num,r0,]),colMeans(all_cue_fr[unit_num,r1,]),colMeans(all_cue_fr[unit_num,r2,]),colMeans(all_cue_fr[unit_num,r3,]))
    r_res_avgs <- list(colMeans(all_res_fr[unit_num,r0,]),colMeans(all_res_fr[unit_num,r1,]),colMeans(all_res_fr[unit_num,r2,]),colMeans(all_res_fr[unit_num,r3,]))
    
        
    #r_avgs <- data.frame(r_vals=rp_vals,time=time,r_cue=r_cue_avgs,r_res=r_res_avgs)
    
    
    r_cue_avgs <- data.frame(time=time,r0=colMeans(all_cue_fr[unit_num,r0,]),r1=colMeans(all_cue_fr[unit_num,r1,]),r2=colMeans(all_cue_fr[unit_num,r2,]),r3=colMeans(all_cue_fr[unit_num,r3,]))
    r_res_avgs <- data.frame(time=time,r0=colMeans(all_res_fr[unit_num,r0,]),r1=colMeans(all_res_fr[unit_num,r1,]),r2=colMeans(all_res_fr[unit_num,r2,]),r3=colMeans(all_res_fr[unit_num,r3,]))
    
    
    r_cue_avgs.m <- melt(r_cue_avgs,id.vars="time",variable_name="r_level")
    plt_cue <- ggplot(r_cue_avgs.m,aes(time,r_level)) + geom_line(aes(colour=r_level))
    plt_cue <- plt_cue + scale_color_brewer(palette="RdYlGn")

    
    plt_val_all_cue <- ggplot(df_val_all_cue, aes(time,value)) + geom_line(aes(colour=vals),size=1.5)
    plt_val_all_cue <- plt_val_all_cue + scale_color_brewer(palette="RdYlGn") +labs(title=paste(region_list[region_index],"unit",j,"cue all value"),y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)
    
  }
  
  
  
  
  
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

