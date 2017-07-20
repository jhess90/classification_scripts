library(openxlsx)
library(ggplot2)
library(reshape2)
source("~/dropbox/mult_rp_files/r_test/multiplot.R")
#source("~/Dropbox/mult_rp_files/r_test/multiplot.R")
library(zoo)
library(gplots)
library(RColorBrewer)
library(abind)
library(gridGraphics)
library(grid)
library(gridExtra)

saveAsPng <- T

file_list <- c('nl_mv_avg_data_M1_dict_total.xlsx','nl_mv_avg_data_S1_dict_total.xlsx','nl_mv_avg_data_PmD_dict_total.xlsx')
region_list <- c('M1','S1','PmD')
time <- seq(from=-0.45,to=0.9,by=0.05)

gf_total <- read.xlsx('gf_mv_avg_data_M1_dict_total.xlsx',sheet=1,colNames=T)
gf_time <- seq(-0.45,0.90,by=0.005)

#########

plot_gf <- function(gf_value,std_value,key){
  
  png(paste(key,"_mv_gf.png",sep=""),width=6,height=4,units="in",res=500)

  gf_avg <- gf_value[10:280]
  gf_std <- std_value[10:280]
  
  upper <- gf_avg + gf_std
  lower <- gf_avg - gf_std
  gf_df <- data.frame(gf_time,gf_avg,upper,lower)
  gf_melt <- melt(gf_df,id.vars="gf_time",variable_name="value")
  p <- ggplot(data=gf_melt$gf_avg,aes(x=gf_time,y=gf_avg)) + geom_line() + theme(plot.margin=unit(c(0.5,1.5,0.5,3.0),"cm")) #margin order: top,right,btm,left

  p <- p + geom_ribbon(aes(ymin=gf_df$lower,ymax=gf_df$upper,alpha=0.15),show.legend = F) + geom_vline(xintercept=0)
  p <- p + labs(title=paste("gripforce",key),y="unit", x="time(s)")
  plot(p)
  dev.off()
  
}

for(region_index in 1:length(file_list)){
  cat("\nplotting region:",region_list[region_index])
  
  filename = file_list[region_index]
  wb <- loadWorkbook(filename)
  num_sheets <- length(sheets(wb))
  
  total_array_name <- paste(region_list[region_index],"_unit_info",sep="")
  total_array <- array(NA,dim=c(28,70,num_sheets))
  
  val_all_cue__3_name <- paste(region_list[region_index],"_val_all_cue_-3",sep="")
  val_all_cue__2_name <- paste(region_list[region_index],"_val_all_cue_-2",sep="")
  val_all_cue__1_name <- paste(region_list[region_index],"_val_all_cue_-1",sep="")
  val_all_cue_0_name <- paste(region_list[region_index],"_val_all_cue_0",sep="")
  val_all_cue_1_name <- paste(region_list[region_index],"_val_all_cue_1",sep="")
  val_all_cue_2_name <- paste(region_list[region_index],"_val_all_cue_2",sep="")
  val_all_cue_3_name <- paste(region_list[region_index],"_val_all_cue_3",sep="")
  val_succ_cue__3_name <- paste(region_list[region_index],"_val_succ_cue_-3",sep="")
  val_succ_cue__2_name <- paste(region_list[region_index],"_val_succ_cue_-2",sep="")
  val_succ_cue__1_name <- paste(region_list[region_index],"_val_succ_cue_-1",sep="")
  val_succ_cue_0_name <- paste(region_list[region_index],"_val_succ_cue_0",sep="")
  val_succ_cue_1_name <- paste(region_list[region_index],"_val_succ_cue_1",sep="")
  val_succ_cue_2_name <- paste(region_list[region_index],"_val_succ_cue_2",sep="")
  val_succ_cue_3_name <- paste(region_list[region_index],"_val_succ_cue_3",sep="")
  val_fail_cue__3_name <- paste(region_list[region_index],"_val_fail_cue_-3",sep="")
  val_fail_cue__2_name <- paste(region_list[region_index],"_val_fail_cue_-2",sep="")
  val_fail_cue__1_name <- paste(region_list[region_index],"_val_fail_cue_-1",sep="")
  val_fail_cue_0_name <- paste(region_list[region_index],"_val_fail_cue_0",sep="")
  val_fail_cue_1_name <- paste(region_list[region_index],"_val_fail_cue_1",sep="")
  val_fail_cue_2_name <- paste(region_list[region_index],"_val_fail_cue_2",sep="")
  val_fail_cue_3_name <- paste(region_list[region_index],"_val_fail_cue_3",sep="")
  val_succ_result__3_name <- paste(region_list[region_index],"_val_succ_result_-3",sep="")
  val_succ_result__2_name <- paste(region_list[region_index],"_val_succ_result_-2",sep="")
  val_succ_result__1_name <- paste(region_list[region_index],"_val_succ_result_-1",sep="")
  val_succ_result_0_name <- paste(region_list[region_index],"_val_succ_result_0",sep="")
  val_succ_result_1_name <- paste(region_list[region_index],"_val_succ_result_1",sep="")
  val_succ_result_2_name <- paste(region_list[region_index],"_val_succ_result_2",sep="")
  val_succ_result_3_name <- paste(region_list[region_index],"_val_succ_result_3",sep="")
  val_fail_result__3_name <- paste(region_list[region_index],"_val_fail_result_-3",sep="")
  val_fail_result__2_name <- paste(region_list[region_index],"_val_fail_result_-2",sep="")
  val_fail_result__1_name <- paste(region_list[region_index],"_val_fail_result_-1",sep="")
  val_fail_result_0_name <- paste(region_list[region_index],"_val_fail_result_0",sep="")
  val_fail_result_1_name <- paste(region_list[region_index],"_val_fail_result_1",sep="")
  val_fail_result_2_name <- paste(region_list[region_index],"_val_fail_result_2",sep="")
  val_fail_result_3_name <- paste(region_list[region_index],"_val_fail_result_3",sep="")
  
  mtv_all_cue_0_name <- paste(region_list[region_index],"_mtv_all_cue_0",sep="")
  mtv_all_cue_1_name <- paste(region_list[region_index],"_mtv_all_cue_1",sep="")
  mtv_all_cue_2_name <- paste(region_list[region_index],"_mtv_all_cue_2",sep="")
  mtv_all_cue_3_name <- paste(region_list[region_index],"_mtv_all_cue_3",sep="")
  mtv_all_cue_4_name <- paste(region_list[region_index],"_mtv_all_cue_4",sep="")
  mtv_all_cue_5_name <- paste(region_list[region_index],"_mtv_all_cue_5",sep="")
  mtv_all_cue_6_name <- paste(region_list[region_index],"_mtv_all_cue_6",sep="")
  mtv_succ_cue_0_name <- paste(region_list[region_index],"_mtv_succ_cue_0",sep="")
  mtv_succ_cue_1_name <- paste(region_list[region_index],"_mtv_succ_cue_1",sep="")
  mtv_succ_cue_2_name <- paste(region_list[region_index],"_mtv_succ_cue_2",sep="")
  mtv_succ_cue_3_name <- paste(region_list[region_index],"_mtv_succ_cue_3",sep="")
  mtv_succ_cue_4_name <- paste(region_list[region_index],"_mtv_succ_cue_4",sep="")
  mtv_succ_cue_5_name <- paste(region_list[region_index],"_mtv_succ_cue_5",sep="")
  mtv_succ_cue_6_name <- paste(region_list[region_index],"_mtv_succ_cue_6",sep="")
  mtv_fail_cue_0_name <- paste(region_list[region_index],"_mtv_fail_cue_0",sep="")
  mtv_fail_cue_1_name <- paste(region_list[region_index],"_mtv_fail_cue_1",sep="")
  mtv_fail_cue_2_name <- paste(region_list[region_index],"_mtv_fail_cue_2",sep="")
  mtv_fail_cue_3_name <- paste(region_list[region_index],"_mtv_fail_cue_3",sep="")
  mtv_fail_cue_4_name <- paste(region_list[region_index],"_mtv_fail_cue_4",sep="")
  mtv_fail_cue_5_name <- paste(region_list[region_index],"_mtv_fail_cue_5",sep="")
  mtv_fail_cue_6_name <- paste(region_list[region_index],"_mtv_fail_cue_6",sep="")
  mtv_succ_result_0_name <- paste(region_list[region_index],"_mtv_succ_result_0",sep="")
  mtv_succ_result_1_name <- paste(region_list[region_index],"_mtv_succ_result_1",sep="")
  mtv_succ_result_2_name <- paste(region_list[region_index],"_mtv_succ_result_2",sep="")
  mtv_succ_result_3_name <- paste(region_list[region_index],"_mtv_succ_result_3",sep="")
  mtv_succ_result_4_name <- paste(region_list[region_index],"_mtv_succ_result_4",sep="")
  mtv_succ_result_5_name <- paste(region_list[region_index],"_mtv_succ_result_5",sep="")
  mtv_succ_result_6_name <- paste(region_list[region_index],"_mtv_succ_result_6",sep="")
  mtv_fail_result_0_name <- paste(region_list[region_index],"_mtv_fail_result_0",sep="")
  mtv_fail_result_1_name <- paste(region_list[region_index],"_mtv_fail_result_1",sep="")
  mtv_fail_result_2_name <- paste(region_list[region_index],"_mtv_fail_result_2",sep="")
  mtv_fail_result_3_name <- paste(region_list[region_index],"_mtv_fail_result_3",sep="")
  mtv_fail_result_4_name <- paste(region_list[region_index],"_mtv_fail_result_4",sep="")
  mtv_fail_result_5_name <- paste(region_list[region_index],"_mtv_fail_result_5",sep="")
  mtv_fail_result_6_name <- paste(region_list[region_index],"_mtv_fail_result_6",sep="")

  val_all_cue__3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_all_cue__2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_all_cue__1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_all_cue_0_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_all_cue_1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_all_cue_2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_all_cue_3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_cue__3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_cue__2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_cue__1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_cue_0_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_cue_1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_cue_2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_cue_3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_cue__3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_cue__2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_cue__1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_cue_0_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_cue_1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_cue_2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_cue_3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_result__3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_result__2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_result__1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_result_0_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_result_1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_result_2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_succ_result_3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_result__3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_result__2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_result__1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_result_0_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_result_1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_result_2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  val_fail_result_3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  
  mtv_all_cue_0_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_all_cue_1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_all_cue_2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_all_cue_3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_all_cue_4_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_all_cue_5_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_all_cue_6_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_cue_0_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_cue_1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_cue_2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_cue_3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_cue_4_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_cue_5_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_cue_6_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_cue_0_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_cue_1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_cue_2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_cue_3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_cue_4_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_cue_5_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_cue_6_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_result_0_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_result_1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_result_2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_result_3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_result_4_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_result_5_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_succ_result_6_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_result_0_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_result_1_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_result_2_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_result_3_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_result_4_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_result_5_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  mtv_fail_result_6_matrix <- matrix(NA,nrow=num_sheets,ncol=28,dimnames=list(1:num_sheets,time))
  
  for (j in 1:num_sheets){
    tmp <- read.xlsx(filename,sheet = j, colNames=T)
    png(paste('mv_',region_list[region_index],"unit_",j,".png",sep=""),width=8,height=6,units="in",res=500)
    
    tmp2 <- tmp[1:28,1:70]
    for (k in 1:length(tmp2)){
      tmp2[,k] <- rollmeanr(tmp[,k],3)
    }
    total_array[1:28,1:70,j] = data.matrix(tmp2)
    
    #making array to compare between units
    val_all_cue__3_matrix[j,] <- tmp2$`val_all_cue_-3`
    val_all_cue__2_matrix[j,] <- tmp2$`val_all_cue_-2`
    val_all_cue__1_matrix[j,] <- tmp2$`val_all_cue_-1`
    val_all_cue_0_matrix[j,] <- tmp2$`val_all_cue_0`
    val_all_cue_1_matrix[j,] <- tmp2$`val_all_cue_1`
    val_all_cue_2_matrix[j,] <- tmp2$`val_all_cue_2`
    val_all_cue_3_matrix[j,] <- tmp2$`val_all_cue_3`
    val_succ_cue__3_matrix[j,] <- tmp2$`val_succ_cue_-3`
    val_succ_cue__2_matrix[j,] <- tmp2$`val_succ_cue_-2`
    val_succ_cue__1_matrix[j,] <- tmp2$`val_succ_cue_-1`
    val_succ_cue_0_matrix[j,] <- tmp2$`val_succ_cue_0`
    val_succ_cue_1_matrix[j,] <- tmp2$`val_succ_cue_1`
    val_succ_cue_2_matrix[j,] <- tmp2$`val_succ_cue_2`
    val_succ_cue_3_matrix[j,] <- tmp2$`val_succ_cue_3`
    val_fail_cue__3_matrix[j,] <- tmp2$`val_fail_cue_-3`
    val_fail_cue__2_matrix[j,] <- tmp2$`val_fail_cue_-2`
    val_fail_cue__1_matrix[j,] <- tmp2$`val_fail_cue_-1`
    val_fail_cue_0_matrix[j,] <- tmp2$`val_fail_cue_0`
    val_fail_cue_1_matrix[j,] <- tmp2$`val_fail_cue_1`
    val_fail_cue_2_matrix[j,] <- tmp2$`val_fail_cue_2`
    val_fail_cue_3_matrix[j,] <- tmp2$`val_fail_cue_3`
    val_succ_result__3_matrix[j,] <- tmp2$`val_succ_result_-3`
    val_succ_result__2_matrix[j,] <- tmp2$`val_succ_result_-2`
    val_succ_result__1_matrix[j,] <- tmp2$`val_succ_result_-1`
    val_succ_result_0_matrix[j,] <- tmp2$`val_succ_result_0`
    val_succ_result_1_matrix[j,] <- tmp2$`val_succ_result_1`
    val_succ_result_2_matrix[j,] <- tmp2$`val_succ_result_2`
    val_succ_result_3_matrix[j,] <- tmp2$`val_succ_result_3`
    val_fail_result__3_matrix[j,] <- tmp2$`val_fail_result_-3`
    val_fail_result__2_matrix[j,] <- tmp2$`val_fail_result_-2`
    val_fail_result__1_matrix[j,] <- tmp2$`val_fail_result_-1`
    val_fail_result_0_matrix[j,] <- tmp2$`val_fail_result_0`
    val_fail_result_1_matrix[j,] <- tmp2$`val_fail_result_1`
    val_fail_result_2_matrix[j,] <- tmp2$`val_fail_result_2`
    val_fail_result_3_matrix[j,] <- tmp2$`val_fail_result_3`
    
    mtv_all_cue_0_matrix[j,] <- tmp2$`mtv_all_cue_0`
    mtv_all_cue_1_matrix[j,] <- tmp2$`mtv_all_cue_1`
    mtv_all_cue_2_matrix[j,] <- tmp2$`mtv_all_cue_2`
    mtv_all_cue_3_matrix[j,] <- tmp2$`mtv_all_cue_3`
    mtv_all_cue_4_matrix[j,] <- tmp2$`mtv_all_cue_4`
    mtv_all_cue_5_matrix[j,] <- tmp2$`mtv_all_cue_5`
    mtv_all_cue_6_matrix[j,] <- tmp2$`mtv_all_cue_6`
    mtv_succ_cue_0_matrix[j,] <- tmp2$`mtv_succ_cue_0`
    mtv_succ_cue_1_matrix[j,] <- tmp2$`mtv_succ_cue_1`
    mtv_succ_cue_2_matrix[j,] <- tmp2$`mtv_succ_cue_2`
    mtv_succ_cue_3_matrix[j,] <- tmp2$`mtv_succ_cue_3`
    mtv_succ_cue_4_matrix[j,] <- tmp2$`mtv_succ_cue_4`
    mtv_succ_cue_5_matrix[j,] <- tmp2$`mtv_succ_cue_5`
    mtv_succ_cue_6_matrix[j,] <- tmp2$`mtv_succ_cue_6`
    mtv_fail_cue_0_matrix[j,] <- tmp2$`mtv_fail_cue_0`
    mtv_fail_cue_1_matrix[j,] <- tmp2$`mtv_fail_cue_1`
    mtv_fail_cue_2_matrix[j,] <- tmp2$`mtv_fail_cue_2`
    mtv_fail_cue_3_matrix[j,] <- tmp2$`mtv_fail_cue_3`
    mtv_fail_cue_4_matrix[j,] <- tmp2$`mtv_fail_cue_4`
    mtv_fail_cue_5_matrix[j,] <- tmp2$`mtv_fail_cue_5`
    mtv_fail_cue_6_matrix[j,] <- tmp2$`mtv_fail_cue_6`
    mtv_succ_result_0_matrix[j,] <- tmp2$`mtv_succ_result_0`
    mtv_succ_result_1_matrix[j,] <- tmp2$`mtv_succ_result_1`
    mtv_succ_result_2_matrix[j,] <- tmp2$`mtv_succ_result_2`
    mtv_succ_result_3_matrix[j,] <- tmp2$`mtv_succ_result_3`
    mtv_succ_result_4_matrix[j,] <- tmp2$`mtv_succ_result_4`
    mtv_succ_result_5_matrix[j,] <- tmp2$`mtv_succ_result_5`
    mtv_succ_result_6_matrix[j,] <- tmp2$`mtv_succ_result_6`
    mtv_fail_result_0_matrix[j,] <- tmp2$`mtv_fail_result_0`
    mtv_fail_result_1_matrix[j,] <- tmp2$`mtv_fail_result_1`
    mtv_fail_result_2_matrix[j,] <- tmp2$`mtv_fail_result_2`
    mtv_fail_result_3_matrix[j,] <- tmp2$`mtv_fail_result_3`
    mtv_fail_result_4_matrix[j,] <- tmp2$`mtv_fail_result_4`
    mtv_fail_result_5_matrix[j,] <- tmp2$`mtv_fail_result_5`
    mtv_fail_result_6_matrix[j,] <- tmp2$`mtv_fail_result_6`
    
    df_val_all_cue <- data.frame(time,val__3=tmp2$`val_all_cue_-3`,val__2=tmp2$`val_all_cue_-2`,val__1=tmp2$`val_all_cue_-2`,val_0=tmp2$`val_all_cue_0`,val_1=tmp2$`val_all_cue_1`,val_2=tmp2$`val_all_cue_2`,val_3=tmp2$`val_all_cue_3`)
    df_val_all_cue <- melt(df_val_all_cue,id.vars="time",variable_name="value")
    plt_val_all_cue <- ggplot(df_val_all_cue, aes(time,value)) + geom_line(aes(colour=variable))
    plt_val_all_cue <- plt_val_all_cue + scale_color_brewer(palette="RdYlGn") +labs(title=paste(region_list[region_index],"unit",j,"cue all value"),y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)
    
    df_val_succ_result <- data.frame(time,val__3=tmp2$`val_succ_result_-3`,val__2=tmp2$`val_succ_result_-2`,val__1=tmp2$`val_succ_result_-2`,val_0=tmp2$`val_succ_result_0`,val_1=tmp2$`val_succ_result_1`,val_2=tmp2$`val_succ_result_2`,val_3=tmp2$`val_succ_result_3`)
    df_val_succ_result <- melt(df_val_succ_result,id.vars="time",variable_name="value")
    plt_val_succ_result <- ggplot(df_val_succ_result, aes(time,value)) + geom_line(aes(colour=variable))
    plt_val_succ_result <- plt_val_succ_result + scale_color_brewer(palette="RdYlGn") +labs(title=paste(region_list[region_index],"unit",j,"result succ value"),y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)
    
    df_val_fail_result <- data.frame(time,val__3=tmp2$`val_fail_result_-3`,val__2=tmp2$`val_fail_result_-2`,val__1=tmp2$`val_fail_result_-2`,val_0=tmp2$`val_fail_result_0`,val_1=tmp2$`val_fail_result_1`,val_2=tmp2$`val_fail_result_2`,val_3=tmp2$`val_fail_result_3`)
    df_val_fail_result <- melt(df_val_fail_result,id.vars="time",variable_name="value")
    plt_val_fail_result <- ggplot(df_val_fail_result, aes(time,value)) + geom_line(aes(colour=variable))
    plt_val_fail_result <- plt_val_fail_result + scale_color_brewer(palette="RdYlGn") +labs(title=paste(region_list[region_index],"unit",j,"result fail value"),y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)
    
    df_mtv_all_cue <- data.frame(time,mtv_0=tmp2$`mtv_all_cue_0`,mtv_1=tmp2$`mtv_all_cue_1`,mtv_2=tmp2$`mtv_all_cue_2`,mtv_3=tmp2$`mtv_all_cue_3`,mtv_4=tmp2$`mtv_all_cue_4`,mtv_5=tmp2$`mtv_all_cue_5`,mtv_6=tmp2$`mtv_all_cue_6`)
    df_mtv_all_cue <- melt(df_mtv_all_cue,id.vars="time",variable_name="value")
    plt_mtv_all_cue <- ggplot(df_mtv_all_cue, aes(time,value)) + geom_line(aes(colour=variable))
    plt_mtv_all_cue <- plt_mtv_all_cue + scale_color_brewer(palette="RdYlGn") +labs(title=paste(region_list[region_index],"unit",j,"cue all motivation"),y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)
    
    df_mtv_succ_result <- data.frame(time,mtv_0=tmp2$`mtv_succ_result_0`,mtv_1=tmp2$`mtv_succ_result_1`,mtv_2=tmp2$`mtv_succ_result_2`,mtv_3=tmp2$`mtv_succ_result_3`,mtv_4=tmp2$`mtv_succ_result_4`,mtv_5=tmp2$`mtv_succ_result_5`,mtv_6=tmp2$`mtv_succ_result_6`)
    df_mtv_succ_result <- melt(df_mtv_succ_result,id.vars="time",variable_name="value")
    plt_mtv_succ_result <- ggplot(df_mtv_succ_result, aes(time,value)) + geom_line(aes(colour=variable))
    plt_mtv_succ_result <- plt_mtv_succ_result + scale_color_brewer(palette="RdYlGn") +labs(title=paste(region_list[region_index],"unit",j,"result succ motivation"),y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)
    
    df_mtv_fail_result <- data.frame(time,mtv_0=tmp2$`mtv_fail_result_0`,mtv_1=tmp2$`mtv_fail_result_1`,mtv_2=tmp2$`mtv_fail_result_2`,mtv_3=tmp2$`mtv_fail_result_3`,mtv_4=tmp2$`mtv_fail_result_4`,mtv_5=tmp2$`mtv_fail_result_5`,mtv_6=tmp2$`mtv_fail_result_6`)
    df_mtv_fail_result <- melt(df_mtv_fail_result,id.vars="time",variable_name="value")
    plt_mtv_fail_result <- ggplot(df_mtv_fail_result, aes(time,value)) + geom_line(aes(colour=variable))
    plt_mtv_fail_result <- plt_mtv_fail_result + scale_color_brewer(palette="RdYlGn") +labs(title=paste(region_list[region_index],"unit",j,"result fail motivation"),y="normalized firing rate", x="time(s)") + geom_vline(xintercept=0)

    png(paste(region_list[region_index],"_mv_multilevels_unit_",j,".png",sep=""),width=8,height=6,units="in",res=500)
    multiplot(plt_val_all_cue,plt_val_succ_result,plt_val_fail_result,plt_mtv_all_cue,plt_mtv_succ_result,plt_mtv_fail_result,cols=2)
    
    #dev.off()
    graphics.off()
    rm(tmp)
    }

  assign(val_all_cue__3_name,val_all_cue__3_matrix)
  assign(val_all_cue__2_name,val_all_cue__2_matrix)
  assign(val_all_cue__1_name,val_all_cue__1_matrix)
  assign(val_all_cue_0_name,val_all_cue_0_matrix)
  assign(val_all_cue_1_name,val_all_cue_1_matrix)
  assign(val_all_cue_2_name,val_all_cue_2_matrix)
  assign(val_all_cue_3_name,val_all_cue_3_matrix)
  assign(val_succ_cue__3_name,val_succ_cue__3_matrix)
  assign(val_succ_cue__2_name,val_succ_cue__2_matrix)
  assign(val_succ_cue__1_name,val_succ_cue__1_matrix)
  assign(val_succ_cue_0_name,val_succ_cue_0_matrix)
  assign(val_succ_cue_1_name,val_succ_cue_1_matrix)
  assign(val_succ_cue_2_name,val_succ_cue_2_matrix)
  assign(val_succ_cue_3_name,val_succ_cue_3_matrix)
  assign(val_fail_cue__3_name,val_fail_cue__3_matrix)
  assign(val_fail_cue__2_name,val_fail_cue__2_matrix)
  assign(val_fail_cue__1_name,val_fail_cue__1_matrix)
  assign(val_fail_cue_0_name,val_fail_cue_0_matrix)
  assign(val_fail_cue_1_name,val_fail_cue_1_matrix)
  assign(val_fail_cue_2_name,val_fail_cue_2_matrix)
  assign(val_fail_cue_3_name,val_fail_cue_3_matrix)
  assign(val_succ_result__3_name,val_succ_result__3_matrix)
  assign(val_succ_result__2_name,val_succ_result__2_matrix)
  assign(val_succ_result__1_name,val_succ_result__1_matrix)
  assign(val_succ_result_0_name,val_succ_result_0_matrix)
  assign(val_succ_result_1_name,val_succ_result_1_matrix)
  assign(val_succ_result_2_name,val_succ_result_2_matrix)
  assign(val_succ_result_3_name,val_succ_result_3_matrix)
  assign(val_fail_result__3_name,val_fail_result__3_matrix)
  assign(val_fail_result__2_name,val_fail_result__2_matrix)
  assign(val_fail_result__1_name,val_fail_result__1_matrix)
  assign(val_fail_result_0_name,val_fail_result_0_matrix)
  assign(val_fail_result_1_name,val_fail_result_1_matrix)
  assign(val_fail_result_2_name,val_fail_result_2_matrix)
  assign(val_fail_result_3_name,val_fail_result_3_matrix)
  
  assign(mtv_all_cue_0_name,mtv_all_cue_0_matrix)
  assign(mtv_all_cue_1_name,mtv_all_cue_1_matrix)
  assign(mtv_all_cue_2_name,mtv_all_cue_2_matrix)
  assign(mtv_all_cue_3_name,mtv_all_cue_3_matrix)
  assign(mtv_all_cue_4_name,mtv_all_cue_4_matrix)
  assign(mtv_all_cue_5_name,mtv_all_cue_5_matrix)
  assign(mtv_all_cue_6_name,mtv_all_cue_6_matrix)
  assign(mtv_succ_cue_0_name,mtv_succ_cue_0_matrix)
  assign(mtv_succ_cue_1_name,mtv_succ_cue_1_matrix)
  assign(mtv_succ_cue_2_name,mtv_succ_cue_2_matrix)
  assign(mtv_succ_cue_3_name,mtv_succ_cue_3_matrix)
  assign(mtv_succ_cue_4_name,mtv_succ_cue_4_matrix)
  assign(mtv_succ_cue_5_name,mtv_succ_cue_5_matrix)
  assign(mtv_succ_cue_6_name,mtv_succ_cue_6_matrix)
  assign(mtv_fail_cue_0_name,mtv_fail_cue_0_matrix)
  assign(mtv_fail_cue_1_name,mtv_fail_cue_1_matrix)
  assign(mtv_fail_cue_2_name,mtv_fail_cue_2_matrix)
  assign(mtv_fail_cue_3_name,mtv_fail_cue_3_matrix)
  assign(mtv_fail_cue_4_name,mtv_fail_cue_4_matrix)
  assign(mtv_fail_cue_5_name,mtv_fail_cue_5_matrix)
  assign(mtv_fail_cue_6_name,mtv_fail_cue_6_matrix)
  assign(mtv_succ_result_0_name,mtv_succ_result_0_matrix)
  assign(mtv_succ_result_1_name,mtv_succ_result_1_matrix)
  assign(mtv_succ_result_2_name,mtv_succ_result_2_matrix)
  assign(mtv_succ_result_3_name,mtv_succ_result_3_matrix)
  assign(mtv_succ_result_4_name,mtv_succ_result_4_matrix)
  assign(mtv_succ_result_5_name,mtv_succ_result_5_matrix)
  assign(mtv_succ_result_6_name,mtv_succ_result_6_matrix)
  assign(mtv_fail_result_0_name,mtv_fail_result_0_matrix)
  assign(mtv_fail_result_1_name,mtv_fail_result_1_matrix)
  assign(mtv_fail_result_2_name,mtv_fail_result_2_matrix)
  assign(mtv_fail_result_3_name,mtv_fail_result_3_matrix)
  assign(mtv_fail_result_4_name,mtv_fail_result_4_matrix)
  assign(mtv_fail_result_5_name,mtv_fail_result_5_matrix)
  assign(mtv_fail_result_6_name,mtv_fail_result_6_matrix)
  
  assign(total_array_name,total_array)
}

#TODO succ/fail cue

M1_matrices <- abind(`M1_val_all_cue_-3`,`M1_val_all_cue_-2`,`M1_val_all_cue_-1`,`M1_val_all_cue_0`,`M1_val_all_cue_1`,`M1_val_all_cue_2`,`M1_val_all_cue_3`, 
                     `M1_val_succ_result_-3`,`M1_val_succ_result_-2`,`M1_val_succ_result_-1`,`M1_val_succ_result_0`,`M1_val_succ_result_1`,`M1_val_succ_result_2`,`M1_val_succ_result_3`,
                     `M1_val_fail_result_-3`,`M1_val_fail_result_-2`,`M1_val_fail_result_-1`,`M1_val_fail_result_0`,`M1_val_fail_result_1`,`M1_val_fail_result_2`,`M1_val_fail_result_3`,
                     `M1_mtv_all_cue_0`,`M1_mtv_all_cue_1`,`M1_mtv_all_cue_2`,`M1_mtv_all_cue_3`,`M1_mtv_all_cue_4`,`M1_mtv_all_cue_5`,`M1_mtv_all_cue_6`,
                     `M1_mtv_succ_result_0`,`M1_mtv_succ_result_1`,`M1_mtv_succ_result_2`,`M1_mtv_succ_result_3`,`M1_mtv_succ_result_4`,`M1_mtv_succ_result_5`,`M1_mtv_succ_result_6`,
                     `M1_mtv_fail_result_0`,`M1_mtv_fail_result_1`,`M1_mtv_fail_result_2`,`M1_mtv_fail_result_3`,`M1_mtv_fail_result_4`,`M1_mtv_fail_result_5`,`M1_mtv_fail_result_6`)

M1_matrix_keys <- c('M1_val_all_cue_-3','M1_val_all_cue_-2','M1_val_all_cue_-1','M1_val_all_cue_0','M1_val_all_cue_1','M1_val_all_cue_2','M1_val_all_cue_3', 
                     'M1_val_succ_result_-3','M1_val_succ_result_-2','M1_val_succ_result_-1','M1_val_succ_result_0','M1_val_succ_result_1','M1_val_succ_result_2','M1_val_succ_result_3',
                     'M1_val_fail_result_-3','M1_val_fail_result_-2','M1_val_fail_result_-1','M1_val_fail_result_0','M1_val_fail_result_1','M1_val_fail_result_2','M1_val_fail_result_3',
                     'M1_mtv_all_cue_0','M1_mtv_all_cue_1','M1_mtv_all_cue_2','M1_mtv_all_cue_3','M1_mtv_all_cue_4','M1_mtv_all_cue_5','M1_mtv_all_cue_6',
                     'M1_mtv_succ_result_0','M1_mtv_succ_result_1','M1_mtv_succ_result_2','M1_mtv_succ_result_3','M1_mtv_succ_result_4','M1_mtv_succ_result_5','M1_mtv_succ_result_6',
                     'M1_mtv_fail_result_0','M1_mtv_fail_result_1','M1_mtv_fail_result_2','M1_mtv_fail_result_3','M1_mtv_fail_result_4','M1_mtv_fail_result_5','M1_mtv_fail_result_6')

S1_matrices <- abind(`S1_val_all_cue_-3`,`S1_val_all_cue_-2`,`S1_val_all_cue_-1`,`S1_val_all_cue_0`,`S1_val_all_cue_1`,`S1_val_all_cue_2`,`S1_val_all_cue_3`, 
                     `S1_val_succ_result_-3`,`S1_val_succ_result_-2`,`S1_val_succ_result_-1`,`S1_val_succ_result_0`,`S1_val_succ_result_1`,`S1_val_succ_result_2`,`S1_val_succ_result_3`,
                     `S1_val_fail_result_-3`,`S1_val_fail_result_-2`,`S1_val_fail_result_-1`,`S1_val_fail_result_0`,`S1_val_fail_result_1`,`S1_val_fail_result_2`,`S1_val_fail_result_3`,
                     `S1_mtv_all_cue_0`,`S1_mtv_all_cue_1`,`S1_mtv_all_cue_2`,`S1_mtv_all_cue_3`,`S1_mtv_all_cue_4`,`S1_mtv_all_cue_5`,`S1_mtv_all_cue_6`,
                     `S1_mtv_succ_result_0`,`S1_mtv_succ_result_1`,`S1_mtv_succ_result_2`,`S1_mtv_succ_result_3`,`S1_mtv_succ_result_4`,`S1_mtv_succ_result_5`,`S1_mtv_succ_result_6`,
                     `S1_mtv_fail_result_0`,`S1_mtv_fail_result_1`,`S1_mtv_fail_result_2`,`S1_mtv_fail_result_3`,`S1_mtv_fail_result_4`,`S1_mtv_fail_result_5`,`S1_mtv_fail_result_6`)

S1_matrix_keys <- c('S1_val_all_cue_-3','S1_val_all_cue_-2','S1_val_all_cue_-1','S1_val_all_cue_0','S1_val_all_cue_1','S1_val_all_cue_2','S1_val_all_cue_3', 
                    'S1_val_succ_result_-3','S1_val_succ_result_-2','S1_val_succ_result_-1','S1_val_succ_result_0','S1_val_succ_result_1','S1_val_succ_result_2','S1_val_succ_result_3',
                    'S1_val_fail_result_-3','S1_val_fail_result_-2','S1_val_fail_result_-1','S1_val_fail_result_0','S1_val_fail_result_1','S1_val_fail_result_2','S1_val_fail_result_3',
                    'S1_mtv_all_cue_0','S1_mtv_all_cue_1','S1_mtv_all_cue_2','S1_mtv_all_cue_3','S1_mtv_all_cue_4','S1_mtv_all_cue_5','S1_mtv_all_cue_6',
                    'S1_mtv_succ_result_0','S1_mtv_succ_result_1','S1_mtv_succ_result_2','S1_mtv_succ_result_3','S1_mtv_succ_result_4','S1_mtv_succ_result_5','S1_mtv_succ_result_6',
                    'S1_mtv_fail_result_0','S1_mtv_fail_result_1','S1_mtv_fail_result_2','S1_mtv_fail_result_3','S1_mtv_fail_result_4','S1_mtv_fail_result_5','S1_mtv_fail_result_6')

PmD_matrices <- abind(`PmD_val_all_cue_-3`,`PmD_val_all_cue_-2`,`PmD_val_all_cue_-1`,`PmD_val_all_cue_0`,`PmD_val_all_cue_1`,`PmD_val_all_cue_2`,`PmD_val_all_cue_3`, 
                     `PmD_val_succ_result_-3`,`PmD_val_succ_result_-2`,`PmD_val_succ_result_-1`,`PmD_val_succ_result_0`,`PmD_val_succ_result_1`,`PmD_val_succ_result_2`,`PmD_val_succ_result_3`,
                     `PmD_val_fail_result_-3`,`PmD_val_fail_result_-2`,`PmD_val_fail_result_-1`,`PmD_val_fail_result_0`,`PmD_val_fail_result_1`,`PmD_val_fail_result_2`,`PmD_val_fail_result_3`,
                     `PmD_mtv_all_cue_0`,`PmD_mtv_all_cue_1`,`PmD_mtv_all_cue_2`,`PmD_mtv_all_cue_3`,`PmD_mtv_all_cue_4`,`PmD_mtv_all_cue_5`,`PmD_mtv_all_cue_6`,
                     `PmD_mtv_succ_result_0`,`PmD_mtv_succ_result_1`,`PmD_mtv_succ_result_2`,`PmD_mtv_succ_result_3`,`PmD_mtv_succ_result_4`,`PmD_mtv_succ_result_5`,`PmD_mtv_succ_result_6`,
                     `PmD_mtv_fail_result_0`,`PmD_mtv_fail_result_1`,`PmD_mtv_fail_result_2`,`PmD_mtv_fail_result_3`,`PmD_mtv_fail_result_4`,`PmD_mtv_fail_result_5`,`PmD_mtv_fail_result_6`)

PmD_matrix_keys <- c('PmD_val_all_cue_-3','PmD_val_all_cue_-2','PmD_val_all_cue_-1','PmD_val_all_cue_0','PmD_val_all_cue_1','PmD_val_all_cue_2','PmD_val_all_cue_3', 
                    'PmD_val_succ_result_-3','PmD_val_succ_result_-2','PmD_val_succ_result_-1','PmD_val_succ_result_0','PmD_val_succ_result_1','PmD_val_succ_result_2','PmD_val_succ_result_3',
                    'PmD_val_fail_result_-3','PmD_val_fail_result_-2','PmD_val_fail_result_-1','PmD_val_fail_result_0','PmD_val_fail_result_1','PmD_val_fail_result_2','PmD_val_fail_result_3',
                    'PmD_mtv_all_cue_0','PmD_mtv_all_cue_1','PmD_mtv_all_cue_2','PmD_mtv_all_cue_3','PmD_mtv_all_cue_4','PmD_mtv_all_cue_5','PmD_mtv_all_cue_6',
                    'PmD_mtv_succ_result_0','PmD_mtv_succ_result_1','PmD_mtv_succ_result_2','PmD_mtv_succ_result_3','PmD_mtv_succ_result_4','PmD_mtv_succ_result_5','PmD_mtv_succ_result_6',
                    'PmD_mtv_fail_result_0','PmD_mtv_fail_result_1','PmD_mtv_fail_result_2','PmD_mtv_fail_result_3','PmD_mtv_fail_result_4','PmD_mtv_fail_result_5','PmD_mtv_fail_result_6')

cat("\nM1 heatmaps")
for (i in 1:length(M1_matrix_keys)){
  if (any(is.na(M1_matrices[,,i]))){
    M1_matrices[,,i][is.na(M1_matrices[,,i])] = 0
    cat("\nna in",M1_matrix_keys[i])
  }
  png(paste(M1_matrix_keys[i],"_mv.png",sep=""),width=8,height=6,units="in",res=500)
  heatmap.2(M1_matrices[,,i],Colv=F,dendrogram="row",scale="row",col=rev(brewer.pal(11,"RdBu")),main=M1_matrix_keys[i],trace="none",cexRow=0.5,ylab="unit",xlab="time (s)",colsep=9)
  dev.off()
}

cat("\nS1 heatmaps")
for (i in 1:length(S1_matrix_keys)){
  if (any(is.na(S1_matrices[,,i]))){
    S1_matrices[,,i][is.na(S1_matrices[,,i])] = 0
    cat("\nna in",S1_matrix_keys[i])
  }
  png(paste(S1_matrix_keys[i],"_mv.png",sep=""),width=8,height=6,units="in",res=500)
  heatmap.2(S1_matrices[,,i],Colv=F,dendrogram="row",scale="row",col=rev(brewer.pal(11,"RdBu")),main=S1_matrix_keys[i],trace="none",cexRow=0.5,ylab="unit",xlab="time (s)",colsep=9)
  dev.off()
}

cat("\nPmD heatmaps")
for (i in 1:length(PmD_matrix_keys)){
  if (any(is.na(PmD_matrices[,,i]))){
    PmD_matrices[,,i][is.na(PmD_matrices[,,i])] = 0
    cat("\nna in",PmD_matrix_keys[i])
  }
  png(paste(PmD_matrix_keys[i],"_mv.png",sep=""),width=8,height=6,units="in",res=500)
  heatmap.2(PmD_matrices[,,i],Colv=F,dendrogram="row",scale="row",col=rev(brewer.pal(11,"RdBu")),main=PmD_matrix_keys[i],trace="none",cexRow=0.5,ylab="unit",xlab="time (s)",colsep=9)
  dev.off()
}

cat("\ngripforce plots")

gf_matrix_keys <- c('val_all_cue_-3','val_all_cue_-2','val_all_cue_-1','val_all_cue_0','val_all_cue_1','val_all_cue_2','val_all_cue_3', 
                     'val_succ_result_-3','val_succ_result_-2','val_succ_result_-1','val_succ_result_0','val_succ_result_1','val_succ_result_2','val_succ_result_3',
                     'val_fail_result_-3','val_fail_result_-2','val_fail_result_-1','val_fail_result_0','val_fail_result_1','val_fail_result_2','val_fail_result_3',
                     'mtv_all_cue_0','mtv_all_cue_1','mtv_all_cue_2','mtv_all_cue_3','mtv_all_cue_4','mtv_all_cue_5','mtv_all_cue_6',
                     'mtv_succ_result_0','mtv_succ_result_1','mtv_succ_result_2','mtv_succ_result_3','mtv_succ_result_4','mtv_succ_result_5','mtv_succ_result_6',
                     'mtv_fail_result_0','mtv_fail_result_1','mtv_fail_result_2','mtv_fail_result_3','mtv_fail_result_4','mtv_fail_result_5','mtv_fail_result_6')

for (i in 1:length(gf_matrix_keys)){
  avg_key <- paste(gf_matrix_keys[i],'_avg',sep="")
  std_key <- paste(gf_matrix_keys[i],'_std',sep="")
  
  gf <- gf_total[[avg_key]]
  std <- gf_total[[std_key]]
  
  plot_gf(gf,std,gf_matrix_keys[i])
  
}

cat("\nsaving")

save.image(file="rearranged_data_mv.RData")
rm(list=ls())

