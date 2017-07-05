library(openxlsx)
library(ggplot2)
library(reshape2)
#source("~/dropbox/mult_rp_files/r_test/multiplot.R")
library(zoo)
library(gplots)
library(RColorBrewer)
library(car)
library(FSA)

options(warn=0) #keep warnings 
#options(warn=-1) #suppress warnings (currently showing up b/ small CI's in boxplots)

load('rearranged_data.RData')

stat_compare <- function(i,region_key){

  #TODO remove rollmean??
  r0_succ_cue <- get(paste(region_key,"_r0_succ_cue_all",sep=""))[i,10:28]
  r1_succ_cue <- get(paste(region_key,"_r1_succ_cue_all",sep=""))[i,10:28]
  r2_succ_cue <- get(paste(region_key,"_r2_succ_cue_all",sep=""))[i,10:28]
  r3_succ_cue <- get(paste(region_key,"_r3_succ_cue_all",sep=""))[i,10:28]
  r0_fail_cue <- get(paste(region_key,"_r0_fail_cue_all",sep=""))[i,10:28]
  r1_fail_cue <- get(paste(region_key,"_r1_fail_cue_all",sep=""))[i,10:28]
  r2_fail_cue <- get(paste(region_key,"_r2_fail_cue_all",sep=""))[i,10:28]
  r3_fail_cue <- get(paste(region_key,"_r3_fail_cue_all",sep=""))[i,10:28]
  p0_succ_cue <- get(paste(region_key,"_p0_succ_cue_all",sep=""))[i,10:28]
  p1_succ_cue <- get(paste(region_key,"_p1_succ_cue_all",sep=""))[i,10:28]
  p2_succ_cue <- get(paste(region_key,"_p2_succ_cue_all",sep=""))[i,10:28]
  p3_succ_cue <- get(paste(region_key,"_p3_succ_cue_all",sep=""))[i,10:28]
  p0_fail_cue <- get(paste(region_key,"_p0_fail_cue_all",sep=""))[i,10:28]
  p1_fail_cue <- get(paste(region_key,"_p1_fail_cue_all",sep=""))[i,10:28]
  p2_fail_cue <- get(paste(region_key,"_p2_fail_cue_all",sep=""))[i,10:28]
  p3_fail_cue <- get(paste(region_key,"_p3_fail_cue_all",sep=""))[i,10:28]
  
  r0_succ_result <- get(paste(region_key,"_r0_succ_result_all",sep=""))[i,10:28]
  r1_succ_result <- get(paste(region_key,"_r1_succ_result_all",sep=""))[i,10:28]
  r2_succ_result <- get(paste(region_key,"_r2_succ_result_all",sep=""))[i,10:28]
  r3_succ_result <- get(paste(region_key,"_r3_succ_result_all",sep=""))[i,10:28]
  r0_fail_result <- get(paste(region_key,"_r0_fail_result_all",sep=""))[i,10:28]
  r1_fail_result <- get(paste(region_key,"_r1_fail_result_all",sep=""))[i,10:28]
  r2_fail_result <- get(paste(region_key,"_r2_fail_result_all",sep=""))[i,10:28]
  r3_fail_result <- get(paste(region_key,"_r3_fail_result_all",sep=""))[i,10:28]
  p0_succ_result <- get(paste(region_key,"_p0_succ_result_all",sep=""))[i,10:28]
  p1_succ_result <- get(paste(region_key,"_p1_succ_result_all",sep=""))[i,10:28]
  p2_succ_result <- get(paste(region_key,"_p2_succ_result_all",sep=""))[i,10:28]
  p3_succ_result <- get(paste(region_key,"_p3_succ_result_all",sep=""))[i,10:28]
  p0_fail_result <- get(paste(region_key,"_p0_fail_result_all",sep=""))[i,10:28]
  p1_fail_result <- get(paste(region_key,"_p1_fail_result_all",sep=""))[i,10:28]
  p2_fail_result <- get(paste(region_key,"_p2_fail_result_all",sep=""))[i,10:28]
  p3_fail_result <- get(paste(region_key,"_p3_fail_result_all",sep=""))[i,10:28]
  
  after_time = time[10:28]
  
  #run with a covariate of 0,1,2,3 and then again with 1,2,3 (same w/ neg), because 0 could be outlier or diff frame of ref
  #r_vals <- c(r0_,r1,r2,r3)
  vals_cue <- c(p3_succ_cue,p2_succ_cue,p1_succ_cue,p0_succ_cue,p3_fail_cue,p2_fail_cue,p1_fail_cue,p0_fail_cue,r0_succ_cue,r1_succ_cue,r2_succ_cue,r3_succ_cue,r0_fail_cue,r1_fail_cue,r2_fail_cue,r3_fail_cue)
  vals_result <- c(p3_succ_result,p2_succ_result,p1_succ_result,p0_succ_result,p3_fail_result,p2_fail_result,p1_fail_result,p0_fail_cue,r0_succ_result,r1_succ_result,r2_succ_result,r3_succ_cue,r0_fail_result,r1_fail_result,r2_fail_result,r3_fail_result)
  vals_result_succ <- c(p3_succ_result,p2_succ_result,p1_succ_result,p0_succ_result,r0_succ_result,r1_succ_result,r2_succ_result,r3_succ_cue)
  vals_result_fail <-c(p3_fail_result,p2_fail_result,p1_fail_result,p0_fail_cue,r0_fail_result,r1_fail_result,r2_fail_result,r3_fail_result)
  
  bins <- length(after_time)
  level <- as.factor(c(rep(-3,bins),rep(-2,bins),rep(-1,bins),rep(0,bins),rep(-3,bins),rep(-2,bins),rep(-1,bins),rep(0,bins),rep(0,bins),rep(1,bins),rep(2,bins),rep(3,bins),rep(0,bins),rep(1,bins),rep(2,bins),rep(3,bins)))
  sep_level <- as.factor(c(rep(-3,bins),rep(-2,bins),rep(-1,bins),rep(0,bins),rep(0,bins),rep(1,bins),rep(2,bins),rep(3,bins)))
  
  cue_frame <- data.frame(value=vals_cue,after_time,level)
  result_frame <- data.frame(value=vals_result,after_time,level)
  result_succ_frame <- data.frame(value=vals_result_succ,after_time,sep_level)
  result_fail_frame <- data.frame(value=vals_result_fail,after_time,sep_level)
  
  cue_ancova_model <- aov(cue_frame$value ~ cue_frame$level + level)
  cue_anova_model <- aov(cue_frame$value ~ cue_frame$level)
  result_ancova_model <- aov(result_frame$value ~ result_frame$level + level)
  result_anova_model <- aov(result_frame$value ~ result_frame$level)
  
  cue_ancova_output <- anova(cue_ancova_model)
  cue_anova_output <- anova(cue_anova_model)
  result_ancova_output <- anova(result_ancova_model)
  result_anova_output <- anova(result_anova_model)
  
  plot1 <- plotmeans(cue_frame$value ~ cue_frame$level)
  plot2 <- plotmeans(result_frame$value ~ result_frame$level)
  plot3 <- plotmeans(result_succ_frame$value ~ result_succ_frame$sep_level)
  plot4 <- plotmeans(result_fail_frame$value ~ result_fail_frame$sep_level)
  
  png("test.png")
  #png(paste(region_key,"_plotmeans_unit_",i,".png",sep=""),width=8,height=6,units="in",res=500)
  multiplot(plot1, plot2, plot3, plot4, cols=2)
  dev.off()
  
  cue_ancova_pval <- cue_ancova_output$'Pr(>f)'[1]
  cue_anova_pval <- cue_anova_output$'Pr(>f)'[1]
  result_ancova_pval <- result_ancova_output$'Pr(>f)'[1]
  result_anova_pval <- result_anova_output$'Pr(>f)'[1]
  
  cue_anocva_tukey <- TukeyHSD(cue_ancova_model)
  cue_anova_tukey <- TukeyHSD(cue_anova_model)
  result_ancova_tukey <- TukeyHSD(result_ancova_model)
  result_anova_tukey <- TukeyHSD(result_anova_model)
  
  cue_kw_model <- kruskal.test(cue_frame$value ~ cue_frame$level)
  result_kw_model <- kruskal.test(result_frame$value ~ result_frame$level)
  
  cue_kw_pval <- cue_kw_model$p.value
  result_kw_pval <- result_kw_model$p.value
  
  if (cue_kw_pval < 0.05 || result_kw_pval < 0.05){
    cue_dunn <- dunnTest(cue_frame$value ~ cue_frame$level)
    result_dunn <- dunnTest(result_frame$value ~ result_frame$level)
    
    cat("unit",i,"significant\n")
  }

}

region_list <- c('M1','S1','PmD')
for (j in 1:length(region_list)){
  region_key <- region_list[j]
  cat("\n",region_key,"\n",sep="")
  unit_num <- dim(get(paste(region_key,"_r0_succ_cue_all",sep="")))[1]
  
  for (unit in 1:unit_num){
    stat_compare(unit,region_key)
  }

}






#level <- as.factor(c(rep(0,length(r0)),rep(1,length(r1)),rep(2,length(r2)),rep(3,length(r3))))
#r_levels <- data.frame(value=c(r0,r1,r2,r3),after_time,level)
#levene_output <- leveneTest(r_vals,level)

#ancova_model <- aov(r_levels$value ~ r_levels$after_time + r_levels$level)
#anova_model <- aov(r_levels$value ~ r_levels$after_time)
#other_anova <- aov(r_levels$value ~ r_levels$level)
#plotmeans(r_levels$value~r_levels$level)
#anova_output <- anova(anova_model)
#p_val <- anova_output$'Pr(>f)'[1]



