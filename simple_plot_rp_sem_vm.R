library(openxlsx)
library(ggplot2)
library(reshape2)
#source("~/dropbox/mult_rp_files/r_test/multiplot.R")
#source("~/Dropbox/mult_rp_files/r_test/multiplot.R")
source("~/workspace/classification_scripts/multiplot.R")
#source("~/documents/lab/workspace/Classification_scripts/multiplot.R")
#source("~/workspace/classification_scripts/multiplot.R")
library(zoo)
library(gplots)
library(RColorBrewer)
library(abind)
library(gridGraphics)
library(grid)
library(gridExtra)
library(R.matlab)
library(egg)

saveAsPng <- T

region_list <- c('M1','S1','PmD')

#TODO make plot bool so can just run the summary stats on some blocks

#########

sem <- function(x){sd(x)/sqrt(length(x))}

fnlist <- function(x, fil){ z <- deparse(substitute(x))
  cat(z, "\n", file=fil)
  nams=names(x) 
  for (i in seq_along(x) ){ cat(nams[i], "\t",  x[[i]], "\n",file=fil, append=TRUE) }
}

##########

attach('block.RData')

M1_sig_unit_list <- c()
sig_info_cue <- list()
sig_info_res <- list()
for(name in names(M1_p_val_list)){
  #just 1 and 5, BC/AC and BR/AR
  temp_sig <- which(rowSums(cbind(M1_p_val_list[[name]][,1] < 0.05,M1_p_val_list[[name]][,5] < 0.05)) > 0)
  sig_info_cue[[name]] <- which(M1_p_val_list[[name]][,1] < 0.05) 
  sig_info_res[[name]] <- which(M1_p_val_list[[name]][,5] < 0.05) 
  M1_sig_unit_list <- append(M1_sig_unit_list,temp_sig)
}
if(exists("M1_sm_sig_p_r_levels_ac")){
  binary_bool <- FALSE
  t1 <- c(M1_sm_sig_p_r_levels_ac[,1],M1_sm_sig_p_p_levels_ac[,1],M1_sm_sig_p_r_outcome_levels_ac[,1],M1_sm_sig_p_p_outcome_levels_ac[,1],M1_sm_sig_p_outcome_levels_ac[,1],M1_sm_sig_p_comb_levels_ac[,1],M1_sig_p_r_delivery_levels_ac[,1],M1_sig_p_p_delivery_levels_ac[,1])
  t2 <- c(M1_sm_sig_p_r_levels_br[,1],M1_sm_sig_p_p_levels_br[,1],M1_sm_sig_p_r_outcome_levels_br[,1],M1_sm_sig_p_p_outcome_levels_br[,1],M1_sm_sig_p_outcome_levels_br[,1],M1_sm_sig_p_comb_levels_br[,1],M1_sig_p_r_delivery_levels_br[,1],M1_sig_p_p_delivery_levels_br[,1])
  t3 <- c(M1_sm_sig_p_r_levels_ar[,1],M1_sm_sig_p_p_levels_ar[,1],M1_sm_sig_p_r_outcome_levels_ar[,1],M1_sm_sig_p_p_outcome_levels_ar[,1],M1_sm_sig_p_outcome_levels_ar[,1],M1_sm_sig_p_comb_levels_ar[,1],M1_sig_p_r_delivery_levels_ar[,1],M1_sig_p_p_delivery_levels_ar[,1])

  list1 <- list('r_levels'=M1_sm_sig_p_r_levels_ac[,1],'p_levels'=M1_sm_sig_p_p_levels_ac[,1],'r_outcome'=M1_sm_sig_p_r_outcome_levels_ac[,1],'p_outcome'=M1_sm_sig_p_p_outcome_levels_ac[,1],'outcome'=M1_sm_sig_p_outcome_levels_ac[,1],'comb'=M1_sm_sig_p_comb_levels_ac[,1],'r_delivery'=M1_sig_p_r_delivery_levels_ac[,1],'p_delivery'=M1_sig_p_p_delivery_levels_ac[,1])
  list2 <- list('r_levels'=M1_sm_sig_p_r_levels_br[,1],'p_levels'=M1_sm_sig_p_p_levels_br[,1],'r_outcome'=M1_sm_sig_p_r_outcome_levels_br[,1],'p_outcome'=M1_sm_sig_p_p_outcome_levels_br[,1],'outcome'=M1_sm_sig_p_outcome_levels_br[,1],'comb'=M1_sm_sig_p_comb_levels_br[,1],'r_delivery'=M1_sig_p_r_delivery_levels_br[,1],'p_delivery'=M1_sig_p_p_delivery_levels_br[,1])
  list3 <- list('r_levels'=M1_sm_sig_p_r_levels_ar[,1],'p_levels'=M1_sm_sig_p_p_levels_ar[,1],'r_outcome'=M1_sm_sig_p_r_outcome_levels_ar[,1],'p_outcome'=M1_sm_sig_p_p_outcome_levels_ar[,1],'outcome'=M1_sm_sig_p_outcome_levels_ar[,1],'comb'=M1_sm_sig_p_comb_levels_ar[,1],'r_delivery'=M1_sig_p_r_delivery_levels_ar[,1],'p_delivery'=M1_sig_p_p_delivery_levels_ar[,1])
}else{
  binary_bool <- TRUE
  t1 <- c(M1_sm_sig_p_r_outcome_levels_ac[,1],M1_sm_sig_p_p_outcome_levels_ac[,1],M1_sm_sig_p_outcome_levels_ac[,1],M1_sm_sig_p_comb_levels_ac[,1],M1_sig_p_r_delivery_levels_ac[,1],M1_sig_p_p_delivery_levels_ac[,1])
  t2 <- c(M1_sm_sig_p_r_outcome_levels_br[,1],M1_sm_sig_p_p_outcome_levels_br[,1],M1_sm_sig_p_outcome_levels_br[,1],M1_sm_sig_p_comb_levels_br[,1],M1_sig_p_r_delivery_levels_br[,1],M1_sig_p_p_delivery_levels_br[,1])
  t3 <- c(M1_sm_sig_p_r_outcome_levels_ar[,1],M1_sm_sig_p_p_outcome_levels_ar[,1],M1_sm_sig_p_outcome_levels_ar[,1],M1_sm_sig_p_comb_levels_ar[,1],M1_sig_p_r_delivery_levels_ar[,1],M1_sig_p_p_delivery_levels_ar[,1])
  
  list1 <- list('r_outcome'=M1_sm_sig_p_r_outcome_levels_ac[,1],'p_outcome'=M1_sm_sig_p_p_outcome_levels_ac[,1],'outcome'=M1_sm_sig_p_outcome_levels_ac[,1],'comb'=M1_sm_sig_p_comb_levels_ac[,1],'r_delivery'=M1_sig_p_r_delivery_levels_ac[,1],'p_delivery'=M1_sig_p_p_delivery_levels_ac[,1])
  list2 <- list('r_outcome'=M1_sm_sig_p_r_outcome_levels_br[,1],'p_outcome'=M1_sm_sig_p_p_outcome_levels_br[,1],'outcome'=M1_sm_sig_p_outcome_levels_br[,1],'comb'=M1_sm_sig_p_comb_levels_br[,1],'r_delivery'=M1_sig_p_r_delivery_levels_br[,1],'p_delivery'=M1_sig_p_p_delivery_levels_br[,1])
  list3 <- list('r_outcome'=M1_sm_sig_p_r_outcome_levels_ar[,1],'p_outcome'=M1_sm_sig_p_p_outcome_levels_ar[,1],'outcome'=M1_sm_sig_p_outcome_levels_ar[,1],'comb'=M1_sm_sig_p_comb_levels_ar[,1],'r_delivery'=M1_sig_p_r_delivery_levels_ar[,1],'p_delivery'=M1_sig_p_p_delivery_levels_ar[,1])
  
  
}
  
M1_sig_unit_list <- c(M1_sig_unit_list,t1,t2,t3)
M1_sig_unit_list <- sort(unique(M1_sig_unit_list))

fnlist(sig_info_cue,"M1_bw_wind_cue.txt")
fnlist(sig_info_res,"M1_bw_wind_res.txt")
fnlist(list1,"M1_win_wind_ac.txt")
fnlist(list2,"M1_win_wind_br.txt")
fnlist(list3,"M1_win_wind_ar.txt")

#
S1_sig_unit_list <- c()
sig_info_cue <- list()
sig_info_res <- list()
for(name in names(S1_p_val_list)){
  #just 1 and 5, BC/AC and BR/AR
  temp_sig <- which(rowSums(cbind(S1_p_val_list[[name]][,1] < 0.05,S1_p_val_list[[name]][,5] < 0.05)) > 0)
  sig_info_cue[[name]] <- which(S1_p_val_list[[name]][,1] < 0.05) 
  sig_info_res[[name]] <- which(S1_p_val_list[[name]][,5] < 0.05) 
  S1_sig_unit_list <- append(S1_sig_unit_list,temp_sig)
}
if(!binary_bool){
  t1 <- c(S1_sm_sig_p_r_levels_ac[,1],S1_sm_sig_p_p_levels_ac[,1],S1_sm_sig_p_r_outcome_levels_ac[,1],S1_sm_sig_p_p_outcome_levels_ac[,1],S1_sm_sig_p_outcome_levels_ac[,1],S1_sm_sig_p_comb_levels_ac[,1],S1_sig_p_r_delivery_levels_ac[,1],S1_sig_p_p_delivery_levels_ac[,1])
  t2 <- c(S1_sm_sig_p_r_levels_br[,1],S1_sm_sig_p_p_levels_br[,1],S1_sm_sig_p_r_outcome_levels_br[,1],S1_sm_sig_p_p_outcome_levels_br[,1],S1_sm_sig_p_outcome_levels_br[,1],S1_sm_sig_p_comb_levels_br[,1],S1_sig_p_r_delivery_levels_br[,1],S1_sig_p_p_delivery_levels_br[,1])
  t3 <- c(S1_sm_sig_p_r_levels_ar[,1],S1_sm_sig_p_p_levels_ar[,1],S1_sm_sig_p_r_outcome_levels_ar[,1],S1_sm_sig_p_p_outcome_levels_ar[,1],S1_sm_sig_p_outcome_levels_ar[,1],S1_sm_sig_p_comb_levels_ar[,1],S1_sig_p_r_delivery_levels_ar[,1],S1_sig_p_p_delivery_levels_ar[,1])

  list1 <- list('r_levels'=S1_sm_sig_p_r_levels_ac[,1],'p_levels'=S1_sm_sig_p_p_levels_ac[,1],'r_outcome'=S1_sm_sig_p_r_outcome_levels_ac[,1],'p_outcome'=S1_sm_sig_p_p_outcome_levels_ac[,1],'outcome'=S1_sm_sig_p_outcome_levels_ac[,1],'comb'=S1_sm_sig_p_comb_levels_ac[,1],'r_delivery'=S1_sig_p_r_delivery_levels_ac[,1],'p_delivery'=S1_sig_p_p_delivery_levels_ac[,1])
  list2 <- list('r_levels'=S1_sm_sig_p_r_levels_br[,1],'p_levels'=S1_sm_sig_p_p_levels_br[,1],'r_outcome'=S1_sm_sig_p_r_outcome_levels_br[,1],'p_outcome'=S1_sm_sig_p_p_outcome_levels_br[,1],'outcome'=S1_sm_sig_p_outcome_levels_br[,1],'comb'=S1_sm_sig_p_comb_levels_br[,1],'r_delivery'=S1_sig_p_r_delivery_levels_br[,1],'p_delivery'=S1_sig_p_p_delivery_levels_br[,1])
  list3 <- list('r_levels'=S1_sm_sig_p_r_levels_ar[,1],'p_levels'=S1_sm_sig_p_p_levels_ar[,1],'r_outcome'=S1_sm_sig_p_r_outcome_levels_ar[,1],'p_outcome'=S1_sm_sig_p_p_outcome_levels_ar[,1],'outcome'=S1_sm_sig_p_outcome_levels_ar[,1],'comb'=S1_sm_sig_p_comb_levels_ar[,1],'r_delivery'=S1_sig_p_r_delivery_levels_ar[,1],'p_delivery'=S1_sig_p_p_delivery_levels_ar[,1])
}else{
  t1 <- c(S1_sm_sig_p_r_outcome_levels_ac[,1],S1_sm_sig_p_p_outcome_levels_ac[,1],S1_sm_sig_p_outcome_levels_ac[,1],S1_sm_sig_p_comb_levels_ac[,1],S1_sig_p_r_delivery_levels_ac[,1],S1_sig_p_p_delivery_levels_ac[,1])
  t2 <- c(S1_sm_sig_p_r_outcome_levels_br[,1],S1_sm_sig_p_p_outcome_levels_br[,1],S1_sm_sig_p_outcome_levels_br[,1],S1_sm_sig_p_comb_levels_br[,1],S1_sig_p_r_delivery_levels_br[,1],S1_sig_p_p_delivery_levels_br[,1])
  t3 <- c(S1_sm_sig_p_r_outcome_levels_ar[,1],S1_sm_sig_p_p_outcome_levels_ar[,1],S1_sm_sig_p_outcome_levels_ar[,1],S1_sm_sig_p_comb_levels_ar[,1],S1_sig_p_r_delivery_levels_ar[,1],S1_sig_p_p_delivery_levels_ar[,1])
  
  list1 <- list('r_outcome'=S1_sm_sig_p_r_outcome_levels_ac[,1],'p_outcome'=S1_sm_sig_p_p_outcome_levels_ac[,1],'outcome'=S1_sm_sig_p_outcome_levels_ac[,1],'comb'=S1_sm_sig_p_comb_levels_ac[,1],'r_delivery'=S1_sig_p_r_delivery_levels_ac[,1],'p_delivery'=S1_sig_p_p_delivery_levels_ac[,1])
  list2 <- list('r_outcome'=S1_sm_sig_p_r_outcome_levels_br[,1],'p_outcome'=S1_sm_sig_p_p_outcome_levels_br[,1],'outcome'=S1_sm_sig_p_outcome_levels_br[,1],'comb'=S1_sm_sig_p_comb_levels_br[,1],'r_delivery'=S1_sig_p_r_delivery_levels_br[,1],'p_delivery'=S1_sig_p_p_delivery_levels_br[,1])
  list3 <- list('r_outcome'=S1_sm_sig_p_r_outcome_levels_ar[,1],'p_outcome'=S1_sm_sig_p_p_outcome_levels_ar[,1],'outcome'=S1_sm_sig_p_outcome_levels_ar[,1],'comb'=S1_sm_sig_p_comb_levels_ar[,1],'r_delivery'=S1_sig_p_r_delivery_levels_ar[,1],'p_delivery'=S1_sig_p_p_delivery_levels_ar[,1])
}
  
S1_sig_unit_list <- c(S1_sig_unit_list,t1,t2,t3)
S1_sig_unit_list <- sort(unique(S1_sig_unit_list))

fnlist(sig_info_cue,"S1_bw_wind_cue.txt")
fnlist(sig_info_res,"S1_bw_wind_res.txt")
fnlist(list1,"S1_win_wind_ac.txt")
fnlist(list2,"S1_win_wind_br.txt")
fnlist(list3,"S1_win_wind_ar.txt")

#
PmD_sig_unit_list <- c()
sig_info_cue <- list()
sig_info_res <- list()
for(name in names(PmD_p_val_list)){
  #just 1 and 5, BC/AC and BR/AR
  temp_sig <- which(rowSums(cbind(PmD_p_val_list[[name]][,1] < 0.05,PmD_p_val_list[[name]][,5] < 0.05)) > 0)
  sig_info_cue[[name]] <- which(PmD_p_val_list[[name]][,1] < 0.05) 
  sig_info_res[[name]] <- which(PmD_p_val_list[[name]][,5] < 0.05) 
  PmD_sig_unit_list <- append(PmD_sig_unit_list,temp_sig)
}
if(!binary_bool){
  t1 <- c(PmD_sm_sig_p_r_levels_ac[,1],PmD_sm_sig_p_p_levels_ac[,1],PmD_sm_sig_p_r_outcome_levels_ac[,1],PmD_sm_sig_p_p_outcome_levels_ac[,1],PmD_sm_sig_p_outcome_levels_ac[,1],PmD_sm_sig_p_comb_levels_ac[,1],PmD_sig_p_r_delivery_levels_ac[,1],PmD_sig_p_p_delivery_levels_ac[,1])
  t2 <- c(PmD_sm_sig_p_r_levels_br[,1],PmD_sm_sig_p_p_levels_br[,1],PmD_sm_sig_p_r_outcome_levels_br[,1],PmD_sm_sig_p_p_outcome_levels_br[,1],PmD_sm_sig_p_outcome_levels_br[,1],PmD_sm_sig_p_comb_levels_br[,1],PmD_sig_p_r_delivery_levels_br[,1],PmD_sig_p_p_delivery_levels_br[,1])
  t3 <- c(PmD_sm_sig_p_r_levels_ar[,1],PmD_sm_sig_p_p_levels_ar[,1],PmD_sm_sig_p_r_outcome_levels_ar[,1],PmD_sm_sig_p_p_outcome_levels_ar[,1],PmD_sm_sig_p_outcome_levels_ar[,1],PmD_sm_sig_p_comb_levels_ar[,1],PmD_sig_p_r_delivery_levels_ar[,1],PmD_sig_p_p_delivery_levels_ar[,1])

  list1 <- list('r_levels'=PmD_sm_sig_p_r_levels_ac[,1],'p_levels'=PmD_sm_sig_p_p_levels_ac[,1],'r_outcome'=PmD_sm_sig_p_r_outcome_levels_ac[,1],'p_outcome'=PmD_sm_sig_p_p_outcome_levels_ac[,1],'outcome'=PmD_sm_sig_p_outcome_levels_ac[,1],'comb'=PmD_sm_sig_p_comb_levels_ac[,1],'r_delivery'=PmD_sig_p_r_delivery_levels_ac[,1],'p_delivery'=PmD_sig_p_p_delivery_levels_ac[,1])
  list2 <- list('r_levels'=PmD_sm_sig_p_r_levels_br[,1],'p_levels'=PmD_sm_sig_p_p_levels_br[,1],'r_outcome'=PmD_sm_sig_p_r_outcome_levels_br[,1],'p_outcome'=PmD_sm_sig_p_p_outcome_levels_br[,1],'outcome'=PmD_sm_sig_p_outcome_levels_br[,1],'comb'=PmD_sm_sig_p_comb_levels_br[,1],'r_delivery'=PmD_sig_p_r_delivery_levels_br[,1],'p_delivery'=PmD_sig_p_p_delivery_levels_br[,1])
  list3 <- list('r_levels'=PmD_sm_sig_p_r_levels_ar[,1],'p_levels'=PmD_sm_sig_p_p_levels_ar[,1],'r_outcome'=PmD_sm_sig_p_r_outcome_levels_ar[,1],'p_outcome'=PmD_sm_sig_p_p_outcome_levels_ar[,1],'outcome'=PmD_sm_sig_p_outcome_levels_ar[,1],'comb'=PmD_sm_sig_p_comb_levels_ar[,1],'r_delivery'=PmD_sig_p_r_delivery_levels_ar[,1],'p_delivery'=PmD_sig_p_p_delivery_levels_ar[,1])
}else{
  t1 <- c(PmD_sm_sig_p_r_outcome_levels_ac[,1],PmD_sm_sig_p_p_outcome_levels_ac[,1],PmD_sm_sig_p_outcome_levels_ac[,1],PmD_sm_sig_p_comb_levels_ac[,1],PmD_sig_p_r_delivery_levels_ac[,1],PmD_sig_p_p_delivery_levels_ac[,1])
  t2 <- c(PmD_sm_sig_p_r_outcome_levels_br[,1],PmD_sm_sig_p_p_outcome_levels_br[,1],PmD_sm_sig_p_outcome_levels_br[,1],PmD_sm_sig_p_comb_levels_br[,1],PmD_sig_p_r_delivery_levels_br[,1],PmD_sig_p_p_delivery_levels_br[,1])
  t3 <- c(PmD_sm_sig_p_r_outcome_levels_ar[,1],PmD_sm_sig_p_p_outcome_levels_ar[,1],PmD_sm_sig_p_outcome_levels_ar[,1],PmD_sm_sig_p_comb_levels_ar[,1],PmD_sig_p_r_delivery_levels_ar[,1],PmD_sig_p_p_delivery_levels_ar[,1])
  
  list1 <- list('r_outcome'=PmD_sm_sig_p_r_outcome_levels_ac[,1],'p_outcome'=PmD_sm_sig_p_p_outcome_levels_ac[,1],'outcome'=PmD_sm_sig_p_outcome_levels_ac[,1],'comb'=PmD_sm_sig_p_comb_levels_ac[,1],'r_delivery'=PmD_sig_p_r_delivery_levels_ac[,1],'p_delivery'=PmD_sig_p_p_delivery_levels_ac[,1])
  list2 <- list('r_outcome'=PmD_sm_sig_p_r_outcome_levels_br[,1],'p_outcome'=PmD_sm_sig_p_p_outcome_levels_br[,1],'outcome'=PmD_sm_sig_p_outcome_levels_br[,1],'comb'=PmD_sm_sig_p_comb_levels_br[,1],'r_delivery'=PmD_sig_p_r_delivery_levels_br[,1],'p_delivery'=PmD_sig_p_p_delivery_levels_br[,1])
  list3 <- list('r_outcome'=PmD_sm_sig_p_r_outcome_levels_ar[,1],'p_outcome'=PmD_sm_sig_p_p_outcome_levels_ar[,1],'outcome'=PmD_sm_sig_p_outcome_levels_ar[,1],'comb'=PmD_sm_sig_p_comb_levels_ar[,1],'r_delivery'=PmD_sig_p_r_delivery_levels_ar[,1],'p_delivery'=PmD_sig_p_p_delivery_levels_ar[,1])
  
  
}
  
PmD_sig_unit_list <- c(PmD_sig_unit_list,t1,t2,t3)
PmD_sig_unit_list <- sort(unique(PmD_sig_unit_list))

fnlist(sig_info_cue,"PmD_bw_wind_cue.txt")
fnlist(sig_info_res,"PmD_bw_wind_res.txt")
fnlist(list1,"PmD_win_wind_ac.txt")
fnlist(list2,"PmD_win_wind_br.txt")
fnlist(list3,"PmD_win_wind_ar.txt")



##########################################

for(region_index in 1:length(region_list)){
  cat("\nplotting region:",region_list[region_index],"\n")
  
  readin <- readMat(paste('simple_output_',region_list[region_index],'.mat',sep=""))
  
  all_cue_fr <- readin$return.dict[,,1]$all.cue.fr
  all_res_fr <- readin$return.dict[,,1]$all.res.fr
  condensed <- readin$return.dict[,,1]$condensed
  bin_size <- readin$return.dict[,,1]$params[,,1]$bin.size[,]
  rp_vals <- c(0,1,2,3)
  
  
  #TODO make for diff bfr and aft times
  #TODO unhardcode rollmean (time and rollmean val)
  old_time <- seq(from=-0.5,to=(1.0-bin_size/1000),by=bin_size/1000)
  time <- seq(from=-0.5+2*bin_size/1000,to=(1.0-3*bin_size/1000),by=bin_size/1000)
  
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
  
  r0_fail <- res0[which(res0 %in% r0)]
  r1_fail <- res0[which(res0 %in% r1)]
  r2_fail <- res0[which(res0 %in% r2)]
  r3_fail <- res0[which(res0 %in% r3)]
  r0_succ <- res1[which(res1 %in% r0)]
  r1_succ <- res1[which(res1 %in% r1)]
  r2_succ <- res1[which(res1 %in% r2)]
  r3_succ <- res1[which(res1 %in% r3)]
  
  #
  rx <- which(condensed[,4] >= 1)
  px <- which(condensed[,5] >= 1)
  
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
  
  #if(region_index == 1){unit_list <- M1_sig_unit_list}else if (region_index == 2){unit_list <- S1_sig_unit_list}else{unit_list <- PmD_sig_unit_list}
  
  #set here if want to just plot certain units by number
  #0059
  if(region_index == 1){unit_list <- c(26,45)}else if (region_index == 2){unit_list <- c(62,69)}else{unit_list <- c(78,36)}
  #504
  #if(region_index == 1){unit_list <- c(65,47)}else if (region_index == 2){unit_list <- c(27,44)}else{unit_list <- c(7,51)}
  
  
  for(unit_num in unit_list){
    cat(unit_num,'\n')

    # 
    # #value  (initial colortest3: value = scale_color_brewer YlOrRd)
    # png(paste("colortest3_sem_",region_list[region_index],"_v_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    # 
    # v_cue_avgs <- data.frame(time=time,v_3=rollmean(colMeans(all_cue_fr[unit_num,v_3,]),5),v_2=rollmean(colMeans(all_cue_fr[unit_num,v_2,]),5),v_1=rollmean(colMeans(all_cue_fr[unit_num,v_1,]),5),v0=rollmean(colMeans(all_cue_fr[unit_num,v0,]),5),v1=rollmean(colMeans(all_cue_fr[unit_num,v1,]),5),v2=rollmean(colMeans(all_cue_fr[unit_num,v2,]),5),v3=rollmean(colMeans(all_cue_fr[unit_num,v3,]),5))
    # v_res_avgs <- data.frame(time=time,v_3=rollmean(colMeans(all_res_fr[unit_num,v_3,]),5),v_2=rollmean(colMeans(all_res_fr[unit_num,v_2,]),5),v_1=rollmean(colMeans(all_res_fr[unit_num,v_1,]),5),v0=rollmean(colMeans(all_res_fr[unit_num,v0,]),5),v1=rollmean(colMeans(all_res_fr[unit_num,v1,]),5),v2=rollmean(colMeans(all_res_fr[unit_num,v2,]),5),v3=rollmean(colMeans(all_res_fr[unit_num,v3,]),5))
    # 
    # v_cue_avgs.m <- melt(v_cue_avgs,id.vars="time",variable="v_level")
    # plt_cue <- ggplot(v_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=v_level),size=1) + theme_classic()
    # plt_cue <- plt_cue +labs(title="Cue",y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0) + scale_color_brewer(palette="YlOrRd")
    # #plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    # plt_cue <- plt_cue + theme(axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    # 
    # v_res_avgs.m <- melt(v_res_avgs,id.vars="time",variable="v_level")
    # plt_res <- ggplot(v_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=v_level),size=1) + theme_classic()
    # plt_res <- plt_res +labs(title="Result",y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0) + scale_color_brewer(palette="YlOrRd")
    # plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    # 
    # ggarrange(plt_cue,plt_res,ncol=1)
    # graphics.off()
    # 
    # 
    # #value  
    # png(paste("colortest4_sem_",region_list[region_index],"_v_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    # 
    # v_cue_avgs <- data.frame(time=time,v_3=rollmean(colMeans(all_cue_fr[unit_num,v_3,]),5),v_2=rollmean(colMeans(all_cue_fr[unit_num,v_2,]),5),v_1=rollmean(colMeans(all_cue_fr[unit_num,v_1,]),5),v0=rollmean(colMeans(all_cue_fr[unit_num,v0,]),5),v1=rollmean(colMeans(all_cue_fr[unit_num,v1,]),5),v2=rollmean(colMeans(all_cue_fr[unit_num,v2,]),5),v3=rollmean(colMeans(all_cue_fr[unit_num,v3,]),5))
    # v_res_avgs <- data.frame(time=time,v_3=rollmean(colMeans(all_res_fr[unit_num,v_3,]),5),v_2=rollmean(colMeans(all_res_fr[unit_num,v_2,]),5),v_1=rollmean(colMeans(all_res_fr[unit_num,v_1,]),5),v0=rollmean(colMeans(all_res_fr[unit_num,v0,]),5),v1=rollmean(colMeans(all_res_fr[unit_num,v1,]),5),v2=rollmean(colMeans(all_res_fr[unit_num,v2,]),5),v3=rollmean(colMeans(all_res_fr[unit_num,v3,]),5))
    # 
    # v_cue_avgs.m <- melt(v_cue_avgs,id.vars="time",variable="v_level")
    # plt_cue <- ggplot(v_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=v_level),size=1) + theme_classic()
    # plt_cue <- plt_cue +labs(title="Cue",y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0) + scale_color_manual(values=c("salmon4","salmon3","lightsalmon","khaki","aquamarine2","aquamarine4","darkgreen"))    #
    # #plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    # plt_cue <- plt_cue + theme(axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    # 
    # v_res_avgs.m <- melt(v_res_avgs,id.vars="time",variable="v_level")
    # plt_res <- ggplot(v_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=v_level),size=1) + theme_classic()
    # plt_res <- plt_res +labs(title="Result",y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0) + scale_color_manual(values=c("salmon4","salmon3","lightsalmon","khaki","aquamarine2","aquamarine4","darkgreen"))    #scale_color_brewer(palette="GnBu") +
    # plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    # 
    # ggarrange(plt_cue,plt_res,ncol=1)
    # graphics.off()
    # 
    # # motivation
    # png(paste("colortest4_sem_",region_list[region_index],"_m_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    # 
    # m_cue_avgs <- data.frame(time=time,m0=rollmean(colMeans(all_cue_fr[unit_num,m0,]),5),m1=rollmean(colMeans(all_cue_fr[unit_num,m1,]),5),m2=rollmean(colMeans(all_cue_fr[unit_num,m2,]),5),m3=rollmean(colMeans(all_cue_fr[unit_num,m3,]),5),m4=rollmean(colMeans(all_cue_fr[unit_num,m4,]),5),m5=rollmean(colMeans(all_cue_fr[unit_num,m5,]),5),m6=rollmean(colMeans(all_cue_fr[unit_num,m6,]),5))
    # m_res_avgs <- data.frame(time=time,m0=rollmean(colMeans(all_res_fr[unit_num,m0,]),5),m1=rollmean(colMeans(all_res_fr[unit_num,m1,]),5),m2=rollmean(colMeans(all_res_fr[unit_num,m2,]),5),m3=rollmean(colMeans(all_res_fr[unit_num,m3,]),5),m4=rollmean(colMeans(all_res_fr[unit_num,m4,]),5),m5=rollmean(colMeans(all_res_fr[unit_num,m5,]),5),m6=rollmean(colMeans(all_res_fr[unit_num,m6,]),5))
    # 
    # m_cue_avgs.m <- melt(m_cue_avgs,id.vars="time",variable="m_level")
    # plt_cue <- ggplot(m_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=m_level),size=2) + theme_classic()
    # plt_cue <- plt_cue + labs(title="Cue") + geom_vline(xintercept=0) + scale_color_manual(values=c("thistle2","thistle3","slategray2","slategray3","slateblue1","slateblue3","slateblue4"))    #scale_color_brewer(palette="GnBu") +
    # plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    # 
    # m_res_avgs.m <- melt(m_res_avgs,id.vars="time",variable="m_level")
    # plt_res <- ggplot(m_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=m_level),size=2) + theme_classic()
    # plt_res <- plt_res + labs(title="Result") + geom_vline(xintercept=0) + scale_color_manual(values=c("thistle2","thistle3","slategray2","slategray3","slateblue1","slateblue3","slateblue4"))    #scale_color_brewer(palette="GnBu") +
    # #plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    # plt_res <- plt_res + theme(axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    # 
    # ggarrange(plt_cue,plt_res,ncol=1)
    # graphics.off()

    ##############
    #value
    png(paste("paper_nl_sem_1_",region_list[region_index],"_v_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)

    v_cue_avgs <- data.frame(time=time,v_3=rollmean(colMeans(all_cue_fr[unit_num,v_3,]),5),v_2=rollmean(colMeans(all_cue_fr[unit_num,v_2,]),5),v_1=rollmean(colMeans(all_cue_fr[unit_num,v_1,]),5),v0=rollmean(colMeans(all_cue_fr[unit_num,v0,]),5),v1=rollmean(colMeans(all_cue_fr[unit_num,v1,]),5),v2=rollmean(colMeans(all_cue_fr[unit_num,v2,]),5),v3=rollmean(colMeans(all_cue_fr[unit_num,v3,]),5))
    v_res_avgs <- data.frame(time=time,v_3=rollmean(colMeans(all_res_fr[unit_num,v_3,]),5),v_2=rollmean(colMeans(all_res_fr[unit_num,v_2,]),5),v_1=rollmean(colMeans(all_res_fr[unit_num,v_1,]),5),v0=rollmean(colMeans(all_res_fr[unit_num,v0,]),5),v1=rollmean(colMeans(all_res_fr[unit_num,v1,]),5),v2=rollmean(colMeans(all_res_fr[unit_num,v2,]),5),v3=rollmean(colMeans(all_res_fr[unit_num,v3,]),5))

    v_cue_avgs.m <- melt(v_cue_avgs,id.vars="time",variable="v_level")
    plt_cue <- ggplot(v_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=v_level),size=1) + theme_classic()
    plt_cue <- plt_cue +labs(title="Cue",y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0) + scale_color_manual(values=c(rgb(254,227,145,maxColorValue = 255),rgb(254,196,79,maxColorValue = 255),rgb(254,153,41,maxColorValue = 255),rgb(236,112,20,maxColorValue = 255),rgb(204,76,2,maxColorValue = 255),rgb(153,52,4,maxColorValue = 255),rgb(102,37,6,maxColorValue = 255)))
    plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))

    v_res_avgs.m <- melt(v_res_avgs,id.vars="time",variable="v_level")
    plt_res <- ggplot(v_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=v_level),size=1) + theme_classic()
    plt_res <- plt_res +labs(title="Result",y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0) + scale_color_manual(values=c(rgb(254,227,145,maxColorValue = 255),rgb(254,196,79,maxColorValue = 255),rgb(254,153,41,maxColorValue = 255),rgb(236,112,20,maxColorValue = 255),rgb(204,76,2,maxColorValue = 255),rgb(153,52,4,maxColorValue = 255),rgb(102,37,6,maxColorValue = 255)))
    plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))

    ggarrange(plt_cue,plt_res,ncol=1)
    graphics.off()

    # motivation (final)
    png(paste("paper_nl_sem_1_",region_list[region_index],"_m_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)

    m_cue_avgs <- data.frame(time=time,m0=rollmean(colMeans(all_cue_fr[unit_num,m0,]),5),m1=rollmean(colMeans(all_cue_fr[unit_num,m1,]),5),m2=rollmean(colMeans(all_cue_fr[unit_num,m2,]),5),m3=rollmean(colMeans(all_cue_fr[unit_num,m3,]),5),m4=rollmean(colMeans(all_cue_fr[unit_num,m4,]),5),m5=rollmean(colMeans(all_cue_fr[unit_num,m5,]),5),m6=rollmean(colMeans(all_cue_fr[unit_num,m6,]),5))
    m_res_avgs <- data.frame(time=time,m0=rollmean(colMeans(all_res_fr[unit_num,m0,]),5),m1=rollmean(colMeans(all_res_fr[unit_num,m1,]),5),m2=rollmean(colMeans(all_res_fr[unit_num,m2,]),5),m3=rollmean(colMeans(all_res_fr[unit_num,m3,]),5),m4=rollmean(colMeans(all_res_fr[unit_num,m4,]),5),m5=rollmean(colMeans(all_res_fr[unit_num,m5,]),5),m6=rollmean(colMeans(all_res_fr[unit_num,m6,]),5))

    m_cue_avgs.m <- melt(m_cue_avgs,id.vars="time",variable="m_level")
    plt_cue <- ggplot(m_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=m_level),size=1) + theme_classic()
    #plt_cue <- plt_cue + labs(title="Cue") + geom_vline(xintercept=0) + scale_color_manual(values=c(rgb(0,0,1),rgb(0,0.5,0),rgb(1,0,0),rgb(0,0.75,0.75),rgb(0.75,0,0.75),rgb(0.75,0.75,0),rgb(0.25,0.25,0.25)))    #matlab (old default colors)
    plt_cue <- plt_cue + labs(title="Cue") + geom_vline(xintercept=0) + scale_color_manual(values=c(rgb(160,160,160,maxColorValue = 255),rgb(51,255,255,maxColorValue = 255),rgb(51,153,255,maxColorValue = 255),rgb(51,51,255,maxColorValue = 255),rgb(153,51,255,maxColorValue = 255),rgb(255,51,255,maxColorValue = 255),rgb(255,51,153,maxColorValue = 255)))
    plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))

    m_res_avgs.m <- melt(m_res_avgs,id.vars="time",variable="m_level")
    plt_res <- ggplot(m_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=m_level),size=1) + theme_classic()
    #plt_res <- plt_res + labs(title="Result") + geom_vline(xintercept=0) + scale_color_manual(values=c(rgb(0,0.447,0.741),rgb(0.85,0.325,0.098),rgb(0.929,0.694,0.125),rgb(0.494,0.184,0.556),rgb(0.466,0.674,0.188),rgb(0.301,0.745,0.933),rgb(0.635,0.078,0.184)))    #new default colors
    plt_res <- plt_res + labs(title="Result") + geom_vline(xintercept=0) + scale_color_manual(values=c(rgb(160,160,160,maxColorValue = 255),rgb(51,255,255,maxColorValue = 255),rgb(51,153,255,maxColorValue = 255),rgb(51,51,255,maxColorValue = 255),rgb(153,51,255,maxColorValue = 255),rgb(255,51,255,maxColorValue = 255),rgb(255,51,153,maxColorValue = 255)))
    plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))

    ggarrange(plt_cue,plt_res,ncol=1)
    graphics.off()
    
  }
}


# save.image(file="rearranged_data.RData")
#rm(list=ls())

