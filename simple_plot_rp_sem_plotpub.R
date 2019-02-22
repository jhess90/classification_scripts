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
  
  #set here if want to just plot certain units by number (M1, S1, PMd)
  #if(region_index == 1){unit_list <- c(7,14,50,49,8,60)}else if (region_index == 2){unit_list <- c(35,73,49,38,7)}else{unit_list <- c(107,62,80)} #Soph cued block 3
  #if(region_index == 1){unit_list <- c(8,65,68,47,76)}else if (region_index == 2){unit_list <- c(19,25,13,21,5)}else{unit_list <- c(44,17,49,7,15,20)} #PG cued block 3 
  #if(region_index == 1){unit_list <- c(17,67,68,69)}else if (region_index == 2){unit_list <- c(89,66,42,81,31)}else{unit_list <- c(49,93,91,5,79,93)} #soph uncued block 1
  if(region_index == 1){unit_list <- c(35,25,20,114)}else if (region_index == 2){unit_list <- c(78)}else{unit_list <- c(72,68,65,75,52,6,67,28)} #PG uncued block 1
  
  for(unit_num in unit_list){
    cat(unit_num,'\n')
    
    #can change this and rest to binary_bool now that it's set
    if (!((length(r1) == 0 | length(r2) == 0) | length(r3) == 0)){
      ## reward
      png(paste("sem_",region_list[region_index],"_r_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
      
      r_cue_avgs <- data.frame(time=time,r0=rollmean(colMeans(all_cue_fr[unit_num,r0,]),5),r1=rollmean(colMeans(all_cue_fr[unit_num,r1,]),5),r2=rollmean(colMeans(all_cue_fr[unit_num,r2,]),5),r3=rollmean(colMeans(all_cue_fr[unit_num,r3,]),5))
      r_res_avgs <- data.frame(time=time,r0=rollmean(colMeans(all_res_fr[unit_num,r0,]),5),r1=rollmean(colMeans(all_res_fr[unit_num,r1,]),5),r2=rollmean(colMeans(all_res_fr[unit_num,r2,]),5),r3=rollmean(colMeans(all_res_fr[unit_num,r3,]),5))
      
      r_cue_sems <-  data.frame(time=time,r0=rollmean(sapply(data.frame(all_cue_fr[unit_num,r0,]),function(x)sd(x)/sqrt(length(x))),5),r1=rollmean(sapply(data.frame(all_cue_fr[unit_num,r1,]),function(x)sd(x)/sqrt(length(x))),5),r2=rollmean(sapply(data.frame(all_cue_fr[unit_num,r2,]),function(x)sd(x)/sqrt(length(x))),5),r3=rollmean(sapply(data.frame(all_cue_fr[unit_num,r3,]),function(x)sd(x)/sqrt(length(x))),5))
      r_res_sems <-  data.frame(time=time,r0=rollmean(sapply(data.frame(all_res_fr[unit_num,r0,]),function(x)sd(x)/sqrt(length(x))),5),r1=rollmean(sapply(data.frame(all_res_fr[unit_num,r1,]),function(x)sd(x)/sqrt(length(x))),5),r2=rollmean(sapply(data.frame(all_res_fr[unit_num,r2,]),function(x)sd(x)/sqrt(length(x))),5),r3=rollmean(sapply(data.frame(all_res_fr[unit_num,r3,]),function(x)sd(x)/sqrt(length(x))),5))
      
      r_cue_avgs.m <- melt(r_cue_avgs,id.vars="time",variable="r_level")
      
      plt_cue <- ggplot(r_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=r_level),size=1) + theme_classic()
      plt_cue <- plt_cue + scale_colour_manual(values=c("goldenrod","palegreen3","seagreen","darkgreen")) + labs(title="Cue") + geom_vline(xintercept=0)
      plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
      
      #
      plt_cue <- plt_cue + geom_ribbon(data=r_cue_avgs,aes(x=time,ymin=r_cue_avgs$r0-r_cue_sems$r0,ymax=r_cue_avgs$r0+r_cue_sems$r0),inherit.aes=FALSE,alpha=0.2,fill="goldenrod")
      plt_cue <- plt_cue + geom_ribbon(data=r_cue_avgs,aes(x=time,ymin=r_cue_avgs$r1-r_cue_sems$r1,ymax=r_cue_avgs$r1+r_cue_sems$r1),inherit.aes=FALSE,alpha=0.2,fill="palegreen3")
      plt_cue <- plt_cue + geom_ribbon(data=r_cue_avgs,aes(x=time,ymin=r_cue_avgs$r2-r_cue_sems$r2,ymax=r_cue_avgs$r2+r_cue_sems$r2),inherit.aes=FALSE,alpha=0.2,fill="seagreen")
      plt_cue <- plt_cue + geom_ribbon(data=r_cue_avgs,aes(x=time,ymin=r_cue_avgs$r3-r_cue_sems$r3,ymax=r_cue_avgs$r3+r_cue_sems$r3),inherit.aes=FALSE,alpha=0.2,fill="darkgreen")
      
      r_res_avgs.m <- melt(r_res_avgs,id.vars="time",variable="r_level")
      plt_res <- ggplot(r_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=r_level),size=1) + theme_classic()
      plt_res <- plt_res + scale_colour_manual(values=c("goldenrod","palegreen3","seagreen","darkgreen")) + labs(title="Result",y="z-score", x="Time(s)",colour="Reward Level") + geom_vline(xintercept=0)
      plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
      
      #
      plt_res <- plt_res + geom_ribbon(data=r_res_avgs,aes(x=time,ymin=r_res_avgs$r0-r_res_sems$r0,ymax=r_res_avgs$r0+r_res_sems$r0),inherit.aes=FALSE,alpha=0.2,fill="goldenrod")
      plt_res <- plt_res + geom_ribbon(data=r_res_avgs,aes(x=time,ymin=r_res_avgs$r1-r_res_sems$r1,ymax=r_res_avgs$r1+r_res_sems$r1),inherit.aes=FALSE,alpha=0.2,fill="palegreen3")
      plt_res <- plt_res + geom_ribbon(data=r_res_avgs,aes(x=time,ymin=r_res_avgs$r2-r_res_sems$r2,ymax=r_res_avgs$r2+r_res_sems$r2),inherit.aes=FALSE,alpha=0.2,fill="seagreen")
      plt_res <- plt_res + geom_ribbon(data=r_res_avgs,aes(x=time,ymin=r_res_avgs$r3-r_res_sems$r3,ymax=r_res_avgs$r3+r_res_sems$r3),inherit.aes=FALSE,alpha=0.2,fill="darkgreen")
      
      
      multiplot(plt_cue,plt_res)
      graphics.off()
      
      ## punishment
      png(paste("sem_",region_list[region_index],"_p_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
      
      p_cue_avgs <- data.frame(time=time,p0=rollmean(colMeans(all_cue_fr[unit_num,p0,]),5),p1=rollmean(colMeans(all_cue_fr[unit_num,p1,]),5),p2=rollmean(colMeans(all_cue_fr[unit_num,p2,]),5),p3=rollmean(colMeans(all_cue_fr[unit_num,p3,]),5))
      p_res_avgs <- data.frame(time=time,p0=rollmean(colMeans(all_res_fr[unit_num,p0,]),5),p1=rollmean(colMeans(all_res_fr[unit_num,p1,]),5),p2=rollmean(colMeans(all_res_fr[unit_num,p2,]),5),p3=rollmean(colMeans(all_res_fr[unit_num,p3,]),5))
      
      #
      p_cue_sems <-  data.frame(time=time,p0=rollmean(sapply(data.frame(all_cue_fr[unit_num,p0,]),function(x)sd(x)/sqrt(length(x))),5),p1=rollmean(sapply(data.frame(all_cue_fr[unit_num,p1,]),function(x)sd(x)/sqrt(length(x))),5),p2=rollmean(sapply(data.frame(all_cue_fr[unit_num,p2,]),function(x)sd(x)/sqrt(length(x))),5),p3=rollmean(sapply(data.frame(all_cue_fr[unit_num,p3,]),function(x)sd(x)/sqrt(length(x))),5))
      p_res_sems <-  data.frame(time=time,p0=rollmean(sapply(data.frame(all_res_fr[unit_num,p0,]),function(x)sd(x)/sqrt(length(x))),5),p1=rollmean(sapply(data.frame(all_res_fr[unit_num,p1,]),function(x)sd(x)/sqrt(length(x))),5),p2=rollmean(sapply(data.frame(all_res_fr[unit_num,p2,]),function(x)sd(x)/sqrt(length(x))),5),p3=rollmean(sapply(data.frame(all_res_fr[unit_num,p3,]),function(x)sd(x)/sqrt(length(x))),5))
      
      p_cue_avgs.m <- melt(p_cue_avgs,id.vars="time",variable="p_level")
      plt_cue <- ggplot(p_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=p_level),size=1) + theme_classic()
      plt_cue <- plt_cue + scale_colour_manual(values=c("goldenrod","coral","firebrick1","red2")) + labs(title="Cue") + geom_vline(xintercept=0)
      plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
      
      #
      plt_cue <- plt_cue + geom_ribbon(data=p_cue_avgs,aes(x=time,ymin=p_cue_avgs$p0-p_cue_sems$p0,ymax=p_cue_avgs$p0+p_cue_sems$p0),inherit.aes=FALSE,alpha=0.2,fill="goldenrod")
      plt_cue <- plt_cue + geom_ribbon(data=p_cue_avgs,aes(x=time,ymin=p_cue_avgs$p1-p_cue_sems$p1,ymax=p_cue_avgs$p1+p_cue_sems$p1),inherit.aes=FALSE,alpha=0.2,fill="coral")
      plt_cue <- plt_cue + geom_ribbon(data=p_cue_avgs,aes(x=time,ymin=p_cue_avgs$p2-p_cue_sems$p2,ymax=p_cue_avgs$p2+p_cue_sems$p2),inherit.aes=FALSE,alpha=0.2,fill="firebrick1")
      plt_cue <- plt_cue + geom_ribbon(data=p_cue_avgs,aes(x=time,ymin=p_cue_avgs$p3-p_cue_sems$p3,ymax=p_cue_avgs$p3+p_cue_sems$p3),inherit.aes=FALSE,alpha=0.2,fill="red2")
      
      p_res_avgs.m <- melt(p_res_avgs,id.vars="time",variable="p_level")
      plt_res <- ggplot(p_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=p_level),size=1) + theme_classic()
      plt_res <- plt_res + scale_colour_manual(values=c("goldenrod","coral","firebrick1","red2")) + labs(title="Result",y="z-score", x="Time(s)",colour="Punishment Level") + geom_vline(xintercept=0)
      plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
      
      #
      plt_res <- plt_res + geom_ribbon(data=p_res_avgs,aes(x=time,ymin=p_res_avgs$p0-p_res_sems$p0,ymax=p_res_avgs$p0+p_res_sems$p0),inherit.aes=FALSE,alpha=0.2,fill="goldenrod")
      plt_res <- plt_res + geom_ribbon(data=p_res_avgs,aes(x=time,ymin=p_res_avgs$p1-p_res_sems$p1,ymax=p_res_avgs$p1+p_res_sems$p1),inherit.aes=FALSE,alpha=0.2,fill="coral")
      plt_res <- plt_res + geom_ribbon(data=p_res_avgs,aes(x=time,ymin=p_res_avgs$p2-p_res_sems$p2,ymax=p_res_avgs$p2+p_res_sems$p2),inherit.aes=FALSE,alpha=0.2,fill="firebrick1")
      plt_res <- plt_res + geom_ribbon(data=p_res_avgs,aes(x=time,ymin=p_res_avgs$p3-p_res_sems$p3,ymax=p_res_avgs$p3+p_res_sems$p3),inherit.aes=FALSE,alpha=0.2,fill="red2")
      
      multiplot(plt_cue,plt_res)
      graphics.off()
    }
    ############
    png(paste("sem_",region_list[region_index],"_r_bin_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    r_cue_avgs <- data.frame(time=time,r0=rollmean(colMeans(all_cue_fr[unit_num,r0,]),5),rx=rollmean(colMeans(all_cue_fr[unit_num,rx,]),5))
    r_res_avgs <- data.frame(time=time,r0=rollmean(colMeans(all_res_fr[unit_num,r0,]),5),rx=rollmean(colMeans(all_res_fr[unit_num,rx,]),5))
    
    r_cue_sems <-  data.frame(time=time,r0=rollmean(sapply(data.frame(all_cue_fr[unit_num,r0,]),function(x)sd(x)/sqrt(length(x))),5),rx=rollmean(sapply(data.frame(all_cue_fr[unit_num,rx,]),function(x)sd(x)/sqrt(length(x))),5))
    r_res_sems <-  data.frame(time=time,r0=rollmean(sapply(data.frame(all_res_fr[unit_num,r0,]),function(x)sd(x)/sqrt(length(x))),5),rx=rollmean(sapply(data.frame(all_res_fr[unit_num,rx,]),function(x)sd(x)/sqrt(length(x))),5))
    
    r_cue_avgs.m <- melt(r_cue_avgs,id.vars="time",variable="r_level")
    
    plt_cue <- ggplot(r_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=r_level),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("goldenrod","darkgreen")) + labs(title="Cue") + geom_vline(xintercept=0)
    
    #
    plt_cue <- plt_cue + geom_ribbon(data=r_cue_avgs,aes(x=time,ymin=r_cue_avgs$r0-r_cue_sems$r0,ymax=r_cue_avgs$r0+r_cue_sems$r0),inherit.aes=FALSE,alpha=0.2,fill="goldenrod")
    plt_cue <- plt_cue + geom_ribbon(data=r_cue_avgs,aes(x=time,ymin=r_cue_avgs$rx-r_cue_sems$rx,ymax=r_cue_avgs$rx+r_cue_sems$rx),inherit.aes=FALSE,alpha=0.2,fill="darkgreen")
    plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    
    r_res_avgs.m <- melt(r_res_avgs,id.vars="time",variable="r_level")
    plt_res <- ggplot(r_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=r_level),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("goldenrod","darkgreen")) + labs(title="Result",y="z-score", x="Time(s)",colour="Reward Level") + geom_vline(xintercept=0)
    
    #
    plt_res <- plt_res + geom_ribbon(data=r_res_avgs,aes(x=time,ymin=r_res_avgs$r0-r_res_sems$r0,ymax=r_res_avgs$r0+r_res_sems$r0),inherit.aes=FALSE,alpha=0.2,fill="goldenrod")
    plt_res <- plt_res + geom_ribbon(data=r_res_avgs,aes(x=time,ymin=r_res_avgs$rx-r_res_sems$rx,ymax=r_res_avgs$rx+r_res_sems$rx),inherit.aes=FALSE,alpha=0.2,fill="darkgreen")
    plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    ## punishment
    png(paste("sem_",region_list[region_index],"_p_bin_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    p_cue_avgs <- data.frame(time=time,p0=rollmean(colMeans(all_cue_fr[unit_num,p0,]),5),px=rollmean(colMeans(all_cue_fr[unit_num,px,]),5))
    p_res_avgs <- data.frame(time=time,p0=rollmean(colMeans(all_res_fr[unit_num,p0,]),5),px=rollmean(colMeans(all_res_fr[unit_num,px,]),5))
    
    #
    p_cue_sems <-  data.frame(time=time,p0=rollmean(sapply(data.frame(all_cue_fr[unit_num,p0,]),function(x)sd(x)/sqrt(length(x))),5),px=rollmean(sapply(data.frame(all_cue_fr[unit_num,px,]),function(x)sd(x)/sqrt(length(x))),5))
    p_res_sems <-  data.frame(time=time,p0=rollmean(sapply(data.frame(all_res_fr[unit_num,p0,]),function(x)sd(x)/sqrt(length(x))),5),px=rollmean(sapply(data.frame(all_res_fr[unit_num,px,]),function(x)sd(x)/sqrt(length(x))),5))
    
    p_cue_avgs.m <- melt(p_cue_avgs,id.vars="time",variable="p_level")
    plt_cue <- ggplot(p_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=p_level),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("goldenrod","red2")) + labs(title="Cue") + geom_vline(xintercept=0)
    
    #
    plt_cue <- plt_cue + geom_ribbon(data=p_cue_avgs,aes(x=time,ymin=p_cue_avgs$p0-p_cue_sems$p0,ymax=p_cue_avgs$p0+p_cue_sems$p0),inherit.aes=FALSE,alpha=0.2,fill="goldenrod")
    plt_cue <- plt_cue + geom_ribbon(data=p_cue_avgs,aes(x=time,ymin=p_cue_avgs$px-p_cue_sems$px,ymax=p_cue_avgs$px+p_cue_sems$px),inherit.aes=FALSE,alpha=0.2,fill="red2")
    plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    
    p_res_avgs.m <- melt(p_res_avgs,id.vars="time",variable="p_level")
    plt_res <- ggplot(p_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=p_level),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("goldenrod","coral","firebrick1","red2")) + labs(title="Result",y="z-score", x="Time(s)",colour="Punishment Level") + geom_vline(xintercept=0)
    
    #
    plt_res <- plt_res + geom_ribbon(data=p_res_avgs,aes(x=time,ymin=p_res_avgs$p0-p_res_sems$p0,ymax=p_res_avgs$p0+p_res_sems$p0),inherit.aes=FALSE,alpha=0.2,fill="goldenrod")
    plt_res <- plt_res + geom_ribbon(data=p_res_avgs,aes(x=time,ymin=p_res_avgs$px-p_res_sems$px,ymax=p_res_avgs$px+p_res_sems$px),inherit.aes=FALSE,alpha=0.2,fill="red2")
    plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    ## value
    #png(paste("sem_",region_list[region_index],"_v_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    #v_cue_avgs <- data.frame(time=time,v_3=rollmean(colMeans(all_cue_fr[unit_num,v_3,]),5),v_2=rollmean(colMeans(all_cue_fr[unit_num,v_2,]),5),v_1=rollmean(colMeans(all_cue_fr[unit_num,v_1,]),5),v0=rollmean(colMeans(all_cue_fr[unit_num,v0,]),5),v1=rollmean(colMeans(all_cue_fr[unit_num,v1,]),5),v2=rollmean(colMeans(all_cue_fr[unit_num,v2,]),5),v3=rollmean(colMeans(all_cue_fr[unit_num,v3,]),5))
    #v_res_avgs <- data.frame(time=time,v_3=rollmean(colMeans(all_res_fr[unit_num,v_3,]),5),v_2=rollmean(colMeans(all_res_fr[unit_num,v_2,]),5),v_1=rollmean(colMeans(all_res_fr[unit_num,v_1,]),5),v0=rollmean(colMeans(all_res_fr[unit_num,v0,]),5),v1=rollmean(colMeans(all_res_fr[unit_num,v1,]),5),v2=rollmean(colMeans(all_res_fr[unit_num,v2,]),5),v3=rollmean(colMeans(all_res_fr[unit_num,v3,]),5))
    
    #v_cue_avgs.m <- melt(v_cue_avgs,id.vars="time",variable="v_level")
    #plt_cue <- ggplot(v_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=v_level),size=1) + theme_classic()
    #plt_cue <- plt_cue + scale_color_brewer(palette="BrBG") +labs(title=paste("Value: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0)
    
    #v_res_avgs.m <- melt(v_res_avgs,id.vars="time",variable="v_level")
    #plt_res <- ggplot(v_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=v_level),size=1) + theme_classic()
    #plt_res <- plt_res + scale_color_brewer(palette="BrBG") +labs(title="Result",y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0)
    
    #multiplot(plt_cue,plt_res)
    #graphics.off()
    
    ## motivation
    #png(paste("sem_",region_list[region_index],"_m_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    #m_cue_avgs <- data.frame(time=time,m0=rollmean(colMeans(all_cue_fr[unit_num,m0,]),5),m1=rollmean(colMeans(all_cue_fr[unit_num,m1,]),5),m2=rollmean(colMeans(all_cue_fr[unit_num,m2,]),5),m3=rollmean(colMeans(all_cue_fr[unit_num,m3,]),5),m4=rollmean(colMeans(all_cue_fr[unit_num,m4,]),5),m5=rollmean(colMeans(all_cue_fr[unit_num,m5,]),5),m6=rollmean(colMeans(all_cue_fr[unit_num,m6,]),5))
    #m_res_avgs <- data.frame(time=time,m0=rollmean(colMeans(all_res_fr[unit_num,m0,]),5),m1=rollmean(colMeans(all_res_fr[unit_num,m1,]),5),m2=rollmean(colMeans(all_res_fr[unit_num,m2,]),5),m3=rollmean(colMeans(all_res_fr[unit_num,m3,]),5),m4=rollmean(colMeans(all_res_fr[unit_num,m4,]),5),m5=rollmean(colMeans(all_res_fr[unit_num,m5,]),5),m6=rollmean(colMeans(all_res_fr[unit_num,m6,]),5))
    
    #m_cue_avgs.m <- melt(m_cue_avgs,id.vars="time",variable="m_level")
    #plt_cue <- ggplot(m_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=m_level),size=1) + theme_classic()
    #plt_cue <- plt_cue + scale_color_brewer(palette="GnBu") +labs(title=paste("Motivation: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0)
    
    #m_res_avgs.m <- melt(m_res_avgs,id.vars="time",variable="m_level")
    #plt_res <- ggplot(m_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=m_level),size=1) + theme_classic()
    #plt_res <- plt_res + scale_color_brewer(palette="GnBu") +labs(title="Result",y="z-score", x="Time(s)",colour="Value Level") + geom_vline(xintercept=0)
    
    #multiplot(plt_cue,plt_res)
    #graphics.off()
    
    ## result
    png(paste("sem_",region_list[region_index],"_res_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    res_cue_avgs <- data.frame(time=time,fail=rollmean(colMeans(all_cue_fr[unit_num,res0,]),5),succ=rollmean(colMeans(all_cue_fr[unit_num,res1,]),5))
    res_res_avgs <- data.frame(time=time,fail=rollmean(colMeans(all_res_fr[unit_num,res0,]),5),succ=rollmean(colMeans(all_res_fr[unit_num,res1,]),5))
    
    res_cue_sems <-  data.frame(time=time,fail=rollmean(sapply(data.frame(all_cue_fr[unit_num,res0,]),function(x)sd(x)/sqrt(length(x))),5),succ=rollmean(sapply(data.frame(all_cue_fr[unit_num,res1,]),function(x)sd(x)/sqrt(length(x))),5))
    res_res_sems <-  data.frame(time=time,fail=rollmean(sapply(data.frame(all_res_fr[unit_num,res0,]),function(x)sd(x)/sqrt(length(x))),5),succ=rollmean(sapply(data.frame(all_res_fr[unit_num,res1,]),function(x)sd(x)/sqrt(length(x))),5))
    
    res_cue_avgs.m <- melt(res_cue_avgs,id.vars="time",variable="res_level")
    plt_cue <- ggplot(res_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=res_level),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("skyblue4","purple")) + labs(title="Cue") +  geom_vline(xintercept=0)
    
    plt_cue <- plt_cue + geom_ribbon(data=res_cue_avgs,aes(x=time,ymin=res_cue_avgs$fail-res_cue_sems$fail,ymax=res_cue_avgs$fail+res_cue_sems$fail),inherit.aes=FALSE,alpha=0.2,fill="skyblue4")
    plt_cue <- plt_cue + geom_ribbon(data=res_cue_avgs,aes(x=time,ymin=res_cue_avgs$succ-res_cue_sems$succ,ymax=res_cue_avgs$succ+res_cue_sems$succ),inherit.aes=FALSE,alpha=0.2,fill="purple")
    plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    
    res_res_avgs.m <- melt(res_res_avgs,id.vars="time",variable="res_level")
    plt_res <- ggplot(res_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=res_level),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("skyblue4","purple")) + labs(title="Result",y="z-score", x="Time(s)",colour="Result") + geom_vline(xintercept=0)
    
    plt_res <- plt_res + geom_ribbon(data=res_res_avgs,aes(x=time,ymin=res_res_avgs$fail-res_res_sems$fail,ymax=res_res_avgs$fail+res_res_sems$fail),inherit.aes=FALSE,alpha=0.2,fill="skyblue4")
    plt_res <- plt_res + geom_ribbon(data=res_res_avgs,aes(x=time,ymin=res_res_avgs$succ-res_res_sems$succ,ymax=res_res_avgs$succ+res_res_sems$succ),inherit.aes=FALSE,alpha=0.2,fill="purple")
    plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    
    ## comb
    png(paste("sem_",region_list[region_index],"_comb_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    comb_cue_avgs <- data.frame(time=time,r0_p0=rollmean(colMeans(all_cue_fr[unit_num,r0_p0,]),5),rx_p0=rollmean(colMeans(all_cue_fr[unit_num,rx_p0,]),5),r0_px=rollmean(colMeans(all_cue_fr[unit_num,r0_px,]),5),rx_px=rollmean(colMeans(all_cue_fr[unit_num,rx_px,]),5))
    comb_res_avgs <- data.frame(time=time,r0_p0=rollmean(colMeans(all_res_fr[unit_num,r0_p0,]),5),rx_p0=rollmean(colMeans(all_res_fr[unit_num,rx_p0,]),5),r0_px=rollmean(colMeans(all_res_fr[unit_num,r0_px,]),5),rx_px=rollmean(colMeans(all_res_fr[unit_num,rx_px,]),5))
    comb_cue_sems <-  data.frame(time=time,r0_p0=rollmean(sapply(data.frame(all_cue_fr[unit_num,r0_p0,]),function(x)sd(x)/sqrt(length(x))),5),rx_p0=rollmean(sapply(data.frame(all_cue_fr[unit_num,rx_p0,]),function(x)sd(x)/sqrt(length(x))),5),r0_px=rollmean(sapply(data.frame(all_cue_fr[unit_num,r0_px,]),function(x)sd(x)/sqrt(length(x))),5),rx_px=rollmean(sapply(data.frame(all_cue_fr[unit_num,rx_px,]),function(x)sd(x)/sqrt(length(x))),5))
    comb_res_sems <-  data.frame(time=time,r0_p0=rollmean(sapply(data.frame(all_res_fr[unit_num,r0_p0,]),function(x)sd(x)/sqrt(length(x))),5),rx_p0=rollmean(sapply(data.frame(all_res_fr[unit_num,rx_p0,]),function(x)sd(x)/sqrt(length(x))),5),r0_px=rollmean(sapply(data.frame(all_res_fr[unit_num,r0_px,]),function(x)sd(x)/sqrt(length(x))),5),rx_px=rollmean(sapply(data.frame(all_res_fr[unit_num,rx_px,]),function(x)sd(x)/sqrt(length(x))),5))
    
    comb_cue_avgs.m <- melt(comb_cue_avgs,id.vars="time",variable="comb")
    plt_cue <- ggplot(comb_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("black","forestgreen","firebrick","mediumblue")) + labs(title="Cue") + geom_vline(xintercept=0)
    
    #
    plt_cue <- plt_cue + geom_ribbon(data=comb_cue_avgs,aes(x=time,ymin=comb_cue_avgs$r0_p0-comb_cue_sems$r0_p0,ymax=comb_cue_avgs$r0_p0+comb_cue_sems$r0_p0),inherit.aes=FALSE,alpha=0.2,fill="black")
    plt_cue <- plt_cue + geom_ribbon(data=comb_cue_avgs,aes(x=time,ymin=comb_cue_avgs$rx_p0-comb_cue_sems$rx_p0,ymax=comb_cue_avgs$rx_p0+comb_cue_sems$rx_p0),inherit.aes=FALSE,alpha=0.2,fill="forestgreen")
    plt_cue <- plt_cue + geom_ribbon(data=comb_cue_avgs,aes(x=time,ymin=comb_cue_avgs$r0_px-comb_cue_sems$r0_px,ymax=comb_cue_avgs$r0_px+comb_cue_sems$r0_px),inherit.aes=FALSE,alpha=0.2,fill="firebrick")
    plt_cue <- plt_cue + geom_ribbon(data=comb_cue_avgs,aes(x=time,ymin=comb_cue_avgs$rx_px-comb_cue_sems$rx_px,ymax=comb_cue_avgs$rx_px+comb_cue_sems$rx_px),inherit.aes=FALSE,alpha=0.2,fill="mediumblue")
    plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    
    
    comb_res_avgs.m <- melt(comb_res_avgs,id.vars="time",variable="comb")
    plt_res <- ggplot(comb_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("black","forestgreen","firebrick","mediumblue")) + labs(title="Result",y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0)
    
    #
    plt_res <- plt_res + geom_ribbon(data=comb_res_avgs,aes(x=time,ymin=comb_res_avgs$r0_p0-comb_res_sems$r0_p0,ymax=comb_res_avgs$r0_p0+comb_res_sems$r0_p0),inherit.aes=FALSE,alpha=0.2,fill="black")
    plt_res <- plt_res + geom_ribbon(data=comb_res_avgs,aes(x=time,ymin=comb_res_avgs$rx_p0-comb_res_sems$rx_p0,ymax=comb_res_avgs$rx_p0+comb_res_sems$rx_p0),inherit.aes=FALSE,alpha=0.2,fill="forestgreen")
    plt_res <- plt_res + geom_ribbon(data=comb_res_avgs,aes(x=time,ymin=comb_res_avgs$r0_px-comb_res_sems$r0_px,ymax=comb_res_avgs$r0_px+comb_res_sems$r0_px),inherit.aes=FALSE,alpha=0.2,fill="firebrick")
    plt_res <- plt_res + geom_ribbon(data=comb_res_avgs,aes(x=time,ymin=comb_res_avgs$rx_px-comb_res_sems$rx_px,ymax=comb_res_avgs$rx_px+comb_res_sems$rx_px),inherit.aes=FALSE,alpha=0.2,fill="mediumblue")
    plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    
    ## comb res
    #png(paste(region_list[region_index],"_comb_res_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    #if (length(r0_p0_f) == 1){r0_p0_f_fr <- rollmean(all_cue_fr[unit_num,r0_p0_f,],5)}else if(length(r0_p0_f) == 0){r0_p0_f_fr <- rep(0,146)}else{r0_p0_f_fr <- rollmean(colMeans(all_cue_fr[unit_num,r0_p0_f,]),5)}
    #if (length(r0_px_f) == 1){r0_px_f_fr <- rollmean(all_cue_fr[unit_num,r0_px_f,],5)}else if(length(r0_px_f) == 0){r0_px_f_fr <- rep(0,146)}else{r0_px_f_fr <- rollmean(colMeans(all_cue_fr[unit_num,r0_px_f,]),5)}
    #if (length(rx_p0_f) == 1){rx_p0_f_fr <- rollmean(all_cue_fr[unit_num,rx_p0_f,],5)}else if(length(rx_p0_f) == 0){rx_p0_f_fr <- rep(0,146)}else{rx_p0_f_fr <- rollmean(colMeans(all_cue_fr[unit_num,rx_p0_f,]),5)}
    #if (length(rx_px_f) == 1){rx_px_f_fr <- rollmean(all_cue_fr[unit_num,rx_px_f,],5)}else if(length(rx_px_f) == 0){rx_px_f_fr <- rep(0,146)}else{rx_px_f_fr <- rollmean(colMeans(all_cue_fr[unit_num,rx_px_f,]),5)}
    
    #if (length(r0_p0_s) == 1){r0_p0_s_fr <- rollmean(all_cue_fr[unit_num,r0_p0_s,],5)}else if(length(r0_p0_s) == 0){r0_p0_s_fr <- rep(0,146)}else{r0_p0_s_fr <- rollmean(colMeans(all_cue_fr[unit_num,r0_p0_s,]),5)}
    #if (length(r0_px_s) == 1){r0_px_s_fr <- rollmean(all_cue_fr[unit_num,r0_px_s,],5)}else if(length(r0_px_s) == 0){r0_px_s_fr <- rep(0,146)}else{r0_px_s_fr <- rollmean(colMeans(all_cue_fr[unit_num,r0_px_s,]),5)}
    #if (length(rx_p0_s) == 1){rx_p0_s_fr <- rollmean(all_cue_fr[unit_num,rx_p0_s,],5)}else if(length(rx_p0_s) == 0){rx_p0_s_fr <- rep(0,146)}else{rx_p0_s_fr <- rollmean(colMeans(all_cue_fr[unit_num,rx_p0_s,]),5)}
    #if (length(rx_px_s) == 1){rx_px_s_fr <- rollmean(all_cue_fr[unit_num,rx_px_s,],5)}else if(length(rx_px_s) == 0){rx_px_s_fr <- rep(0,146)}else{rx_px_s_fr <- rollmean(colMeans(all_cue_fr[unit_num,rx_px_s,]),5)}
    
    #comb_cue_avgs <- data.frame(time=time,r0_p0_s=r0_p0_s_fr,rx_p0_s=rx_p0_s_fr,r0_px_s=r0_px_s_fr,rx_px_s=rx_px_s_fr,r0_p0_f=r0_p0_f_fr,rx_p0_f=rx_p0_f_fr,r0_px_f=r0_px_f_fr,rx_px_f=rx_px_f_fr)
    
    #if (length(r0_px_f) == 1){r0_px_f_fr <- rollmean(all_res_fr[unit_num,r0_px_f,],5)}else if(length(r0_px_f) == 0){r0_px_f_fr <- rep(0,146)}else{r0_px_f_fr <- rollmean(colMeans(all_res_fr[unit_num,r0_px_f,]),5)}
    #if (length(rx_p0_f) == 1){rx_p0_f_fr <- rollmean(all_res_fr[unit_num,rx_p0_f,],5)}else if(length(rx_p0_f) == 0){rx_p0_f_fr <- rep(0,146)}else{rx_p0_f_fr <- rollmean(colMeans(all_res_fr[unit_num,rx_p0_f,]),5)}
    #if (length(rx_px_f) == 1){rx_px_f_fr <- rollmean(all_res_fr[unit_num,rx_px_f,],5)}else if(length(rx_px_f) == 0){rx_px_f_fr <- rep(0,146)}else{rx_px_f_fr <- rollmean(colMeans(all_res_fr[unit_num,rx_px_f,]),5)}
    
    #if (length(r0_p0_s) == 1){r0_p0_s_fr <- rollmean(all_res_fr[unit_num,r0_p0_s,],5)}else if(length(r0_p0_s) == 0){r0_p0_s_fr <- rep(0,146)}else{r0_p0_s_fr <- rollmean(colMeans(all_res_fr[unit_num,r0_p0_s,]),5)}
    #if (length(r0_px_s) == 1){r0_px_s_fr <- rollmean(all_res_fr[unit_num,r0_px_s,],5)}else if(length(r0_px_s) == 0){r0_px_s_fr <- rep(0,146)}else{r0_px_s_fr <- rollmean(colMeans(all_res_fr[unit_num,r0_px_s,]),5)}
    #if (length(rx_p0_s) == 1){rx_p0_s_fr <- rollmean(all_res_fr[unit_num,rx_p0_s,],5)}else if(length(rx_p0_s) == 0){rx_p0_s_fr <- rep(0,146)}else{rx_p0_s_fr <- rollmean(colMeans(all_res_fr[unit_num,rx_p0_s,]),5)}
    #if (length(rx_px_s) == 1){rx_px_s_fr <- rollmean(all_res_fr[unit_num,rx_px_s,],5)}else if(length(rx_px_s) == 0){rx_px_s_fr <- rep(0,146)}else{rx_px_s_fr <- rollmean(colMeans(all_res_fr[unit_num,rx_px_s,]),5)}
    
    #comb_res_avgs <- data.frame(time=time,r0_p0_s=r0_p0_s_fr,rx_p0_s=rx_p0_s_fr,r0_px_s=r0_px_s_fr,rx_px_s=rx_px_s_fr,r0_p0_f=r0_p0_f_fr,rx_p0_f=rx_p0_f_fr,r0_px_f=r0_px_f_fr,rx_px_f=rx_px_f_fr)
    
    #comb_cue_avgs.m <- melt(comb_cue_avgs,id.vars="time",variable="comb")
    #plt_cue <- ggplot(comb_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=0.5) + theme_classic()
    #plt_cue <- plt_cue + scale_colour_manual(values=c("black","forestgreen","firebrick","mediumblue","gray69","darkseagreen2","lightpink2","darkslategray1")) + labs(title=paste("Combination: Unit",unit_num,"\nCue"),y="z-score", x="Time(s)",colour="Combination") #+ geom_vline(xintercept=0) +  scale_linetype_manual(values=c("solid","solid","solid","solid","dashed","dashed","dashed","dashed"))
    
    
    #comb_res_avgs.m <- melt(comb_res_avgs,id.vars="time",variable="comb")
    #plt_res <- ggplot(comb_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=0.5) + theme_classic()
    #plt_res <- plt_res + scale_colour_manual(values=c("black","forestgreen","firebrick","mediumblue","gray69","darkseagreen2","lightpink2","darkslategray1")) + labs(title="Result",y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0)
    
    #multiplot(plt_cue,plt_res)
    #graphics.off()
    
    ## reward sf
    png(paste("sem_",region_list[region_index],"_r_sf_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    if (length(r0_s) == 1){r0_s_fr_cue <- rollmean(all_cue_fr[unit_num,r0_s,],5)}else if(length(r0_s) == 0){r0_s_fr_cue <- rep(0,146)}else{r0_s_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,r0_s,]),5)}
    if (length(r0_f) == 1){r0_f_fr_cue <- rollmean(all_cue_fr[unit_num,r0_f,],5)}else if(length(r0_f) == 0){r0_f_fr_cue <- rep(0,146)}else{r0_f_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,r0_f,]),5)}
    if (length(rx_s) == 1){rx_s_fr_cue <- rollmean(all_cue_fr[unit_num,rx_s,],5)}else if(length(rx_s) == 0){rx_s_fr_cue <- rep(0,146)}else{rx_s_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,rx_s,]),5)}
    if (length(rx_f) == 1){rx_f_fr_cue <- rollmean(all_cue_fr[unit_num,rx_f,],5)}else if(length(rx_f) == 0){rx_f_fr_cue <- rep(0,146)}else{rx_f_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,rx_f,]),5)}
    
    if (length(r0_s) == 1){r0_s_fr_res <- rollmean(all_res_fr[unit_num,r0_s,],5)}else if(length(r0_s) == 0){r0_s_fr_res <- rep(0,146)}else{r0_s_fr_res <- rollmean(colMeans(all_res_fr[unit_num,r0_s,]),5)}
    if (length(r0_f) == 1){r0_f_fr_res <- rollmean(all_res_fr[unit_num,r0_f,],5)}else if(length(r0_f) == 0){r0_f_fr_res <- rep(0,146)}else{r0_f_fr_res <- rollmean(colMeans(all_res_fr[unit_num,r0_f,]),5)}
    if (length(rx_s) == 1){rx_s_fr_res <- rollmean(all_res_fr[unit_num,rx_s,],5)}else if(length(rx_s) == 0){rx_s_fr_res <- rep(0,146)}else{rx_s_fr_res <- rollmean(colMeans(all_res_fr[unit_num,rx_s,]),5)}
    if (length(rx_f) == 1){rx_f_fr_res <- rollmean(all_res_fr[unit_num,rx_f,],5)}else if(length(rx_f) == 0){rx_f_fr_res <- rep(0,146)}else{rx_f_fr_res <- rollmean(colMeans(all_res_fr[unit_num,rx_f,]),5)}
    
    r_sf_cue_avgs <- data.frame(time=time,r0_s_cue=r0_s_fr_cue,r0_f_cue=r0_f_fr_cue,rx_s_cue=rx_s_fr_cue,rx_f_cue=rx_f_fr_cue)
    r_sf_res_avgs <- data.frame(time=time,r0_s_res=r0_s_fr_res,r0_f_res=r0_f_fr_res,rx_s_res=rx_s_fr_res,rx_f_res=rx_f_fr_res)
    
    if (length(r0_s) <= 1){r0_s_fr_cue <- rep(0,146)}else{r0_s_fr_cue <- rollmean(sapply(data.frame(all_cue_fr[unit_num,r0_s,]),function(x)sd(x)/sqrt(length(x))),5)}
    if (length(r0_f) <= 1){r0_f_fr_cue <- rep(0,146)}else{r0_f_fr_cue <- rollmean(sapply(data.frame(all_cue_fr[unit_num,r0_f,]),function(x)sd(x)/sqrt(length(x))),5)}
    if (length(rx_s) <= 1){rx_s_fr_cue <- rep(0,146)}else{rx_s_fr_cue <- rollmean(sapply(data.frame(all_cue_fr[unit_num,rx_s,]),function(x)sd(x)/sqrt(length(x))),5)}
    if (length(rx_f) <= 1){rx_f_fr_cue <- rep(0,146)}else{rx_f_fr_cue <- rollmean(sapply(data.frame(all_cue_fr[unit_num,rx_f,]),function(x)sd(x)/sqrt(length(x))),5)}
    
    if (length(r0_s) <= 1){r0_s_fr_res <- rep(0,146)}else{r0_s_fr_res <- rollmean(sapply(data.frame(all_cue_fr[unit_num,r0_s,]),function(x)sd(x)/sqrt(length(x))),5)}
    if (length(r0_f) <= 1){r0_f_fr_res <- rep(0,146)}else{r0_f_fr_res <- rollmean(sapply(data.frame(all_cue_fr[unit_num,r0_f,]),function(x)sd(x)/sqrt(length(x))),5)}
    if (length(rx_s) <= 1){rx_s_fr_res <- rep(0,146)}else{rx_s_fr_res <- rollmean(sapply(data.frame(all_cue_fr[unit_num,rx_s,]),function(x)sd(x)/sqrt(length(x))),5)}
    if (length(rx_f) <= 1){rx_f_fr_res <- rep(0,146)}else{rx_f_fr_res <- rollmean(sapply(data.frame(all_cue_fr[unit_num,rx_f,]),function(x)sd(x)/sqrt(length(x))),5)}

    r_sf_cue_sems <- data.frame(time=time,r0_s_cue=r0_s_fr_cue,r0_f_cue=r0_f_fr_cue,rx_s_cue=rx_s_fr_cue,rx_f_cue=rx_f_fr_cue)
    r_sf_res_sems <- data.frame(time=time,r0_s_res=r0_s_fr_res,r0_f_res=r0_f_fr_res,rx_s_res=rx_s_fr_res,rx_f_res=rx_f_fr_res)
    

    r_sf_cue_avgs.m <- melt(r_sf_cue_avgs,id.vars="time",variable="comb")
    plt_cue <- ggplot(r_sf_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("black","gray69","forestgreen","darkseagreen2")) +labs(title="Cue") + geom_vline(xintercept=0)
    
    plt_cue <- plt_cue + geom_ribbon(data=r_sf_cue_avgs,aes(x=time,ymin=r_sf_cue_avgs$r0_s-r_sf_cue_sems$r0_s,ymax=r_sf_cue_avgs$r0_s+r_sf_cue_sems$r0_s),inherit.aes=FALSE,alpha=0.2,fill="black")
    plt_cue <- plt_cue + geom_ribbon(data=r_sf_cue_avgs,aes(x=time,ymin=r_sf_cue_avgs$r0_f-r_sf_cue_sems$r0_f,ymax=r_sf_cue_avgs$r0_f+r_sf_cue_sems$r0_f),inherit.aes=FALSE,alpha=0.2,fill="gray69")
    plt_cue <- plt_cue + geom_ribbon(data=r_sf_cue_avgs,aes(x=time,ymin=r_sf_cue_avgs$rx_s-r_sf_cue_sems$rx_s,ymax=r_sf_cue_avgs$rx_s+r_sf_cue_sems$rx_s),inherit.aes=FALSE,alpha=0.2,fill="forestgreen")
    plt_cue <- plt_cue + geom_ribbon(data=r_sf_cue_avgs,aes(x=time,ymin=r_sf_cue_avgs$rx_f-r_sf_cue_sems$rx_f,ymax=r_sf_cue_avgs$rx_f+r_sf_cue_sems$rx_f),inherit.aes=FALSE,alpha=0.2,fill="darkseagreen2")
    plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    
    
    r_sf_res_avgs.m <- melt(r_sf_res_avgs,id.vars="time",variable="comb")
    plt_res <- ggplot(r_sf_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("black","gray69","forestgreen","darkseagreen2")) + labs(title="Result",y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0)
    
    plt_res <- plt_res + geom_ribbon(data=r_sf_res_avgs,aes(x=time,ymin=r_sf_res_avgs$r0_s-r_sf_res_sems$r0_s,ymax=r_sf_res_avgs$r0_s+r_sf_res_sems$r0_s),inherit.aes=FALSE,alpha=0.2,fill="black")
    plt_res <- plt_res + geom_ribbon(data=r_sf_res_avgs,aes(x=time,ymin=r_sf_res_avgs$r0_f-r_sf_res_sems$r0_f,ymax=r_sf_res_avgs$r0_f+r_sf_res_sems$r0_f),inherit.aes=FALSE,alpha=0.2,fill="gray69")
    plt_res <- plt_res + geom_ribbon(data=r_sf_res_avgs,aes(x=time,ymin=r_sf_res_avgs$rx_s-r_sf_res_sems$rx_s,ymax=r_sf_res_avgs$rx_s+r_sf_res_sems$rx_s),inherit.aes=FALSE,alpha=0.2,fill="forestgreen")
    plt_res <- plt_res + geom_ribbon(data=r_sf_res_avgs,aes(x=time,ymin=r_sf_res_avgs$rx_f-r_sf_res_sems$rx_f,ymax=r_sf_res_avgs$rx_f+r_sf_res_sems$rx_f),inherit.aes=FALSE,alpha=0.2,fill="darkseagreen2")
    plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
    ## punishment sf
    png(paste("sem_",region_list[region_index],"_p_sf_unit_",unit_num,".png",sep=""),width=8,height=6,units="in",res=500)
    
    if (length(p0_s) == 1){p0_s_fr_cue <- rollmean(all_cue_fr[unit_num,p0_s,],5)}else if(length(p0_s) == 0){p0_s_fr_cue <- rep(0,146)}else{p0_s_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,p0_s,]),5)}
    if (length(p0_f) == 1){p0_f_fr_cue <- rollmean(all_cue_fr[unit_num,p0_f,],5)}else if(length(p0_f) == 0){p0_f_fr_cue <- rep(0,146)}else{p0_f_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,p0_f,]),5)}
    if (length(px_s) == 1){px_s_fr_cue <- rollmean(all_cue_fr[unit_num,px_s,],5)}else if(length(px_s) == 0){px_s_fr_cue <- rep(0,146)}else{px_s_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,px_s,]),5)}
    if (length(px_f) == 1){px_f_fr_cue <- rollmean(all_cue_fr[unit_num,px_f,],5)}else if(length(px_f) == 0){px_f_fr_cue <- rep(0,146)}else{px_f_fr_cue <- rollmean(colMeans(all_cue_fr[unit_num,px_f,]),5)}
    
    if (length(p0_s) == 1){p0_s_fr_res <- rollmean(all_res_fr[unit_num,p0_s,],5)}else if(length(p0_s) == 0){p0_s_fr_res <- rep(0,146)}else{p0_s_fr_res <- rollmean(colMeans(all_res_fr[unit_num,p0_s,]),5)}
    if (length(p0_f) == 1){p0_f_fr_res <- rollmean(all_res_fr[unit_num,p0_f,],5)}else if(length(p0_f) == 0){p0_f_fr_res <- rep(0,146)}else{p0_f_fr_res <- rollmean(colMeans(all_res_fr[unit_num,p0_f,]),5)}
    if (length(px_s) == 1){px_s_fr_res <- rollmean(all_res_fr[unit_num,px_s,],5)}else if(length(px_s) == 0){px_s_fr_res <- rep(0,146)}else{px_s_fr_res <- rollmean(colMeans(all_res_fr[unit_num,px_s,]),5)}
    if (length(px_f) == 1){px_f_fr_res <- rollmean(all_res_fr[unit_num,px_f,],5)}else if(length(px_f) == 0){px_f_fr_res <- rep(0,146)}else{px_f_fr_res <- rollmean(colMeans(all_res_fr[unit_num,px_f,]),5)}
    
    
    p_sf_cue_avgs <- data.frame(time=time,p0_s_cue=p0_s_fr_cue,p0_f_cue=p0_f_fr_cue,px_s_cue=px_s_fr_cue,px_f_cue=px_f_fr_cue)
    p_sf_res_avgs <- data.frame(time=time,p0_s_res=p0_s_fr_res,p0_f_res=p0_f_fr_res,px_s_res=px_s_fr_res,px_f_res=px_f_fr_res)

    if (length(p0_s) <= 1){p0_s_fr_cue <- rep(0,146)}else{p0_s_fr_cue <- rollmean(sapply(data.frame(all_cue_fr[unit_num,p0_s,]),function(x)sd(x)/sqrt(length(x))),5)}
    if (length(p0_f) <= 1){p0_f_fr_cue <- rep(0,146)}else{p0_f_fr_cue <- rollmean(sapply(data.frame(all_cue_fr[unit_num,p0_f,]),function(x)sd(x)/sqrt(length(x))),5)}
    if (length(px_s) <= 1){px_s_fr_cue <- rep(0,146)}else{px_s_fr_cue <- rollmean(sapply(data.frame(all_cue_fr[unit_num,px_s,]),function(x)sd(x)/sqrt(length(x))),5)}
    if (length(px_f) <= 1){px_f_fr_cue <- rep(0,146)}else{px_f_fr_cue <- rollmean(sapply(data.frame(all_cue_fr[unit_num,px_f,]),function(x)sd(x)/sqrt(length(x))),5)}
    
    if (length(p0_s) <= 1){p0_s_fr_res <- rep(0,146)}else{p0_s_fr_res <- rollmean(sapply(data.frame(all_cue_fr[unit_num,p0_s,]),function(x)sd(x)/sqrt(length(x))),5)}
    if (length(p0_f) <= 1){p0_f_fr_res <- rep(0,146)}else{p0_f_fr_res <- rollmean(sapply(data.frame(all_cue_fr[unit_num,p0_f,]),function(x)sd(x)/sqrt(length(x))),5)}
    if (length(px_s) <= 1){px_s_fr_res <- rep(0,146)}else{px_s_fr_res <- rollmean(sapply(data.frame(all_cue_fr[unit_num,px_s,]),function(x)sd(x)/sqrt(length(x))),5)}
    if (length(px_f) <= 1){px_f_fr_res <- rep(0,146)}else{px_f_fr_res <- rollmean(sapply(data.frame(all_cue_fr[unit_num,px_f,]),function(x)sd(x)/sqrt(length(x))),5)}
    
    p_sf_cue_sems <- data.frame(time=time,p0_s_cue=p0_s_fr_cue,p0_f_cue=p0_f_fr_cue,px_s_cue=px_s_fr_cue,px_f_cue=px_f_fr_cue)
    p_sf_res_sems <- data.frame(time=time,p0_s_res=p0_s_fr_res,p0_f_res=p0_f_fr_res,px_s_res=px_s_fr_res,px_f_res=px_f_fr_res)
    
    p_sf_cue_avgs.m <- melt(p_sf_cue_avgs,id.vars="time",variable="comb")
    plt_cue <- ggplot(p_sf_cue_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=1) + theme_classic()
    plt_cue <- plt_cue + scale_colour_manual(values=c("black","gray69","firebrick","lightpink2")) + labs(title="Cue") +  geom_vline(xintercept=0)
    
    plt_cue <- plt_cue + geom_ribbon(data=p_sf_cue_avgs,aes(x=time,ymin=p_sf_cue_avgs$p0_s-p_sf_cue_sems$p0_s,ymax=p_sf_cue_avgs$p0_s+p_sf_cue_sems$p0_s),inherit.aes=FALSE,alpha=0.2,fill="black")
    plt_cue <- plt_cue + geom_ribbon(data=p_sf_cue_avgs,aes(x=time,ymin=p_sf_cue_avgs$p0_f-p_sf_cue_sems$p0_f,ymax=p_sf_cue_avgs$p0_f+p_sf_cue_sems$p0_f),inherit.aes=FALSE,alpha=0.2,fill="gray69")
    plt_cue <- plt_cue + geom_ribbon(data=p_sf_cue_avgs,aes(x=time,ymin=p_sf_cue_avgs$px_s-p_sf_cue_sems$px_s,ymax=p_sf_cue_avgs$px_s+p_sf_cue_sems$px_s),inherit.aes=FALSE,alpha=0.2,fill="firebrick")
    plt_cue <- plt_cue + geom_ribbon(data=p_sf_cue_avgs,aes(x=time,ymin=p_sf_cue_avgs$px_f-p_sf_cue_sems$px_f,ymax=p_sf_cue_avgs$px_f+p_sf_cue_sems$px_f),inherit.aes=FALSE,alpha=0.2,fill="lightpink2")
    plt_cue <- plt_cue + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    
    p_sf_res_avgs.m <- melt(p_sf_res_avgs,id.vars="time",variable="comb")
    plt_res <- ggplot(p_sf_res_avgs.m,aes(x=time,y=value)) + geom_line(aes(colour=comb),size=1) + theme_classic()
    plt_res <- plt_res + scale_colour_manual(values=c("black","gray69","firebrick","lightpink2")) + labs(title="Result",y="z-score", x="Time(s)",colour="Combination") + geom_vline(xintercept=0)
    
    plt_res <- plt_res + geom_ribbon(data=p_sf_res_avgs,aes(x=time,ymin=p_sf_res_avgs$p0_s-p_sf_res_sems$p0_s,ymax=p_sf_res_avgs$p0_s+p_sf_res_sems$p0_s),inherit.aes=FALSE,alpha=0.2,fill="black")
    plt_res <- plt_res + geom_ribbon(data=p_sf_res_avgs,aes(x=time,ymin=p_sf_res_avgs$p0_f-p_sf_res_sems$p0_f,ymax=p_sf_res_avgs$p0_f+p_sf_res_sems$p0_f),inherit.aes=FALSE,alpha=0.2,fill="gray69")
    plt_res <- plt_res + geom_ribbon(data=p_sf_res_avgs,aes(x=time,ymin=p_sf_res_avgs$px_s-p_sf_res_sems$px_s,ymax=p_sf_res_avgs$px_s+p_sf_res_sems$px_s),inherit.aes=FALSE,alpha=0.2,fill="firebrick")
    plt_res <- plt_res + geom_ribbon(data=p_sf_res_avgs,aes(x=time,ymin=p_sf_res_avgs$px_f-p_sf_res_sems$px_f,ymax=p_sf_res_avgs$px_f+p_sf_res_sems$px_f),inherit.aes=FALSE,alpha=0.2,fill="lightpink2")
    plt_res <- plt_res + theme(legend.position="none",axis.title=element_blank(),axis.text.y=element_text(size=rel(1.5)),axis.text.x=element_blank(),plot.title=element_text(size=rel(2)))
    
    multiplot(plt_cue,plt_res)
    graphics.off()
    
  
  }
}


# save.image(file="rearranged_data.RData")
rm(list=ls())

