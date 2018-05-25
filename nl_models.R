rm(list=ls())

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

tryCatch({
  source("~/documents/lab/workspace/Classification_scripts/multiplot.R")
  source("~/documents/lab/workspace/Classification_scripts/Ski.Mack.JH.R")
  print('on laptop')
},warning=function(war){(print('on beaver'))
  source("~/workspace/classification_scripts/multiplot.R")
  source("~/workspace/classification_scripts/Ski.Mack.JH.R")
},finally={print('sourced multiplot and ski mack')})

saveAsPng <- T

region_list <- c('M1','S1','PmD')
window_names <- c('ac','br','ar','concat')
#TODO unhardcode
#ac = 51:150, br = 151:200, ar = 201:300, concat= 51:300
windows <- rbind(c(51,150),c(151,200),c(201,300),c(51,300))

#########


for(region_index in 1:length(region_list)){
  cat("\nregion:",region_list[region_index])
  
  readin <- readMat(paste('simple_output_',region_list[region_index],'.mat',sep=""))
  
  all_cue_fr <- readin$return.dict[,,1]$all.cue.fr
  all_res_fr <- readin$return.dict[,,1]$all.res.fr
  condensed <- readin$return.dict[,,1]$condensed
  bin_size <- readin$return.dict[,,1]$params[,,1]$bin.size[,]
  
  r_vals <- condensed[,4]
  p_vals <- condensed[,5]
  total_units <- dim(all_cue_fr)[1]
  
  total_fr <- abind(all_cue_fr,all_res_fr,along=3)
  
  for (i in length(window_names)){
    frs <- total_fr[,,windows[i,1]:windows[i,2]]
    mean_frs <- rowMeans(frs,dims=2)
    
    
    for(unit_num in 1:total_units){
      
      input <- data.frame(mean_fr = mean_frs[unit_num,],r = r_vals, p = p_vals)
      
      #linear (simple model, same as in python)
      linear_model <- lm(formula = mean_fr ~ r + p,data=input)
      
    }
    
    
    
    
     
  }
  
  
  
  
  
  

  
  
}


# save.image(file="nl_model.RData")
#rm(list=ls())

