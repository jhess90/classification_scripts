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
library(scatterplot3d)
library(rgl)
library(minpack.lm)

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
  #readin <- readMat(paste('non_z_simple_output_',region_list[region_index],'.mat',sep=""))
  
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
      input_diff <- data.frame(mean_fr = mean_frs[unit_num,],d = r_vals - p_vals)
      input_nl <- data.frame(mean_fr = mean_frs[unit_num,],r = r_vals, p = p_vals, a = 1, b = 1, c = 1, d = 1)
      
      #linear (simple linear model, same as in python): mean_fr = ax + by + c
      #should these all be fit with same way (nonlinear least squares? something else?)
      linear_model <- lm(formula = mean_fr ~ r + p,data=input)
      
      #difference (simple linear model): mean_fr = a(r-p) + b
      diff_model <- lm(formula=mean_fr ~ d, data=input_diff)
      
      #divisive nl: mean_fr = (a*r + b*p) / (a + c*b + d)
      #nl_form <- formula(mean_fr ~ (a*r + b*p) / (a + c*b + d) + e)
      #nl_model <- lm(formula = nl_form,data = input_nl)
      
      
      #start_list <- list(a=-1 * mean(input$mean_fr[input$r == 0]),b=mean(input$mean_fr[input$p -- 0]), c = .1, d = 1, e = -1)
      #out <- nls(mean_fr ~ (a*r + b*p) / (a + c*b + d),data=input,start=start_list)
      
      
      #form <- formula(mean_fr ~ a + b(r - p))
      ##have to expand out? a + br - bp
      #diff_model <- lm(formula=form,data=input)
      #
      #scatterplot3d(input$r,input$p,input$mean_fr)
      #r <- 0:3
      #p <- 0:3
      #a <- -1*mean(input$mean_fr[input$r == 0])
      #b <- mean(input$mean_fr[input$p -- 0])
      #c <- .5
        
        
      #need e offset b/ z-score, can be < 0. (Or switch to non-z-scored?)
      #e <- -.2
      #d <- (a * mean(input$mean_fr[input$r == 0]) + b * mean(input$mean_fr[input$p == 0])) / (mean(input$mean_fr) - e) - (c*b)

      #
      #f <- function(r,p,a,b,c,d,e){mean_fr = (a*r + b*p) / (a + c*b + d) + e}
      f <- function(r,p){mean_fr = (a*r + b*p) / (a + c*b + d) + e}
      
      #make formula to work with NLS? 
      #f_formula <- formula(mean_fr ~ (a*r + b*p) / (a + c*b + d) + e)
      
      #z <- outer(r,p,f)
      
      #persp3d(r, p, z, col="skyblue")
      
      
      #start_list <- list(a=-1 * mean(input$mean_fr[input$r == 0]),b=mean(input$mean_fr[input$p -- 0]), c = .1, d = 1, e = -.2)
      #out <- nls(f_formula,data=input,start=start_list)
      
      #out_2 <- nlsLM(f,data=input,start=start_list)
      #start_list <- list(a=-1 * mean(input$mean_fr[input$r == 0]),b=mean(input$mean_fr[input$p -- 0]), c = .1, d = 1, e = -1)
      ##start_list <- list(a = -50, b = -70, c = -1, d = 1, e = -1)
      ##out_2.1 <- nlsLM(mean_fr ~ (a*r + b*p) / (a + c*b + d) + e ,data=input,start=start_list,trace=T,control=list(maxiter=500),
      ##                lower = c(a = -1000, b = -100, c = -10, d = -10, e = -2), 
      ##                upper = c(a = 1000, b = 1000, c = 10, d = 10, e = 2))
      
      #start_list.2 <- list(a=-1 * mean(input$mean_fr[input$r == 0]),b=mean(input$mean_fr[input$p -- 0]), c = 0)
      #out_2.2 <- nlsLM(mean_fr ~ (a*r + b*p) / (a + b) + c,data=input,start=start_list.2,trace=T,control=list(maxiter=500),
      #                lower = c(a = -10, b = -10, c = -10), 
      #                upper = c(a = 10, b = 10, c = 10))
      
      
      tryCatch({
        start_list.2 <- list(a=mean(input$mean_fr[input$r == 0]),b=mean(input$mean_fr[input$p == 0]), c = -1)
        #start_list.2 <- list(a = -50, b - -60, c = -2)
        #start_list.2 <- list(a=1,b=1,c=1)
        #lm(input$mean_fr~input$r + input$p) to give estimates
        #start_list.2 <- list(a=-0.033, b = -0.0017, c = 0.093373)
        out_2.2 <- nlsLM(mean_fr ~ (a*r + b*p) / (a + b) + c ,data=input,start=start_list.2,control=list(maxiter=500)) #,trace=T)
        #                 lower = c(a = -10, b = -10, c = -10), 
        #                 upper = c(a = 10, b = 10, c = 10)) #,control = nls.control(minFactor = 1/2048))
      
        cat('\n','success 1: unit',unit_num, window_names[i])
        
        
      },error=function(e){
        #cat('error: unit ',unit_num,'\n')
      },finally={})
        
      tryCatch({
        #start_list.23 <- list(a=mean(input$mean_fr[input$r == 0]),b=mean(input$mean_fr[input$p == 0]), c = -1, d = -1)
        start_list.23 <- list(a = -50, b - -60, c = -2)
        #start_list.2 <- list(a=1,b=1,c=1)
        #lm(input$mean_fr~input$r + input$p) to give estimates
        #start_list.2 <- list(a=-0.033, b = -0.0017, c = 0.093373)
        out_2.3 <- nlsLM(mean_fr ~ (a*r + b*p) / (a + b) + c ,data=input,start=start_list.23,control=list(maxiter=500)) #,trace=T)
        #                 lower = c(a = -10, b = -10, c = -10), 
        #                 upper = c(a = 10, b = 10, c = 10)) #,control = nls.control(minFactor = 1/2048))
        
        cat('\n','success 2: unit',unit_num,window_names[i])
        
      },error=function(e){
        #cat('error: unit ',unit_num,'\n')
      },finally={})
      
        
      #x <- getInitial(mean_fr ~SSlogis(r,p,a,b),data=input, trace=T)
      
      
      #out_3 <- nls.lm(par=start_list,fn = f)
      
      #now trying to plot to find appropriate starting point. If this doesn't work, have to solve the initiation problem. But then will be diff for each unit?
      
    }
  }
}