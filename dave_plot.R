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
library(R.matlab)
library(broom)
library(plyr)
library(reshape)


data <- read.csv('SigUnits.csv',header=T,sep=',')

postcue_504_c <- c(data$X..significant.units[1],data$X..significant.units[9],data$X..significant.units[17])
postcue_504_nc <- c(data$X..significant.units[2],data$X..significant.units[10],data$X..significant.units[18])
postreward_504_c <- c(data$X..significant.units[5],data$X..significant.units[13],data$X..significant.units[21])
postreward_504_nc <-c(data$X..significant.units[6],data$X..significant.units[14],data$X..significant.units[22])

postcue_0059_c <- c(data$X..significant.units[3],data$X..significant.units[11],data$X..significant.units[19])
postcue_0059_nc <- c(data$X..significant.units[4],data$X..significant.units[12],data$X..significant.units[20])
postreward_0059_c <- c(data$X..significant.units[7],data$X..significant.units[15],data$X..significant.units[23])
postreward_0059_nc <-c(data$X..significant.units[8],data$X..significant.units[16],data$X..significant.units[24])

levels <- c('0-1','0-2','0-3')

postcue_504_c.df <- data.frame(perc = postcue_504_c,levels=levels,group='cue')
postcue_504_nc.df <- data.frame(perc = postcue_504_nc,levels=levels,group='no cue')
postreward_504_c.df <- data.frame(perc = postreward_504_c,levels=levels,group='cue')
postreward_504_nc.df <- data.frame(perc = postreward_504_nc,levels=levels,group='no cue')

postcue_0059_c.df <- data.frame(perc = postcue_0059_c,levels=levels,group='cue')
postcue_0059_nc.df <- data.frame(perc = postcue_0059_nc,levels=levels,group='no cue')
postreward_0059_c.df <- data.frame(perc = postreward_0059_c,levels=levels,group='cue')
postreward_0059_nc.df <- data.frame(perc = postreward_0059_nc,levels=levels,group='no cue')

postcue_504 <- rbind(postcue_504_c.df,postcue_504_nc.df)
postreward_504 <- rbind(postreward_504_c.df,postreward_504_nc.df)
postcue_0059 <- rbind(postcue_0059_c.df,postcue_0059_nc.df)
postreward_0059 <- rbind(postreward_0059_c.df,postreward_0059_nc.df)

#png('perc_sig_units.png',width=8,height=6,units="in",res=500)
svg('perc_sig_units.svg',width=8,height=6,pointsize=12)

postcue_504_plt <- ggplot(data=postcue_504,aes(x=levels,y=perc,group=group,color=group)) + geom_line() + geom_point(size=3)
postcue_504_plt <- postcue_504_plt + labs(title='504 Post Cue',x="Comparison",y="Percentage Significant Units",color="") + ylim(0,80)

postreward_504_plt <- ggplot(data=postreward_504,aes(x=levels,y=perc,group=group,color=group)) + geom_line() + geom_point(size=3)
postreward_504_plt <- postreward_504_plt + labs(title='504 Post Reward',x="Comparison",y="Percentage Significant Units",color="") + ylim(0,80)

postcue_0059_plt <- ggplot(data=postcue_0059,aes(x=levels,y=perc,group=group,color=group)) + geom_line() + geom_point(size=3)
postcue_0059_plt <- postcue_0059_plt + labs(title='0059 Post Cue',x="Comparison",y="Percentage Significant Units",color="") + ylim(0,80)

postreward_0059_plt <- ggplot(data=postreward_0059,aes(x=levels,y=perc,group=group,color=group)) + geom_line() + geom_point(size=3) 
postreward_0059_plt <- postreward_0059_plt + labs(title='0059 Post Reward',x="Comparison",y="Percentage Significant Units",color="") + ylim(0,90)

multiplot(postcue_504_plt,postreward_504_plt,postcue_0059_plt,postreward_0059_plt,cols=2)

graphics.off()
