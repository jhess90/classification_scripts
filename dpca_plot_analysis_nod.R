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

all_data <- read.xlsx('perc_variance.xlsx',sheet=1,colNames=T)

PmD_total <- as.numeric(all_data$X17[7])
S1_total <- as.numeric(all_data$X33[7])
M1_total <- as.numeric(all_data$X49[7])

total_percs <- c(M1_total,S1_total,PmD_total)
names <- c('S1','M1','PmD')

perc.df <- data.frame(value=total_percs,Region=names)

total_perc_plt <- ggplot(perc.df) + geom_bar(aes(x=Region,y=value),stat="identity") + geom_text(aes(x=perc.df$Region,y=perc.df$value + 0.025),label=scales::percent(perc.df$value))
total_perc_plt <- total_perc_plt + scale_y_continuous(labels = scales::percent) + labs(title = "Explained Variance",y="Percentage")

png('explained_variance.png',width=8,height=6,units="in",res=500)
plot(total_perc_plt) 

graphics.off()




