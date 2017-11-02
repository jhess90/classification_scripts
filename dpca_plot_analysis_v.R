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

all_data <- read.xlsx('perc_variance_v.xlsx',sheet=1,colNames=T)

PmD_total <- as.numeric(all_data$X17[6])
S1_total <- as.numeric(all_data$X33[6])
M1_total <- as.numeric(all_data$X49[6])

total_percs <- c(M1_total,S1_total,PmD_total)
names <- c('S1','M1','PmD')

perc.df <- data.frame(value=total_percs,Region=names)

total_perc_plt <- ggplot(perc.df) + geom_bar(aes(x=Region,y=value),stat="identity") + geom_text(aes(x=perc.df$Region,y=perc.df$value + 0.025),label=scales::percent(perc.df$value))
total_perc_plt <- total_perc_plt + scale_y_continuous(labels = scales::percent) + labs(title = "Explained Variance",y="Percentage")

png('explained_variance_v.png',width=8,height=6,units="in",res=500)
plot(total_perc_plt) 

graphics.off()

var_names <- c()
PmD_var <- c()
S1_var <- c()
M1_var <- c()

for(latent_var in 1:(dim(all_data)[1] -2)){
  var_names <- c(var_names,all_data[latent_var + 1,1])
  PmD_var <- c(PmD_var,as.double(all_data[latent_var+1,17]))
  S1_var <- c(S1_var,as.double(all_data[latent_var+1,33]))
  M1_var <- c(M1_var,as.double(all_data[latent_var+1,49]))

}

PmD_total <- as.double(all_data[latent_var+2,17])
S1_total <- as.double(all_data[latent_var+2,33])
M1_total <- as.double(all_data[latent_var+2,49])

PmD_percs <- PmD_var / PmD_total
S1_percs <- S1_var / S1_total
M1_percs <- M1_var / M1_total

percs.df <- data.frame(PmD_percs = PmD_percs,S1_percs=S1_percs,M1_percs=M1_percs,row.names = var_names)
percs.melt <- melt(as.matrix(percs.df))

png('explained_variance_variables_v.png',width=8,height=6,units="in",res=500)

plt <- ggplot(percs.melt) + geom_bar(aes(x=Var1,y=value,fill=Var2),stat="identity",position="dodge") + labs(title="Explained Variance",y="Percentage",x="Variables",fill="Region")
plt <- plt + scale_y_continuous(labels = scales::percent) #+ geom_text(aes(x=Var1,y=percs.melt$value + 0.025),label=scales::percent(percs.melt$value),position="dodge")

plot(plt)
graphics.off()

rm(list=ls())
