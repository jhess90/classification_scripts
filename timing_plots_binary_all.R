library(openxlsx)
library(ggplot2)
library(reshape2)
#source("~/documents/lab/workspace/Classification_scripts/multiplot.R")
source("~/workspace/classification_scripts/multiplot.R")
library(zoo)
library(gplots)
library(RColorBrewer)
library(abind)
library(gridGraphics)
library(grid)
library(gridExtra)
library(R.matlab)

saveAsPng <- T


#########


nhp_id <- '0059'

if (nhp_id == '0059'){condensed <- readRDS('0059_total_condensed.rds')
}else if (nhp_id == '504'){condensed <- readRDS('504_total_condensed.rds')}


r0 <- which(condensed[,4] == 0)
rx <- which(condensed[,4] >= 1)

p0 <- which(condensed[,5] == 0)
px <- which(condensed[,5] >= 1)

v_x <- which(condensed[,7] <= -1)
v0 <- which(condensed[,7] == 0)
vx <- which(condensed[,7] >= 1)

if (max(condensed[,8]) == 6){
  m0 <- which(condensed[,8] == 0)
  mx <- which(condensed[,8] == 3)
  m2x <- which(condensed[,8] == 6)
}else{
  m0 <- which(condensed[,8] == 0)
  mx <- which(condensed[,8] == 1)
  m2x <- which(condensed[,8] == 2)
}

res0 <- which(condensed[,6] == 0)
res1 <- which(condensed[,6] == 1)

r0_fail <- which(res0 %in% r0)
rx_fail <- which(res0 %in% rx)
r0_succ <- which(res1 %in% r0)
rx_succ <- which(res1 %in% rx)

p0_fail <- which(res0 %in% p0)
px_fail <- which(res0 %in% px)
p0_succ <- which(res1 %in% p0)
px_succ <- which(res1 %in% px)

#reward
png(paste("time_", nhp_id,"_reward_binary.png",sep=""),width=8,height=6,units="in",res=500)

ravgs <- data.frame(r_values=c(0,1),reach_time = c(mean(condensed[r0,9]),mean(condensed[rx,9])),intertrial = c(mean(condensed[r0,10]),mean(condensed[rx,10])))
rstds <- data.frame(r_values=c(0,1),reach_time = c(sd(condensed[r0,9]),sd(condensed[rx,9])),intertrial = c(sd(condensed[r0,10]),sd(condensed[rx,10])))

avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='r_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=r_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9)) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Reward",fill="") + scale_x_discrete(limits=0:1,labels=c("R0","RX"))

plot(plt)
graphics.off()

#punishment
png(paste("time_", nhp_id,"_punishment_binary.png",sep=""),width=8,height=6,units="in",res=500)

pavgs <- data.frame(p_values=c(0,1),reach_time = c(mean(condensed[p0,9]),mean(condensed[px,9])),intertrial = c(mean(condensed[p0,10]),mean(condensed[px,10])))
pstds <- data.frame(p_values=c(0,1),reach_time = c(sd(condensed[p0,9]),sd(condensed[px,9])),intertrial = c(sd(condensed[p0,10]),sd(condensed[px,10])))

avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='p_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=p_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9)) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Punishment",fill="") + scale_x_discrete(limits=0:1,labels=c("P0","PX"))

plot(plt)
graphics.off()

#result
png(paste("time_", nhp_id,"_result_binary.png",sep=""),width=8,height=6,units="in",res=500)

resavgs <- data.frame(res_values=c(0,1),reach_time = c(mean(condensed[res0,9]),mean(condensed[res1,9])),intertrial = c(mean(condensed[res0,10]),mean(condensed[res1,10])))
resstds <- data.frame(res_values=c(0,1),reach_time = c(sd(condensed[res0,9]),sd(condensed[res1,9])),intertrial = c(sd(condensed[res0,10]),sd(condensed[res1,10])))

avg_melt <- melt(resavgs,id="res_values",variable.name='type',value.name='avg')
std_melt <- melt(resstds,id="res_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='res_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=res_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9)) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Result",fill="") + scale_x_discrete(limits=0:1,labels=c("fail","succ"))

plot(plt)
graphics.off()

#value
png(paste("time_", nhp_id,"_value_binary.png",sep=""),width=8,height=6,units="in",res=500)

vavgs <- data.frame(v_values=c(-1,0,1),reach_time = c(mean(condensed[v_x,9]),mean(condensed[v0,9]),mean(condensed[vx,9])),intertrial = c(mean(condensed[v_x,10]),mean(condensed[v0,10]),mean(condensed[vx,10])))
vstds <- data.frame(v_values=c(-1,0,1),reach_time = c(sd(condensed[v_x,9]),sd(condensed[v0,9]),sd(condensed[vx,9])),intertrial = c(sd(condensed[v_x,10]),sd(condensed[v0,10]),sd(condensed[vx,10])))

avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='v_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=v_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9)) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Value",fill="") + scale_x_discrete(limits=-1:1,labels=c("V_X","V0","VX"))

plot(plt)
graphics.off()

#motivation
png(paste("time_", nhp_id,"_motivation_binary.png",sep=""),width=8,height=6,units="in",res=500)

mavgs <- data.frame(m_values=c(0,1,2),reach_time = c(mean(condensed[m0,9]),mean(condensed[mx,9]),mean(condensed[m2x,9])),intertrial = c(mean(condensed[m0,10]),mean(condensed[mx,10]),mean(condensed[m2x,10])))
mstds <- data.frame(m_values=c(0,1,2),reach_time = c(sd(condensed[m0,9]),sd(condensed[mx,9]),sd(condensed[m2x,9])),intertrial = c(sd(condensed[m0,10]),sd(condensed[mx,10]),sd(condensed[m2x,10])))

avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='m_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=m_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9)) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Motivation",fill="") + scale_x_discrete(limits=0:2,labels=c("M0","MX","M2X"))

plot(plt)
graphics.off()

##########
##########

#reward sf
png(paste("time_", nhp_id,"_reward_sf_binary.png",sep=""),width=8,height=6,units="in",res=500)

r_s_avgs <- data.frame(r_values=c(0,1),reach_time_s = c(mean(condensed[r0_succ,9]),mean(condensed[rx_succ,9])),intertrial_s = c(mean(condensed[r0_succ,10]),mean(condensed[rx_succ,10])))
r_s_stds <- data.frame(r_values=c(0,1),reach_time_s = c(sd(condensed[r0_succ,9]),sd(condensed[rx_succ,9])),intertrial_s = c(sd(condensed[r0_succ,10]),sd(condensed[rx_succ,10])))
r_f_avgs <- data.frame(r_values=c(0,1),reach_time_f = c(mean(condensed[r0_fail,9]),mean(condensed[rx_fail,9])),intertrial_f = c(mean(condensed[r0_fail,10]),mean(condensed[rx_fail,10])))
r_f_stds <- data.frame(r_values=c(0,1),reach_time_f = c(sd(condensed[r0_fail,9]),sd(condensed[rx_fail,9])),intertrial_f = c(sd(condensed[r0_fail,10]),sd(condensed[rx_fail,10])))

avg_s_melt <- melt(r_s_avgs,id="r_values",variable.name='type',value.name='avg')
std_s_melt <- melt(r_s_stds,id="r_values",variable.name='type',value.name='std')
avg_f_melt <- melt(r_f_avgs,id="r_values",variable.name='type',value.name='avg')
std_f_melt <- melt(r_f_stds,id="r_values",variable.name='type',value.name='std')

test_s <- merge(std_s_melt,avg_s_melt,row.names='r_values')
test_f <- merge(std_f_melt,avg_f_melt,row.names='r_values')
test <- rbind(test_s,test_f)
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=r_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen","paleturquoise","lightgreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Reward",fill="") + scale_x_discrete(limits=0:1,labels=c("R0","RX")) # + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9))

plot(plt)
graphics.off()

#punishment sf
png(paste("time_", nhp_id,"_punishment_sf_binary.png",sep=""),width=8,height=6,units="in",res=500)

p_s_avgs <- data.frame(p_values=c(0,1),reach_time_s = c(mean(condensed[p0_succ,9]),mean(condensed[px_succ,9])),intertrial_s = c(mean(condensed[p0_succ,10]),mean(condensed[px_succ,10])))
p_s_stds <- data.frame(p_values=c(0,1),reach_time_s = c(sd(condensed[p0_succ,9]),sd(condensed[px_succ,9])),intertrial_s = c(sd(condensed[p0_succ,10]),sd(condensed[px_succ,10])))
p_f_avgs <- data.frame(p_values=c(0,1),reach_time_f = c(mean(condensed[p0_fail,9]),mean(condensed[px_fail,9])),intertrial_f = c(mean(condensed[p0_fail,10]),mean(condensed[px_fail,10])))
p_f_stds <- data.frame(p_values=c(0,1),reach_time_f = c(sd(condensed[p0_fail,9]),sd(condensed[px_fail,9])),intertrial_f = c(sd(condensed[p0_fail,10]),sd(condensed[px_fail,10])))

avg_s_melt <- melt(p_s_avgs,id="p_values",variable.name='type',value.name='avg')
std_s_melt <- melt(p_s_stds,id="p_values",variable.name='type',value.name='std')
avg_f_melt <- melt(p_f_avgs,id="p_values",variable.name='type',value.name='avg')
std_f_melt <- melt(p_f_stds,id="p_values",variable.name='type',value.name='std')

test_s <- merge(std_s_melt,avg_s_melt,row.names='p_values')
test_f <- merge(std_f_melt,avg_f_melt,row.names='p_values')
test <- rbind(test_s,test_f)
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=p_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen","paleturquoise","lightgreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Punishment",fill="") + scale_x_discrete(limits=0:1,labels=c("P0","PX"))  # + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9))

plot(plt)
graphics.off()


rm(list=ls())










