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

readin <- readMat('simple_output_M1.mat')
condensed <- readin$return.dict[,,1]$condensed

condensed[,9] <- (condensed[,2] + condensed[,3])- condensed[,1]
condensed[2:length(condensed[,1]),10] <- diff(condensed[,1])

condensed <- condensed[2:length(condensed[,1]),]


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

p0_fail <- res0[which(res0 %in% p0)]
p1_fail <- res0[which(res0 %in% p1)]
p2_fail <- res0[which(res0 %in% p2)]
p3_fail <- res0[which(res0 %in% p3)]
p0_succ <- res1[which(res1 %in% p0)]
p1_succ <- res1[which(res1 %in% p1)]
p2_succ <- res1[which(res1 %in% p2)]
p3_succ <- res1[which(res1 %in% p3)]

#reward
png("time_reward.png",width=8,height=6,units="in",res=500)

ravgs <- data.frame(r_values=c(0,1,2,3),reach_time = c(mean(condensed[r0,9]),mean(condensed[r1,9]),mean(condensed[r2,9]),mean(condensed[r3,9])),intertrial = c(mean(condensed[r0,10]),mean(condensed[r1,10]),mean(condensed[r2,10]),mean(condensed[r3,10])))
rstds <- data.frame(r_values=c(0,1,2,3),reach_time = c(sd(condensed[r0,9]),sd(condensed[r1,9]),sd(condensed[r2,9]),sd(condensed[r3,9])),intertrial = c(sd(condensed[r0,10]),sd(condensed[r1,10]),sd(condensed[r2,10]),sd(condensed[r3,10])))

avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='r_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=r_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9)) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Reward",fill="")

plot(plt)
graphics.off()

#punishment
png("time_punishment.png",width=8,height=6,units="in",res=500)

pavgs <- data.frame(p_values=c(0,1,2,3),reach_time = c(mean(condensed[p0,9]),mean(condensed[p1,9]),mean(condensed[p2,9]),mean(condensed[p3,9])),intertrial = c(mean(condensed[p0,10]),mean(condensed[p1,10]),mean(condensed[p2,10]),mean(condensed[p3,10])))
pstds <- data.frame(p_values=c(0,1,2,3),reach_time = c(sd(condensed[p0,9]),sd(condensed[p1,9]),sd(condensed[p2,9]),sd(condensed[p3,9])),intertrial = c(sd(condensed[p0,10]),sd(condensed[p1,10]),sd(condensed[p2,10]),sd(condensed[p3,10])))

avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='p_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=p_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9)) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Punishment",fill="")

plot(plt)
graphics.off()

#result
png("time_result.png",width=8,height=6,units="in",res=500)

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
png("time_value.png",width=8,height=6,units="in",res=500)

vavgs <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),reach_time = c(mean(condensed[v_3,9]),mean(condensed[v_2,9]),mean(condensed[v_1,9]),mean(condensed[v0,9]),mean(condensed[v1,9]),mean(condensed[v2,9]),mean(condensed[v3,9])),intertrial = c(mean(condensed[v_3,10]),mean(condensed[v_2,10]),mean(condensed[v_1,10]),mean(condensed[v0,10]),mean(condensed[v1,10]),mean(condensed[v2,10]),mean(condensed[v3,10])))
vstds <- data.frame(v_values=c(-3,-2,-1,0,1,2,3),reach_time = c(sd(condensed[v_3,9]),sd(condensed[v_2,9]),sd(condensed[v_1,9]),sd(condensed[v0,9]),sd(condensed[v1,9]),sd(condensed[v2,9]),sd(condensed[v3,9])),intertrial = c(sd(condensed[v_3,10]),sd(condensed[v_2,10]),sd(condensed[v_1,10]),sd(condensed[v0,10]),sd(condensed[v1,10]),sd(condensed[v2,10]),sd(condensed[v3,10])))

avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='v_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=v_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9)) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Value",fill="")

plot(plt)
graphics.off()

#motivation
png("time_motivation.png",width=8,height=6,units="in",res=500)

mavgs <- data.frame(m_values=c(0,1,2,3,4,5,6),reach_time = c(mean(condensed[m0,9]),mean(condensed[m1,9]),mean(condensed[m2,9]),mean(condensed[m3,9]),mean(condensed[m4,9]),mean(condensed[m5,9]),mean(condensed[m6,9])),intertrial = c(mean(condensed[m0,10]),mean(condensed[m1,10]),mean(condensed[m2,10]),mean(condensed[m3,10]),mean(condensed[m4,10]),mean(condensed[m5,10]),mean(condensed[m6,10])))
mstds <- data.frame(m_values=c(0,1,2,3,4,5,6),reach_time = c(sd(condensed[m0,9]),sd(condensed[m1,9]),sd(condensed[m2,9]),sd(condensed[m3,9]),sd(condensed[m4,9]),sd(condensed[m5,9]),sd(condensed[m6,9])),intertrial = c(sd(condensed[m0,10]),sd(condensed[m1,10]),sd(condensed[m2,10]),sd(condensed[m3,10]),sd(condensed[m4,10]),sd(condensed[m5,10]),sd(condensed[m6,10])))

avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='m_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=m_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9)) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Motivation",fill="")

plot(plt)
graphics.off()

##########
##########

#reward sf
png("time_reward_sf.png",width=8,height=6,units="in",res=500)

r_s_avgs <- data.frame(r_values=c(0,1,2,3),reach_time_s = c(mean(condensed[r0_succ,9]),mean(condensed[r1_succ,9]),mean(condensed[r2_succ,9]),mean(condensed[r3_succ,9])),intertrial_s = c(mean(condensed[r0_succ,10]),mean(condensed[r1_succ,10]),mean(condensed[r2_succ,10]),mean(condensed[r3_succ,10])))
r_s_stds <- data.frame(r_values=c(0,1,2,3),reach_time_s = c(sd(condensed[r0_succ,9]),sd(condensed[r1_succ,9]),sd(condensed[r2_succ,9]),sd(condensed[r3_succ,9])),intertrial_s = c(sd(condensed[r0_succ,10]),sd(condensed[r1_succ,10]),sd(condensed[r2_succ,10]),sd(condensed[r3_succ,10])))
r_f_avgs <- data.frame(r_values=c(0,1,2,3),reach_time_f = c(mean(condensed[r0_fail,9]),mean(condensed[r1_fail,9]),mean(condensed[r2_fail,9]),mean(condensed[r3_fail,9])),intertrial_f = c(mean(condensed[r0_fail,10]),mean(condensed[r1_fail,10]),mean(condensed[r2_fail,10]),mean(condensed[r3_fail,10])))
r_f_stds <- data.frame(r_values=c(0,1,2,3),reach_time_f = c(sd(condensed[r0_fail,9]),sd(condensed[r1_fail,9]),sd(condensed[r2_fail,9]),sd(condensed[r3_fail,9])),intertrial_f = c(sd(condensed[r0_fail,10]),sd(condensed[r1_fail,10]),sd(condensed[r2_fail,10]),sd(condensed[r3_fail,10])))

avg_s_melt <- melt(r_s_avgs,id="r_values",variable.name='type',value.name='avg')
std_s_melt <- melt(r_s_stds,id="r_values",variable.name='type',value.name='std')
avg_f_melt <- melt(r_f_avgs,id="r_values",variable.name='type',value.name='avg')
std_f_melt <- melt(r_f_stds,id="r_values",variable.name='type',value.name='std')

test_s <- merge(std_s_melt,avg_s_melt,row.names='r_values')
test_f <- merge(std_f_melt,avg_f_melt,row.names='r_values')
test <- rbind(test_s,test_f)
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=r_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen","paleturquoise","lightgreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Reward",fill="")  # + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9))

plot(plt)
graphics.off()

#punishment sf
png("time_punishment_sf.png",width=8,height=6,units="in",res=500)

p_s_avgs <- data.frame(p_values=c(0,1,2,3),reach_time_s = c(mean(condensed[p0_succ,9]),mean(condensed[p1_succ,9]),mean(condensed[p2_succ,9]),mean(condensed[p3_succ,9])),intertrial_s = c(mean(condensed[p0_succ,10]),mean(condensed[p1_succ,10]),mean(condensed[p2_succ,10]),mean(condensed[p3_succ,10])))
p_s_stds <- data.frame(p_values=c(0,1,2,3),reach_time_s = c(sd(condensed[p0_succ,9]),sd(condensed[p1_succ,9]),sd(condensed[p2_succ,9]),sd(condensed[p3_succ,9])),intertrial_s = c(sd(condensed[p0_succ,10]),sd(condensed[p1_succ,10]),sd(condensed[p2_succ,10]),sd(condensed[p3_succ,10])))
p_f_avgs <- data.frame(p_values=c(0,1,2,3),reach_time_f = c(mean(condensed[p0_fail,9]),mean(condensed[p1_fail,9]),mean(condensed[p2_fail,9]),mean(condensed[p3_fail,9])),intertrial_f = c(mean(condensed[p0_fail,10]),mean(condensed[p1_fail,10]),mean(condensed[p2_fail,10]),mean(condensed[p3_fail,10])))
p_f_stds <- data.frame(p_values=c(0,1,2,3),reach_time_f = c(sd(condensed[p0_fail,9]),sd(condensed[p1_fail,9]),sd(condensed[p2_fail,9]),sd(condensed[p3_fail,9])),intertrial_f = c(sd(condensed[p0_fail,10]),sd(condensed[p1_fail,10]),sd(condensed[p2_fail,10]),sd(condensed[p3_fail,10])))

avg_s_melt <- melt(p_s_avgs,id="p_values",variable.name='type',value.name='avg')
std_s_melt <- melt(p_s_stds,id="p_values",variable.name='type',value.name='std')
avg_f_melt <- melt(p_f_avgs,id="p_values",variable.name='type',value.name='avg')
std_f_melt <- melt(p_f_stds,id="p_values",variable.name='type',value.name='std')

test_s <- merge(std_s_melt,avg_s_melt,row.names='p_values')
test_f <- merge(std_f_melt,avg_f_melt,row.names='p_values')
test <- rbind(test_s,test_f)
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=p_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen","paleturquoise","lightgreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Punishment",fill="")  # + geom_text(aes(y=0.25),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9))

plot(plt)
graphics.off()


rm(list=ls())










