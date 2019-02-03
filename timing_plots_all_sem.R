rm(list=ls())

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


########

nhp_id <- 'nhp'

sem <- function(x){sd(x)/sqrt(length(x))}


######

file_list <- Sys.glob("block*.RData")

ind = 1
for(block_name in file_list){
  cat(block_name)
  
  if(ind == 1){
    attach(block_name)
    temp <- condensed
    
    temp[,9] <- (temp[,2] + temp[,3])- temp[,1]
    temp[2:length(temp[,1]),10] <- diff(temp[,1])
    
    temp <- temp[2:length(temp[,1]),]
    condensed_all <- temp
    
    detach()
  }else{
    attach(block_name)
    temp <- condensed
    
    temp[,9] <- (temp[,2] + temp[,3])- temp[,1]
    temp[2:length(temp[,1]),10] <- diff(temp[,1])
    
    temp <- temp[2:length(temp[,1]),]

    condensed_all <- rbind(condensed_all,temp)
  
    detach() 
  }
  ind <- ind + 1
}


condensed <- condensed_all

condensed_all <- condensed_all[condensed_all[,9] > 0,]
condensed_all <- condensed_all[condensed_all[,10] > 0,]


r0 <- which(condensed[,4] == 0)
r1 <- which(condensed[,4] == 1)
r2 <- which(condensed[,4] == 2)
r3 <- which(condensed[,4] == 3)

p0 <- which(condensed[,5] == 0)
p1 <- which(condensed[,5] == 1)
p2 <- which(condensed[,5] == 2)
p3 <- which(condensed[,5] == 3)

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

rx <- which(condensed[,4] >= 1)
px <- which(condensed[,5] >= 1)

r0_fail <- res0[which(res0 %in% r0)]
rx_fail <- res0[which(res0 %in% rx)]
r0_succ <- res1[which(res1 %in% r0)]
rx_succ <- res1[which(res1 %in% rx)]

p0_fail <- res0[which(res0 %in% p0)]
px_fail <- res0[which(res0 %in% px)]
p0_succ <- res1[which(res1 %in% p0)]
px_succ <- res1[which(res1 %in% px)]

r0_p0 <- r0[which(r0 %in% p0)]
rx_p0 <- rx[which(rx %in% p0)]
r0_px <- r0[which(r0 %in% px)]
rx_px <- rx[which(rx %in% px)]

#reward
png(paste("time_", nhp_id,"_reward.png",sep=""),width=8,height=6,units="in",res=500)

ravgs <- data.frame(r_values=c(0,1,2,3),reach_time = c(mean(condensed[r0,9]),mean(condensed[r1,9]),mean(condensed[r2,9]),mean(condensed[r3,9])),intertrial = c(mean(condensed[r0,10]),mean(condensed[r1,10]),mean(condensed[r2,10]),mean(condensed[r3,10])))
rstds <- data.frame(r_values=c(0,1,2,3),reach_time = c(sd(condensed[r0,9]),sd(condensed[r1,9]),sd(condensed[r2,9]),sd(condensed[r3,9])),intertrial = c(sd(condensed[r0,10]),sd(condensed[r1,10]),sd(condensed[r2,10]),sd(condensed[r3,10])))

avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='r_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=r_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Reward",fill="") + theme(legend.position="none")

plot(plt)
graphics.off()

#punishment
png(paste("time_", nhp_id,"_punishment.png",sep=""),width=8,height=6,units="in",res=500)

pavgs <- data.frame(p_values=c(0,1,2,3),reach_time = c(mean(condensed[p0,9]),mean(condensed[p1,9]),mean(condensed[p2,9]),mean(condensed[p3,9])),intertrial = c(mean(condensed[p0,10]),mean(condensed[p1,10]),mean(condensed[p2,10]),mean(condensed[p3,10])))
pstds <- data.frame(p_values=c(0,1,2,3),reach_time = c(sd(condensed[p0,9]),sd(condensed[p1,9]),sd(condensed[p2,9]),sd(condensed[p3,9])),intertrial = c(sd(condensed[p0,10]),sd(condensed[p1,10]),sd(condensed[p2,10]),sd(condensed[p3,10])))

avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='p_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=p_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Punishment",fill="") + theme(legend.position="none")

plot(plt)
graphics.off()


#reward
png(paste("time_", nhp_id,"_reward_binary.png",sep=""),width=8,height=6,units="in",res=500)

ravgs <- data.frame(r_values=c(0,1),reach_time = c(mean(condensed[r0,9]),mean(condensed[rx,9])),intertrial = c(mean(condensed[r0,10]),mean(condensed[rx,10])))
rstds <- data.frame(r_values=c(0,1),reach_time = c(sem(condensed[r0,9]),sem(condensed[rx,9])),intertrial = c(sem(condensed[r0,10]),sem(condensed[rx,10])))

avg_melt <- melt(ravgs,id="r_values",variable.name='type',value.name='avg')
std_melt <- melt(rstds,id="r_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='r_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=r_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Reward",fill="") + scale_x_discrete(limits=0:1,labels=c("R0","RX")) + theme(legend.position="none")

plot(plt)
graphics.off()

#punishment
png(paste("time_", nhp_id,"_punishment_binary.png",sep=""),width=8,height=6,units="in",res=500)

pavgs <- data.frame(p_values=c(0,1),reach_time = c(mean(condensed[p0,9]),mean(condensed[px,9])),intertrial = c(mean(condensed[p0,10]),mean(condensed[px,10])))
pstds <- data.frame(p_values=c(0,1),reach_time = c(sem(condensed[p0,9]),sem(condensed[px,9])),intertrial = c(sem(condensed[p0,10]),sem(condensed[px,10])))

avg_melt <- melt(pavgs,id="p_values",variable.name='type',value.name='avg')
std_melt <- melt(pstds,id="p_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='p_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=p_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Punishment",fill="") + scale_x_discrete(limits=0:1,labels=c("P0","PX")) + theme(legend.position="none")

plot(plt)
graphics.off()

#result
png(paste("time_", nhp_id,"_result_binary.png",sep=""),width=8,height=6,units="in",res=500)

resavgs <- data.frame(res_values=c(0,1),reach_time = c(mean(condensed[res0,9]),mean(condensed[res1,9])),intertrial = c(mean(condensed[res0,10]),mean(condensed[res1,10])))
resstds <- data.frame(res_values=c(0,1),reach_time = c(sem(condensed[res0,9]),sem(condensed[res1,9])),intertrial = c(sem(condensed[res0,10]),sem(condensed[res1,10])))

avg_melt <- melt(resavgs,id="res_values",variable.name='type',value.name='avg')
std_melt <- melt(resstds,id="res_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='res_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=res_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Result",fill="") + scale_x_discrete(limits=0:1,labels=c("fail","succ")) + theme(legend.position="none")

plot(plt)
graphics.off()

# #value
# png(paste("time_", nhp_id,"_value_binary.png",sep=""),width=8,height=6,units="in",res=500)
# 
# vavgs <- data.frame(v_values=c(-1,0,1),reach_time = c(mean(condensed[v_x,9]),mean(condensed[v0,9]),mean(condensed[vx,9])),intertrial = c(mean(condensed[v_x,10]),mean(condensed[v0,10]),mean(condensed[vx,10])))
# vstds <- data.frame(v_values=c(-1,0,1),reach_time = c(sem(condensed[v_x,9]),sem(condensed[v0,9]),sem(condensed[vx,9])),intertrial = c(sem(condensed[v_x,10]),sem(condensed[v0,10]),sem(condensed[vx,10])))
# 
# avg_melt <- melt(vavgs,id="v_values",variable.name='type',value.name='avg')
# std_melt <- melt(vstds,id="v_values",variable.name='type',value.name='std')
# 
# test <- merge(std_melt,avg_melt,row.names='v_values')
# test[is.na(test)] <- 0
# 
# plt <- ggplot(data=test,aes(x=v_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
# plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + geom_text(aes(y=0.75),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9)) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Value",fill="") + scale_x_discrete(limits=-1:1,labels=c("V_X","V0","VX"))
# 
# plot(plt)
# graphics.off()
# 
# #motivation
# png(paste("time_", nhp_id,"_motivation_binary.png",sep=""),width=8,height=6,units="in",res=500)
# 
# mavgs <- data.frame(m_values=c(0,1,2),reach_time = c(mean(condensed[m0,9]),mean(condensed[mx,9]),mean(condensed[m2x,9])),intertrial = c(mean(condensed[m0,10]),mean(condensed[mx,10]),mean(condensed[m2x,10])))
# mstds <- data.frame(m_values=c(0,1,2),reach_time = c(sem(condensed[m0,9]),sem(condensed[mx,9]),sem(condensed[m2x,9])),intertrial = c(sem(condensed[m0,10]),sem(condensed[mx,10]),sem(condensed[m2x,10])))
# 
# avg_melt <- melt(mavgs,id="m_values",variable.name='type',value.name='avg')
# std_melt <- melt(mstds,id="m_values",variable.name='type',value.name='std')
# 
# test <- merge(std_melt,avg_melt,row.names='m_values')
# test[is.na(test)] <- 0
# 
# plt <- ggplot(data=test,aes(x=m_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
# plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + geom_text(aes(y=0.75),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9)) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Motivation",fill="") + scale_x_discrete(limits=0:2,labels=c("M0","MX","M2X"))
# 
# # plot(plt)
# graphics.off()
# 
# ##########
# ##########
# 
# #reward sf
# png(paste("time_", nhp_id,"_reward_sf_binary.png",sep=""),width=8,height=6,units="in",res=500)
# 
# r_s_avgs <- data.frame(r_values=c(0,1),reach_time_s = c(mean(condensed[r0_succ,9]),mean(condensed[rx_succ,9])),intertrial_s = c(mean(condensed[r0_succ,10]),mean(condensed[rx_succ,10])))
# r_s_stds <- data.frame(r_values=c(0,1),reach_time_s = c(sem(condensed[r0_succ,9]),sem(condensed[rx_succ,9])),intertrial_s = c(sem(condensed[r0_succ,10]),sem(condensed[rx_succ,10])))
# r_f_avgs <- data.frame(r_values=c(0,1),reach_time_f = c(mean(condensed[r0_fail,9]),mean(condensed[rx_fail,9])),intertrial_f = c(mean(condensed[r0_fail,10]),mean(condensed[rx_fail,10])))
# r_f_stds <- data.frame(r_values=c(0,1),reach_time_f = c(sem(condensed[r0_fail,9]),sem(condensed[rx_fail,9])),intertrial_f = c(sem(condensed[r0_fail,10]),sem(condensed[rx_fail,10])))
# 
# avg_s_melt <- melt(r_s_avgs,id="r_values",variable.name='type',value.name='avg')
# std_s_melt <- melt(r_s_stds,id="r_values",variable.name='type',value.name='std')
# avg_f_melt <- melt(r_f_avgs,id="r_values",variable.name='type',value.name='avg')
# std_f_melt <- melt(r_f_stds,id="r_values",variable.name='type',value.name='std')
# 
# test_s <- merge(std_s_melt,avg_s_melt,row.names='r_values')
# test_f <- merge(std_f_melt,avg_f_melt,row.names='r_values')
# test <- rbind(test_s,test_f)
# test[is.na(test)] <- 0
# 
# plt <- ggplot(data=test,aes(x=r_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
# plt <- plt + scale_fill_manual(values=c("royalblue","seagreen","paleturquoise","lightgreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Reward",fill="") + scale_x_discrete(limits=0:1,labels=c("R0","RX"))+ theme(legend.position="none") # + geom_text(aes(y=0.75),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9))
# 
# plot(plt)
# graphics.off()
# 
# #punishment sf
# png(paste("time_", nhp_id,"_punishment_sf_binary.png",sep=""),width=8,height=6,units="in",res=500)
# 
# p_s_avgs <- data.frame(p_values=c(0,1),reach_time_s = c(mean(condensed[p0_succ,9]),mean(condensed[px_succ,9])),intertrial_s = c(mean(condensed[p0_succ,10]),mean(condensed[px_succ,10])))
# p_s_stds <- data.frame(p_values=c(0,1),reach_time_s = c(sem(condensed[p0_succ,9]),sem(condensed[px_succ,9])),intertrial_s = c(sem(condensed[p0_succ,10]),sem(condensed[px_succ,10])))
# p_f_avgs <- data.frame(p_values=c(0,1),reach_time_f = c(mean(condensed[p0_fail,9]),mean(condensed[px_fail,9])),intertrial_f = c(mean(condensed[p0_fail,10]),mean(condensed[px_fail,10])))
# p_f_stds <- data.frame(p_values=c(0,1),reach_time_f = c(sem(condensed[p0_fail,9]),sem(condensed[px_fail,9])),intertrial_f = c(sem(condensed[p0_fail,10]),sem(condensed[px_fail,10])))
# 
# avg_s_melt <- melt(p_s_avgs,id="p_values",variable.name='type',value.name='avg')
# std_s_melt <- melt(p_s_stds,id="p_values",variable.name='type',value.name='std')
# avg_f_melt <- melt(p_f_avgs,id="p_values",variable.name='type',value.name='avg')
# std_f_melt <- melt(p_f_stds,id="p_values",variable.name='type',value.name='std')
# 
# test_s <- merge(std_s_melt,avg_s_melt,row.names='p_values')
# test_f <- merge(std_f_melt,avg_f_melt,row.names='p_values')
# test <- rbind(test_s,test_f)
# test[is.na(test)] <- 0
# 
# plt <- ggplot(data=test,aes(x=p_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
# plt <- plt + scale_fill_manual(values=c("royalblue","seagreen","paleturquoise","lightgreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Punishment",fill="") + scale_x_discrete(limits=0:1,labels=c("P0","PX")) + theme(legend.position="none")
# std_f_melt <- melt(p_f_stds,id="c_values",variable.name='type',value.name='std')
# 
# test_s <- merge(std_s_melt,avg_s_melt,row.names='c_values')
# test_f <- merge(std_f_melt,avg_f_melt,row.names='c_values')
# test <- rbind(test_s,test_f)
# test[is.na(test)] <- 0
# 
# plt <- ggplot(data=test,aes(x=c_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
# plt <- plt + scale_fill_manual(values=c("royalblue","seagreen","paleturquoise","lightgreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Combination",fill="") + scale_x_discrete(limits=0:3,labels=c("R0_P0","R0_PX","RX_P0","RX_PX")) # + geom_text(aes(y=0.75),size=3,label=sprintf("%0.2f", round(test$avg, digits = 2)),position=position_dodge(width=0.9))
# 
# plot(plt)
# graphics.off()


#comb 
png(paste("time_", nhp_id,"_comb_binary.png",sep=""),width=8,height=6,units="in",res=500)

comb_avgs <- data.frame(c_values=c(0,1,2,3),reach_time = c(mean(condensed[r0_p0,9]),mean(condensed[r0_px,9]),mean(condensed[rx_p0,9]),mean(condensed[rx_px,9])),intertrial = c(mean(condensed[r0_p0,10]),mean(condensed[r0_px,10]),mean(condensed[rx_p0,10]),mean(condensed[rx_px,10])))
comb_stds <- data.frame(c_values=c(0,1,2,3),reach_time = c(sem(condensed[r0_p0,9]),sem(condensed[r0_px,9]),sem(condensed[rx_p0,9]),sem(condensed[rx_px,9])),intertrial = c(sem(condensed[r0_p0,10]),sem(condensed[r0_px,10]),sem(condensed[rx_p0,10]),sem(condensed[rx_px,10])))

avg_melt <- melt(comb_avgs,id="c_values",variable.name='type',value.name='avg')
std_melt <- melt(comb_stds,id="c_values",variable.name='type',value.name='std')

test <- merge(std_melt,avg_melt,row.names='c_values')
test[is.na(test)] <- 0

plt <- ggplot(data=test,aes(x=c_values,y=avg,ymax=avg+std,ymin=avg-std,fill=type)) + geom_bar(position="dodge",stat="identity") + geom_errorbar(position=position_dodge(width=0.9),color="gray32",width=0.25)
plt <- plt + scale_fill_manual(values=c("royalblue","seagreen")) + theme_classic() + labs(title="Average Times",y="Time (s)",x="Combination",fill="") + scale_x_discrete(limits=0:3,labels=c("R0_P0","R0_PX","RX_P0","RX_PX")) + theme(legend.position="none")
plot(plt)
graphics.off()








