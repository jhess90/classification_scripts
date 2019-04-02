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
library(data.table)

attach('summary.RData')

M1_perc_diff_ac <- list()
for(name in names(M1_diffs_length_list_ac_total)){
  M1_perc_diff_ac[[name]] <- M1_diffs_length_list_ac_total[[name]] / M1_total_unit_num * 100
}
M1_perc_diff_br <- list()
for(name in names(M1_diffs_length_list_br_total)){
  M1_perc_diff_br[[name]] <- M1_diffs_length_list_br_total[[name]] / M1_total_unit_num * 100
}
M1_perc_diff_ar <- list()
for(name in names(M1_diffs_length_list_ar_total)){
  M1_perc_diff_ar[[name]] <- M1_diffs_length_list_ar_total[[name]] / M1_total_unit_num * 100
}  
M1_diffs <- as.data.frame(rbind("M1_ac"=M1_perc_diff_ac,"M1_br" =M1_perc_diff_br,"M1_ar" =M1_perc_diff_ar))

S1_perc_diff_ac <- list()
for(name in names(S1_diffs_length_list_ac_total)){
  S1_perc_diff_ac[[name]] <- S1_diffs_length_list_ac_total[[name]] / S1_total_unit_num * 100
}
S1_perc_diff_br <- list()
for(name in names(S1_diffs_length_list_br_total)){
  S1_perc_diff_br[[name]] <- S1_diffs_length_list_br_total[[name]] / S1_total_unit_num * 100
}
S1_perc_diff_ar <- list()
for(name in names(S1_diffs_length_list_ar_total)){
  S1_perc_diff_ar[[name]] <- S1_diffs_length_list_ar_total[[name]] / S1_total_unit_num * 100
}  
S1_diffs <- as.data.frame(rbind("S1_ac"=S1_perc_diff_ac,"S1_br" =S1_perc_diff_br,"S1_ar" =S1_perc_diff_ar))


PmD_perc_diff_ac <- list()
for(name in names(PmD_diffs_length_list_ac_total)){
  PmD_perc_diff_ac[[name]] <- PmD_diffs_length_list_ac_total[[name]] / PmD_total_unit_num * 100
}
PmD_perc_diff_br <- list()
for(name in names(PmD_diffs_length_list_br_total)){
  PmD_perc_diff_br[[name]] <- PmD_diffs_length_list_br_total[[name]] / PmD_total_unit_num * 100
}
PmD_perc_diff_ar <- list()
for(name in names(PmD_diffs_length_list_ar_total)){
  PmD_perc_diff_ar[[name]] <- PmD_diffs_length_list_ar_total[[name]] / PmD_total_unit_num * 100
}  
PmD_diffs <- as.data.frame(rbind("PmD_ac"=PmD_perc_diff_ac,"PmD_br" =PmD_perc_diff_br,"PmD_ar" =PmD_perc_diff_ar))



fwrite(M1_diffs,file='perc_diff_within_win.csv',append=T,row.names=T)
fwrite(S1_diffs,file='perc_diff_within_win.csv',append=T,row.names=T)
fwrite(PmD_diffs,file='perc_diff_within_win.csv',append=T,row.names=T)


#########

M1_win_r <- data.frame(pos=c(sum(M1_win_r_ac[,3]==1),sum(M1_win_r_br[,3]==1),sum(M1_win_r_ar[,3]==1)),neg=c(sum(M1_win_r_ac[,3]==-1),sum(M1_win_r_br[,3]==-1),sum(M1_win_r_ar[,3]==-1))) / M1_total_unit_num
row.names(M1_win_r) <- c('AC','BF','AF')
M1_win_p <- data.frame(pos=c(sum(M1_win_p_ac[,3]==1),sum(M1_win_p_br[,3]==1),sum(M1_win_p_ar[,3]==1)),neg=c(sum(M1_win_p_ac[,3]==-1),sum(M1_win_p_br[,3]==-1),sum(M1_win_p_ar[,3]==-1))) / M1_total_unit_num
row.names(M1_win_p) <- c('AC','BF','AF')
M1_win_r_s <- data.frame(pos=c(sum(M1_win_r_s_ac[,3]==1),sum(M1_win_r_s_br[,3]==1),sum(M1_win_r_s_ar[,3]==1)),neg=c(sum(M1_win_r_s_ac[,3]==-1),sum(M1_win_r_s_br[,3]==-1),sum(M1_win_r_s_ar[,3]==-1))) / M1_total_unit_num
row.names(M1_win_r_s) <- c('AC','BF','AF')
M1_win_p_f <- data.frame(pos=c(sum(M1_win_p_f_ac[,3]==1),sum(M1_win_p_f_br[,3]==1),sum(M1_win_p_f_ar[,3]==1)),neg=c(sum(M1_win_p_f_ac[,3]==-1),sum(M1_win_p_f_br[,3]==-1),sum(M1_win_p_f_ar[,3]==-1))) / M1_total_unit_num
row.names(M1_win_p_f) <- c('AC','BF','AF')
M1_win_res <- data.frame(pos=c(sum(M1_win_res_ac[,3]==1),sum(M1_win_res_br[,3]==1),sum(M1_win_res_ar[,3]==1)),neg=c(sum(M1_win_res_ac[,3]==-1),sum(M1_win_res_br[,3]==-1),sum(M1_win_res_ar[,3]==-1))) / M1_total_unit_num
row.names(M1_win_res) <- c('AC','BF','AF')

S1_win_r <- data.frame(pos=c(sum(S1_win_r_ac[,3]==1),sum(S1_win_r_br[,3]==1),sum(S1_win_r_ar[,3]==1)),neg=c(sum(S1_win_r_ac[,3]==-1),sum(S1_win_r_br[,3]==-1),sum(S1_win_r_ar[,3]==-1))) / S1_total_unit_num
row.names(S1_win_r) <- c('AC','BF','AF')
S1_win_p <- data.frame(pos=c(sum(S1_win_p_ac[,3]==1),sum(S1_win_p_br[,3]==1),sum(S1_win_p_ar[,3]==1)),neg=c(sum(S1_win_p_ac[,3]==-1),sum(S1_win_p_br[,3]==-1),sum(S1_win_p_ar[,3]==-1)))/ S1_total_unit_num
row.names(S1_win_p) <- c('AC','BF','AF')
S1_win_r_s <- data.frame(pos=c(sum(S1_win_r_s_ac[,3]==1),sum(S1_win_r_s_br[,3]==1),sum(S1_win_r_s_ar[,3]==1)),neg=c(sum(S1_win_r_s_ac[,3]==-1),sum(S1_win_r_s_br[,3]==-1),sum(S1_win_r_s_ar[,3]==-1)))/ S1_total_unit_num
row.names(S1_win_r_s) <- c('AC','BF','AF')
S1_win_p_f <- data.frame(pos=c(sum(S1_win_p_f_ac[,3]==1),sum(S1_win_p_f_br[,3]==1),sum(S1_win_p_f_ar[,3]==1)),neg=c(sum(S1_win_p_f_ac[,3]==-1),sum(S1_win_p_f_br[,3]==-1),sum(S1_win_p_f_ar[,3]==-1)))/ S1_total_unit_num
row.names(S1_win_p_f) <- c('AC','BF','AF')
S1_win_res <- data.frame(pos=c(sum(S1_win_res_ac[,3]==1),sum(S1_win_res_br[,3]==1),sum(S1_win_res_ar[,3]==1)),neg=c(sum(S1_win_res_ac[,3]==-1),sum(S1_win_res_br[,3]==-1),sum(S1_win_res_ar[,3]==-1)))/ S1_total_unit_num
row.names(S1_win_res) <- c('AC','BF','AF')

PmD_win_r <- data.frame(pos=c(sum(PmD_win_r_ac[,3]==1),sum(PmD_win_r_br[,3]==1),sum(PmD_win_r_ar[,3]==1)),neg=c(sum(PmD_win_r_ac[,3]==-1),sum(PmD_win_r_br[,3]==-1),sum(PmD_win_r_ar[,3]==-1))) / PmD_total_unit_num
row.names(PmD_win_r) <- c('AC','BF','AF')
PmD_win_p <- data.frame(pos=c(sum(PmD_win_p_ac[,3]==1),sum(PmD_win_p_br[,3]==1),sum(PmD_win_p_ar[,3]==1)),neg=c(sum(PmD_win_p_ac[,3]==-1),sum(PmD_win_p_br[,3]==-1),sum(PmD_win_p_ar[,3]==-1))) / PmD_total_unit_num
row.names(PmD_win_p) <- c('AC','BF','AF')
PmD_win_r_s <- data.frame(pos=c(sum(PmD_win_r_s_ac[,3]==1),sum(PmD_win_r_s_br[,3]==1),sum(PmD_win_r_s_ar[,3]==1)),neg=c(sum(PmD_win_r_s_ac[,3]==-1),sum(PmD_win_r_s_br[,3]==-1),sum(PmD_win_r_s_ar[,3]==-1))) / PmD_total_unit_num
row.names(PmD_win_r_s) <- c('AC','BF','AF')
PmD_win_p_f <- data.frame(pos=c(sum(PmD_win_p_f_ac[,3]==1),sum(PmD_win_p_f_br[,3]==1),sum(PmD_win_p_f_ar[,3]==1)),neg=c(sum(PmD_win_p_f_ac[,3]==-1),sum(PmD_win_p_f_br[,3]==-1),sum(PmD_win_p_f_ar[,3]==-1))) / PmD_total_unit_num
row.names(PmD_win_p_f) <- c('AC','BF','AF')
PmD_win_res <- data.frame(pos=c(sum(PmD_win_res_ac[,3]==1),sum(PmD_win_res_br[,3]==1),sum(PmD_win_res_ar[,3]==1)),neg=c(sum(PmD_win_res_ac[,3]==-1),sum(PmD_win_res_br[,3]==-1),sum(PmD_win_res_ar[,3]==-1))) / PmD_total_unit_num
row.names(PmD_win_res) <- c('AC','BF','AF')


##use later: Colors
#purples: 118,42,131    153,112,171   194,165,207
#greens:  166,219,160   90,174,97     27,120,55

########

window_labels <-  c("AC","BF","AF")

png(paste('M1_win_posneg.png',sep=""),width=8,height=6,units="in",res=500)

diffs_pos.df <- data.frame(window_labels=c(window_labels),R=M1_win_r$pos,R_S=M1_win_r_s$pos,P=M1_win_p$pos,P_F=M1_win_p_f$pos,Result=M1_win_res$pos,position=c(1,2,3))
diffs_neg.df <- data.frame(window_labels=c(window_labels),R=M1_win_r$neg*-1,R_S=M1_win_r_s$neg*-1,P=M1_win_p$neg*-1,P_F=M1_win_p_f$neg*-1,Result=M1_win_res$neg*-1,position=c(1,2,3))

diffs_pos.m <-melt(diffs_pos.df,id.vars=c('window_labels','position'))
diffs_neg.m <-melt(diffs_neg.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs_pos.m,aes(variable,value)) + geom_bar(data=diffs_pos.m,aes(fill=window_labels,group=position),stat="identity",position=position_dodge()) 
plt <- plt + geom_bar(data=diffs_neg.m,aes(fill=window_labels,group=position),stat="identity",position=position_dodge())
plt <- plt + scale_fill_manual(values=c(rgb(27,158,119,maxColorValue = 255),rgb(217,95,2,maxColorValue = 255),rgb(117,112,179,maxColorValue = 255)))

plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12),legend.position="none")
plt <- plt + geom_hline(yintercept=0,size=0.25)
plt <- plt + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F","Result"))

plot(plt)
graphics.off()

#
png(paste('S1_win_posneg.png',sep=""),width=8,height=6,units="in",res=500)

diffs_pos.df <- data.frame(window_labels=c(window_labels),R=S1_win_r$pos,R_S=S1_win_r_s$pos,P=S1_win_p$pos,P_F=S1_win_p_f$pos,Result=S1_win_res$pos,position=c(1,2,3))
diffs_neg.df <- data.frame(window_labels=c(window_labels),R=S1_win_r$neg*-1,R_S=S1_win_r_s$neg*-1,P=S1_win_p$neg*-1,P_F=S1_win_p_f$neg*-1,Result=S1_win_res$neg*-1,position=c(1,2,3))

diffs_pos.m <-melt(diffs_pos.df,id.vars=c('window_labels','position'))
diffs_neg.m <-melt(diffs_neg.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs_pos.m,aes(variable,value)) + geom_bar(data=diffs_pos.m,aes(fill=window_labels,group=position),stat="identity",position=position_dodge()) 
plt <- plt + geom_bar(data=diffs_neg.m,aes(fill=window_labels,group=position),stat="identity",position=position_dodge())
plt <- plt + scale_fill_manual(values=c(rgb(27,158,119,maxColorValue = 255),rgb(217,95,2,maxColorValue = 255),rgb(117,112,179,maxColorValue = 255)))

plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12),legend.position="none")
plt <- plt + geom_hline(yintercept=0,size=0.25)
plt <- plt + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F","Result"))

plot(plt)
graphics.off()

#
png(paste('PmD_win_posneg.png',sep=""),width=8,height=6,units="in",res=500)

diffs_pos.df <- data.frame(window_labels=c(window_labels),R=PmD_win_r$pos,R_S=PmD_win_r_s$pos,P=PmD_win_p$pos,P_F=PmD_win_p_f$pos,Result=PmD_win_res$pos,position=c(1,2,3))
diffs_neg.df <- data.frame(window_labels=c(window_labels),R=PmD_win_r$neg*-1,R_S=PmD_win_r_s$neg*-1,P=PmD_win_p$neg*-1,P_F=PmD_win_p_f$neg*-1,Result=PmD_win_res$neg*-1,position=c(1,2,3))

diffs_pos.m <-melt(diffs_pos.df,id.vars=c('window_labels','position'))
diffs_neg.m <-melt(diffs_neg.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs_pos.m,aes(variable,value)) + geom_bar(data=diffs_pos.m,aes(fill=window_labels,group=position),stat="identity",position=position_dodge()) 
plt <- plt + geom_bar(data=diffs_neg.m,aes(fill=window_labels,group=position),stat="identity",position=position_dodge())
plt <- plt + scale_fill_manual(values=c(rgb(27,158,119,maxColorValue = 255),rgb(217,95,2,maxColorValue = 255),rgb(117,112,179,maxColorValue = 255)))

plt <- plt + theme_classic() + labs(title=paste('PmD\nTotal Units:',PmD_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12),legend.position="none")
plt <- plt + geom_hline(yintercept=0,size=0.25)
plt <- plt + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F","Result"))

plot(plt)
graphics.off()



#diff_labels <-  c("Result","Comb","R0/RX","P0/PX")

# #
# png(paste('M1_win_wind_sig1_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,Result=c(M1_diffs$outcome$M1_ac,M1_diffs$outcome$M1_br,M1_diffs$outcome$M1_ar),Comb=c(M1_diffs$comb$M1_ac,M1_diffs$comb$M1_br,M1_diffs$comb$M1_ar),R=c(M1_diffs$r_bin$M1_ac,M1_diffs$r_bin$M1_br,M1_diffs$r_bin$M1_ar),P=c(M1_diffs$p_bin$M1_ac,M1_diffs$p_bin$M1_br,M1_diffs$p_bin$M1_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge())
# plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("Result","Comb","R0/RX","P0/PX"))
# 
# plot(plt)
# graphics.off()
# 
# #
# png(paste('S1_win_wind_sig1_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,Result=c(S1_diffs$outcome$S1_ac,S1_diffs$outcome$S1_br,S1_diffs$outcome$S1_ar),Comb=c(S1_diffs$comb$S1_ac,S1_diffs$comb$S1_br,S1_diffs$comb$S1_ar),R=c(S1_diffs$r_bin$S1_ac,S1_diffs$r_bin$S1_br,S1_diffs$r_bin$S1_ar),P=c(S1_diffs$p_bin$S1_ac,S1_diffs$p_bin$S1_br,S1_diffs$p_bin$S1_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge())
# plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("Result","Comb","R0/RX","P0/PX"))
# 
# plot(plt)
# graphics.off()
# 
# 
# png(paste('PmD_win_wind_sig1_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,Result=c(PmD_diffs$outcome$PmD_ac,PmD_diffs$outcome$PmD_br,PmD_diffs$outcome$PmD_ar),Comb=c(PmD_diffs$comb$PmD_ac,PmD_diffs$comb$PmD_br,PmD_diffs$comb$PmD_ar),R=c(PmD_diffs$r_bin$PmD_ac,PmD_diffs$r_bin$PmD_br,PmD_diffs$r_bin$PmD_ar),P=c(PmD_diffs$p_bin$PmD_ac,PmD_diffs$p_bin$PmD_br,PmD_diffs$p_bin$PmD_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge())
# plt <- plt + theme_classic() + labs(title=paste('PmD\nTotal Units:',PmD_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("Result","Comb","R0/RX","P0/PX"))
# 
# plot(plt)
# graphics.off()
# 
# ################
# 
# diff_labels <-  c("R Levels","P Levels","R Outcome","P Outcome","Comb Outcome")
# 
# png(paste('M1_win_wind_sig2_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,r=c(M1_diffs$r$M1_ac,M1_diffs$r$M1_br,M1_diffs$r$M1_ar),p=c(M1_diffs$p$M1_ac,M1_diffs$p$M1_br,M1_diffs$p$M1_ar),r_outcome=c(M1_diffs$r_outcome$M1_ac,M1_diffs$r_outcome$M1_br,M1_diffs$r_outcome$M1_ar),p_outcome=c(M1_diffs$p_outcome$M1_ac,M1_diffs$p_outcome$M1_br,M1_diffs$p_outcome$M1_ar),comb_outcome=c(M1_diffs$comb_outcome$M1_ac,M1_diffs$comb_outcome$M1_br,M1_diffs$comb_outcome$M1_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge())
# plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.3), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.3), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.3), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4"))  + scale_x_discrete(labels=c("R Levels","P Levels","R Outcome","P Outcome","Comb Outcome"))
# 
# plot(plt)
# graphics.off()
# 
# #
# png(paste('S1_win_wind_sig2_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,r=c(S1_diffs$r$S1_ac,S1_diffs$r$S1_br,S1_diffs$r$S1_ar),p=c(S1_diffs$p$S1_ac,S1_diffs$p$S1_br,S1_diffs$p$S1_ar),r_outcome=c(S1_diffs$r_outcome$S1_ac,S1_diffs$r_outcome$S1_br,S1_diffs$r_outcome$S1_ar),p_outcome=c(S1_diffs$p_outcome$S1_ac,S1_diffs$p_outcome$S1_br,S1_diffs$p_outcome$S1_ar),comb_outcome=c(S1_diffs$comb_outcome$S1_ac,S1_diffs$comb_outcome$S1_br,S1_diffs$comb_outcome$S1_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge())
# plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.3), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.3), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.3), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4"))  + scale_x_discrete(labels=c("R Levels","P Levels","R Outcome","P Outcome","Comb Outcome"))
# 
# plot(plt)
# graphics.off()
# 
# #
# png(paste('PmD_win_wind_sig2_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,r=c(PmD_diffs$r$PmD_ac,PmD_diffs$r$PmD_br,PmD_diffs$r$PmD_ar),p=c(PmD_diffs$p$PmD_ac,PmD_diffs$p$PmD_br,PmD_diffs$p$PmD_ar),r_outcome=c(PmD_diffs$r_outcome$PmD_ac,PmD_diffs$r_outcome$PmD_br,PmD_diffs$r_outcome$PmD_ar),p_outcome=c(PmD_diffs$p_outcome$PmD_ac,PmD_diffs$p_outcome$PmD_br,PmD_diffs$p_outcome$PmD_ar),comb_outcome=c(PmD_diffs$comb_outcome$PmD_ac,PmD_diffs$comb_outcome$PmD_br,PmD_diffs$comb_outcome$PmD_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge())
# plt <- plt + theme_classic() + labs(title=paste('PmD\nTotal Units:',PmD_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.3), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.3), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.3), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4"))  + scale_x_discrete(labels=c("R Levels","P Levels","R Outcome","P Outcome","Comb Outcome"))
# 
# plot(plt)
# graphics.off()
# 
# 
# ###########
# diff_labels <-  c("R0/RX","P0/PX","Outcome")
# 
# #
# png(paste('M1_win_wind_sig3_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,R=c(M1_diffs$r_bin$M1_ac,M1_diffs$r_bin$M1_br,M1_diffs$r_bin$M1_ar),P=c(M1_diffs$p_bin$M1_ac,M1_diffs$p_bin$M1_br,M1_diffs$p_bin$M1_ar),Result=c(M1_diffs$outcome$M1_ac,M1_diffs$outcome$M1_br,M1_diffs$outcome$M1_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
# plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","P0/PX","Result"))
# plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))
# 
# plot(plt)
# graphics.off()
# 
# #
# png(paste('S1_win_wind_sig3_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,R=c(S1_diffs$r_bin$S1_ac,S1_diffs$r_bin$S1_br,S1_diffs$r_bin$S1_ar),P=c(S1_diffs$p_bin$S1_ac,S1_diffs$p_bin$S1_br,S1_diffs$p_bin$S1_ar),Result=c(S1_diffs$outcome$S1_ac,S1_diffs$outcome$S1_br,S1_diffs$outcome$S1_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
# plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","P0/PX","Result"))
# plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))
# 
# plot(plt)
# graphics.off()
# 
# 
# png(paste('PmD_win_wind_sig3_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,R=c(PmD_diffs$r_bin$PmD_ac,PmD_diffs$r_bin$PmD_br,PmD_diffs$r_bin$PmD_ar),P=c(PmD_diffs$p_bin$PmD_ac,PmD_diffs$p_bin$PmD_br,PmD_diffs$p_bin$PmD_ar),Result=c(PmD_diffs$outcome$PmD_ac,PmD_diffs$outcome$PmD_br,PmD_diffs$outcome$PmD_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
# plt <- plt + theme_classic() + labs(title=paste('PmD\nTotal Units:',PmD_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","P0/PX","Result"))
# plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))
# 
# plot(plt)
# graphics.off()
# 
# #######
# diff_labels <-  c("P0/PX")
# 
# #
# png(paste('M1_win_wind_sig4_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,P=c(M1_diffs$p_bin$M1_ac,M1_diffs$p_bin$M1_br,M1_diffs$p_bin$M1_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
# plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("P0/PX"))
# plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=16),axis.title=element_text(size=12),axis.text=element_text(size=12))
# 
# plot(plt)
# graphics.off()
# 
# #
# png(paste('S1_win_wind_sig4_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,P=c(S1_diffs$p_bin$S1_ac,S1_diffs$p_bin$S1_br,S1_diffs$p_bin$S1_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
# plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("P0/PX"))
# plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=16),axis.title=element_text(size=12),axis.text=element_text(size=12))
# 
# plot(plt)
# graphics.off()
# 
# 
# png(paste('PmD_win_wind_sig4_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,P=c(PmD_diffs$p_bin$PmD_ac,PmD_diffs$p_bin$PmD_br,PmD_diffs$p_bin$PmD_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
# plt <- plt + theme_classic() + labs(title=paste('PmD\nTotal Units:',PmD_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("P0/PX"))
# plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=16),axis.title=element_text(size=12),axis.text=element_text(size=12))
# 
# plot(plt)
# graphics.off()
# 
# 
# ###############################
# diff_labels <-  c("P0/PX","P0 Fail/PX Fail")
# 
# #
# png(paste('M1_win_wind_sig5_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,P=c(M1_diffs$p_bin$M1_ac,M1_diffs$p_bin$M1_br,M1_diffs$p_bin$M1_ar),P_F=c(M1_diffs$p_f_bin$M1_ac,M1_diffs$p_f_bin$M1_br,M1_diffs$p_f_bin$M1_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
# plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("P0/PX","P0 F/PX F"))
# plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=16),axis.title=element_text(size=12),axis.text=element_text(size=12))
# 
# plot(plt)
# graphics.off()
# 
# #
# png(paste('S1_win_wind_sig5_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,P=c(S1_diffs$p_bin$S1_ac,S1_diffs$p_bin$S1_br,S1_diffs$p_bin$S1_ar),P_F=c(S1_diffs$p_f_bin$S1_ac,S1_diffs$p_f_bin$S1_br,S1_diffs$p_f_bin$S1_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
# plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("P0/PX","P0 F/PX F"))
# plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=16),axis.title=element_text(size=12),axis.text=element_text(size=12))
# 
# plot(plt)
# graphics.off()
# 
# #
# png(paste('PmD_win_wind_sig5_perc.png',sep=""),width=8,height=6,units="in",res=500)
# 
# diffs.df <- data.frame(window_labels=window_labels,P=c(PmD_diffs$p_bin$PmD_ac,PmD_diffs$p_bin$PmD_br,PmD_diffs$p_bin$PmD_ar),P_F=c(PmD_diffs$p_f_bin$PmD_ac,PmD_diffs$p_f_bin$PmD_br,PmD_diffs$p_f_bin$PmD_ar),position=c(1,2,3))
# diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))
# 
# plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
# plt <- plt + theme_classic() + labs(title=paste('PmD\nTotal Units:',PmD_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
# plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("P0/PX","P0 F/PX F"))
# plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=16),axis.title=element_text(size=12),axis.text=element_text(size=12))
# 
# plot(plt)
# graphics.off()

#######
# diff_labels <-  c("R0/RX","P0/PX","Outcome","R0 S/RX S","P0 F/PX F")
# 
# #
png(paste('M1_win_wind_sig6_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(M1_diffs$r_bin$M1_ac,M1_diffs$r_bin$M1_br,M1_diffs$r_bin$M1_ar),P=c(M1_diffs$p_bin$M1_ac,M1_diffs$p_bin$M1_br,M1_diffs$p_bin$M1_ar),Result=c(M1_diffs$outcome$M1_ac,M1_diffs$outcome$M1_br,M1_diffs$outcome$M1_ar),R_S=c(M1_diffs$r_s_bin$M1_ac,M1_diffs$r_s_bin$M1_br,M1_diffs$r_s_bin$M1_ar),P_F=c(M1_diffs$p_f_bin$M1_ac,M1_diffs$p_f_bin$M1_br,M1_diffs$p_f_bin$M1_ar),position=c(1,2,3))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","P0/PX","Result","R0 S/RX S","P0 F/PX F"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

#
png(paste('S1_win_wind_sig6_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(S1_diffs$r_bin$S1_ac,S1_diffs$r_bin$S1_br,S1_diffs$r_bin$S1_ar),P=c(S1_diffs$p_bin$S1_ac,S1_diffs$p_bin$S1_br,S1_diffs$p_bin$S1_ar),Result=c(S1_diffs$outcome$S1_ac,S1_diffs$outcome$S1_br,S1_diffs$outcome$S1_ar),R_S=c(S1_diffs$r_s_bin$S1_ac,S1_diffs$r_s_bin$S1_br,S1_diffs$r_s_bin$S1_ar),P_F=c(S1_diffs$p_f_bin$S1_ac,S1_diffs$p_f_bin$S1_br,S1_diffs$p_f_bin$S1_ar),position=c(1,2,3))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","P0/PX","Result","R0 S/RX S","P0 F/PX F"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

#
png(paste('PmD_win_wind_sig6_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(PmD_diffs$r_bin$PmD_ac,PmD_diffs$r_bin$PmD_br,PmD_diffs$r_bin$PmD_ar),P=c(PmD_diffs$p_bin$PmD_ac,PmD_diffs$p_bin$PmD_br,PmD_diffs$p_bin$PmD_ar),Result=c(PmD_diffs$outcome$PmD_ac,PmD_diffs$outcome$PmD_br,PmD_diffs$outcome$PmD_ar),R_S=c(PmD_diffs$r_s_bin$PmD_ac,PmD_diffs$r_s_bin$PmD_br,PmD_diffs$r_s_bin$PmD_ar),P_F=c(PmD_diffs$p_f_bin$PmD_ac,PmD_diffs$p_f_bin$PmD_br,PmD_diffs$p_f_bin$PmD_ar),position=c(1,2,3))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('PMd\nTotal Units:',PmD_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","P0/PX","Result","R0 S/RX S","P0 F/PX F"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

################
diff_labels <-  c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F","Outcome")

#
png(paste('M1_win_wind_sig7_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(M1_diffs$r_bin$M1_ac,M1_diffs$r_bin$M1_br,M1_diffs$r_bin$M1_ar),R_S=c(M1_diffs$r_s_bin$M1_ac,M1_diffs$r_s_bin$M1_br,M1_diffs$r_s_bin$M1_ar),P=c(M1_diffs$p_bin$M1_ac,M1_diffs$p_bin$M1_br,M1_diffs$p_bin$M1_ar),P_F=c(M1_diffs$p_f_bin$M1_ac,M1_diffs$p_f_bin$M1_br,M1_diffs$p_f_bin$M1_ar),Result=c(M1_diffs$outcome$M1_ac,M1_diffs$outcome$M1_br,M1_diffs$outcome$M1_ar),position=c(1,2,3))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F","Result"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

#
png(paste('S1_win_wind_sig7_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(S1_diffs$r_bin$S1_ac,S1_diffs$r_bin$S1_br,S1_diffs$r_bin$S1_ar),R_S=c(S1_diffs$r_s_bin$S1_ac,S1_diffs$r_s_bin$S1_br,S1_diffs$r_s_bin$S1_ar),P=c(S1_diffs$p_bin$S1_ac,S1_diffs$p_bin$S1_br,S1_diffs$p_bin$S1_ar),P_F=c(S1_diffs$p_f_bin$S1_ac,S1_diffs$p_f_bin$S1_br,S1_diffs$p_f_bin$S1_ar),Result=c(S1_diffs$outcome$S1_ac,S1_diffs$outcome$S1_br,S1_diffs$outcome$S1_ar),position=c(1,2,3))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F","Result"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

#
png(paste('PmD_win_wind_sig7_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(PmD_diffs$r_bin$PmD_ac,PmD_diffs$r_bin$PmD_br,PmD_diffs$r_bin$PmD_ar),R_S=c(PmD_diffs$r_s_bin$PmD_ac,PmD_diffs$r_s_bin$PmD_br,PmD_diffs$r_s_bin$PmD_ar),P=c(PmD_diffs$p_bin$PmD_ac,PmD_diffs$p_bin$PmD_br,PmD_diffs$p_bin$PmD_ar),P_F=c(PmD_diffs$p_f_bin$PmD_ac,PmD_diffs$p_f_bin$PmD_br,PmD_diffs$p_f_bin$PmD_ar),Result=c(PmD_diffs$outcome$PmD_ac,PmD_diffs$outcome$PmD_br,PmD_diffs$outcome$PmD_ar),position=c(1,2,3))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('PMd\nTotal Units:',PmD_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F","Result"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

##########
diff_labels <-  c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F")

#
png(paste('M1_win_wind_sig8_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(M1_diffs$r_bin$M1_ac,M1_diffs$r_bin$M1_br,M1_diffs$r_bin$M1_ar),R_S=c(M1_diffs$r_s_bin$M1_ac,M1_diffs$r_s_bin$M1_br,M1_diffs$r_s_bin$M1_ar),P=c(M1_diffs$p_bin$M1_ac,M1_diffs$p_bin$M1_br,M1_diffs$p_bin$M1_ar),P_F=c(M1_diffs$p_f_bin$M1_ac,M1_diffs$p_f_bin$M1_br,M1_diffs$p_f_bin$M1_ar),position=c(1,2,3))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

#
png(paste('S1_win_wind_sig8_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(S1_diffs$r_bin$S1_ac,S1_diffs$r_bin$S1_br,S1_diffs$r_bin$S1_ar),R_S=c(S1_diffs$r_s_bin$S1_ac,S1_diffs$r_s_bin$S1_br,S1_diffs$r_s_bin$S1_ar),P=c(S1_diffs$p_bin$S1_ac,S1_diffs$p_bin$S1_br,S1_diffs$p_bin$S1_ar),P_F=c(S1_diffs$p_f_bin$S1_ac,S1_diffs$p_f_bin$S1_br,S1_diffs$p_f_bin$S1_ar),position=c(1,2,3))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

#
png(paste('PmD_win_wind_sig8_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(PmD_diffs$r_bin$PmD_ac,PmD_diffs$r_bin$PmD_br,PmD_diffs$r_bin$PmD_ar),R_S=c(PmD_diffs$r_s_bin$PmD_ac,PmD_diffs$r_s_bin$PmD_br,PmD_diffs$r_s_bin$PmD_ar),P=c(PmD_diffs$p_bin$PmD_ac,PmD_diffs$p_bin$PmD_br,PmD_diffs$p_bin$PmD_ar),P_F=c(PmD_diffs$p_f_bin$PmD_ac,PmD_diffs$p_f_bin$PmD_br,PmD_diffs$p_f_bin$PmD_ar),position=c(1,2,3))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('PMd\nTotal Units:',PmD_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()


##########
window_labels <- c("Cue","Feedback")
diff_labels <-  c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F")

#
png(paste('M1_win_wind_sig9_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(M1_diffs$r_bin$M1_ac,M1_diffs$r_bin$M1_ar),R_S=c(M1_diffs$r_s_bin$M1_ac,M1_diffs$r_s_bin$M1_ar),P=c(M1_diffs$p_bin$M1_ac,M1_diffs$p_bin$M1_ar),P_F=c(M1_diffs$p_f_bin$M1_ac,M1_diffs$p_f_bin$M1_ar),position=c(1,2))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray")) + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

#
png(paste('S1_win_wind_sig9_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(S1_diffs$r_bin$S1_ac,S1_diffs$r_bin$S1_ar),R_S=c(S1_diffs$r_s_bin$S1_ac,S1_diffs$r_s_bin$S1_ar),P=c(S1_diffs$p_bin$S1_ac,S1_diffs$p_bin$S1_ar),P_F=c(S1_diffs$p_f_bin$S1_ac,S1_diffs$p_f_bin$S1_ar),position=c(1,2))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray")) + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

#
png(paste('PmD_win_wind_sig9_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(PmD_diffs$r_bin$PmD_ac,PmD_diffs$r_bin$PmD_ar),R_S=c(PmD_diffs$r_s_bin$PmD_ac,PmD_diffs$r_s_bin$PmD_ar),P=c(PmD_diffs$p_bin$PmD_ac,PmD_diffs$p_bin$PmD_ar),P_F=c(PmD_diffs$p_f_bin$PmD_ac,PmD_diffs$p_f_bin$PmD_ar),position=c(1,2))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('PMd\nTotal Units:',PmD_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray")) + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

###########
window_labels <- c("Cue","Feedback")
diff_labels <-  c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F","Outcome")

#
png(paste('M1_win_wind_sig10_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(M1_diffs$r_bin$M1_ac,M1_diffs$r_bin$M1_ar),R_S=c(M1_diffs$r_s_bin$M1_ac,M1_diffs$r_s_bin$M1_ar),P=c(M1_diffs$p_bin$M1_ac,M1_diffs$p_bin$M1_ar),P_F=c(M1_diffs$p_f_bin$M1_ac,M1_diffs$p_f_bin$M1_ar),Result=c(M1_diffs$outcome$M1_ac,M1_diffs$outcome$M1_ar),position=c(1,2))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray")) + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F","Result"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

#
png(paste('S1_win_wind_sig10_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(S1_diffs$r_bin$S1_ac,S1_diffs$r_bin$S1_ar),R_S=c(S1_diffs$r_s_bin$S1_ac,S1_diffs$r_s_bin$S1_ar),P=c(S1_diffs$p_bin$S1_ac,S1_diffs$p_bin$S1_ar),P_F=c(S1_diffs$p_f_bin$S1_ac,S1_diffs$p_f_bin$S1_ar),Result=c(S1_diffs$outcome$S1_ac,S1_diffs$outcome$S1_ar),position=c(1,2))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray")) + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F","Result"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

#
png(paste('PmD_win_wind_sig10_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(PmD_diffs$r_bin$PmD_ac,PmD_diffs$r_bin$PmD_ar),R_S=c(PmD_diffs$r_s_bin$PmD_ac,PmD_diffs$r_s_bin$PmD_ar),P=c(PmD_diffs$p_bin$PmD_ac,PmD_diffs$p_bin$PmD_ar),P_F=c(PmD_diffs$p_f_bin$PmD_ac,PmD_diffs$p_f_bin$PmD_ar),Result=c(PmD_diffs$outcome$PmD_ac,PmD_diffs$outcome$PmD_ar),position=c(1,2))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('PMd\nTotal Units:',PmD_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray")) + scale_x_discrete(labels=c("R0/RX","R0 S/RX S","P0/PX","P0 F/PX F","Result"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

###########
window_labels <- c("Cue","Feedback")
diff_labels <-  c("R0/RX","P0/PX","Outcome")

#
png(paste('M1_win_wind_sig11_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(M1_diffs$r_bin$M1_ac,M1_diffs$r_bin$M1_ar),P=c(M1_diffs$p_bin$M1_ac,M1_diffs$p_bin$M1_ar),Result=c(M1_diffs$outcome$M1_ac,M1_diffs$outcome$M1_ar),position=c(1,2))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","P0/PX","Result"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

#
png(paste('S1_win_wind_sig11_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(S1_diffs$r_bin$S1_ac,S1_diffs$r_bin$S1_ar),P=c(S1_diffs$p_bin$S1_ac,S1_diffs$p_bin$S1_ar),Result=c(S1_diffs$outcome$S1_ac,S1_diffs$outcome$S1_ar),position=c(1,2))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","P0/PX","Result"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()

#
png(paste('PmD_win_wind_sig11_perc.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,R=c(PmD_diffs$r_bin$PmD_ac,PmD_diffs$r_bin$PmD_ar),P=c(PmD_diffs$p_bin$PmD_ac,PmD_diffs$p_bin$PmD_ar),Result=c(PmD_diffs$outcome$PmD_ac,PmD_diffs$outcome$PmD_ar),position=c(1,2))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge(),show.legend=F)
plt <- plt + theme_classic() + labs(title=paste('PMd\nTotal Units:',M1_total_unit_num),y='Percent Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4")) + scale_x_discrete(labels=c("R0/RX","P0/PX","Result"))
plt <- plt + theme(panel.grid.major.y=element_line(color='lightgrey',size=0.25),plot.title=element_text(size=rel(2)),axis.title=element_blank(),axis.text=element_text(size=12))

plot(plt)
graphics.off()


