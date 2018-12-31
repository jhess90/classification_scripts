rm(list=ls())

library(data.table)

attach('summary.RData')

M1_perc_diff_ac <- list()
for(name in names(M1_diffs_length_list_ac_total)){
  M1_perc_diff_ac[[name]] <- M1_diffs_length_list_ac_total[[name]] / M1_total_unit_num
}
M1_perc_diff_br <- list()
for(name in names(M1_diffs_length_list_br_total)){
  M1_perc_diff_br[[name]] <- M1_diffs_length_list_br_total[[name]] / M1_total_unit_num
}
M1_perc_diff_ar <- list()
for(name in names(M1_diffs_length_list_ar_total)){
  M1_perc_diff_ar[[name]] <- M1_diffs_length_list_ar_total[[name]] / M1_total_unit_num
}  
M1_diffs <- as.data.frame(rbind("M1_ac"=M1_perc_diff_ac,"M1_br" =M1_perc_diff_br,"M1_ar" =M1_perc_diff_ar))

S1_perc_diff_ac <- list()
for(name in names(S1_diffs_length_list_ac_total)){
  S1_perc_diff_ac[[name]] <- S1_diffs_length_list_ac_total[[name]] / S1_total_unit_num
}
S1_perc_diff_br <- list()
for(name in names(S1_diffs_length_list_br_total)){
  S1_perc_diff_br[[name]] <- S1_diffs_length_list_br_total[[name]] / S1_total_unit_num
}
S1_perc_diff_ar <- list()
for(name in names(S1_diffs_length_list_ar_total)){
  S1_perc_diff_ar[[name]] <- S1_diffs_length_list_ar_total[[name]] / S1_total_unit_num
}  
S1_diffs <- as.data.frame(rbind("S1_ac"=S1_perc_diff_ac,"S1_br" =S1_perc_diff_br,"S1_ar" =S1_perc_diff_ar))


PmD_perc_diff_ac <- list()
for(name in names(PmD_diffs_length_list_ac_total)){
  PmD_perc_diff_ac[[name]] <- PmD_diffs_length_list_ac_total[[name]] / PmD_total_unit_num
}
PmD_perc_diff_br <- list()
for(name in names(PmD_diffs_length_list_br_total)){
  PmD_perc_diff_br[[name]] <- PmD_diffs_length_list_br_total[[name]] / PmD_total_unit_num
}
PmD_perc_diff_ar <- list()
for(name in names(PmD_diffs_length_list_ar_total)){
  PmD_perc_diff_ar[[name]] <- PmD_diffs_length_list_ar_total[[name]] / PmD_total_unit_num
}  
PmD_diffs <- as.data.frame(rbind("PmD_ac"=PmD_perc_diff_ac,"PmD_br" =PmD_perc_diff_br,"PmD_ar" =PmD_perc_diff_ar))



fwrite(M1_diffs,file='perc_diff_within_win.csv',append=T,row.names=T)
fwrite(S1_diffs,file='perc_diff_within_win.csv',append=T,row.names=T)
fwrite(PmD_diffs,file='perc_diff_within_win.csv',append=T,row.names=T)


window_labels <-  c("AC","BR","AR")
diff_labels <-  c("Result","Comb","R0/RX","P0/PX")

#
png(paste('M1_win_wind_sig.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,Result=c(M1_diffs$outcome$M1_ac,M1_diffs$outcome$M1_br,M1_diffs$outcome$M1_ar),Comb=c(M1_diffs$comb$M1_ac,M1_diffs$comb$M1_br,M1_diffs$comb$M1_ar),R=c(M1_diffs$r_bin$M1_ac,M1_diffs$r_bin$M1_br,M1_diffs$r_bin$M1_ar),P=c(M1_diffs$p_bin$M1_ac,M1_diffs$p_bin$M1_br,M1_diffs$p_bin$M1_ar),position=c(1,2,3))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge())
plt <- plt + theme_classic() + labs(title=paste('M1\nTotal Units:',M1_total_unit_num),y='Proportion Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4"))

plot(plt)
graphics.off()

#
png(paste('S1_win_wind_sig.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,Result=c(S1_diffs$outcome$S1_ac,S1_diffs$outcome$S1_br,S1_diffs$outcome$S1_ar),Comb=c(S1_diffs$comb$S1_ac,S1_diffs$comb$S1_br,S1_diffs$comb$S1_ar),R=c(S1_diffs$r_bin$S1_ac,S1_diffs$r_bin$S1_br,S1_diffs$r_bin$S1_ar),P=c(S1_diffs$p_bin$S1_ac,S1_diffs$p_bin$S1_br,S1_diffs$p_bin$S1_ar),position=c(1,2,3))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge())
plt <- plt + theme_classic() + labs(title=paste('S1\nTotal Units:',S1_total_unit_num),y='Proportion Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4"))

plot(plt)
graphics.off()


png(paste('PmD_win_wind_sig.png',sep=""),width=8,height=6,units="in",res=500)

diffs.df <- data.frame(window_labels=window_labels,Result=c(PmD_diffs$outcome$PmD_ac,PmD_diffs$outcome$PmD_br,PmD_diffs$outcome$PmD_ar),Comb=c(PmD_diffs$comb$PmD_ac,PmD_diffs$comb$PmD_br,PmD_diffs$comb$PmD_ar),R=c(PmD_diffs$r_bin$PmD_ac,PmD_diffs$r_bin$PmD_br,PmD_diffs$r_bin$PmD_ar),P=c(PmD_diffs$p_bin$PmD_ac,PmD_diffs$p_bin$PmD_br,PmD_diffs$p_bin$PmD_ar),position=c(1,2,3))
diffs.m <-melt(diffs.df,id.vars=c('window_labels','position'))

plt <- ggplot(diffs.m,aes(variable,value)) + geom_bar(aes(fill=window_labels,group=position),stat='identity',position=position_dodge())
plt <- plt + theme_classic() + labs(title=paste('PmD\nTotal Units:',PmD_total_unit_num),y='Proportion Significant',x='',fill='Time Window') + theme(axis.text.x = element_text(size = rel(1.6), angle = 00)) + theme(axis.text.y = element_text(size = rel(1.6), angle = 00)) + theme(axis.title.y = element_text(size = rel(1.6), angle = 90))
plt <- plt + scale_fill_manual(values=c("darkslategray2","darkslategray","darkslategray4"))

plot(plt)
graphics.off()


 