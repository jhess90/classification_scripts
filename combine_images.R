library(grid)
library(png)
library(gridExtra)
library(magick)


combine_fnct <- function(name,files){

#files <- Sys.glob("M1_all_r*.png")
cat("\n",name)

img_1 <- image_read(files[1])
img_2 <- image_read(files[2])
img_3 <- image_read(files[3])
img_4 <- image_read(files[4])

top_row <- c(img_1,img_2)
bot_row <- c(img_3,img_4)

top_half <- image_append(image_scale(top_row))
bot_half <- image_append(image_scale(bot_row))
#image_background(top_half,"white",flatten=T)

#png("test.png",width=8,height=6,units="in",res=500)
both_halves <- c(top_half,bot_half)
whole <- image_append(image_scale(both_halves),stack=T)
image_background(whole,"white",flatten=T)

image_write(whole,path=paste(name,".png",sep=""),format="png")
return()
}

region_list <- c('M1','S1','PmD')

for (i in 1:length(region_list)){
  region_key=region_list[i]
  
  name1 <- paste(region_key,"_all_r",sep="")
  files1 <- c(paste(region_key,"_all_r_succ_cue.png",sep=""),paste(region_key,"_all_r_fail_cue.png",sep=""),paste(region_key,"_all_r_succ_result.png",sep=""),paste(region_key,"_all_r_fail_result.png",sep=""))
  combine_fnct(name1,files1)
  
  name2 <- paste(region_key,"_all_p",sep="")
  files2 <- c(paste(region_key,"_all_p_succ_cue.png",sep=""),paste(region_key,"_all_p_fail_cue.png",sep=""),paste(region_key,"_all_p_succ_result.png",sep=""),paste(region_key,"_all_p_fail_result.png",sep=""))
  combine_fnct(name2,files2)
  
  name3 <- paste(region_key,"_no_r",sep="")
  files3 <- c(paste(region_key,"_no_r_succ_cue.png",sep=""),paste(region_key,"_no_r_fail_cue.png",sep=""),paste(region_key,"_no_r_succ_result.png",sep=""),paste(region_key,"_no_r_fail_result.png",sep=""))
  combine_fnct(name3,files3)
  
  name4 <- paste(region_key,"_no_p",sep="")
  files4 <- c(paste(region_key,"_no_p_succ_cue.png",sep=""),paste(region_key,"_no_p_fail_cue.png",sep=""),paste(region_key,"_no_p_succ_result.png",sep=""),paste(region_key,"_no_p_fail_result.png",sep=""))
  combine_fnct(name4,files4)
  
  name5 <- paste(region_key,"_catch",sep="")
  files5 <- c(paste(region_key,"_r_all_catch_cue.png",sep=""),paste(region_key,"_p_all_catch_cue.png",sep=""),paste(region_key,"_r_all_catch_result.png",sep=""),paste(region_key,"_p_all_catch_result.png",sep=""))
  combine_fnct(name5,files5)
  
}


