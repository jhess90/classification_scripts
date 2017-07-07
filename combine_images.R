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
  img_5 <- image_read(files[5])
  img_6 <- image_read(files[6])
  img_7 <- image_read(files[7])
  img_8 <- image_read(files[8])
  
  row1 <- c(img_1,img_2)
  row2 <- c(img_3,img_4)
  row3 <- c(img_5,img_6)
  row4 <- c(img_7,img_8)
            
  row1_img <- image_append(image_scale(row1))
  row2_img <- image_append(image_scale(row2))
  row3_img <- image_append(image_scale(row3))
  row4_img <- image_append(image_scale(row4))
  
  whole <- c(row1_img,row2_img,row3_img,row4_img)
  whole_img <- image_append(image_scale(whole),stack=T)
  image_background(whole_img,"white",flatten=T)
  image_write(whole,path=paste(name,".png",sep=""),format="png")
return()
}

region_list <- c('M1','S1','PmD')

for (i in 1:length(region_list)){
  region_key=region_list[i]
  
  name1 <- paste(region_key,"_all_r",sep="")
  files1 <- c(paste(region_key,"_all_r_succ_cue.png",sep=""),paste(region_key,"_all_r_fail_cue.png",sep=""),"ra_succ_cue_gf.png","ra_fail_cue_gf.png",paste(region_key,"_all_r_succ_result.png",sep=""),paste(region_key,"_all_r_fail_result.png",sep=""),"ra_succ_result_gf.png","ra_fail_resul_gf.png")
  combine_fnct(name1,files1)
  
  name2 <- paste(region_key,"_all_p",sep="")
  files2 <- c(paste(region_key,"_all_p_succ_cue.png",sep=""),paste(region_key,"_all_p_fail_cue.png",sep=""),"pa_succ_cue_gf.png","pa_fail_cue_gf.png",paste(region_key,"_all_p_succ_result.png",sep=""),paste(region_key,"_all_p_fail_result.png",sep=""),"pa_succ_result_gf.png","pa_fail_resul_gf.png")
  combine_fnct(name2,files2)
  
  name3 <- paste(region_key,"_no_r",sep="")
  files3 <- c(paste(region_key,"_no_r_succ_cue.png",sep=""),paste(region_key,"_no_r_fail_cue.png",sep=""),"r0_succ_cue_gf.png","r0_fail_cue_gf.png",paste(region_key,"_no_r_succ_result.png",sep=""),paste(region_key,"_no_r_fail_result.png",sep=""),"r0_succ_result_gf.png","r0_fail_resul_gf.png")
  combine_fnct(name3,files3)
  
  name4 <- paste(region_key,"_no_p",sep="")
  files4 <- c(paste(region_key,"_no_p_succ_cue.png",sep=""),paste(region_key,"_no_p_fail_cue.png",sep=""),"p0_succ_cue_gf.png","p0_fail_cue_gf.png",paste(region_key,"_no_p_succ_result.png",sep=""),paste(region_key,"_no_p_fail_result.png",sep=""),"p0_succ_result_gf.png","p0_fail_resul_gf.png")
  combine_fnct(name4,files4)
  
  name5 <- paste(region_key,"_catch",sep="")
  files5 <- c(paste(region_key,"_r_all_catch_cue.png",sep=""),paste(region_key,"_p_all_catch_cue.png",sep=""),"r_all_catch_cue_gf.png","p_all_catch_cue_gf.png",paste(region_key,"_r_all_catch_result.png",sep=""),paste(region_key,"_p_all_catch_result.png",sep=""),"r_all_catch_result_gf.png","p_all_catch_resul_gf.png")
  combine_fnct(name5,files5)
  
}




###### NEW hopefully faster #####
#make heatmap
#grid.echo()
#grid.grab()
#grid.newpage()
#grid.arrange(heatmap1,heatmap2,ncol=1)

