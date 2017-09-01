library(grid)
library(png)
library(gridExtra)
library(magick)

avg_bool = TRUE

combine_fnct <- function(name,files){
  #files <- Sys.glob("M1_all_r*.png")
  cat("\n",name)

  img_1 <- image_read(files[1])
  img_2 <- image_read(files[2])
  img_3 <- image_read(files[3])
  img_4 <- image_read(files[4])
  
  row1 <- c(img_1,img_2)
  row2 <- c(img_3,img_4)
  
  row1_img <- image_append(image_scale(row1)) #,"2400"))
  row2_img <- image_append(image_scale(row2)) #,"800"))
  
  new_whole <- c(row1_img,row2_img)
  new_whole_img <- image_append(image_scale(new_whole),stack=T)

  image_background(new_whole_img,"white",flatten=T)
  image_write(new_whole_img,path=paste(name,".png",sep=""),format="png")
  
return()
}

combine3_fnct <- function(name,files){
  #files <- Sys.glob("M1_all_r*.png")
  cat("\n",name)
  
  img_1 <- image_read(files[1])
  img_2 <- image_read(files[2])
  img_3 <- image_read(files[3])

  row <- c(img_1,img_2,img_3)

  row_img <- image_append(image_scale(row)) #,"2400"))

  image_background(row_img,"white",flatten=T)
  image_write(row_img,path=paste(name,".png",sep=""),format="png")
  
  return()
}



region_list <- c('M1','S1','PmD')
for (i in 1:length(region_list)){
  region_key=region_list[i]
  
  if (avg_bool){
    hist_all_files <- c(paste('slope_hist_all_',region_key,'_bfr_cue.png',sep=''),paste('slope_hist_all_',region_key,'_bfr_result.png',sep=''),paste('slope_hist_all_',region_key,'_aft_cue.png',sep=''),paste('slope_hist_all_',region_key,'_aft_result.png',sep=''))
    hist_avg_files <- c(paste('slope_hist_avg_',region_key,'_bfr_cue.png',sep=''),paste('slope_hist_avg_',region_key,'_bfr_result.png',sep=''),paste('slope_hist_avg_',region_key,'_aft_cue.png',sep=''),paste('slope_hist_avg_',region_key,'_aft_result.png',sep=''))
    
    combine_fnct(paste('slope_hist_all_',region_key,sep=""),hist_all_files)
    combine_fnct(paste('slope_hist_avg_',region_key,sep=""),hist_avg_files)
    
  }else{
    hist_files <-c(paste('slope_hist_',region_key,'_bfr_cue.png',sep=''),paste('slope_hist_',region_key,'_bfr_result.png',sep=''),paste('slope_hist_',region_key,'_aft_cue.png',sep=''),paste('slope_hist_',region_key,'_aft_result.png',sep=''))
    combine_fnct(paste('slope_hist_',region_key,sep=""),hist_files)
    
  }
}


if (avg_bool){
  
  avg_plots_list <- c('M1_avg_slopes.png','S1_avg_slopes.png','PmD_avg_slopes.png')
  
  avg_abs_plots_list <- c('M1_avg_abs_slopes.png','S1_avg_abs_slopes.png','PmD_avg_abs_slopes.png')
  
  avg_avg_plots_list <- c('M1_avg_avg_slopes.png','S1_avg_avg_slopes.png','PmD_avg_avg_slopes.png')
  
  avg_avg_plots_list <- c('M1_avg_avg_slopes.png','S1_avg_avg_slopes.png','PmD_avg_avg_slopes.png')
  
  
  sig_plots_list <- c('sig_plotsM1.png','sig_plotsS1.png','sig_plotsPmD.png')
  
  signs_bar_plotted_all_list <- c('signs_bar_plotted_all_M1.png','signs_bar_plotted_all_S1.png','signs_bar_plotted_all_PmD.png')
  
  signs_bar_plotted_avg_list <- c('signs_bar_plotted_avg_M1.png','signs_bar_plotted_avg_S1.png','signs_bar_plotted_avg_PmD.png')
  
  signs_bar_plotted_list <- c('signs_bar_plotted_M1.png','signs_bar_plotted_S1.png','signs_bar_plotted_PmD.png')
  
  
  
}else{
  sig_plots_list <- c('sig_plotsM1.png','sig_plotsS1.png','sig_plotsPmD.png')
  
  signs_bar_plotted_list <- c('signs_bar_plotted_M1.png','signs_bar_plotted_S1.png','signs_bar_plotted_PmD.png')
  
}


