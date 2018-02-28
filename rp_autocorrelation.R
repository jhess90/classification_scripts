library(R.matlab)
#source("~/documents/lab/workspace/Classification_scripts/multiplot.R")
source("~/workspace/classification_scripts/multiplot.R")


condensed <- readMat('corr_output_M1.mat')$condensed

r_vector <- condensed[,4]
p_vector <- condensed[,5]

#r0p0: 0, r1p0: 1, r2p0: 2, r3p0: 3
#r0p1: 4, r1p1: 5, r2p1: 6, r3p1: 7
#r0p2: 8, r1p2: 9, r2p2: 10, r3p2: 11
#r0p3: 12, r1p3: 13, r2p3: 14, r3p3: 15

combo <- integer(length(r_vector))
for (i in 1:length(r_vector)){
  if (r_vector[i] == 0 & p_vector[i] == 0){combo[i] <- 0
  }else if(r_vector[i] == 1 & p_vector[i] == 0){combo[i] <- 1
  }else if(r_vector[i] == 2 & p_vector[i] == 0){combo[i] <- 2
  }else if(r_vector[i] == 3 & p_vector[i] == 0){combo[i] <- 3
  }else if(r_vector[i] == 0 & p_vector[i] == 1){combo[i] <- 4
  }else if(r_vector[i] == 1 & p_vector[i] == 1){combo[i] <- 5
  }else if(r_vector[i] == 2 & p_vector[i] == 1){combo[i] <- 6
  }else if(r_vector[i] == 3 & p_vector[i] == 1){combo[i] <- 7
  }else if(r_vector[i] == 0 & p_vector[i] == 2){combo[i] <- 8
  }else if(r_vector[i] == 1 & p_vector[i] == 2){combo[i] <- 9
  }else if(r_vector[i] == 2 & p_vector[i] == 2){combo[i] <- 10
  }else if(r_vector[i] == 3 & p_vector[i] == 2){combo[i] <- 11
  }else if(r_vector[i] == 0 & p_vector[i] == 3){combo[i] <- 12
  }else if(r_vector[i] == 1 & p_vector[i] == 3){combo[i] <- 13
  }else if(r_vector[i] == 2 & p_vector[i] == 3){combo[i] <- 14
  }else if(r_vector[i] == 3 & p_vector[i] == 3){combo[i] <- 15
  }else{cat('ERROR')}
}

#############
#plot
#############

png(paste('autocorrelation.png',sep=""),width=8,height=6,units="in",res=500)


par(mfrow=(c(2,2)))

acf(r_vector,plot=T)
acf(p_vector,plot=T)
acf(combo,plot=T)

graphics.off()



