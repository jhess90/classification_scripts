rm(list=ls())

library(R.matlab)
#source("~/documents/lab/workspace/Classification_scripts/multiplot.R")
source("~/workspace/classification_scripts/multiplot.R")


#########

file_list <- Sys.glob("block*.RData")

ind = 1
for(block_name in file_list){
  cat(block_name)
  
  if(ind == 1){
    attach(block_name)
    temp <- condensed
    
    condensed_all <- temp
    
    detach()
  }else{
    attach(block_name)
    temp <- condensed
    
    condensed_all <- rbind(condensed_all,temp)
    
    detach() 
  }
  ind <- ind + 1
}

condensed <- condensed_all

r_vector <- condensed[,4]
p_vector <- condensed[,5]

res_vector <- condensed[,6]

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

#png(paste('autocorrelation.png',sep=""),width=8,height=6,units="in",res=500)
#par(mfrow=(c(2,2)))

r_acf <- acf(r_vector,plot=F) 
p_acf <- acf(p_vector,plot=F)
c_acf <- acf(combo,plot=F)
res_acf <- acf(res_vector,plot=F)

#multiplot(r_acf,p_acf,c_acf,res_acf,cols=2)

png('autocorr_reward.png',width=8,height=6,units='in',res=500)
plot(r_acf,main="Reward")
graphics.off()

png('autocorr_punishment.png',width=8,height=6,units='in',res=500)
plot(p_acf,main="Punishment")
graphics.off()

png('autocorr_comb.png',width=8,height=6,units='in',res=500)
plot(c_acf,main="RP Combination")
graphics.off()

png('autocorr_result.png',width=8,height=6,units='in',res=500)
plot(res_acf,main="Result")
graphics.off()



