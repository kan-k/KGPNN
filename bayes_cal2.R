#Function for taking in multiple runs' predictions, and calculate MAE, MSE and R^2

#Version 2, should include Gender

if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)
p_load(nimble)
p_load(extraDistr)
p_load(dplyr)
JobId=as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
print(JobId)
set.seed(JobId)

#calculating r^2
rsqcal <- function(true,pred){
  RSS <-sum((true - pred)^2)
  TSS <- sum((true - mean(true))^2)
  return((1 - RSS/TSS)*100)
}

meanstats.re<- function(filename, runs, mala_if, Sex = "all"){
  in.rmse.vec<- vector(mode="numeric")
  in.mae.vec<- vector(mode="numeric")
  in.rsq.vec<- vector(mode="numeric")
  out.rmse.vec<- vector(mode="numeric")
  out.mae.vec<- vector(mode="numeric")
  out.rsq.vec<- vector(mode="numeric")
  in.cover<- vector(mode="numeric")
  in.width<- vector(mode="numeric")
  out.cover<- vector(mode="numeric")
  out.width<- vector(mode="numeric")
  
  in.rmse.vec.female<- vector(mode="numeric")
  in.mae.vec.female<- vector(mode="numeric")
  in.rsq.vec.female<- vector(mode="numeric")
  out.rmse.vec.female<- vector(mode="numeric")
  out.mae.vec.female<- vector(mode="numeric")
  out.rsq.vec.female<- vector(mode="numeric")
  in.cover.female<- vector(mode="numeric")
  in.width.female<- vector(mode="numeric")
  out.cover.female<- vector(mode="numeric")
  out.width.female<- vector(mode="numeric")
  
  in.rmse.vec.male<- vector(mode="numeric")
  in.mae.vec.male<- vector(mode="numeric")
  in.rsq.vec.male<- vector(mode="numeric")
  out.rmse.vec.male<- vector(mode="numeric")
  out.mae.vec.male<- vector(mode="numeric")
  out.rsq.vec.male<- vector(mode="numeric")
  in.cover.male<- vector(mode="numeric")
  in.width.male<- vector(mode="numeric")
  out.cover.male<- vector(mode="numeric")
  out.width.male<- vector(mode="numeric")
  
  
  train.test.ind <- list()
  train.test.ind$test <- read.csv('/well/nichols/users/qcv214/KGPNN/sex_test_index.csv')$x
  train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/KGPNN/sex_train_index.csv')$x
  n.train <- length(train.test.ind$train) #[1:200] only active if it is mala or sgld200
  n.test <- length(train.test.ind$test)
  
  if(Sex == "all"){
    #age_tab<-as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/age_sex.feather'))
    age_tab<-as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/age_sex_strat.feather'))
  } else if(Sex == "m"){
    age_tab <- as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/age_M.feather'))
  } else if(Sex == "f"){
    age_tab <- as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/age_F.feather'))
  } else if(Sex == "old"){
    age_tab<-as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/age_sex.feather'))
    age_tab <- age_tab[order(age_tab$id),]         
    age_tab <- rbind(age_tab[1:3101,], c(24612030,NA,NA),age_tab[3102:nrow(age_tab),] )
  }
  
  #age_tab <- age_tab[order(age_tab$id),]
  age <- age_tab$age
  if(Sex == "all"){
    #age_tab<-as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/age_sex.feather'))
    sex <-  as.numeric(age_tab$sex)
  }
  
  train.true <- age_tab[train.test.ind$train, 2:3]
  true.train<- data.frame(train.test.ind$train,train.true,row.names = NULL)
  colnames(true.train) <- c('id','true','sex')
  
  test.true <- age_tab[train.test.ind$test, 2:3]
  true.test<- data.frame(train.test.ind$test,test.true,row.names = NULL)
  colnames(true.test) <- c('id','true','sex')
  
  num.it <-1200
  # runs <- c(2:4,7:10)
  res.mat <- array(,dim=c(2,length(runs),num.it))
  for(i in runs){
    res.mat[,which(i==runs),] <- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_",filename,"_loss__jobid_",i,".csv")))
  }
  mod.mse <- mean(rowMeans(res.mat[1,,]))
  
  for(i in runs){
    print(paste0("run: ",i))
    dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_inpred__jobid_',i,'.feather'))))
    #print(dim(dat.in))
    #print(head(dat.in))
    if(mala_if){ 
      dat.in <- tail(dat.in,200*1000) #burn in of 1000
    } else {
      dat.in <- tail(dat.in,500*1000) # tail(dat.in,50*1000) only true if it is sgld 200.  #I think this is the _size_ of minibatch (was 500) * _burn_in_ 
    }
    
    colnames(dat.in) <- c('id','pred')
    dat.in.grouped <- dat.in %>%
      group_by(id) %>%
      summarize(mean_pred = mean(pred), lwr_ppi = mean(pred)-1.96*sqrt(mod.mse+var(pred)), upr_ppi = mean(pred)+1.96*sqrt(mod.mse+var(pred)))
    
    joined_data.train <- left_join(true.train, dat.in.grouped, by = "id")
    joined_data.train <- joined_data.train %>%
      mutate(within_interval = ifelse(true>= lwr_ppi & true <= upr_ppi, TRUE, FALSE))
    colnames(joined_data.train) <- c("subject_id", "truth","sex","mean_pred", "lwr_ppi", "upr_ppi", "within_interval")
    
    
    ####Added 18 Aug
    ############For MALA, 200 training
    #joined_data.train <- joined_data.train[1:200,]
    
    
    #Test
    dat.test <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_outpred__jobid_',i,'.feather'))))
    dat.test <- tail(dat.test,n.test*1000) #This is the _size_ of test*burn in, current there are 1200 iterations of SGLD, taking last 1k mean only 200 was burn in
    colnames(dat.test) <- c('id','pred')
    dat.test.grouped <- dat.test %>%
      group_by(id) %>%
      summarize(mean_pred = mean(pred), lwr_ppi = mean(pred)-1.96*sqrt(mod.mse+var(pred)), upr_ppi = mean(pred)+1.96*sqrt(mod.mse+var(pred)))
    
    joined_data.test <- left_join(true.test, dat.test.grouped, by = "id")
    joined_data.test <- joined_data.test %>%
      mutate(within_interval = ifelse(true>= lwr_ppi & true <= upr_ppi, TRUE, FALSE))
    colnames(joined_data.test) <- c("subject_id", "truth",'sex',"mean_pred", "lwr_ppi", "upr_ppi", "within_interval")
    
    # print(paste0(" %Correct interval: ", sum(joined_data.test$within_interval)*100/length(joined_data.test$subject_id), " of Run ",i,", Mean PPI width: ",mean(joined_data.test$upr_ppi - joined_data.test$lwr_ppi)))
    ##Metrics
    in.rmse <- sqrt(mean((joined_data.train$mean_pred-joined_data.train$truth)^2))
    in.mae <- (mean(abs(joined_data.train$mean_pred-joined_data.train$truth)))
    in.rsq <- rsqcal(joined_data.train$truth,joined_data.train$mean_pred)
    
    ##Metrics
    out.rmse <- sqrt(mean((joined_data.test$mean_pred-joined_data.test$truth)^2))
    out.mae <- (mean(abs(joined_data.test$mean_pred-joined_data.test$truth)))
    out.rsq <- rsqcal(joined_data.test$truth,joined_data.test$mean_pred)
    
    #concat result
    in.rmse.vec <- c(in.rmse.vec,in.rmse)
    in.mae.vec <- c(in.mae.vec,in.mae)
    in.rsq.vec <- c(in.rsq.vec,in.rsq)
    
    out.rmse.vec <- c(out.rmse.vec,out.rmse)
    out.mae.vec <- c(out.mae.vec,out.mae)
    out.rsq.vec <- c(out.rsq.vec,out.rsq)
    
    in.cover  <- c(in.cover,sum(joined_data.train$within_interval)*100/length(joined_data.train$subject_id)  )
    in.width <- c(in.width, mean(joined_data.train$upr_ppi - joined_data.train$lwr_ppi))
    out.cover <- c(out.cover,sum(joined_data.test$within_interval)*100/length(joined_data.test$subject_id) )
    out.width <- c(out.width,mean(joined_data.test$upr_ppi - joined_data.test$lwr_ppi) )
    print(paste0("in-rmse: ",in.rmse.vec))

  ########################################################################### Female ##################################################
  

  in.rmse.female <- sqrt(mean((joined_data.train$mean_pred[joined_data.train$sex==0]-joined_data.train$truth[joined_data.train$sex==0])^2))
  in.mae.female <- (mean(abs(joined_data.train$mean_pred[joined_data.train$sex==0]-joined_data.train$truth[joined_data.train$sex==0])))
  in.rsq.female <- rsqcal(joined_data.train$truth[joined_data.train$sex==0],joined_data.train$mean_pred[joined_data.train$sex==0])
  
  ##Metrics
  out.rmse.female <- sqrt(mean((joined_data.test$mean_pred[joined_data.test$sex==0]-joined_data.test$truth[joined_data.test$sex==0])^2))
  out.mae.female <- (mean(abs(joined_data.test$mean_pred[joined_data.test$sex==0]-joined_data.test$truth[joined_data.test$sex==0])))
  out.rsq.female <- rsqcal(joined_data.test$truth[joined_data.test$sex==0],joined_data.test$mean_pred[joined_data.test$sex==0])
  
  #concat result
  in.rmse.vec.female <- c(in.rmse.vec.female,in.rmse.female)
  in.mae.vec.female <- c(in.mae.vec.female,in.mae.female)
  in.rsq.vec.female <- c(in.rsq.vec.female,in.rsq.female)
  
  out.rmse.vec.female <- c(out.rmse.vec.female,out.rmse.female)
  out.mae.vec.female <- c(out.mae.vec.female,out.mae.female)
  out.rsq.vec.female <- c(out.rsq.vec.female,out.rsq.female)
  
  in.cover.female  <- c(in.cover.female,sum(joined_data.train$within_interval[joined_data.train$sex==0])*100/length(joined_data.train$subject_id[joined_data.train$sex==0])  )
  in.width.female <- c(in.width.female, mean(joined_data.train$upr_ppi[joined_data.train$sex==0] - joined_data.train$lwr_ppi[joined_data.train$sex==0]))
  out.cover.female <- c(out.cover.female,sum(joined_data.test$within_interval[joined_data.test$sex==0])*100/length(joined_data.test$subject_id[joined_data.test$sex==0]) )
  out.width.female <- c(out.width.female,mean(joined_data.test$upr_ppi[joined_data.test$sex==0] - joined_data.test$lwr_ppi[joined_data.test$sex==0]) )
  print(paste0("in-rmse-female: ",in.rmse.vec.female))
  #print(paste0("kength rmse female: ", length(in.rmse.vec.female)))
  
  ########################################################################### Male ##################################################

  
  in.rmse.male <- sqrt(mean((joined_data.train$mean_pred[joined_data.train$sex==1]-joined_data.train$truth[joined_data.train$sex==1])^2))
  in.mae.male <- (mean(abs(joined_data.train$mean_pred[joined_data.train$sex==1]-joined_data.train$truth[joined_data.train$sex==1])))
  in.rsq.male <- rsqcal(joined_data.train$truth[joined_data.train$sex==1],joined_data.train$mean_pred[joined_data.train$sex==1])
  
  ##Metrics
  out.rmse.male <- sqrt(mean((joined_data.test$mean_pred[joined_data.test$sex==1]-joined_data.test$truth[joined_data.test$sex==1])^2))
  out.mae.male <- (mean(abs(joined_data.test$mean_pred[joined_data.test$sex==1]-joined_data.test$truth[joined_data.test$sex==1])))
  out.rsq.male <- rsqcal(joined_data.test$truth[joined_data.test$sex==1],joined_data.test$mean_pred[joined_data.test$sex==1])
  
  #concat result
  in.rmse.vec.male <- c(in.rmse.vec.male,in.rmse.male)
  in.mae.vec.male <- c(in.mae.vec.male,in.mae.male)
  in.rsq.vec.male <- c(in.rsq.vec.male,in.rsq.male)
  
  out.rmse.vec.male <- c(out.rmse.vec.male,out.rmse.male)
  out.mae.vec.male <- c(out.mae.vec.male,out.mae.male)
  out.rsq.vec.male <- c(out.rsq.vec.male,out.rsq.male)
  
  in.cover.male  <- c(in.cover.male,sum(joined_data.train$within_interval[joined_data.train$sex==1])*100/length(joined_data.train$subject_id[joined_data.train$sex==1])  )
  in.width.male <- c(in.width.male, mean(joined_data.train$upr_ppi[joined_data.train$sex==1] - joined_data.train$lwr_ppi[joined_data.train$sex==1]))
  out.cover.male <- c(out.cover.male,sum(joined_data.test$within_interval[joined_data.test$sex==1])*100/length(joined_data.test$subject_id[joined_data.test$sex==1]) )
  out.width.male <- c(out.width.male,mean(joined_data.test$upr_ppi[joined_data.test$sex==1] - joined_data.test$lwr_ppi[joined_data.test$sex==1]) )
  print(paste0("in-rmse-male: ",in.rmse.vec.male))
  #print(paste0("length rmse male: ", length(in.rmse.vec.male)))
  
  ########################################################################### 
}
  out <- matrix(,nrow=10,ncol=2)
  out[1,] <- c(median(in.rmse.vec),sd(in.rmse.vec))
  out[9,] <- c(median(in.mae.vec),sd(in.mae.vec))
  out[3,] <- c(median(in.rsq.vec),sd(in.rsq.vec))
  out[2,] <- c(median(out.rmse.vec),sd(out.rmse.vec))
  out[10,] <- c(median(out.mae.vec),sd(out.mae.vec))
  out[4,] <- c(median(out.rsq.vec),sd(out.rsq.vec))
  out[5,] <- c(median(in.cover),sd(in.cover))
  out[6,] <- c(median(out.cover),sd(out.cover))
  out[7,] <- c(median(in.width),sd(in.width))
  out[8,] <- c(median(out.width),sd(out.width))
  
  out <-  as.data.frame(out)
  colnames(out) <- c("median", "sd")
  rownames(out) <- c("inRMSE","outRMSE","inR2","outR2","in-Coverage","out-Coverage","in-PPIwidth","out-PPIwidth","inMAE","outMAE")

  out.female <- matrix(,nrow=10,ncol=2)
  out.female[1,] <- c(median(in.rmse.vec.female),sd(in.rmse.vec.female))
  out.female[9,] <- c(median(in.mae.vec.female),sd(in.mae.vec.female))
  out.female[3,] <- c(median(in.rsq.vec.female),sd(in.rsq.vec.female))
  out.female[2,] <- c(median(out.rmse.vec.female),sd(out.rmse.vec.female))
  out.female[10,] <- c(median(out.mae.vec.female),sd(out.mae.vec.female))
  out.female[4,] <- c(median(out.rsq.vec.female),sd(out.rsq.vec.female))
  out.female[5,] <- c(median(in.cover.female),sd(in.cover.female))
  out.female[6,] <- c(median(out.cover.female),sd(out.cover.female))
  out.female[7,] <- c(median(in.width.female),sd(in.width.female))
  out.female[8,] <- c(median(out.width.female),sd(out.width.female))
  
  out.female <-  as.data.frame(out.female)
  colnames(out.female) <- c("median", "sd")
  rownames(out.female) <- c("inRMSE.female","outRMSE.female","inR2.female","outR2.female","in-Coverage.female","out-Coverage.female","in-PPIwidth.female","out-PPIwidth.female","inMAE.female","outMAE.female")
  
  out.male <- matrix(,nrow=10,ncol=2)
  out.male[1,] <- c(median(in.rmse.vec.male),sd(in.rmse.vec.male))
  out.male[9,] <- c(median(in.mae.vec.male),sd(in.mae.vec.male))
  out.male[3,] <- c(median(in.rsq.vec.male),sd(in.rsq.vec.male))
  out.male[2,] <- c(median(out.rmse.vec.male),sd(out.rmse.vec.male))
  out.male[10,] <- c(median(out.mae.vec.male),sd(out.mae.vec.male))
  out.male[4,] <- c(median(out.rsq.vec.male),sd(out.rsq.vec.male))
  out.male[5,] <- c(median(in.cover.male),sd(in.cover.male))
  out.male[6,] <- c(median(out.cover.male),sd(out.cover.male))
  out.male[7,] <- c(median(in.width.male),sd(in.width.male))
  out.male[8,] <- c(median(out.width.male),sd(out.width.male))
  
  out.male <-  as.data.frame(out.male)
  colnames(out.male) <- c("median", "sd")
  rownames(out.male) <- c("inRMSE.male","outRMSE.male","inR2.male","outR2.male","in-Coverage.male","out-Coverage.male","in-PPIwidth.male","out-PPIwidth.male","inMAE.male","outMAE.male")
  
  return(rbind(out,out.female,out.male))

}

# print("start=========")
# stat.ig <- meanstats.re("apr5_mala_weights_sub_adapt1_opt_V",c(1:10), mala_if=TRUE)
# write.csv(stat.ig,paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_summary_may10_mala_mode.csv"))
#print("=========IG DONE=========")
#stat.eb <-meanstats.re("may7_mala_weights_sub_adapt1_diffinit", c(2:4,7:10), mala_if=FALSE)
# write.csv(stat.eb,paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_summary_may10_mala_near.csv"))
# print("=========IG DONE=========")
# stat.eb <-meanstats.re("apr27_sgld_bb_ig_a5_b0_V_200", c(1:10), mala_if=FALSE)
# write.csv(stat.eb,paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_summary_may10_sgld_near.csv"))

#7 Aug
############## for spiltting sex in data, no longer used
# print("=========IG DONE=========")
# stat.eb <-meanstats.re("sep18_m_sgld_bbig_a6_b0_near", c(1:10), mala_if=FALSE, Sex = "m")
# write.csv(stat.eb,paste0("/well/nichols/users/qcv214/KGPNN/pile/re_summary_sep18_m_a6b0_near.csv"))
# stat.eb <-meanstats.re("sep18_f_sgld_bbig_a6_b0_near", c(1:10), mala_if=FALSE, Sex = "f")
# write.csv(stat.eb,paste0("/well/nichols/users/qcv214/KGPNN/pile/re_summary_sep18_f_a6b0_near.csv"))
# print("=========IG DONE=========")
# stat.eb <-meanstats.re("sep15_m_sgld_bbig_a4_b0_near", c(1:10), mala_if=FALSE, Sex = "m")
# write.csv(stat.eb,paste0("/well/nichols/users/qcv214/KGPNN/pile/re_summary_sep15_m_a4b0_near.csv"))
# stat.eb <-meanstats.re("sep15_f_sgld_bbig_a4_b0_near", c(1:10), mala_if=FALSE, Sex = "f")
# write.csv(stat.eb,paste0("/well/nichols/users/qcv214/KGPNN/pile/re_summary_sep15_f_a4b0_near.csv"))
# 
# print("=========IG DONE=========")
# stat.eb <-meanstats.re("sep18_m_sgld_bbig_a8_b0_near", c(1:10), mala_if=FALSE, Sex = "m")
# write.csv(stat.eb,paste0("/well/nichols/users/qcv214/KGPNN/pile/re_summary_sep18_m_a8b0_near.csv"))
# stat.eb <-meanstats.re("sep18_f_sgld_bbig_a8_b0_near", c(1:10), mala_if=FALSE, Sex = "f")
# write.csv(stat.eb,paste0("/well/nichols/users/qcv214/KGPNN/pile/re_summary_sep18_f_a8b0_near.csv"))

#2 Oct

print("=========IG DONE=========")
stat.eb <-meanstats.re("oct2_gender_gpgp_sgld_a4_b0_near", c(1:7,9:10), mala_if=FALSE)
write.csv(stat.eb,paste0("/well/nichols/users/qcv214/KGPNN/pile/re_summary_oct2_gender_gpgp_sgld_near.csv"))
print("=========IG DONE=========")
stat.eb <-meanstats.re("oct2_nogender_gpgp_sgld_a4_b0_near", c(1:7,9:10), mala_if=FALSE)
write.csv(stat.eb,paste0("/well/nichols/users/qcv214/KGPNN/pile/re_summary_oct2_nogender_gpgp_sgld_near.csv"))



