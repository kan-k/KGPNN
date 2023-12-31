#Function for taking in multiple runs' predictions, and calculate MAE, MSE and R^2
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
  
  
  train.test.ind <- list()
  train.test.ind$test <- read.csv('/well/nichols/users/qcv214/KGPNN/test_index.csv')$x
  train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/KGPNN/train_index.csv')$x
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
  
  train.true <- age[train.test.ind$train]
  true.train<- data.frame(train.test.ind$train,train.true,row.names = NULL)
  colnames(true.train) <- c('id','true')
  
  test.true <- age[train.test.ind$test]
  true.test<- data.frame(train.test.ind$test,test.true,row.names = NULL)
  colnames(true.test) <- c('id','true')
  
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
    print(dim(dat.in))
    print(head(dat.in))
    if(mala_if){ 
      dat.in <- tail(dat.in,200*1000) #burn in of 1000
    } else {
      dat.in <- tail(dat.in,350*1000) # tail(dat.in,50*1000) only true if it is sgld 200.  #I think this is the _size_ of minibatch (was 500) * _burn_in_ 
    }
    
    colnames(dat.in) <- c('id','pred')
    dat.in.grouped <- dat.in %>%
      group_by(id) %>%
      summarize(mean_pred = mean(pred), lwr_ppi = mean(pred)-1.96*sqrt(mod.mse+var(pred)), upr_ppi = mean(pred)+1.96*sqrt(mod.mse+var(pred)))
    
    joined_data.train <- left_join(true.train, dat.in.grouped, by = "id")
    joined_data.train <- joined_data.train %>%
      mutate(within_interval = ifelse(true>= lwr_ppi & true <= upr_ppi, TRUE, FALSE))
    colnames(joined_data.train) <- c("subject_id", "truth","mean_pred", "lwr_ppi", "upr_ppi", "within_interval")
    
    
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
    colnames(joined_data.test) <- c("subject_id", "truth","mean_pred", "lwr_ppi", "upr_ppi", "within_interval")
    
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
  
  
  
  ########################################################################################################Female
  n.train.female <- length(which(sex[train.test.ind$train] == 0))
  n.test.female <- length(which(sex[train.test.ind$test] == 0))
  
  for(i in runs){
    print(paste0("run: ",i))
    dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_inpred__jobid_',i,'.feather'))))
    
    #####. need to find a way to only compute male and female
    
    print(dim(dat.in))
    print(head(dat.in))
    if(mala_if){ 
      dat.in <- tail(dat.in,200*1000) #burn in of 1000
    } else {
      dat.in <- tail(dat.in,350*1000) # tail(dat.in,50*1000) only true if it is sgld 200.  #I think this is the _size_ of minibatch (was 500) * _burn_in_ 
    }
    
    colnames(dat.in) <- c('id','pred')
    dat.in.grouped <- dat.in %>%
      group_by(id) %>%
      summarize(mean_pred = mean(pred), lwr_ppi = mean(pred)-1.96*sqrt(mod.mse+var(pred)), upr_ppi = mean(pred)+1.96*sqrt(mod.mse+var(pred)))
    
    joined_data.train <- left_join(true.train, dat.in.grouped, by = "id")
    joined_data.train <- joined_data.train %>%
      mutate(within_interval = ifelse(true>= lwr_ppi & true <= upr_ppi, TRUE, FALSE))
    colnames(joined_data.train) <- c("subject_id", "truth","mean_pred", "lwr_ppi", "upr_ppi", "within_interval")
    
    
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
    colnames(joined_data.test) <- c("subject_id", "truth","mean_pred", "lwr_ppi", "upr_ppi", "within_interval")
    
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
  
  ##################################################################################################################################Male
  
  
  ##################################################################################################################################
  
  return(out)
  
  #if split, copy the above twice
  
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





