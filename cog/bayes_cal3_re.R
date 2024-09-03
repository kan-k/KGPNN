# 20 May: Bayes calculation for SGLD, simulation

#2 june, change from Nmi to rand index

#this is taken fromn bayes_cal3, but note that we have no "true" classes


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
p_load(infotheo)
p_load(mclust)

JobId=as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
print(JobId)
set.seed(JobId)

#calculating r^2
rsqcal <- function(true,pred){
  RSS <-sum((true - pred)^2)
  TSS <- sum((true - mean(true))^2)
  return((1 - RSS/TSS)*100)
}

rmse <- function(truth, prediction) {
  sqrt(mean((truth - prediction)^2, na.rm = TRUE))
}

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

getproportion <- function(true_class,pred_class){
  return(sum(pred_class==true_class)/length(pred_class))
}
compute_AMI <- function(true_class, pred_class, num_permutations = 100) {
  # Helper function to compute the mutual information
  mi <- mutinformation(true_class, pred_class)
  # Compute the entropies
  entropy_true <- entropy(true_class)
  entropy_pred <- entropy(pred_class)
  # Compute the expected mutual information (EMI)
  # Approximate EMI using a permutation method
  expected_mi <- function(true_class, pred_class, num_permutations) {
    n <- length(true_class)
    mi_values <- numeric(num_permutations)
    for (i in 1:num_permutations) {
      permuted_pred <- sample(pred_class)
      mi_values[i] <- mutinformation(true_class, permuted_pred)
    }
    mean(mi_values)
  }
  
  emi <- expected_mi(true_class, pred_class, num_permutations)
  # Compute the Adjusted Mutual Information (AMI)
  ami <- (mi - emi) / ((entropy_true + entropy_pred) / 2 - emi)
  return(ami)
}
compute_NMI <- function(true_class, pred_class) {
  2*mutinformation(true_class,pred_class)/(entropy(pred_class)+ entropy(true_class))
}


meanstats.re<- function(filename, runs, mala_if){
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
  
  
  age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/cog/agesex_strat2.feather'))
  train.test.ind <- list()
  train.test.ind$test <- read.csv('/well/nichols/users/qcv214/KGPNN/cog/cog_test_index.csv')$x
  train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/KGPNN/cog/cog_train_index.csv')$x
  lognum <- age_tab$pm_tf
  train.true <- lognum[train.test.ind$train]
  #add class
  true.train<- data.frame(train.test.ind$train,train.true,row.names = NULL)
  colnames(true.train) <- c('id','true')
  
  test.true <- lognum[train.test.ind$test]
  true.test<- data.frame(train.test.ind$test,test.true,row.names = NULL)
  colnames(true.test) <- c('id','true')
  
  num.it <-2000 #500 epochs, 4 minibatches
  
  
  #This does not seem to work on 26 May. I will change it to indidual runs instead
  # res.mat<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_loss__jobid_",runs,".csv")))
  # mod.mse <- mean(res.mat[1,])
  
  for(i in runs){
    print(paste0("run: ",i))
    res.mat<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_loss__jobid_",i,".csv")))
    mod.mse <- mean(res.mat[1,])
    dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_inpred__jobid_',i,'.feather'))))[,1:3] #Taking only the indices and the predictions but not class assignments
    print(dim(dat.in))
    print(head(dat.in))
    if(mala_if){ 
      dat.in <- tail(dat.in,200*1000) #burn in of 1000 #for MALA only
    } else {
      dat.in <- tail(dat.in,(length(train.test.ind$train)/4)*1000) # tail(dat.in,50*1000) only true if it is sgld 200
    }
    
    colnames(dat.in) <- c('id','pred','class')
    dat.in.grouped <- dat.in %>%
      group_by(id) %>%
      summarize(mean_pred = mean(pred), lwr_ppi = mean(pred)-1.96*sqrt(mod.mse+var(pred)), upr_ppi = mean(pred)+1.96*sqrt(mod.mse+var(pred)), pred_class = getmode(class))
    
    joined_data.train <- left_join(true.train, dat.in.grouped, by = "id")
    joined_data.train <- joined_data.train %>%
      mutate(within_interval = ifelse(true>= lwr_ppi & true <= upr_ppi, TRUE, FALSE))
    colnames(joined_data.train) <- c("subject_id", "truth","mean_pred", "lwr_ppi", "upr_ppi","pred_class", "within_interval")
    
    result.train <- joined_data.train %>%
      group_by(pred_class) %>%
      summarize(
        RMSE = rmse(truth, mean_pred),
        interval_width = mean(upr_ppi - lwr_ppi, na.rm = TRUE),
        coverage = sum(within_interval) * 100 / n()
      )
    # Define the function to be applied
    
    ##### this is not exactly right. yet as classes may not coincide... I think classes need to reference to its underlying non-imaging covariates
    # get_class_proportion <- function(i, reference_table, data) {
    #   true_class <- reference_table$true_class[reference_table$subject_id == i]
    #   predicted_class <- data$class[data$id == i]
    #   getproportion(true_class = true_class, pred_class = predicted_class)
    # }
    # # Use sapply to apply the function over dat.in$id
    # class_prop <- sapply(joined_data.train$subject_id, get_class_proportion, reference_table = joined_data.train, data = dat.in)
    # joined_data.train <- cbind(joined_data.train,class_prop)
    
    ####Added 18 Aug
    ############For MALA, 200 training
    # joined_data.train <- joined_data.train[1:200,]
    
    
    #Test
    dat.test <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_outpred__jobid_',i,'.feather'))))[,1:3]
    dat.test <- tail(dat.test,length(train.test.ind$test)*1000)
    colnames(dat.test) <- c('id','pred','class')
    dat.test.grouped <- dat.test %>%
      group_by(id) %>%
      summarize(mean_pred = mean(pred), lwr_ppi = mean(pred)-1.96*sqrt(mod.mse+var(pred)), upr_ppi = mean(pred)+1.96*sqrt(mod.mse+var(pred)), pred_class = getmode(class))
    
    joined_data.test <- left_join(true.test, dat.test.grouped, by = "id")
    joined_data.test <- joined_data.test %>%
      mutate(within_interval = ifelse(true>= lwr_ppi & true <= upr_ppi, TRUE, FALSE))
    colnames(joined_data.test) <- c("subject_id", "truth","mean_pred", "lwr_ppi", "upr_ppi","pred_class", "within_interval")
    
    result.test <- joined_data.test %>%
      group_by(pred_class) %>%
      summarize(
        RMSE = rmse(truth, mean_pred),
        interval_width = mean(upr_ppi - lwr_ppi, na.rm = TRUE),
        coverage = sum(within_interval) * 100 / n()
      )
    # Define the function to be applied
    # Use sapply to apply the function over dat.in$id
    # class_prop <- sapply(joined_data.test$subject_id, get_class_proportion, reference_table = joined_data.test, data = dat.test )
    # joined_data.test <- cbind(joined_data.test,class_prop)
    
    
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
  
  #I don't need the below due to processing 1 at a time
  
  out <- c(in.rmse.vec,out.rmse.vec,in.rsq.vec,out.rsq.vec,in.cover,out.cover,in.width,out.width,in.mae.vec,out.mae.vec)
  names(out) <- c("inRMSE","outRMSE","inR2","outR2","in-Coverage","out-Coverage","in-PPIwidth","out-PPIwidth","inMAE","outMAE")
  
  out <- c(out,unlist(result.train),unlist(result.test))
  names(out)[1:10] <- c("inRMSE","outRMSE","inR2","outR2","in-Coverage","out-Coverage","in-PPIwidth","out-PPIwidth","inMAE","outMAE")
  return(round(out,3))
}

# print("=========IG DONE=========")

# success.run <- c(1:10)
# for(run in success.run){
#   stat.eb <-meanstats.re("aug9_pm_gpols_12init_sgld", run, mala_if=FALSE)
#   write.csv(stat.eb,paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/summary_re_aug9_pm_gpols_sgld_",run,".csv"))
# }

success.run <- c(1:10)
for(run in success.run){
  stat.eb <-meanstats.re("aug22_pm_sm_gpols_12init_sgld_K2", run, mala_if=FALSE)
  write.csv(stat.eb,paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/summary_re_aug22_pm_gpols_sgld_K2_",run,".csv"))
}
