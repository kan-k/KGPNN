
#Nov 5 , changed data from sim15_age.feather to sim15_age_gpr

# R script


#lasso 
if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)
p_load(dplyr)

##3 Dec with white matter, stem removed and thresholded

print("stage 1")
JobId=as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
print(JobId)



#Age
# age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/sim15_age.feather'))

age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/sim15_age_gpr.feather'))
#age_tab <- age_tab[order(age_tab$id),].     #DOES THIS MESS UP ORDER
dat.age <- age_tab$pred_amp
sex <-  as.numeric(age_tab$sex)
sex <- sapply(sex, function(x) replace(x, x==0,-1)) #Change female to -1, male to 1

#subset data
ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
ind.to.use <- list()
ind.to.use$test <- read.csv('/well/nichols/users/qcv214/KGPNN/sim15_test_index.csv')$x
ind.to.use$train <- read.csv('/well/nichols/users/qcv214/KGPNN/sim15_train_index.csv')$x

depind <- age_tab$DepInd
quantile_thresholds.15 <- median(depind) #can use MEDIAN since length is even, so it takes the middle value, it's 3 and 31
dep.group1 <- ifelse(depind < quantile_thresholds.15, yes =1, no = 0) #bottom 15
dep.group2 <- ifelse(depind >= quantile_thresholds.15, yes =1, no = 0) #top 15
#Define another group variable called age group which is directly associated with what we are predicting.
#age.group <- ifelse(age > mean(age), yes = 1, no = -1)

#mask
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
dat_allmat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz'))

co.dat <- cbind(sex,dep.group1,dep.group2)

dat_allmat <- cbind(dat_allmat, co.dat) #Combining imaging and non-imaging

n.mask <- length(res3.mask.reg)
# n.expan <- choose(6+3,3)
# p.dat <- ncol(res3.dat)
# n.dat <- nrow(res3.dat)



print("stage 2")

#func

rsqcal2<-function(old,new,ind.old,ind.new){
  ridge_pred<-old
  ridge_pred_new<-new
  no<-length(old)
  nn<-length(new)
  #insample
  y<-dat.age[ind.old]
  sserr_ridge<-sum((y-ridge_pred)^2)
  sstot<-var(y)*length(y)
  #outsample
  y_new<-dat.age[ind.new]
  sstot_new<-sum((y_new-mean(y))^2)
  sserr_ridge_new<-sum((y_new-ridge_pred_new)^2)
  #Output
  #print(paste0('In-sameple Variance of prediction explained: ',round(1-sserr_ridge/sstot,5)*100,' || RMSE: ', round(sqrt(mean((y-ridge_pred)^2)),4)))
  #print(paste0('Out-sample Variance of prediction explained: ',round(1-sserr_ridge_new/sstot_new,5)*100,' || RMSE: ', round(sqrt(mean((y_new-ridge_pred_new)^2)),4)))
  print('Done')
  out=list()
  out$inrsq<-round(1-sserr_ridge/sstot,5)*100
  out$inmae<-round(mean(abs(y-ridge_pred)),4)
  out$outrsq<-round(1-sserr_ridge_new/sstot_new,5)*100
  out$outmae<-round(mean(abs(y_new-ridge_pred_new)),4)
  out$inrmse <-round(sqrt(mean((y-ridge_pred)^2)),4)
  out$outrmse <-round(sqrt(mean((y_new-ridge_pred_new)^2)),4)
  return(out)
}

print("stage 3")
time.train <-  Sys.time()


#Tested.... doing set seed(4) get_ind is the same as loading sim_wb2_index


#get beta(v)
time.train <-  Sys.time()
print("fitting")
lassofit<- cv.glmnet(x = data.matrix(as.matrix(dat_allmat[ind.to.use$train, ])) ,y = dat.age[ind.to.use$train], alpha = 0, lambda = NULL,standardize = FALSE) #alpha does matter here, 0 is ridge
print("in-predicting")
print(dim(data.matrix(as.matrix(dat_allmat[ind.to.use$train,]))))
pred_prior<-predict(lassofit, data.matrix(as.matrix(dat_allmat[ind.to.use$train,])), s= "lambda.min")
print("out-predicting")
print(dim(data.matrix(as.matrix(dat_allmat[ind.to.use$test,]))))
pred_prior_new<-predict(lassofit, data.matrix(as.matrix(dat_allmat[ind.to.use$test,])), s= "lambda.min")

time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

write.csv( c(unlist(t(as.matrix(rsqcal2(pred_prior,pred_prior_new,ind.old = ind.to.use$train,ind.new = ind.to.use$test)))),as.numeric(sub('.*:', '', summary(coef(lassofit, s=lassofit$lambda.min)[-1,]))),sum(abs(coef(lassofit, s=lassofit$lambda.min))>1e-8)),
           paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_nov5_ridge_depind15_noscale_",JobId,".csv"), row.names = FALSE)

print("write prediction")

write.csv(c(pred_prior_new),paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_nov5_ridge_depind15_outpred_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior),paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_nov5_ridge_depind15_inpred_noscale_",JobId,".csv"), row.names = FALSE)


######LASSO
#get beta(v)
time.train <-  Sys.time()
print("fitting")
lassofit<- cv.glmnet(x = data.matrix(as.matrix(dat_allmat[ind.to.use$train, ])) ,y = dat.age[ind.to.use$train], alpha = 1, lambda = NULL,standardize = FALSE) #alpha does matter here, 0 is ridge
print("in-predicting")
print(dim(data.matrix(as.matrix(dat_allmat[ind.to.use$train,]))))
pred_prior<-predict(lassofit, data.matrix(as.matrix(dat_allmat[ind.to.use$train,])), s= "lambda.min")
print("out-predicting")
print(dim(data.matrix(as.matrix(dat_allmat[ind.to.use$test,]))))
pred_prior_new<-predict(lassofit, data.matrix(as.matrix(dat_allmat[ind.to.use$test,])), s= "lambda.min")



time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

write.csv( c(unlist(t(as.matrix(rsqcal2(pred_prior,pred_prior_new,ind.old = ind.to.use$train,ind.new = ind.to.use$test)))),as.numeric(sub('.*:', '', summary(coef(lassofit, s=lassofit$lambda.min)[-1,]))),sum(abs(coef(lassofit, s=lassofit$lambda.min))>1e-8)),
           paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_nov5_lasso_depind15_noscale_",JobId,".csv"), row.names = FALSE)

print("write prediction")

write.csv(c(pred_prior_new),paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_nov5_lasso_depind15_outpred_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior),paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_nov5_lasso_depind15_inpred_noscale_",JobId,".csv"), row.names = FALSE)