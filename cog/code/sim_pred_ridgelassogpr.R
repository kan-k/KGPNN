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
######
#Age
age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/cog/sim_agesex_strat2.feather'))
#age_tab <- age_tab[order(age_tab$id),].     #DOES THIS MESS UP ORDER
dat.age <- (age_tab$pred)

age <- as.numeric(age_tab$age)
sex <-  as.numeric(age_tab$sex)
sex <- sapply(sex, function(x) replace(x, x==0,-1)) #Change female to -1, male to 1

train.test.ind <- list()
train.test.ind$test <- read.csv('/well/nichols/users/qcv214/KGPNN/cog/cog_test_index.csv')$x
train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/KGPNN/cog/cog_train_index.csv')$x
n.train <- length(train.test.ind$train)

depind <- age_tab$DepInd
quantile_thresholds <- quantile(depind[train.test.ind$train], probs = seq(0, 1, by = 0.34))
#age.group <- ifelse(age > mean(age), yes = 1, no = -1)
dep.group1 <- ifelse(depind <=quantile_thresholds[[2]], yes =1, no = 0)
dep.group2 <- ifelse(depind > quantile_thresholds[[2]] & depind <=quantile_thresholds[[3]], yes =1, no = 0)
dep.group3 <- ifelse(depind > quantile_thresholds[[3]], yes =1, no = 0)

quantile_thresholds <- quantile(age[train.test.ind$train], probs = seq(0, 1, by = 0.34))
#age.group <- ifelse(age > mean(age), yes = 1, no = -1)
age.group1 <- ifelse(age <=quantile_thresholds[[2]], yes =1, no = 0)
age.group2 <- ifelse(age > quantile_thresholds[[2]] & age <=quantile_thresholds[[3]], yes =1, no = 0)
age.group3 <- ifelse(age > quantile_thresholds[[3]], yes =1, no = 0)
#####
co.dat <- cbind(sex,dep.group1,dep.group2,dep.group3 ,age.group1,age.group2,age.group3)



#subset data
ind.to.use <- list()
ind.to.use$test <-  read.csv('/well/nichols/users/qcv214/KGPNN/cog/cog_test_index.csv')$x
ind.to.use$train <- read.csv('/well/nichols/users/qcv214/KGPNN/cog/cog_train_index.csv')$x


#mask
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
dat_allmat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz'))


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
           paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/sim_oct1_pred_ridge_noscale_",JobId,".csv"), row.names = FALSE)

print("write prediction")

write.csv(c(pred_prior_new),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/sim_oct1_pred_ridge_outpred_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/sim_oct1_pred_ridge_inpred_noscale_",JobId,".csv"), row.names = FALSE)


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
           paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/sim_oct1_pred_lasso_noscale_",JobId,".csv"), row.names = FALSE)

print("write prediction")

write.csv(c(pred_prior_new),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/sim_oct1_pred_lasso_outpred_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/sim_oct1_pred_lasso_inpred_noscale_",JobId,".csv"), row.names = FALSE)


####GPR
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id[1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
nb <- find_brain_image_neighbors(img1, res3.mask, radius=1)


norm.func <- function(x){ 2*(x - min(x))/(max(x)-min(x)) -1 }

print("stage 3")
time.train <-  Sys.time()
poly_degree = 40 #Was 20
a_concentration = 0.5
b_smoothness = 40

nb.centred<- apply(nb$maskcoords,2,norm.func)
#get psi
psi.mat.nb <- GP.eigen.funcs.fast(nb.centred, poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
#get lambda
lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = 3)
#Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
sqrt.lambda.nb <- sqrt(lambda.nb)
bases.nb <- t(psi.mat.nb)*sqrt.lambda.nb

print("before cbind")
#Get design matrix
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
dat_allmat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz'))


dat_allmat <- cbind(t(bases.nb%*%t(dat_allmat)), co.dat) #Combining imaging and non-imaging

z.nb <- cbind(1,dat_allmat)


print("after cbind")


#subset data
train.test.ind <- list()
train.test.ind$test <- read.csv('/well/nichols/users/qcv214/KGPNN/cog/cog_test_index.csv')$x
train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/KGPNN/cog/cog_train_index.csv')$x
n.train <- length(train.test.ind$train)
n.test <- length(train.test.ind$test)


train_Y <- dat.age[train.test.ind$train]
train_Z <- z.nb[train.test.ind$train,]
test_Y <- dat.age[train.test.ind$test]
# test_img <- dat_allmat[train.test.ind$test,]
test_Z <- z.nb[train.test.ind$test,]

#get beta(v)
time.train <-  Sys.time()
mc_sample <- 300L
lassofit <- fast_normal_lm(X = train_Z ,y =train_Y,mcmc_sample = mc_sample) #Change smaples to 400 ###Was fast_horseshoe
time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)
pred_prior<-predict_fast_lm(lassofit, train_Z )#$mean
pred_prior_new<-predict_fast_lm(lassofit, test_Z)#$mean


##########################################################################################################################################################
#print(adsfasjdfoasgasjpog) #Just want to check time
##########################################################################################################################################################

write.csv(c(unlist(t(as.matrix(rsqcal2(pred_prior$mean,pred_prior_new$mean,ind.old = train.test.ind$train,ind.new = train.test.ind$test)))),as.numeric(sub('.*:', '', summary(lassofit$post_mean$betacoef[-1,]))),sum(abs(lassofit$post_mean$betacoef[-1,])>1e-5)),
          paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/sim_oct1_pred_gpr_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior_new$mean),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/sim_oct1_pred_gpr_outpred_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior$mean),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/sim_oct1_pred_gpr_inpred_noscale_",JobId,".csv"), row.names = FALSE)
####Result to use
write.csv(rbind(c(train.test.ind$train),c(train.test.ind$test)),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/sim_oct1_pred_gpr_index_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(lassofit$post_mean$betacoef),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/sim_oct1_pred_gpr_coef_',JobId,'.feather'))

##Plotting 
print("bases.nb")
print(dim(bases.nb))
print("length coef")
print(length(lassofit$post_mean$betacoef[2:(ncol(bases.nb)+1)]))
print("fit t(bases.nb)")
beta_fit <- data.frame(NM = crossprod(t(bases.nb),lassofit$post_mean$betacoef[2:(ncol(bases.nb)+1)]))
print("past")
gp.mask.nm <- res3.mask
gp.mask.nm[gp.mask.nm!=0] <-beta_fit$NM
gp.mask.nm@datatype = 16
gp.mask.nm@bitpix = 32
writeNIfTI(gp.mask.nm,paste0('/well/nichols/users/qcv214/KGPNN/cog/viz/sim_oct1_pred_gpr_',poly_degree,a_concentration,b_smoothness,JobId))





#Posterior Predictive mean in-sampleMSE
in.mse <- mean((pred_prior$mean-age_tab$pred[train.test.ind$train])^2)

#######In-sample################################################
stat.in.ig.stderrmod <- sqrt(in.mse+pred_prior$sd^2)
stat.in.ig.lwrmod <- pred_prior$mean - qt(0.975,mc_sample-1)*stat.in.ig.stderrmod
stat.in.ig.uprmod <- pred_prior$mean + qt(0.975,mc_sample-1)*stat.in.ig.stderrmod
##for no personalised variance
stat.in.ig.lwrmod2 <- pred_prior$mean - qt(0.975,mc_sample-1)*sqrt(in.mse)
stat.in.ig.uprmod2 <- pred_prior$mean + qt(0.975,mc_sample-1)*sqrt(in.mse)

#Define proportion counting

#True
within.pred <- vector(mode='numeric')
within.pred2 <- vector(mode='numeric')
for(i in 1:length(age_tab$pred[train.test.ind$train])){
  within.pred <- c(within.pred,(between(age_tab$pred[train.test.ind$train][i],stat.in.ig.lwrmod[i],stat.in.ig.uprmod[i])))
  within.pred2 <- c(within.pred2,(between(age_tab$pred[train.test.ind$train][i],stat.in.ig.lwrmod2[i],stat.in.ig.uprmod2[i])))
}
stat.in.ig.true.covermod <- within.pred
stat.in.ig.true.covermod2 <- within.pred2

print(paste0('Proprtion of true lying within subject 95% prediction interval: ',sum(stat.in.ig.true.covermod)/n.train*100))

#######Out-of-sample################################################
stat.out.ig.stderrmod <- sqrt(in.mse+pred_prior_new$sd^2)
stat.out.ig.lwrmod <- pred_prior_new$mean - qt(0.975,mc_sample-1)*stat.out.ig.stderrmod
stat.out.ig.uprmod <- pred_prior_new$mean + qt(0.975,mc_sample-1)*stat.out.ig.stderrmod
##for no personalised variance
stat.out.ig.lwrmod2 <- pred_prior_new$mean - qt(0.975,mc_sample-1)*sqrt(in.mse)
stat.out.ig.uprmod2 <- pred_prior_new$mean + qt(0.975,mc_sample-1)*sqrt(in.mse)

#Define proportion counting

#True
within.pred <- vector(mode='numeric')
within.pred2 <- vector(mode='numeric')
for(i in 1:length(age_tab$pred[train.test.ind$test])){
  within.pred <- c(within.pred,(between(age_tab$pred[train.test.ind$test][i],stat.out.ig.lwrmod[i],stat.out.ig.uprmod[i])))
  within.pred2 <- c(within.pred2,(between(age_tab$pred[train.test.ind$test][i],stat.out.ig.lwrmod2[i],stat.out.ig.uprmod2[i])))
  
}
stat.out.ig.true.covermod <- within.pred
stat.out.ig.true.covermod2 <- within.pred2
print(paste0('Proprtion of true lying within subject 95% prediction interval: ',sum(stat.out.ig.true.covermod)/n.test*100))

print("done heree")

print(sum(stat.in.ig.true.covermod))
print(sum(stat.out.ig.true.covermod))
print(sum(stat.in.ig.true.covermod2))
print(sum(stat.out.ig.true.covermod2))
print(c(n.train,n.test,n.train,n.test))

cover.mat <- t(matrix(c(sum(stat.in.ig.true.covermod),sum(stat.out.ig.true.covermod),sum(stat.in.ig.true.covermod2),sum(stat.out.ig.true.covermod2))))/c(n.train,n.test,n.train,n.test)*100
colnames(cover.mat) <- c("train","test","npvtrain","npvtest")
print('Dine')
write.csv(cover.mat,paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/sim_oct1_pred_gpr_coverage_",JobId,".csv"), row.names = FALSE)
print("all done")

