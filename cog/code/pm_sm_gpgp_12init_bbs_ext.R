# R script

#Making held-out predictions

#28 Oct, for sanity check, let's make a prediction for all data


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
JobId=as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
print(JobId)
set.seed(JobId)

print("Starting")
print(Sys.time())

print('############### Test Optimised ###############')


filename <- "aug9_pm_sm_gpgp_12init_bbs"

success.run <- c(1:10)
init.num <- ifelse(JobId %in% success.run, yes = JobId, no = sample(success.run,1))

start.time <- Sys.time()
#1 Split data into mini batches (train and validation)
print("Load Rcpp")
source("/well/nichols/users/qcv214/KGPNN/code/rcpp_funcs.R")
time.taken <- Sys.time() - start.time
cat("load rcpp file completed in: ", time.taken)

train_test_split<- function(num_datpoint, num_test,num_train){
  full.ind<-1:num_datpoint
  #test
  test<-sample(x=full.ind,size=num_test,replace=FALSE)
  #train
  left <-sample(x=setdiff(full.ind,test),size=num_train,replace=FALSE)
  mini_batches <- split(left,ceiling(seq_along(left)/batch_size))
  out=list()
  out$test<-test
  out$train<-left
  return(out)
}
batch_split <- function(left, batch_size){
  left <- sample(x=left,size=length(left),replace=FALSE)
  mini_batches <- split(left,ceiling(seq_along(left)/batch_size))
  out=list()
  out$train<-mini_batches
  return(out)
}
rsqcal <- function(true,pred){
  RSS <-sum((true - pred)^2)
  TSS <- sum((true - mean(true))^2)
  return((1 - RSS/TSS)*100)
}

#Define ReLU
relu <- function(x) sapply(x, function(z) max(0,z))
relu.prime <- function(x) sapply(x, function(z) 1.0*(z>0))
# #Define shifted sigmoid
# sigmoid <- function(x) sapply(x, function(z) 2/(1+exp(-z))-1)
# sigmoid.prime <- function(x) sapply(x, function(z) 2*exp(-z)/(exp(-z)+1)^2) #this makes sense
#Define scaled and shifted sigmoid
sigmoid <- function(x) sapply(x, function(z) 2/(1+exp(-10*z))-1)
# sigmoid.prime <- function(x) sapply(x, function(z) 20*exp(-10*z)/(exp(-10*z)+1)^2) #this makes sense
sigmoid.prime <- function(x) {    #This breaks at 71, exactly the same as on Desmos
  sapply(x, function(z) {
    #Applying constraint to prevent gradient instability.... not sure if this is too harsh
    z <- ifelse(abs(z) > 1, 1 * sign(z), z)
    derivative = 20*exp(-10*z)/(exp(-10*z)+1)^2
    return(derivative)
  })
}

softmax <-function(z){
  t(apply(z,1,function(x){
    x<-10*x
    xmax <- max(x)
    return(exp(x-max(x))/sum(exp(x-max(x))))
  }))
}

softmax.prime <- function(x, l.grad) {
  a <- x %*% -t(x)*10
  diag(a) <- x*(10-10*x)
  return(a*l.grad) #deleted a*l.grad here
}

#For creating interaction effect
elementwise_product <- function(vec1, vec2) {
  return(vec1 * vec2)
}


#Define Mean Squared Error
mse <- function(pred, true){mean((pred-true)^2)}
phi <- function(k){(k+1)}



print("Loading data")


#Age
age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/cog/agesex_strat2.feather'))
#age_tab <- age_tab[order(age_tab$id),].     #DOES THIS MESS UP ORDER
lognum <- (age_tab$pm_tf)

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

co.dat <- cbind(sex,dep.group1,dep.group2,dep.group3 ,age.group1,age.group2,age.group3)


# res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz')
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
# res3.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz'))
res3.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz'))


n.mask <- length(res3.mask.reg)
# n.expan <- choose(10+3,3) #this should correspond to dec_vec given in res4_first_layer_gp
p.dat <- ncol(res3.dat)
n.dat <- nrow(res3.dat)

#For specifying n.expan.degree
# n.expan.degree <- 30
n.expan <- choose(20+3,3)

#Pre-transform
print(paste0("load up res3 gp"))
source("/well/nichols/users/qcv214/bnn2/res3/first_layer_gp5_50.R")

print(paste0("load up res3 gp [DONE]"))

partial.gp.centroid.img<-t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/KGPNN/partial_gp_centroids_fixed_300.540.feather"))))
l.expan <- ncol(partial.gp.centroid.img)

mult.dat <- function(X) as.matrix(res3.dat %*% X)
mult.thea <- function(X,theta) rowSums(X*theta)
res3.dat <- array(t(apply(partial.gp,MARGIN = c(1),mult.dat)), dim =c(n.mask,n.dat,n.expan)) #rename res3.dat as the product of data and expansion


################################################################################################################################################
for(num.lat.class.select in c(2)){
  
  #Losses
  loss.train <- vector(mode = "numeric")
  loss.val <- vector(mode = "numeric")
  
  loss.train.male <- vector(mode = "numeric")
  loss.val.male <- vector(mode = "numeric")
  
  loss.train.fmale <- vector(mode = "numeric")
  loss.val.fmale <- vector(mode = "numeric")
  
  map.train <- vector(mode = "numeric")
  #r squared
  rsq.train <- vector(mode = "numeric")
  rsq.val <- vector(mode = "numeric")
  
  rsq.train.male <- vector(mode = "numeric")
  rsq.val.male <- vector(mode = "numeric")
  
  rsq.train.fmale <- vector(mode = "numeric")
  rsq.val.fmale <- vector(mode = "numeric")
  #Predictions
  pred.train.ind <- vector(mode = "numeric")
  pred.train.val <- vector(mode = "numeric")
  pred.test.ind <- vector(mode = "numeric")
  pred.test.val <- vector(mode = "numeric")
  
  #class assignment
  class.train.val <-vector(mode = "numeric")
  class.test.val <-vector(mode = "numeric")
  #Epoch learning rate
  lr.vec <- vector(mode = "numeric")
  pre.lr.vec <- vector(mode = "numeric")
  prod.lr.vec <-  vector(mode = "numeric")
  ck.old <- 1
  
  #min held-out RMSE
  min.mse <- 1e+8
  
  
  time.taken <- Sys.time() - start.time
  cat("Loading data complete in: ", time.taken)
  print(Sys.time())
  
  
  print("Initialisation")
  #1 Initialisation
  #1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
  # using minimum recorded
  min.theta <- as.matrix(read_feather(paste0( "/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_minweights__jobid_",init.num,'.feather')))
  min.gp.coef <- as.matrix(read_feather(paste0( "/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_minLCgp__jobid_",init.num,'.feather')))
  min.co.weights<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_mincoweights__jobid_",init.num,".csv")))
  min.lweights <- c(unlist(read_feather(paste0( "/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_minlweights__jobid_",init.num,'.feather'))))
  min.bias <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_minbias__jobid_",init.num,".csv"))$x
  min.co.bias <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_mincobias__jobid_",init.num,".csv"))$x
  min.lbias <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_minlbias__jobid_",init.num,".csv"))$x
  
  time.train <-  Sys.time()
  
  # hidden.layer.test <- apply(t(apply(res3.dat[, train.test.ind$unseen, ],MARGIN = 2,FUN = mult.thea,thet=theta.matrix) + bias), 2, FUN = relu)
  hidden.layer <- apply(t(apply(res3.dat,MARGIN = 2,FUN = mult.thea,thet=min.theta) + min.bias), 2, FUN = relu)
  
  co.pre.hidden.layer <- t(t(co.dat %*% t(min.co.weights)) + min.co.bias)
  co.hidden.layer <- softmax(co.pre.hidden.layer)
  
  hidden.layer.mixed <- cbind(hidden.layer, co.hidden.layer)
  
  # Generate polynomial features (linear terms)
  partial.gp.centroid <- rbind(partial.gp.centroid.img,min.gp.coef)
  poly_features <- as.matrix(hidden.layer.mixed %*% partial.gp.centroid) #
  
  # z.nb <- cbind(1,poly_features) #This is different from LASIR in the sense that the subgroup latent directly affect the output, whereas the group themselves dont. But then that can be modified easily.
  
  hs_pred_SOI <- min.lbias + poly_features %*% min.lweights
  pred.test.ind <- 1:n.dat
  pred.test.val <- c(hs_pred_SOI)
  class.test.val <- c(apply(co.hidden.layer,1,which.max))
  
  
  #BB
  
  time.taken <- Sys.time() - time.train
  cat("Training complete in: ", time.taken)
  
  print(length(pred.test.ind))
  print(head(pred.test.ind))
  print(length(pred.test.val))
  print(head(pred.test.val))
  print(length(class.test.val))
  print(head(class.test.val))
  
  temp.frame <- as.data.frame(rbind(pred.test.ind,pred.test.val,class.test.val))
  colnames(temp.frame) <- NULL
  colnames(temp.frame) <- 1:ncol(temp.frame)
  write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_min_outpred_ext_test_',"_jobid_",JobId,'.feather'))
  
  
}


