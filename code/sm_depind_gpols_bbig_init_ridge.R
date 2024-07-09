

###Note that this analysis only works if the classes exist in both training and test data


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


filename <- "june13_sm_depind_gpols_once_init_ridge" 
success.run <- 1:10
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



#Define Mean Squared Error
mse <- function(pred, true){mean((pred-true)^2)}
phi <- function(k){(k+1)}

#Losses
loss.train <- vector(mode = "numeric")
loss.val <- vector(mode = "numeric")

#r squared
rsq.train <- vector(mode = "numeric")
rsq.val <- vector(mode = "numeric")

#Predictions
pred.train.ind <- vector(mode = "numeric")
pred.train.val <- vector(mode = "numeric")
pred.test.ind <- vector(mode = "numeric")
pred.test.val <- vector(mode = "numeric")

#class assignment
class.train.val <-vector(mode = "numeric")
class.test.val <-vector(mode = "numeric")

#class performance
class.performance.ind <- vector(mode = "numeric")

print("Loading data")

#Age
age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/age_sex_strat_depind.feather'))
#age_tab <- age_tab[order(age_tab$id),].     #DOES THIS MESS UP ORDER
age <- age_tab$age

train.test.ind <- list()
train.test.ind$test <- read.csv('/well/nichols/users/qcv214/KGPNN/sex_test_index.csv')$x
train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/KGPNN/sex_train_index.csv')$x
n.train <- length(train.test.ind$train)
n.test <- length(train.test.ind$test)

#mask
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
res3.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz'))

n.mask <- length(res3.mask.reg)
# n.expan <- choose(10+3,3) #this should correspond to dec_vec given in res4_first_layer_gp
p.dat <- ncol(res3.dat)
n.dat <- nrow(res3.dat)


#Load in class predictions from gpnn
dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/KGPNN/pile/re_june13_sm_depind_gpols_once_init_bbs_inpred__jobid_', init.num, '.feather'))))[, c(1:3)]
dat.in <- tail(dat.in, n.train)
colnames(dat.in) <- c('id_ind', 'pred', 'class')

dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/KGPNN/pile/re_june13_sm_depind_gpols_once_init_bbs_outpred__jobid_', init.num, '.feather'))))[, c(1:3)]
dat.out <- tail(dat.out, n.test)
colnames(dat.out) <- c('id_ind', 'pred', 'class')

#Join age_tab and dat.in and dat.out using id_ind as index
age_tab$class <- NA

age_tab$class[dat.in$id_ind] <- dat.in$class
age_tab$class[dat.out$id_ind] <- dat.out$class


#make fit ridge per class ###Note that this analysis only works if the classes exist in both training and test data
classes <- intersect(unique(dat.in$class), unique(dat.out$class))

for(class in classes){
  class.index <- which(age_tab$class==class)
  training.ind <- intersect(train.test.ind$train, class.index)
  test.ind <- intersect(train.test.ind$test, class.index)
  x.train <- res3.dat[training.ind,]
  x.test <- res3.dat[test.ind,]
  y.train <- age_tab$age[training.ind]
  y.test <- age_tab$age[test.ind]
  
  #fit Ridge
  ridge.fit <- cv.glmnet(x = data.matrix(as.matrix(x.train)) ,y = y.train, alpha = 0, lambda = NULL,standardize = FALSE)
  #in-sample pred
  pred_prior<-predict(ridge.fit, data.matrix(as.matrix(x.train)), s= "lambda.min")
  #out-sample pred
  pred_prior_new<-predict(ridge.fit, data.matrix(as.matrix(x.test)), s= "lambda.min")
  
  #Subject-level
  pred.train.ind <- c(pred.train.ind,training.ind) 
  pred.train.val <- c(pred.train.val,pred_prior)
  pred.test.ind <- c(pred.test.ind,test.ind) 
  pred.test.val <- c(pred.test.val,pred_prior_new) 
  #class
  class.train.val <- c(class.train.val, rep(class,length(training.ind)))
  class.test.val <- c(class.test.val, rep(class,length(test.ind)))
  
  #Overall
  
  
  loss.train <- c(loss.train, mseCpp(pred_prior,y.train))
  rsq.train <- c(rsq.train, rsqCpp(y.train,pred_prior))
  
  loss.val <- c(loss.val, mseCpp(pred_prior_new,y.test))
  rsq.val <- c(rsq.val, rsqCpp(y.test,pred_prior_new))
  
  class.performance.ind <- c(class.performance.ind, class)
  
}

write.csv(rbind(class.performance.ind,loss.train,loss.val,rsq.train,rsq.val),paste0("/well/nichols/users/qcv214/KGPNN/pile/re_",filename,"_class_perf_","_jobid_",JobId,".csv"), row.names = FALSE)

temp.frame <- as.data.frame(rbind(pred.train.ind,pred.train.val,class.train.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
# temp.frame<- t(tail(t(temp.frame),length(train.test.ind$train)*5))
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_inpred_',"_jobid_",JobId,'.feather'))
temp.frame <- as.data.frame(rbind(pred.test.ind,pred.test.val,class.test.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
# temp.frame<- t(tail(t(temp.frame),length(train.test.ind$test)*5))
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_outpred_',"_jobid_",JobId,'.feather'))



