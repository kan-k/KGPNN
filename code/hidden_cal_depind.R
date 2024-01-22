# R script

#Now saving the IG params
#Nov11: bias corrected
#Nov13: co.weights loading corrected and BBSmoothing adjusted for non-img covar
#dec1: correcting imaging update, latent instead of sex

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

print('############### Test Optimised ###############')


filename <- "dec18_testhidden_sm_depind_gpnn_bbig_init_bbs" 
# init.num <- sample(c(5,7,8,10),1)#JobId
# prior.var <- 0.05 #was 0.05
# learning_rate <- 0.99 #for slow decay starting less than 1
# prior.var.bias <- 1
# epoch <- 150 #was 500
# beta.bb<- 0.5
# lr.init <- learning_rate


start.time <- Sys.time()
#1 Split data into mini batches (train and validation)

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
    xmax <- max(x)
    return(exp(x-max(x))/sum(exp(x-max(x))))
  }))
}

softmax.prime <- function(x, l.grad) {
  a <- x %*% -t(x)
  diag(a) <- x*(1-x)
  return(a*l.grad) #deleted a*l.grad here
}

#For creating interaction effect
elementwise_product <- function(vec1, vec2) {
  return(vec1 * vec2)
}


#Define Mean Squared Error
mse <- function(pred, true){mean((pred-true)^2)}
phi <- function(k){(k+1)}

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
#Epoch learning rate
lr.vec <- vector(mode = "numeric")
pre.lr.vec <- vector(mode = "numeric")
prod.lr.vec <-  vector(mode = "numeric")
ck.old <- 1

print("Loading data")

#Age
age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/age_sex_strat_depind.feather'))
#age_tab <- age_tab[order(age_tab$id),].     #DOES THIS MESS UP ORDER
age <- age_tab$age
sex <-  as.numeric(age_tab$sex)
sex <- sapply(sex, function(x) replace(x, x==0,-1)) #Change female to -1, male to 1
#Define another group variable called age group which is directly associated with what we are predicting.
train.test.ind <- list()
train.test.ind$test <- read.csv('/well/nichols/users/qcv214/KGPNN/sex_test_index.csv')$x
train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/KGPNN/sex_train_index.csv')$x
n.train <- length(train.test.ind$train)

depind <- age_tab$DepInd
quantile_thresholds <- quantile(depind[train.test.ind$train], probs = seq(0, 1, by = 0.1))

#Define another group variable called age group which is directly associated with what we are predicting.
#age.group <- ifelse(age > mean(age), yes = 1, no = -1)
dep.group1 <- ifelse(depind >= quantile_thresholds[[1]] & depind <=quantile_thresholds[[2]], yes =1, no = 0)
dep.group2 <- ifelse(depind > quantile_thresholds[[2]] & depind <=quantile_thresholds[[3]], yes =1, no = 0)
dep.group3 <- ifelse(depind > quantile_thresholds[[3]] & depind <=quantile_thresholds[[4]], yes =1, no = 0)
dep.group4 <- ifelse(depind > quantile_thresholds[[4]] & depind <=quantile_thresholds[[5]], yes =1, no = 0)
dep.group5 <- ifelse(depind > quantile_thresholds[[5]] & depind <=quantile_thresholds[[6]], yes =1, no = 0)
dep.group6 <- ifelse(depind > quantile_thresholds[[6]] & depind <=quantile_thresholds[[7]], yes =1, no = 0)
dep.group7 <- ifelse(depind > quantile_thresholds[[7]] & depind <=quantile_thresholds[[8]], yes =1, no = 0)
dep.group8 <- ifelse(depind > quantile_thresholds[[8]] & depind <=quantile_thresholds[[9]], yes =1, no = 0)
dep.group9 <- ifelse(depind > quantile_thresholds[[9]] & depind <=quantile_thresholds[[10]], yes =1, no = 0)
dep.group10 <- ifelse(depind > quantile_thresholds[[10]] & depind <=quantile_thresholds[[11]], yes =1, no = 0)

co.dat <- cbind(sex,dep.group1,dep.group2,dep.group3,dep.group4,dep.group5,dep.group6,dep.group7,dep.group8,dep.group9,dep.group10)
#mask
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
res3.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz'))

n.mask <- length(res3.mask.reg)
# n.expan <- choose(6+3,3)
p.dat <- ncol(res3.dat)
n.dat <- nrow(res3.dat)


time.taken <- Sys.time() - start.time
cat("Loading data complete in: ", time.taken)


print("Getting mini batch")
#Get minibatch index 
batch_size <- 500


#NN parameters
it.num <- 1


print("Initialisation")
#1 Initialisation
#1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
# theta.matrix <- matrix(,nrow=n.mask, ncol= n.expan)
weights9 <- as.matrix(read_feather(paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_dec16_sm_depind_gpnn_bbig_init_bbs_weights__jobid_',8,'.feather')))
weights10 <- as.matrix(read_feather(paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_dec16_sm_depind_gpnn_bbig_init_bbs_weights__jobid_',1,'.feather')))
weights6 <- as.matrix(read_feather(paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_dec16_sm_depind_gpnn_bbig_init_bbs_weights__jobid_',9,'.feather')))

#Initialising bias (to 0)
bias9 <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_dec16_sm_depind_gpnn_bbig_init_bbs_bias__jobid_",8,".csv"))$x
bias10 <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_dec16_sm_depind_gpnn_bbig_init_bbs_bias__jobid_",1,".csv"))$x
bias6 <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_dec16_sm_depind_gpnn_bbig_init_bbs_bias__jobid_",9,".csv"))$x




time.train <-  Sys.time()




hidden.layer9 <- apply(t(t(res3.dat[train.test.ind$train,]  %*% t(weights9)) + bias9), 2, FUN = relu)

hidden.layer10 <- apply(t(t(res3.dat[train.test.ind$train,]  %*% t(weights10)) + bias10), 2, FUN = relu)

hidden.layer6 <- apply(t(t(res3.dat[train.test.ind$train,]  %*% t(weights6)) + bias6), 2, FUN = relu)


time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

write_feather(as.data.frame(hidden.layer9),paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_hidden8_',"_jobid_",JobId,'.feather'))
write_feather(as.data.frame(hidden.layer10),paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_hidden1_',"_jobid_",JobId,'.feather'))
write_feather(as.data.frame(hidden.layer6),paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_hidden9_',"_jobid_",JobId,'.feather'))

