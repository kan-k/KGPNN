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


filename <- "apr1_sm_depind15_gpnn_bbig_init_famp_min" 
#success.run <- c(3,8)
#init.num <- ifelse(JobId %in% success.run, yes = JobId, no = sample(success.run,1))
init.num <- 10
prior.var <- 0.05 #was 0.05
learning_rate <- 0.99 #for slow decay starting less than 1
prior.var.bias <- 1
epoch <- 400 #was 500
beta.bb<- 0.5
lr.init <- learning_rate


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

print("Loading data")

#Age
#age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/sim15_age.feather'))
age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/sim15_age_femaleAmp.feather'))
#age_tab <- age_tab[order(age_tab$id),].     #DOES THIS MESS UP ORDER
age <- age_tab$pred_amp
sex <-  as.numeric(age_tab$sex)
sex <- sapply(sex, function(x) replace(x, x==0,-1)) #Change female to -1, male to 1

train.test.ind <- list()
train.test.ind$test <- read.csv('/well/nichols/users/qcv214/KGPNN/sim15_test_index.csv')$x
train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/KGPNN/sim15_train_index.csv')$x
n.train <- length(train.test.ind$train)

depind <- age_tab$DepInd
# quantile_thresholds <- quantile(depind[train.test.ind$train], probs = seq(0, 1, by = 0.1))

#Define another group variable called age group which is directly associated with what we are predicting.
#age.group <- ifelse(age > mean(age), yes = 1, no = -1)

quantile_thresholds.15 <- median(depind) #can use MEDIAN since length is even, so it takes the middle value, it's 3 and 31
dep.group1 <- ifelse(depind < quantile_thresholds.15, yes =1, no = 0) #bottom 15
dep.group2 <- ifelse(depind >= quantile_thresholds.15, yes =1, no = 0) #top 15
co.dat <- cbind(sex,dep.group1,dep.group2)

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


# source("/well/nichols/users/qcv214/bnn2/res3/first_layer_gp4.R")
partial.gp.centroid<-t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/KGPNN/partial_gp_centroids_fixed_300.540.feather"))))
l.expan <- ncol(partial.gp.centroid)


time.taken <- Sys.time() - start.time
cat("Loading data complete in: ", time.taken)


#Initial parameters for inverse gamma
alpha.init <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_apr1_sm_depind15_gpnn_bbig_init_famp_minalpha__jobid_",init.num,".csv"))$x #shape
beta.init <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_apr1_sm_depind15_gpnn_bbig_init_famp_minbeta__jobid_",init.num,".csv"))$x #scale



#Define init var
prior.var <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_apr1_sm_depind15_gpnn_bbig_init_famp_minpriorvar__jobid_",init.num,".csv"))$x#Mean of IG

#Fix prior var to be 0.1
# prior.var <- 1.5
y.sigma <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_apr1_sm_depind15_gpnn_bbig_init_famp_minsigma__jobid_",init.num,".csv"))$x
y.sigma.vec <- y.sigma

print("Initialisation")
#1 Initialisation
#1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
# theta.matrix <- matrix(,nrow=n.mask, ncol= n.expan)
weights <- as.matrix(read_feather(paste0( '/well/nichols/users/qcv214/KGPNN/pile/sim_apr1_sm_depind15_gpnn_bbig_init_famp_minweights__jobid_',init.num,'.feather')))
co.weights<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_apr1_sm_depind15_gpnn_bbig_init_famp_mincoweights__jobid_",init.num,".csv")))

#Initialising bias (to 0)
bias <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_apr1_sm_depind15_gpnn_bbig_init_famp_minbias__jobid_",init.num,".csv"))$x
co.bias <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_apr1_sm_depind15_gpnn_bbig_init_famp_mincobias__jobid_",init.num,".csv"))$x

num.lat.class<- length(co.bias)


time.train <-  Sys.time()

########### Training
hidden.layer <-computeHiddenLayer(res3.dat[train.test.ind$train, ], t(weights), bias)
hidden.layer <-ReLU(hidden.layer)
co.pre.hidden.layer <- t(t(co.dat[train.test.ind$train, ] %*% t(co.weights)) + co.bias)
co.hidden.layer <- softmax(co.pre.hidden.layer)

# Generate polynomial features (linear terms)
poly_features <- as.matrix(hidden.layer %*% partial.gp.centroid) #
# Create the interaction features
interaction_features <- sapply(1:ncol(co.hidden.layer), function(i) {
  sapply(1:ncol(poly_features), function(j) {
    elementwise_product(co.hidden.layer[, i], poly_features[, j])
  })
})
# Create the design matrix
interaction_features <- array(data = interaction_features, dim = c(nrow(co.hidden.layer), ncol(co.hidden.layer) * ncol(poly_features))) #m1n1m1n2m1n3m1n4

# Create the design matrix
z.nb <- cbind(1,poly_features, co.hidden.layer, interaction_features) #This is different from LASIR in the sense that the subgroup latent directly affect the output, whereas the group themselves dont. But then that can be modified easily.

hs_fit_SOI <- fast_normal_lm(age[train.test.ind$train],z.nb) #This also gives the bias term 


beta_fit <- data.frame(HS = c(partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[2:(l.expan+1)], #Main imaging effects
                              hs_fit_SOI$post_mean$betacoef[(l.expan+2):(l.expan+2+num.lat.class-1)],            #Main Latent effect
                              partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[(l.expan+2+num.lat.class):(l.expan+2+num.lat.class+l.expan-1)],
                              partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[(l.expan+2+num.lat.class+l.expan):(l.expan+2+num.lat.class+l.expan*2-1)],
                              partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[(l.expan+2+num.lat.class+l.expan*2):(l.expan+2+num.lat.class+l.expan*3-1)],
                              partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[(l.expan+2+num.lat.class+l.expan*3):(l.expan+2+num.lat.class+l.expan*4-1)]
))
l.bias <- hs_fit_SOI$post_mean$betacoef[1]

hs_in.pred_SOI <- predict_fast_lm(hs_fit_SOI, z.nb, alpha = 0.95)$mean


loss.train <- c(loss.train, mseCpp(hs_in.pred_SOI,age[train.test.ind$train]))
rsq.train <- c(rsq.train, rsqCpp(age[train.test.ind$train],hs_in.pred_SOI))

loss.train.male <- c(loss.train.male, mseCpp(hs_in.pred_SOI[which(sex[train.test.ind$train] == 1)],age[train.test.ind$train][which(sex[train.test.ind$train] == 1)]))
rsq.train.male <- c(rsq.train.male, rsqCpp(age[train.test.ind$train][which(sex[train.test.ind$train] == 1)],hs_in.pred_SOI[which(sex[train.test.ind$train] == 1)]))

loss.train.fmale <- c(loss.train.fmale, mseCpp(hs_in.pred_SOI[which(sex[train.test.ind$train] == -1)],age[train.test.ind$train][which(sex[train.test.ind$train] == -1)]))
rsq.train.f <- c(rsq.train.fmale, rsqCpp(age[train.test.ind$train][which(sex[train.test.ind$train] == -1)],hs_in.pred_SOI[which(sex[train.test.ind$train] == -1)]))


#Validation
#Layers

hidden.layer.test <-computeHiddenLayer(res3.dat[train.test.ind$test, ], t(weights), bias)
hidden.layer.test <-ReLU(hidden.layer.test)

poly_features.test <- as.matrix(hidden.layer.test %*% partial.gp.centroid)

co.hidden.layer.test <- softmax(t(t(co.dat[train.test.ind$test, ] %*% t(co.weights)) + co.bias))

poly_features.test <- as.matrix(hidden.layer.test %*% partial.gp.centroid)
# Create the interaction features
interaction_features.test <- sapply(1:ncol(co.hidden.layer.test), function(i) {
  sapply(1:ncol(poly_features.test), function(j) {
    elementwise_product(co.hidden.layer.test[, i], poly_features.test[, j])
  })
})
# Create the design matrix
interaction_features.test <- array(data = interaction_features.test, dim = c(nrow(co.hidden.layer.test), ncol(co.hidden.layer.test) * ncol(poly_features.test))) #m1n1m1n2m1n3m1n4
# Create the design matrix
z.nb.test <- cbind(1,poly_features.test, co.hidden.layer.test, interaction_features.test)

#z.nb.test<- model.matrix(age[train.test.ind$test] ~ poly(as.matrix(hidden.layer.test %*% partial.gp.centroid), 1)*sex[train.test.ind$test])

#Loss calculation
# hs_pred_SOI <- hs_fit_SOI$post_mean$betacoef[1] + hidden.layer.test %*%beta_fit$HS
hs_pred_SOI <- predict_fast_lm(hs_fit_SOI, z.nb.test, alpha = 0.95)$mean


loss.val <- c(loss.val, mseCpp(hs_pred_SOI,age[train.test.ind$test]))
rsq.val <- c(rsq.val, rsqCpp(age[train.test.ind$test],hs_pred_SOI))

loss.val.male <- c(loss.val.male, mseCpp(hs_pred_SOI[which(sex[train.test.ind$test] == 1)],age[train.test.ind$test][which(sex[train.test.ind$test] == 1)]))
rsq.val.male <- c(rsq.val.male, rsqCpp(age[train.test.ind$test][which(sex[train.test.ind$test] == 1)],hs_pred_SOI[which(sex[train.test.ind$test] == 1)]))

loss.val.fmale <- c(loss.val.fmale, mseCpp(hs_pred_SOI[which(sex[train.test.ind$test] == -1)],age[train.test.ind$test][which(sex[train.test.ind$test] == -1)]))
rsq.val.fmale <- c(rsq.val.fmale, rsqCpp(age[train.test.ind$test][which(sex[train.test.ind$test] == -1)],hs_pred_SOI[which(sex[train.test.ind$test] == -1)]))


pred.train.ind <- c(pred.train.ind,train.test.ind$train) 
pred.train.val <- c(pred.train.val,hs_in.pred_SOI)
pred.test.ind <- c(pred.test.ind,train.test.ind$test) 
pred.test.val <- c(pred.test.val,hs_pred_SOI) 
#class
class.train.val <- c(class.train.val, apply(co.hidden.layer,1,which.max) )
class.test.val <- c(class.test.val, apply(co.hidden.layer.test,1,which.max) )

time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)
write.csv(rbind(loss.train,loss.val),paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_",filename,"_loss_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(rsq.train,rsq.val),paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_",filename,"_rsq_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(loss.train.male,loss.val.male),paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_",filename,"_lossM_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(rsq.train.male,rsq.val.male),paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_",filename,"_rsqM_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(loss.train.fmale,loss.val.fmale),paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_",filename,"_lossF_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(rsq.train.fmale,rsq.val.fmale),paste0("/well/nichols/users/qcv214/KGPNN/pile/sim_",filename,"_rsqF_","_jobid_",JobId,".csv"), row.names = FALSE)

temp.frame <- as.data.frame(rbind(pred.train.ind,pred.train.val,class.train.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
# temp.frame<- t(tail(t(temp.frame),length(train.test.ind$train)*5))
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/pile/sim_',filename,'_inpred_',"_jobid_",JobId,'.feather'))
temp.frame <- as.data.frame(rbind(pred.test.ind,pred.test.val,class.test.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
# temp.frame<- t(tail(t(temp.frame),length(train.test.ind$test)*5))
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/pile/sim_',filename,'_outpred_',"_jobid_",JobId,'.feather'))

