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


filename <- "dec20_sm_depind_gpnn_bbig_init_bbs" 
success.run <- c(1,3,4,5,7,8,9)
init.num <- ifelse(JobId %in% success.run, yes = JobId, no = sample(success.run,1))
prior.var <- 0.05 #was 0.05
learning_rate <- 0.99 #for slow decay starting less than 1
prior.var.bias <- 1
epoch <- 300 #was 500
beta.bb<- 0.5
lr.init <- learning_rate


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

print("Loading data")

#Age
age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/age_sex_strat_depind.feather'))
#age_tab <- age_tab[order(age_tab$id),].     #DOES THIS MESS UP ORDER
age <- age_tab$age
sex <-  as.numeric(age_tab$sex)
sex <- sapply(sex, function(x) replace(x, x==0,-1)) #Change female to -1, male to 1

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


# source("/well/nichols/users/qcv214/bnn2/res3/first_layer_gp4.R")
partial.gp.centroid<-t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/KGPNN/partial_gp_centroids_fixed_300.540.feather"))))
l.expan <- ncol(partial.gp.centroid)


time.taken <- Sys.time() - start.time
cat("Loading data complete in: ", time.taken)


print("Getting mini batch")
#Get minibatch index 
batch_size <- 500


#NN parameters
it.num <- 1

#Initial parameters for inverse gamma
alpha.init <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_dec15_sm_depind_gpnn_bbig_init_minalpha__jobid_",init.num,".csv"))$x #shape
beta.init <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_dec15_sm_depind_gpnn_bbig_init_minbeta__jobid_",init.num,".csv"))$x #scale


#Storing inv gamma
conj.alpha <- matrix(, nrow=n.mask,ncol=epoch*4)
conj.beta <- matrix(, nrow=n.mask,ncol=epoch*4)
conj.invgamma <-matrix(, nrow=n.mask,ncol=epoch*4)
# conj.cv <- matrix(, nrow=n.mask,ncol=epoch*4)

#Define init var
prior.var <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_dec15_sm_depind_gpnn_bbig_init_minpriorvar__jobid_",init.num,".csv"))$x#Mean of IG

#Fix prior var to be 0.1
# prior.var <- 1.5
y.sigma <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_dec15_sm_depind_gpnn_bbig_init_minsigma__jobid_",init.num,".csv"))$x
y.sigma.vec <- y.sigma

print("Initialisation")
#1 Initialisation
#1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
# theta.matrix <- matrix(,nrow=n.mask, ncol= n.expan)
weights <- as.matrix(read_feather(paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_dec15_sm_depind_gpnn_bbig_init_minweights__jobid_',init.num,'.feather')))
co.weights<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_dec15_sm_depind_gpnn_bbig_init_mincoweights__jobid_",init.num,".csv")))

#Initialising bias (to 0)
bias <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_dec15_sm_depind_gpnn_bbig_init_minbias__jobid_",init.num,".csv"))$x
co.bias <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_dec15_sm_depind_gpnn_bbig_init_mincobias__jobid_",init.num,".csv"))$x

num.lat.class<- length(co.bias)


time.train <-  Sys.time()

#Start epoch
for(e in 1:epoch){
  
  #randomise data
  mini.batch <- batch_split(left = train.test.ind$train, batch_size = batch_size)
  num.batch <- length(mini.batch$train)
  
  grad_x <- 0 #For BB
  
  time.epoch <-  Sys.time()
  #Start batch
  for(b in 1:num.batch){
    
    minibatch.size <- length(mini.batch$train[[b]])
    
    print(paste0("Epoch: ",e, ", batch number: ", b))
    #3 Feed it to next layer
    
    hidden.layer <- apply(t(t(res3.dat[mini.batch$train[[b]], ]  %*% t(weights)) + bias), 2, FUN = relu) #n x n.mask
    co.pre.hidden.layer <- t(t(co.dat[mini.batch$train[[b]], ] %*% t(co.weights)) + co.bias)
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
    
    hs_fit_SOI <- fast_normal_lm(age[mini.batch$train[[b]]],z.nb) #This also gives the bias term
    
    
    beta_fit <- data.frame(HS = c(partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[2:(l.expan+1)], #Main imaging effects
                                  hs_fit_SOI$post_mean$betacoef[(l.expan+2):(l.expan+2+num.lat.class-1)],            #Main Latent effect
                                  partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[(l.expan+2+num.lat.class):(l.expan+2+num.lat.class+l.expan-1)],
                                  partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[(l.expan+2+num.lat.class+l.expan):(l.expan+2+num.lat.class+l.expan*2-1)],
                                  partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[(l.expan+2+num.lat.class+l.expan*2):(l.expan+2+num.lat.class+l.expan*3-1)],
                                  partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[(l.expan+2+num.lat.class+l.expan*3):(l.expan+2+num.lat.class+l.expan*4-1)]
    ))
    l.bias <- hs_fit_SOI$post_mean$betacoef[1]
    
    hs_in.pred_SOI <- predict_fast_lm(hs_fit_SOI, z.nb, alpha = 0.95)$mean
    
    
    loss.train <- c(loss.train, mse(hs_in.pred_SOI,age[mini.batch$train[[b]]]))
    rsq.train <- c(rsq.train, rsqcal(age[mini.batch$train[[b]]],hs_in.pred_SOI))
    
    loss.train.male <- c(loss.train.male, mse(hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == 1)],age[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == 1)]))
    rsq.train.male <- c(rsq.train.male, rsqcal(age[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == 1)],hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == 1)]))
    
    loss.train.fmale <- c(loss.train.fmale, mse(hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == -1)],age[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == -1)]))
    rsq.train.f <- c(rsq.train.fmale, rsqcal(age[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == -1)],hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == -1)]))
    
    temp.sum.sum.sq <- apply(weights, 1, FUN = function(x) sum(x^2))
    
    #Note wrong MAP here. I have NOT incorporated intercept
    map.train <- c(map.train,n.train/2*log(y.sigma) +1/(2*y.sigma)*n.train*mse(hs_in.pred_SOI,age[mini.batch$train[[b]]]) +n.mask/2*log(y.sigma) +n.mask*p.dat/2*log(y.sigma) + 1/(2*y.sigma)*sum(1/prior.var*(temp.sum.sum.sq))  +1/2*sum(c(bias)^2) )
    
    #Validation
    #Layers
    
    
    hidden.layer.test <- apply(t(t(res3.dat[train.test.ind$test, ] %*% t(weights)) + bias), 2, FUN = relu)
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
    
    
    loss.val <- c(loss.val, mse(hs_pred_SOI,age[train.test.ind$test]))
    rsq.val <- c(rsq.val, rsqcal(age[train.test.ind$test],hs_pred_SOI))
    
    loss.val.male <- c(loss.val.male, mse(hs_pred_SOI[which(sex[train.test.ind$test] == 1)],age[train.test.ind$test][which(sex[train.test.ind$test] == 1)]))
    rsq.val.male <- c(rsq.val.male, rsqcal(age[train.test.ind$test][which(sex[train.test.ind$test] == 1)],hs_pred_SOI[which(sex[train.test.ind$test] == 1)]))
    
    loss.val.fmale <- c(loss.val.fmale, mse(hs_pred_SOI[which(sex[train.test.ind$test] == -1)],age[train.test.ind$test][which(sex[train.test.ind$test] == -1)]))
    rsq.val.fmale <- c(rsq.val.fmale, rsqcal(age[train.test.ind$test][which(sex[train.test.ind$test] == -1)],hs_pred_SOI[which(sex[train.test.ind$test] == -1)]))
    
    ##Keeping the last 5 epochs predictions
    if(e >= (epoch-5)){
      pred.train.ind <- c(pred.train.ind,mini.batch$train[[b]]) 
      pred.train.val <- c(pred.train.val,hs_in.pred_SOI)
      pred.test.ind <- c(pred.test.ind,train.test.ind$test) 
      pred.test.val <- c(pred.test.val,hs_pred_SOI) 
      #class
      class.train.val <- c(class.train.val, apply(co.hidden.layer,1,which.max) )
      class.test.val <- c(class.test.val, apply(co.hidden.layer.test,1,which.max) )
    }
    
    if(it.num < epoch*num.batch){
      #Update weight
      
      #4Update the full weights, fit GP against the full weights using HS-prior model to get normally dist thetas
      grad.loss <- age[mini.batch$train[[b]]] - hs_in.pred_SOI
      
      #Update weight
      grad <- array(,dim = c(minibatch.size,dim(weights)))
      
      #Each 
      for(j in 1:n.mask){ #nrow of weights = n.mask
        grad[,j,] <- -1/y.sigma*c(grad.loss)*(beta_fit$HS[j]+
                                                beta_fit$HS[n.mask+num.lat.class+1+(j-1)*num.lat.class]*c(co.hidden.layer[,1])+
                                                beta_fit$HS[n.mask+num.lat.class+2+(j-1)*num.lat.class]*c(co.hidden.layer[,2])+ 
                                                beta_fit$HS[n.mask+num.lat.class+3+(j-1)*num.lat.class]*c(co.hidden.layer[,3])+ 
                                                beta_fit$HS[n.mask+num.lat.class+4+(j-1)*num.lat.class]*c(co.hidden.layer[,4])
        )*c(relu.prime(hidden.layer[,j]))*res3.dat[mini.batch$train[[b]], ]  #######this is wrong shttt. sex shouldnt be there, it should be the sigmoid
      }
      #Take batch average
      grad.m <- apply(grad, c(2,3), mean)
      
      #update weights for non-imaging covariate
      #Update weight
      l.grad <- beta_fit$HS[(n.mask+1):(n.mask+4)] #Main effect first
      for(j in 1:n.mask){
        l.grad <- l.grad +beta_fit$HS[(n.mask+num.lat.class+1+(j-1)*num.lat.class):(n.mask+num.lat.class+4+(j-1)*num.lat.class)]
      }
      
      
      co.sm.grad <- apply(co.hidden.layer,1,softmax.prime,l.grad = l.grad) #Note that softmax.prime take in softmax output rather than the pre-softmax input
      #Then I need to times co.sm.grad by l.grad. I think I need l.grad to be inside softmax.prime
      #Then for the resulting post-3D-transformation of co.sm.grad, I want to time each 1st dim by c(grad.loss). This is as simple as ...*c(grad.loss) [have verified]
      
      co.sm.grad <- array(t(co.sm.grad),dim = c(ncol(co.sm.grad),sqrt(nrow(co.sm.grad)),sqrt(nrow(co.sm.grad))))
      grad.sum <- apply(co.sm.grad*(-1/y.sigma*c(grad.loss)), c(1,3), sum)
      
      
      co.grad.m<- t(grad.sum)%*%co.dat[mini.batch$train[[b]], ]/nrow(co.dat[mini.batch$train[[b]], ]) #n.lat class * num attr
      
      #Changed here, deleted - from 1/7
      co.grad.b.m <- c(colMeans(grad.sum))
      
      co.weights <- co.weights*(1-learning_rate) - learning_rate*co.grad.m 
      
      co.bias <- co.bias*(1-learning_rate) - learning_rate*co.grad.b.m
      
      
      #########Here is inefficiency
      grad.b <- matrix(, nrow = minibatch.size, ncol = length(bias))
      for(j in 1:n.mask){ #nrow of weights = n.mask
        grad.b[,j] <- -1/y.sigma*c(grad.loss)*(beta_fit$HS[j]+
                                                 beta_fit$HS[n.mask+num.lat.class+1+(j-1)*num.lat.class]*c(co.hidden.layer[,1])+ #r1c1
                                                 beta_fit$HS[n.mask+num.lat.class+2+(j-1)*num.lat.class]*c(co.hidden.layer[,2])+ #r1c2
                                                 beta_fit$HS[n.mask+num.lat.class+3+(j-1)*num.lat.class]*c(co.hidden.layer[,3])+ #r1c3
                                                 beta_fit$HS[n.mask+num.lat.class+4+(j-1)*num.lat.class]*c(co.hidden.layer[,4]) #r1c4
        )*c(relu.prime(hidden.layer[,j]))  #######this is wrong shttt. sex shouldnt be there, it should be the sigmoid
      }
      #Take batch average
      grad.b.m <- c(apply(grad.b, c(2), mean))
      
      grad.sigma.m <- mean(length(train.test.ind$train)/(2*y.sigma) - length(train.test.ind$train)/(2*y.sigma^2)*c(grad.loss)^2-1/(2*y.sigma^2)*sum(c(weights/prior.var)^2)+1/(2*y.sigma)*p.dat*n.mask)
      ####Note here of the static equal prior.var
      #Update theta matrix
      weights <- weights*(1-learning_rate*1/(prior.var*y.sigma)) - learning_rate*grad.m * length(train.test.ind$train)
      #Note that updating weights at the end will be missing the last batch of last epoch
      
      #Update bias
      bias <- bias*(1-learning_rate*1/(prior.var.bias)) - learning_rate*c(grad.b.m) * length(train.test.ind$train)
      
      # Update sigma
      y.sigma <- y.sigma - learning_rate*(grad.sigma.m)
      y.sigma.vec <- c(y.sigma.vec,y.sigma)
      
      delta_f <- c(c(weights/(prior.var*y.sigma) + grad.m*n.train),c(bias/prior.var.bias + grad.b.m*(n.train)),c(co.weights+co.grad.m*n.train),c(co.bias + co.grad.b.m*n.train))
      grad_x <- beta.bb*delta_f + (1-beta.bb)*grad_x
      x.param <- c(c(weights),c(bias),c(co.weights),c(co.bias))
      #Update Cv
      for(i in 1:n.mask){
        alpha.shape <- alpha.init[i] + length(weights[i,])/2
        # alpha.shape <- alpha.init[i] # Keep alpha the same
        beta.scale <- beta.init[i] + sum(weights[i,]^2)/(2*y.sigma)
        prior.var[i] <- rinvgamma(n = 1, alpha.shape, beta.scale)
        
        conj.alpha[i,it.num] <- alpha.shape
        conj.beta[i,it.num] <- beta.scale
        conj.invgamma[i,it.num] <- prior.var[i]
      }
    }
    
    it.num <- it.num +1
    
    # invisible(capture.output(ifelse(it.num >=2000, learning_rate <- lr.init*0.001,ifelse(it.num >=1000, learning_rate <- lr.init*0.01, learning_rate <- lr.init) )))
    # if((it.num %% 200) ==0){
    #   learning_rate <- learning_rate*0.1
    # }
    # learning_rate <- learning_rate
    print(paste0("training loss: ",mse(hs_in.pred_SOI,age[mini.batch$train[[b]]])))
    print(paste0("validation loss: ",mse(hs_pred_SOI,age[train.test.ind$test])))
  }
  
  print(paste0("epoch: ",e," out of ",epoch, ", time taken for this epoch: ",Sys.time() -time.epoch))
  print(paste0("sigma^2: ",y.sigma))
  if(e %in% c(epoch)){ #300 and 500
    
    salient.mat <- t(t(beta_fit$HS[1:n.mask]*weights) %*% t(apply(t(t(res3.dat[train.test.ind$train, ]  %*% t(weights)) + bias), 2, FUN = relu.prime)))
    
    gp.mask.hs <- res3.mask
    gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
    gp.mask.hs@datatype = 16
    gp.mask.hs@bitpix = 32
    writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/KGPNN/viz/',filename,"_epoch_",e,'_mainsal_',JobId))
    
    
    
    salient.mat <- t(t(beta_fit$HS[n.mask+num.lat.class+1+(seq(n.mask)-1)*num.lat.class]*weights) %*% t(c(co.hidden.layer[,1]) * apply(t(t(res3.dat[mini.batch$train[[b]], ]  %*% t(weights)) + bias), 2, FUN = relu.prime)))
    gp.mask.hs <- res3.mask
    gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
    gp.mask.hs@datatype = 16
    gp.mask.hs@bitpix = 32
    writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/KGPNN/viz/',filename,"_epoch_",e,'_inter1sal_',JobId))
    
    salient.mat <- t(t(beta_fit$HS[n.mask+num.lat.class+2+(seq(n.mask)-1)*num.lat.class]*weights) %*% t(c(co.hidden.layer[,2]) * apply(t(t(res3.dat[mini.batch$train[[b]], ]  %*% t(weights)) + bias), 2, FUN = relu.prime)))
    gp.mask.hs <- res3.mask
    gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
    gp.mask.hs@datatype = 16
    gp.mask.hs@bitpix = 32
    writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/KGPNN/viz/',filename,"_epoch_",e,'_inter2sal_',JobId))
    
    salient.mat <- t(t(beta_fit$HS[n.mask+num.lat.class+3+(seq(n.mask)-1)*num.lat.class]*weights) %*% t(c(co.hidden.layer[,3]) * apply(t(t(res3.dat[mini.batch$train[[b]], ]  %*% t(weights)) + bias), 2, FUN = relu.prime)))
    gp.mask.hs <- res3.mask
    gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
    gp.mask.hs@datatype = 16
    gp.mask.hs@bitpix = 32
    writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/KGPNN/viz/',filename,"_epoch_",e,'_inter3sal_',JobId))
    
    salient.mat <- t(t(beta_fit$HS[n.mask+num.lat.class+4+(seq(n.mask)-1)*num.lat.class]*weights) %*% t(c(co.hidden.layer[,4]) * apply(t(t(res3.dat[mini.batch$train[[b]], ]  %*% t(weights)) + bias), 2, FUN = relu.prime)))
    gp.mask.hs <- res3.mask
    gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
    gp.mask.hs@datatype = 16
    gp.mask.hs@bitpix = 32
    writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/KGPNN/viz/',filename,"_epoch_",e,'_inter4sal_',JobId))
    
    # salient.mat <- apply(t(t(res3.dat[train.test.ind$train, ]  %*% t(weights)) + bias), 2, FUN = relu.prime) %*%beta_fit$HS
    #salient.mat <- t(t(beta_fit$HS*weights) %*% t(apply(t(t(res3.dat[train.test.ind$train, ]  %*% t(weights)) + bias), 2, FUN = relu.prime)))
    # salient.mat <- t(t(beta_fit$HS[1:n.mask]*weights) %*% t(apply(t(t(res3.dat[train.test.ind$train, ]  %*% t(weights)) + bias), 2, FUN = relu.prime))) + t(t(beta_fit$HS[(n.mask+2):(n.mask*2+1)]*weights) %*% t(sex[train.test.ind$train] * apply(t(t(res3.dat[train.test.ind$train, ]  %*% t(weights)) + bias), 2, FUN = relu.prime))) 
    # 
    # gp.mask.hs <- res3.mask
    # gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
    # gp.mask.hs@datatype = 16
    # gp.mask.hs@bitpix = 32
    # writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/KGPNN/viz/',filename,"_epoch_",e,'_allsal_',JobId))
  }
  
  #BB
  #1 Feb, change indexing (3,2) to 2,1)... it's actually wrong. I am not saving the 1st lr, so 1st-3rd lr are literally the same.
  if(e >=2){
    diff_x = x.param - prev_x
    diff_grad_x = grad_x - prev_grad_x
    pre.learning_rate <- 1/num.batch*sum(diff_x*diff_x)/abs(sum(diff_x*diff_grad_x)) 
    pre.lr.vec <- c(pre.lr.vec, pre.learning_rate)
    
    ck.new <- ck.old^(1/(e-1))^(e-2)*(pre.learning_rate*phi(e))^(1/(e-1))
    
    # prod.lr.vec<- c(prod.lr.vec,prod(pre.lr.vec*phi(2:e))^(1/(e-1)))
    
    # learning_rate <- prod(pre.lr.vec*phi(2:e))^(1/(e-1))/phi(e)
    learning_rate <- ck.new/phi(e)
    lr.vec <- c(lr.vec, learning_rate)
    print(paste0("at epoch ",e," learning rate is ",learning_rate, ' (pre) ', pre.learning_rate))
    print(paste0("at epoch ",e,"product of pre.lr.vec is ", prod(pre.lr.vec),", product of phi is ",prod(phi(2:e)), " phi e is ", phi(e)))
    print(paste0("at epoch ",e,", ck.new is ", ck.new))
    
    ck.old <- ck.new
  }
  prev_x <- x.param
  prev_grad_x <- grad_x
  
}

pre.lr.vec <- c(lr.init,lr.init,pre.lr.vec[-length(pre.lr.vec)]) #Add first two learning rates
lr.vec <- c(lr.init,lr.init,lr.vec[-length(lr.vec)]) #Add first two learning rates

time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

write.csv(rbind(loss.train,loss.val),paste0("/well/nichols/users/qcv214/KGPNN/pile/re_",filename,"_loss_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(rsq.train,rsq.val),paste0("/well/nichols/users/qcv214/KGPNN/pile/re_",filename,"_rsq_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(loss.train.male,loss.val.male),paste0("/well/nichols/users/qcv214/KGPNN/pile/re_",filename,"_lossM_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(rsq.train.male,rsq.val.male),paste0("/well/nichols/users/qcv214/KGPNN/pile/re_",filename,"_rsqM_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(loss.train.fmale,loss.val.fmale),paste0("/well/nichols/users/qcv214/KGPNN/pile/re_",filename,"_lossF_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(rsq.train.fmale,rsq.val.fmale),paste0("/well/nichols/users/qcv214/KGPNN/pile/re_",filename,"_rsqF_","_jobid_",JobId,".csv"), row.names = FALSE)


write.csv(map.train,paste0("/well/nichols/users/qcv214/KGPNN/pile/re_",filename,"_map_","_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(weights),paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_weights_',"_jobid_",JobId,'.feather'))
write.csv(bias,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_bias_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(y.sigma.vec,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_sigma_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(l.bias,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_lbias_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(lr.vec,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_lr_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(pre.lr.vec,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_prelr_',"_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(c(beta_fit$HS )),paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_lweights_',"_jobid_",JobId,'.feather'))
write.csv(co.weights,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_coweights_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(co.bias,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_cobias_',"_jobid_",JobId,".csv"), row.names = FALSE)

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

#inv gamme param
write.csv(conj.alpha,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_alpha_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(conj.beta,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_beta_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(conj.invgamma,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_invgam_',"_jobid_",JobId,".csv"), row.names = FALSE)