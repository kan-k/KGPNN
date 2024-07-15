# R script

#May 6: Capped BBS.
#May 19, saving the last 200 epochs of predictions (and classes)

#28 May: change the res4 GP from R^2 to R and `____sm_depind_gpols_bbig_init_bbs` to `_____sm_depind_gpols_once_init_bbs`

#june 13, change number of subclasses from 4 to 10 , and param search from 1000 to 500


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


filename <- "july10_num_sm_gpnn_init_bbs" 
success.run <- c(3,5:10)
init.num <- ifelse(JobId %in% success.run, yes = JobId, no = sample(success.run,1))
prior.var <- 0.05 #was 0.05
learning_rate <- 0.99 #for slow decay starting less than 1
prior.var.bias <- 1
epoch <- 250 #was 500
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
age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/cog/agesex_strat.feather'))
#age_tab <- age_tab[order(age_tab$id),].     #DOES THIS MESS UP ORDER
lognum <- log(age_tab$num)

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


#mask
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
res3.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz'))

n.mask <- length(res3.mask.reg)
# n.expan <- choose(6+3,3)
p.dat <- ncol(res3.dat)
n.dat <- nrow(res3.dat)

# source("/well/nichols/users/qcv214/bnn2/res3/first_layer_gp4.R")
partial.gp.centroid.img<-t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/KGPNN/res4_partial_gp_centroids_fixed_200.540.feather"))))
l.expan <- ncol(partial.gp.centroid.img)

time.taken <- Sys.time() - start.time
cat("Data Loading data complete in: ", time.taken)
print(Sys.time())



print("Getting mini batch")
#Get minibatch index 
batch_size <- 500


#NN parameters
it.num <- 1

time.load <-  Sys.time()
print(Sys.time())

#Initial parameters for inverse gamma
alpha.init <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_july10_num_sm_gpnn_init_minalpha__jobid_",init.num,".csv"))$x #shape
beta.init <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_july10_num_sm_gpnn_init_minbeta__jobid_",init.num,".csv"))$x #scale


#Storing inv gamma
conj.alpha <- matrix(, nrow=n.mask,ncol=epoch*4)
conj.beta <- matrix(, nrow=n.mask,ncol=epoch*4)
conj.invgamma <-matrix(, nrow=n.mask,ncol=epoch*4)
# conj.cv <- matrix(, nrow=n.mask,ncol=epoch*4)

#Define init var
prior.var <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_july10_num_sm_gpnn_init_minpriorvar__jobid_",init.num,".csv"))$x#Mean of IG

#Fix prior var to be 0.1
# prior.var <- 1.5
y.sigma <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_july10_num_sm_gpnn_init_minsigma__jobid_",init.num,".csv"))$x
y.sigma.vec <- y.sigma

print("Initialisation")
#1 Initialisation
#1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
# theta.matrix <- matrix(,nrow=n.mask, ncol= n.expan)
weights <- as.matrix(read_feather(paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_july10_num_sm_gpnn_init_minweights__jobid_',init.num,'.feather')))
co.weights<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_july10_num_sm_gpnn_init_mincoweights__jobid_",init.num,".csv")))

#Initialising bias (to 0)
bias <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_july10_num_sm_gpnn_init_minbias__jobid_",init.num,".csv"))$x
co.bias <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_july10_num_sm_gpnn_init_mincobias__jobid_",init.num,".csv"))$x

num.lat.class<- length(co.bias)

#non-imaging covar
gp.coef <- as.matrix(read_feather(paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_july10_num_sm_gpnn_init_minLCgp__jobid_',init.num,'.feather')))
partial.gp.centroid <- rbind(partial.gp.centroid.img,gp.coef)

time.taken <- Sys.time() - time.load
cat("Previous Param load complete in: ", time.taken)
print(Sys.time())

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
    
    hidden.layer <-computeHiddenLayer(res3.dat[mini.batch$train[[b]], ], t(weights), bias)
    hidden.layer <-ReLU(hidden.layer)
    
    co.pre.hidden.layer <- t(t(co.dat[mini.batch$train[[b]], ] %*% t(co.weights)) + co.bias)
    co.hidden.layer <- softmax(co.pre.hidden.layer)
    
    hidden.layer.mixed <- cbind(hidden.layer, co.hidden.layer)
    
    ##
    
    
    # Generate polynomial features (linear terms)
    poly_features <- as.matrix(hidden.layer.mixed %*% partial.gp.centroid) #
    
    z.nb <- cbind(1,poly_features) #This is different from LASIR in the sense that the subgroup latent directly affect the output, whereas the group themselves dont. But then that can be modified easily.
    
    hs_fit_SOI <- fast_normal_lm(lognum[mini.batch$train[[b]]],z.nb) #This also gives the bias term
    
    
    beta_fit <- data.frame(HS = matrixVectorMultiply(partial.gp.centroid, hs_fit_SOI$post_mean$betacoef[2:(l.expan+1)]))
    
    l.bias <- hs_fit_SOI$post_mean$betacoef[1]
    
    hs_in.pred_SOI <- predict_fast_lm(hs_fit_SOI, z.nb, alpha = 0.95)$mean
    
    
    loss.train <- c(loss.train, mseCpp(hs_in.pred_SOI,lognum[mini.batch$train[[b]]]))
    rsq.train <- c(rsq.train, rsqCpp(lognum[mini.batch$train[[b]]],hs_in.pred_SOI))
    
    loss.train.male <- c(loss.train.male, mseCpp(hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == 1)],lognum[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == 1)]))
    rsq.train.male <- c(rsq.train.male, rsqCpp(lognum[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == 1)],hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == 1)]))
    
    loss.train.fmale <- c(loss.train.fmale, mseCpp(hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == -1)],lognum[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == -1)]))
    rsq.train.f <- c(rsq.train.fmale, rsqCpp(lognum[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == -1)],hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == -1)]))
    
    temp.sum.sum.sq <- apply(weights, 1, FUN = function(x) sum(x^2))
    
    #Note wrong MAP here. I have NOT incorporated intercept
    map.train <- c(map.train,n.train/2*log(y.sigma) +1/(2*y.sigma)*n.train*mse(hs_in.pred_SOI,lognum[mini.batch$train[[b]]]) +n.mask/2*log(y.sigma) +n.mask*p.dat/2*log(y.sigma) + 1/(2*y.sigma)*sum(1/prior.var*(temp.sum.sum.sq))  +1/2*sum(c(bias)^2) )
    
    
    #Validation
    #Layers
    
    hidden.layer.test <- apply(t(t(res3.dat[train.test.ind$test, ] %*% t(weights)) + bias), 2, FUN = relu)
    
    co.hidden.layer.test <- softmax(t(t(co.dat[train.test.ind$test, ] %*% t(co.weights)) + co.bias))
    
    hidden.layer.mixed.test <- cbind(hidden.layer.test, co.hidden.layer.test)
    
    poly_features.test <- as.matrix(hidden.layer.mixed.test %*% partial.gp.centroid)
    
    # Create the design matrix
    z.nb.test <- cbind(1,poly_features.test)
    
    #z.nb.test<- model.matrix(lognum[train.test.ind$test] ~ poly(as.matrix(hidden.layer.test %*% partial.gp.centroid), 1)*sex[train.test.ind$test])
    
    #Loss calculation
    # hs_pred_SOI <- hs_fit_SOI$post_mean$betacoef[1] + hidden.layer.test %*%beta_fit$HS
    hs_pred_SOI <- predict_fast_lm(hs_fit_SOI, z.nb.test, alpha = 0.95)$mean
    
    
    loss.val <- c(loss.val, mseCpp(hs_pred_SOI,lognum[train.test.ind$test]))
    rsq.val <- c(rsq.val, rsqCpp(lognum[train.test.ind$test],hs_pred_SOI))
    
    loss.val.male <- c(loss.val.male, mseCpp(hs_pred_SOI[which(sex[train.test.ind$test] == 1)],lognum[train.test.ind$test][which(sex[train.test.ind$test] == 1)]))
    rsq.val.male <- c(rsq.val.male, rsqCpp(lognum[train.test.ind$test][which(sex[train.test.ind$test] == 1)],hs_pred_SOI[which(sex[train.test.ind$test] == 1)]))
    
    loss.val.fmale <- c(loss.val.fmale, mseCpp(hs_pred_SOI[which(sex[train.test.ind$test] == -1)],lognum[train.test.ind$test][which(sex[train.test.ind$test] == -1)]))
    rsq.val.fmale <- c(rsq.val.fmale, rsqCpp(lognum[train.test.ind$test][which(sex[train.test.ind$test] == -1)],hs_pred_SOI[which(sex[train.test.ind$test] == -1)]))
    
    ##Keeping the last 5 epochs predictions
    if(e >= (epoch-200)){ #let's save the last 200 epochs.
      pred.train.ind <- c(pred.train.ind,mini.batch$train[[b]]) 
      pred.train.val <- c(pred.train.val,hs_in.pred_SOI)
      pred.test.ind <- c(pred.test.ind,train.test.ind$test) 
      pred.test.val <- c(pred.test.val,hs_pred_SOI) 
      #class
      class.train.val <- c(class.train.val, apply(co.hidden.layer,1,which.max) )
      class.test.val <- c(class.test.val, apply(co.hidden.layer.test,1,which.max) )
    }
    
    if(it.num < epoch*num.batch){
      
      beta.img <- beta_fit$HS[1:n.mask]
      
      #non-Imaging weights betas
      beta.nonimg <- beta_fit$HS[(n.mask+1):(n.mask+num.lat.class)]
      #Update weight
      
      #4Update the full weights, fit GP against the full weights using HS-prior model to get normally dist thetas
      grad.loss <- lognum[mini.batch$train[[b]]] - hs_in.pred_SOI
      
      #Update weight
      
      ##########      ##########      ##########      ##########      ##########
      #I need to update the gradient calculation.  Need to check how `updateWeights` handles the new beta.
      
      grad <- updateWeights(minibatch.size, n.mask,y.sigma, grad.loss, beta.img, hidden.layer,res3.dat[mini.batch$train[[b]], ]) #This already output 3D
      grad.m <- computeMean(grad)
      
      grad.b <- updateGradB(minibatch.size, n.mask,y.sigma, grad.loss, beta.img, hidden.layer)
      grad.b.m <- calculateColumnMeans(grad.b)
      
      
      #update weights for non-imaging covariate
      #########Here is inefficiency
      
      
      co.sm.grad <- apply(co.hidden.layer,1,softmax.prime,l.grad = beta.nonimg) #Note that softmax.prime take in softmax output rather than the pre-softmax input
      #Then I need to times co.sm.grad by l.grad. I think I need l.grad to be inside softmax.prime
      #Then for the resulting post-3D-transformation of co.sm.grad, I want to time each 1st dim by c(grad.loss). This is as simple as ...*c(grad.loss) [have verified]
      co.sm.grad <- array(t(co.sm.grad),dim = c(ncol(co.sm.grad),sqrt(nrow(co.sm.grad)),sqrt(nrow(co.sm.grad))))
      grad.sum <- apply(co.sm.grad*(-1/y.sigma*c(grad.loss)), c(1,3), sum)
      
      co.grad.m<- t(grad.sum)%*%co.dat[mini.batch$train[[b]], ]/nrow(co.dat[mini.batch$train[[b]], ]) #n.lat class * num attr
      
      co.grad.b.m <- c(colMeans(grad.sum))
      
      co.weights <- co.weights*(1-learning_rate) - learning_rate*co.grad.m 
      
      co.bias <- co.bias*(1-learning_rate) - learning_rate*co.grad.b.m
      
      
      # Update sigma
      
      ####This has to be changed
      
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
      
      #Update GP representation of latent classes
      # grad.gp.coef <- t(t(co.hidden.layer)*c(beta.nonimg))
      # grad.gp.coef.m <- 
      
      #######Need to check if this is right but hopefully
      gp.coef.grad <- (c(grad.loss)*co.hidden.layer)
      gp.coef.grad <- array(apply(gp.coef.grad,1,function(y) y %o% hs_fit_SOI$post_mean$betacoef[2:(l.expan+1)]),dim = c(num.lat.class,l.expan,minibatch.size)) #Now is num class x K x n
      gp.coef.grad.m <- apply(gp.coef.grad,c(1,2),mean)*(-1/y.sigma) #compute mean over minibatch (3rd dim), then factor by y.sigma
      
      #end result should be (n x 4 x 5456)
      
      gp.coef <- gp.coef*(1-learning_rate) - learning_rate*gp.coef.grad.m* n.train
      
      partial.gp.centroid <- rbind(partial.gp.centroid.img,gp.coef)
      
      #### I have added hidden layer here for non-imaging here
      delta_f <- c(c(weights/(prior.var*y.sigma) + grad.m*n.train),c(bias/prior.var.bias + grad.b.m*(n.train)),c(co.weights+co.grad.m*n.train),c(co.bias + co.grad.b.m*n.train), c(gp.coef + gp.coef.grad.m* n.train))
      
      grad_x <- beta.bb*delta_f + (1-beta.bb)*grad_x
      # x.param <- c(c(weights),c(bias))
      x.param <- c(c(weights),c(bias),c(co.weights),c(co.bias), c(gp.coef))
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
    
    print(paste0("training loss: ",mseCpp(hs_in.pred_SOI,lognum[mini.batch$train[[b]]])))
    print(paste0("validation loss: ",mseCpp(hs_pred_SOI,lognum[train.test.ind$test])))
  }
  
  print(paste0("epoch: ",e," out of ",epoch, ", time taken for this epoch: ",Sys.time() -time.epoch))
  print(paste0("sigma^2: ",y.sigma))
  # if(e %in% c(epoch)){ #300 and 500
  #   
  #   salient.mat <- t(t(beta_fit$HS[1:n.mask]*weights) %*% t(apply(t(t(res3.dat[train.test.ind$train, ]  %*% t(weights)) + bias), 2, FUN = relu.prime)))
  #   
  #   gp.mask.hs <- res3.mask
  #   gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
  #   gp.mask.hs@datatype = 16
  #   gp.mask.hs@bitpix = 32
  #   writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/KGPNN/viz/re_',filename,"_epoch_",e,'_mainsal_',JobId))
  #   
  #   
  #   
  #   salient.mat <- t(t(beta_fit$HS[n.mask+num.lat.class+1+(seq(n.mask)-1)*num.lat.class]*weights) %*% t(c(co.hidden.layer[,1]) * apply(t(t(res3.dat[mini.batch$train[[b]], ]  %*% t(weights)) + bias), 2, FUN = relu.prime)))
  #   gp.mask.hs <- res3.mask
  #   gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
  #   gp.mask.hs@datatype = 16
  #   gp.mask.hs@bitpix = 32
  #   writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/KGPNN/viz/re_',filename,"_epoch_",e,'_inter1sal_',JobId))
  #   
  #   salient.mat <- t(t(beta_fit$HS[n.mask+num.lat.class+2+(seq(n.mask)-1)*num.lat.class]*weights) %*% t(c(co.hidden.layer[,2]) * apply(t(t(res3.dat[mini.batch$train[[b]], ]  %*% t(weights)) + bias), 2, FUN = relu.prime)))
  #   gp.mask.hs <- res3.mask
  #   gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
  #   gp.mask.hs@datatype = 16
  #   gp.mask.hs@bitpix = 32
  #   writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/KGPNN/viz/re_',filename,"_epoch_",e,'_inter2sal_',JobId))
  #   
  #   salient.mat <- t(t(beta_fit$HS[n.mask+num.lat.class+3+(seq(n.mask)-1)*num.lat.class]*weights) %*% t(c(co.hidden.layer[,3]) * apply(t(t(res3.dat[mini.batch$train[[b]], ]  %*% t(weights)) + bias), 2, FUN = relu.prime)))
  #   gp.mask.hs <- res3.mask
  #   gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
  #   gp.mask.hs@datatype = 16
  #   gp.mask.hs@bitpix = 32
  #   writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/KGPNN/viz/re_',filename,"_epoch_",e,'_inter3sal_',JobId))
  #   
  #   salient.mat <- t(t(beta_fit$HS[n.mask+num.lat.class+4+(seq(n.mask)-1)*num.lat.class]*weights) %*% t(c(co.hidden.layer[,4]) * apply(t(t(res3.dat[mini.batch$train[[b]], ]  %*% t(weights)) + bias), 2, FUN = relu.prime)))
  #   gp.mask.hs <- res3.mask
  #   gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
  #   gp.mask.hs@datatype = 16
  #   gp.mask.hs@bitpix = 32
  #   writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/KGPNN/viz/re_',filename,"_epoch_",e,'_inter4sal_',JobId))
  #   
  #   # salient.mat <- apply(t(t(res3.dat[train.test.ind$train, ]  %*% t(weights)) + bias), 2, FUN = relu.prime) %*%beta_fit$HS
  #   #salient.mat <- t(t(beta_fit$HS*weights) %*% t(apply(t(t(res3.dat[train.test.ind$train, ]  %*% t(weights)) + bias), 2, FUN = relu.prime)))
  #   # salient.mat <- t(t(beta_fit$HS[1:n.mask]*weights) %*% t(apply(t(t(res3.dat[train.test.ind$train, ]  %*% t(weights)) + bias), 2, FUN = relu.prime))) + t(t(beta_fit$HS[(n.mask+2):(n.mask*2+1)]*weights) %*% t(sex[train.test.ind$train] * apply(t(t(res3.dat[train.test.ind$train, ]  %*% t(weights)) + bias), 2, FUN = relu.prime))) 
  #   # 
  #   # gp.mask.hs <- res3.mask
  #   # gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
  #   # gp.mask.hs@datatype = 16
  #   # gp.mask.hs@bitpix = 32
  #   # writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/KGPNN/viz/',filename,"_epoch_",e,'_allsal_',JobId))
  # }
  
  #BB
  #1 Feb, change indexing (3,2) to 2,1)... it's actually wrong. I am not saving the 1st lr, so 1st-3rd lr are literally the same.
  if(e >=2){
    diff_x = x.param - prev_x
    diff_grad_x = grad_x - prev_grad_x
    
    ########
    if (abs(sum(diff_x*diff_grad_x)) == 0){
      pre.learning_rate <- 0.025 #0.25
    } else { 
      pre.learning_rate <- 1/num.batch*sum(diff_x*diff_x)/abs(sum(diff_x*diff_grad_x))
    }
    pre.learning_rate <- sign(pre.learning_rate)*min(abs(pre.learning_rate),0.1) #was 0.8
    ########
    
    # pre.learning_rate <- 1/num.batch*sum(diff_x*diff_x)/abs(sum(diff_x*diff_grad_x)) 
    pre.lr.vec <- c(pre.lr.vec, pre.learning_rate)
    
    ck.new <- ck.old^(1/(e-1))^(e-2)*(pre.learning_rate*phi(e))^(1/(e-1))
    
    # prod.lr.vec<- c(prod.lr.vec,prod(pre.lr.vec*phi(2:e))^(1/(e-1)))
    
    # learning_rate <- prod(pre.lr.vec*phi(2:e))^(1/(e-1))/phi(e)
    learning_rate <- ck.new/phi(e)
    lr.vec <- c(lr.vec, learning_rate)
    print(paste0("at epoch ",e," learning rate is ",learning_rate, ' (pre) ', pre.learning_rate))
    # print(paste0("at epoch ",e,"product of pre.lr.vec is ", prod(pre.lr.vec),", product of phi is ",prod(phi(2:e)), " phi e is ", phi(e)))
    # print(paste0("at epoch ",e,", ck.new is ", ck.new))
    
    ck.old <- ck.new
  }
  prev_x <- x.param
  prev_grad_x <- grad_x
  
}

pre.lr.vec <- c(lr.init,lr.init,pre.lr.vec[-length(pre.lr.vec)]) #Add first two learning rates
lr.vec <- c(lr.init,lr.init,lr.vec[-length(lr.vec)]) #Add first two learning rates

time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

write.csv(rbind(loss.train,loss.val),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_loss_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(rsq.train,rsq.val),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_rsq_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(loss.train.male,loss.val.male),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_lossM_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(rsq.train.male,rsq.val.male),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_rsqM_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(loss.train.fmale,loss.val.fmale),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_lossF_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(rsq.train.fmale,rsq.val.fmale),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_rsqF_","_jobid_",JobId,".csv"), row.names = FALSE)

write.csv(map.train,paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_map_","_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(weights),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_weights_',"_jobid_",JobId,'.feather'))
write.csv(bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_bias_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(y.sigma.vec,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_sigma_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(l.bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_lbias_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(lr.vec,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_lr_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(pre.lr.vec,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_prelr_',"_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(c(beta_fit$HS )),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_concatlweights_',"_jobid_",JobId,'.feather'))
write.csv(co.weights,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_coweights_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(co.bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_cobias_',"_jobid_",JobId,".csv"), row.names = FALSE)


temp.frame <- as.data.frame(rbind(pred.train.ind,pred.train.val,class.train.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
# temp.frame<- t(tail(t(temp.frame),length(train.test.ind$train)*5))
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_inpred_',"_jobid_",JobId,'.feather'))
temp.frame <- as.data.frame(rbind(pred.test.ind,pred.test.val,class.test.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
# temp.frame<- t(tail(t(temp.frame),length(train.test.ind$test)*5))
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_outpred_',"_jobid_",JobId,'.feather'))


#inv gamme param
write.csv(conj.alpha,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_alpha_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(conj.beta,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_beta_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(conj.invgamma,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_invgam_',"_jobid_",JobId,".csv"), row.names = FALSE)