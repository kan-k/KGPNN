# R script

#sep 22, adjusting MAP loss, deleting the term for hidden layer prior since we use lm() now


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


filename <- "sep22_pm_sm_gpols_12init_sgld_lr99_run2" 
success.run <- 1:10
init.num <- ifelse(JobId %in% success.run, yes = JobId, no = sample(success.run,1))
prior.var <- 0.05 #was 0.05

# start.b <- 1 #Originally 1e9
# start.a <- 1e-6
# start.gamma <- 3
# learning_rate <- start.a*(start.b+1)^(-start.gamma) #for slow decay starting less than 1 #
learning_rate <- 0.99#1e-6 Change below

prior.var.bias <- 1
epoch <- 750 #was 500
record.epoch <- 500 #epoch 
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


print("Loading data")

#Age
age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/cog/agesex_strat2.feather'))
#age_tab <- age_tab[order(age_tab$id),].     #DOES THIS MESS UP ORDER
lognum <- age_tab$pm_tf

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
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')

res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
res3.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz'))

n.mask <- length(res3.mask.reg)
# n.expan <- choose(10+3,3) #this should correspond to dec_vec given in res4_first_layer_gp
p.dat <- ncol(res3.dat)
n.dat <- nrow(res3.dat)

#For specifying n.expan.degree
n.expan <- choose(20+3,3)

#Pre-transform
print(paste0("load up res3 gp"))
source("/well/nichols/users/qcv214/bnn2/res3/first_layer_gp5_50.R")

print(paste0("load up res3 gp [DONE]"))
print(paste0("dim res3 gp"))
print(dim(res3.dat))
mult.dat <- function(X) as.matrix(res3.dat %*% X)
mult.thea <- function(X,theta) rowSums(X*theta)
res3.dat <- array(t(apply(partial.gp,MARGIN = c(1),mult.dat)), dim =c(n.mask,n.dat,n.expan)) #rename res3.dat as the product of data and expansion

################################################################################################################################################
for(num.lat.class.select in c(2)){
  
  prior.var <- 0.05 #was 0.05
  learning_rate <- 0.99 #for slow decay starting less than 1
  # learning_rate <- 1e-6
  
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
  
  
  time.taken <- Sys.time() - start.time
  cat("Loading data complete in: ", time.taken)
  print(Sys.time())
  
  
  print("Getting mini batch")
  #Get minibatch index 
  batch_size <- 500
  
  
  #NN parameters
  it.num <- 1
  
  #Initial parameters for inverse gamma
  alpha.init <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_aug22_pm_sm_gpols_12init",'_K',num.lat.class.select,"_minalpha__jobid_",2,".csv"))$x #shape
  beta.init <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_aug22_pm_sm_gpols_12init",'_K',num.lat.class.select,"_minbeta__jobid_",2,".csv"))$x #scale
  
  
  #Storing inv gamma
  conj.alpha <- matrix(, nrow=n.mask,ncol=epoch*4)
  conj.beta <- matrix(, nrow=n.mask,ncol=epoch*4)
  conj.invgamma <-matrix(, nrow=n.mask,ncol=epoch*4)
  # conj.cv <- matrix(, nrow=n.mask,ncol=epoch*4)
  
  #Define init var
  prior.var <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_aug22_pm_sm_gpols_12init",'_K',num.lat.class.select,"_minpriorvar__jobid_",2,".csv"))$x#Mean of IG
  
  #Fix prior var to be 0.1
  # prior.var <- 1.5
  y.sigma <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_aug22_pm_sm_gpols_12init",'_K',num.lat.class.select,"_minsigma__jobid_",2,".csv"))$x
  y.sigma.vec <- y.sigma
  
  print("Initialisation")
  #1 Initialisation
  #1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
  # theta.matrix <- matrix(,nrow=n.mask, ncol= n.expan)
  theta.matrix <- as.matrix(read_feather(paste0( "/well/nichols/users/qcv214/KGPNN/cog/pile/re_aug22_pm_sm_gpols_12init",'_K',num.lat.class.select,"_mintheta__jobid_",2,'.feather')))
  co.weights<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_aug22_pm_sm_gpols_12init",'_K',num.lat.class.select,"_mincoweights__jobid_",2,".csv")))
  
  #Initialising bias (to 0)
  bias <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_aug22_pm_sm_gpols_12init",'_K',num.lat.class.select,"_minbias__jobid_",2,".csv"))$x
  co.bias <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_aug22_pm_sm_gpols_12init",'_K',num.lat.class.select,"_mincobias__jobid_",2,".csv"))$x
  
  num.lat.class<- length(co.bias)
  
  gaus.sd <- 0
  
  
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
      
      # print(dim(theta.matrix))
      # print(dim(res3.dat[, mini.batch$train[[b]], ]))
      hidden.layer <- apply(t(apply(res3.dat[, mini.batch$train[[b]], ],MARGIN = 2,FUN = mult.thea,thet=theta.matrix) + bias), 2, FUN = relu) #not C++ optim, this collapes into two dimension
      #########Can optimise later since I already have the raw code. OPTIMISE the above
      co.pre.hidden.layer <- t(t(co.dat[mini.batch$train[[b]], ] %*% t(co.weights)) + co.bias)
      co.hidden.layer <- softmax(co.pre.hidden.layer)
      # Generate polynomial features (linear terms)
      poly_features <- as.matrix(hidden.layer) #
      interaction_features <- sapply(1:ncol(co.hidden.layer), function(i) {
        sapply(1:ncol(poly_features), function(j) {
          elementwise_product(co.hidden.layer[, i], poly_features[, j])
        })
      })
      # Create the design matrix
      interaction_features <- array(data = interaction_features, dim = c(nrow(co.hidden.layer), ncol(co.hidden.layer) * ncol(poly_features))) #m1n1m1n2m1n3m1n4
      z.nb <- cbind(poly_features, co.hidden.layer, interaction_features) #This is different from LASIR in the sense that the subgroup latent directly affect the output, whereas the group themselves dont. But then that can be modified easily.
      
      fit.lm <- lm(lognum[mini.batch$train[[b]]] ~ z.nb)
      
      l.weights <- coefficients(fit.lm)[-1]
      l.weights[is.na(l.weights)] <- 0
      beta_fit <- data.frame(HS = l.weights)
      l.bias <- coefficients(fit.lm)[1]
      
      hs_in.pred_SOI <- l.bias + z.nb%*%beta_fit$HS
      
      loss.train <- c(loss.train, mseCpp(hs_in.pred_SOI,lognum[mini.batch$train[[b]]]))
      rsq.train <- c(rsq.train, rsqCpp(lognum[mini.batch$train[[b]]],hs_in.pred_SOI))
      
      loss.train.male <- c(loss.train.male, mseCpp(hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == 1)],lognum[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == 1)]))
      rsq.train.male <- c(rsq.train.male, rsqCpp(lognum[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == 1)],hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == 1)]))
      
      loss.train.fmale <- c(loss.train.fmale, mseCpp(hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == -1)],lognum[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == -1)]))
      rsq.train.f <- c(rsq.train.fmale, rsqCpp(lognum[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == -1)],hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == -1)]))
      
      temp.sum.sum.sq <- apply(theta.matrix, 1, FUN = function(x) sum(x^2))
      
      #Note wrong MAP here. I have NOT incorporated intercept
      # map.train <- c(map.train,n.train/2*log(y.sigma) +1/(2*y.sigma)*n.train*mseCpp(hs_in.pred_SOI,lognum[mini.batch$train[[b]]]) +n.mask/2*log(y.sigma) +n.mask*n.expan/2*log(y.sigma) + 1/(2*y.sigma)*sum(1/prior.var*(temp.sum.sum.sq))  +1/2*sum(c(bias)^2) )
      
      map.train <- c(map.train,n.train/2*log(y.sigma) +1/(2*y.sigma)*n.train*mseCpp(hs_in.pred_SOI,lognum[mini.batch$train[[b]]]) +n.mask*n.expan/2*log(y.sigma) + 1/(2*y.sigma)*sum(1/prior.var*(temp.sum.sum.sq))  +1/2*sum(c(bias)^2) )
      
      hidden.layer.test <- apply(t(apply(res3.dat[, train.test.ind$test, ],MARGIN = 2,FUN = mult.thea,thet=theta.matrix) + bias), 2, FUN = relu)
      poly_features.test <- as.matrix(hidden.layer.test)
      
      co.hidden.layer.test <- softmax(t(t(co.dat[train.test.ind$test, ] %*% t(co.weights)) + co.bias))
      interaction_features.test <- sapply(1:ncol(co.hidden.layer.test), function(i) {
        sapply(1:ncol(poly_features.test), function(j) {
          elementwise_product(co.hidden.layer.test[, i], poly_features.test[, j])
        })
      })
      # Create the design matrix
      interaction_features.test <- array(data = interaction_features.test, dim = c(nrow(co.hidden.layer.test), ncol(co.hidden.layer.test) * ncol(poly_features.test))) #m1n1m1n2m1n3m1n4
      
      # Create the design matrix
      z.nb.test <- cbind(poly_features.test, co.hidden.layer.test, interaction_features.test)
      hs_pred_SOI <- l.bias + z.nb.test %*%beta_fit$HS
      
      
      loss.val <- c(loss.val, mseCpp(hs_pred_SOI,lognum[train.test.ind$test]))
      rsq.val <- c(rsq.val, rsqCpp(lognum[train.test.ind$test],hs_pred_SOI))
      
      loss.val.male <- c(loss.val.male, mseCpp(hs_pred_SOI[which(sex[train.test.ind$test] == 1)],lognum[train.test.ind$test][which(sex[train.test.ind$test] == 1)]))
      rsq.val.male <- c(rsq.val.male, rsqCpp(lognum[train.test.ind$test][which(sex[train.test.ind$test] == 1)],hs_pred_SOI[which(sex[train.test.ind$test] == 1)]))
      
      loss.val.fmale <- c(loss.val.fmale, mseCpp(hs_pred_SOI[which(sex[train.test.ind$test] == -1)],lognum[train.test.ind$test][which(sex[train.test.ind$test] == -1)]))
      rsq.val.fmale <- c(rsq.val.fmale, rsqCpp(lognum[train.test.ind$test][which(sex[train.test.ind$test] == -1)],hs_pred_SOI[which(sex[train.test.ind$test] == -1)]))
      
      ##Keeping the last 5 epochs predictions
      if(e >= (epoch-record.epoch)){ #let's save the last 200 epochs.
        pred.train.ind <- c(pred.train.ind,mini.batch$train[[b]]) 
        pred.train.val <- c(pred.train.val,hs_in.pred_SOI)
        pred.test.ind <- c(pred.test.ind,train.test.ind$test) 
        pred.test.val <- c(pred.test.val,hs_pred_SOI) 
        #class
        class.train.val <- c(class.train.val, apply(co.hidden.layer,1,which.max) )
        class.test.val <- c(class.test.val, apply(co.hidden.layer.test,1,which.max) )
      }
      
      if(it.num < epoch*num.batch){
        
        #4Update the full weights, fit GP against the full weights using HS-prior model to get normally dist thetas
        grad.loss <- lognum[mini.batch$train[[b]]] - hs_in.pred_SOI      
        #Update weight
        grad <- updateThetainter(minibatch.size, n.mask, num.lat.class,y.sigma, grad.loss, beta_fit$HS, hidden.layer,co.hidden.layer,res3.dat[,mini.batch$train[[b]], ]) #This already output 3D
        grad.m <- computeMean(grad)
        grad.b <- updateBiasGPinter(minibatch.size, n.mask, num.lat.class,y.sigma, grad.loss, beta_fit$HS, hidden.layer,co.hidden.layer)
        grad.b.m <- calculateColumnMeans(grad.b)
        l.grad <- beta_fit$HS[(n.mask+1):(n.mask+num.lat.class)] #Main effect first
        for(j in 1:n.mask){
          l.grad <- l.grad +beta_fit$HS[(n.mask+num.lat.class+1+(j-1)*num.lat.class):(n.mask+num.lat.class+num.lat.class+(j-1)*num.lat.class)]
        }
        co.sm.grad <- apply(co.hidden.layer,1,softmax.prime,l.grad = l.grad) #Note that softmax.prime take in softmax output rather than the pre-softmax input
        co.sm.grad <- array(t(co.sm.grad),dim = c(ncol(co.sm.grad),sqrt(nrow(co.sm.grad)),sqrt(nrow(co.sm.grad))))
        grad.sum <- apply(co.sm.grad*(-1/y.sigma*c(grad.loss)), c(1,3), sum)
        
        co.grad.m<- t(grad.sum)%*%co.dat[mini.batch$train[[b]], ]/nrow(co.dat[mini.batch$train[[b]], ]) #n.lat class * num attr
        co.grad.b.m <- c(colMeans(grad.sum))
        
        co.weights <- co.weights*(1-learning_rate) - learning_rate*co.grad.m - matrix(rnorm(ncol(co.dat)*num.lat.class,0,gaus.sd), ncol = ncol(co.dat), nrow = num.lat.class)
        co.bias <- co.bias*(1-learning_rate) - learning_rate*co.grad.b.m - rnorm(num.lat.class,0,gaus.sd)
        
        grad.sigma.m <- mean(length(train.test.ind$train)/(2*y.sigma) - length(train.test.ind$train)/(2*y.sigma^2)*c(grad.loss)^2-1/(2*y.sigma^2)*sum(c(theta.matrix/prior.var)^2)+1/(2*y.sigma)*n.expan*n.mask)
        
        ####Note here of the static equal prior.var
        #Update theta matrix
        theta.matrix <- theta.matrix*(1-learning_rate*1/(prior.var*y.sigma)) - learning_rate*grad.m * length(train.test.ind$train) - matrix(rnorm(n.mask*n.expan,0,gaus.sd), ncol = n.expan, nrow = n.mask)
        #Update bias
        bias <- bias*(1-learning_rate*1/(prior.var.bias)) - learning_rate*c(grad.b.m) * length(train.test.ind$train) - rnorm(n.mask,0,gaus.sd)
        
        # Update sigma
        y.sigma <- y.sigma - learning_rate*(grad.sigma.m) - rnorm(1,0,gaus.sd)
        y.sigma.vec <- c(y.sigma.vec,y.sigma)
        
        
        #Update Cv
        for(i in 1:n.mask){
          alpha.shape <- alpha.init[i] + length(theta.matrix[i,])/2
          # alpha.shape <- alpha.init[i] # Keep alpha the same
          beta.scale <- beta.init[i] + sum(theta.matrix[i,]^2)/(2*y.sigma)
          prior.var[i] <- rinvgamma(n = 1, alpha.shape, beta.scale)
          
          conj.alpha[i,it.num] <- alpha.shape
          conj.beta[i,it.num] <- beta.scale
          conj.invgamma[i,it.num] <- prior.var[i]
        }
      }
      
      it.num <- it.num +1
      
      # learning_rate <- start.a*(start.b+it.num)^(-start.gamma)
      learning_rate <- learning_rate/it.num
      gaus.sd <- sqrt(2*learning_rate)
      
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
    
  }
  
  pre.lr.vec <- c(lr.init,lr.init,pre.lr.vec[-length(pre.lr.vec)]) #Add first two learning rates
  lr.vec <- c(lr.init,lr.init,lr.vec[-length(lr.vec)]) #Add first two learning rates
  
  time.taken <- Sys.time() - time.train
  cat("Training complete in: ", time.taken)
  
  
  
  write.csv(rbind(loss.train,loss.val),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_loss_","_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(rbind(rsq.train,rsq.val),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_rsq_","_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(rbind(loss.train.male,loss.val.male),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_lossM_","_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(rbind(rsq.train.male,rsq.val.male),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_rsqM_","_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(rbind(loss.train.fmale,loss.val.fmale),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_lossF_","_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(rbind(rsq.train.fmale,rsq.val.fmale),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_rsqF_","_jobid_",JobId,".csv"), row.names = FALSE)
  
  
  write.csv(map.train,paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_map_","_jobid_",JobId,".csv"), row.names = FALSE)
  write_feather(as.data.frame(theta.matrix),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_theta_',"_jobid_",JobId,'.feather'))
  write.csv(bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_bias_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(y.sigma.vec,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_sigma_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(l.bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_lbias_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(lr.vec,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_lr_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(pre.lr.vec,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_prelr_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write_feather(as.data.frame(c(beta_fit$HS )),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_lweights_',"_jobid_",JobId,'.feather'))
  write.csv(co.weights,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_coweights_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(co.bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_cobias_',"_jobid_",JobId,".csv"), row.names = FALSE)
  
  
  
  temp.frame <- as.data.frame(rbind(pred.train.ind,pred.train.val,class.train.val))
  colnames(temp.frame) <- NULL
  colnames(temp.frame) <- 1:ncol(temp.frame)
  # temp.frame<- t(tail(t(temp.frame),length(train.test.ind$train)*5))
  write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_inpred_',"_jobid_",JobId,'.feather'))
  temp.frame <- as.data.frame(rbind(pred.test.ind,pred.test.val,class.test.val))
  colnames(temp.frame) <- NULL
  colnames(temp.frame) <- 1:ncol(temp.frame)
  # temp.frame<- t(tail(t(temp.frame),length(train.test.ind$test)*5))
  write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_outpred_',"_jobid_",JobId,'.feather'))
  
  
  #inv gamme param
  write.csv(conj.alpha,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_alpha_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(conj.beta,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_beta_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(conj.invgamma,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_invgam_',"_jobid_",JobId,".csv"), row.names = FALSE)
}