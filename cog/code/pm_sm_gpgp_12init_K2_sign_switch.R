#10 july
##Try to model log num ~ brain, 2x gender, 3x deprivation, 3x age? all quantile-based
#14 july, modellling raw num instead of log num
#12 oct, also doing likelihood and prior

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

filename <- "oct12_pm_sm_gpgp_12init_sign_switch"
# prior.var <- 0.05 #was 0.05
learning_rate <- 0.99 #for slow decay starting less than 1
prior.var.bias <- 1
epoch <- 50 #was 400
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
#Define scaled and shifted sigmoid
sigmoid <- function(x) sapply(x, function(z) 2/(1+exp(-10*z))-1)
sigmoid.prime <- function(x) sapply(x, function(z) 20*exp(-10*z)/(exp(-10*z)+1)^2) #this makes sense

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

#Define sigmoid

#Define Mean Squared Error
mse <- function(pred, true){mean((pred-true)^2)}


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
# res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz')
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
# n.expan.degree <- 30
n.expan <- choose(20+3,3)

#Pre-transform
print(paste0("load up res3 gp"))
# source("/well/nichols/users/qcv214/bnn2/res3/res4_first_layer_gp.R")
# source("/well/nichols/users/qcv214/KGPNN/res4_first_layer_gp_tf.R") #Added 30 apr
# source("/well/nichols/users/qcv214/KGPNN/res4_first_layer_gpOnce_tf.R") #Use R GPs rather than R^2
source("/well/nichols/users/qcv214/bnn2/res3/first_layer_gp5_50.R")

partial.gp.centroid.img<-t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/KGPNN/partial_gp_centroids_fixed_300.540.feather"))))
l.expan <- ncol(partial.gp.centroid.img)

print(paste0("load up res3 gp [DONE]"))
print(paste0("dim res3 gp"))
print(dim(res3.dat))
mult.dat <- function(X) as.matrix(res3.dat %*% X)
mult.thea <- function(X,theta) rowSums(X*theta)
res3.dat <- array(t(apply(partial.gp,MARGIN = c(1),mult.dat)), dim =c(n.mask,n.dat,n.expan)) #rename res3.dat as the product of data and expansion

####################################################################################################################################

for(num.lat.class.select in c(2)){
  
  learning_rate <- 0.99
  
  #Losses
  loss.train <- vector(mode = "numeric")
  loss.val <- vector(mode = "numeric")
  
  loss.train.male <- vector(mode = "numeric")
  loss.val.male <- vector(mode = "numeric")
  
  loss.train.fmale <- vector(mode = "numeric")
  loss.val.fmale <- vector(mode = "numeric")
  
  map.train <- vector(mode = "numeric")
  lik.train <- vector(mode = "numeric")
  prior.train <- vector(mode = "numeric")
  
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
  
  
  
  time.taken <- Sys.time() - start.time
  cat("Loading data complete in: ", time.taken)
  
  
  print("Getting mini batch")
  #Get minibatch index 
  batch_size <- 500 #Correspondds to 4 batches 500,500,500,471
  
  #NN parameters
  it.num <- 1
  
  
  #Initial parameters for inverse gamma
  alpha.init <- rep(11,n.mask) #shape
  beta.init <- rep(0.5,n.mask) #scale
  
  
  #Storing inv gamma
  conj.alpha <- matrix(, nrow=n.mask,ncol=epoch*4)
  conj.beta <- matrix(, nrow=n.mask,ncol=epoch*4)
  conj.invgamma <-matrix(, nrow=n.mask,ncol=epoch*4)
  # conj.cv <- matrix(, nrow=n.mask,ncol=epoch*4)
  
  #Define init var
  prior.var <- beta.init/(alpha.init-1) #Mean of IG
  
  #Fix prior var to be 0.1
  # prior.var <- 1.5
  y.sigma <- var(lognum[train.test.ind$train])
  y.sigma.vec <- y.sigma
  
  print("Initialisation")
  #1 Initialisation
  #1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
  
  theta.matrix <- matrix(,nrow=n.mask, ncol= n.expan)
  for(i in 1:n.mask){
    theta.matrix[i,] <- rnorm(n.expan,0,sqrt(prior.var*y.sigma)) #was 500
  }
  
  #Weight for non-imaging covariates
  
  num.lat.class<- num.lat.class.select
  co.weights <- matrix(rnorm(ncol(co.dat)*num.lat.class,0,0.01), ncol = ncol(co.dat), nrow = num.lat.class) #4 number of latent subgroup #Note that this is 
  co.bias <- rnorm(num.lat.class,0,0.1) #I think this one is still one
  #Minimum values
  min.mse <- 1e+8
  
  
  #Initialising bias (to 0)
  bias <- rnorm(n.mask)
  
  #non-imaging covar
  gp.coef <- matrix(rnorm(num.lat.class*l.expan),nrow = num.lat.class )
  partial.gp.centroid <- rbind(partial.gp.centroid.img,gp.coef)
  
  
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
      
      #################
      hidden.layer <- apply(t(apply(res3.dat[, mini.batch$train[[b]], ],MARGIN = 2,FUN = mult.thea,thet=theta.matrix) + bias), 2, FUN = relu) #not C++ optim, this collapes into two dimension
      
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
      
      #################
      
      loss.train <- c(loss.train, mseCpp(hs_in.pred_SOI,lognum[mini.batch$train[[b]]]))
      rsq.train <- c(rsq.train, rsqCpp(lognum[mini.batch$train[[b]]],hs_in.pred_SOI))
      
      loss.train.male <- c(loss.train.male, mseCpp(hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == 1)],lognum[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == 1)]))
      rsq.train.male <- c(rsq.train.male, rsqCpp(lognum[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == 1)],hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == 1)]))
      
      loss.train.fmale <- c(loss.train.fmale, mseCpp(hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == -1)],lognum[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == -1)]))
      rsq.train.f <- c(rsq.train.fmale, rsqCpp(lognum[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == -1)],hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == -1)]))
      
      temp.sum.sum.sq <- apply(theta.matrix, 1, FUN = function(x) sum(x^2))
      
      map.train <- c(map.train,n.train/2*log(y.sigma) +1/(2*y.sigma)*n.train*mseCpp(hs_in.pred_SOI,lognum[mini.batch$train[[b]]]) +l.expan/2*log(y.sigma) + 1/(2*y.sigma)*(1/1)*sum((hs_fit_SOI$post_mean$betacoef[2:(l.expan+1)])^2) + 1/2*l.bias^2)
      lik.train <- c(lik.train,n.train/2*log(y.sigma) +1/(2*y.sigma)*n.train*mseCpp(hs_in.pred_SOI,lognum[mini.batch$train[[b]]]))
      prior.train <- c(prior.train,l.expan/2*log(y.sigma) + 1/(2*y.sigma)*(1/1)*sum((hs_fit_SOI$post_mean$betacoef[2:(l.expan+1)])^2) + 1/2*l.bias^2)
      
      #Validation
      #Layers
      hidden.layer.test <- apply(t(apply(res3.dat[, train.test.ind$test, ],MARGIN = 2,FUN = mult.thea,thet=theta.matrix) + bias), 2, FUN = relu)
      co.hidden.layer.test <- softmax(t(t(co.dat[train.test.ind$test, ] %*% t(co.weights)) + co.bias))
      
      hidden.layer.mixed.test <- cbind(hidden.layer.test, co.hidden.layer.test)
      
      poly_features.test <- as.matrix(hidden.layer.mixed.test %*% partial.gp.centroid)
      
      # Create the design matrix
      z.nb.test <- cbind(1,poly_features.test)
      
      hs_pred_SOI <- predict_fast_lm(hs_fit_SOI, z.nb.test, alpha = 0.95)$mean
      
      
      loss.val <- c(loss.val, mseCpp(hs_pred_SOI,lognum[train.test.ind$test]))
      rsq.val <- c(rsq.val, rsqCpp(lognum[train.test.ind$test],hs_pred_SOI))
      
      loss.val.male <- c(loss.val.male, mseCpp(hs_pred_SOI[which(sex[train.test.ind$test] == 1)],lognum[train.test.ind$test][which(sex[train.test.ind$test] == 1)]))
      rsq.val.male <- c(rsq.val.male, rsqCpp(lognum[train.test.ind$test][which(sex[train.test.ind$test] == 1)],hs_pred_SOI[which(sex[train.test.ind$test] == 1)]))
      
      loss.val.fmale <- c(loss.val.fmale, mseCpp(hs_pred_SOI[which(sex[train.test.ind$test] == -1)],lognum[train.test.ind$test][which(sex[train.test.ind$test] == -1)]))
      rsq.val.fmale <- c(rsq.val.fmale, rsqCpp(lognum[train.test.ind$test][which(sex[train.test.ind$test] == -1)],hs_pred_SOI[which(sex[train.test.ind$test] == -1)]))
      
      #For keeping the minimum
      if(isTRUE((tail(loss.val,1) < min.mse)) && (e >2 )){
        min.theta.matrix <- theta.matrix
        min.bias <- bias
        min.y.sigma <- y.sigma
        min.lr <- learning_rate
        min.alpha <- conj.alpha[,(it.num-1)]
        min.beta <- conj.beta[,(it.num-1)]
        min.prior.var <- conj.invgamma[,(it.num-1)]
        min.mse <- tail(loss.val,1)
        min.co.weights <- co.weights
        min.co.bias <- co.bias
        min.gp.coef <- gp.coef
      }
      
      ##Keeping the last 5 epochs predictions
      if(e >= (epoch-5)){
        pred.train.ind <- c(pred.train.ind,mini.batch$train[[b]]) 
        pred.train.val <- c(pred.train.val,hs_in.pred_SOI)
        pred.test.ind <- c(pred.test.ind,train.test.ind$test) 
        pred.test.val <- c(pred.test.val,hs_pred_SOI) 
      }
      
      if(it.num < epoch*num.batch){
        #Update weight
        
        beta.img <- beta_fit$HS[1:n.mask]
        
        #non-Imaging weights betas
        beta.nonimg <- beta_fit$HS[(n.mask+1):(n.mask+num.lat.class)]
        #Update weight
        
        #4Update the full weights, fit GP against the full weights using HS-prior model to get normally dist thetas
        grad.loss <- lognum[mini.batch$train[[b]]] - hs_in.pred_SOI
        
        grad <- updateWeightsGP(minibatch.size, n.mask,y.sigma, grad.loss, beta.img, hidden.layer,res3.dat[,mini.batch$train[[b]], ]) #This already output 3D
        grad.m <- computeMean(grad)
        
        grad.b <- updateGradB(minibatch.size, n.mask,y.sigma, grad.loss, beta.img, hidden.layer)
        grad.b.m <- calculateColumnMeans(grad.b)
        
        co.sm.grad <- apply(co.hidden.layer,1,softmax.prime,l.grad = beta.nonimg) #Note that softmax.prime take in softmax output rather than the pre-softmax input
        #Then I need to times co.sm.grad by l.grad. I think I need l.grad to be inside softmax.prime
        #Then for the resulting post-3D-transformation of co.sm.grad, I want to time each 1st dim by c(grad.loss). This is as simple as ...*c(grad.loss) [have verified]
        co.sm.grad <- array(t(co.sm.grad),dim = c(ncol(co.sm.grad),sqrt(nrow(co.sm.grad)),sqrt(nrow(co.sm.grad))))
        grad.sum <- apply(co.sm.grad*(-1/y.sigma*c(grad.loss)), c(1,3), sum)
        
        co.grad.m<- t(grad.sum)%*%co.dat[mini.batch$train[[b]], ]/nrow(co.dat[mini.batch$train[[b]], ]) #n.lat class * num attr
        
        co.grad.b.m <- c(colMeans(grad.sum))
        
        co.weights <- co.weights*(1+learning_rate) + learning_rate*co.grad.m 
        
        co.bias <- co.bias*(1+learning_rate) + learning_rate*co.grad.b.m
        
        # Update sigma
        grad.sigma.m <- mean(length(train.test.ind$train)/(2*y.sigma) - length(train.test.ind$train)/(2*y.sigma^2)*c(grad.loss)^2-1/(2*y.sigma^2)*sum(c(theta.matrix/prior.var)^2)+1/(2*y.sigma)*n.expan*n.mask)
        #update theta
        theta.matrix <- theta.matrix*(1+learning_rate*1/(prior.var*y.sigma)) + learning_rate*grad.m * length(train.test.ind$train)
        #Update bias
        bias <- bias*(1+learning_rate*1/(prior.var.bias)) + learning_rate*c(grad.b.m) * length(train.test.ind$train)
        # Update sigma
        y.sigma <- y.sigma + learning_rate*(grad.sigma.m)
        y.sigma.vec <- c(y.sigma.vec,y.sigma)
        
        
        #2nd hidden non-imaging layer
        gp.coef.grad <- (c(grad.loss)*co.hidden.layer)
        gp.coef.grad <- array(apply(gp.coef.grad,1,function(y) y %o% hs_fit_SOI$post_mean$betacoef[2:(l.expan+1)]),dim = c(num.lat.class,l.expan,minibatch.size)) #Now is num class x K x n
        gp.coef.grad.m <- apply(gp.coef.grad,c(1,2),mean)*(-1/y.sigma) #compute mean over minibatch (3rd dim), then factor by y.sigma
        gp.coef <- gp.coef*(1-learning_rate) - learning_rate*gp.coef.grad.m* n.train
        partial.gp.centroid <- rbind(partial.gp.centroid.img,gp.coef)
        
        
        
        # delta_f <- c(c(theta.matrix/(prior.var*y.sigma) + grad.m*n.train),c(bias/prior.var.bias + grad.b.m*(n.train)),c(co.weights+co.grad.m*n.train),c(co.bias + co.grad.b.m*n.train), c(gp.coef + gp.coef.grad.m* n.train))
        
        # grad_x <- beta.bb*delta_f + (1-beta.bb)*grad_x
        # x.param <- c(c(weights),c(bias))
        # x.param <- c(c(theta.matrix),c(bias),c(co.weights),c(co.bias), c(gp.coef))
        
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
      
      # invisible(capture.output(ifelse(it.num >=2000, learning_rate <- lr.init*0.001,ifelse(it.num >=1000, learning_rate <- lr.init*0.01, learning_rate <- lr.init) )))
      # if((it.num %% 200) ==0){
      #   learning_rate <- learning_rate*0.1
      # }
      # learning_rate <- learning_rate
      print(paste0("training loss: ",mse(hs_in.pred_SOI,lognum[mini.batch$train[[b]]])))
      print(paste0("validation loss: ",mse(hs_pred_SOI,lognum[train.test.ind$test])))
    }
    
    print(paste0("epoch: ",e," out of ",epoch, ", time taken for this epoch: ",Sys.time() -time.epoch))
    print(paste0("sigma^2: ",y.sigma))
    
    # if(e %in% c(epoch/2,epoch)){
    #   salient.mat <- matrix(,nrow=length(train.test.ind$train), ncol = p.dat)
    #   for(o in train.test.ind$train){
    #     salient.mat[which(o==train.test.ind$train),]<- c(beta_fit$HS %*% (relu.prime(bias+weights%*%res3.dat[o,])*weights))
    #   }
    #   gp.mask.hs <- res3.mask
    #   gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
    #   gp.mask.hs@datatype = 16
    #   gp.mask.hs@bitpix = 32
    #   writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/',filename,'_sal_',JobId,"_epoch_",e))
    # }
    
    #BB
    
    # if(e >=2){
    #   
    #   diff_x = x.param - prev_x
    #   diff_grad_x = grad_x - prev_grad_x
    #   
    #   if (abs(sum(diff_x*diff_grad_x)) == 0){
    #     learning_rate <- 0.025 #0.25
    #   } else { 
    #     learning_rate <- 1/num.batch*sum(diff_x*diff_x)/abs(sum(diff_x*diff_grad_x))
    #   }
    #   
    #   
    #   ####Cap learning rate 
    #   learning_rate <- sign(learning_rate)*min(abs(learning_rate),0.1) #was 0.8
    #   print(paste0("CAPPED learning rate of above epoch: ", learning_rate))
    # }
    # prev_x <- x.param
    # prev_grad_x <- grad_x
    learning_rate <- learning_rate/it.num
    print(paste0("learning rate of above epoch: ", learning_rate))
    lr.vec <- c(lr.vec, learning_rate)
  }
  
  time.taken <- Sys.time() - time.train
  cat("Training complete in: ", time.taken)
  print(Sys.time())
  
  lr.vec <- c(lr.init,lr.vec[-length(lr.vec)]) #Add first learning rate
  
  
  write.csv(rbind(loss.train,loss.val),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_loss_","_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(rbind(rsq.train,rsq.val),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_rsq_","_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(rbind(loss.train.male,loss.val.male),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_lossM_","_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(rbind(rsq.train.male,rsq.val.male),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_rsqM_","_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(rbind(loss.train.fmale,loss.val.fmale),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_lossF_","_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(rbind(rsq.train.fmale,rsq.val.fmale),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_rsqF_","_jobid_",JobId,".csv"), row.names = FALSE)
  
  write.csv(map.train,paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_map_","_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(lik.train,paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_likelihood_","_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(prior.train,paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_prior_","_jobid_",JobId,".csv"), row.names = FALSE)
  
  write_feather(as.data.frame(theta.matrix),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_theta_',"_jobid_",JobId,'.feather'))
  write.csv(bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_bias_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(y.sigma.vec,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_sigma_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(l.bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_lbias_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(lr.vec,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_lr_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write_feather(as.data.frame(c(beta_fit$HS )),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_concatlweights_',"_jobid_",JobId,'.feather'))
  write.csv(co.weights,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_coweights_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(co.bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_cobias_',"_jobid_",JobId,".csv"), row.names = FALSE)
  
  
  
  temp.frame <- as.data.frame(rbind(pred.train.ind,pred.train.val))
  colnames(temp.frame) <- NULL
  colnames(temp.frame) <- 1:ncol(temp.frame)
  # temp.frame<- t(tail(t(temp.frame),length(train.test.ind$train)*5))
  write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_inpred_',"_jobid_",JobId,'.feather'))
  temp.frame <- as.data.frame(rbind(pred.test.ind,pred.test.val))
  colnames(temp.frame) <- NULL
  colnames(temp.frame) <- 1:ncol(temp.frame)
  # temp.frame<- t(tail(t(temp.frame),length(train.test.ind$test)*5))
  
  
  #Write Minimum 
  write.csv(min.mse,paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,'_K',num.lat.class.select,"_minloss_","_jobid_",JobId,".csv"), row.names = FALSE)
  write_feather(as.data.frame(min.theta.matrix),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_mintheta_',"_jobid_",JobId,'.feather'))
  write.csv(min.bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_minbias_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(min.y.sigma,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_minsigma_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(min.lr,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_minlr_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(min.alpha,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_minalpha_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(min.beta,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_minbeta_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(min.prior.var,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_minpriorvar_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(min.co.weights,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_mincoweights_',"_jobid_",JobId,".csv"), row.names = FALSE)
  write.csv(min.co.bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_mincobias_',"_jobid_",JobId,".csv"), row.names = FALSE)
  
  
  write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_K',num.lat.class.select,'_outpred_',"_jobid_",JobId,'.feather'))
  
}