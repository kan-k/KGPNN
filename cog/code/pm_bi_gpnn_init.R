# R script


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
print("Starting")
print("Load Rcpp")
start.time <- Sys.time()
source("/well/nichols/users/qcv214/KGPNN/code/rcpp_funcs.R")
time.taken <- Sys.time() - start.time
print(paste0("load rcpp file completed in: ", time.taken))


filename <- "july26_pm_bi_gpnn_init"
# prior.var <- 0.05 #was 0.05
learning_rate <- 0.99 #for slow decay starting less than 1
epoch <- 250 #was 500
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
#Define scaled and shifted sigmoid
sigmoid <- function(x) sapply(x, function(z) 2/(1+exp(-10*z))-1)
sigmoid.prime <- function(x) sapply(x, function(z) 20*exp(-10*z)/(exp(-10*z)+1)^2) #this makes sense
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

#Define sigmoid

#Define Mean Squared Error
mse <- function(pred, true){mean((pred-true)^2)}

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
print("Loading data")


#Age
age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/cog/age_dmn_sex_strat.feather'))
age <- age_tab$pm_tf


#mask
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
res3.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz'))

list_of_all_images.dmn<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/fMRI/rfMRI_25.dr/dr_stage2.nii.gz')

res3.dat.dmn <-  as.matrix(fast_read_imgs_mask(list_of_all_images.dmn,'/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz'))

n.mask <- length(res3.mask.reg)
# n.expan <- choose(6+3,3)
p.dat <- ncol(res3.dat)
n.dat <- nrow(res3.dat)

train.test.ind <- list()
train.test.ind$test <- read.csv('/well/nichols/users/qcv214/KGPNN/cog/cog_dmn_test_index.csv')$x
train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/KGPNN/cog/cog_dmn_train_index.csv')$x
n.train <- length(train.test.ind$train)


# source("/well/nichols/users/qcv214/bnn2/res3/first_layer_gp4.R")
# partial.gp.centroid<-t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/KGPNN/partial_gp_centroids_fixed_100.540.feather")))) #12 x 4
partial.gp.centroid<-t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/KGPNN/partial_gp_centroids_fixed_300.540.feather"))))

l.expan <- ncol(partial.gp.centroid) #4

#Length


time.taken <- Sys.time() - start.time
cat("Loading data complete in: ", time.taken)


print("Getting mini batch")
#Get minibatch index 
batch_size <- 500 #Correspondds to 4 batches 500,500,500, 222


#NN parameters
it.num <- 1

prior.var.bias <- 1
prior.var.bias.dmn <- 1

############################################## prior var for structural
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

############################################## prior var for DMN
alpha.init.dmn <- rep(11,n.mask) #shape
beta.init.dmn <- rep(0.5,n.mask) #scale

conj.alpha.dmn <- matrix(, nrow=n.mask,ncol=epoch*4)
conj.beta.dmn <- matrix(, nrow=n.mask,ncol=epoch*4)
conj.invgamma.dmn <-matrix(, nrow=n.mask,ncol=epoch*4)

prior.var.dmn <- beta.init.dmn/(alpha.init.dmn-1)

##############################################Global param sigma^2
y.sigma <- var(age[train.test.ind$train])
y.sigma.vec <- y.sigma
##############################################
print("Initialisation")
#1 Initialisation
#1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
# weights <- matrix(,nrow=n.mask, ncol= n.expan)
weights <- matrix(, ncol = p.dat, nrow = n.mask)
for(i in 1:n.mask){
  weights[i,] <- rnorm(p.dat,0,sqrt(prior.var*y.sigma))
}
bias <- rnorm(n.mask)


#Weight for DMN
weights.dmn <- matrix(, ncol = p.dat, nrow = n.mask)
for(i in 1:n.mask){
  weights.dmn[i,] <- rnorm(p.dat,0,sqrt(prior.var.dmn*y.sigma))
}
bias.dmn <- rnorm(n.mask)

#Minimum values
min.mse <- 1e+8


#Initialising bias (to 0)


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
    
    # hidden.layer <- apply(t(t(res3.dat[mini.batch$train[[b]], ]  %*% t(weights)) + bias), 2, FUN = relu) #n x n.mask
    hidden.layer <-computeHiddenLayer(res3.dat[mini.batch$train[[b]], ], t(weights), bias)
    
    
    # hidden.layer.dmn <- apply(t(t(res3.dat.dmn[mini.batch$train[[b]], ]  %*% t(weights.dmn)) + bias.dmn), 2, FUN = relu)
    hidden.layer.dmn <-computeHiddenLayer(res3.dat.dmn[mini.batch$train[[b]], ], t(weights.dmn), bias.dmn)
    
    
    # Generate polynomial features (linear terms)
    # hidden.features <- as.matrix(hidden.layer %*% partial.gp.centroid) # (n x n.mask) x (n.mask x l.expan) = (n x l.expan)
    hidden.features <- matrixMultiply(hidden.layer, partial.gp.centroid)
    
    # hidden.dmn.features <- as.matrix(hidden.layer.dmn %*% partial.gp.centroid)
    hidden.dmn.features <- matrixMultiply(hidden.layer.dmn, partial.gp.centroid)
    z.nb <- cbind(1,hidden.features, hidden.dmn.features, hidden.dmn.features*hidden.features) #This is different from LASIR in the sense that the subgroup latent directly affect the output, whereas the group themselves dont. But then that can be modified easily.
    
    hs_fit_SOI <- fast_normal_lm(age[mini.batch$train[[b]]],z.nb) #This also gives the bias term
    
    ############################################################################################ 
    
    beta_fit <- data.frame(HS = c(partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[(1+1):(l.expan+1)],
                                  partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[(1+l.expan+1):(2*l.expan+1)],
                                  (partial.gp.centroid^2)%*%hs_fit_SOI$post_mean$betacoef[-(1:(2*l.expan+1))]
    ))
    
    l.bias <- hs_fit_SOI$post_mean$betacoef[1]
    
    hs_in.pred_SOI <- predict_fast_lm(hs_fit_SOI, z.nb, alpha = 0.95)$mean
    
    
    loss.train <- c(loss.train, mseCpp(hs_in.pred_SOI,age[mini.batch$train[[b]]]))
    rsq.train <- c(rsq.train, rsqCpp(age[mini.batch$train[[b]]],hs_in.pred_SOI))
    
    # loss.train.male <- c(loss.train.male, mseCpp(hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == 1)],age[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == 1)]))
    # rsq.train.male <- c(rsq.train.male, rsqCpp(age[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == 1)],hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == 1)]))
    
    # loss.train.fmale <- c(loss.train.fmale, mseCpp(hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == -1)],age[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == -1)]))
    # rsq.train.f <- c(rsq.train.fmale, rsqCpp(age[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == -1)],hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == -1)]))
    
    temp.sum.sum.sq <- apply(weights, 1, FUN = function(x) sum(x^2))
    
    #Note wrong MAP here. I have NOT incorporated intercept
    map.train <- c(map.train,n.train/2*log(y.sigma) +1/(2*y.sigma)*n.train*mse(hs_in.pred_SOI,age[mini.batch$train[[b]]]) +n.mask/2*log(y.sigma) +n.mask*p.dat/2*log(y.sigma) + 1/(2*y.sigma)*sum(1/prior.var*(temp.sum.sum.sq))  +1/2*sum(c(bias)^2) )
    
    ################################################################# TEST #################################################################
    
    # hidden.layer.test <- apply(t(t(res3.dat[train.test.ind$test, ] %*% t(weights)) + bias), 2, FUN = relu)
    hidden.layer.test <-computeHiddenLayer(res3.dat[train.test.ind$test, ], t(weights), bias)
    
    # hidden.layer.dmn.test <- apply(t(t(res3.dat.dmn[train.test.ind$test, ] %*% t(weights.dmn)) + bias.dmn), 2, FUN = relu)
    hidden.layer.dmn.test <-computeHiddenLayer(res3.dat.dmn[train.test.ind$test, ], t(weights.dmn), bias.dmn)
    
    
    # Generate polynomial features (linear terms)
    # hidden.features.test <- as.matrix(hidden.layer.test %*% partial.gp.centroid) # (n x n.mask) x (n.mask x l.expan) = (n x l.expan)
    hidden.features.test <- matrixMultiply(hidden.layer.test, partial.gp.centroid)
    
    # hidden.dmn.features.test <- as.matrix(hidden.layer.dmn.test %*% partial.gp.centroid)
    hidden.dmn.features.test <- matrixMultiply(hidden.layer.dmn.test, partial.gp.centroid)
    z.nb.test <- cbind(1,hidden.features.test, hidden.dmn.features.test, hidden.dmn.features.test*hidden.features.test) #This is different from LASIR in the sense that the subgroup latent directly affect the output, whereas the group themselves dont. But then that can be modified easily.
    
    hs_pred_SOI <- predict_fast_lm(hs_fit_SOI, z.nb.test, alpha = 0.95)$mean
    
    loss.val <- c(loss.val, mseCpp(hs_pred_SOI,age[train.test.ind$test]))
    rsq.val <- c(rsq.val, rsqCpp(age[train.test.ind$test],hs_pred_SOI))
    
    # loss.val.male <- c(loss.val.male, mseCpp(hs_pred_SOI[which(sex[train.test.ind$test] == 1)],age[train.test.ind$test][which(sex[train.test.ind$test] == 1)]))
    # rsq.val.male <- c(rsq.val.male, rsqCpp(age[train.test.ind$test][which(sex[train.test.ind$test] == 1)],hs_pred_SOI[which(sex[train.test.ind$test] == 1)]))
    # 
    # loss.val.fmale <- c(loss.val.fmale, mseCpp(hs_pred_SOI[which(sex[train.test.ind$test] == -1)],age[train.test.ind$test][which(sex[train.test.ind$test] == -1)]))
    # rsq.val.fmale <- c(rsq.val.fmale, rsqCpp(age[train.test.ind$test][which(sex[train.test.ind$test] == -1)],hs_pred_SOI[which(sex[train.test.ind$test] == -1)]))
    # 
    #For keeping the minimum
    if((tail(loss.val,1) < min.mse) & (e >2 )){
      min.weights <- weights
      min.bias <- bias
      min.y.sigma <- y.sigma
      min.lr <- learning_rate
      min.alpha <- conj.alpha[,(it.num-1)]
      min.beta <- conj.beta[,(it.num-1)]
      min.prior.var <- conj.invgamma[,(it.num-1)]
      min.mse <- tail(loss.val,1)
      min.weights.dmn <- weights.dmn
      min.bias.dmn <- bias.dmn
      min.alpha.dmn <- conj.alpha.dmn[,(it.num-1)]
      min.beta.dmn <- conj.beta.dmn[,(it.num-1)]
      min.prior.var.dmn <- conj.invgamma.dmn[,(it.num-1)]
      
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
      
      #4Update the full weights, fit GP against the full weights using HS-prior model to get normally dist thetas
      grad.loss <- age[mini.batch$train[[b]]] - hs_in.pred_SOI
      
      #Update weight and weight.dmn
      grad <- updateDoubleWeightsFirst(minibatch.size, n.mask,y.sigma, grad.loss, beta_fit$HS, hidden.layer,res3.dat[mini.batch$train[[b]], ], hidden.layer.dmn,res3.dat.dmn[mini.batch$train[[b]], ])
      grad.dmn <- updateDoubleWeightsSecond(minibatch.size, n.mask,y.sigma, grad.loss, beta_fit$HS, hidden.layer,res3.dat[mini.batch$train[[b]], ], hidden.layer.dmn,res3.dat.dmn[mini.batch$train[[b]], ])
      
      #Sex shouldnt be in the above, it should be sigmoid. That means all of my sigmoid are wrong.
      #Take batch average
      grad.m <- computeMean(grad)
      grad.dmn.m <- computeMean(grad.dmn)
      
      
      #####
      #########Here is inefficiency
      grad.b <- updateGradBFirst(minibatch.size, n.mask,y.sigma, grad.loss, beta_fit$HS, hidden.layer,hidden.layer.dmn)
      grad.b.dmn <- updateGradBSecond(minibatch.size, n.mask,y.sigma, grad.loss, beta_fit$HS, hidden.layer,hidden.layer.dmn)
      
      #Take batch average
      grad.b.m <- c(apply(grad.b, c(2), mean)) #I am applying -grad.b here. Is it right!?!?! 2 nov
      grad.b.dmn.m <- c(apply(grad.b.dmn, c(2), mean)) #I am applying -grad.b here. Is it right!?!?! 2 nov
      
      
      # Update sigma
      
      ####This has to be changed
      
      grad.sigma.m <- mean(length(train.test.ind$train)/(2*y.sigma) - length(train.test.ind$train)/(2*y.sigma^2)*c(grad.loss)^2 -1/(2*y.sigma^2)*sum(c(weights/prior.var)^2)+1/(2*y.sigma)*p.dat*n.mask
                           -1/(2*y.sigma^2)*sum(c(weights.dmn/prior.var.dmn)^2)+1/(2*y.sigma)*p.dat*n.mask)
      ####Note here of the static equal prior.var
      #Update theta matrix
      weights <- weights*(1-learning_rate*1/(prior.var*y.sigma)) - learning_rate*grad.m * length(train.test.ind$train)
      weights.dmn <- weights.dmn*(1-learning_rate*1/(prior.var.dmn*y.sigma)) - learning_rate*grad.dmn.m * length(train.test.ind$train)
      
      #Note that updating weights at the end will be missing the last batch of last epoch
      
      #Update bias
      bias <- bias*(1-learning_rate*1/(prior.var.bias)) - learning_rate*c(grad.b.m) * length(train.test.ind$train)
      bias.dmn <- bias.dmn*(1-learning_rate*1/(prior.var.bias.dmn)) - learning_rate*c(grad.b.dmn.m) * length(train.test.ind$train)
      
      # Update sigma
      y.sigma <- y.sigma - learning_rate*(grad.sigma.m)
      y.sigma.vec <- c(y.sigma.vec,y.sigma)
      
      delta_f <- c(c(weights/(prior.var*y.sigma) + grad.m*n.train),c(bias/prior.var.bias + grad.b.m*(n.train)),c(weights.dmn/(prior.var.dmn*y.sigma) + grad.dmn.m*n.train),c(bias.dmn/prior.var.bias.dmn + grad.b.dmn.m*(n.train)))
      
      grad_x <- beta.bb*delta_f + (1-beta.bb)*grad_x
      # x.param <- c(c(weights),c(bias))
      x.param <- c(c(weights),c(bias),c(weights.dmn),c(bias.dmn))
      
      
      #Update Cv
      for(i in 1:n.mask){
        alpha.shape <- alpha.init[i] + length(weights[i,])/2
        beta.scale <- beta.init[i] + sum(weights[i,]^2)/(2*y.sigma)
        prior.var[i] <- rinvgamma(n = 1, alpha.shape, beta.scale)
        
        conj.alpha[i,it.num] <- alpha.shape
        conj.beta[i,it.num] <- beta.scale
        conj.invgamma[i,it.num] <- prior.var[i]
        ###########################
        alpha.shape.dmn <- alpha.init.dmn[i] + length(weights.dmn[i,])/2
        beta.scale.dmn <- beta.init.dmn[i] + sum(weights.dmn[i,]^2)/(2*y.sigma)
        prior.var.dmn[i] <- rinvgamma(n = 1, alpha.shape.dmn, beta.scale.dmn)
        
        conj.alpha.dmn[i,it.num] <- alpha.shape.dmn
        conj.beta.dmn[i,it.num] <- beta.scale.dmn
        conj.invgamma.dmn[i,it.num] <- prior.var.dmn[i]
      }
      
      
      
    }
    
    it.num <- it.num +1
    
    # invisible(capture.output(ifelse(it.num >=2000, learning_rate <- lr.init*0.001,ifelse(it.num >=1000, learning_rate <- lr.init*0.01, learning_rate <- lr.init) )))
    # if((it.num %% 200) ==0){
    #   learning_rate <- learning_rate*0.1
    # }
    # learning_rate <- learning_rate
    print(paste0("training loss: ",mseCpp(hs_in.pred_SOI,age[mini.batch$train[[b]]])))
    print(paste0("validation loss: ",mseCpp(hs_pred_SOI,age[train.test.ind$test])))
  }
  
  print(paste0("epoch: ",e," out of ",epoch, ", time taken for this epoch: ",Sys.time() -time.epoch))
  print(paste0("sigma^2: ",y.sigma))
  
  
  #BB
  
  if(e >=2){
    diff_x = x.param - prev_x
    diff_grad_x = grad_x - prev_grad_x
    learning_rate <- 1/num.batch*sum(diff_x*diff_x)/abs(sum(diff_x*diff_grad_x))
  }
  prev_x <- x.param
  prev_grad_x <- grad_x
  lr.vec <- c(lr.vec, learning_rate)
}

time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

lr.vec <- c(lr.init,lr.vec[-length(lr.vec)]) #Add first learning rate


write.csv(rbind(loss.train,loss.val),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_loss_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(rsq.train,rsq.val),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_rsq_","_jobid_",JobId,".csv"), row.names = FALSE)
# write.csv(rbind(loss.train.male,loss.val.male),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_lossM_","_jobid_",JobId,".csv"), row.names = FALSE)
# write.csv(rbind(rsq.train.male,rsq.val.male),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_rsqM_","_jobid_",JobId,".csv"), row.names = FALSE)
# write.csv(rbind(loss.train.fmale,loss.val.fmale),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_lossF_","_jobid_",JobId,".csv"), row.names = FALSE)
# write.csv(rbind(rsq.train.fmale,rsq.val.fmale),paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_rsqF_","_jobid_",JobId,".csv"), row.names = FALSE)

write.csv(map.train,paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_map_","_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(weights),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_weights_',"_jobid_",JobId,'.feather'))
write.csv(bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_bias_',"_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(weights.dmn),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_weightsdmn_',"_jobid_",JobId,'.feather'))
write.csv(bias.dmn,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_biasdmn_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(y.sigma.vec,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_sigma_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(l.bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_lbias_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(lr.vec,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_lr_',"_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(c(beta_fit$HS )),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_lweights_',"_jobid_",JobId,'.feather'))


temp.frame <- as.data.frame(rbind(pred.train.ind,pred.train.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
# temp.frame<- t(tail(t(temp.frame),length(train.test.ind$train)*5))
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_inpred_',"_jobid_",JobId,'.feather'))
temp.frame <- as.data.frame(rbind(pred.test.ind,pred.test.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
# temp.frame<- t(tail(t(temp.frame),length(train.test.ind$test)*5))


#Write Minimum 
write.csv(min.mse,paste0("/well/nichols/users/qcv214/KGPNN/cog/pile/re_",filename,"_minloss_","_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(min.weights),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_minweights_',"_jobid_",JobId,'.feather'))
write.csv(min.bias,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_minbias_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(min.y.sigma,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_minsigma_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(min.lr,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_minlr_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(min.alpha,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_minalpha_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(min.beta,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_minbeta_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(min.prior.var,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_minpriorvar_',"_jobid_",JobId,".csv"), row.names = FALSE)

write_feather(as.data.frame(min.weights.dmn),paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_minweightsdmn_',"_jobid_",JobId,'.feather'))
write.csv(min.bias.dmn,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_minbiasdmn_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(min.alpha.dmn,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_minalphadmn_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(min.beta.dmn,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_minbetadmn_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(min.prior.var.dmn,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_minpriorvardmn_',"_jobid_",JobId,".csv"), row.names = FALSE)

write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/cog/pile/re_',filename,'_outpred_',"_jobid_",JobId,'.feather'))