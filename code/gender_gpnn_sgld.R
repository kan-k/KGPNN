# R script

# 11 Nov, after 1000 steps, change learning rate to 5e-11 from 5e-10, and after 2000, to 5e-12
#differ from gpnn_gp from the init variance of theta
# Adding gradient noise
#from BBS
#30 Mar add recording of out of sample predictions to 1000


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


filename <- "oct21_gender_gpnn_sgld_a3_b0_near" 
init.num <- JobId #WAS 2 [12/4/23]
prior.var <- 0.05 #was 0.05
start.b <- 1 #Originally 1e90
start.a <- 1e-3
start.gamma <- 1
learning_rate <- start.a*(start.b+1)^(-start.gamma) #for slow decay starting less than 1 #
prior.var.bias <- 1
epoch <- 300 
# burnin.epoch <- 100 #Epochs until SGLD kicks in
record.epoch <- 300 #Record last 100 epochs
# beta.bb<- 0.5
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
#Recaord noise of gradients
#####Need to change in other cases
grad.noise <- matrix(nrow = 12*2,ncol=epoch)

print("Loading data")

#Load data and mask and GP 
#Age
age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/age_sex_strat.feather'))
#age_tab <- age_tab[order(age_tab$id),].     #DOES THIS MESS UP ORDER
age <- age_tab$age
sex <-  as.numeric(age_tab$sex)
sex <- sapply(sex, function(x) replace(x, x==0,-1)) #Change female to -1, male to 1

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

train.test.ind <- list()
train.test.ind$test <- read.csv('/well/nichols/users/qcv214/KGPNN/sex_test_index.csv')$x
train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/KGPNN/sex_train_index.csv')$x
n.train <- length(train.test.ind$train)


# source("/well/nichols/users/qcv214/bnn2/res3/first_layer_gp4.R")
partial.gp.centroid<-t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/KGPNN/partial_gp_centroids_fixed_300.540.feather"))))
l.expan <- ncol(partial.gp.centroid)

#Length


time.taken <- Sys.time() - start.time
cat("Loading data complete in: ", time.taken)


print("Getting mini batch")
#Get minibatch index 
batch_size <- 500


#NN parameters
it.num <- 1

#Initial parameters for inverse gamma
alpha.init <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_sep26_gender_gpnn_bbig_init_minalpha__jobid_",init.num,".csv"))$x #shape
beta.init <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_sep26_gender_gpnn_bbig_init_minbeta__jobid_",init.num,".csv"))$x #scale

#Storing inv gamma
conj.alpha <- matrix(, nrow=n.mask,ncol=epoch*4)
conj.beta <- matrix(, nrow=n.mask,ncol=epoch*4)
conj.invgamma <-matrix(, nrow=n.mask,ncol=epoch*4)
# conj.cv <- matrix(, nrow=n.mask,ncol=epoch*4)

#Define init var
prior.var <-  read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_sep26_gender_gpnn_bbig_init_minpriorvar__jobid_",init.num,".csv"))$x#Mean of IG

#Fix prior var to be 0.1
# prior.var <- 1.5
y.sigma <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_sep26_gender_gpnn_bbig_init_minsigma__jobid_",init.num,".csv"))$x
y.sigma.vec <- y.sigma
####

gaus.sd <- 0

print("Initialisation")
#1 Initialisation
#1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
# theta.matrix <- matrix(,nrow=n.mask, ncol= n.expan)
weights <- as.matrix(read_feather(paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_sep26_gender_gpnn_bbig_init_minweights__jobid_',init.num,'.feather')))

#Initialising bias (to 0)
bias <- read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_sep26_gender_gpnn_bbig_init_minbias__jobid_",init.num,".csv"))$x
time.train <-  Sys.time()

#Start epoch
for(e in 1:epoch){
  
  #randomise data
  mini.batch <- batch_split(left = train.test.ind$train, batch_size = batch_size)
  num.batch <- length(mini.batch$train)
  
  # grad_x <- 0 #For BB
  
  #Storing epoch gradients of the first element of each region
  grad.select <- matrix(, nrow=n.mask*2,ncol=num.batch)
  
  
  time.epoch <-  Sys.time()
  #Start batch
  for(b in 1:num.batch){
    
    
    minibatch.size <- length(mini.batch$train[[b]])
    
    print(paste0("Epoch: ",e, ", batch number: ", b))
    #3 Feed it to next layer
    
    hidden.layer <- apply(t(t(res3.dat[mini.batch$train[[b]], ]  %*% t(weights)) + bias), 2, FUN = relu)
    
    # Generate polynomial features (linear terms)
    poly_features <- as.matrix(hidden.layer %*% partial.gp.centroid)
    # Create the interaction features
    interaction_features <- poly_features * sex[mini.batch$train[[b]]]
    # Create the design matrix
    z.nb <- cbind(1,poly_features, sex[mini.batch$train[[b]]], interaction_features)
    #z.nb <- model.matrix(age[mini.batch$train[[b]]] ~ poly(as.matrix(hidden.layer %*% partial.gp.centroid), 1)*sex[mini.batch$train[[b]]])
    
    # print("pass z.nb")
    
    hs_fit_SOI <- fast_normal_lm(age[mini.batch$train[[b]]],z.nb) #This also gives the bias term
    
    beta_fit <- data.frame(HS = c(partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[2:(l.expan+1)],
                                  hs_fit_SOI$post_mean$betacoef[l.expan+2],
                                  partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[-1:-(l.expan+2)] ))
    l.bias <- hs_fit_SOI$post_mean$betacoef[1]
    
    hs_in.pred_SOI <- predict_fast_lm(hs_fit_SOI, z.nb, alpha = 0.95)$mean
    
    loss.train <- c(loss.train, mse(hs_in.pred_SOI,age[mini.batch$train[[b]]]))
    rsq.train <- c(rsq.train, rsqcal(age[mini.batch$train[[b]]],hs_in.pred_SOI))
    
    loss.train.male <- c(loss.train.male, mse(hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == 1)],age[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == 1)]))
    rsq.train.male <- c(rsq.train.male, rsqcal(age[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == 1)],hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == 1)]))
    
    loss.train.fmale <- c(loss.train.fmale, mse(hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == -1)],age[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == -1)]))
    rsq.train.f <- c(rsq.train.fmale, rsqcal(age[mini.batch$train[[b]]][which(sex[mini.batch$train[[b]]] == -1)],hs_in.pred_SOI[which(sex[mini.batch$train[[b]]] == -1)]))
    
    temp.sum.sum.sq <- apply(weights, 1, FUN = function(x) sum(x^2))
    map.train <- c(map.train,n.train/2*log(y.sigma) +1/(2*y.sigma)*n.train*mse(hs_in.pred_SOI,age[mini.batch$train[[b]]]) +n.mask/2*log(y.sigma) +n.mask*p.dat/2*log(y.sigma) + 1/(2*y.sigma)*sum(1/prior.var*(temp.sum.sum.sq))  +1/2*sum(c(bias)^2) )
    
    
    #Validation
    #Layers
    hidden.layer.test <- apply(t(t(res3.dat[train.test.ind$test, ] %*% t(weights)) + bias), 2, FUN = relu)
    
    poly_features.test <- as.matrix(hidden.layer.test %*% partial.gp.centroid)
    # Create the interaction features
    interaction_features.test <- poly_features.test * sex[train.test.ind$test]
    # Create the design matrix
    z.nb.test <- cbind(1,poly_features.test, sex[train.test.ind$test], interaction_features.test)
    hs_pred_SOI <- predict_fast_lm(hs_fit_SOI, z.nb.test, alpha = 0.95)$mean
    
    loss.val <- c(loss.val, mse(hs_pred_SOI,age[train.test.ind$test]))
    rsq.val <- c(rsq.val, rsqcal(age[train.test.ind$test],hs_pred_SOI))
    
    loss.val.male <- c(loss.val.male, mse(hs_pred_SOI[which(sex[train.test.ind$test] == 1)],age[train.test.ind$test][which(sex[train.test.ind$test] == 1)]))
    rsq.val.male <- c(rsq.val.male, rsqcal(age[train.test.ind$test][which(sex[train.test.ind$test] == 1)],hs_pred_SOI[which(sex[train.test.ind$test] == 1)]))
    
    loss.val.fmale <- c(loss.val.fmale, mse(hs_pred_SOI[which(sex[train.test.ind$test] == -1)],age[train.test.ind$test][which(sex[train.test.ind$test] == -1)]))
    rsq.val.fmale <- c(rsq.val.fmale, rsqcal(age[train.test.ind$test][which(sex[train.test.ind$test] == -1)],hs_pred_SOI[which(sex[train.test.ind$test] == -1)]))
    
    
    ##Keeping the last 5 epochs predictions
    if(e >= (epoch-record.epoch)){
      pred.train.ind <- c(pred.train.ind,mini.batch$train[[b]]) 
      pred.train.val <- c(pred.train.val,hs_in.pred_SOI)
      pred.test.ind <- c(pred.test.ind,train.test.ind$test) 
      pred.test.val <- c(pred.test.val,hs_pred_SOI) 
      
      if(e==(epoch-record.epoch+1)){ #Added +1 here
        mean.num <- 1
        mean.weights <- weights
        mean.bias <- bias
        
        mean.abs.weights <- abs(weights)
        mean.abs.bias <- abs(bias)
        
      }else{
        mean.weights <- (mean.abs.weights*mean.num + weights)/(mean.num+1)
        mean.bias <- (mean.abs.bias*mean.num + bias)/(mean.num+1)
        #Mean Magnitude
        mean.abs.weights <- (mean.abs.weights*mean.num + abs(weights))/(mean.num+1)
        mean.abs.bias <- (mean.abs.bias*mean.num + abs(bias))/(mean.num+1)
        
        mean.num <- mean.num + 1
      }
      
    }
    
    if(it.num < epoch*num.batch){
      
      #Update weight
      
      #4Update the full weights, fit GP against the full weights using HS-prior model to get normally dist thetas
      grad.loss <- age[mini.batch$train[[b]]] - hs_in.pred_SOI
      
      #Update weight
      #Update weight
      grad <- array(,dim = c(minibatch.size,dim(weights)))
      for(j in 1:n.mask){ #nrow of weights = n.mask
        grad[,j,] <- -1/y.sigma*c(grad.loss)*(beta_fit$HS[j]+beta_fit$HS[j+n.mask+1]*sex[mini.batch$train[[b]]])*c(relu.prime(hidden.layer[,j]))*res3.dat[mini.batch$train[[b]], ]  
      }
      
      #Take batch average
      grad.m <- apply(grad, c(2,3), mean)
      #####

      grad.b <- 1/y.sigma* c(grad.loss)*t(beta_fit$HS[1:n.mask]*t(apply(hidden.layer, 2, FUN = relu.prime))) +1/y.sigma* c(grad.loss)*t(beta_fit$HS[(n.mask+2):(n.mask*2+1)]*t(apply(hidden.layer, 2, FUN = relu.prime)*sex[mini.batch$train[[b]]]))
      
      #Take batch average
      grad.b.m <- c(apply(grad.b, c(2), mean))
      
      # Update sigma
      grad.sigma.m <- mean(length(train.test.ind$train)/(2*y.sigma) - length(train.test.ind$train)/(2*y.sigma^2)*c(grad.loss)^2-1/(2*y.sigma^2)*sum(c(weights/prior.var)^2)+1/(2*y.sigma)*p.dat*n.mask)
      ####Note here of the static equal prior.var
      #Update theta matrix
      weights <- weights*(1-learning_rate*1/(prior.var*y.sigma)) - learning_rate*grad.m * length(train.test.ind$train) - matrix(rnorm(p.dat*n.mask,0,gaus.sd), ncol = p.dat, nrow = n.mask)
      #Note that updating weights at the end will be missing the last batch of last epoch
      
      #Update bias
      bias <- bias*(1-learning_rate*1/(prior.var.bias)) - learning_rate*c(grad.b.m) * length(train.test.ind$train) - rnorm(n.mask,0,gaus.sd)
      
      # Update sigma
      y.sigma <- y.sigma - learning_rate*(grad.sigma.m) - rnorm(1,0,gaus.sd)
      y.sigma.vec <- c(y.sigma.vec,y.sigma)
      
      #Update Cv
      for(i in 1:n.mask){
        alpha.shape <- alpha.init[i] + length(weights[i,])/2
        # print(paste0("i =",i,', alpha = ',alpha.shape))
        # alpha.shape <- alpha.init[i] # Keep alpha the same
        beta.scale <- beta.init[i] + sum(weights[i,]^2)/(2*y.sigma)
        prior.var[i] <- rinvgamma(n = 1, alpha.shape, beta.scale)
        # print(paste0("i =",i,', prior.var[i] = ',prior.var[i]))
        
        conj.alpha[i,it.num] <- alpha.shape
        conj.beta[i,it.num] <- beta.scale
        conj.invgamma[i,it.num] <- prior.var[i]
      }
    }
    
    it.num <- it.num +1
    learning_rate <- start.a*(start.b+it.num)^(-start.gamma)
    gaus.sd <- sqrt(2*learning_rate)
    
    #Record grad.select
    #16 Mar, should grad.m be positive or negative?
    grad.select[,b] <- c(c((weights/(prior.var*y.sigma) - grad.m * length(train.test.ind$train))[,1]),c(bias/(prior.var.bias) - c(grad.b.m) * length(train.test.ind$train)))
    
    
    print(paste0("training loss: ",mse(hs_in.pred_SOI,age[mini.batch$train[[b]]])))
    print(paste0("validation loss: ",mse(hs_pred_SOI,age[train.test.ind$test])))
  }
  
  #Record variance of grad.select
  grad.noise[,e] <- apply(grad.select, 1, var)
  
  print(paste0("epoch: ",e," out of ",epoch, ", time taken for this epoch: ",Sys.time() -time.epoch))
  print(paste0("sigma^2: ",y.sigma))
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
# write.csv(pre.lr.vec,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_prelr_',"_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(c(partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[2:(l.expan+1)],
                              hs_fit_SOI$post_mean$betacoef[l.expan+2],
                              partial.gp.centroid%*%hs_fit_SOI$post_mean$betacoef[-1:-(l.expan+2)] )),paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_lweights_',"_jobid_",JobId,'.feather'))

#gradient Noise
write.csv(grad.noise,paste0("/well/nichols/users/qcv214/KGPNN/pile/re_",filename,"_gradnoise_","_jobid_",JobId,".csv"), row.names = FALSE)

#Mean parameters
write_feather(as.data.frame(mean.weights),paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_meanweights_',"_jobid_",JobId,'.feather'))
write.csv(mean.bias,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_meanbias_',"_jobid_",JobId,".csv"), row.names = FALSE)

#Mean absparameters
write_feather(as.data.frame(mean.abs.weights),paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_meanabsweights_',"_jobid_",JobId,'.feather'))
write.csv(mean.abs.bias,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_meanabsbias_',"_jobid_",JobId,".csv"), row.names = FALSE)


temp.frame <- as.data.frame(rbind(pred.train.ind,pred.train.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
# temp.frame<- t(tail(t(temp.frame),length(train.test.ind$train)*5))
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_inpred_',"_jobid_",JobId,'.feather'))
temp.frame <- as.data.frame(rbind(pred.test.ind,pred.test.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
# temp.frame<- t(tail(t(temp.frame),length(train.test.ind$test)*5))
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_outpred_',"_jobid_",JobId,'.feather'))

#Inverse gamma param
write.csv(conj.alpha,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_alpha_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(conj.beta,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_beta_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(conj.invgamma,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_',filename,'_invgam_',"_jobid_",JobId,".csv"), row.names = FALSE)
