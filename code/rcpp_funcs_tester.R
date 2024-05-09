#Testing Rcpp

library(Rcpp)


############ apply(t(t(x  %*% t(weights)) + bias), 2, FUN = relu) 
cppFunction('
NumericMatrix computeHiddenLayer(NumericMatrix data, NumericMatrix weights, NumericVector bias) {
  int n = data.nrow();
  int m = weights.ncol();
  
  NumericMatrix hiddenLayer(n, m);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      double sum = 0;
      for (int k = 0; k < data.ncol(); ++k) {
        sum += data(i, k) * weights(k, j);
      }
      hiddenLayer(i, j) = std::max(0.0, sum + bias[j]);
    }
  }
  
  return hiddenLayer;
}
', depends = "Rcpp")

######### Matrix %*% Matrix
cppFunction('
NumericMatrix matrixMultiply(NumericMatrix A, NumericMatrix B) {
  if (A.ncol() != B.nrow()) {
    stop("Incompatible dimensions");
  }
  int n = A.nrow(), k = B.ncol(), m = A.ncol();
  NumericMatrix C(n, k);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      double sum = 0;
      for (int l = 0; l < m; ++l) {
        sum += A(i, l) * B(l, j);
      }
      C(i, j) = sum;
    }
  }
  return C;
}')

######### Matrix %*% vector
cppFunction('
NumericVector matrixVectorMultiply(NumericMatrix A, NumericVector vec) {
  if (A.ncol() != vec.size()) {
    stop("Incompatible dimensions");
  }
  int n = A.nrow(), m = A.ncol();
  NumericVector result(n);
  for (int i = 0; i < n; ++i) {
    double sum = 0;
    for (int j = 0; j < m; ++j) {
      sum += A(i, j) * vec[j];
    }
    result[i] = sum;
  }
  return result;
}')


#### MSE calculator
cppFunction('
double mseCpp(NumericVector pred, NumericVector trueVals) {
  int n = pred.size();
  double sum = 0;
  for(int i = 0; i < n; ++i) {
    sum += std::pow(pred[i] - trueVals[i], 2);
  }
  return sum / n;
}')
pred <- c(1, 2, 3, 4)
trueVals <- c(1, 2, 4, 4)
mseCpp(pred, trueVals)
mse(pred,trueVals)

####R squared calculator
cppFunction('
double rsqCpp(NumericVector trueVals, NumericVector pred) {
  int n = trueVals.size();
  double rss = 0;
  double tss = 0;
  double meanTrue = mean(trueVals);
  
  for(int i = 0; i < n; ++i) {
    rss += std::pow(trueVals[i] - pred[i], 2);
    tss += std::pow(trueVals[i] - meanTrue, 2);
  }
  
  double rsq = (1 - rss / tss) * 100;
  return rsq;
}')
rsqCpp(trueVals, pred)
rsqcal(trueVals, pred)

# Assuming you have defined `res3.dat`, `weights`, `bias`, and `partial.gp.centroid` appropriately

x<- matrix(1:10,nrow = 2,ncol = 5)
weights <- matrix(c(1:50,-(51:60)), ncol = 5, nrow = 12)
bias <- 1:12

time.train <-  Sys.time()
hidden_layer <- computeHiddenLayer(x, t(weights), bias)
time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)
time.train <-  Sys.time()
hidden.layer <- apply(t(t(x  %*% t(weights)) + bias), 2, FUN = relu) 
time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

time.train <-  Sys.time()
poly_features <- matrixMultiply(hidden.layer, partial.gp.centroid)
time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

time.train <-  Sys.time()
poly_features <- as.matrix(hidden.layer %*% partial.gp.centroid)
time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)


matrixVectorMultiply(hidden.layer, 1:12) #This already does c() for you



## Grad update
cppFunction('
NumericVector updateWeights(int minibatchSize, int nMask, double ySigma, 
                            NumericVector gradLoss, NumericVector betaHS, 
                            NumericMatrix hiddenLayer, NumericMatrix res3dat, int weightCols) {
  
  // Manually calculate the product of the dimensions for the grad array
  int totalSize = minibatchSize * nMask * weightCols;
  NumericVector grad(totalSize); // Create a vector with the calculated total size
            
            for (int i = 0; i < minibatchSize; ++i) {
              for (int j = 0; j < nMask; ++j) {
                for (int k = 0; k < weightCols; ++k) {
                  int index = i + minibatchSize * (j + nMask * k); // Correctly calculate the index
                  double reluPrimeValue = hiddenLayer(i, j) > 0 ? 1.0 : 0.0;
                  grad[index] = -1.0 / ySigma * gradLoss[i] * betaHS[j] * reluPrimeValue * res3dat(i, k);
                }
              }
            }
            
            // Optionally, if you need to return the vector with dimensions set as a 3D array
            grad.attr("dim") = IntegerVector::create(minibatchSize, nMask, weightCols);
            
            return grad;
            }')

weightCols <- dim(weights)[2]
grad <- updateWeights(minibatchSize, nMask, ySigma, gradLoss, betaHS, hiddenLayer, res3dat, weightCols)
grad2 <- updateWeights(nrow(x), 12,1, c(3,4), 1:12, hidden.layer,x, weightCols) #This already output 3D


## Grad mean
cppFunction('
NumericMatrix calculateMeanAdjusted(NumericVector grad) {
  // Extract dimensions from grad
  Dimension dim = grad.attr("dim");
  int minibatchSize = dim[0];
  int nMask = dim[1];
  int weightCols = dim[2];
  
  NumericMatrix gradMean(nMask, weightCols);
  
  for (int j = 0; j < nMask; ++j) {
    for (int k = 0; k < weightCols; ++k) {
      double sum = 0;
      for (int i = 0; i < minibatchSize; ++i) {
        // Correctly calculate the linear index for the 1D storage of the 3D array
        int index = i + minibatchSize * (j + nMask * k);
        sum += grad[index];
      }
      gradMean(j, k) = sum / minibatchSize;
    }
  }
  
  return gradMean;
}')

calculateMeanAdjusted(grad) # same as apply(grad, c(2,3), mean) All correct.


## Grad bias update
cppFunction('
NumericMatrix updateGradB(int minibatchSize, int nMask, double ySigma, 
                          NumericVector gradLoss, NumericVector betaHS, 
                          NumericMatrix hiddenLayer) {
  NumericMatrix gradB(minibatchSize, nMask);
  
  for (int j = 0; j < nMask; ++j) {
    for (int i = 0; i < minibatchSize; ++i) {
      double reluPrimeValue = hiddenLayer(i, j) > 0 ? 1.0 : 0.0;
      gradB(i, j) = -1.0 / ySigma * gradLoss[i] * betaHS[j] * reluPrimeValue;
    }
  }
  
  return gradB;
}')

## Grad bias col mean
cppFunction('
NumericVector calculateColumnMeans(NumericMatrix gradB) {
  int nCols = gradB.ncol();
  NumericVector columnMeans(nCols);
  
  for (int j = 0; j < nCols; ++j) {
    double sum = 0;
    for (int i = 0; i < gradB.nrow(); ++i) {
      sum += gradB(i, j);
    }
    columnMeans[j] = sum / gradB.nrow();
  }
  
  return columnMeans;
}
')

calculateColumnMeans(weights) #seems right




