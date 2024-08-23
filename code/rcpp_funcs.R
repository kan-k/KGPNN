library(Rcpp)
library(RcppArmadillo)
############ apply(t(t(x  %*% t(weights)) + bias), 2, FUN = relu) 
# cppFunction('
# NumericMatrix computeHiddenLayer(NumericMatrix data, NumericMatrix weights, NumericVector bias) {
#   int n = data.nrow();
#   int m = weights.ncol();
# 
#   NumericMatrix hiddenLayer(n, m);
#   for (int i = 0; i < n; ++i) {
#     for (int j = 0; j < m; ++j) {
#       double sum = 0;
#       for (int k = 0; k < data.ncol(); ++k) {
#         sum += data(i, k) * weights(k, j);
#       }
#       hiddenLayer(i, j) = std::max(0.0, sum + bias[j]);
#     }
#   }
# 
#   return hiddenLayer;
# }
# ', depends = "Rcpp")

# cppFunction('arma::mat computeHiddenLayer(arma::mat& data, arma::mat& weights, arma::vec& bias) {
#   arma::mat hiddenLayer = data * weights;
#   hiddenLayer.each_row() += bias.t();
#   hiddenLayer.for_each([](arma::mat::elem_type& val) { val = std::max(0.0, val); });
#   return hiddenLayer;
# }', depends = "RcppArmadillo") #So this works!!!!

#below `computeHiddenLayer` doesn't have ReLU
cppFunction('arma::mat computeHiddenLayer(arma::mat& data, arma::mat& weights, arma::vec& bias) {
  arma::mat hiddenLayer = data * weights;
  hiddenLayer.each_row() += bias.t();
  return hiddenLayer;
}', depends = "RcppArmadillo") 

cppFunction('arma::mat ReLU(arma::mat& x){
  arma::mat y = x;
  y.elem(arma::find(x<=0)).zeros();
  // y = 1.0/(1.0 + exp(-x)); #sigmoid
  return y;
}', depends = "RcppArmadillo")

# hiddenLayer <- computeHiddenLayer(x, t(weights),bias)


######### Matrix %*% Matrix
# cppFunction('
# NumericMatrix matrixMultiply(NumericMatrix A, NumericMatrix B) {
#   if (A.ncol() != B.nrow()) {
#     stop("Incompatible dimensions");
#   }
#   int n = A.nrow(), k = B.ncol(), m = A.ncol();
#   NumericMatrix C(n, k);
#   for (int i = 0; i < n; ++i) {
#     for (int j = 0; j < k; ++j) {
#       double sum = 0;
#       for (int l = 0; l < m; ++l) {
#         sum += A(i, l) * B(l, j);
#       }
#       C(i, j) = sum;
#     }
#   }
#   return C;
# }')

cppFunction('arma::mat matrixMultiply(arma::mat A, arma::mat B) {
  return A * B;
}', depends = "RcppArmadillo")

######### Matrix %*% vector
# cppFunction('
# NumericVector matrixVectorMultiply(NumericMatrix A, NumericVector vec) {
#   if (A.ncol() != vec.size()) {
#     stop("Incompatible dimensions");
#   }
#   int n = A.nrow(), m = A.ncol();
#   NumericVector result(n);
#   for (int i = 0; i < n; ++i) {
#     double sum = 0;
#     for (int j = 0; j < m; ++j) {
#       sum += A(i, j) * vec[j];
#     }
#     result[i] = sum;
#   }
#   return result;
# }')
cppFunction('arma::mat matrixVectorMultiply(arma::mat A, arma::vec vec) {
  return A * vec;
}', depends = "RcppArmadillo")



#### MSE calculator
# cppFunction('
# double mseCpp(NumericVector pred, NumericVector trueVals) {
#   int n = pred.size();
#   double sum = 0;
#   for(int i = 0; i < n; ++i) {
#     sum += std::pow(pred[i] - trueVals[i], 2);
#   }
#   return sum / n;
# }')
cppFunction('double mseCpp(arma::vec predictions, arma::vec targets) {
  return arma::mean(arma::square(predictions - targets));
}', depends = "RcppArmadillo")

####R squared calculator
# cppFunction('
# double rsqCpp(NumericVector trueVals, NumericVector pred) {
#   int n = trueVals.size();
#   double rss = 0;
#   double tss = 0;
#   double meanTrue = mean(trueVals);
#   
#   for(int i = 0; i < n; ++i) {
#     rss += std::pow(trueVals[i] - pred[i], 2);
#     tss += std::pow(trueVals[i] - meanTrue, 2);
#   }
#   
#   double rsq = (1 - rss / tss) * 100;
#   return rsq;
# }')
cppFunction('
double rsqCpp(arma::vec trueVals, arma::vec pred) {
  double rss = arma::sum(arma::square(trueVals - pred));
  double tss = arma::sum(arma::square(trueVals - arma::mean(trueVals)));
  double rsq = (1 - rss / tss) * 100;
  return rsq;
}', depends = "RcppArmadillo")


## Grad update
# cppFunction('
# NumericVector updateWeights2(int minibatchSize, int nMask, double ySigma,
#                             NumericVector gradLoss, NumericVector betaHS,
#                             NumericMatrix hiddenLayer, NumericMatrix res3dat, int weightCols) {
# 
#   // Manually calculate the product of the dimensions for the grad array
#   int totalSize = minibatchSize * nMask * weightCols;
#   NumericVector grad(totalSize); // Create a vector with the calculated total size
# 
#             for (int i = 0; i < minibatchSize; ++i) {
#               for (int j = 0; j < nMask; ++j) {
#                 for (int k = 0; k < weightCols; ++k) {
#                   int index = i + minibatchSize * (j + nMask * k); // Correctly calculate the index
#                   double reluPrimeValue = hiddenLayer(i, j) > 0 ? 1.0 : 0.0;
#                   grad[index] = -1.0 / ySigma * gradLoss[i] * betaHS[j] * reluPrimeValue * res3dat(i, k);
#                 }
#               }
#             }
# 
#             // Optionally, if you need to return the vector with dimensions set as a 3D array
#             grad.attr("dim") = IntegerVector::create(minibatchSize, nMask, weightCols);
# 
#             return grad;
#             }')

# cppFunction('
# arma::cube updateWeights(int minibatchSize, int nMask, double ySigma, 
#                                   const arma::vec& gradLoss, const arma::vec& betaHS, 
#                                   const arma::mat& hiddenLayer, const arma::mat& res3dat) {
#     using namespace std;
#     arma::cube grad(nMask, minibatchSize, res3dat.n_cols, arma::fill::zeros);
#     
#     for (int j = 0; j < nMask; ++j) {
#         arma::mat test = -1.0 / ySigma * gradLoss * betaHS(j) % arma::sign(hiddenLayer.col(j));
#         arma::mat replicatedMat = arma::repmat(test, 1, res3dat.n_cols);
#         arma::mat gradMat = replicatedMat % res3dat;
#         // arma::colvec reluPrime = arma::conv_to<arma::colvec>::from(hiddenLayer.col(j) > 0);
#         // arma::colvec tempGrad = (-1.0 / ySigma) * gradLoss % reluPrime * betaHS(j);
#         
#         grad.slice(j) = gradMat;
# 
#         // Broadcasting tempGrad across all columns of res3dat
#         // for (size_t i = 0; i < tempGrad.n_elem; ++i) {
#         //    grad.tube(i, j) = tempGrad(i) * res3dat.row(i);
#         // }
#     }
# 
#     // Reshape grad to have dimensions as required (if needed)
#     // grad.reshape(minibatchSize, nMask, res3dat.n_cols);
#     
#     return grad;
# }', depends = "RcppArmadillo")


# cppFunction('
# arma::cube updateWeights2(int minibatchSize, int nMask, double ySigma, 
#                          const arma::vec& gradLoss, const arma::vec& betaHS, 
#                          const arma::mat& hiddenLayer, const arma::mat& res3dat) {
#   using namespace std;
#   arma::cube grad(minibatchSize,res3dat.n_cols, nMask, arma::fill::zeros);
#   
#   for (int j = 0; j < nMask; ++j) {
#     arma::mat test = -1.0 / ySigma * gradLoss * betaHS(j) % arma::sign(hiddenLayer.col(j)); //vector of size number of subjects
#     arma::mat replicatedMat = arma::repmat(test, 1, res3dat.n_cols); //a matrix of size number subjects x number of voxels  [res3dat.n_cols = 150k]
#     arma::mat gradMat = replicatedMat % res3dat;
#     grad.slice(j) = gradMat;
#   }
#   return grad;
# }', depends = "RcppArmadillo")

# uw2<- updateWeights(nrow(x), 12,1, c(3,4), 1:12, hidden.layer,x)
#########.  FIX below
cppFunction('
arma::cube updateWeights(int minibatchSize, int nMask, double ySigma, 
                                  const arma::vec& gradLoss, const arma::vec& betaHS, 
                                  const arma::mat& hiddenLayer, const arma::mat& res3dat) {
    using namespace std;
    arma::cube grad(minibatchSize,res3dat.n_cols, nMask, arma::fill::zeros);
    
    for (int j = 0; j < nMask; ++j) {
        arma::mat test = -1.0 / ySigma * gradLoss * betaHS(j) % arma::sign(hiddenLayer.col(j)); //vector of size number of subjects
        arma::mat gradMat = res3dat;
        gradMat.each_col() %= test; // Efficient element-wise multiplication
        grad.slice(j) = gradMat;
      }
    return grad;
}', depends = "RcppArmadillo")

####Bimodal calculation with interactions....
##### Note that I dont know how to return double 3D matrices for now, so I will create two weights functions for each mode
cppFunction('
arma::cube updateDoubleWeightsFirst(int minibatchSize, int nMask, double ySigma, 
                                  const arma::vec& gradLoss, const arma::vec& betaHS, 
                                  const arma::mat& hiddenLayer, const arma::mat& res3dat,
                                  const arma::mat& hiddenLayerdmn, const arma::mat& res3datdmn) {
    using namespace std;
    arma::cube grad(minibatchSize,res3dat.n_cols, nMask, arma::fill::zeros);
    for (int j = 0; j < nMask; ++j) {
        arma::mat test = (-1.0 / ySigma) * gradLoss % (betaHS(j) + (betaHS((j+(nMask*2))) * hiddenLayerdmn.col(j))) % arma::sign(hiddenLayer.col(j));
        arma::mat gradMat = res3dat;
        gradMat.each_col() %= test; // Efficient element-wise multiplication
        grad.slice(j) = gradMat;
    }
    return grad;
}', depends = "RcppArmadillo")

cppFunction('
arma::cube updateDoubleWeightsSecond(int minibatchSize, int nMask, double ySigma, 
                                  const arma::vec& gradLoss, const arma::vec& betaHS, 
                                  const arma::mat& hiddenLayer, const arma::mat& res3dat,
                                  const arma::mat& hiddenLayerdmn, const arma::mat& res3datdmn) {
    using namespace std;
    arma::cube grad(minibatchSize,res3dat.n_cols, nMask, arma::fill::zeros);
    for (int j = 0; j < nMask; ++j) {
        arma::mat test = (-1.0 / ySigma) * gradLoss % (betaHS(j+nMask) + (betaHS((j+(nMask*2))) * hiddenLayer.col(j))) % arma::sign(hiddenLayerdmn.col(j));
        arma::mat gradMat = res3datdmn;
        gradMat.each_col() %= test; // Efficient element-wise multiplication
        grad.slice(j) = gradMat;
    }
    return grad;
}', depends = "RcppArmadillo")
# 
# x<- matrix(1:10,nrow = 2,ncol = 5)
# weights <- matrix(c(1:50,-(51:60)), ncol = 5, nrow = 12)
# bias <- 1:12
# relu <- function(x) sapply(x, function(z) max(0,z))
# relu.prime <- function(x) sapply(x, function(z) 1.0*(z>0))
# hidden.layer <- apply(t(t(x  %*% t(weights)) + bias), 2, FUN = relu)

# 
# 
# grad3 <- updateWeights(nrow(x), 12,1, c(3,4), 1:12, hidden.layer,x)
# grad32<- lapply(grad3, 1, t)
# apply(grad, c(3,2), mean)
# 
# grad4 <- updateWeights2(nrow(x), 12,1, c(3,4), 1:12, hidden.layer,x,ncol(x))



## Grad mean
# cppFunction('
# NumericMatrix calculateMeanAdjusted(NumericVector grad) {
#   // Extract dimensions from grad
#   Dimension dim = grad.attr("dim");
#   int minibatchSize = dim[0];
#   int nMask = dim[1];
#   int weightCols = dim[2];
#   
#   NumericMatrix gradMean(nMask, weightCols);
#   
#   for (int j = 0; j < nMask; ++j) {
#     for (int k = 0; k < weightCols; ++k) {
#       double sum = 0;
#       for (int i = 0; i < minibatchSize; ++i) {
#         // Correctly calculate the linear index for the 1D storage of the 3D array
#         int index = i + minibatchSize * (j + nMask * k);
#         sum += grad[index];
#       }
#       gradMean(j, k) = sum / minibatchSize;
#     }
#   }
#   
#   return gradMean;
# }')

cppFunction('
arma::mat computeMean(arma::cube grad) {
  arma::mat gradMean = arma::mean(grad, 0);
  return gradMean.t();
}',depends = "RcppArmadillo")


## Grad bias update
# cppFunction('
# NumericMatrix updateGradB(int minibatchSize, int nMask, double ySigma, 
#                           NumericVector gradLoss, NumericVector betaHS, 
#                           NumericMatrix hiddenLayer) {
#   NumericMatrix gradB(minibatchSize, nMask);
#   
#   for (int j = 0; j < nMask; ++j) {
#     for (int i = 0; i < minibatchSize; ++i) {
#       double reluPrimeValue = hiddenLayer(i, j) > 0 ? 1.0 : 0.0;
#       gradB(i, j) = -1.0 / ySigma * gradLoss[i] * betaHS[j] * reluPrimeValue;
#     }
#   }
#   
#   return gradB;
# }')
# updateGradB(nrow(x), 12,1, c(3,4), 1:12, hidden.layer)

cppFunction('
arma::mat updateGradB(int minibatchSize, int nMask, double ySigma, 
                               const arma::vec& gradLoss, const arma::vec& betaHS, 
                               const arma::mat& hiddenLayer) {
    arma::mat gradB(minibatchSize, nMask, arma::fill::zeros);

    for (int j = 0; j < nMask; ++j) {
        // Compute ReLU prime for the j-th mask across all minibatch samples
        arma::vec reluPrime = arma::conv_to<arma::vec>::from(hiddenLayer.col(j) > 0);

        // Vectorized computation for the j-th column of gradB
        gradB.col(j) = (-1.0 / ySigma) * gradLoss % reluPrime * betaHS[j];
    }

    return gradB;
}', depends = "RcppArmadillo")
# updateGradBOptimized(nrow(x), 12,1, c(3,4), 1:12, hidden.layer)

cppFunction('
arma::mat updateGradBFirst(int minibatchSize, int nMask, double ySigma, 
                               const arma::vec& gradLoss, const arma::vec& betaHS, 
                               const arma::mat& hiddenLayer, const arma::mat& hiddenLayerdmn) {
    arma::mat gradB(minibatchSize, nMask, arma::fill::zeros);

    for (int j = 0; j < nMask; ++j) {
        // Compute ReLU prime for the j-th mask across all minibatch samples
        arma::vec reluPrime = arma::conv_to<arma::vec>::from(hiddenLayer.col(j) > 0);

        // Vectorized computation for the j-th column of gradB
        gradB.col(j) = (-1.0 / ySigma) * gradLoss % (betaHS(j) + (betaHS((j+(nMask*2))) * hiddenLayerdmn.col(j)))  % reluPrime;
    }

    return gradB;
}', depends = "RcppArmadillo")

cppFunction('
arma::mat updateGradBSecond(int minibatchSize, int nMask, double ySigma, 
                               const arma::vec& gradLoss, const arma::vec& betaHS, 
                               const arma::mat& hiddenLayer, const arma::mat& hiddenLayerdmn) {
    arma::mat gradB(minibatchSize, nMask, arma::fill::zeros);

    for (int j = 0; j < nMask; ++j) {
        // Compute ReLU prime for the j-th mask across all minibatch samples
        arma::vec reluPrime = arma::conv_to<arma::vec>::from(hiddenLayerdmn.col(j) > 0);

        // Vectorized computation for the j-th column of gradB
        gradB.col(j) = (-1.0 / ySigma) * gradLoss % (betaHS(j+nMask) + (betaHS((j+(nMask*2))) * hiddenLayer.col(j)))  % reluPrime;
    }

    return gradB;
}', depends = "RcppArmadillo")


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



########################################For GPNN-GP-OLS version###################################
#Updating theta
cppFunction('
arma::cube updateWeightsGP(int minibatchSize, int nMask, double ySigma, 
                                  const arma::vec& gradLoss, const arma::vec& betaHS, 
                                  const arma::mat& hiddenLayer, const arma::cube& res3dat) {
    using namespace std;
    arma::cube grad(minibatchSize,res3dat.n_slices, nMask, arma::fill::zeros);
    
    for (int j = 0; j < nMask; ++j) {
        arma::mat test = -1.0 / ySigma * gradLoss * betaHS(j) % arma::sign(hiddenLayer.col(j)); //vector of size number of subjects
        arma::mat gradMat = res3dat.row(j);
        gradMat.each_col() %= test; // Efficient element-wise multiplication
        grad.slice(j) = gradMat;
      }
    return grad;
}', depends = "RcppArmadillo")

#Updating theta with interactions
cppFunction('
arma::cube updateThetainter(int minibatchSize, int nMask, int numLatClass, 
                                double ySigma, const arma::vec& gradLoss, 
                                const arma::vec& betaHS, const arma::mat& hiddenLayer, 
                                const arma::mat& coHiddenLayer, const arma::cube& res3dat) {
    using namespace std;
    arma::cube grad(minibatchSize,res3dat.n_slices, nMask, arma::fill::zeros);
    
    for (int j = 0; j < nMask; ++j) {
        // Calculate index offsets for the additional terms
        int offset = nMask + numLatClass + 1 + (j - 1) * numLatClass;

        // Initialize additionalTerms with the first component
        arma::vec additionalTerms = betaHS(j) + betaHS(offset) * coHiddenLayer.col(0);

        // Compute the remaining additional terms using a for loop
        for (int k = 1; k < numLatClass; ++k) {
            additionalTerms += betaHS(offset + k) * coHiddenLayer.col(k);
        }

        // Complete gradient term calculation
        arma::vec test = -1.0 / ySigma * gradLoss % additionalTerms % arma::sign(hiddenLayer.col(j));
        arma::mat gradMat = res3dat.row(j);
        gradMat.each_col() %= test; // Efficient element-wise multiplication
        grad.slice(j) = gradMat;
    }
    return grad;
}', depends = "RcppArmadillo")

#Updating bias with interactions
cppFunction('
arma::mat updateBiasGPinter(int minibatchSize, int nMask, int numLatClass, 
                                double ySigma, const arma::vec& gradLoss, 
                                const arma::vec& betaHS, const arma::mat& hiddenLayer, 
                                const arma::mat& coHiddenLayer) {
    arma::mat gradB(minibatchSize, nMask, arma::fill::zeros);

    for (int j = 0; j < nMask; ++j) {
        // Calculate index offsets for the additional terms
        int offset = nMask + numLatClass + 1 + (j - 1) * numLatClass;

        // Initialize additionalTerms with the first component
        arma::vec additionalTerms = betaHS(j) + betaHS(offset) * coHiddenLayer.col(0);

        // Compute the remaining additional terms using a for loop
        for (int k = 1; k < numLatClass; ++k) {
            additionalTerms += betaHS(offset + k) * coHiddenLayer.col(k);
        }

        // Compute the result vector for the current weight index j
        arma::vec resultVec = -1.0 / ySigma * (gradLoss % additionalTerms % arma::sign(hiddenLayer.col(j)));

        // Assign the result to the appropriate column in gradB
        gradB.col(j) = resultVec;
    }
    return gradB;
}', depends = "RcppArmadillo")


#Bimodal GP-OLS
cppFunction('
arma::cube updateDoubleWeightsGPFirst(int minibatchSize, int nMask, double ySigma, 
                                  const arma::vec& gradLoss, const arma::vec& betaHS, 
                                  const arma::mat& hiddenLayer, const arma::cube& res3dat,
                                  const arma::mat& hiddenLayerdmn, const arma::cube& res3datdmn) {
    using namespace std;
    arma::cube grad(minibatchSize,res3dat.n_slices, nMask, arma::fill::zeros);
    for (int j = 0; j < nMask; ++j) {
        arma::mat test = (-1.0 / ySigma) * gradLoss % (betaHS(j) + (betaHS((j+(nMask*2))) * hiddenLayerdmn.col(j))) % arma::sign(hiddenLayer.col(j));
        arma::mat gradMat = res3dat.row(j);
        gradMat.each_col() %= test; // Efficient element-wise multiplication
        grad.slice(j) = gradMat;
    }
    return grad;
}', depends = "RcppArmadillo")

cppFunction('
arma::cube updateDoubleWeightsGPSecond(int minibatchSize, int nMask, double ySigma, 
                                  const arma::vec& gradLoss, const arma::vec& betaHS, 
                                  const arma::mat& hiddenLayer, const arma::cube& res3dat,
                                  const arma::mat& hiddenLayerdmn, const arma::cube& res3datdmn) {
    using namespace std;
    arma::cube grad(minibatchSize,res3dat.n_slices, nMask, arma::fill::zeros);
    for (int j = 0; j < nMask; ++j) {
        arma::mat test = (-1.0 / ySigma) * gradLoss % (betaHS(j+nMask) + (betaHS((j+(nMask*2))) * hiddenLayer.col(j))) % arma::sign(hiddenLayerdmn.col(j));
        arma::mat gradMat = res3datdmn.row(j);
        gradMat.each_col() %= test; // Efficient element-wise multiplication
        grad.slice(j) = gradMat;
    }
    return grad;
}', depends = "RcppArmadillo")


