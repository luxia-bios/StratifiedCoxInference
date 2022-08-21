#################################################
### High-D inference for stratified Cox model ###
#################################################

### This file contains code that simulates data in a stratified Cox model
### and draws inference on regression coefficients using the proposed method
### De-biased Lasso via Quadratic Programming (DBL-QP).

rm(list=ls())
library(mvtnorm)
library(glmnet)
library(Rcpp)
library(RcppArmadillo)
library(survival)
library(quadprog)

### set working directory
# setwd("your directory")
setwd("./")

# include R library before Rcpp library
source("strat_cox_Rlib.R")
sourceCpp("strat_cox_Clib.cpp")

### setup for strata and true beta
set.seed(2020)

################################
##### simulation scenarios #####
################################

##### Note: scenario 1 runs the fastest and thus is used for illustration
#####       other scenarios can run after being uncommented (but will take considerably longer time)

## scenario 1: (K, n, p) = (10,100,10)
K <- 10 # number of strata
n <- 100 # number of patients within each stratum
size_f <- rep(n, times=K) # vector of stratum size
total_n <- sum(size_f) # total number of observations
p <- 10 # number of covariates
nfold_cv_kk <- K # number of folds for CV in de-biased lasso; strata as sampling units (gamma)

## scenario 2: (K, n, p) = (10,100,100)
# K <- 10 # number of strata
# n <- 100 # number of patients within each stratum
# size_f <- rep(n, times=K) # vector of stratum size
# total_n <- sum(size_f) # total number of observations
# p <- 100 # number of covariates
# nfold_cv_kk <- K # number of folds for CV in de-biased lasso; strata as sampling units (gamma)

## scenario 3: (K, n, p) = (5,200,100)
# K <- 5 # number of strata
# n <- 200 # number of patients within each stratum
# size_f <- rep(n, times=K) # vector of stratum size
# total_n <- sum(size_f) # total number of observations
# p <- 100 # number of covariates
# nfold_cv_kk <- K # number of folds for CV in de-biased lasso; strata as sampling units (gamma)

## scenario 4: (K, p) = (40, 100), n~Poisson(40)
# K <- 40 # number of strata
# n <- 40 # number of patients within each stratum
# size_f <- rpois(n=K, lambda=n) # vector of stratum size
# total_n <- sum(size_f) # total number of observations
# p <- 100 # number of covariates
# nfold_cv_kk <- 10 # number of folds for CV in de-biased lasso; strata as sampling units (gamma)


##### remaining of the fixed setup #####

struct <- "ar1" # covariance structure for p-dim covariates
s0 <- 4 # number of true model size
rho <- 0.5 # correlation parameter for covariates
covmat <- matrix(0, nrow=p, ncol=p) # covariance structure for covariates
if(struct == "indep") {
  covmat <- diag(p)
} else if(struct == "ar1") {
  covmat <- rho^(abs(outer(1:p,1:p,"-")))
} else if(struct == "cs") {
  covmat <- rho*rep(1,p)%*%t(rep(1,p)) + (1-rho)*diag(p)
} else if(struct == "invcs") {
  covmat <- rho*rep(1,p)%*%t(rep(1,p)) + (1-rho)*diag(p)
  covmat <- solve(covmat)
  covmat <- diag(1/sqrt(diag(covmat)))%*%covmat%*%diag(1/sqrt(diag(covmat)))
}
large_signal <- 1 # large signal strength
small_signal <- 0.3 # small signal strength
# true beta
len_beta <- 11
idx_beta <- 5 # changing idx_beta from 1 to 11 will give different values of beta_true[1], ranging from 0 to 2
beta_level_seq <- seq(0, 2, length.out=len_beta) 
beta_true <- rep(0,p)
beta_true[sample(2:p, size=s0)] <- c(rep(small_signal,s0/2), rep(large_signal,s0/2))
beta_true[1] <- beta_level_seq[idx_beta]
signal_pos <- which(beta_true != 0) # true signal positions

# baseline hazards (fixed throughout simulations)
baseline_haz <- seq(from=0.5, to=1, length.out=K)
baseline_haz <- rep(baseline_haz, times=size_f)
baseline_cens <- seq(from=0.5, to=1, length.out=K)/5
baseline_cens <- rep(baseline_cens, time=size_f)

#########################################
pre_time <- proc.time()
## the following survival outcomes and covariates should be run with changing seeds if more than one replications
## set.seed(your_seed)
set.seed(123456789) 

#### data generation ####

# covariates
X <- rmvnorm(total_n, mean = rep(0,p), sigma = covmat)
X <- ifelse(abs(X)>3, sign(X)*3, X) 
# latent survival and censoring times
rand_unif <- runif(total_n)
latent_t <- rexp(total_n, rate=(baseline_haz*as.vector(exp(X%*%beta_true)))) 
latent_c <- rexp(total_n, rate=(baseline_cens*as.vector(exp(X%*%beta_true))))
# event indicator
delta <- as.numeric(latent_t<=latent_c); mean(1-delta)
# observed survival time
time <- latent_t*delta+latent_c*(1-delta)
# vector of stratum index
strata_idx <- rep(1:K, times=size_f)

# order data as time increases within each stratum
tmp_dat <- data.frame(strata_idx=strata_idx, time=time, delta=delta, X=X)
tmp_dat <- tmp_dat[order(strata_idx, time),]
time <- tmp_dat$time
delta <- tmp_dat$delta
strata_idx <- tmp_dat$strata_idx
X <- as.matrix(tmp_dat[,(1:p)+3])
rm(list="tmp_dat")

### obtain lasso estimator ###

# setup for cross-validation for lasso
nfold <- 5 # nfold for cross-validation in lasso
n_lambda <- 30 # number of lambdas used in cross-validation for lasso
tol <- 1.0e-6 # tolerance level for convergence of algorithm
maxiter <- 50000 # max number of iterations for lasso

# generating random index vector for K-fold cross-validation in lasso
cv_idx0 <- NULL
for(kk in 1:K) {
  tmp_cv_idx <- cut(seq(1,sum(strata_idx==kk)), breaks=nfold, labels=F)
  tmp_cv_idx <- sample(x=tmp_cv_idx, size=length(tmp_cv_idx), replace=F)
  cv_idx0 <- c(cv_idx0, tmp_cv_idx)
}

# lasso estimator selected from cross-validation
beta_init <- rep(0,p)
obj_lasso <- cv_lasso_stratCox_cpp(X, time, delta, beta_init, cv_idx0, strata_idx, K,
                                               nfold, n_lambda, tol, maxiter)
beta_glmnet <- as.vector(obj_lasso$beta_opt) # lasso estimator

# function cv_lasso_stratCox_cpp automatically outputs 
# first and second order derivatives of neg log partial likelihood
# and Sigma_hat (score_sq_glmnet) with lasso estimates plugged in
neg_dloglik_glmnet <- obj_lasso$neg_dloglik
neg_ddloglik_glmnet <- obj_lasso$neg_ddloglik
score_sq_glmnet <- obj_lasso$score_sq
r <- eigen(score_sq_glmnet)
r$values[r$values<=1.0e-14] <- 0


#############################################################################
## proposed DBL-QP method for drawing inference on regression coefficients ##
#############################################################################

### first step: cross validation for DBL-QP to select tuning parameter gamma ###

# setup for cross-validation

n_multi <- 30
multiplier_seq <- exp(seq(from=log(0.01), to=log(0.6), length.out=n_multi))
alpha_cv <- 0.1 # FWER control level for constructing active set via multiple testing for CV

all_cvpl2 <- array(NA, length(multiplier_seq)) # lik evaluation on left-out data
cut_strata_idx <- cut(1:K, breaks=nfold_cv_kk, labels=F)
split_strata_idx <- sample(x=cut_strata_idx, size=K, replace=F) # index for cross-validation folds

# partial likelihood evaluation on left-out data as selection criterion on all potential gamma values
all_cvpl2 <- array(NA, length(multiplier_seq)) 

# start of cross-validation to select gamma

for(jj in 1:length(multiplier_seq)) { # for cvpl
  multiplier <- multiplier_seq[jj] # gamma = multiplier*sqrt(log(p)/total_n)

  cvpl2 <- 0 # partial likelihood evaluation for current gamma value
  
  for(k in 1:nfold_cv_kk) { # for cv
    cv_idx <- which(strata_idx %in% (c(1:K)[split_strata_idx==k])) # test data index; rest as training data
    train_x <- X[-c(cv_idx),] # training data x
    test_x <- X[cv_idx,] # test data x
    train_time <- time[-c(cv_idx)]
    test_time <- time[cv_idx]
    train_delta <- delta[-c(cv_idx)]
    test_delta <- delta[cv_idx]
    train_strata_idx <- strata_idx[-c(cv_idx)]
    test_strata_idx <- strata_idx[cv_idx]
    
    # taking too long for lasso cross-validation within DBL-QP cross-validation when p is large
    # using lambda_opt from all data here
    # return lasso estimator on training data
    train_strata_idx_tmp <- table(train_strata_idx)
    train_strata_idx_tmp <- rep(1:length(train_strata_idx_tmp), times=train_strata_idx_tmp) # reset training data strata_idx to consecutive numbers
    beta_glmnet_train <- lasso_stratCox_cpp_ext(obj_lasso$lambda_opt, train_x, train_time, 
                                                train_delta, rep(0,p), train_strata_idx_tmp, length(unique(train_strata_idx_tmp)))
    
    # compute first and second order derivatives of neg log partial likelihood,
    # and Sigma_hat (score_sq_train) on training data by plugging in lasso estimates
    neg_loglik_glmnet_train <- 0
    neg_dloglik_glmnet_train <- rep(0, p)
    neg_ddloglik_glmnet_train <- matrix(0, nrow=p, ncol=p)
    score_sq_train <- matrix(0, nrow=p, ncol=p)
    all_neg_loglik_functions_cpp_ext(neg_loglik_glmnet_train, neg_dloglik_glmnet_train, neg_ddloglik_glmnet_train,
                                     score_sq_train, train_x, train_time, train_delta,
                                     beta_glmnet_train, train_strata_idx_tmp, length(unique(train_strata_idx_tmp)))
    r_train <- eigen(score_sq_train)
    r_train$values[r_train$values<=1.0e-14] <- 0
    
    ### compute de-biased lasso estimator based on training data and current gamma
    
    b_hat_new <- rep(NA, p)
    se_new <- rep(NA, p)
    mu <- multiplier*sqrt(log(p)/total_n) # current gamma
    my_pos <- which(r_train$values > 0)
    my_rank <- sum(r_train$values > 0)
    Dmat <- diag(r_train$values[my_pos])
    dvec <- rep(0,my_rank)
    Amat <- t(rbind(-r_train$vectors[,my_pos]%*%Dmat, r_train$vectors[,my_pos]%*%Dmat))
    # solve QP problems using solve.QP()
    for(j in 1:p) {
      e_j <- rep(0, p); e_j[j] <- 1
      bvec <- (c(-e_j, e_j) - mu*rep(1,2*p))
      res <- solve.QP(Dmat=Dmat, dvec=dvec, Amat=Amat, bvec=bvec)
      m <- as.vector(r_train$vectors[,my_pos]%*%res$solution) +
        as.vector(r_train$vectors[,-my_pos]%*%rep(0, p-my_rank))
      # de-biased lasso
      b_hat_new[j] <- beta_glmnet_train[j] - as.numeric(m%*%neg_dloglik_glmnet_train)
      # model-based standard error
      se_new[j] <- sqrt(m[j]/nrow(train_x)) 
    }
    
    # threshold pval_new at level alpha_cv/p (Bonferroni correction) 
    # to obtain thresholded de-biased lasso estimator for CV evaluation
    pval_new <- 2*pnorm(abs(b_hat_new/se_new), lower.tail = F)
    tmp_beta <- b_hat_new*as.numeric(pval_new < (alpha_cv/p))
    # cross-validation evaluation
    test_strata_idx_tmp <- table(test_strata_idx)
    test_strata_idx_tmp <- rep(1:length(test_strata_idx_tmp), times=test_strata_idx_tmp)
    cvpl2 <- cvpl2 - all_neg_loglik_cpp_ext(test_x, test_time, test_delta, 
                                            tmp_beta, test_strata_idx_tmp, length(unique(test_strata_idx_tmp)))*nrow(test_x)
  } # end for cv
  all_cvpl2[jj] <- cvpl2
  
} # end for cvpl

##### second step: compute de-biased lasso estimator and final results for DBL-QP #####

# choosing the optimal tuning parameter gamma that maximizes all_cvpl2
multiplier <- multiplier_seq[which.max(all_cvpl2)]

## de-biased lasso results with optimal gamma selected from above
b_hat_new <- array(NA, p) # final de-biased lasso estimator
se_new <- array(NA, p) # final model-based standard error
theta_new <- matrix(NA, ncol=p, nrow=p) # final Theta_hat matrix estimation
mu_new <- multiplier*sqrt(log(p)/total_n) # optimal tuning parameter gamma selected
my_pos <- which(r$values > 0)
my_rank <- sum(r$values > 0)
Dmat <- diag(r$values[my_pos])
dvec <- rep(0,my_rank)
Amat <- t(rbind(-r$vectors[,my_pos]%*%Dmat, r$vectors[,my_pos]%*%Dmat))
for(j in 1:p) {
  e_j <- rep(0, p); e_j[j] <- 1
  bvec <- (c(-e_j, e_j) - mu_new*rep(1,2*p))
  res <- solve.QP(Dmat=Dmat, dvec=dvec, Amat=Amat, bvec=bvec)
  m <- as.vector(r$vectors[,my_pos]%*%res$solution) + 
    as.vector(r$vectors[,-my_pos]%*%rep(0, p-my_rank))
  b_hat_new[j] <- beta_glmnet[j] - as.numeric(m%*%neg_dloglik_glmnet)
  se_new[j] <- sqrt(m[j]/total_n) 
  theta_new[j,] <- m
}

## recording final confidence intervals for each beta_j and associated pvalues

alpha <- 0.05 # level alpha confidence interval for final results
v <- qnorm(alpha/2, lower.tail=F) # 1-alpha/2 upper quantitle in standard normal
# coverage of level alpha confidence intervals for each beta_j
cov_new <- as.numeric((beta_true <= (b_hat_new+v*se_new)) &
                        (beta_true >= (b_hat_new-v*se_new)))
# associated p-values by DBL-QP
pval_new <- 2*pnorm(abs(b_hat_new/se_new), lower.tail=F)


### MSPLE as comparison ###

obj_mle <- coxph(Surv(time, delta) ~ X + strata(strata_idx))
beta_mle <- as.vector(summary(obj_mle)$coefficients[,1])
se_mle <- as.vector(summary(obj_mle)$coefficients[,3])
cov_mle <- as.numeric((beta_true <= (beta_mle+v*se_mle)) &
                        (beta_true >= (beta_mle-v*se_mle)))
pval_mle <- 2*pnorm(abs(beta_mle/se_mle), lower.tail=F)


### oracle estimator as comparison ###

beta_oracle <- rep(0, p)
tmp_model <- coxph(Surv(time, delta) ~ X[,signal_pos] + strata(strata_idx))
beta_oracle[signal_pos] <- coef(tmp_model)
names(beta_oracle) <- NULL
se_oracle <- rep(0, p)
se_oracle[signal_pos] <- as.vector(summary(tmp_model)$coefficients[,3])
cov_oracle <- rep(0, p)
cov_oracle[signal_pos] <- as.numeric((beta_true[signal_pos] <= (beta_oracle[signal_pos]+v*se_oracle[signal_pos])) &
                                       (beta_true[signal_pos] >= (beta_oracle[signal_pos]-v*se_oracle[signal_pos])))
pval_oracle <- rep(0, p)
pval_oracle[signal_pos] <- 2*pnorm(abs(beta_oracle[signal_pos]/se_oracle[signal_pos]), lower.tail=F)

### checking three estimators combined ###

# from left to right: DBL-QP, MSPLE, Oracle
cbind(b_hat_new, beta_mle, beta_oracle)

# total time
total_time <- proc.time() - pre_time
total_time


