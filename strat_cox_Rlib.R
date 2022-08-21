##### lasso wrapper in R
##### This function will be called iteratively in "strat_cox_lib.cpp" 
##### to solve an approximated lasso problem.
##### Please make sure to include this function before "strat_cox_lib.cpp".
# dl: first-order derivative of loss function
# ddl: second-order derivative of loss function
# beta_old: estimated coefficient vector beta from last iteration step
# lam: a scaler tuning parameter used for lasso penalty
lasso_wrapper_r <- function(dl, ddl, beta_old, lam) {
  A <- chol(ddl) # upper triangular A such that t(A)%*%A = ddl
  phi <- as.vector(dl) - as.vector(ddl%*%beta_old)
  yy <- as.vector(solve(a=t(A), b=-phi))
  nobs <- length(yy)
  beta_new <- as.vector(coef(glmnet(x=A, y=yy, family="gaussian", alpha=1, standardize=F, intercept=F, lambda=lam/nobs))[-1])
  return(beta_new)
}


