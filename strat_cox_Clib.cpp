#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

double sign(double a) {
  double mysign;
  if(a == 0) {mysign=0;} else {mysign = (a>0 ? 1 : -1);}
  return mysign;
}

double absf(double a) {
  double value = (a>0 ? a : ((-1)*a));
  return value;
}

// parameters:
// lambda: tuning parameter lambda
// X: matrix of covariates, one row for one observation
// time: vector of observed survival time
// delta: vector of event indicator
// beta_init: vector of initial value for beta
// cv_idx: vector of cross-validation index; has the same length as time and take values in 1,...,nfold
// strata_idx: vector of stratum index
// N: number of strata
// tol: tolerance level for algorithm convergence
// maxiter: maximum number of iterations in algorithm

// calculate within stratum neg loglik functions
void within_neg_loglik_functions_cpp(double& neg_loglik, arma::colvec& neg_dloglik, arma::mat& neg_ddloglik, 
                              arma::mat& score_sq, arma::mat& X, arma::colvec& time, arma::colvec& delta, arma::colvec& beta) {
  int ncol = X.n_cols;
  int nrow = X.n_rows;
  
  // initialization
  arma::mat xbeta = X*beta;
  arma::mat exp_xbeta = exp(xbeta);
  double mu0_i;
  arma::colvec mu1_i(ncol);
  arma::mat mu2_i(ncol, ncol);
  arma::mat at_risk(nrow, 1);
  double time_i;
  neg_loglik = 0;
  neg_dloglik.fill(0);
  neg_ddloglik.fill(0);
  score_sq.fill(0);
  
  // calculation
  neg_loglik += arma::as_scalar(xbeta.t()*delta);
  for(int i=0; i<nrow; i++) {
    time_i = time(i);
    if(delta(i) != 0) {
      for(int k=0; k<nrow; k++) {
        at_risk(k,0) = (time(k) >= time_i ? 1 : 0);
      }
      mu0_i = arma::as_scalar(exp_xbeta.t()*at_risk)/(double)nrow;
      mu1_i = trans(X)*(at_risk%exp_xbeta)/(double)nrow;
      mu2_i = trans(X)*diagmat(vectorise(at_risk%exp_xbeta))*X/(double)nrow;
      neg_loglik += 0 - log(mu0_i);
      neg_dloglik += trans(X.row(i)) - mu1_i/mu0_i;
      neg_ddloglik += mu2_i/mu0_i - (mu1_i/mu0_i)*trans(mu1_i/mu0_i);
      score_sq += (trans(X.row(i)) - mu1_i/mu0_i)*(X.row(i) - trans(mu1_i/mu0_i));
    }
  }
  
  neg_loglik = (0.0-1.0)*neg_loglik;
  neg_dloglik = (0.0-1.0)*neg_dloglik;
} 


// calculate all stratum neg loglik functions -- internal
void all_neg_loglik_functions_cpp(double& neg_loglik, arma::colvec& neg_dloglik, arma::mat& neg_ddloglik, 
                                     arma::mat& score_sq, arma::mat& X, arma::colvec& time, arma::colvec& delta, 
                                     arma::colvec& beta, arma::uvec& strata_idx, int N) {
  int ncol = X.n_cols;
  int nrow = X.n_rows;
  
  std::vector< std::vector<int> > facility_idx(N); // index of subjects within each stratum
  for (int i = 0; i < nrow; i++)
    facility_idx[strata_idx(i) - 1].push_back(i);
  
  // initialization
  neg_loglik = 0.0;
  neg_dloglik = arma::zeros<arma::colvec>(ncol);
  neg_ddloglik = arma::zeros<arma::mat>(ncol, ncol);
  score_sq = arma::zeros<arma::mat>(ncol, ncol);
  double tmp_neg_loglik = 0;
  arma::colvec tmp_neg_dloglik = arma::zeros<arma::colvec>(ncol);
  arma::mat tmp_neg_ddloglik = arma::zeros<arma::mat>(ncol, ncol);
  arma::mat tmp_score_sq = arma::zeros<arma::mat>(ncol, ncol);

  unsigned int size_strat;
  arma::mat X_i;
  arma::colvec time_i = arma::zeros<colvec>(N);
  arma::colvec delta_i = arma::zeros<colvec>(N);

  
  for(int kk=0; kk<N; kk++) {
    size_strat = facility_idx[kk].size();
    X_i.set_size(size_strat, ncol);
    X_i = X.rows(arma::conv_to<arma::uvec>::from(facility_idx[kk]));
    time_i.resize(size_strat);
    time_i = time.elem(arma::conv_to<arma::uvec>::from(facility_idx[kk]));
    delta_i.resize(size_strat);
    delta_i = delta.elem(arma::conv_to<arma::uvec>::from(facility_idx[kk]));
    within_neg_loglik_functions_cpp(tmp_neg_loglik, tmp_neg_dloglik, tmp_neg_ddloglik, 
                                    tmp_score_sq, X_i, time_i, delta_i, beta);
    neg_loglik = neg_loglik + tmp_neg_loglik;
    neg_dloglik = neg_dloglik + tmp_neg_dloglik;
    neg_ddloglik = neg_ddloglik + tmp_neg_ddloglik;
    score_sq = score_sq + tmp_score_sq;
  }
  neg_loglik = neg_loglik/((double)(nrow));
  neg_dloglik = neg_dloglik/((double)(nrow));
  neg_ddloglik = neg_ddloglik/((double)(nrow));
  score_sq = score_sq/((double)(nrow));
}


// calculate within stratum neg loglik (no derivative) -- internal
double within_neg_loglik_cpp(arma::mat& X, arma::colvec& time, arma::colvec& delta, arma::colvec& beta) {
  int nrow = X.n_rows;
  double neg_loglik = 0.0;
    
  // initialization
  arma::mat xbeta = X*beta;
  arma::mat exp_xbeta = exp(xbeta);
  double mu0_i = 0.0;
  arma::mat at_risk(nrow, 1);
  arma::uvec at_risk1(nrow);
  double time_i = 0.0;
  
  // calculation
  neg_loglik += arma::as_scalar((trans(xbeta))*delta);
  for(int i=0; i<nrow; i++) {
    time_i = time.at(i);
    if(delta.at(i) != 0.0) {
      at_risk1 = (time >= time_i);
      at_risk.col(0) = arma::conv_to<arma::colvec>::from(at_risk1);
      
      mu0_i = arma::as_scalar((trans(exp_xbeta))*at_risk)/((double)(nrow));
      neg_loglik += 0 - log(mu0_i);
    }
  }
  
  neg_loglik = (0.0-1.0)*neg_loglik;
  return neg_loglik;
} 

// calculate all stratum neg loglik -- internal
double all_neg_loglik_cpp(arma::mat& X, arma::colvec& time, arma::colvec& delta, 
                                  arma::colvec& beta, arma::uvec& strata_idx, int N) {
  arma::colvec K = arma::zeros<colvec>(N); // number of subjects within each stratum
  int ncol = X.n_cols;
  int nrow = X.n_rows;
  unsigned int tmp_strata_idx;
  for(int i=0; i<nrow; i++) {
    tmp_strata_idx = strata_idx.at(i) - 1;
    K.at(tmp_strata_idx) ++;
  }
  
  // initialization
  double neg_loglik=0.0;
  double tmp_neg_loglik = 0.0;
  int size_strat;
  arma::uvec q;
  arma::mat X_i;
  arma::colvec time_i;
  arma::colvec delta_i;
  
  for(int kk=0; kk<N; kk++) {
    size_strat = (int)K.at(kk);
    X_i.set_size(size_strat, ncol);
    X_i = X.rows(find(strata_idx == (kk+1)));
    time_i.set_size(size_strat);
    time_i = time.elem(find(strata_idx == (kk+1)));
    delta_i.set_size(size_strat);
    delta_i = delta.elem(find(strata_idx == (kk+1)));
    tmp_neg_loglik = within_neg_loglik_cpp(X_i, time_i, delta_i, beta);
    neg_loglik += tmp_neg_loglik;
  }
  
  return neg_loglik/((double)(nrow));
}

// lasso for stratified Cox model -- internal
arma::colvec lasso_stratCox_cpp(double lambda, arma::mat& X, arma::colvec& time, arma::colvec& delta, 
                               arma::colvec& beta_init, arma::uvec& strata_idx, int N,
                               double tol=1.0e-6, int maxiter=1000) {
  int iter=1;
  double beta_diff=100000.0;  
  int ncol = X.n_cols;
  arma::colvec beta_old = beta_init;
  arma::colvec beta_new = beta_init;
  double neg_loglik; 
  arma::colvec neg_dloglik(ncol);
  arma::mat neg_ddloglik(ncol, ncol);
  arma::mat score_sq(ncol, ncol);
  
  Rcpp::Environment myEnv = Rcpp::Environment::global_env();
  Rcpp::Function my_lasso_r = myEnv["lasso_wrapper_r"];
  
  while((beta_diff>tol) & (iter<=maxiter)) {
    neg_loglik = 0;
    neg_dloglik = arma::zeros<arma::colvec>(ncol);
    neg_ddloglik = arma::zeros<arma::mat>(ncol, ncol);
    score_sq = arma::zeros<arma::mat>(ncol, ncol);
    all_neg_loglik_functions_cpp(neg_loglik, neg_dloglik, neg_ddloglik, score_sq, X, time, delta, 
                                 beta_old, strata_idx, N);
   
    beta_new = Rcpp::as<arma::colvec>(my_lasso_r(Rcpp::Named("dl", neg_dloglik),
                                                 Rcpp::Named("ddl", neg_ddloglik),
                                                 Rcpp::Named("beta_old", beta_old),
                                                 Rcpp::Named("lam", lambda)));
    
    beta_diff = sqrt(sum(pow(beta_new-beta_old, 2)));
    iter = iter + 1;
    beta_old = beta_new;
  }// end while
  
  return beta_new;
} 

// cross-validation for lasso: cannot specify lambda sequence by user
// [[Rcpp::export]]
List cv_lasso_stratCox_cpp(arma::mat& X, arma::colvec& time, arma::colvec& delta, 
                          arma::colvec& beta_init, arma::uvec& cv_idx, arma::uvec& strata_idx, int N,  
                          int nfold=5, int nlambda=50, double tol=1.0e-6, int maxiter=1000) {
    // parameters:
    // X: matrix of covariates, one row for one observation
    // time: vector of observed survival time
    // delta: vector of event indicator
    // beta_init: vector of initial value for beta
    // cv_idx: vector of cross-validation index; has the same length as time and take values in 1,...,nfold
    // strata_idx: vector of stratum index
    // N: number of strata
    // nfold: number of folds in cross-validation
    // nlambda: number of lambdas used in cross-validation (lambda sequence is automatically generated)
    // tol: tolerance level for algorithm convergence
    // maxiter: maximum number of iterations in algorithm
    
  // initialization
  int ncol = X.n_cols;
  int nrow = X.n_rows;
  arma::colvec zeros_p = arma::zeros<arma::colvec>(ncol);
  arma::vec lambda_seq(nlambda);
  double lambda_max = 0.0;
  double lam;
  double lambda_min_ratio = (nrow > ncol ? 0.01 : 0.05);
  arma::vec cv_value = arma::zeros<arma::vec>(nlambda);
  arma::colvec beta_tmp(beta_init);
  
  arma::colvec n_cv_group(nfold); // number of subjects within each cv fold
  n_cv_group = arma::zeros<arma::colvec>(nfold);
  unsigned int tmp_cv_idx;
  for(int i=0; i<nrow; i++) {
    tmp_cv_idx = cv_idx.at(i) - 1;
    n_cv_group.at(tmp_cv_idx) ++;
  }
  
  arma::uvec keep_idx = find(cv_idx != 1); // _keep_ is for training data set
  int n_keep = keep_idx.n_elem;
  arma::mat X_tmp(n_keep, ncol);
  arma::colvec time_tmp(n_keep), delta_tmp(n_keep);
  arma::uvec strata_idx_tmp(n_keep);
  
  double neg_loglik=0;
  arma::colvec neg_dloglik(ncol);
  neg_dloglik.fill(0);
  arma::mat neg_ddloglik(ncol, ncol);
  neg_ddloglik.fill(0);
  arma::mat score_sq(ncol, ncol);
  score_sq.fill(0);
  double a_d;
  
  arma::colvec beta_opt(ncol);
  double lambda_opt;
  
  // compute lambda sequence 
  all_neg_loglik_functions_cpp(neg_loglik, neg_dloglik, neg_ddloglik, score_sq, X, time, delta, 
                               zeros_p, strata_idx, N);
  for(int d=0; d<ncol; d++) {
    a_d = neg_dloglik.at(d);
    if(absf(a_d) > lambda_max) lambda_max = absf(a_d);
  }
  for(int m=0; m<nlambda; m++) {
    lambda_seq.at(m) = lambda_max*pow(lambda_min_ratio, (double)m/(double)(nlambda-1));
  }
  
  // cross-validation to get cv_value
  for(int m=0; m<nlambda; m++) {
    lam = lambda_seq.at(m);
    for(int j=1; j<=nfold; j++) {
      n_keep = nrow - n_cv_group.at(j-1);
      keep_idx.set_size(n_keep);
      keep_idx = find(cv_idx != j);
      X_tmp.set_size(n_keep, ncol);
      X_tmp = X.rows(keep_idx);
      time_tmp.set_size(n_keep);
      time_tmp = time.elem(keep_idx);
      delta_tmp.set_size(n_keep);
      delta_tmp = delta.elem(keep_idx);
      strata_idx_tmp.set_size(n_keep);
      strata_idx_tmp = strata_idx.elem(keep_idx);
      
      // calculate beta_tmp with data withheld
      beta_tmp = lasso_stratCox_cpp(lam, X_tmp, time_tmp, delta_tmp, beta_tmp, strata_idx_tmp, N, tol, maxiter);
      cv_value(m) += (0.0-1)*all_neg_loglik_cpp(X, time, delta, beta_tmp, strata_idx, N)*X.n_rows -
        (0.0-1)*all_neg_loglik_cpp(X_tmp, time_tmp, delta_tmp, beta_tmp, strata_idx_tmp, N)*X_tmp.n_rows;
    } // end for (nfold)
  } // end for (nlambda)
  
  lambda_opt = lambda_seq(0);
  double tmp = cv_value(0);
  for(int i=1; i<nlambda; i++) {
    if(cv_value(i) > tmp) {
      tmp = cv_value(i);
      lambda_opt = lambda_seq(i);
    }
  }
  
  beta_opt = lasso_stratCox_cpp(lambda_opt, X, time, delta, beta_tmp, strata_idx, N, tol, maxiter);
  neg_loglik=0;
  neg_dloglik.fill(0);
  neg_ddloglik.fill(0);
  score_sq.fill(0);
  all_neg_loglik_functions_cpp(neg_loglik, neg_dloglik, neg_ddloglik, score_sq, X, time, delta, 
                               beta_opt, strata_idx, N);
  
  return List::create(_["beta_opt"] = beta_opt, _["lambda_seq"] = lambda_seq, 
                      _["cv_value"] = cv_value, _["lambda_opt"] = lambda_opt,
                      _["neg_loglik"] = neg_loglik, _["neg_dloglik"] = neg_dloglik, _["neg_ddloglik"] = neg_ddloglik,
                      _["score_sq"] = score_sq);
}


// calculate all stratum neg loglik -- external
// [[Rcpp::export]]
double all_neg_loglik_cpp_ext(arma::mat& X, arma::colvec& time, arma::colvec& delta, 
                          arma::colvec& beta, arma::uvec& strata_idx, int N) {
  arma::colvec K(N); // number of subjects within each stratum
  K.fill(0);
  int ncol = X.n_cols;
  int nrow = X.n_rows;
  unsigned int tmp_strata_idx;
  for(int i=0; i<nrow; i++) {
    tmp_strata_idx = strata_idx(i) - 1;
    K(tmp_strata_idx) ++;
  }
  
  // initialization
  double neg_loglik=0;
  double tmp_neg_loglik = 0;
  int size_strat;
  arma::uvec q;
  arma::mat X_i;
  arma::colvec time_i;
  arma::colvec delta_i;
  
  for(int kk=0; kk<N; kk++) {
    size_strat = (int)K(kk);
    q.set_size(size_strat);
    q = find(strata_idx == (kk+1));
    X_i.set_size(size_strat, ncol);
    X_i = X.rows(q);
    time_i.set_size(size_strat);
    time_i = time.elem(q);
    delta_i.set_size(size_strat);
    delta_i = delta.elem(q);
    tmp_neg_loglik = within_neg_loglik_cpp(X_i, time_i, delta_i, beta);
    neg_loglik += tmp_neg_loglik;
  }
  
  return neg_loglik/((double)(nrow));
}

// lasso for stratified Cox model -- external
// [[Rcpp::export]]
arma::colvec lasso_stratCox_cpp_ext(double lambda, arma::mat& X, arma::colvec& time, arma::colvec& delta, 
                                arma::colvec& beta_init, arma::uvec& strata_idx, int N,
                                double tol=1.0e-6, int maxiter=1000) {
    
  int iter=1;
    double beta_diff=100000.0;
    int ncol = X.n_cols;
    arma::colvec beta_old = beta_init;
    arma::colvec beta_new = beta_init;
    double neg_loglik;
    arma::colvec neg_dloglik(ncol);
    arma::mat neg_ddloglik(ncol, ncol);
    arma::mat score_sq(ncol, ncol);
    
    Rcpp::Environment myEnv = Rcpp::Environment::global_env();
    Rcpp::Function my_lasso_r = myEnv["lasso_wrapper_r"];
    
    while((beta_diff>tol) & (iter<=maxiter)) {
      neg_loglik = 0;
      neg_dloglik = arma::zeros<arma::colvec>(ncol);
      neg_ddloglik = arma::zeros<arma::mat>(ncol, ncol);
      score_sq = arma::zeros<arma::mat>(ncol, ncol);
      all_neg_loglik_functions_cpp(neg_loglik, neg_dloglik, neg_ddloglik, score_sq, X, time, delta,
                                   beta_old, strata_idx, N);
     
      beta_new = Rcpp::as<arma::colvec>(my_lasso_r(Rcpp::Named("dl", neg_dloglik),
                                                   Rcpp::Named("ddl", neg_ddloglik),
                                                   Rcpp::Named("beta_old", beta_old),
                                                   Rcpp::Named("lam", lambda)));
      
      beta_diff = sqrt(sum(pow(beta_new-beta_old, 2)));
      iter = iter + 1;
      beta_old = beta_new;
    }// end while
    
    return beta_new;
} 


// calculate all stratum neg loglik functions -- external
// [[Rcpp::export]]
void all_neg_loglik_functions_cpp_ext(double& neg_loglik, arma::colvec& neg_dloglik, arma::mat& neg_ddloglik,
                                     arma::mat& score_sq, arma::mat& X, arma::colvec& time, arma::colvec& delta,
                                     arma::colvec& beta, arma::uvec& strata_idx, int N) {
    // parameters:
    // neg_loglik: negative log likelihood (as output)
    // neg_dloglik: first-order derivative of neg log likelihood (as output)
    // neg_ddloglik: second-order derivative of neg log likelihood (as output)
    // score_sq: matrix Sigma_hat (as output)
    // X: matrix of covariates, one row for one observation
    // time: vector of observed survival time
    // delta: vector of event indicator
    // beta: vector of beta
    // strata_idx: vector of stratum index
    // N: number of strata
    
  int ncol = X.n_cols;
  int nrow = X.n_rows;
  
  std::vector< std::vector<int> > facility_idx(N); // index of subjects within each stratum
  for (int i = 0; i < nrow; i++)
    facility_idx[strata_idx(i) - 1].push_back(i);
  
  // initialization
  neg_loglik = 0.0;
  neg_dloglik = arma::zeros<arma::colvec>(ncol);
  neg_ddloglik = arma::zeros<arma::mat>(ncol, ncol);
  score_sq = arma::zeros<arma::mat>(ncol, ncol);
  double tmp_neg_loglik = 0;
  arma::colvec tmp_neg_dloglik = arma::zeros<arma::colvec>(ncol);
  arma::mat tmp_neg_ddloglik = arma::zeros<arma::mat>(ncol, ncol);
  arma::mat tmp_score_sq = arma::zeros<arma::mat>(ncol, ncol);

  unsigned int size_strat;
  arma::mat X_i;
  arma::colvec time_i = arma::zeros<colvec>(N);
  arma::colvec delta_i = arma::zeros<colvec>(N);

  
  for(int kk=0; kk<N; kk++) {
    size_strat = facility_idx[kk].size();
    X_i.set_size(size_strat, ncol);
    X_i = X.rows(arma::conv_to<arma::uvec>::from(facility_idx[kk]));
    time_i.resize(size_strat);
    time_i = time.elem(arma::conv_to<arma::uvec>::from(facility_idx[kk]));
    delta_i.resize(size_strat);
    delta_i = delta.elem(arma::conv_to<arma::uvec>::from(facility_idx[kk]));
    within_neg_loglik_functions_cpp(tmp_neg_loglik, tmp_neg_dloglik, tmp_neg_ddloglik,
                                    tmp_score_sq, X_i, time_i, delta_i, beta);
    neg_loglik = neg_loglik + tmp_neg_loglik;
    neg_dloglik = neg_dloglik + tmp_neg_dloglik;
    neg_ddloglik = neg_ddloglik + tmp_neg_ddloglik;
    score_sq = score_sq + tmp_score_sq;
  }
  neg_loglik = neg_loglik/((double)(nrow));
  neg_dloglik = neg_dloglik/((double)(nrow));
  neg_ddloglik = neg_ddloglik/((double)(nrow));
  score_sq = score_sq/((double)(nrow));
}


