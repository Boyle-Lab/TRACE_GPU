/*
 *  File: emutils.c
 *
 *  functions involved in parameter calculation including emission probability,
 *  motif scores and other model parameters.
 *
 *  The HMM structure and some codes are borrowed and modified from Kanungo's
 *  original HMM program.
 *  Tapas Kanungo, "UMDHMM: Hidden Markov Model Toolkit," in "Extended Finite State Models of Language," A. Kornai (editor), Cambridge University Press, 1999. http://www.kanungo.com/software/software.html.
 *  The matrix functions were borrowed and slightly modified from GSL library (https://www.gnu.org/software/gsl/doc/html/index.html)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>

#include "nrutil.h"
#include "hmm.h"
#include "logmath.h"
#include <omp.h>

/* The following two functions were borrowed and slightly modified from GSL library */
int gsl_ran_multivariate_gaussian_log_pdf (const gsl_vector * x,
                                           const gsl_vector * mu,
                                           const gsl_matrix * L,
                                           double * result,
                                           gsl_vector * work);

int gsl_linalg_cholesky_decomp_check (gsl_matrix * A, int *error_row,
                                      gsl_matrix * covar, HMM* phmm, int state);
static inline 
double
quiet_sqrt (double x)  
     /* avoids runtime error, for checking matrix for positive definiteness */
{
  return (x >= 0) ? sqrt(x) : GSL_NAN;
}


/*calculate motif score for each position for all motif in the model*/
void CalMotifScore_P(HMM *phmm, gsl_matrix * S, int *O1, int P, int *peakPos)
{
  int k, t, j, i, n, m, D;
  int start, end;
  double tempF, tempR, tmpL, tmpR, tmp;
  gsl_vector * tempList;
  for (m = 0; m < phmm->M; m++) {
    D = phmm->D[m]; /* length of motif */
#pragma omp parallel num_threads(THREAD_NUM) \
  private(start, end, tempF, tempR, tmpL, tmpR, tmp, t, i, j, n, tempList)
  {
#pragma omp for /* Do the calculation in parallel, peak by peak */
    for (k = 0; k < P; k++) {
      tempList = gsl_vector_alloc(D);
      start = peakPos[k];
      end = peakPos[k+1] - 1; /* The starting and ending position for peak k */
      
      /* First, calculate motif score for forward and reverse strand for the
       * first and last D bases separately */
      tmpL = -INFINITY;
      t = start - 1;
      i = 1;
      tempF = 0.0;
      tempR = 0.0;
      for (j = 1; j <= D; j++){
        if (O1[t-i+j] != -1){
          tempF += log_2(phmm->pwm[m][j-1][O1[t-i+j]]);
          tempR += log_2(phmm->pwm[m][D-j][3-O1[t-i+j]]);
          tempF -= log_2(phmm->bg[O1[t-i+j]]);
          tempR -= log_2(phmm->bg[3-O1[t-i+j]]);
        }
      }
      tmpL=MAX(tmpL, MAX(tempF, tempR));
      tmpR = -INFINITY;
      t = end - D;
      i = 1;
      tempF = 0.0;
      tempR = 0.0;
      for (j = 1; j <= D; j++){
        if (O1[t-i+j] != -1){
          tempF += log_2(phmm->pwm[m][j-1][O1[t-i+j]]);
          tempR += log_2(phmm->pwm[m][D-j][3-O1[t-i+j]]);
          tempF -= log_2(phmm->bg[O1[t-i+j]]);
          tempR -= log_2(phmm->bg[3-O1[t-i+j]]);
        }
      }
      tmpR=MAX(tmpR, MAX(tempF, tempR));
      /* Calculate motif score for forward and reverse strand for all other positions */
      t = start + D - 2;
      for (i = 1; i <= D; i++){
        tempF = 0.0;
        tempR = 0.0;
        for (j = 1; j <= D; j++){
          if (O1[t-i+j] != -1){
            tempF += log_2(phmm->pwm[m][j-1][O1[t-i+j]]);
            tempR += log_2(phmm->pwm[m][D-j][3-O1[t-i+j]]);
            tempF -= log_2(phmm->bg[O1[t-i+j]]);
            tempR -= log_2(phmm->bg[3-O1[t-i+j]]);
          }
        }
        gsl_vector_set(tempList, i-1, MAX(tempF, tempR));
      }
      
      for (t = start - 1; t < end; t++) {
        if (t < (start + D - 1)){
          gsl_matrix_set (S, m, t, tmpL);
        }
        else if (t > (end - D - 1)) {
          gsl_matrix_set (S, m, t, tmpR);
        }
        else{
          tempF = 0.0;
          tempR = 0.0;
          i = 1;
          for (j = 1; j <= D; j++){
            if (O1[t-i+j] != -1){
              tempF += log_2(phmm->pwm[m][j-1][O1[t-i+j]]);
              tempR += log_2(phmm->pwm[m][D-j][3-O1[t-i+j]]);
              tempF -= log_2(phmm->bg[O1[t-i+j]]);
              tempR -= log_2(phmm->bg[3-O1[t-i+j]]);
            }
          }
          for (n = D-1; n > 0; n--){
            gsl_vector_swap_elements(tempList, n-1, n);
          } 
          gsl_vector_set(tempList, 0, MAX(tempF, tempR));
          gsl_matrix_set (S, m, t, gsl_vector_max(tempList));
        }
      }
      gsl_vector_free(tempList);  //TODO:change in old file
    }
  }
  }
}


/* calculate motif score for each position for all motif in the model
 including maternal and paternal*/
void CalMotifScore_P_va(HMM *phmm, gsl_matrix * S, int *O1, int *O2, int P, int *peakPos)
{
  int thread_id, nloops;
  int k, t, j, i, n, m, D;
  int start, end;
  double tempF, tempR, tmpL, tmp_p, tmp_m, tmpR, tmp;
  gsl_vector * tempList;
  for (m = 0; m < phmm->M; m++) {
    D = phmm->D[m]; /* length of motif */
#pragma omp parallel num_threads(THREAD_NUM) \
  private(thread_id, start, end, tempF, tempR, tmpL, tmp_p, tmp_m, tmpR, \
          tmp, t, i, j, n, tempList)
  {
#pragma omp for /* Do the calculation in parallel, peak by peak */
    for (k = 0; k < P; k++) {
      tempList = gsl_vector_alloc(D);
      start = peakPos[k];
      end = peakPos[k+1] - 1; /* The starting and ending position for peak k */

      /* First, calculate motif score for forward and reverse strand for the
       * first and last D bases separately */
      tmpL = -INFINITY;
      tmpR = -INFINITY;
      t = start - 1;
      i = 1;
      tempF = 0.0;
      tempR = 0.0;
      for (j = 1; j <= D; j++){
        if (O1[t-i+j] != -1){
          tempF += log_2(phmm->pwm[m][j-1][O1[t-i+j]]);
          tempR += log_2(phmm->pwm[m][D-j][3-O1[t-i+j]]);
          tempF -= log_2(phmm->bg[O1[t-i+j]]);
          tempR -= log_2(phmm->bg[3-O1[t-i+j]]);
        }
      }
      tmp_m=MAX(tmpL, MAX(tempF, tempR));
      tempF = 0.0;
      tempR = 0.0;
      for (j = 1; j <= D; j++){
        if (O2[t-i+j] != -1){
          tempF += log_2(phmm->pwm[m][j-1][O2[t-i+j]]);
          tempR += log_2(phmm->pwm[m][D-j][3-O2[t-i+j]]);
          tempF -= log_2(phmm->bg[O2[t-i+j]]);
          tempR -= log_2(phmm->bg[3-O2[t-i+j]]);
        }
      }
      tmp_p = MAX(tmpL, MAX(tempF, tempR));
      tmpL = MAX(tmp_p, tmp_m);
      t = end - D;
      i = 1;
      tempF = 0.0;
      tempR = 0.0;
      for (j = 1; j <= D; j++){
        if (O1[t-i+j] != -1){
          tempF += log_2(phmm->pwm[m][j-1][O1[t-i+j]]);
          tempR += log_2(phmm->pwm[m][D-j][3-O1[t-i+j]]);
          tempF -= log_2(phmm->bg[O1[t-i+j]]);
          tempR -= log_2(phmm->bg[3-O1[t-i+j]]);
        }
      }
      tmp_m = MAX(tmpR, MAX(tempF, tempR));
      tempF = 0.0;
      tempR = 0.0;
      for (j = 1; j <= D; j++){
        if (O2[t-i+j] != -1){
          tempF += log_2(phmm->pwm[m][j-1][O2[t-i+j]]);
          tempR += log_2(phmm->pwm[m][D-j][3-O2[t-i+j]]);
          tempF -= log_2(phmm->bg[O2[t-i+j]]);
          tempR -= log_2(phmm->bg[3-O2[t-i+j]]);
        }
      }
      tmp_p = MAX(tmpR, MAX(tempF, tempR));
      tmpR = MAX(tmp_p, tmp_m);
      /* Calculate motif score for forward and reverse strand for all other positions */
      t = start + D - 2;
      for (i = 1; i <= D; i++){
        tempF = 0.0;
        tempR = 0.0;
        for (j = 1; j <= D; j++){
          if (O1[t-i+j] != -1){
            tempF += log_2(phmm->pwm[m][j-1][O1[t-i+j]]);
            tempR += log_2(phmm->pwm[m][D-j][3-O1[t-i+j]]);
            tempF -= log_2(phmm->bg[O1[t-i+j]]);
            tempR -= log_2(phmm->bg[3-O1[t-i+j]]);
          }
        }
        tmp_m=MAX(tempF, tempR);
        tempF = 0.0;
        tempR = 0.0;
        for (j = 1; j <= D; j++){
          if (O2[t-i+j] != -1){
            tempF += log_2(phmm->pwm[m][j-1][O2[t-i+j]]);
            tempR += log_2(phmm->pwm[m][D-j][3-O2[t-i+j]]);
            tempF -= log_2(phmm->bg[O2[t-i+j]]);
            tempR -= log_2(phmm->bg[3-O2[t-i+j]]);
          }
        }
        tmp_p=MAX(tempF, tempR);
        gsl_vector_set(tempList, i-1, MAX(tmp_p, tmp_m));
      }
      for (t = start - 1; t < end; t++) {
        if (t < (start + D - 1)){
          gsl_matrix_set (S, m, t, tmpL);
        }
        else if (t > (end - D - 1)) {
          gsl_matrix_set (S, m, t, tmpR);
        }
        else{
          tempF = 0.0;
          tempR = 0.0;
          i = 1;
          for (j = 1; j <= D; j++){
            if (O1[t-i+j] != -1){
              tempF += log_2(phmm->pwm[m][j-1][O1[t-i+j]]);
              tempR += log_2(phmm->pwm[m][D-j][3-O1[t-i+j]]);
              tempF -= log_2(phmm->bg[O1[t-i+j]]);
              tempR -= log_2(phmm->bg[3-O1[t-i+j]]);
            }
          }
          tmp_m=MAX(tempF, tempR);
          tempF = 0.0;
          tempR = 0.0;
          for (j = 1; j <= D; j++){
            if (O2[t-i+j] != -1){
              tempF += log_2(phmm->pwm[m][j-1][O2[t-i+j]]);
              tempR += log_2(phmm->pwm[m][D-j][3-O2[t-i+j]]);
              tempF -= log_2(phmm->bg[O2[t-i+j]]);
              tempR -= log_2(phmm->bg[3-O2[t-i+j]]);
            }
          }
          tmp_p=MAX(tempF, tempR);
          for (n = D-1; n > 0; n--){
            gsl_vector_swap_elements(tempList, n-1, n);
          }
          gsl_vector_set(tempList, 0, MAX(tmp_p, tmp_m));
          gsl_matrix_set (S, m, t, gsl_vector_max(tempList));
        }
      }
      free(tempList);
    }
  }
  }
}

/*compute emission probability treating each feature as independent */
void EmissionMatrix(HMM* phmm, gsl_matrix * obs_matrix, int P, int *peakPos,
                    gsl_matrix * emission_matrix, int T)
{
  int i, j, t, thread_id;
  double mean, sd, x_minus_mu, emission;
  gsl_matrix_set_zero(emission_matrix);
#pragma omp parallel num_threads(THREAD_NUM) \
  private(thread_id, i, j, t)
  {
#pragma omp for
    for (i = 0; i < phmm->N; i++){
      for (j = 0; j < phmm->K; j++){
        mean =  gsl_matrix_get(phmm->mean_matrix, j, i);
        sd = gsl_matrix_get(phmm->var_matrix, j, i);
        for (t = 0; t < T; t++){
          x_minus_mu = gsl_matrix_get(obs_matrix, j, t) - mean;
          emission = gsl_matrix_get(emission_matrix, i, t);
          gsl_matrix_set(emission_matrix, i, t, emission +
                         log(gsl_ran_gaussian_pdf(x_minus_mu, sd)));
        }
      }
    }
  }
}


/*compute emission probability on k-dimensional multivariate Gaussian distribution*/
void EmissionMatrix_mv(HMM* phmm, gsl_matrix * obs_matrix, int P, int *peakPos, 
                       gsl_matrix * emission_matrix, int T)
{
  int thread_id, nloops;
  int k, t, i, j, start, end, m, n;
  double tmp;
  if (phmm->K == 1){
    double pNorm, data, mean, std;
    for (k = 0; k < P; k++){
      start = peakPos[k];
      end = peakPos[k+1] - 1;
      for (t = start-1; t < end; t++) {
        data = gsl_matrix_get(obs_matrix,0, t);
        for (i = 0; i < phmm->N; i++){
          mean = gsl_matrix_get(phmm->mean_matrix, 0, i);
          std = sqrt(gsl_matrix_get(phmm->cov_matrix[i], 0, 0)) + TINY;
          pNorm = (1.0/(SQRT_TWO_PI*std)) * exp((-0.5/(std*std))*
                  ((data-mean)*(data-mean))) + TINY; //add a tiny value to avoid math problem
          gsl_matrix_set (emission_matrix, i, t, pNorm);
        }
      }
    }
  }
  /* If there are only two features, use the bivariate function */
  else if (phmm->K == 2){
    gsl_vector * mu_vector[phmm->N];
    int n = -1;
    int l;
    int error_row;
    int x, y;
#pragma omp parallel num_threads(THREAD_NUM) \
  private(thread_id, j, n, l, x, y)
    {
#pragma omp for
      for (i = 0; i < phmm->N; i++){
        mu_vector[i] = gsl_vector_alloc(phmm->K);
        gsl_matrix_get_col(mu_vector[i], phmm->mean_matrix, i);
      }
    }
    gsl_vector * data_vector;

#pragma omp parallel num_threads(THREAD_NUM) \
  private(thread_id, start, end, i, t, tmp, data_vector)
    {
#pragma omp for
      for (k = 0; k < P; k++){
        start = peakPos[k];
        end = peakPos[k+1] - 1;
        data_vector = gsl_vector_alloc(phmm->K);

        for (t = start-1; t < end; t++) {
          gsl_matrix_get_col(data_vector, obs_matrix, t);
          for (i = 0; i < phmm->N; i++){
            tmp = gsl_ran_bivariate_gaussian_pdf(gsl_vector_get(data_vector, 0) -
                    gsl_vector_get(mu_vector[i], 0),
                    gsl_vector_get(data_vector, 1) - gsl_vector_get(mu_vector[i], 1),
                    sqrt(gsl_matrix_get(phmm->cov_matrix[i], 0, 0)) + TINY,
                    sqrt(gsl_matrix_get(phmm->cov_matrix[i], 1, 1)) + TINY, //add a tiny value to avoid math problem
                    gsl_matrix_get(phmm->cov_matrix[i], 0, 1) /
                    (sqrt(gsl_matrix_get(phmm->cov_matrix[i], 0, 0)) *
                    sqrt(gsl_matrix_get(phmm->cov_matrix[i], 1, 1))));
            gsl_matrix_set (emission_matrix, i, t, tmp);
          }
        }
        gsl_vector_free(data_vector);
      }
      thread_id = omp_get_thread_num();
    }
    for (i = 0; i < phmm->N; i++){
      gsl_vector_free(mu_vector[i]);
    }
  }
  /* If there are more than two features, use the multivariate function */
  if (phmm->K > 2){
    gsl_matrix * cov_matrix_tmp[phmm->N];
    gsl_matrix * tmp_matrix;
    gsl_vector * mu_vector[phmm->N]; 
    int n = -1;
    int l;
    int error_row;
    int x, y;
#pragma omp parallel num_threads(THREAD_NUM) \
  private(thread_id, j, n, l, x, y, tmp_matrix, error_row)
  {
#pragma omp for
    for (i = 0; i < phmm->N; i++){
      cov_matrix_tmp[i] = gsl_matrix_alloc(phmm->K, phmm->K);
      gsl_matrix_memcpy(cov_matrix_tmp[i], phmm->cov_matrix[i]);
      mu_vector[i] = gsl_vector_alloc(phmm->K);
      gsl_matrix_get_col(mu_vector[i], phmm->mean_matrix, i);
      gsl_set_error_handler_off ();
      gsl_linalg_cholesky_decomp_check(cov_matrix_tmp[i], &error_row,
                                       phmm->cov_matrix[i], phmm, i);
    }
  }
  gsl_vector * data_vector;  
  gsl_vector * workspace;
#pragma omp parallel num_threads(THREAD_NUM) \
  private(thread_id, start, end, i, t,workspace, tmp, data_vector)
  {
#pragma omp for
    for (k = 0; k < P; k++){
      start = peakPos[k];
      end = peakPos[k+1] - 1;
      data_vector = gsl_vector_alloc(phmm->K);
      workspace = gsl_vector_alloc(phmm->K);
      for (t = start-1; t < end; t++) {
        gsl_matrix_get_col(data_vector, obs_matrix, t);
        for (i = 0; i < phmm->N; i++){
          gsl_ran_multivariate_gaussian_log_pdf(data_vector, mu_vector[i],
                                                cov_matrix_tmp[i], &tmp, workspace);
          gsl_matrix_set (emission_matrix, i, t, tmp);
        }
      }
      gsl_vector_free(workspace);
      gsl_vector_free(data_vector);  
    }
    thread_id = omp_get_thread_num();
  }
  
    for (i = 0; i < phmm->N; i++){
      gsl_vector_free(mu_vector[i]);
      gsl_matrix_free(cov_matrix_tmp[i]);
    }
  } 
}

/*compute emission probability on k-dimensional multivariate Gaussian distribution,
 * the difference between this function and EmissionMatrix_mv is:
 * if a hidden state has a emission probability of Inf, that state will be removed*/
void EmissionMatrix_mv_reduce(HMM* phmm, gsl_matrix * obs_matrix, int P, int *peakPos, 
                       gsl_matrix * emission_matrix, int T)
{
  int thread_id, nloops;
  int k, t, i, j, start, end, m, n;
  int TF = 0;
  int *stateList = ivector(phmm->N); //TODO: adapt to one states
  for (j = 0; j < phmm->M; j++){
    for (i = TF; i < TF + phmm->D[j]; i++) {
      stateList[i] = j;
    }
    TF += phmm->D[j];
    if (phmm->inactive == 1){
      for (i = TF; i < TF + phmm->D[j]; i++) {
        stateList[i] = j;
      }
      TF += phmm->D[j];
    }
  }
  TF -= 1;

  for (j = phmm->N - phmm->extraState; j < phmm->N; j++){
    stateList[j] = phmm->M;
  }
  if (phmm->K == 1) {
    fprintf(stderr, "Please set --model as 0 \n");
    exit (1);
  }
  else {
    gsl_matrix *cov_matrix_tmp[phmm->N];
    gsl_matrix *tmp_matrix;
    gsl_vector *mu_vector[phmm->N];
    gsl_vector *tmp_vector;
    gsl_vector *tmp_vector2;
    gsl_vector *deleted_vector[phmm->N];
    int FULL = -1;
    int EMPTY = -2;
    int n;
    int l;
    int error_row = FULL;
    int x, y;

#pragma omp parallel num_threads(THREAD_NUM) \
  private(thread_id, j, n, l, x, y, tmp_matrix, error_row, \
          tmp_vector, tmp_vector2)
    {
#pragma omp for
      for (i = 0; i < phmm->N; i++) {
        n = -1;
        deleted_vector[i] = gsl_vector_alloc(1);
        gsl_vector_set(deleted_vector[i], 0, FULL);
        cov_matrix_tmp[i] = gsl_matrix_alloc(phmm->K, phmm->K);
        gsl_matrix_memcpy(cov_matrix_tmp[i], phmm->cov_matrix[i]);
        mu_vector[i] = gsl_vector_alloc(phmm->K);
        gsl_matrix_get_col(mu_vector[i], phmm->mean_matrix, i);
        do {
          n++;
          if (n != 0) {
            /* store errow_row in deleted_vector, which will be used to
             * remove features later in pdf calculation*/
            if (n == 1) {
              gsl_vector_set(deleted_vector[i], 0, error_row);
            } else {
              tmp_vector2 = gsl_vector_alloc(n - 1);
              gsl_vector_memcpy(tmp_vector2, deleted_vector[i]);
              gsl_vector_free(deleted_vector[i]);
              deleted_vector[i] = gsl_vector_alloc(n);
              for (j = 0; j < n - 1; j++)
                gsl_vector_set(deleted_vector[i], j, gsl_vector_get( tmp_vector2, j));
              gsl_vector_free(tmp_vector2);
              gsl_vector_set(deleted_vector[i], n - 1, error_row);
            }
            gsl_matrix_free(cov_matrix_tmp[i]);
            cov_matrix_tmp[i] = gsl_matrix_alloc(phmm->K - n, phmm->K - n);
            tmp_vector = gsl_vector_alloc(phmm->K - n + 1);
            gsl_vector_memcpy(tmp_vector, mu_vector[i]);
            gsl_vector_free(mu_vector[i]);
            mu_vector[i] = gsl_vector_alloc(phmm->K - n);
            x = 0;
            for (j = 0; j < phmm->K - n + 1; j++) {
              if (j != error_row) {
                gsl_vector_set(mu_vector[i], x, gsl_vector_get(tmp_vector, j));
                y = 0;
                for (l = 0; l < phmm->K - n + 1; l++) {
                  if (l != error_row) {
                    gsl_matrix_set(cov_matrix_tmp[i], x, y,
                                   gsl_matrix_get(tmp_matrix, j, l));
                    y++;
                  }
                }
                x++;
              }
            }
            gsl_matrix_free(tmp_matrix);
            gsl_vector_free(tmp_vector);
          }
          tmp_matrix = gsl_matrix_alloc(phmm->K - n, phmm->K - n);
          gsl_matrix_memcpy(tmp_matrix, cov_matrix_tmp[i]);

          error_row = FULL;
          gsl_set_error_handler_off();
          /* Check if the matrix is positive-definite, if not,
           * store the row that caused the problem in error_row*/
          gsl_linalg_cholesky_decomp_check(cov_matrix_tmp[i], &error_row,
                                           phmm->cov_matrix[i], phmm, i);
          /* If non positive-definite problem started from first row in matrix,
           * set deleted_vector[i] to EMPTY, which means the state i will be
           * removed from model */
          if (error_row == 0) {
            gsl_vector_set(deleted_vector[i], 0, EMPTY);
            error_row = FULL;
          }
          
        } while (error_row != FULL);
        gsl_matrix_free(tmp_matrix);
      }
    
  }
  
  /*
  gsl_vector *kept_vector[phmm->N];
  for (i = 0; i < phmm->N; i++){
    x = 0;
    j = 0;
    if (gsl_vector_get(deleted_vector[i], 0) == FULL) {
      kept_vector[i] = gsl_vector_alloc(phmm->K);
      for (m = 0; m < phmm->K; m++){
         gsl_vector_set(kept_vector[i], m, m);
      }
    }
    else{
      kept_vector[i] = gsl_vector_alloc(phmm->K-deleted_vector[i]->size);
      for (m = 0; m < phmm->K; m++) {
        if ((m-j) != gsl_vector_get(deleted_vector[i], j)){
          gsl_vector_set(kept_vector[i], x, m);
          x++;
        }
        else {
          j++;
        }
      }
    }
  }
  for (i = 0; i < phmm->N; i++){
      printf("i: %d size: %d\t: ", i, kept_vector[i]->size);
      for (j = 0; j < kept_vector[i]->size; j++){
          printf("j: %d\t", gsl_vector_get(kept_vector[i], j));
      }
  }
  */
      
  gsl_vector * data_vector;
  gsl_vector * deleted_data_vector;  
  gsl_vector * workspace;
  double tmp;
  double mean, sd, data_mean, emission;

#pragma omp parallel num_threads(THREAD_NUM) \
  private(thread_id, nloops, start, end, i, j, t, m, x, workspace, tmp, \
          data_vector, tmp_vector, deleted_data_vector, mean, sd, data_mean, emission)
  {
    nloops = 0;
#pragma omp for   
    for (k = 0; k < P; k++){
      ++nloops;
      start = peakPos[k];
      end = peakPos[k+1] - 1;
      data_vector = gsl_vector_alloc(phmm->K);
      workspace = gsl_vector_alloc(phmm->K);
      for (t = start-1; t < end; t++) {
        gsl_matrix_get_col(data_vector, obs_matrix, t);
        for (i = 0; i < phmm->N; i++){
          
          /* If no data needs to be deleted, calculate log pdf directly */
          //else
          if (gsl_vector_get(deleted_vector[i], 0) == FULL) {
            gsl_ran_multivariate_gaussian_log_pdf(data_vector, mu_vector[i],
                    cov_matrix_tmp[i], &tmp, workspace);
          }
          /* If all data needs to be deleted, the log emission probability
           * will be -INFINITY */
          else if (gsl_vector_get(deleted_vector[i], 0) == EMPTY) {
            tmp = -INFINITY;
          }
          /* If some of the data need to be deleted, remove those ones first,
           * then calculate log pdf */
          else {
            emission = 0.0;
            deleted_data_vector = gsl_vector_alloc(phmm->K-deleted_vector[i]->size);
              x = 0;
              j = 0;
              for (m = 0; m < data_vector->size; m++) {
                if ((m-j) != gsl_vector_get(deleted_vector[i], j)){
                  gsl_vector_set(deleted_data_vector, x, gsl_vector_get(data_vector, m));
                  x++;
                }
                else {
                  mean =  gsl_matrix_get(phmm->mean_matrix, m-j, i);
                  sd = sqrt(gsl_matrix_get(phmm->cov_matrix[i], m-j, m-j));
                  if (sd == 0.0) {emission += -INFINITY;}
                  else {
                  data_mean = gsl_matrix_get(obs_matrix, m-j, t) - mean;    
                  emission += log(gsl_ran_gaussian_pdf(data_mean, sd));  
                  }
                  j++;
                }
              }
            gsl_ran_multivariate_gaussian_log_pdf(deleted_data_vector, mu_vector[i],
                                                  cov_matrix_tmp[i], &tmp, workspace);
            gsl_vector_free(deleted_data_vector);
            tmp += emission;
          }
          gsl_matrix_set (emission_matrix, i, t, tmp);
        }
      }
      gsl_vector_free(workspace);
      gsl_vector_free(data_vector);  
    }
    thread_id = omp_get_thread_num();
  }
    for (i = 0; i < phmm->N; i++){
      gsl_vector_free(mu_vector[i]);
      gsl_matrix_free(cov_matrix_tmp[i]);
      gsl_vector_free(deleted_vector[i]);
    }
  } 
}

/* Calculate the covariance matrix based on known standard deviation and correlation */
void covarMatrix_GSL(HMM *phmm, int state, gsl_matrix * cov_matrix)
{
  int i, j, n, k;
  double corr;
  k = 0;
    for (i = 0; i < phmm->K; i++) {
      for (j = 0; j < phmm->K; j++) {
        if (i == j) {
          gsl_matrix_set (cov_matrix, i, j, 
                          (gsl_matrix_get(phmm->var_matrix, i, state) * 
                          gsl_matrix_get(phmm->var_matrix, i, state)));
        }
        else if (i < j) {
          corr = phmm->rho[k][state];
          gsl_matrix_set (cov_matrix, i, j,
                          (gsl_matrix_get(phmm->var_matrix, i, state) *
                          gsl_matrix_get(phmm->var_matrix, j, state) * corr));
          k++;
        }
        else {
          gsl_matrix_set(cov_matrix, i, j, gsl_matrix_get(cov_matrix, j, i));
        }
      }
    }
}


int gsl_ran_multivariate_gaussian_log_pdf (const gsl_vector * x,
                                           const gsl_vector * mu,
                                           const gsl_matrix * L, double * result,
                                           gsl_vector * work)
{
  const size_t M = L->size1;
  const size_t N = L->size2;
  if (M != N)
    {
      GSL_ERROR("requires square matrix", GSL_ENOTSQR);
    }
  else if (mu->size != M)
    {
      GSL_ERROR("incompatible dimension of mean vector with variance-covariance matrix", GSL_EBADLEN);
    }
  else if (x->size != M)
    {
      GSL_ERROR("incompatible dimension of quantile vector", GSL_EBADLEN);
    }
  else if (work->size != M)
    {
      GSL_ERROR("incompatible dimension of work vector", GSL_EBADLEN);
    }
  else
    {
      size_t i;
      double quadForm;        /* (x - mu)' Sigma^{-1} (x - mu) */
      double logSqrtDetSigma; /* log [ sqrt(|Sigma|) ] */

      /* compute: work = x - mu */
      for (i = 0; i < M; ++i)
        {
          double xi = gsl_vector_get(x, i);
          double mui = gsl_vector_get(mu, i);
          gsl_vector_set(work, i, xi - mui);
        }

      /* compute: work = L^{-1} * (x - mu) */
      gsl_blas_dtrsv(CblasLower, CblasNoTrans, CblasNonUnit, L, work);

      /* compute: quadForm = (x - mu)' Sigma^{-1} (x - mu) */
      gsl_blas_ddot(work, work, &quadForm);

      /* compute: log [ sqrt(|Sigma|) ] = sum_i log L_{ii} */
      logSqrtDetSigma = 0.0;
      for (i = 0; i < M; ++i)
      {
        double Lii = gsl_matrix_get(L, i, i);
        logSqrtDetSigma += log(Lii);
      }
      *result = -0.5*quadForm - logSqrtDetSigma - 0.5*M*log(2.0*M_PI);

      return GSL_SUCCESS;
    }
}


/*check if matrix is positive-definite, modified from GSL library*/
int gsl_linalg_cholesky_decomp_check (gsl_matrix * A, int *error_row,
                                      gsl_matrix * covar, HMM* phmm, int state)
{
  const size_t M = A->size1;
  const size_t N = A->size2;
  int m, n;
  FILE	* tmp_fp;
  if (M != N) {
    GSL_ERROR("cholesky decomposition requires square matrix", GSL_ENOTSQR);
  }
  else
    {
      size_t i,j,k;
      int status = 0;

      /* Do the first 2 rows explicitly.  It is simple, and faster.  And
       * one can return if the matrix has only 1 or 2 rows.  
       */

      double A_00 = gsl_matrix_get (A, 0, 0);
      if (A_00 != A_00)
       {
          status = GSL_EDOM ;
          *error_row = 0;
          //printf("nan %d\t", state);
          return GSL_SUCCESS;
       }
      double L_00 = quiet_sqrt(A_00);
      
      if (A_00 <= 0)
        {
          status = GSL_EDOM ;
          *error_row = 0;
          return GSL_SUCCESS;
        }

      gsl_matrix_set (A, 0, 0, L_00);
  
      if (M > 1)
        {
          double A_10 = gsl_matrix_get (A, 1, 0);
          double A_11 = gsl_matrix_get (A, 1, 1);
          
          double L_10 = A_10 / L_00;
          double diag = A_11 - L_10 * L_10;
          double L_11 = quiet_sqrt(diag);
          
          if (diag <= 0)
            {
              //printf("%d %d\t", state,1);

              status = GSL_EDOM;
              *error_row = 1;
              return GSL_SUCCESS;
            }

          gsl_matrix_set (A, 1, 0, L_10);        
          gsl_matrix_set (A, 1, 1, L_11);
        }
      
      for (k = 2; k < M; k++)
        {
          double A_kk = gsl_matrix_get (A, k, k);
          
          for (i = 0; i < k; i++)
            {
              double sum = 0;

              double A_ki = gsl_matrix_get (A, k, i);
              double A_ii = gsl_matrix_get (A, i, i);

              gsl_vector_view ci = gsl_matrix_row (A, i);
              gsl_vector_view ck = gsl_matrix_row (A, k);

              if (i > 0) {
                gsl_vector_view di = gsl_vector_subvector(&ci.vector, 0, i);
                gsl_vector_view dk = gsl_vector_subvector(&ck.vector, 0, i);
                
                gsl_blas_ddot (&di.vector, &dk.vector, &sum);
              }

              A_ki = (A_ki - sum) / A_ii;
              gsl_matrix_set (A, k, i, A_ki);
            } 

          {
            gsl_vector_view ck = gsl_matrix_row (A, k);
            gsl_vector_view dk = gsl_vector_subvector (&ck.vector, 0, k);
            
            double sum = gsl_blas_dnrm2 (&dk.vector);
            double diag = A_kk - sum * sum;

            double L_kk = quiet_sqrt(diag);
            
            if (diag <= 0)
              {
                status = GSL_EDOM;
                *error_row = k;
                return GSL_SUCCESS;
              }
            
            gsl_matrix_set (A, k, k, L_kk);
          }
        }

      /* Now copy the transposed lower triangle to the upper triangle,
       * the diagonal is common.  
       */
      
      for (i = 1; i < M; i++)
        {
          for (j = 0; j < i; j++)
            {
              double A_ij = gsl_matrix_get (A, i, j);
              gsl_matrix_set (A, j, i, A_ij);
            }
        } 
      
      if (status == GSL_EDOM)
        {
          GSL_ERROR ("matrix must be positive definite", GSL_EDOM);
          
          
        }
      
      return GSL_SUCCESS;
    }
}
