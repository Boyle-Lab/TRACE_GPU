/*
 *  File: viterbi.c
 *
 *  viterbi step and hidden states labeling.
 *
 *  The HMM structure and some codes are borrowed and modified from Kanungo's
 *  original HMM program.
 *  Tapas Kanungo, "UMDHMM: Hidden Markov Model Toolkit," in "Extended Finite State Models of Language," A. Kornai (editor), Cambridge University Press, 1999. http://www.kanungo.com/software/software.html.
 *
 */

#include <math.h>
#include "hmm.h"
#include "nrutil.h"
#include "logmath.h"
#include <omp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#define VITHUGE  100000000000.0
#define VITTINY  -100000000000.0
#define PROBTINY  -10000000000.0

/* Get the hidden states sequence using Viterbi algorithm */
void Viterbi(HMM *phmm, int T, double *g, double **alpha, double **beta,
             double	**gamma, double  *logprobf, double **delta, 
             int **psi, int *q, double *vprob, double *pprob, 
             double **posterior, int P, int *peakPos,
             gsl_matrix *emission_matrix, gsl_matrix *pwm_matrix)
{
  int thread_id, nloops;
  int  i, j, k, m, x, y, TF;   /* state indices */
  int  t = 0;      /* time index */
  int  maxvalind;
  int  l = 0;
  int *TFlist, *lengthList;
  int *motifList, *indexList;
  int nonInf;
  double  maxval, val;
  double plogprobinit, plogprobfinal;
  double temp;
    
  TF = 0;
  TFlist = ivector(phmm->M * (phmm->inactive+1));
  indexList = ivector(phmm->M * (phmm->inactive+1));
  for (j = 0; j < phmm->M; j++){
    TF += phmm->D[j];
    TFlist[j * (phmm->inactive+1)] = TF - 1;
    indexList[j * (phmm->inactive+1)] = j;
    if (phmm->inactive == 1){
      TF += phmm->D[j];
      TFlist[j * (phmm->inactive+1) + 1] = TF - 1;
      indexList[j * (phmm->inactive+1) + 1] = j;
    }
  }
  lengthList = ivector(phmm->M * (phmm->inactive+1));
  if (phmm->inactive == 1){
    for (j = 0; j < phmm->M; j++){
      lengthList[j*2] = phmm->D[j];
      lengthList[j*2+1] = phmm->D[j];
    }
  }
  else lengthList = phmm->D;
  
  motifList = ivector(phmm->N);
  TF = 0;
  for (j = 0; j < phmm->M; j++) {
    for (i = TF; i < TF + phmm->D[j]; i++) {
      motifList[i] = (phmm->inactive+1)*j;
    }
    TF += phmm->D[j];
    if (phmm->inactive == 1) {
      for (i = TF; i < TF + phmm->D[j]; i++) {
        motifList[i] = (phmm->inactive+1)*j+1;
      }
      TF += phmm->D[j];
    }
  }
  for (i = TF; i < TF + phmm->extraState; i++) {
    motifList[i] = i;
  }
#pragma omp parallel num_threads(THREAD_NUM) \
  private(thread_id, nloops, val, maxval, maxvalind, t, j, i, temp, x, y, nonInf)
  {
    nloops = 0;
#pragma omp for
  for (k = 0; k < P; k++){
    ++nloops;
    /* 1. Initialization  */     
    for (i = 0; i < phmm->N; i++) {
      delta[peakPos[k]-1][i] = log(phmm->pi[i]) + 
                               gsl_matrix_get(emission_matrix, i, peakPos[k]-1);
      psi[peakPos[k]-1][i] = phmm->N;
      posterior[peakPos[k]-1][i] = alpha[i][peakPos[k]-1] + 
                                   beta[i][peakPos[k]-1] - logprobf[k];
    }
    for (x = 0; x < phmm->M * (phmm->inactive+1); x++){
        posterior[peakPos[k]-1][TFlist[x]] = PROBTINY;
    }  
    /* 2. Recursion */
    for (t = peakPos[k]; t < peakPos[k+1] - 1; t++) {
      for (j = 0; j < phmm->N; j++) {
        maxval = VITTINY;
        maxvalind = phmm->N-1;
        for (i = 0; i < phmm->N; i++) {
          val = delta[t-1][i] + gsl_matrix_get(phmm->log_A_matrix, i, j);
          if (val > maxval) {
            maxval = val;
            maxvalind = i;
          }
        }
        temp = gsl_matrix_get(emission_matrix, j, t);
        if (j < TF){
          if (gsl_matrix_get(pwm_matrix,indexList[motifList[j]],t) < (phmm->thresholds[indexList[motifList[j]]]-0.5)){
            temp = VITTINY;
              //(phmm->thresholds[(int)floor((double)motifList[j]/(phmm->inactive+1))]
          }
        }
        delta[t][j] = maxval + temp;
        psi[t][j] = maxvalind;
        
        posterior[t][j] = alpha[j][t] + beta[j][t] - logprobf[k];
        /* treat t as the last position of a motif */
        for (x = 0; x < phmm->M * (phmm->inactive+1); x++){
          if (j == TFlist[x]){
            if (t <= (peakPos[k] + lengthList[x])){
              posterior[t][j] = log(0.0);
            }
            /* check pwm threshold*/
            //else if (gsl_matrix_get(pwm_matrix,indexList[motifList[j]],t) < (phmm->thresholds[indexList[motifList[j]]]-0.1)){
              //for (y = t - lengthList[x] + 1; y <= t; y++) posterior[y][j] = log(0.0);
            //}
            else {
              temp=0.0;
              nonInf = 0;
              for (i = 0; i < lengthList[x]; i++) {
                if ((alpha[j-i][t-i] + beta[j-i][t-i] - logprobf[k]) != -INFINITY){
                  temp += (alpha[j-i][t-i] + beta[j-i][t-i] - logprobf[k]); 
                  nonInf += 1;  
                }
              }
              if (nonInf == 0) temp = PROBTINY;
              for (y = t - lengthList[x] + 1; y <= t; y++) posterior[y][j] = temp;
            }
              
            break;
          }
        }
      }
    }
 
    /* 3. Termination */
    pprob[k] = -VITHUGE;
    q[peakPos[k+1]-2] = 1;
    for (i = 0; i < phmm->N; i++) {
      if (delta[peakPos[k+1]-2][i] > pprob[k]) {
        pprob[k] = delta[peakPos[k+1]-2][i];
        q[peakPos[k+1]-2] = i;
      }
    }
    g[peakPos[k+1]-2] = gamma[q[peakPos[k+1]-2]][peakPos[k+1]-2];
    vprob[peakPos[k+1]-2] = pprob[k]; 
    /* 4. Path (state sequence) backtracking */
  	for (t = peakPos[k+1] - 3; t >= peakPos[k] - 1; t--){
	  	q[t] = psi[t+1][q[t+1]];
      vprob[t] = delta[t+1][q[t+1]];
      g[t] = gamma[q[t]][t];
    }
  }
  }
  free_ivector(TFlist, phmm->M * (phmm->inactive+1));
  free_ivector(indexList, phmm->M * (phmm->inactive+1));
  free_ivector(motifList, phmm->N);
  free_ivector(lengthList, phmm->M * (phmm->inactive+1));
}


/* Motif-centric approach. This function will calculate marginal posterior probabilities
 * for all motif sites provided  */
void getPosterior_motif(FILE *fpOut, int T, int *peakPos, int P, double **posterior,
                        HMM *phmm, char **chrm, int *peakStart, int *peakEnd,
                        int *motifStart, int *motifEnd, int *motifLength, int *motifIndex)
{
  int start, end, TFstart, TFend, length, init, t, i, j, m, n;
  int old_start = -1;
  int TF, maxTF, indexTF_end, state, pos, motif;
  double prob;
  char *chr;
  //char chr[BUFSIZE];
  //char chr_[BUFSIZE];
  int *lengthList = ivector(phmm->M * (phmm->inactive+1));
  int *motifList = ivector(phmm->N);
  indexTF_end = -1;
  if (phmm->inactive == 1){
    for (j = 0; j < phmm->M; j++){
      lengthList[j*2] = phmm->D[j];
      lengthList[j*2+1] = phmm->D[j];
      indexTF_end += phmm->D[j] * 2;
    }
  }
  else {
    lengthList = phmm->D;
    for (j = 0; j < phmm->M; j++){
      indexTF_end += phmm->D[j];
    }
  }

  TF = 0;
  for (j = 0; j < phmm->M; j++) {
    for (i = TF; i < TF + phmm->D[j]; i++) {
      motifList[i] = (phmm->inactive+1)*j;
    }
    TF += phmm->D[j];
    if (phmm->inactive == 1) {
      for (i = TF; i < TF + phmm->D[j]; i++) {
        motifList[i] = (phmm->inactive+1)*j+1;
      }
      TF += phmm->D[j];
    }
  }
  for (i = TF; i < TF + phmm->extraState; i++) {
    motifList[i] = i;
  }
  TF -= 1;
  //i = -1;
  fprintf(stdout,"scanning motif sites and calculating posterior probabilities \n");
  //while (fscanf(fpIn, "%s\t%d\t%d\t%s\t%d\t%d\t%d", chr, &start, &end, chr_, &TFstart, &TFend, &length) != EOF) {
  for (i = 0; i < P; i++){
    start = peakStart[i];
    end = peakEnd[i];
    chr = (char *) malloc(BUFSIZE);
    chr[0] = '\0';
    strcat(chr, chrm[i]);
    for (n = motifIndex[i]; n < motifIndex[i+1]; n++){
      TFstart = motifStart[n];
      TFend = motifEnd[n];
      length = motifLength[n];
    /* Skip repetitive regions */
   // if (start != old_start) {
    //  i++;
   // }
        //printf("P:%s %d %d %d %d %d %d %d\n", chr, start, end, TFstart, TFend, peakPos[i] - 1+TFstart - start + length - 1,peakPos[P], T);
        //fflush(stdout);
      if (TFstart != -1){
        if (length >= (TFend-TFstart) && TFend <= end) {
          init = peakPos[i] - 1;
        
          /* Print posterior probabilities of being every active and inactive
           * motif states and generic footprints states for all motif sites */
          fprintf(fpOut,"%s\t%d\t%d", chr, TFstart, TFend);
          TF = -1;
          for (j = 0; j < phmm->M * (phmm->inactive+1); j++){
            TF += lengthList[j];
            //prob = -INFINITY;
            //for (m = 0; m <= length+MIN(lengthList[j]-1,end-TFend); m ++) prob = MAX(prob, posterior[(init + TFstart - start + m - 1)][TF]);
            m = MIN(lengthList[j],end-TFstart); //TODO:change in original file
            prob = posterior[(init + TFstart - start + m - 1)][TF];
            fprintf(fpOut,"\t%e", prob);
          }
          /* posterior probability of being a generic footprint */
          prob = 0.0;
          for (m = 0; m < length; m ++) prob += posterior[(init + TFstart - start + m)][phmm->N-4];
          fprintf(fpOut,"\t%e", prob/length);
          prob = 0.0;
          for (m = 0; m < length; m ++) prob += posterior[(init + TFstart - start + m)][phmm->N-3];
          fprintf(fpOut,"\t%e", prob/length);
          fprintf(fpOut,"\n");
        }
      }
    }
    
    free(chr);
    //old_start = start;
  }
  free_ivector(motifList, phmm->N);
  free_ivector(lengthList, phmm->M * (phmm->inactive+1));
}

/* Get all binding sites predictions from viterbi */
void getPosterior_all(FILE *fpOut, int T, int *q, int *peakPos, int P, double **posterior,
                      HMM *phmm, char **chrm, int *peakStart, int *peakEnd)
{
  int start, end, TFstart, TFend, length, init, t, i, j;
  int stateStart, stateEnd, dataStart, dataEnd, stateLength;
  int old_start = -1;
  int TF, maxTF;
  double prob;
  char *chr;
  fprintf(stdout,"scanning peak file and calculating posterior probabilities for all positions\n");
  int * stateList = ivector(phmm->N);
  TF = 0;
  for (j = 0; j < phmm->M; j++){
    for (i = TF; i < TF + phmm->D[j]; i++) {
      stateList[i] = j * (phmm->inactive + 1);
    }
    TF += phmm->D[j];

    if (phmm->inactive == 1){
      for (i = TF; i < TF + phmm->D[j]; i++) {
        stateList[i] = j * (phmm->inactive + 1) + 1;
      }
      TF += phmm->D[j];
    }
  }
  TF -= 1;
  for (j = phmm->N - phmm->extraState; j < phmm->N; j++){
    stateList[j] = j;
  }
  i = -1;
  //while (fscanf(fpIn, "%s\t%d\t%d", chr, &start, &end) != EOF) {
  for (j = 0; j < P; j ++){
    start = peakStart[j];
    end = peakEnd[j];
    chr = (char *) malloc(BUFSIZE);
    chr[0] = '\0';
    strcat(chr, chrm[j]);
    if (start != old_start){
      i++;
      dataStart = peakPos[i] - 1;
      dataEnd = peakPos[i+1] - 2;
      stateStart = 0;
      stateEnd = stateStart;
      t = dataStart;
      if (posterior[t][q[t]] != -INFINITY) {
        prob = posterior[t][q[t]];
        stateLength = 1;
      }
      else {
        prob = 0;
        stateLength = 0;
      }
      for (t = dataStart+1; t <= dataEnd; t++){
        if (stateList[q[t]] == stateList[q[t-1]]){
          stateEnd ++;
          if (posterior[t][q[t]] != -INFINITY){
            prob += posterior[t][q[t]];
            stateLength ++;
          }
          maxTF = stateList[q[t]] + 1;
          if (t == dataEnd)
            fprintf(fpOut,"%s\t%d\t%d\t%d\t%e\t%e\n", chr, start + stateStart,
                    start + stateEnd + 1, maxTF, prob/stateLength, prob/stateLength);
        }
        else {
          maxTF = stateList[q[t-1]] + 1;
          if (maxTF <= phmm->M * (phmm->inactive+1) && phmm->M != 0) {
            if (maxTF % 2 == 0) {
              fprintf(fpOut, "%s\t%d\t%d\t%d\t%e\t%e\n", chr, start + stateStart,
                      start + stateEnd + 1, maxTF, posterior[t - 1][q[t - 1]],
                      posterior[t - 1][q[t - 1] - phmm->D[maxTF / 2 - 1]]);
            }
            else {
              fprintf(fpOut, "%s\t%d\t%d\t%d\t%e\t%e\n", chr, start + stateStart,
                      start + stateEnd + 1, maxTF, posterior[t - 1][q[t - 1]],
                      posterior[t - 1][q[t - 1] + phmm->D[(maxTF - 1) / 2]]);
            }
          }
          else fprintf(fpOut,"%s\t%d\t%d\t%d\t%e\t%e\n", chr, start + stateStart,
                       start + stateEnd + 1, maxTF, prob/stateLength, prob/stateLength);
          stateEnd ++;
          stateStart = stateEnd;
          if (posterior[t][q[t]] != -INFINITY) {
            prob = posterior[t][q[t]];
            stateLength = 1;
          }
          else {
            prob = 0;
            stateLength = 0;
          }
          if (t == dataEnd)
            fprintf(fpOut,"%s\t%d\t%d\t%d\t%e\t%e\n", chr, start + stateStart,
                    start + stateEnd + 1, stateList[q[t]] + 1, prob/stateLength,
                    prob/stateLength);
        }
      }
    }
    old_start = start;
    free(chr);
  }
  free_ivector(stateList, phmm->N);
}


