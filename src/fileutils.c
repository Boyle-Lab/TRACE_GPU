/*
 *  File: fileutils.c
 *
 *  functions involved in readign and writing files
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

/* Read the sequence file to get length T */
void ReadLength(FILE *fp, int *pT)
{
  if(fscanf(fp, "T= %d\n", pT) == EOF){
    fprintf(stderr, "Error: model file error \n");
    exit (1);
  }
}

/* Read the sequence file to get length T, GC content, sequence O,
 * number of peaks P and peak start position peakPos */
void ReadSequence(FILE *fp, int T, double *GC, int *O, int *pP, int **peakPos)
{
  int unused_num, *peaks;
  int i;
  unused_num = fscanf(fp, "GC: ");
  for (i = 0; i < 4; i++) {
    if(fscanf(fp, "%lf\t", &GC[i]) == EOF){
      fprintf(stderr, "Error: sequence file error \n");
      exit (1);
    }
  }
  unused_num = fscanf(fp,"\n");
  for (i = 0; i < T; i++) {
    if(fscanf(fp,"%d", &O[i]) == EOF){
      fprintf(stderr, "Error: sequence file error \n");
      exit (1);
    }
  }
  unused_num = fscanf(fp,"\n");
  unused_num = fscanf(fp, "P= %d\n", pP);
  peaks = ivector(*pP + 1);
  for (i=0; i < *pP + 1; i++){
    if(fscanf(fp,"%d", &peaks[i]) == EOF){
      fprintf(stderr, "Error: sequence file error \n");
      exit (1);
    }
  }
  *peakPos = peaks;
}

/* Read the sequence file with maternal and paternal to get length T, GC content,
* sequence O, number of peaks P and peak start position peakPos */
void ReadSequence_va(FILE *fp, int *pT, double *GC, int **pO1, int **pO2, int *pP, int **pPeakPos)
{
  int unused_num, *peaks, *O1, *O2;
  int i;
  if(fscanf(fp, "T= %d\n", pT) == EOF){
    fprintf(stderr, "Error: model file error \n");
    exit (1);
  }
  unused_num = fscanf(fp, "GC: ");
  for (i = 0; i < 4; i++) {
    if(fscanf(fp, "%lf\t", &GC[i]) == EOF){
      fprintf(stderr, "Error: sequence file error \n");
      exit (1);
    }
  }
  O1 = ivector(*pT);
  O2 = ivector(*pT);
  unused_num = fscanf(fp,"\n");
  for (i = 0; i < *pT; i++) {
    if(fscanf(fp,"%d | %d", &O1[i], &O2[i]) == EOF){
      fprintf(stderr, "Error: sequence file error \n");
      exit (1);
    }
  }
  unused_num = fscanf(fp,"\n");
  unused_num = fscanf(fp, "P= %d\n", pP);
  peaks = ivector(*pP + 1);
  for (i=0; i < *pP + 1; i++){
    if(fscanf(fp,"%d", &peaks[i]) == EOF){
      fprintf(stderr, "Error: sequence file error \n");
      exit (1);
    }
  }
  *pO1 = O1;
  *pO2 = O2;
  *pPeakPos = peaks;
}

/* Read count or slope file to store numbers in data_vector,
 * with a optional adjust, which is used to change the scale of original numbers*/
void ReadTagFile(FILE * fp, int T, gsl_vector * data_vector, double adjust)
{
  double tmp;
  int i;
  for (i=0; i < T; i++) {
    if(fscanf(fp,"%lf\t", &tmp) == EOF){
      fprintf(stderr, "Error: input file error \n");
      exit (1);
    }
    gsl_vector_set(data_vector, i, tmp*adjust);
  }
}

void ReadPeakFile(FILE * fp, int P, char ** chr, int * posStart, int * posEnd)
{
  int i;
  for (i=0; i < P; i++) {
    chr[i] = malloc(BUFSIZE);
    if(fscanf(fp,"%s\t%d\t%d", chr[i], &posStart[i], &posEnd[i]) == EOF){
      fprintf(stderr, "Error: input file error \n");
      exit (1);
    }
  }
}

void ReadMotifFile(char * motiffile, int P, int L, int ** motifStart,
                   int ** motifEnd, int ** motifLength, int ** motifIndex)
{
  int i, j, start, end, start_, end_, length, start_old;
  char chr[BUFSIZE];
  char chr_[BUFSIZE];
  checkFile(motiffile, "r");
  FILE *fp = fopen(motiffile, "r");
  L = 0;
  while(fscanf(fp, "%s\t%d\t%d\t%s\t%d\t%d\t%d", chr, &start, &end, chr_, &start_, &end_, &length) != EOF){
    L++;
  }
  fclose(fp);
  int * index = ivector(P+1);
  int * TFstart = ivector(L);
  int * TFend = ivector(L);
  int * TFlength = ivector(L);
  //TFchr = (char **)malloc(L * sizeof(char *));
  fp = fopen(motiffile, "r");
  start_old = -1;
  j = 0;
  for (i=0; i < L; i++) {
    //TFchr[i] = malloc(BUFSIZE);
    if (fscanf(fp, "%s\t%d\t%d\t%s\t%d\t%d\t%d", chr, &start, &end, chr_, &TFstart[i], &TFend[i], &TFlength[i]) != EOF){
      if (start_old != start){
        index[j]=i;
        j++;
        start_old = start;
      }
    }
  }
  index[j]=i;
  fclose(fp);
  *motifStart = TFstart;
  *motifEnd = TFend;
  *motifLength = TFlength;
  *motifIndex = index;
}

void ReadTFlistFile(FILE *fp, int L, char **array)
{
  int i;
  for (i=0; i < L; i++) {
    array[i] = malloc(BUFSIZE);
    if(fgets(array[i], BUFSIZE, fp) == NULL){
      fprintf(stderr, "Error: input file error \n");
      exit (1);
    }
    array[i][strlen(array[i])-1] = '\0';
  }
}


void PrintSequenceProb(FILE *fp, int T, int *O, double *vprob, double *g,
                       double **posterior, int indexTF)
{
  int i;
  fprintf(fp,"%d\t%lf\t%lf\t%lf", O[0], vprob[0], g[0], posterior[0][indexTF]);
  for (i=1; i < T; i++) {
    fprintf(fp,"\n%d\t%lf\t%lf\t%lf", O[i], vprob[i], g[i], posterior[i][indexTF]);
  }
}

/* check if the provided file path is valid */
void checkFile(char *filename, char *mode)
{
  FILE *fp = fopen(filename, mode);
  if (fp == NULL) {
    fprintf(stderr, "Error: File %s not valid \n", filename);
    exit(1);
  }
  fclose(fp);
}

/* check if the provided file path is valid */
int checkFile_noExit(char *filename, char *mode)
{
  FILE *fp = fopen(filename, mode);
  if (fp == NULL) {
    //fprintf(stderr, "Error: File %s not valid \n", filename);
    return 0;
  }
  fclose(fp);
  return 1;
}
