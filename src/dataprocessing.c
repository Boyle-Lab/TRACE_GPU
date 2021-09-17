/*
 *  File: dataprocessing.c
 *
 *
 *
 *
 *
 */
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <sys/types.h>
#include <unistd.h>
#include "hmm.h"
#include "nrutil.h"
#include <dlfcn.h>
#include <sys/time.h>

#define error(msg) do { printf("%s\n", msg); exit(1); } while (1)

/* compute counts, slope and genotypes for main TRACE program,
   in variant-aware cases, O1 O2 will be maternal and paternal sequence,
   otherwise O1 will be sequence and O2 will be empty*/
void DataProcessing(char *peak_file, char *bam_file, char *fasta_file, char *size_file,
                    char *vcf_file, char *prefix, gsl_vector **slope_vector_in, gsl_vector **counts_vector_in,
                    double *GC, int *pT,int *pP, int **peakPos, int **O1,  int **O2) {
  int i, j, *pos, *seq, *seq1, *seq2;
  double s, gc;
  PyObject *pmod, *pclass, *pargs, *pinst, *pmeth, *pres;
  PyObject *pos_obj, *seq_obj, *next, *loessSignal, *slope_1st, *slope_2nd, *bc_loess;
    
  Py_Initialize();
  PyObject *sys = PyImport_ImportModule("sys");
  PyObject *path = PyObject_GetAttrString(sys, "path");
  PyList_Append(path, PyUnicode_FromString("."));
  
  //Py_Finalize();
  pmod = PyImport_ImportModule("dataProcessing_c"); /* fetch module */
  if (pmod == NULL){
    PyErr_Print();
    error("Can't load dataProcessing_c");
  }
  pclass = PyObject_GetAttrString(pmod, "Signal");   /* fetch module.class */
  Py_DECREF(pmod);
  if (pclass == NULL){
    PyErr_Print();
    error("Can't get dataProcessing_c.Signal");
  }
  pargs  = Py_BuildValue("(ssss)", bam_file, peak_file, size_file, fasta_file);
  if (pargs == NULL) {
    PyErr_Print();
    Py_DECREF(pclass);
    error("Can't build arguments list");
  }
  pinst = PyEval_CallObject(pclass, pargs);        /* call class(  ) */
  Py_DECREF(pclass);
  Py_DECREF(pargs);
  if (pinst == NULL){
    PyErr_Print();
    error("Error calling dataProcessing_c.Signal(  )");
  }
  if (vcf_file[0] == '\0'){
    printf("processing signal ... ");
    pmeth  = PyObject_GetAttrString(pinst, "get_signal_all"); /* fetch get_signal_all method */
    if (pmeth == NULL){
      PyErr_Print();
      error("Can't fetch get_signal_all");
    }
    pargs = Py_BuildValue("(fOii)", 0.05, Py_False, 0, THREAD_NUM); //b ? Py_True: Py_False
    if (pargs == NULL) {
      PyErr_Print();
      Py_DECREF(pmeth);
      error("Can't build arguments list");
    }
    pres = PyEval_CallObject(pmeth, pargs);    /* call get_signal_all */
    Py_DECREF(pmeth);
    Py_DECREF(pargs);
    if (pres == NULL){
      PyErr_Print();
      error("Error calling get_signal_all");
    }
  }
  else{
    printf("processing signal with variants ... ");
    pmeth  = PyObject_GetAttrString(pinst, "get_signal_SNP"); /* fetch get_signal_all method */
    if (pmeth == NULL){
      PyErr_Print();
      error("Can't fetch get_signal_SNP");
    }
    pargs = Py_BuildValue("(sfOiii)", vcf_file, 0.05, Py_False, 0, *pT, THREAD_NUM); //b ? Py_True: Py_False
    if (pargs == NULL) {
      PyErr_Print();
      Py_DECREF(pmeth);
      error("Can't build arguments list");
    }
    pres = PyEval_CallObject(pmeth, pargs);    /* call get_signal_all */
    Py_DECREF(pmeth);
    Py_DECREF(pargs);
    if (pres == NULL){
      PyErr_Print();
      error("Error calling get_signal_SNP");
    }
  }
  Py_DECREF(pinst);
  if(!(*pT)){
    PyArg_Parse(PyTuple_GetItem(pres, 0), "i", pP);
    PyArg_Parse(PyTuple_GetItem(pres, 1), "i", pT);
    PyArg_Parse(PyTuple_GetItem(pres, 2), "d", &gc);
    GC[0] = GC[3] = (1.0 - gc) / 2.0;
    GC[1] = GC[2] = gc / 2.0;
  
    pos_obj = PyTuple_GetItem(pres, 3);
    pos = ivector(*pP + 1);
    if (PyList_Check(pos_obj)) {
      for (j = 0; j < (int) PyList_Size(pos_obj); j++) {
        next = PyList_GetItem(pos_obj, j);
        PyArg_Parse(next, "i", &i);
        pos[j] = i;
      }
    }
    *peakPos = pos;
    
    if (vcf_file[0] != '\0'){
      seq_obj = PyTuple_GetItem(pres, 6);
      seq1 = ivector(*pT);
      if (PyList_Check(seq_obj)) {
        for (j = 0; j < (int) PyList_Size(seq_obj); j++) {
          next = PyList_GetItem(seq_obj, j);
          PyArg_Parse(next, "i", &i);
          seq1[j] = i;
        }
      }
      *O1 = seq1;
      seq_obj = PyTuple_GetItem(pres, 7);
      seq2 = ivector(*pT);
      if (PyList_Check(seq_obj)) {
        for (j = 0; j < (int) PyList_Size(seq_obj); j++) {
          next = PyList_GetItem(seq_obj, j);
          PyArg_Parse(next, "i", &i);
          seq2[j] = i;
        }
      }
      *O2 = seq2;
   }
    else{
      seq_obj = PyTuple_GetItem(pres, 6);
      seq = ivector(*pT);
      if (PyList_Check(seq_obj)) {
        for (j = 0; j < (int) PyList_Size(seq_obj); j++) {
          next = PyList_GetItem(seq_obj, j);
          PyArg_Parse(next, "i", &i);
          seq[j] = i;
        }
      }
      *O1 = seq;
    }
  }
  loessSignal = PyTuple_GetItem(pres, 4);
  slope_2nd = PyTuple_GetItem(pres, 5);
  gsl_vector *counts_vector = gsl_vector_alloc(*pT);
  gsl_vector *slope_vector = gsl_vector_alloc(*pT);
  if (PyList_Check(loessSignal) && PyList_Check(slope_2nd)) {
    for (j = 0; j < (int) PyList_Size(loessSignal); j++) {
      next = PyList_GetItem(loessSignal, j);
      PyArg_Parse(next, "d", &s);                    /* convert to C */
      gsl_vector_set(counts_vector, j ,s);
      next = PyList_GetItem(slope_2nd, j);
      PyArg_Parse(next, "d", &s);
      gsl_vector_set(slope_vector, j ,s);
    }
  }
  *counts_vector_in = counts_vector;
  *slope_vector_in = slope_vector;
  Py_DECREF(pres);

    if (prefix[0] != '\0'){
        printf("... writing files ... ");

        char slopefile[BUFSIZE], countfile[BUFSIZE], seqfile[BUFSIZE];
        slopefile[0] = '\0';
        countfile[0] = '\0';
        seqfile[0] = '\0';
        strcat(slopefile, prefix);
        strcat(countfile, prefix);
        strcat(seqfile, prefix);
        if (vcf_file[0] != '\0'){
           strcat(slopefile, "_slope_va_2.txt");
           strcat(countfile, "_count_va.txt");
           strcat(seqfile, "_seq_va.txt");
           checkFile(slopefile, "w");
           checkFile(countfile, "w");
           checkFile(seqfile, "w");
           FILE *fp1, *fp2, *fp3;
           fp1 = fopen(seqfile, "w");
           fp2 = fopen(slopefile, "w");
           fp3 = fopen(countfile, "w");
           fprintf(fp1, "T=%d GC: %lf\t%lf\t%lf\t%lf\n", *pT, GC[0],GC[1],GC[2],GC[3]);
           for (j = 0; j < *pT; j++) {
             fprintf(fp1,"%d|%d\n", (*O1)[j],(*O2)[j]);
             fprintf(fp2,"%lf\n", gsl_vector_get (slope_vector, j));
             fprintf(fp3,"%lf\n", gsl_vector_get (counts_vector, j));
           }
           fprintf(fp1,"P= %d\n", *pP);
           for (j = 0; j < *pP + 1; j++) {
             fprintf(fp1,"%d\n", (*peakPos)[j]);
           }
           fclose(fp1);
           fclose(fp2);
           fclose(fp3);
        }
        else{
          strcat(slopefile, "_slope_2.txt");
          strcat(countfile, "_count.txt");
          strcat(seqfile, "_seq.txt");
          checkFile(slopefile, "w");
          checkFile(countfile, "w");
          checkFile(seqfile, "w");

          FILE *fp1, *fp2, *fp3;
          fp1 = fopen(seqfile, "w");
          fp2 = fopen(slopefile, "w");
          fp3 = fopen(countfile, "w");
          fprintf(fp1, "T=%d GC: %lf\t%lf\t%lf\t%lf\n", *pT, GC[0],GC[1],GC[2],GC[3]);
          for (j = 0; j < *pT; j++) {
            fprintf(fp1,"%d\n", (*O1)[j]);
            fprintf(fp2,"%lf\n", gsl_vector_get (slope_vector, j));
            fprintf(fp3,"%lf\n", gsl_vector_get (counts_vector, j));
          }
          fprintf(fp1,"P= %d\n", *pP);
          for (j = 0; j < *pP + 1; j++) {
            fprintf(fp1,"%d\n", (*peakPos)[j]);
          }
          fclose(fp1);
          fclose(fp2);
          fclose(fp3);
        }
        
    }
  printf("... Done\n");
}

/*
void main(  ) {
  char *peak_file="/home/nouyang/tmp_file/test_peak.bed", *bam_file="/data/projects/FootPrinting/ENCFF577DUO.bam", *fasta_file="/data/genomes/hg38/seq/hg38.fa", *size_file="/data/genomes/hg38/hg38.chrom.sizes";

  FILE *fp1, *fp2, *fp3;
  int j, *peakPos, *O1, T, P;
  double *GC;
  gsl_vector *slope_vector, *counts_vector;
  GC = dvector(4);
  DataProcessing(peak_file, bam_file, fasta_file, size_file, &slope_vector, &counts_vector, GC, &T, &O1, &P, &peakPos);
    //DataProcessing(char *peak_file, char *bam_file, char *fasta_file, char *size_file, gsl_vector **slope_vector_in, gsl_vector **counts_vector_in, double *pGC, int *pT, int *O, int *pP, int **peakPos) {
  fp1 = fopen("/home/nouyang/tmp_file/test_count.txt", "w");
  fp2 = fopen("/home/nouyang/tmp_file/test_slope_2.txt", "w");
  printf("%d %d %d %d %d %f\n", (int)counts_vector->size, T, P, peakPos[0], O1[0], GC[0]);
  for (j = 0; j < (int)counts_vector->size; j++) {
    fprintf(fp1,"%f\n", gsl_vector_get (counts_vector, j));
    fprintf(fp2,"%f\n", gsl_vector_get (slope_vector, j));
  }
  fclose(fp1);
  fclose(fp2);
}

*/
