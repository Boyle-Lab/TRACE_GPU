/*
 *  File:
 *
 *
 *
 *
 *
 */
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#define error(msg) do { printf("%s\n", msg); exit(1); } while (1)

void DataProcessing(gsl_vector **slope_vector_in, gsl_vector **counts_vector_in,
                    char *peak_file, char *bam_file, char *fasta_file, char *size_file) {
  int i, j, P, T;
  double s;
  
  PyObject *pmod, *pclass, *pargs, *pinst, *pmeth, *pres;
  PyObject* next, *loessSignal, *slope_1st, *slope_2nd, *bc_loess;
    
  Py_Initialize();
  PyObject *sys = PyImport_ImportModule("sys");
  PyObject *path = PyObject_GetAttrString(sys, "path");
  PyList_Append(path, PyUnicode_FromString("."));
  
  pmod = PyImport_ImportModule("dataProcessing_"); /* fetch module */
  if (pmod == NULL)
    error("Can't load dataProcessing_");
  pclass = PyObject_GetAttrString(pmod, "Signal");   /* fetch module.class */
  Py_DECREF(pmod);
  if (pclass == NULL)
    error("Can't get dataProcessing_.Signal");
  pargs  = Py_BuildValue("(sss)", bam_file, peak_file, size_file);
  if (pargs == NULL) {
    Py_DECREF(pclass);
    error("Can't build arguments list");
  }
  pinst = PyEval_CallObject(pclass, pargs);        /* call class(  ) */
  Py_DECREF(pclass);
  Py_DECREF(pargs);
  if (pinst == NULL)
    error("Error calling dataProcessing_.Signal(  )");
  pmeth  = PyObject_GetAttrString(pinst, "load_sequence"); /* fetch load_sequence method */
  if (pmeth == NULL)
    error("Can't fetch load_sequence");
  
  pargs  = Py_BuildValue("(ss)", fasta_file, "/home/nouyang/tmp_file/test_seq.txt");       /* convert to Python */
  if (pargs == NULL) {
    Py_DECREF(pmeth);
    error("Can't build arguments list");
  }
  pres = PyEval_CallObject(pmeth, pargs);    /* call load_sequence */
  Py_DECREF(pmeth);
  Py_DECREF(pargs);
  if (pres == NULL)
    error("Error calling load_sequence");
  PyArg_Parse(PyTuple_GetItem(pres, 0), "i", &P);
  PyArg_Parse(PyTuple_GetItem(pres, 1), "i", &T);         /* convert to C */
  Py_DECREF(pres);
  pmeth  = PyObject_GetAttrString(pinst, "get_signal_all"); /* fetch get_signal method */
  if (pmeth == NULL)
    error("Can't fetch get_signal");
  Py_DECREF(pinst);
  pargs = Py_BuildValue("(fOiii)", 0.05, Py_False, 0, T, P); //b ? Py_True: Py_False
  if (pargs == NULL) {
    Py_DECREF(pmeth);
    error("Can't build arguments list");
  }
  pres = PyEval_CallObject(pmeth, pargs);    /* call get_signal */
  Py_DECREF(pargs);
  if (pres == NULL)
    error("Error calling get_signal");
  //loessSignal, slope_2nd, slope_1st, bc_loess
  loessSignal = PyTuple_GetItem(pres, 0);
  slope_2nd = PyTuple_GetItem(pres, 1);
  
  gsl_vector *counts_vector = gsl_vector_alloc(T);
  gsl_vector *slope_vector = gsl_vector_alloc(T);
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
  Py_DECREF(pres);
  Py_DECREF(pmeth);
  *counts_vector_in = counts_vector;
  *slope_vector_in = slope_vector;
}

void main(  ) {
  char *peak_file="/home/nouyang/tmp_file/test_peak.bed", *bam_file="/data/projects/FootPrinting/ENCFF577DUO.bam", *fasta_file="/data/genomes/hg38/seq/hg38.fa", *size_file="/data/genomes/hg38/hg38.chrom.sizes";

  FILE *fp1, *fp2;
  int j;
  gsl_vector *slope_vector, *counts_vector;
  DataProcessing(&slope_vector, &counts_vector, peak_file, bam_file, fasta_file, size_file);
  fp1 = fopen("/home/nouyang/tmp_file/test_count.txt", "w");
  fp2 = fopen("/home/nouyang/tmp_file/test_slope_2.txt", "w");
  printf("%d\n", (int)counts_vector->size);
  for (j = 0; j < (int)counts_vector->size; j++) {
    fprintf(fp1,"%f\n", gsl_vector_get (counts_vector, j));
    fprintf(fp2,"%f\n", gsl_vector_get (slope_vector, j));
  }
  fclose(fp1);
  fclose(fp2);
}
