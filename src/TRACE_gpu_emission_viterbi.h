#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <omp.h> 
#include "hmm.h"
__host__ void EmissionMatrix_viterbi_gpu(FILE *fpOut,char *TFname, HMM* phmm, gsl_matrix * obs_matrix, int P, int *peakPos, 
                       gsl_matrix ** cov_matrix_tmp, gsl_vector ** deleted_vector, gsl_matrix * pwm_matrix, int T,
                       char **chr,int *peakStart, int *peakEnd);