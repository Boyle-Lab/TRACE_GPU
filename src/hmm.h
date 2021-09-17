#ifndef HMM_H
#define HMM_H
#include "logmath.h"
#include "nrutil.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

typedef struct {
/* make sure to put TF motif at the first states */
  int N; /* number of states;  Q={1,2,...,N} */
  int M; /* number of TFs*/
  int K; /*number of values in each state. (slop and pwm scrore for each TF)*/
  int inactive;
  int model; /* 0: independent, 1: multivariance */
  int lPeak; /* length of states in peak surrounding FPs */
  int extraState; /* number of states that are not FPs */
  double **A; /* A[1..N][1..N]. a[i][j] is the transition prob
			   of going from state i at time t to state j
			   at time t+1 */
  //int P; /*number of the peaks*/
  //int peakPos; /*position for each peak */
  double **mu; /* mu[i] is mean of slop in state i */
  double **sigma; /* sigma[i] is variance in state i */
  double **rho; /*correlation*/
  double ***pwm; /* pwm[i][n][x] is probability of each base in each state from PWM for TF i*/
  int *D; /* D[i] is number of positions in motif */
  double *bg; /*background percentage for ACGT */
  double	*pi;	/* pi[1..N] pi[i] is the initial state distribution. */
  double *thresholds; /* thresholds of pwm scores */
  gsl_matrix ** cov_matrix; /* N x N Covariance matrix */
  gsl_matrix * mean_matrix; /* K x N mean matrix, each row is a vector of means
                             * for each state for each feature*/
  gsl_matrix * var_matrix; /* K x N standard deviation matrix, each row is a vector
                            * of std for each state for each feature*/
  gsl_matrix * log_A_matrix; /* N x N transition prob matrix */
} HMM;

/* BaumWelch.c */
void BaumWelch(HMM *phmm, int T, gsl_matrix * obs_matrix, int *pniter, int P, 
               int *peakPos, double *logprobf, double **alpha, double **beta, 
               double **gamma, gsl_matrix * emission_matrix);
void UpdateVariance_2(HMM *phmm, gsl_matrix * obs_matrix, 
                      gsl_vector * prob_sum, gsl_matrix *prob_matrix, 
                      int T, int TF);
void UpdateCovariance_2(HMM *phmm, gsl_matrix * obs_matrix, 
                        gsl_vector * prob_sum, gsl_matrix *prob_matrix, 
                        int T, int TF);
void ComputeGamma(HMM *phmm, int T, gsl_matrix * alpha_matrix, 
                  gsl_matrix * beta_matrix, gsl_matrix * gamma_matrix);
void ComputeXi_sum(HMM* phmm, gsl_matrix * alpha_matrix, 
                   gsl_matrix * beta_matrix, gsl_vector * xi_sum_vector,
                   gsl_matrix * emission_matrix, int T);
void ComputeXi_sum_P(HMM* phmm, double **alpha, double **beta, double *xi_sum, 
                   gsl_matrix * emission_matrix, int T);
void ComputeGamma_P(HMM *phmm, int T, double **alpha, double **beta, 
                    double **gamma);

/*fwd_bwd.c*/
void Forward_P(HMM *phmm, int T, double **alpha, double *pprob, int P, 
               int *peakPos, gsl_matrix * emission_matrix);
void Backward_P(HMM *phmm, int T, double **beta, int P, int *peakPos, 
                gsl_matrix * emission_matrix);
                
/* emutils.c */
/*void CalMotifScore_P(HMM *phmm, gsl_matrix * S, int *O1, int P, int *peakPos);
void CalMotifScore_P_va(HMM *phmm, gsl_matrix * S, int *O1, int *O2, int P, int *peakPos);
void EmissionMatrix(HMM* phmm, gsl_matrix * obs_matrix, int P, int *peakPos, 
                    gsl_matrix * emission_matrix, int T);
void EmissionMatrix_mv(HMM* phmm, gsl_matrix * obs_matrix, int P, int *peakPos,
                       gsl_matrix * emission_matrix, int T);
void EmissionMatrix_mv_reduce(HMM* phmm, gsl_matrix * obs_matrix, int P, 
                              int *peakPos, gsl_matrix * emission_matrix, 
                              int T);
void covarMatrix_GSL(HMM *phmm, int state, gsl_matrix * cov_matrix);*/

//if calculate emission matrix on gpu (still need to do cov=LL' on cpu, but run multivariate gaussian modeling on gpu) -> added function EmissionMatrix_mv_reduce_pre_gpu
/* emutils_2.c */
void CalMotifScore_P(HMM *phmm, gsl_matrix * S, int *O1, int P, int *peakPos);
void CalMotifScore_P_va(HMM *phmm, gsl_matrix * S, int *O1, int *O2, int P, int *peakPos);
void EmissionMatrix_mv_reduce_pre_gpu(HMM *phmm, gsl_matrix * obs_matrix, int P, int *peakPos, int T,
                                      gsl_matrix **cov_matrix_tmp,gsl_vector **deleted_vector);
void covarMatrix_GSL(HMM *phmm, int state, gsl_matrix * cov_matrix);
void EmissionMatrix(HMM* phmm, gsl_matrix * obs_matrix, int P, int *peakPos, 
                    gsl_matrix * emission_matrix, int T);
void EmissionMatrix_mv(HMM* phmm, gsl_matrix * obs_matrix, int P, int *peakPos,
                       gsl_matrix * emission_matrix, int T);
void EmissionMatrix_mv_reduce(HMM* phmm, gsl_matrix * obs_matrix, int P, 
                              int *peakPos, gsl_matrix * emission_matrix, 
                              int T);
void CalMotifScore_P_single_TF(HMM *phmm, gsl_vector * S, int *O1, int P, int *peakPos, 
                             int TF_index);
void CalMotifScore_P_partial(HMM *phmm, gsl_matrix * S, int *O1, int P_i, int *peakPos, 
                        int *peakPos_i, int *posIndex_i, int TF_start, int TF_end);
/* fileutils.c */
void ReadLength(FILE *fp, int *pT);
void ReadSequence(FILE *fp, int T, double *GC, int *O, int *pP,
                  int **peakPos);
void ReadSequence_va(FILE *fp, int *pT, double *GC, int **pO1, int **pO2,
                     int *pP, int **pPeakPos);
void ReadTagFile(FILE *fp, int T, gsl_vector * data_vector, double adjust);
void ReadPeakFile(FILE *fp, int P, char **chr, int *posStart, int *posEnd);
void ReadMotifFile(char * motiffile, int P, int L, int ** motifStart,
                   int ** motifEnd, int ** motifLength, int ** motifIndex);
void ReadTFlistFile(FILE *fp, int L, char **Array);
void PrintSequenceProb(FILE *fp, int T, int *O, double *vprob, double *g, 
                       double **posterior, int indexTF);
void checkFile(char *filename, char *mode);
int checkFile_noExit(char *filename, char *mode);

/* hmmutils.c */
void ReadM(FILE *fp, HMM *phmm);
void allocHMM(HMM *phmm);
void ReadInitHMM(FILE *fp, HMM *phmm);
void ReadHMM(FILE *fp, HMM *phmm);
void PrintHMM(FILE *fp, HMM *phmm);
void copyHMM(HMM *phmmIn, HMM *phmmOut);
void getRho(HMM *phmm);
void FreeHMM(HMM *phmm);
                  
/* viterbi.c */
void Viterbi(HMM *phmm, int T, double *g, double **alpha, double **beta,
             double	**gamma, double  *logprobf, double **delta, 
             int **psi, int *q, double *vprob, double *pprob, 
             double **posterior, int P, int *peakPos,
             gsl_matrix *emission_matrix, gsl_matrix *pwm_matrix);
void getPosterior_motif(FILE *fpOut, int T, int *peakPos, int P, double **posterior,
                        HMM *phmm, char **chrm, int *peakStart, int *peakEnd,
                        int *motifStart, int *motifEnd, int *motifLength, int *motifIndex);
void getPosterior_all(FILE *fpOut, int T, int *q, int *peakPos, int P, double **posterior,
                      HMM *phmm, char **chrm, int *peakStart, int *peakEnd);
/* dataprocessing.c */
void DataProcessing(char *peak_file, char *bam_file, char *fasta_file, char *size_file,
                    char *vcf_file, char *prefix, gsl_vector **slope_vector_in, gsl_vector **counts_vector_in,
                    double *GC, int *pT,int *pP, int **peakPos, int **O1,  int **O2);
int hmmgetseed(void);
void hmmsetseed(int seed); 
double hmmgetrand(void); 

#define MAX(x,y)        ((x) > (y) ? (x) : (y))
#define MIN(x,y)        ((x) < (y) ? (x) : (y))

/* some globle values */
#define SQRT_TWO_PI 2.5066282746310002
#define D_LOG2E 1.44269504088896340736
#define TINY 1.0e-15
#define BUFSIZE 150
int MAXITERATION;
int THREAD_NUM;

u_int64_t GetTimeStamp();


#endif /* HMM_H */
