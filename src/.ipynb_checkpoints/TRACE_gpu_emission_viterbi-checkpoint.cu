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
u_int64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * (u_int64_t) 1000000 + tv.tv_usec;
}
u_int64_t start_time;
//reference to c linkage
extern "C" { 
    #include "TRACE_gpu_emission_viterbi.h"
}
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define THREADS_PER_BLOCK 256
#define VITHUGE  100000000000.0 
#define VITTINY  -1000000000000.0
#define M_PI 3.14159265358979323846
__constant__ double DVITHUGE = 100000000000.0; //used in gpu kernel
__constant__ double DVITTINY = -100000000000.0;

///////////functions to linearize matrix///////////
__host__ __device__
void matrix_set(double *arr, int i, int j, int width, double value) {
  arr[width * i + j] = value;
}

__host__ __device__
void matrix_set_int(int *arr, int i, int j, int width, int value) {
  arr[width * i + j] = value;
}

__host__ __device__
double matrix_get(double *arr, int i, int j, int width) {
  return arr[width * i + j];
}

__host__ __device__
int matrix_get_int(int *arr, int i, int j, int width) {
  return arr[width * i + j];
}
void gsl_matrix_to_arr(int height, int width, gsl_matrix *source, double *arr) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      matrix_set(arr, i, j, width, gsl_matrix_get(source, i, j));
    }
  }
}
//3D matrices
__host__ __device__
void matrix_3D_set(double *arr, int x, int y, int z, int width, int height, double value) {
  arr[x + y * width + z * height * width] = value;
}

__host__ __device__
void matrix_3D_set_int(int *arr, int x, int y, int z, int width, int height, int value) {
  arr[x + y * width + z * height * width] = value;
}

__host__ __device__
double matrix_3D_get(double *arr, int x, int y, int z, int width, int height) { // x:width;y:height;z:depth
  return arr[x + y * width + z * height * width];
}
__host__ __device__
double matrix_3D_get_int(int *arr, int x, int y, int z, int width, int height) {
  return arr[x + y * width + z * height * width];
}

///////Log math functions///////
__host__ __device__
double _logadd(const double p, const double q) {
  return p + log1p(exp(q - p));
}
__host__ __device__
double logadd(const double p, const double q) {
  return (p > q) ? _logadd(p, q) : _logadd(q, p);
}
__host__ __device__
double logCheckAdd(const double p, const double q) {
  if (p != -INFINITY && q != -INFINITY){
    return logadd(p,q);
  }
  else if (p == -INFINITY && q == -INFINITY){
    return -INFINITY;
  }
  else if (p == -INFINITY){
    return q;
  }
  else if (q == -INFINITY){
    return p;
  }
}
///////Function for output posterior of final states for each peak/////////////
///Note posterior array should set T as width; also output TF name in last column////
__host__ void getPosterior_all(FILE *fpOut, char *TFname, int T, int *q, int P,
                               int *peakPos, double *posterior, int n_states, int extraState, int *D,
                               char **chrm, int *peakStart, int *peakEnd) {
  //parameter for final model
  int M = 10;
  int inactive = 1;

  int start,end;
  int t, i, j;
  int stateStart, stateEnd, dataStart, dataEnd, stateLength;
  int old_start = -1;
  int TF, maxTF;
  double prob;
  char *chr;

  int *stateList = (int *) calloc(n_states, sizeof(int));
  TF = 0;
  for (j = 0; j < M; j++){
    for (i = TF; i < TF + D[j]; i++) {
      stateList[i] = j * (inactive + 1);
    }
    TF += D[j];

    if (inactive == 1){
      for (i = TF; i < TF + D[j]; i++) {
        stateList[i] = j * (inactive + 1) + 1;
      }
      TF += D[j];
    }
  }
  TF -= 1;
  for (j = n_states - extraState; j < n_states; j++){
    stateList[j] = j;
  }
  i = -1;
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
      if (matrix_get(posterior,q[t],t,T) != -INFINITY) {
        prob = matrix_get(posterior,q[t],t,T);
        //printf("%d\t%d\t%f\n",t,q[t],prob);
        stateLength = 1;
      }
      else {
        prob = 0;
        stateLength = 0;
      }
      for (t = dataStart+1; t <= dataEnd; t++){
        if (stateList[q[t]] == stateList[q[t-1]]){
          stateEnd ++;
          if (matrix_get(posterior,q[t],t,T) != -INFINITY){
            prob += matrix_get(posterior,q[t],t,T);
            stateLength ++;
          }
          maxTF = stateList[q[t]] + 1;
          if (t == dataEnd)
            fprintf(fpOut,"%s\t%d\t%d\t%d\t%e\t%e\t%s\n", chr, start + stateStart,
                    start + stateEnd + 1, maxTF, prob/stateLength, prob/stateLength,TFname);
        }
        else {
          maxTF = stateList[q[t-1]] + 1;
          if (maxTF <= M * (inactive+1) && M != 0) {
            if (maxTF % 2 == 0) {
              fprintf(fpOut, "%s\t%d\t%d\t%d\t%e\t%e\t%s\n", chr, start + stateStart,
                      start + stateEnd + 1, maxTF, matrix_get(posterior,q[t - 1],t-1,T),
                      matrix_get(posterior,q[t - 1] - D[maxTF / 2 - 1],t-1,T),TFname);
            }
            else {
              fprintf(fpOut, "%s\t%d\t%d\t%d\t%e\t%e\t%s\n", chr, start + stateStart,
                      start + stateEnd + 1, maxTF, matrix_get(posterior,q[t - 1],t-1,T),
                      matrix_get(posterior,q[t - 1] + D[(maxTF - 1) / 2],t-1,T),TFname);
            }
          }
          else fprintf(fpOut,"%s\t%d\t%d\t%d\t%e\t%e\t%s\n", chr, start + stateStart,
                       start + stateEnd + 1, maxTF, prob/stateLength, prob/stateLength,TFname);
          stateEnd ++;
          stateStart = stateEnd;
          if (matrix_get(posterior,q[t],t,T) != -INFINITY) {
            prob = matrix_get(posterior,q[t],t,T);
            stateLength = 1;
          }
          else {
            prob = 0;
            stateLength = 0;
          }
          if (t == dataEnd)
            fprintf(fpOut,"%s\t%d\t%d\t%d\t%e\t%e\t%s\n", chr, start + stateStart,
                    start + stateEnd + 1, stateList[q[t]] + 1, prob/stateLength,
                    prob/stateLength,TFname);
        }
      }
    }
    old_start = start;
    free(chr);
  }
  free((int*) (stateList));
}

///////////////////////Start of gpu kernels///////////////////////
__global__ void get_emission_kernel(double *mean_matrix,double *cov_matrix,double *obs_matrix, int n_states, int T,
                                    double *L,int *error_rows,double *emission_matrix) { //need to do cholesky decomp on cpu: cov=LL'
	int const K = 12; // const parameter for final model
	/////To make block threads ~sync -> each block take a state -> same on l, each thread take several positions////
	int const state = blockIdx.x; //state ID
  	int const tidx = threadIdx.x;
  //assign the positions each thread calculates
  int chunk_size = ceil(__int2double_rn(T) / blockDim.x);
  int remain = T - (chunk_size-1) * blockDim.x;
  int T_start, T_end; //depend on tidx
  if (tidx < remain) {//first set of threads (total: remain) calculate chunk_size positions
      T_start = tidx * chunk_size;
      T_end = T_start + chunk_size;
  } else {//threads after remain caclulate chunk_size-1 positions
      T_start = remain * chunk_size + (tidx - remain) * (chunk_size-1);
      T_end  = T_start + chunk_size-1;
  }
	for (int t = T_start; t < T_end; t++){
    	int l = 0;
    	double d_vector[12]; //predefine size as #of features K
        double d_vector_2[12];
        double sum; //for tmp values
    	double data_mean;
    	double quadForm = 0; //(x-mu)'L{-1}'L{-1}(x-mu)
    	double logSqrtDetSigma = 0.0; // log [ sqrt(|Sigma|) ]
    	double prob = 0.0; //log prob
    	if(matrix_get_int(error_rows,0,state,n_states) == -2) { //EMPTY: all rows deleted -> log emission prob = -INFINITY
    		prob = -INFINITY;
    	} else {
    		//First get positive-definite part size l from error_rows matrix
        for (int k = 0; k < K; k++) {
    			data_mean = matrix_get(obs_matrix,k,t,T) - matrix_get(mean_matrix,k,state,n_states);
    			if (matrix_get_int(error_rows,k,state,n_states) == 1){
    				d_vector[l] = data_mean;
    				l ++;
    			} else{
    				//model as single variate gaussian dist for error rows, add to prob individually
            if(matrix_3D_get(cov_matrix,k,k,state,K,K) != 0.0) {
              prob += -0.5*data_mean*data_mean/matrix_3D_get(cov_matrix,k,k,state,K,K) - 0.5*log(matrix_3D_get(cov_matrix,k,k,state,K,K)) - 0.5*log(2.0*M_PI); 
            }
    			}
    		}
    		// already have sigma = LL' where L is a lower triangular matrix then (x - mu)' Sigma^{-1} (x - mu) = (x-mu)'(LL'){-1}(x-mu) = (x-mu)'L{-1}'L{-1}(x-mu) = B'B where B = L{-1}(x-mu)
	    	//need to solve x_2 = L{-1}x -> L*x_2 = x -> can simply use backward/forward substitution for upper/lower triangular matrix L to solve the linear system (i.e. trsv function)
	    	//Hard to use cuBLAS library in cuda 10.1 -> just write function explicitly here			
      /////Forward substitution to compute L{-1}(x-mu)//////
			d_vector_2[0] = d_vector[0]/matrix_3D_get(L,0,0,state,K,K);
			if (l >= 2) {
				for (int i =1; i < l; i++){
					sum = 0;
					for (int j = 0; j < i; j++) {
						sum += matrix_3D_get(L,j,i,state,K,K) * d_vector_2[j];
					}
                    d_vector_2[i] = (d_vector[i]-sum)/matrix_3D_get(L,i,i,state,K,K);
				}	
			}
			
			for (int i = 0; i < l; i++){
				/////Compute quadForm = (x - mu)' Sigma^{-1} (x - mu)/////
				quadForm += d_vector_2[i] * d_vector_2[i];
				/////Compute logSqrtDetSigma=log[sqrt(|Sigma|)]=log[sqrt|L*L'|]=log[|L|]=sum_i log[L_ii]/////
				logSqrtDetSigma += log(matrix_3D_get(L,i,i,state,K,K));
			}
			//model as multivariate gaussian dist (when l=1 equivalent with the single variate gaussian dist above) 
			prob += -0.5*quadForm - logSqrtDetSigma - 0.5*l*log(2.0*M_PI); //M_PI
        } 
		matrix_set(emission_matrix,state,t,T,prob);
    }
}
__global__ void fwd_kernel(double *alpha, double *logprobf, int T, int n_states, int *peakPos, double *pi,
             double *emission_matrix, double *log_A_matrix) {
	extern __shared__ double shared_bank[]; //store two columns (length = n_states) for probs. on time t-1 & t
    double *d_prev_probs = shared_bank;
    double *d_current_probs = d_prev_probs + n_states;    
    int const k = blockIdx.x; //i.e. peakID, each block takes a peak
    int const tidx = threadIdx.x;
    int const peak_start = peakPos[k] -1;
    int const peak_end = peakPos[k+1] -1;
    //assign the states each thread calculates
    int chunk_size = ceil(__int2double_rn(n_states) / blockDim.x);
    int remain = n_states - (chunk_size-1) * blockDim.x;
    int state_start, state_end; //depend on tidx
    if (tidx < remain) {//first set of threads (total: remain) calculate chunk_size states
        state_start = tidx * chunk_size;
        state_end = state_start + chunk_size;
    } else {//threads after remain caclulate chunk_size-1 states
        state_start = remain * chunk_size + (tidx - remain) * (chunk_size-1);
        state_end  = state_start + chunk_size-1;
    }
    double sum;
    //1. Initialization
    for (int j = state_start; j < state_end; j++) {
        if (pi[j] == 0.0){
            d_prev_probs[j] = -INFINITY;
            
        }
        else{
            d_prev_probs[j] = log(pi[j]) + matrix_get(emission_matrix,j,peak_start,T);
        }
        matrix_set(alpha,j,peak_start,T,d_prev_probs[j]);
    }
    __syncthreads();
    //2. Induction
    for (int t=peak_start+1; t <peak_end; t++) {  //for each position on peak calculate sum prob;
        for (int j = state_start; j < state_end; j++) { //state index this thread calculates;
            sum = -INFINITY;
            for (int i = 0; i < n_states; i++) { //transition from i to j
                sum = logCheckAdd(sum, d_prev_probs[i] + matrix_get(log_A_matrix,i,j,n_states));
            }
            d_current_probs[j] = sum + matrix_get(emission_matrix,j,t,T);
            matrix_set(alpha,j,t,T,d_current_probs[j]);
        }
        __syncthreads();
        //update probs on t-1 with t
        for (int j = state_start; j < state_end; j++) {
            d_prev_probs[j] = d_current_probs[j];
        }
        __syncthreads(); 
    }
    //3. Termination
    if (tidx == 0){//Thread 0 calculate log prob. of whole sequence
        double temp;
        temp = -INFINITY; //prob. of whole sequence for calculating posterior prob. later in viterbi_kernel
        for (int i = 0; i < n_states; i++) {
          temp = logCheckAdd(temp, d_prev_probs[i]); //d_prev_probs[i] = alpha[i][peak_end-1]
        }
        logprobf[k] = temp;
    }
    __syncthreads();
}

__global__ void bwd_kernel(double *beta, int T, int n_states, int *peakPos,
             double *emission_matrix, double *log_A_matrix) { 
	extern __shared__ double shared_bank[]; 
	//store two columns (length = n_states) for probs. on time t+1 & t
    double *d_prev_probs = shared_bank;
    double *d_current_probs = d_prev_probs + n_states;
    int const k = blockIdx.x; //i.e. peakID, each block takes a peak
    int const tidx = threadIdx.x;
    int const peak_start = peakPos[k] -1;
    int const peak_end = peakPos[k+1] -1;
    //assign the states each thread calculates
    int chunk_size = ceil(__int2double_rn(n_states) / blockDim.x);
    int remain = n_states - (chunk_size-1) * blockDim.x;
    int state_start, state_end; //depend on tidx
    if (tidx < remain) {//first set of threads (total: remain) calculate chunk_size states
        state_start = tidx * chunk_size;
        state_end = state_start + chunk_size;
    } else {//threads after remain caclulate chunk_size-1 states
        state_start = remain * chunk_size + (tidx - remain) * (chunk_size-1);
        state_end  = state_start + chunk_size-1;
    }
    double sum;
    //1. Initialization
    for (int j = state_start; j < state_end; j++) {
        if(j==n_states-2){
            d_prev_probs[j] = 0.0;
        } else {
            d_prev_probs[j] = -INFINITY;
        }
        matrix_set(beta,j,peak_end-1,T,d_prev_probs[j]);  
    }
    __syncthreads();
    //2. Induction
    for (int t=peak_end-2; t >=peak_start; t--) {
        for (int i = state_start; i<state_end; i++){
            sum = -INFINITY;
            for (int j=0; j<n_states; j++){
                sum = logCheckAdd(sum,d_prev_probs[j] + matrix_get(log_A_matrix,i,j,n_states) + matrix_get(emission_matrix,j,t+1,T));
            }
            d_current_probs[i] = sum;
            matrix_set(beta,i,t,T,sum);
        }
        __syncthreads();
        //update probs on t with t+1
        for (int j = state_start; j < state_end; j++) {
            d_prev_probs[j] = d_current_probs[j];
        }
        __syncthreads(); 
    }
}

__global__ void viterbi_kernel(int *psi, double *posterior,
             double *pwm_matrix, double *phmm_thresholds, 
             int *D, //calculate TF inside kernel & loop
             double *logprobf, double *alpha, double *beta, 
             int *q,
             int T, int n_states, int *peakPos, double *pi, int M, int inactive,
             double *emission_matrix, double *log_A_matrix) { 
    extern __shared__ double shared_bank[]; 
    //store two columns (length = n_states) for max probs. on time t-1 & t
    double *d_prev_probs = shared_bank;
    double *d_current_probs = d_prev_probs + n_states;
    int const k = blockIdx.x; //i.e. peakID, each block takes a peak
    int const tidx = threadIdx.x;
    int const peak_start = peakPos[k] -1;
    int const peak_end = peakPos[k+1] -1;
    //assign the states each thread calculates
    int chunk_size = ceil(__int2double_rn(n_states) / blockDim.x);
    int remain = n_states - (chunk_size-1) * blockDim.x;
    int state_start, state_end; //depend on tidx
    if (tidx < remain) {//first set of threads (total: remain) calculate chunk_size states
        state_start = tidx * chunk_size;
        state_end = state_start + chunk_size;
    } else {//threads after remain caclulate chunk_size-1 states
        state_start = remain * chunk_size + (tidx - remain) * (chunk_size-1);
        state_end  = state_start + chunk_size-1;
    }
    int TF =0;
    double maxval;
    double val;
    int maxvalind;
    double temp;
    double temp_alpha_beta_logprobf;
    int nonInf;
    int l;
    //1. Initialization
    for (int j = state_start; j < state_end; j++) {
        d_prev_probs[j] = log(pi[j]) + matrix_get(emission_matrix,j,peak_start,T);
        matrix_set_int(psi,peak_start,j,n_states,n_states);
        matrix_set(posterior,j,peak_start,T, matrix_get(alpha,j,peak_start,T) + matrix_get(beta,j,peak_start,T) - logprobf[k]);
    }
    __syncthreads();
    if (tidx==0){ //or write in previous loop...
        TF = 0;
        for (int x = 0; x < M; x++){
            //for j in the last position of motif x
            matrix_set(posterior,TF + D[x] -1,peak_start,T,DVITTINY);
            matrix_set(posterior,TF + D[x] * (inactive+1) -1,peak_start,T,DVITTINY);
            TF += D[x] * (inactive+1);
        }  
    }
    __syncthreads();
    //2. Recursion
    for (int t = peak_start + 1; t < peak_end; t++) {
        for (int j = state_start; j< state_end; j++) {
            maxval = DVITTINY;
            maxvalind = n_states - 1;
            for (int i = 0; i<n_states; i++) {
                val = d_prev_probs[i] + matrix_get(log_A_matrix,i,j,n_states);
                if (val > maxval) {
                    maxval = val;
                    maxvalind = i;
                }
            }
            matrix_set_int(psi, t, j, n_states, maxvalind);
            d_current_probs[j] = maxval + matrix_get(emission_matrix,j,t,T);
            matrix_set(posterior, j,t, T, matrix_get(alpha,j,t,T) + matrix_get(beta,j,t,T) - logprobf[k]);
            TF = 0;
            for (int x = 0; x < M ; x++) { //motif index [0,10)
              //if the motif index x maps to current j
              if (j >= TF && j < TF + D[x] * (inactive+1)) {
                //first check pwm thresholds, update delta[j][t]
                if (matrix_get(pwm_matrix, x, t, T) < (phmm_thresholds[x] - 0.5)) { //CHECK the threshold consistent with esthmm_gpu.c
                  d_current_probs[j] = maxval + DVITTINY;
                }
                //then treat t as the last position of motif x
                if (j == TF + D[x] -1 || j == TF + D[x] * (inactive+1) -1) {
                  if (t <= (peak_start + 1 + D[x])) {
                    matrix_set(posterior,j,t,T,log(0.0));
                  }
                  else {
                      temp = 0.0;
                      nonInf = 0;
                      temp_alpha_beta_logprobf = 0.0;
                      for (l = 0; l < D[x]; l++) {
                        temp_alpha_beta_logprobf = matrix_get(alpha,j-l,t-l,T) + matrix_get(beta,j-l,t-l,T) -  logprobf[k];
                        if (temp_alpha_beta_logprobf != -INFINITY) {
                          temp += temp_alpha_beta_logprobf;
                          nonInf += 1;
                        }

                      }
                      if (nonInf == 0) temp = DVITTINY;
                      for (l = 0; l < D[x]; l++) {
                            matrix_set(posterior,j,t-l,T,temp); //set post prob for all positions of this pwm
                      }
                  }
                }
                break;
              }
              //update sum of TF motif lengths  
              TF += D[x] * (inactive+1);
            }
        }
        __syncthreads();
        //update probs on t with t+1
        for (int j = state_start; j < state_end; j++) {
            d_prev_probs[j] = d_current_probs[j];
        }
        __syncthreads();
    }
    // 3. Termination
    if (tidx == 0) {//Thread 0 do termination & backtracing
        maxval = DVITTINY;
        maxvalind = n_states-1;
        for (int i = 0; i<n_states; i++) {
            if (d_prev_probs[i] > maxval) {
                maxval = d_prev_probs[i];
                maxvalind = i;
            }
        }
        q[peak_end-1] = maxvalind; //for each T
        //4. Path backtracing
        for (int t=peak_end-2; t>=peak_start; t--) {
            q[t] = matrix_get_int(psi, t+1, maxvalind, n_states);
            maxvalind = q[t];
        }
    }
    __syncthreads();
}
///////////////////////End of gpu kernels///////////////////////

///////host function to prepare arrays for gpu & invoke gpu kernels///////
__host__ void EmissionMatrix_viterbi_gpu(FILE *fpOut,char *TFname, HMM* phmm, gsl_matrix * obs_matrix, int P, int *peakPos, 
                       gsl_matrix ** cov_matrix_tmp, gsl_vector ** deleted_vector, gsl_matrix * pwm_matrix, int T,
                       char **chr,int *peakStart, int *peakEnd)
{
    int K = phmm->K; //# of features; for easier refering
  	int N = phmm->N; //# of states; for easier refering
    //constant parameters for final model
    int M = 10; //TF number
    int inactive = 1;
  	///////Emission calculation part/////////
    //Assuming already did LL' decomp (i.e. updated cov_matrix_tmp & deleted_vector)
    //Prepare linearized arrays for gpu
  	/* Input arrays: 2D array: i,j; j:width; 3D array: x,y,z:width,height,depth; x + y*width + z*width*height;
  	int *error_rows: K x N 
  	1: rows in positive-definite part; 
  	0: error rows count individaully; 
  	-2 (first element): all rows removed for this state;
  	double *mean_matrix: K x N
  	double *cov_matrix: K x K x N
  	double *L: K x K x N
  	double *obs_matrix: K x T
  	Compute on gpu:
  	double *emission_matrix: N x T  */
  int i,j,t,m;
  int x,y;
  const int FULL = -1;
  const int EMPTY = -2;
  int *h_error_rows;
  size_t error_rows_size = K * N * sizeof(*h_error_rows);
  h_error_rows = (int *) malloc (error_rows_size);
  for (i = 0; i < N; i++){
    j = 0;
    if (gsl_vector_get(deleted_vector[i], 0) == FULL) {
      for (m=0;m<K;m++){
        matrix_set_int(h_error_rows,m,i,N,1);
      }
    } else if (gsl_vector_get(deleted_vector[i], 0) == EMPTY) {
      matrix_set_int(h_error_rows,0,i,N,EMPTY); //assign -2 to first element
    }
    else{
      for (m = 0; m < K; m++) {
        if ((m-j) != gsl_vector_get(deleted_vector[i], j)){
          matrix_set_int(h_error_rows,m,i,N,1);
        }
        else { //is error row
          matrix_set_int(h_error_rows,m-j,i,N,0);
        }
        j++;
      }
    }     
  }

  double *h_mean_matrix;
  size_t mean_matrix_size = K * N * sizeof(*h_mean_matrix);
  h_mean_matrix = (double *) malloc (mean_matrix_size);
  for (i = 0; i< N; i++){
    for (m = 0; m<K; m++) {
      matrix_set(h_mean_matrix,m,i,N,gsl_matrix_get(phmm->mean_matrix,m,i));
    }
  }

  double *h_cov_matrix;
  size_t cov_matrix_size = K * K * N * sizeof(*h_cov_matrix);
  h_cov_matrix = (double *) malloc (cov_matrix_size);
  for (i=0; i<N; i++){
    for (x=0; x<K;x++){
      for (y=0; y<K;y++){
        matrix_3D_set(h_cov_matrix,x,y,i,K,K,gsl_matrix_get(phmm->cov_matrix[i],y,x)); //note x,y order, symmertirc does not matter here though
      }
    }
  }

  double *h_L = 0;
  size_t L_size = K * K * N * sizeof(*h_L);
  h_L = (double *) malloc (L_size);
  for (i = 0; i < N; i++){
    if (gsl_vector_get(deleted_vector[i], 0) == FULL) {
      for (y=0;y<K;y++){
          for (x=0;x<=y;x++){ //only need lower triangle part
               matrix_3D_set(h_L,x,y,i,K,K,gsl_matrix_get(cov_matrix_tmp[i],y,x));
          }
      }
    } else if (gsl_vector_get(deleted_vector[i], 0) != EMPTY) { //if EMPTY -> L remain 0
      for (y=0;y<K - deleted_vector[i]->size;y++){ //size of positive-definite part;
          for (x=0;x<=y;x++){ //only need lower triangle part
               matrix_3D_set(h_L,x,y,i,K,K,gsl_matrix_get(cov_matrix_tmp[i],y,x));
          }
      }
      }
   }

  double *h_obs_matrix;
  size_t obs_matrix_size = K * T * sizeof(*h_obs_matrix);
  h_obs_matrix = (double *) malloc (obs_matrix_size);
  for (m=0;m<K;m++){
    for (t=0;t<T;t++){
      matrix_set(h_obs_matrix,m,t,T,gsl_matrix_get(obs_matrix,m,t));
    }
  }

  //double *h_emission_matrix=0; //temp array to copy back from gpu for checking
  //size_t emission_matrix_size = N * T * sizeof(*h_emission_matrix);
  //h_emission_matrix = (double *) malloc (emission_matrix_size);

  ////Copy arrays to gpu to calculate emission matrix////
  int *d_error_rows;
  HANDLE_ERROR(cudaMalloc(&d_error_rows,error_rows_size));
  HANDLE_ERROR(cudaMemcpy(d_error_rows,h_error_rows,error_rows_size,cudaMemcpyHostToDevice));
  double *d_mean_matrix;
  HANDLE_ERROR(cudaMalloc(&d_mean_matrix,mean_matrix_size));
  HANDLE_ERROR(cudaMemcpy(d_mean_matrix,h_mean_matrix,mean_matrix_size,cudaMemcpyHostToDevice));
  double *d_cov_matrix;
  HANDLE_ERROR(cudaMalloc(&d_cov_matrix,cov_matrix_size));
  HANDLE_ERROR(cudaMemcpy(d_cov_matrix,h_cov_matrix,cov_matrix_size,cudaMemcpyHostToDevice));
  double *d_L;
  HANDLE_ERROR(cudaMalloc(&d_L,L_size));
  HANDLE_ERROR(cudaMemcpy(d_L,h_L,L_size,cudaMemcpyHostToDevice));
  double *d_obs_matrix;
  HANDLE_ERROR(cudaMalloc(&d_obs_matrix,obs_matrix_size));
  HANDLE_ERROR(cudaMemcpy(d_obs_matrix,h_obs_matrix,obs_matrix_size,cudaMemcpyHostToDevice));
  //arrays compute on gpu
  double *d_emission_matrix = 0;
  size_t emission_matrix_size = N * T * sizeof(*d_emission_matrix);
  cudaMalloc(&d_emission_matrix,emission_matrix_size);
  ////Set dimensions for grid,block and invoke kernel////
  //measure kernel running times
  cudaEvent_t start, stop_1,stop_2,stop_3,stop_4,stop_5,stop_6;
  float time; 
  cudaEventCreate(&start);   
  cudaEventCreate(&stop_1);
  cudaEventCreate(&stop_2);
  cudaEventCreate(&stop_3);
  cudaEventCreate(&stop_4);
  cudaEventCreate(&stop_5);
  cudaEventCreate(&stop_6);
  cudaEventRecord(start, 0); // start measuring  the time
  
  get_emission_kernel<<<N,THREADS_PER_BLOCK>>> (d_mean_matrix,d_cov_matrix,d_obs_matrix,N,T,
                                            d_L,d_error_rows,d_emission_matrix);
                                                cudaEventRecord(stop_1, 0);
  cudaEventRecord(stop_1, 0);
  cudaEventSynchronize(stop_1);
  cudaEventElapsedTime(&time, start, stop_1);
  printf("get_emission_kernel time %3.1f ms\n", time);   
  //cudaMemcpy(h_emission_matrix,d_emission_matrix,emission_matrix_size,cudaMemcpyDeviceToHost); //copy back emission matrix from gpu for check
  //Free arrays for emission part
  cudaFree(d_error_rows);
  cudaFree(d_mean_matrix);
  cudaFree(d_cov_matrix);
  cudaFree(d_L);
  cudaFree(d_obs_matrix);
  //cudaFree(d_emission_matrix);
  free(h_error_rows);
  free(h_mean_matrix);
  free(h_cov_matrix);
  free(h_L);
  free(h_obs_matrix);
  //free(h_emission_matrix);
  	
  ///////Viterbi part/////////
  //Final cpu arrays copy from gpu to calculate post prob.
  double *posterior = (double *) calloc(T * N, sizeof(double)); //output all states
  //double *posterior = (double *) calloc(2 * phmm->D[0] * T, sizeof(double)); //only output first two states
  int *q = (int *) calloc(T, sizeof(int));
  //Allocate & copy gpu memory//
  int *d_D;
  size_t D_size = M * sizeof(*d_D);
  HANDLE_ERROR(cudaMalloc(&d_D,D_size));
  HANDLE_ERROR(cudaMemcpy(d_D,phmm->D,D_size,cudaMemcpyHostToDevice));
  double *h_phmm_log_A_matrix = 0;
  double *d_phmm_log_A_matrix = 0;
  size_t phmm_log_A_matrix_size = N * N * sizeof(*h_phmm_log_A_matrix);
  h_phmm_log_A_matrix = (double *) malloc (phmm_log_A_matrix_size);
  gsl_matrix_to_arr(N, N, phmm->log_A_matrix, h_phmm_log_A_matrix);
  HANDLE_ERROR(cudaMalloc(&d_phmm_log_A_matrix, phmm_log_A_matrix_size));
  HANDLE_ERROR(cudaMemcpy(d_phmm_log_A_matrix, h_phmm_log_A_matrix, phmm_log_A_matrix_size, cudaMemcpyHostToDevice)); 
  
  double *h_pwm_matrix = 0;
  double *d_pwm_matrix = 0;
  size_t pwm_matrix_size = M * T * sizeof(*h_pwm_matrix);
  h_pwm_matrix = (double *)malloc (pwm_matrix_size);
  gsl_matrix_to_arr(M, T, pwm_matrix, h_pwm_matrix);
  HANDLE_ERROR(cudaMalloc(&d_pwm_matrix, pwm_matrix_size));
  HANDLE_ERROR(cudaMemcpy(d_pwm_matrix, h_pwm_matrix, pwm_matrix_size, cudaMemcpyHostToDevice));
    
  int *d_peakPos = 0;
  size_t peakPos_size = (P + 1) * sizeof(*d_peakPos);
  HANDLE_ERROR(cudaMalloc(&d_peakPos, peakPos_size));
  HANDLE_ERROR(cudaMemcpy(d_peakPos, peakPos, peakPos_size, cudaMemcpyHostToDevice));
    
  double *d_pi = 0;
  size_t pi_size = N * sizeof(*d_pi);
  HANDLE_ERROR(cudaMalloc(&d_pi, pi_size));
  HANDLE_ERROR(cudaMemcpy(d_pi, phmm->pi, pi_size, cudaMemcpyHostToDevice));

  double *d_thresholds = 0;
  size_t thresholds_size = M * sizeof(*d_thresholds);
  HANDLE_ERROR(cudaMalloc(&d_thresholds, thresholds_size));
  HANDLE_ERROR(cudaMemcpy(d_thresholds, phmm->thresholds, thresholds_size, cudaMemcpyHostToDevice));
  //arrays to compute on gpu
  double *d_alpha = 0;
  size_t alpha_size = N * T * sizeof(*d_alpha);
  HANDLE_ERROR(cudaMalloc(&d_alpha, alpha_size));
    
  double *d_logprobf = 0;
  size_t logprobf_size = P * sizeof(*d_logprobf);
  HANDLE_ERROR(cudaMalloc(&d_logprobf, logprobf_size));    
    
  double *d_beta = 0;
  size_t beta_size = N * T * sizeof(*d_beta);
  HANDLE_ERROR(cudaMalloc(&d_beta, beta_size));

  int *d_psi = 0;
  size_t psi_size = T * N * sizeof(*d_psi);
  HANDLE_ERROR(cudaMalloc(&d_psi, psi_size));

  double *d_posterior = 0;
  size_t posterior_size = N * T * sizeof(*d_posterior);
  HANDLE_ERROR(cudaMalloc(&d_posterior, posterior_size));

  int *d_q = 0;
  size_t q_size = T * sizeof(*d_q);
  HANDLE_ERROR(cudaMalloc(&d_q, q_size));
  /////define shared memory/////
  size_t shared_mem_size = 2 * N * sizeof(double); //two columns for t-1 & t
  cudaEventRecord(stop_2, 0);
  cudaEventSynchronize(stop_2);
  cudaEventElapsedTime(&time, stop_1, stop_2);
  printf("Prepare arrays for viterbi part time %3.1f ms\n", time); 
  fwd_kernel<<<P,THREADS_PER_BLOCK, shared_mem_size>>>(d_alpha,
                                                d_logprobf,
                                                     T,  
    												  N,
    												  d_peakPos,
    												  d_pi,
    												  d_emission_matrix,
    												  d_phmm_log_A_matrix);
  cudaEventRecord(stop_3, 0);
  cudaEventSynchronize(stop_3);
  cudaEventElapsedTime(&time, stop_2, stop_3);
  printf("fwd kernel time %3.1f ms\n", time); 
  bwd_kernel<<<P,THREADS_PER_BLOCK, shared_mem_size>>>(d_beta,
                                                      T,  
    												  N,
    												  d_peakPos,
    												  d_emission_matrix,
    												  d_phmm_log_A_matrix);
  cudaEventRecord(stop_4, 0);
  cudaEventSynchronize(stop_4);
  cudaEventElapsedTime(&time, stop_3, stop_4);
  printf("bwd kernel time %3.1f ms\n", time); 
  viterbi_kernel<<<P,THREADS_PER_BLOCK, shared_mem_size>>>(d_psi,d_posterior,
                     d_pwm_matrix,d_thresholds,
                     d_D,
                     d_logprobf,d_alpha,d_beta,
                     d_q,
                     T,N,d_peakPos,d_pi,M,inactive,
                     d_emission_matrix,d_phmm_log_A_matrix);
  cudaEventRecord(stop_5, 0);
  cudaEventSynchronize(stop_5);
  cudaEventElapsedTime(&time, stop_4, stop_5);
  printf("viterbi kernel time %3.1f ms\n", time); 
  HANDLE_ERROR(cudaMemcpy(q, d_q, T * sizeof(int), cudaMemcpyDeviceToHost));
  //copy complete posterior array
  HANDLE_ERROR(cudaMemcpy(posterior,d_posterior,T * N * sizeof(double), cudaMemcpyDeviceToHost));
  //only copy posterior for state 1,2 in final output to save time -> need to set width as T for posterior array; also use getPosterior_all_2
  //HANDLE_ERROR(cudaMemcpy(posterior,d_posterior,2 * phmm->D[0] * T * sizeof(double), cudaMemcpyDeviceToHost));
  cudaEventRecord(stop_6, 0);
  cudaEventSynchronize(stop_6);
  cudaEventElapsedTime(&time, stop_5, stop_6);
  printf("posterior arrays copy time %3.1f ms\n", time);   
  //Free memory on gpu
  cudaFree(d_alpha);
  cudaFree(d_logprobf);
  cudaFree(d_beta);
  cudaFree(d_peakPos);
  cudaFree(d_pi);
  cudaFree(d_emission_matrix);
  cudaFree(d_phmm_log_A_matrix);
  cudaFree(d_psi);
  cudaFree(d_posterior);
  cudaFree(d_pwm_matrix);
  cudaFree(d_thresholds);
  cudaFree(d_q);
  cudaFree(d_D);
  //output predictions
  getPosterior_all(fpOut,TFname,T,q,P,peakPos,posterior,N,phmm->extraState,phmm->D,chr,peakStart,peakEnd); //need complete posterior array
  //Free memory on cpu
  free((double*) (posterior));
  free((int*) (q));
  free((double*) (h_phmm_log_A_matrix));
  free((double*) (h_pwm_matrix));
}
