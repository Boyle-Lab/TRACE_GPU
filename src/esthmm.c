/*
 *  File: esthmm.c
 *
 *  The main function of training step in TRACE.
 *  (embedded data processing steps)
 *
 *  The HMM structure and some codes are borrowed and modified from Kanungo's
 *  original HMM program.
 *  Tapas Kanungo, "UMDHMM: Hidden Markov Model Toolkit," in "Extended Finite State Models of Language," A. Kornai (editor), Cambridge University Press, 1999. http://www.kanungo.com/software/software.html.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nrutil.h"
#include "hmm.h"
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <getopt.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>
#include "TRACE_gpu_fwd.h"
#include <sys/time.h>

u_int64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * (u_int64_t) 1000000 + tv.tv_usec;
}

void Usage(char *name);

int main (int argc, char **argv)
{
  /* The following are read from input files */
  FILE *fp, *fp1, *fp2;
  HMM hmm; /* Initialize the HMM structure */
  int P; /* Total number of peaks */
  int *peakPos; /* Starting location of each peaks*/
                /*there two are used to seperate values from each peak in concatenated data*/
  int totalT, T; /* Total length and length in each run*/
  int peakLen; /* Length of each peak */
  int peakPer = 5; /* number of peaks processed in each run, default is 1*/
  int *O1, *O2; /* Temporarily store the list of sequence, represented by numbers,
                 * 1=A, 2=C, 3=G, 4=T */
  double *GC; /* GC content */
  gsl_matrix *pwm_matrix, *obs_matrix; /* Matrix of PWM scores, and observation data
             /* right now, there are only three observations: counts, slop and sequence
              might need to change the structure if more data are used in the future */
  gsl_vector *slop_vector, *counts_vector;
  
  int niter; /* Numbers of iterations used in training */
  int i, j, n, m, k, index, PosIndex;
  int c, w = 0, indexTF;
  MAXITERATION = 200; /* Set default numbers for max iteration and number of threads */
  THREAD_NUM = 40;
  /* Get all command-line options */
  extern int optind, opterr, optopt;
  extern char *optarg;
  /* Flags for all command line options */
  int oflg=0, mflg=0, nflg=0, aflg =0, bflg =0, pflg =0, fflg=0, eflg=0, tflg=0, lflg=0, gpuflg = 0;
  int errflg=0, vflg=0, iflg=0, sflg=0, rflg=0, zflg=0, xflg = 0, gflg = 0, cflg = 0, siflg = 0, eiflg = 0;
  int nStart = 0;
  int nTF = 746; //TODO: change TF list length
  int nEnd = nTF;
  char *slopefile, *countfile, *seqfile, *listfile, *thresholdfile;
  char *hmminitfile, *motiffile, *outfile, *scorefile, *predfile;
  char peakfile[BUFSIZE], vcf_file[BUFSIZE], bamfile[BUFSIZE], fastafile[BUFSIZE], sizefile[BUFSIZE];
  char sample[BUFSIZE], dsf[BUFSIZE];
  char chrFile[BUFSIZE];
  int ifMulti = 2; /* defalult model is multivaraint normal distribution
                      with problematic hidden state dropped */
  int ifSkip = 0; /* if skip training step and start viterbi step
                      default is no */
  int GPU = 1;
  u_int64_t start, start0, end;
  
  hmminitfile = (char *)malloc(BUFSIZ*sizeof(char));
  motiffile = (char *)malloc(BUFSIZ*sizeof(char));
  outfile = (char *)malloc(BUFSIZ*sizeof(char));
  predfile = (char *)malloc(BUFSIZ*sizeof(char));
  scorefile = (char *)malloc(BUFSIZ*sizeof(char));
    
  hmminitfile[0] = '\0';
  peakfile[0] = '\0';
  outfile[0] = '\0';
  predfile[0] = '\0';
  scorefile[0] = '\0';
  motiffile[0] = '\0';
  vcf_file[0] = '\0';
  sample[0] = '\0';
  dsf[0] = '\0';
  chrFile[0] = '\0';
  static struct option longopts[] = {
    {"final-model", required_argument, NULL, 'O'},
    {"initial-model", required_argument, NULL, 'I'},
    {"peak-file", required_argument, NULL, 'P'},
    {"motif-file", required_argument, NULL, 'F'},
    {"thread", required_argument, NULL, 'T'},
    {"threshold-file", required_argument, NULL, 'E'},
    {"max-inter", required_argument, NULL, 'M'},
    {"model", required_argument, NULL, 'N'},
    {"predictions-file", required_argument, NULL, 'B'},
    {"scores-file", required_argument, NULL, 'A'},
    {"viterbi", no_argument, NULL, 'V'},
    {"bam.file", required_argument, NULL, 'R'},
    {"fasta.file", required_argument, NULL, 'S'},
    {"size.file", required_argument, NULL, 'Z'},
    {"vcf.file", required_argument, NULL, 'G'},
    {"peak.number", required_argument, NULL, 'X'},
    {"sample-name", required_argument, NULL, 's'},
    {"dsf", required_argument, NULL, 'd'},
    {"chr", required_argument, NULL, 'C'},
    {"startIndex", required_argument, NULL, 'i'},
    {"endIndex", required_argument, NULL, 'e'},
    {"GPU", required_argument, NULL, 'g'},
          //{.name = "seq.file", .has_arg = no_argument, .val = 'Q'},
          //{.name = "count.file", .has_arg = no_argument, .val = 'C'},
          //{.name = "slope.file", .has_arg = no_argument, .val = 'S'},
          //{.name = "init.model", .has_arg = no_argument, .val = 'I'},
    {0,         0,                 0,  0 }
  };
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "vhO:A:B:M:N:P:T:E:F:I:R:S:Z:X:G:s:d:i:e:g:C:V", longopts, &option_index)) != EOF){
    switch (c) {
      case 'v':
        vflg++;
        break;
      case 'h':
        Usage(argv[0]);
        exit(1);
        break;
      case 'T':
      /* set number of threads */
        if (tflg)
          errflg++;
        else {
          tflg++;
          THREAD_NUM = atoi(optarg);
        }
        break;
      case 'E':
      /* get the threshold file name*/
        if (eflg)
          errflg++;
        else {
          eflg++;
          thresholdfile = optarg;
        }
        break;
      case 'M':
      /* get the max number of iteration */
        if (mflg)
          errflg++;
        else {
          mflg++;
          MAXITERATION = atoi(optarg);
        }
        break;
      case 'O':
      /* get the HMM output file name*/
        if (oflg)
          errflg++;
        else {
          oflg++;
          strcat(outfile, optarg);
        }
        break;
      case 'I':
      /* get the initial HMM file name*/
        if (iflg)
          errflg++;
        else {
          iflg++;
          strcat(hmminitfile, optarg);
        }
        break;
      case 'N':
      /* choose independ normal or multivariance, default is independent(0)*/
        if (nflg)
          errflg++;
        else {
          nflg++;
          ifMulti = atoi(optarg);
        }
        break;
      case 'A':
      /* get the output file name */
        if (aflg)
          errflg++;
        else {
          aflg++;
          strcat(scorefile, optarg);
        }
        break;
      case 'B':
      /* get the output file name */
        if (bflg)
          errflg++;
        else {
          bflg++;
          strcat(predfile, optarg);
        }
        break;
      case 'F':
        /* get the peak with motif sites file name */
        if (fflg)
          errflg++;
        else {
          fflg++;
          strcat(motiffile, optarg);
        }
        break;
      case 'P':
      /* get the peak file name */
        if (pflg)
          errflg++;
        else {
          pflg++;
          strcat(peakfile, optarg);
        }
        break;
      case 'S':
      /* get the fasta file name */
        if (sflg)
          errflg++;
        else {
          sflg++;
          strcat(fastafile, optarg);
        }
        break;
      case 'R':
      /* get the bam file name */
        if (rflg)
          errflg++;
        else {
          rflg++;
          strcat(bamfile, optarg);
        }
        break;
      case 'Z':
      /* get the size file name */
        if (zflg)
          errflg++;
        else {
          zflg++;
          strcat(sizefile, optarg);
        }
        break;
      case 'G':
      /* get the size file name */
        if (gflg)
          errflg++;
        else {
          gflg++;
          strcat(vcf_file, optarg);
        }
        break;
      case 'X':
      /* number of peaks to be processed in each run */
        if (xflg)
          errflg++;
        else {
          xflg++;
          peakPer = atoi(optarg);
        }
        break;
      case 'V':
      /* if skip training step */
        if (lflg)
          errflg++;
        else {
          lflg++;
          ifSkip = 1;
        }
        break;
      case 'C':
      /* get the chromosome name */
        if (cflg)
          errflg++;
        else {
          cflg++;
          strcat(chrFile, optarg);
        }
        break;
      case 's':
      /* get the sample name */
        if (vflg > 3)
          errflg++;
        else {
          vflg++;
          strcat(sample, optarg);
        }
        break;
      case 'd':
      /* get the sample name */
        if (vflg > 3)
          errflg++;
        else {
          vflg++;
          strcat(dsf, optarg);
        }
        break;
      case 'i':
      /* start index of TF list */
        if (siflg)
          errflg++;
        else {
          siflg++;
          nStart = atoi(optarg);
        }
        break;
      case 'g':
      /* if use GPU */
        if (gpuflg)
          errflg++;
        else {
          gpuflg++;
          GPU = atoi(optarg);
        }
        break;
      case 'e':
      /* end index of TF list */
        if (eiflg)
          errflg++;
        else {
          eiflg++;
          nEnd = atoi(optarg);
        }
        break;
      case '?':
        errflg++;
    }
  }
  if (errflg) {
    Usage(argv[0]);
    exit (1);
  }
    
    //char *chrFile="chr1_shuf_1";
    
    //char *peak_file="/home/nouyang/tmp_file/GM12878_hg38_Peak_chr1_shuf_1.bed";
    //char *peak_file="/home/nouyang/tmp_file/GM12878_hg38_Peak_chr1.bed";
    //char *peak_file="/home/nouyang/p2C/test_peak_m.bed";
    char *fasta_file="/data/genomes/hg38/seq/hg38.fa", *size_file="/data/genomes/hg38/hg38.chrom.sizes";
    char bam_file[BUFSIZE];
    char peak_file[BUFSIZE];
    if (!cflg) strcat(chrFile, "chr1_shuf_1");
    
    peak_file[0] = '\0';
    strcat(peak_file, "/home/nouyang/tmp_file/GM12878_hg38_Peak_");
    strcat(peak_file, chrFile);
    strcat(peak_file, ".bed");
    bam_file[0] = '\0';
    //strcat(bam_file, "/data/projects/FootPrinting/ENCFF577DUO.bam");
    if (vflg){
      strcat(vcf_file, "/data/projects/FootPrinting/vcf_files/");
      strcat(vcf_file, sample);
      strcat(vcf_file, "_chr1.vcf.gz");
      strcat(bam_file, "/data/projects/FootPrinting/");
      strcat(bam_file, sample);
      strcat(bam_file, "_");
      strcat(bam_file, dsf);
      strcat(bam_file, ".bam");
    }
    //"/home/nouyang/tmp_file/test_peak.bed"
  //char *peak_file="/home/nouyang/tmp_file/test_peak_m.bed", *bam_file="/nfs/boylelab_turbo/data_repo/Ningxin/vg_w_variants/NA18858_R21_L5_R5_L6_Y3_L6.bam", *fasta_file="/data/genomes/hg38/seq/hg38.fa", *size_file="/data/genomes/hg38/hg38.chrom.sizes";
    sflg=1;
    rflg=1;
    zflg=1;
    pflg=1;
    fflg=1;
    //strcat(motiffile, "/home/nouyang/p2C/test_peak_m_CTCF.bed");
  /* Check if the input files were provided */
  if (argc - optind != 2){
    if (sflg + rflg + zflg + pflg != 4){
      fprintf(stderr, "Error: required files not provided \n");
      Usage(argv[0]);
      exit (1);
    }
    else{
      checkFile(peak_file, "r");
      
      GC = dvector(4);
      totalT = 0;
      char prefix[BUFSIZE];
      prefix[0] = '\0';
      strcat(prefix, "_");
      char seq_file[BUFSIZE];
      seq_file[0] = '\0';
      //strcat(seq_file, "/data/projects/FootPrinting/variant_results_new/");
      //strcat(seq_file, sample);
      //strcat(seq_file, "_hg38_chr1_seq.txt");
      strcat(seq_file, "/data/projects/FootPrinting/processed_data/");
      strcat(seq_file, sample);
      strcat(seq_file, "_hg38_");
      strcat(seq_file, chrFile);
      strcat(seq_file, "_seq.txt");
      checkFile(seq_file, "r");
      fp = fopen(seq_file, "r");
      ReadSequence_va(fp, &totalT, GC, &O1, &O2, &P, &peakPos);
      fclose(fp);
      char count_file[BUFSIZE];
      count_file[0] = '\0';
      strcat(count_file, "/data/projects/FootPrinting/processed_data/");
      strcat(count_file, sample);
      strcat(count_file, "_hg38_");
      strcat(count_file, chrFile);
      strcat(count_file, "_count.txt");
      checkFile(count_file, "r");
      fp = fopen(count_file, "r");
      counts_vector = gsl_vector_alloc(totalT);
      ReadTagFile(fp, totalT, counts_vector, 1.0);
      fclose(fp);
      char slope_file[BUFSIZE];
      slope_file[0] = '\0';
      strcat(slope_file, "/data/projects/FootPrinting/processed_data/");
      strcat(slope_file, sample);
      strcat(slope_file, "_hg38_");
      strcat(slope_file, chrFile);
      strcat(slope_file, "_slope_2.txt");
      checkFile(slope_file, "r");
      fp = fopen(slope_file, "r");
      slop_vector = gsl_vector_alloc(totalT);
      ReadTagFile(fp, totalT, slop_vector, 1.0);
      fclose(fp);

      printf("%s %s %s %s %s %s\n DataProcessing finished\n", bam_file, vcf_file, peak_file, seq_file, count_file, slope_file);
      /*data processing functions from python*/
     // start = GetTimeStamp();
       // checkFile(bam_file, "r");
       // checkFile(fasta_file, "r");
       // checkFile(size_file, "r");
    // DataProcessing(peak_file, bam_file, fasta_file, size_file, vcf_file, prefix, &slop_vector, &counts_vector, GC, &totalT, &P, &peakPos, &O1, &O2); //for variant-aware cases, O1 O2 will be maternal and paternal sequence, otherwise O2 will be empty
     // printf("DataProcessing Time: %ld\n", GetTimeStamp() - start);
    }
  }
  else{
    /* Read existing input files */
    index = optind;
    seqfile = argv[index++]; /* Sequence input file */
    countfile = argv[index++]; /* Counts file */
    slopefile = argv[index++]; /* Slopes file */
    /* Read the observed sequence */
    checkFile(seqfile, "r");
    fp = fopen(seqfile, "r");
    GC = dvector(4);
    ReadLength(fp, &totalT);
    O1 = ivector(T);
    ReadSequence(fp, totalT, GC, O1, &P, &peakPos);
    fclose(fp);

    /* Read the slope file */
    checkFile(slopefile, "r");
    fp = fopen(slopefile, "r");
    slop_vector = gsl_vector_alloc(totalT);
    ReadTagFile(fp, totalT, slop_vector, 1.0);
    fclose(fp);

    /* Read the tag counts file */
    checkFile(countfile, "r");
    fp = fopen(countfile, "r");
    counts_vector = gsl_vector_alloc(totalT);
    ReadTagFile(fp, totalT, counts_vector, 1.0);
    fclose(fp);
  }
  /* run through all TFs in the list, if you want to change TF list, */
  /* make sure to change both file name and number of TFs*/
  //int nTF = 746; //TODO: change TF list length
  char *TFArray[nTF]; //TODO: change TF list
  char *motifArray[nTF]; //TODO: change TF list
  //HMM *hmmList = (HMM *) malloc((unsigned)(nTF*sizeof(HMM)));
  HMM hmmList[nTF];
  char *hmmFile;
  HMM *phmm;
  HMM hmm_i;
  checkFile("/data/projects/FootPrinting/TF_list_2020.txt", "r");
  fp = fopen("/data/projects/FootPrinting/TF_list_2020.txt", "r");
  ReadTFlistFile(fp, nTF, TFArray);
  checkFile("/data/projects/FootPrinting/motif_list_2020.txt", "r");
  fp = fopen("/data/projects/FootPrinting/motif_list_2020.txt", "r");
  ReadTFlistFile(fp, nTF, motifArray);
  fclose(fp);
  #pragma omp parallel num_threads(THREAD_NUM) \
  private(hmmFile, thresholdfile, fp, j)
  {
  #pragma omp for
  for (i = 0; i < nTF; i++){
    /* Check if user only wants to run decoding step*/
    if (ifSkip){
      /* Read HMM input file */
      //if (!oflg){
        //fprintf(stderr, "Error: final model file required \n");
        //exit (1);
      //}
      hmmFile = malloc(BUFSIZ*sizeof(char));
      hmmFile[0] = '\0';
      strcat(hmmFile, "/data/projects/FootPrinting/variant_results/GM12878_hg38_");
      strcat(hmmFile, TFArray[i]);
      strcat(hmmFile, "_10_chr1_corrected_new.txt");
      if (checkFile_noExit(hmmFile, "r")){
        //hmmList[i] = malloc(sizeof(HMM));
        //phmm = (HMM *)malloc(sizeof(HMM));
        fp = fopen(hmmFile, "r");
        hmmList[i].model = ifMulti;
        ReadM(fp, &hmmList[i]);
        ReadHMM(fp, &hmmList[i]);
        fclose(fp);
        if (GC){
          hmmList[i].bg[0] = hmmList[i].bg[3] = GC[0];
          hmmList[i].bg[2] = hmmList[i].bg[1] = GC[1];
        }
        else{
          hmm_i.bg[0]=hmm_i.bg[1]=hmm_i.bg[2]=hmm_i.bg[3]=0.25;
        }
        //printf("%d: %s %d\n", i, TFArray[i], hmm_i.N);
        //copyHMM(&hmm_i, &hmmList[i]);
        //FreeHMM(&hmm_i);
      }
      else hmmList[i].N = 0;
      if (hmmFile != NULL) free(hmmFile);
      /* Check region of interest file is provided for decoding */
      if (!pflg && !fflg){
        fprintf(stderr, "Error: peak file required \n");
        exit (1);
      }
    }
    
    else{
      /* Initialize the HMM model */
      /* Read HMM input file */
      if (!iflg){
        fprintf(stderr, "Error: initial model file required \n");
        exit (1);
      }
      checkFile(hmminitfile, "r");
      fp = fopen(hmminitfile, "r");
      hmm.model = ifMulti;
      ReadM(fp, &hmm);
      if (hmm.M == 0) ReadInitHMM(fp, &hmm);
      else ReadHMM(fp, &hmm);
      if (GC){
        hmm.bg[0] = hmm.bg[3] = GC[0];
        hmm.bg[2] = hmm.bg[1] = GC[1];
      }
      else{
        hmm.bg[0]=hmm.bg[1]=hmm.bg[2]=hmm.bg[3]=0.25;
      }
      fclose(fp);
      /* Check given file names are valid */
      /* If trained model file name is not provided, use initial model file name
         with suffix "_final_model.txt" */
      if (!oflg){
        strcat(outfile,hmminitfile);
        strcat(outfile,"_final_model.txt");
      }
      checkFile(outfile, "w");
    }
    
  }
  }
  for (i = 0; i < nTF; i++){
    if(hmmList[i].N){
      hmmList[i].thresholds = (double *) dvector(hmmList[i].M);
      /* with a given threshold file, TRACE can limit the state labeling to the
         regions with PWM score higher than the provided value */
      thresholdfile = malloc(BUFSIZ*sizeof(char));
      thresholdfile[0] = '\0';
      strcat(thresholdfile, "/home/nouyang/tmp_file/");
      strcat(thresholdfile, motifArray[i]);
      strcat(thresholdfile, "_threshold.txt");
      checkFile(thresholdfile, "r");
      eflg=1;
      if (eflg) {
        fp = fopen(thresholdfile, "r");  //TODO: thresholds function
        for (j = 0; j < hmmList[i].M; j++) {
          if(fscanf(fp, "%lf\n", &(hmmList[i].thresholds[j])) == EOF){
            fprintf(stderr, "Error: threshold file error \n");
            exit (1);
          }
          //printf("%d: %lf\n", j, hmmList[i].thresholds[j]);
        }
        fclose(fp);
        //printf("thresholds set\n");
      }
      else{
        for (j = 0; j < hmmList[i].M; j++) {
          hmmList[i].thresholds[j] = -INFINITY;
        }
      }
      if (thresholdfile != NULL) free(thresholdfile);
      //printf("%d: %s %s %d %lf %lf\n", i, TFArray[i], motifArray[i], hmmList[i].N, gsl_matrix_get(hmmList[i].mean_matrix, 0,0),hmmList[i].thresholds[0]);
    }
  }
    
  strcat(peakfile, peak_file);
    
  /* Check given file name is valid and read peak file */
  checkFile(peakfile, "r");
  int *peakStart = ivector(P);
  int *peakEnd = ivector(P);
  char *chr[P];
  //char **chr = (char **)malloc(P * sizeof(char *));
  fp = fopen(peakfile, "r");
  ReadPeakFile(fp, P, chr, peakStart, peakEnd);
  fclose(fp);

  //char **TFchr;
  int *TFstart, *TFend, *TFlength, *motifIndex;
  int P_i, totalT_i, L, L_i;
  int *peakPos_i, *posIndex_i;
  int *O1_i, *O2_i;
  char **chr_i;
  int *peakStart_i, *peakEnd_i, *TFstart_i, *TFend_i, *TFlength_i, *motifIndex_i;
  gsl_vector * counts_vector_i;
  gsl_vector * slop_vector_i;
  gsl_matrix * obs_matrix_i;
  gsl_matrix * tmp_matrix;
  gsl_vector * tmp_vector, * tmp_vector_i;
  
  for (index = nStart; index < nEnd; index++){
    if(hmmList[index].N){
      printf("processing TF %d: %s\n", index, TFArray[index]);
      /* If prediction file name is not provided, use initial model file name
         with suffix "_viterbi_results.txt" */
      if (pflg && (!bflg)) {
        //strcat(predfile, outfile);
        predfile[0] = '\0';
        strcat(predfile, "/data/projects/FootPrinting/variant_results_new/");
        strcat(predfile, sample);
        strcat(predfile, "_");
        strcat(predfile, TFArray[index]);
        strcat(predfile, "_");
        strcat(predfile, chrFile);
        strcat(predfile, "_viterbi_results.txt");
          //strcat(predfile, "/home/nouyang/p2C/test/test_");
        //strcat(predfile, TFArray[index]);
          // strcat(predfile, "_viterbi_results.txt");
      }
      
      if (fflg){
        motiffile[0] = '\0';
        strcat(motiffile, "/home/nouyang/tmp_file/GM12878_hg38_OCAPeak_");
        strcat(motiffile, chrFile);
        strcat(motiffile, "_");
        strcat(motiffile, TFArray[index]);
        strcat(motiffile, ".bed");
        //strcat(motiffile, "/home/nouyang/p2C/test/test_peak_m_");
        //strcat(motiffile, TFArray[index]);
        //strcat(motiffile, ".bed");
        checkFile(motiffile, "r");
        ReadMotifFile(motiffile, P, L, &TFstart, &TFend, &TFlength, &motifIndex);
        /* If prediction file name is not provided, use initial model file name
           with suffix "_with_probs.txt.txt" */
        if (!aflg) {
          scorefile[0] = '\0';
          strcat(scorefile, "/data/projects/FootPrinting/variant_results_new/");
          strcat(scorefile, sample);
          strcat(scorefile, "_");
          strcat(scorefile, TFArray[index]);
          strcat(scorefile, "_");
          strcat(scorefile, chrFile);
          strcat(scorefile, "_with_probs.txt");
          //strcat(scorefile, motiffile);
          //strcat(scorefile, "_with_probs.txt");
        }
        
      }
      tmp_matrix = gsl_matrix_alloc(hmmList[index].M, totalT);
        
      if (gflg) CalMotifScore_P_va(&(hmmList[index]), tmp_matrix, O1, O2, P, peakPos);
      else CalMotifScore_P(&(hmmList[index]), tmp_matrix, O1, P, peakPos);
      peakPos_i = ivector(P+1);
      posIndex_i = ivector(P);
      P_i = 0;
      totalT_i = 0;
      L_i = 0;
      tmp_vector = gsl_vector_alloc(totalT);
      gsl_matrix_get_row(tmp_vector, tmp_matrix, 0);
      peakPos_i[0] = 1;
      j = 0;
    

      for (i = 0; i < P; i++){
        tmp_vector_i = gsl_vector_alloc(peakPos[i+1] - peakPos[i]);
        gsl_vector_get_2(tmp_vector, peakPos[i]-1, peakPos[i+1]-2, tmp_vector_i);
        if (gsl_vector_max(tmp_vector_i) > (hmmList[index].thresholds[0]-0.5)){
          peakPos_i[j+1] = peakPos_i[j] + (peakPos[i+1] - peakPos[i]);
          posIndex_i[j] = i;
          totalT_i = totalT_i + (peakPos[i+1] - peakPos[i]);
          if (fflg) L_i = L_i + (motifIndex[i+1] - motifIndex[i]);
          j ++;
          P_i ++;
          
        }
        gsl_vector_free(tmp_vector_i);
      }
      gsl_vector_free(tmp_vector);
      printf("P: %d L: %d T: %d\n", P_i, L_i, totalT_i);
      
      O1_i = ivector(totalT_i);
      chr_i = malloc(P_i * sizeof(char *));
      peakStart_i = ivector(P_i);
      peakEnd_i = ivector(P_i);
      
      counts_vector_i = gsl_vector_alloc(totalT_i);
      slop_vector_i = gsl_vector_alloc(totalT_i);
      obs_matrix_i = gsl_matrix_alloc(hmmList[index].K, totalT_i);
    
      for (i = 0; i < P_i; i++){
        chr_i[i] = malloc(BUFSIZE);
        chr_i[i][0] = '\0';
        strcat(chr_i[i], chr[posIndex_i[i]]);
        peakStart_i[i] = peakStart[posIndex_i[i]];
        peakEnd_i[i] = peakEnd[posIndex_i[i]];
        for (j = peakPos[posIndex_i[i]], n = 0; j < peakPos[posIndex_i[i]+1]; j++, n++){
        //O1_i[peakPos_i[i]+n-1] = O1[j-1];
        //gsl_vector_set(counts_vector_i, peakPos_i[i]+n-1, gsl_vector_get(counts_vector, j-1));
        //gsl_vector_set(slop_vector_i, peakPos_i[i]+n-1, gsl_vector_get(slop_vector, j-1));
          for (m = 0; m < hmmList[index].M; m++){
            gsl_matrix_set(obs_matrix_i, m, peakPos_i[i]+n-1, gsl_matrix_get(tmp_matrix, m, j-1));
          }
          gsl_matrix_set(obs_matrix_i, hmmList[index].M, peakPos_i[i]+n-1, gsl_vector_get(slop_vector, j-1));
          gsl_matrix_set(obs_matrix_i, hmmList[index].M+1, peakPos_i[i]+n-1, gsl_vector_get(counts_vector, j-1));
        }
      }
      //printf("P: %d L: %d T: %d\n", P_i, L_i, totalT_i);
     // fflush(stdout);
      if (fflg){
        TFstart_i = ivector(L_i);
        TFend_i = ivector(L_i);
        TFlength_i = ivector(L_i);
        motifIndex_i = ivector(P_i + 1);
        motifIndex_i[0] = 0;
        for (i = 0; i < P_i; i++){
          for (j = motifIndex[posIndex_i[i]], n = 0; j < motifIndex[posIndex_i[i]+1]; j++, n++){
            TFstart_i[motifIndex_i[i]+n] = TFstart[j];
            TFend_i[motifIndex_i[i]+n] = TFend[j];
            TFlength_i[motifIndex_i[i]+n] = TFlength[j];
          }
          motifIndex_i[i+1] = motifIndex_i[i] + n;
        }
      }

      gsl_matrix_free(tmp_matrix);
      //printf("%d %f %f %f %f %f %f %f\n", totalT_i, gsl_matrix_get(obs_matrix_i, 0, totalT_i-200), gsl_matrix_get(obs_matrix_i, 0, totalT_i-2), gsl_matrix_get(obs_matrix_i, 0, totalT_i-1), gsl_matrix_get(obs_matrix_i, 0, 200), gsl_matrix_get(obs_matrix_i, hmmList[index].M, totalT_i-1), gsl_matrix_get(obs_matrix_i, hmmList[index].M, 200), gsl_matrix_get(obs_matrix_i, hmmList[index].M+1, totalT_i-1));
        //fflush(stdout);
        
      //fp = fopen(peakfile, "r");
      
      /*                                                */
      /* partition genome, depending on number of peaks */
      /*                                                */
        if (GPU){
          checkFile(predfile, "w");
          fp1 = fopen(predfile, "w");
          checkFile(scorefile, "w");
          fp2 = fopen(scorefile, "w");
          char **chr_tmp;
          int *peakPos_tmp;
          int peakIndex, peakNum;
          int *peakStart_tmp, *peakEnd_tmp;
          int *TFstart_tmp, *TFend_tmp, *TFlength_tmp, *motifIndex_tmp;
          gsl_matrix * emission_matrix;
          fprintf(stdout,"decoding... scanning peak file and calculating posterior probabilities for all positions\n");
          start0 = GetTimeStamp();
          for (peakIndex = 0; peakIndex < P_i; peakIndex += peakPer){
            PosIndex = peakPos_i[peakIndex] - 1;
            peakNum = MIN(peakPer, P_i-peakIndex);
            fprintf(stdout,"viterbi start: %d %d\n", peakIndex, peakNum);

            T = peakPos_i[peakIndex+peakNum] - peakPos_i[peakIndex];
            peakPos_tmp = subivector(peakPos_i,P_i+1,peakIndex,peakIndex+peakNum);
    
            for (i = peakNum; i >= 0; i--){
              peakPos_tmp[i] = peakPos_tmp[i] - peakPos_tmp[0] + 1;
            }
            chr_tmp = malloc(peakNum * sizeof(char *));
            for (i = 0; i < peakNum; i++){
              chr_tmp[i] = malloc(BUFSIZE);
              chr_tmp[i][0] = '\0';
              strcat(chr_tmp[i], chr_i[peakIndex+i]);
            }
            peakStart_tmp = subivector(peakStart_i,P_i,peakIndex,peakIndex+peakNum-1);
            peakEnd_tmp = subivector(peakEnd_i,P_i,peakIndex,peakIndex+peakNum-1);
            if (fflg){
              TFstart_tmp = subivector(TFstart_i,L_i,motifIndex_i[peakIndex],motifIndex_i[peakIndex+peakNum]);
              TFend_tmp = subivector(TFend_i,L_i,motifIndex_i[peakIndex],motifIndex_i[peakIndex+peakNum]);
              TFlength_tmp = subivector(TFlength_i,L_i,motifIndex_i[peakIndex],motifIndex_i[peakIndex+peakNum]);
              motifIndex_tmp = subivector(motifIndex_i,P_i+1, peakIndex,peakIndex+peakNum);
              for (i = peakNum; i >= 0; i--){
                motifIndex_tmp[i] = motifIndex_tmp[i] - motifIndex_tmp[0];
              }
            }
            
           /* Put PWM scores, counts and slopes into a matrix */
            /* Calculate PWM scores for each motif at each position */
            obs_matrix = gsl_matrix_alloc(hmmList[index].K, T);
            tmp_vector = gsl_vector_alloc(T);
            tmp_vector_i = gsl_vector_alloc(totalT_i);
            pwm_matrix = gsl_matrix_alloc(hmmList[index].M, T);
            for (i = 0; i < hmmList[index].M; i++) {
                gsl_matrix_get_row(tmp_vector_i, obs_matrix_i, i);
                gsl_vector_get_2(tmp_vector_i, PosIndex, PosIndex+T-1, tmp_vector);
                gsl_matrix_set_row(obs_matrix, i, tmp_vector);
                gsl_matrix_set_row(pwm_matrix, i, tmp_vector);
            }
            for (i = hmmList[index].M; i < hmmList[index].K; i++) {
                gsl_matrix_get_row(tmp_vector_i, obs_matrix_i, i);
                gsl_vector_get_2(tmp_vector_i, PosIndex, PosIndex+T-1, tmp_vector);
                gsl_matrix_set_row(obs_matrix, i, tmp_vector);
            }
    
            gsl_vector_free(tmp_vector);
            gsl_vector_free(tmp_vector_i);
            //printf("%s %d: %d %d %d %d %d %d data: %f %f %f %f %f %f %f\n", chr_tmp[peakNum-1], peakIndex, peakNum, PosIndex, PosIndex+T-1, peakPos_tmp[0], peakPos_tmp[peakNum],T, gsl_matrix_get(obs_matrix, 0, T-200), gsl_matrix_get(obs_matrix, 0, T-2), gsl_matrix_get(obs_matrix, 0, T-1), gsl_matrix_get(obs_matrix, 0, 200), gsl_matrix_get(obs_matrix, hmmList[index].M, T-1), gsl_matrix_get(obs_matrix, hmmList[index].M, 200), gsl_matrix_get(obs_matrix, hmmList[index].M+1, T-1));
      
            if (!ifSkip){
            printf("GPU version is not available for training step, please set -V to do decoding\n");
            /* Set the initial mean parameters of PWM score feature based on max and min of actual calculation */
              int h_index = 0;
              tmp_vector = gsl_vector_alloc(T);
              for (i = 0; i < hmmList[h_index].M; i++) {
                gsl_matrix_get_row(tmp_vector, pwm_matrix, i);
                for (j = 0; j < hmmList[h_index].N; j++) {
                  gsl_matrix_set(hmmList[h_index].mean_matrix, i, j, gsl_vector_min(tmp_vector) / 6.0);
                  gsl_matrix_set(hmmList[h_index].var_matrix, i, j, 8.0);
                }
                /* For each binding site state, the initial mean parameter for
                 correspondong PWM score will be 1/2 of its greatest score */
                for (j = index; j < index + hmmList[h_index].D[i] * (hmmList[index].inactive + 1); j++) {
                  gsl_matrix_set(hmmList[h_index].mean_matrix, i, j, gsl_vector_max(tmp_vector) / 2.0);
                }
                index += hmmList[h_index].D[i] * (hmmList[h_index].inactive+1);
              }
              gsl_vector_free(tmp_vector);
              /*                     */
              /* Start training step */
              /*                     */
                /* matrix of alpha, beta and gamma in BW and viterbi algorithm */
                double **alpha = dmatrix(hmmList[index].N, T);
                double **beta = dmatrix(hmmList[index].N, T);
                double **gamma = dmatrix(hmmList[index].N, T);
                double *logprobf = dvector(peakNum); /* vector containing log likelihood for each peak */
                emission_matrix = gsl_matrix_alloc(hmmList[index].N, T); /* matrix of emission probabilities */
                /* BW algorithm */
                BaumWelch(&hmmList[index], T, obs_matrix, &niter, peakNum, peakPos_tmp, logprobf, alpha, beta, gamma, emission_matrix);
                fp2 = fopen(outfile, "w");
                /* Print the final model */
                PrintHMM(fp2, &hmmList[index]);
                fclose(fp2);
                free_dvector(logprobf, peakNum);
                free_dmatrix(gamma, hmmList[index].N, T);
                free_dmatrix(alpha, hmmList[index].N, T);
                free_dmatrix(beta, hmmList[index].N, T);
                gsl_matrix_free(emission_matrix);
            }
            else {
              //start = GetTimeStamp();
              emission_matrix = gsl_matrix_alloc(hmmList[index].N, T); /* matrix of emission probabilities */
              if (hmmList[index].model == 0) EmissionMatrix(&hmmList[index], obs_matrix, peakNum, peakPos_tmp, emission_matrix, T);
              if (hmmList[index].model == 1) EmissionMatrix_mv(&hmmList[index], obs_matrix, peakNum, peakPos_tmp, emission_matrix, T);
              if (hmmList[index].model == 2) EmissionMatrix_mv_reduce(&hmmList[index], obs_matrix, peakNum, peakPos_tmp, emission_matrix, T);
              end = GetTimeStamp();
              //printf("GPU Emission Time: %ld\n", end - start);
              //printf("emission_matrix: %f %f %f %f %f %f\n", gsl_matrix_get(emission_matrix, 0, T-1), gsl_matrix_get(emission_matrix, 0, 200), gsl_matrix_get(emission_matrix, hmmList[index].N-2, T-1), gsl_matrix_get(emission_matrix, hmmList[index].N-2, 200),gsl_matrix_get(emission_matrix, hmmList[index].N-1, T-1),gsl_matrix_get(emission_matrix, hmmList[index].N-1, 200));
      
              //start = GetTimeStamp();
              fwd_cuda(fp1,fp2, hmmList[index].pi, hmmList[index].thresholds, hmmList[index].D, hmmList[index].N, hmmList[index].extraState, T, peakNum, peakPos_tmp, emission_matrix, hmmList[index].log_A_matrix, pwm_matrix, chr_tmp, peakStart_tmp, peakEnd_tmp,TFstart_tmp, TFend_tmp, TFlength_tmp, motifIndex_tmp);
              end = GetTimeStamp();
              //printf("GPU viterbi Time: %ld\n", end - start);
              gsl_matrix_free(emission_matrix);
            }
            for (i = 0; i < peakNum; i++) free(chr_tmp[i]);
            free(chr_tmp);
            gsl_matrix_free(obs_matrix);
            free_ivector(peakPos_tmp, peakNum+1);
            free_ivector(peakStart_tmp, peakNum);
            free_ivector(peakEnd_tmp, peakNum);
            if (fflg){
              free_ivector(TFstart_tmp, motifIndex_i[peakIndex+peakNum]-motifIndex_i[peakIndex]+1);
              free_ivector(TFend_tmp, motifIndex_i[peakIndex+peakNum]-motifIndex_i[peakIndex]+1);
              free_ivector(TFlength_tmp, motifIndex_i[peakIndex+peakNum]-motifIndex_i[peakIndex]+1);
              free_ivector(motifIndex_tmp, peakIndex+peakNum-peakIndex+1);
            }
            if (hmm.M > 0){
              gsl_matrix_free(pwm_matrix);
            }
            fprintf(stdout,"peak %d finished\n", peakIndex+peakNum);
          }
          //fclose(fp);
          fclose(fp1);
          fclose(fp2);
          printf("GPU Time: %ld\n", (GetTimeStamp() - start0));
        }
    
        
        if ((!GPU) && (pflg || fflg)){
          fprintf(stdout,"decoding... scanning peak file and calculating posterior probabilities for all positions (CPU)\n");
          //start = GetTimeStamp();
          /* matrix of alpha, beta and gamma in viterbi algorithm */
          double **alpha = dmatrix(hmmList[index].N, totalT_i);
          double **beta = dmatrix(hmmList[index].N, totalT_i);
          double **gamma = dmatrix(hmmList[index].N, totalT_i);
          double *logprobf = dvector(P_i); /* vector containing log likelihood for each peak */
          gsl_matrix * emission_matrix = gsl_matrix_alloc(hmmList[index].N, totalT_i); /* matrix of emission probabilities */
          start = GetTimeStamp();
          if (hmmList[index].model == 2) EmissionMatrix_mv_reduce(&hmmList[index], obs_matrix_i, P_i, peakPos_i, emission_matrix, totalT_i);
          //printf("CPU Emission Time: %ld\n", (GetTimeStamp() - start));
          //printf("emission_matrix: %f %f %f %f %f %f\n", gsl_matrix_get(emission_matrix, 0, totalT_i-1), gsl_matrix_get(emission_matrix, 0, 200), gsl_matrix_get(emission_matrix, hmmList[index].N-2, totalT_i-1), gsl_matrix_get(emission_matrix, hmmList[index].N-2, 200),gsl_matrix_get(emission_matrix, hmmList[index].N-1, totalT_i-1),gsl_matrix_get(emission_matrix, hmmList[index].N-1, 200));
    
        /*                     */
        /* Start decoding step */
        /*                     */
          int *q = ivector(totalT_i); /* state sequence q[1..T] */
          int **psi = imatrix(totalT_i, hmmList[index].N);
          double *g = dvector(totalT_i);
          double *vprob = dvector(totalT_i);
          double **delta = dmatrix(totalT_i, hmmList[index].N);
          double *logproba = dvector(P_i);
          double  **posterior = dmatrix(totalT_i, hmmList[index].N);
          start = GetTimeStamp();
          Forward_P(&hmmList[index], totalT_i, alpha, logprobf, P_i, peakPos_i, emission_matrix);
          Backward_P(&hmmList[index], totalT_i, beta, P_i, peakPos_i, emission_matrix);
          Viterbi(&hmmList[index], totalT_i, g, alpha, beta, gamma, logprobf, delta, psi, q,
            vprob, logproba, posterior, P_i, peakPos_i, emission_matrix, obs_matrix_i);
          free_dmatrix(delta, totalT_i, hmmList[index].N);
          free_dmatrix(gamma, hmmList[index].N, totalT_i);
          free_dmatrix(alpha, hmmList[index].N, totalT_i);
          free_dmatrix(beta, hmmList[index].N, totalT_i);
          free_dvector(logproba, P_i);
          free_ivector(g, totalT_i);
          free_dvector(logprobf, P_i);
          free_imatrix(psi, totalT_i, hmmList[index].N);
          free_dvector(vprob, totalT_i);
          gsl_matrix_free(emission_matrix);
          if (fflg){
            strcat(scorefile, "_CPU.txt");
            fp1 = fopen(scorefile, "w");
            getPosterior_motif(fp1, totalT_i, peakPos_i, P_i, posterior, &hmmList[index], chr_i, peakStart_i, peakEnd_i,TFstart_i, TFend_i, TFlength_i, motifIndex_i);
            fclose(fp1);
          }
          if (pflg){
            strcat(predfile, "_CPU.txt");
            fp1 = fopen(predfile, "w");
            getPosterior_all(fp1, totalT_i, q, peakPos_i, P_i, posterior, &hmmList[index], chr_i, peakStart_i, peakEnd_i);
            fclose(fp1);
          }
          
          printf("CPU viterbi Time: %ld\n", (GetTimeStamp() - start));
          free_ivector(q, totalT_i);
          free_dmatrix(posterior, totalT_i, hmmList[index].N);
        }
        FreeHMM(&(hmmList[index]));
        if(fflg){
          free_ivector(TFstart_i, L_i);
          free_ivector(TFend_i, L_i);
          free_ivector(TFlength_i, L_i);
          free_ivector(motifIndex_i, P_i + 1);
        }
        gsl_matrix_free(obs_matrix_i);
        free_ivector(O1_i, totalT_i);
        free_ivector(peakPos_i, P_i+1);
        gsl_vector_free(slop_vector_i);
        gsl_vector_free(counts_vector_i);
        free_ivector(peakStart_i, P_i);
        free_ivector(peakEnd_i, P_i);
        free_ivector(posIndex_i, P);
        free_ivector(motifIndex, P);
        free_ivector(TFstart, L);
        free_ivector(TFend, L);
        free_ivector(TFlength, L);
        //free(hmminitfile);
        free(motiffile);
        //free(outfile);
        free(scorefile);
        free(predfile);
        for (i = 0; i < P_i; i++) free(chr_i[i]);
        free(chr_i);
        //hmminitfile = (char *)malloc(BUFSIZ*sizeof(char));
        motiffile = (char *)malloc(BUFSIZ*sizeof(char));
        //outfile = (char *)malloc(BUFSIZ*sizeof(char));
        scorefile = (char *)malloc(BUFSIZ*sizeof(char));
        predfile = (char *)malloc(BUFSIZ*sizeof(char));

    }
  }
  //FreeHMM(&hmm);
  free(hmminitfile);
  free(motiffile);
  free(outfile);
  free(scorefile);
  free(predfile);
  free_ivector(O1, totalT);
  free_ivector(peakPos, P+1);
  gsl_vector_free(slop_vector);
  gsl_vector_free(counts_vector);
  for (i = 0; i < P; i++) free(chr[i]);
  //for (i = 0; i < nTF; i++) {
  //  if(hmmList[i].N) FreeHMM(&hmmList[i]);
 // }
  //free(hmmList);
}


void Usage(char *name)
{
  printf("Usage error. \n");
  printf("Usage1: %s [-v] ./TRACE <seq.file> <counts.file> <slope.file> "
         "--initial-model <init.model.file> --final-model <final.model.file> "
         "-peak-file <peak_3.file> --motif-file <peak_7.file>  --thread <thread.num>\n", name);
  printf("Usage2: %s [-v] ./TRACE --viterbi <seq.file> <counts.file> <slope.file> "
         "--final-model <final.model.file> -peak-file <peak_3.file> "
         "--motif-file <peak_7.file>  -T <thread.num>\n", name);
  printf("  seq.file - file containing the obs. sequence\n");
  printf("  count.file - file containing the obs. tag counts\n");
  printf("  slope.file - file containing the obs. slope\n");
  printf("  init.model.file - file with the initial model parameters\n");
  printf("  peak.file - file containing regions to detect TFBSs\n");
  printf("  final.model.file - output file containing the learned HMM\n");
}
