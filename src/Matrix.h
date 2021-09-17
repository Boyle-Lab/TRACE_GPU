/*
*  Matrix.h
*  pvalue
*  
*  Created by Jean-Stéphane Varré on 02/07/07.
*  Copyright 2007 LIFL-USTL-INRIA. All rights reserved.
*  Originally modified for the pytfmpval Python package
*
*  Modified for TRACE program
*
*/

#ifndef Matrix_h
#define Matrix_h

#include <stdio.h>
typedef struct {
  long int totalMapSize;
  long int totalOp;
  
  double ** mat; // the matrix as it is stored in the matrix file
  int length;
  double granularity; // the real granularity used, greater than 1
  long int ** matInt; // the discrete matrix with offset
  double errorMax;
  long int *offsets; // offset of each column
  long int offset; // sum of offsets
  long int *minScoreColumn; // min discrete score at each column
  long int *maxScoreColumn; // max discrete score at each column
  long int *sum;
  long int minScore;  // min total discrete score (normally 0)
  long int maxScore;  // max total discrete score
  long int scoreRange;  // score range = max - min + 1
  long int *bestScore;
  long int *worstScore;
  double background[4];
} Matrix;

#endif /* Matrix_h */
