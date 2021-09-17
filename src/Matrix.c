/*
*  Matrix.cpp
*  pvalue
*
*  Created by Jean-Stéphane Varré on 02/07/07.
*  Copyright 2007 LIFL-USTL-INRIA. All rights reserved.
*  Originally modified for the pytfmpval Python package
*
*  Modified for TRACE program
*
*/

#include "Matrix.h"


void initMatrix(Matrix *pMatrix, double pA, double pC, double pG, double pT) {
  pMatrix->granularity = 1.0;
  pMatrix->offset = 0;
  pMatrix->background[0] = pA;
  pMatrix->background[1] = pC;
  pMatrix->background[2] = pG;
  pMatrix->background[3] = pT;
}


void toLog2OddRatio(Matrix *pMatrix) {
  for (int p = 0; p < pMatrix->length; p++) {
    for (int k = 0; k < 4; k++) {
      pMatrix->mat[k][p] = log2((pMatrix->mat[k][p]) - log2 (pMatrix->background[k]);
    }
  }
}

void computesIntegerMatrix (Matrix *pMatrix, double granularity) {
  double minS = 0, maxS = 0;
  double scoreRange;
  int length = pMatrix->lenght;
  // computes precision
  for (int i = 0; i < length; i++) {
    double min = pMatrix->mat[0][i];
    double max = min;
    for (int k = 1; k < 4; k++ )  {
      min = ((min < pMatrix->mat[k][i])?min:(pMatrix->mat[k][i]));
      max = ((max > pMatrix->mat[k][i])?max:(pMatrix->mat[k][i]));
    }
    minS += min;
    maxS += max;
  }
  
  // score range
  scoreRange = maxS - minS + 1;
  
  if (granularity > 1.0) {
    pMatrix->granularity = granularity / scoreRange;
  } else if (granularity < 1.0) {
    pMatrix->granularity = 1.0 / granularity;
  } else {
    pMatrix->granularity = 1.0;
  }
  
  long int *matInt = limatrix(4, length);
  for (int k = 0; k < 4; k++ ) {
    for (int p = 0 ; p < length; p++) {
      matInt[k][p] = ROUND_TO_INT((double)(mat[k][p]*pMatrix->granularity));
    }
  }
  
  pMatrix->errorMax = 0.0;
  for (int i = 1; i < length; i++) {
    double maxE = mat[0][i] * pMatrix->granularity - (matInt[0][i]);
    for (int k = 1; k < 4; k++) {
      maxE = ((maxE < mat[k][i] * pMatrix->granularity - matInt[k][i])?(pMatrix->mat[k][i] * pMatrix->granularity - (matInt[k][i])):(maxE));
    }
    pMatrix->errorMax += maxE;
  }
  
  if (sortColumns) {
    // sort the columns : the first column is the one with the greatest value
    long long min = 0;
    for (int i = 0; i < length; i++) {
      for (int k = 0; k < 4; k++) {
        min = MIN(min,matInt[k][i]);
      }
    }
    min --;
    long int maxs[length];
    for (int i = 0; i < length; i++) {
      maxs[i] = matInt[0][i];
      for (int k = 1; k < 4; k++) {
        if (maxs[i] < matInt[k][i]) {
          maxs[i] = matInt[k][i];
        }
      }
    }
    long int **mattemp = limatrix(4, length);
    for (int i = 0; i < length; i++) {
      long int max = maxs[0];
      int p = 0;
      for (int j = 1; j < length; j++) {
        if (max < maxs[j]) {
          max = maxs[j];
          p = j;
        }
      }
      maxs[p] = min;
      for (int k = 0; k < 4; k++) {
        mattemp[k][i] = matInt[k][p];
      }
    }
    
    for (int k = 0; k < 4; k++)  {
      for (int i = 0; i < length; i++) {
        matInt[k][i] = mattemp[k][i];
      }
    }

    for (int k = 0; k < 4; k++) {
      free(mattemp[k]);
    }
    free(mattemp);
    free(maxs);
  }
  
  // computes offsets
  pMatrix->offset = 0;
  pMatrix->offsets = livector(length); //
  for (int i = 0; i < length; i++) {
    long int min = matInt[0][i];
    for (int k = 1; k < 4; k++ )  {
      min = ((min < matInt[k][i])?min:(matInt[k][i]));
    }
    pMatrix->offsets[i] = -min;
    for (int k = 0; k < 4; k++ )  {
      matInt[k][i] += pMatrix->offsets[i];
    }
    pMatrix->offset += pMatrix->offsets[i];
  }
  
  // look for the minimum score of the matrix for each column
  pMatrix->minScoreColumn = livector(length);
  pMatrix->maxScoreColumn = livector(length);
  //sum            = new long long [length];
  pMatrix->minScore = 0;
  pMatrix->maxScore = 0;
  for (int i = 0; i < length; i++) {
    pMatrix->minScoreColumn[i] = matInt[0][i];
    pMatrix->maxScoreColumn[i] = matInt[0][i];
    //sum[i] = 0;
    for (int k = 1; k < 4; k++ )  {
      //sum[i] = sum[i] + matInt[k][i];
      if (pMatrix->minScoreColumn[i] > matInt[k][i]) {
        pMatrix->minScoreColumn[i] = matInt[k][i];
      }
      if (pMatrix->maxScoreColumn[i] < matInt[k][i]) {
        pMatrix->maxScoreColumn[i] = matInt[k][i];
      }
    }
    pMatrix->minScore = pMatrix->minScore + pMatrix->minScoreColumn[i];
    pMatrix->maxScore = pMatrix->maxScore + pMatrix->maxScoreColumn[i];
    //cout << "minScoreColumn[" << i << "] = " << minScoreColumn[i] << endl;
    //cout << "maxScoreColumn[" << i << "] = " << maxScoreColumn[i] << endl;
  }
  pMatrix->scoreRange = pMatrix->maxScore - pMatrix->minScore + 1;
  
  pMatrix->bestScore = livector(length);
  pMatrix->worstScore = livector(length);
  pMatrix->bestScore[length-1] = pMatrix->maxScore;
  pMatrix->worstScore[length-1] = pMatrix->minScore;
  for (int i = length - 2; i >= 0; i--) {
    pMatrix->bestScore[i]  = pMatrix->bestScore[i+1]  - pMatrix->maxScoreColumn[i+1];
    pMatrix->worstScore[i] = pMatrix->worstScore[i+1] - pMatrix->minScoreColumn[i+1];
  }
  
}




/**
* Computes the pvalue associated with the threshold score requestedScore.
 */
void Matrix::lookForPvalue (long int requestedScore, long int min, long int max, double *pmin, double *pmax) {
  
  map<long long, double> *nbocc = calcDistribWithMapMinMax(min,max);
  map<long long, double>::iterator iter;
  

  // computes p values and stores them in nbocc[length]
  double sum = nbocc[length][max+1];
  long long s = max + 1;
  map<long long, double>::reverse_iterator riter = nbocc[length-1].rbegin();
  while (riter != nbocc[length-1].rend()) {
    sum += riter->second;
    if (riter->first >= requestedScore) s = riter->first;
    nbocc[length][riter->first] = sum;
    riter++;
  }
  //cout << "   s found : " << s << endl;
  
  iter = nbocc[length].find(s);
  while (iter != nbocc[length].begin() && iter->first >= s - errorMax) {
    iter--;
  }
  //cout << "   s - E found : " << iter->first << endl;
  
#ifdef MEMORYCOUNT
  // for tests, store the number of memory bloc necessary
  for (int pos = 0; pos <= length; pos++) {
    totalMapSize += nbocc[pos].size();
  }
#endif
  
  *pmax = nbocc[length][s];
  *pmin = iter->second;

  delete[] nbocc;
  delete[] minScoreColumn;
  delete[] maxScoreColumn;
  delete[] bestScore;
  delete[] worstScore;
  delete[] offsets;
  for (int k = 0; k < 4; k++) {
      delete[] matInt[k];
    }
  delete[] matInt;
  
}

