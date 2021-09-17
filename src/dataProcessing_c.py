#!/usr/bin/env python3

##-----------------------------------------------------------------------------##
##  Processes DNase-seq or ATAC-seq data, generate data files with required    ##
##  format for TRACE                                                           ##
##  modified for embedding in C                                                ##
##                                                                             ##
##-----------------------------------------------------------------------------##

import os
import argparse
import numpy as np
import pandas
import multiprocessing as mp
from pysam import AlignmentFile, Fastafile
from pybedtools import BedTool
from scipy.signal import savgol_filter
from scipy.stats import scoreatpercentile
from math import exp
from sklearn.preprocessing import scale
from rpy2.robjects import FloatVector, IntVector, globalenv
from rpy2.robjects.packages import importr
stats = importr('stats')
base = importr('base')

#Internal
from biasCorrection import bias_correction_dnase, bias_correction_atac, \
  bias_correction_atac_pe

def seq2int(seq):
  d = {"A": 0, "C": 1, "G": 2, "T": 3}
  seqInt = [d[s] if s in d else s for s in seq.upper()]
  return seqInt

# a generator to returns one header and sequence at a time
def read_fasta(fileObject):
  header = ''
  seq = ''
  # skip any useless leading information
  for line in fileObject:
    if line.startswith('>'):
      header = line.strip()
      break
  for line in fileObject:
    if line.startswith('>'):
      if 'N' in seq.upper():
        yield header, seq, False
      else:
        yield header, seq, True
      header = line.strip()
      seq = ''
    else:
      seq += line.strip()
  if header:
    if 'N' in seq.upper():
      yield header, seq, False
    else:
      yield header, seq, True

def read_seq(fileObject):
  header = ''
  seq = ''
  # skip any useless leading information
  for line in fileObject:
    seq = ''
    seq += line.strip()
  if 'N' in seq.upper():
    yield header, seq, False
  else:
    yield header, seq, True


def get_fasta_info(inFile):
  sumCG = 0
  sumAll = 0
  sequence = []
  pos = []
  pos.append(1)
  k = 1
  for header, seq, bool in read_fasta(inFile):
  #for header, seq, bool in read_seq(inFile):
    # keep adding C, G bases counted to get total CG number
    sumCG += seq.upper().count("C")
    sumCG += seq.upper().count("G")
    sumAll += len(seq)
    sequence += seq2int(seq.upper())
    pos.append(sumAll + 1)
    k += 1
  return (sumCG / sumAll), sequence, pos

def get_fasta_info_variants(inFile, vcf):
  sumCG = 0
  sumAll = 0
  sequence_ma = []
  sequence_pa = []
  pos = []
  pos.append(1)
  k = 1
  vcf_bed = vcf.intersect(inFile,wb=True)
  start=0
  #prev = int(a_with_b[0][11])
  loc_all=0
  for header, seq, bool in read_fasta(open(inFile.seqfn)):
    seq = seq.upper()
    seq_m=seq
    seq_p=seq
    #print(header)
    for i in range(start, len(vcf_bed)):
      loc = int(vcf_bed[i][1])-int(vcf_bed[i][11])-1
      if int(vcf_bed[i][11]) != header.split(":")[1].split("-")[0]:
        start = i
        loc_all=loc_all+len(seq)
        break
      #only consider SNPs now
      if len(vcf_bed[i][3]) > 1 or len(vcf_bed[i][4]) > 1:
        continue
      if seq[loc] != vcf_bed[i][3]:
        print("refference sequence and vcf don't match")
        exit(1)
      if vcf_bed[i][9].split('|')[0] == "1":
        seq_m = seq_m[0:loc] + vcf_bed[i][4] + seq_m[(loc+1):]
      if vcf_bed[i][9].split('|')[1] == "1":
        seq_p = seq_p[0:loc] + vcf_bed[i][4] + seq_p[(loc+1):]
    sumCG += (seq_m.count("C")+seq_p.count("C"))/2
    sumCG += (seq_m.count("G")+seq_p.count("G"))/2
    sumAll += len(seq)
    sequence_ma += seq2int(seq_m.upper())
    sequence_pa += seq2int(seq_p.upper())
    pos.append(sumAll + 1)
    k += 1
  return (sumCG / sumAll), sequence_ma, sequence_pa, pos

def get_fasta_info_variants_mp(inFile, vcf):
  sumCG = 0
  sumAll = 0
  sequence_ma = []
  sequence_pa = []
  pos = []
  pos.append(1)
  k = 1
  vcf_bed = vcf.intersect(inFile,wb=True)
  
  #prev = int(a_with_b[0][11])
  loc_all=0
  fasta_list = []
  for header, seq, bool in read_fasta(open(inFile.seqfn)):
    fasta_list.append([header, seq])
  seq = seq.upper()
  seq_m=seq
  seq_p=seq
  #print(header)
  for i in range(start, len(vcf_bed)):
    loc = int(vcf_bed[i][1])-int(vcf_bed[i][11])-1
    if int(vcf_bed[i][11]) != header.split(":")[1].split("-")[0]:
      start = i
      loc_all=loc_all+len(seq)
      break
    #only consider SNPs now
    if len(vcf_bed[i][3]) > 1 or len(vcf_bed[i][4]) > 1:
      continue
    if seq[loc] != vcf_bed[i][3]:
      print("refference sequence and vcf don't match")
      exit(1)
    if vcf_bed[i][9].split('|')[0] == "1":
      seq_m = seq_m[0:loc] + vcf_bed[i][4] + seq_m[(loc+1):]
    if vcf_bed[i][9].split('|')[1] == "1":
      seq_p = seq_p[0:loc] + vcf_bed[i][4] + seq_p[(loc+1):]
  sumCG += (seq_m.count("C")+seq_p.count("C"))/2
  sumCG += (seq_m.count("G")+seq_p.count("G"))/2
  sumAll += len(seq)
  sequence_ma += seq2int(seq_m.upper())
  sequence_pa += seq2int(seq_p.upper())
  pos.append(sumAll + 1)
  k += 1
  return (sumCG / sumAll), sequence_ma, sequence_pa, pos

def vcf_info(fasta_list):
  start=0

#sigmoid transformation#
def sigmoid(x):
  neg_x = [-i for i in x]
  return ((np.exp(x)-np.exp(neg_x))/(np.exp(x)+np.exp(neg_x))).tolist()

#loess function from R, get fitted value#
def loess_fromR(x, y, f, d = 2):
  x_vector = IntVector(x)
  y_vector = FloatVector(y)
  globalenv["x_vector"] = x_vector
  globalenv["y_vector"] = y_vector
  globalenv["f"] = f
  a = round(f, 2) if round(f, 2) > 0.0 else f
  model = stats.loess('y_vector~x_vector', span = a, degree = d)
  return model.rx2('fitted')

class Signal:
  def __init__(self, bamFile, bedFile, sizeFile, fastaFile):
    """
    Initializes Signal.
    """
    self.bam = None #AlignmentFile(bamFile, "rb")
    self.bam_file = bamFile #store file name to create an object in each process for multiprocessing
    self.bed = BedTool(bedFile)
    self.size = pandas.read_csv(sizeFile, sep='\t', header=None)
    self.fasta_file = fastaFile
    self.vcfFile = None
    self.counts = None
    self.loessSignal = None
    self.slope_2nd = None
    self.pos = None
    
  def init_output(self, T):
    self.loessSignal = [0.0 for x in range(T)]
    self.slope_2nd = [0.0 for x in range(T)]
    
  #get sequence information file in the correct format for TRACE#
  def load_sequence(self, outputFile = None):
    self.bed = self.bed.sequence(fi=self.fasta_file)
    GC, seq, self.pos = get_fasta_info(open(self.bed.seqfn))
    #fastaFile = Fastafile(fastaFile)
    #seq = []
    #for i in range(len(self.bed[0])):
     # seq.append(str(fastaFile.fetch(self.bed[0][i], self.bed[1][i], self.bed[2][i])))
    #GC, seq, pos = get_fasta_info(seq)
    if (outputFile):
      outFile = open(outputFile, 'w')
      print("T=", len(seq), "GC: ", (1.0 - GC) / 2.0, "\t", GC / 2.0, "\t",
          GC / 2.0, "\t", (1.0 - GC) / 2.0, "\n", file = outFile)
      for i in range(len(seq)):
        print(str(seq[i]), "\t", file = outFile)
      print("P= ", len(self.pos) - 1, "\n", file = outFile)
      for i in range(len(self.pos)):
        print(self.pos[i], "\n", file = outFile)
      outFile.close()
    return GC, seq
  
  def load_sequence_SNPs(self, outputFile = None):
    self.bed = self.bed.sequence(fi=self.fasta_file)
    vcf = BedTool(self.vcfFile)
    GC, seq_m, seq_p, self.pos = get_fasta_info_variants(self.bed, vcf)
    if (outputFile):
      outFile = open(outputFile, 'w')
      print("T=", len(seq_m), "GC: ", (1.0 - GC) / 2.0, "\t", GC / 2.0, "\t",
             GC / 2.0, "\t", (1.0 - GC) / 2.0, "\n", file = outFile)
      for i in range(len(seq_m)):
        print(seq_m[i], "|", seq_p[i], file = outFile)
      print("P= ", len(self.pos) - 1, "\n", file = outFile)
      for i in range(len(self.pos)):
        print(self.pos[i], "\n", file = outFile)
      outFile.close()
    return GC, seq_m, seq_p
   
  #count number of 5' end cut at each position#
  def count_read(self, region, ext_l, ext_r, shift = 0):
    reads = self.bam.fetch(reference=region[0], start=int(region[1]) - ext_l,
                           end=int(region[2]) + ext_r + 1)
    tagList = []
    for read in reads:
      if (read.is_reverse):
        tagList.append(int(read.reference_end - 1 - shift))
      else:
        tagList.append(int(read.reference_start + shift))
    list = range(int(region[1]) - ext_l + 1, int(region[2]) + ext_r + 1)
    counts = []
    for i in range(len(list)):
      counts.append(tagList.count(list[i]))
    return counts

  #normalized by 10kb surrounding window#
  def within_norm(self, counts):
    mu = np.mean(counts[np.nonzero(counts)])
    return counts / mu

  #normalized by std and percentile#
  def between_norm(self, counts, perc, std):
    list = [np.sign(x) * (1.0 / (1.0 + (exp(-(np.sign(x) * x - perc) / std)))) if x != 0.0 else 0.0 for x in counts]
    return list

  def get_slope(self, counts, window=9, derivative=1):
    return savgol_filter(np.array(counts), window, 2, deriv=derivative).tolist()


  def bias_correction(self, counts, region, ext_l, ext_r, forward_shift = 0,
                      reverse_shift = 0):
    signal = bias_correction_dnase(self, counts, region[0],
                                   int(region[1]) - ext_l,
                                   int(region[2]) + ext_r, forward_shift,
                                   reverse_shift)
    return signal

  def bias_correction_atac(self, counts, region, ext_l, ext_r,
                           forward_shift = 0, reverse_shift = 0):
    signal_f, signal_r = bias_correction_atac(self, counts, region[0],
                                              int(region[1]) - ext_l,
                                              int(region[2]) + ext_r,
                                              forward_shift, reverse_shift)
    return (np.array(signal_f)+np.array(signal_r)).tolist()

  def bias_correction_atac_pe(self, counts, region, ext_l, ext_r,
                              forward_shift = 0, reverse_shift = 0):
    signal = bias_correction_atac_pe(self, counts, region[0],
                                     int(region[1]) - ext_l,
                                     int(region[2]) + ext_r, forward_shift,
                                     reverse_shift)
    return signal

  def get_signal(self, span, is_atac, shift, i):
    counts_raw=[]
    loessSignal = []
    bc_normed = []
    bc_loess=[]
    slope_2nd = []
    slope_1st = []
    peak = self.bed[i]
    
    #for i in range(len(self.bed[0])):
     # peak = [self.bed[0][i], self.bed[1][i], self.bed[2][i]]
    maximum = int(self.size[1][self.size[0] == peak[0]])
    # Size of flanking region not exceeding genome size
    ext_l = 5000 if int(peak[1]) > 5000 else int(peak[1])
    ext_r = 5000 if int(peak[2]) + 5000 < maximum else maximum - int(peak[2])
    ext_l_50 = 50 if int(peak[1]) > 50 else int(peak[1])
    ext_r_50 = 50 if int(peak[2]) + 50 < maximum else maximum - int(peak[2])
    # Count aligned reads at each position
    counts = self.count_read(peak, ext_l, ext_r, shift)
    counts_raw += counts[(ext_l + 1):(len(counts) - ext_r + 1)]
    mean = np.array(counts).mean()
    std = np.array(counts).std()
    counts = [min(x, mean + 10 * std) for x in counts]
    # Bias correction
    if is_atac == "se":
      counts_bc = self.bias_correction_atac(counts, peak, ext_l, ext_r)
    elif is_atac == "pe":
      counts_bc = self.bias_correction_atac_pe(counts, peak, ext_l, ext_r,
                                                 shift, -shift)
    else:
      counts_bc = self.bias_correction(counts, peak, ext_l, ext_r)
      # Normalize read counts without bias correction by mean of surrounding 10kb
    counts = (self.within_norm(np.array(counts)).tolist())[(ext_l - ext_l_50):(len(counts) - ext_r + ext_r_50)]
      # Smooth read counts without bias correction by local regression from R
    smoothed = loess_fromR(range(len(counts)), counts, 600 * span/len(counts))
    loessSignal += smoothed[(ext_l_50 + 1):(len(smoothed) - ext_r_50 + 1)]

    # Normalize and smooth bias corrected read counts
    counts = self.within_norm(np.array(counts_bc)).tolist()
    perc = scoreatpercentile(np.array(counts), 98)
    std = np.array(counts).std()
    normedSignals = self.between_norm(counts[(ext_l - ext_l_50):(len(counts) - ext_r + ext_r_50)], perc, std)
    bc_normed += normedSignals[(ext_l_50 + 1):(len(normedSignals) - ext_r_50 + 1)]
    ext_l = ext_l_50
    ext_r = ext_r_50
    smoothedSignal = loess_fromR(range(len(normedSignals)), normedSignals,
                                   600 * span / len(normedSignals))
    bc_loess += smoothedSignal[(ext_l_50 + 1):(len(smoothedSignal) - ext_r_50 + 1)]
    # Get the first and second derivatives
    slope_2nd += self.get_slope(smoothedSignal, derivative=2)[
                   (ext_l + 1):(len(smoothedSignal) - ext_r + 1)]
    slope_1st += self.get_slope(smoothedSignal, derivative=1)[
                   (ext_l + 1):(len(smoothedSignal) - ext_r + 1)]

    slope_2nd = sigmoid([-x * 100 for x in slope_2nd])
    
    return loessSignal, slope_2nd#, slope_1st, bc_loess
    #prefix = "/home/nouyang/tmp_file/test"
    #with open(prefix + "_count.txt", "w") as outFile:
     # s = scale(loessSignal)
      #for i in range(len(loessSignal)):
       # print(s[i], file=outFile)
    #with open(prefix +'_slope_2.txt', "w") as outFile:
     # for i in range(len(slope_2nd)):
      #  print(slope_2nd[i], file=outFile)
    #with open(prefix +'_slope_1.txt', "w") as outFile:
     # for i in range(len(slope_1st)):
      #  print(slope_1st[i], file=outFile)

  def get_signal_update(self, span, is_atac, shift, i):
    self.bam = AlignmentFile(self.bam_file, "rb") #create an object in each process
    loessSignal, slope_2nd = self.get_signal(span, is_atac, shift, i)
    return [i, loessSignal, slope_2nd]
    
  def get_signal_all(self, span, is_atac, shift, nThread):
    GC, seq = self.load_sequence(outputFile = None)
    T = len(seq)
    P = len(self.pos) - 1
    self.init_output(T)
    pool = mp.Pool(processes=nThread)
    inputs = [(0.05, False, 0, x) for x in range(P)]
    results = pool.starmap(self.get_signal_update, inputs)
    for item in results:
     # print(self.pos[item[0]]-1, self.pos[item[0]+1]-1,item[0],
     #       len(self.loessSignal[(self.pos[item[0]]-1):(self.pos[item[0]+1]-1)]),len(item[1]),len(item[2]), T)
      self.loessSignal[(self.pos[item[0]]-1):(self.pos[item[0]+1]-1)] = item[1]
      self.slope_2nd[(self.pos[item[0]]-1):(self.pos[item[0]+1]-1)] = item[2]
    self.loessSignal = scale(self.loessSignal).tolist()
    return P, T, GC, self.pos, self.loessSignal, self.slope_2nd, seq

  def get_signal_SNP(self, vcfFile, span, is_atac, shift, T, nThread):
    self.vcfFile = vcfFile
    if (T):
      P = len(self.bed)
      self.pos = [1]
      for i in range(P):
        self.pos.append(int(self.bed[i][2])-int(self.bed[i][1])+self.pos[i])
      self.bed = self.bed.sequence(fi=self.fasta_file)
      
      GC=[]
      seq_m=[]
      seq_p=[]
    else:
      GC, seq_m, seq_p  = self.load_sequence_SNPs(outputFile = "/home/nouyang/p2C/test_seq_va_chr1.txt")
      T = len(seq_m)
      P = len(self.pos) - 1
    self.init_output(T)
    pool = mp.Pool(processes=nThread)
    inputs = [(0.05, False, 0, x) for x in range(P)]
    results = pool.starmap(self.get_signal_update, inputs)
    for item in results:
      #print(self.pos[item[0]]-1, self.pos[item[0]+1]-1,item[0],
       #     len(self.loessSignal[(self.pos[item[0]]-1):(self.pos[item[0]+1]-1)]),len(item[1]),len(item[2]), T)
      self.loessSignal[(self.pos[item[0]]-1):(self.pos[item[0]+1]-1)] = item[1]
      self.slope_2nd[(self.pos[item[0]]-1):(self.pos[item[0]+1]-1)] = item[2]
    self.loessSignal = scale(self.loessSignal).tolist()
    return P, T, GC, self.pos, self.loessSignal, self.slope_2nd, seq_m, seq_p
  
def main():
  parser = argparse.ArgumentParser()
  # Optional parameters
  parser.add_argument("--atac-seq", type=str, dest = "is_atac", default=False,
                      choices=["pe", "se"],
                      help="If set, ATAC-seq based data processing will be used. "
                           "choose between two types: pair-end(pe) and single-end(se). "
                           "DEFAULT: False")
  parser.add_argument("--span", type=float, dest="span", default=0.05,
                      help='span number for loess, DEFAULT: 0.05')
  parser.add_argument("--prefix", type=str, dest = "prefix",
                      default="TRACE",
                      help="The prefix for results files. DEFAULT: TRACE")
  parser.add_argument("--shift", type=int, dest = "shift", default=0,
                      help="Number of bases for reads to be shifted")
  parser.add_argument("--genome", type=str, dest = "genome", default="hg38",
                      help="Specify genome. Currently hg19 and hg38 are available. default:hg38")
  parser.add_argument("--VCF", type=str, dest="VCF", default=False,
                      help="VCF file")
  # Required input
  parser.add_argument(dest = "input_files", metavar="peak_3.file bam.file",
                      type=str, nargs='*', help='BED files of interesting regions, fasta.file'
                                                'BAM file of reads, sequence file')
  args = parser.parse_args()
  if (len(args.input_files) < 3):
    print("Error: missing requird files")
    parser.print_help()
    exit(1)

  #genome_size = os.path.dirname(__file__) + '/../data/' + args.genome + '.chrom.sizes'
  genome_size = "/data/genomes/hg38/hg38.chrom.sizes"
  signal = Signal(args.input_files[1], args.input_files[0], genome_size, args.input_files[2])
  # Generate sequence file
  # GC, seq = signal.load_sequence(outputFile = args.prefix + "_seq.txt")
  # Process DNase-seq or ATAC-seq data
  #loessSignal, deriv2nd, deriv1st, bc_loess = signal.get_signal(args.span, args.is_atac, args.shift, 0)
  signal.get_signal_all(args.span, args.is_atac, args.shift)
  # Generate count and slope files
  with open(args.prefix + "_count.txt", "w") as outFile:
    #s = scale(signal.loessSignal)
    for i in range(len(signal.loessSignal)):
      print(signal.loessSignal[i], file=outFile)
  with open(args.prefix +'_slope_2.txt', "w") as outFile:
    for i in range(len(signal.slope_2nd)):
      print(signal.slope_2nd[i], file=outFile)
  #with open(args.prefix +'_slope_1.txt', "w") as outFile:
  #  for i in range(len(deriv1st)):
    #  print(deriv1st[i], file=outFile)
  return


if __name__ == "__main__":
  main()
