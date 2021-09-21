# TRACE_GPU
Accelerate TRACE algorithm with GPU computing.

GPU computing is used for part of the emission matrix calculation and the viterbi decoding step in TRACE.

## Installation 
Create and activate conda environment for TRACE from ```environment_3.8.yml``` file:
```
$ conda env create -f environment_3.8.yml
$ source activate TRACE_env_3.8
```
**Note:** The lastest version of conda might not work here, we used **conda 4.3.30** on our machine.

The GNU Scientific Library (GSL) is required, you will need to update the GSL lib path and conda environment path in the first two lines in ```Makefile```. 

We used ```nvcc``` from **cuda 10.1** for compiling, and NVIDIA TITAN V for GPU computing. The ```-arch``` in ```Makefile``` needs to match with the GPU card you are using. More info can be found here: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/.

You might also need to add conda env lib path to ```LD_LIBRARY_PATH``` in your environment:
```
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shengchd/.conda/envs/TRACE_env_3.8/lib/
```
Build TRACE_gpu:
```
$ make
```

## Usage
For each DNase-seq data from ENCODE, we used the default peak bed file in hg38, and their corresponding bam file.
We first spanned all peaks with nearby 100 bp and then merged the overlapping ones:
```
bedtools slop -i ENCFF588OCA.bed -g hg38.chrom.sizes -b 100 | bedtools merge -i stdin
```
We then separated the spanned peak file by chromosomes (e.g. ```test_chr21/chr21.bed```).

There are three main steps in ```TRACE_gpu```:

1. Data preprocessing on the DNase-seq data, which processes the signal and generates the features required for TRACE. It takes ```-T``` to run in parallel on CPUs. If setting ```--prefix```, it will generate three intermediate files containing the features (i.e. ```test_chr21_seq.txt```, ```test_chr21_count.txt```, ```test_chr21_slope_2.txt```). Those features are required and same for each TF. 
```
$ ./TRACE_gpu -T 20 --bam-file ENCFF577DUO.bam --peak-file test_chr21/chr21.bed \
--fasta-file hg38.fa --size-file hg38.chrom.sizes --prefix test_chr21
```

Required input: 
* ```<bam.file>```: bam file from DNase-seq, its indexed .bai file is also required to be in the same location
* ```<bed.file>```: peak file called from the input bam file
* ```<fasta.file>```: genome sequence file
* ```<size.file>```: genome size file

Required files:
* ```<table.file>```: two tables used for correct bias in data preprocessing step (currently set as located under ```../data/single_hit_bias_table_*.txt```, can change in ```biasCorrection.py```)

Output: ```test_chr21_seq.txt```,```test_chr21_count.txt```, ```test_chr21_slope_2.txt```

**Note:** This step can be running separately from the following two steps since it only requires CPUs, but needs some modifications on the current script to skip the following steps and only generate the three intermediate files (ends at line 385 of ```src/esthmm_gpu.c```). 

2. CPU preprocess before GPU computing. This step mainly calculates the features unique for each TF (i.e. pwm scores), and stores the features in internal data arrays to copy to GPU in step 3. It also takes ```-T``` to run in parallel on CPUs. 

3. GPU computing. This step calculates the emission matrix and runs the viterbi decoding step in TRACE (needs to put ```--viterbi``` in command). It processes ```-X``` peaks at maximum in each round to fit with the GPU memory. (Our GPU has a 12G memory size, ```-X 1500``` was used.)


step 2-3 are iterated over every TF internally in  ```src/esthmm_gpu.c```.

## Demo
* To run TRACE_gpu from begining with input bam and bed files (no intermediate files will be output in this case without ```--prefix```):
```
$ ./TRACE_gpu --bam-file ENCFF577DUO.bam \ 
--viterbi -T 20 -X 1500 --peak-file test_chr21/chr21.bed --chr chr21
```
Output: ```results/*_chr21_viterbi_results.txt```

* To run TRACE_gpu on step 2-3 with input feature files:
```
$ ./TRACE_gpu test_chr21/test_chr21_seq.txt test_chr21/test_chr21_count.txt test_chr21/test_chr21_slope_2.txt \
--viterbi -T 20 -X 1500 --peak-file test_chr21/chr21.bed --chr chr21
```
Output: ```results/*_chr21_viterbi_results.txt```

**Note:**

* ```--chr``` is the chromosome name used in the output file names.

* The TF list is in ```data/TF_list_2020.txt```. In the main function in ```src/esthmm_gpu.c```, it will iterate over the first N TFs in the TF list (i.e. ```nTF``` in line 72 of ```src/esthmm_gpu.c``` ), TFs without available model files under ```data/model_file``` will be skipped. ```nTF``` is currently set as 3 for a quick test, you should get 2 output files under ```results/```. To iterate over all TFs, set ```nTF``` as **746**, you should then get 666 output files.

* The 2 example output files are under ```results/example_test_chr21```. The running time for each step on our machine can be found in ```results/example_test_chr21/test_chr21.log``` (in microseconds if with no unit).
