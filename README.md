# TRACE_GPU
--------------
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

## Demo
To run TRACE_gpu on a test set from chr21:
```
$ ./TRACE_gpu test_chr21/test_chr21_seq.txt test_chr21/test_chr21_count.txt test_chr21/test_chr21_slope_2.txt --viterbi -T 20 -X 1500 --peak-file test_chr21/chr21.bed --chr chr21
```
Output: ```results/*_chr21_viterbi_results.txt```

**Note:**

* ```-T``` is the number of threads used for pwm calculation on CPU before GPU computing. 

* ```-X``` is the maximum number of input peaks for GPU computing to fit the limited memory on GPU. Our GPU has a 12G memory size, ```-X 1500``` was used. 

* The TF list is in ```data/TF_list_2020.txt```. In the main function in ```src/esthmm_gpu.c```, it will iterate over the first N TFs in the TF list (i.e. ```nTF``` in line 72 of ```src/esthmm_gpu.c``` ), TFs without available model files under ```data/model_file``` will be skipped. ```nTF``` is currently set as **3** for a quick test, you should get 2 output files under ```results/```.

* The example output files are under ```results/example_test_chr21```. The running time for each step on our machine can be found in ```results/example_test_chr21/test_chr21.log```.
