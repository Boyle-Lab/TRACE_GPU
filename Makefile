#PY = $(/data/software/Miniconda3/4.3.30/lib)
#PY = /home/nouyang/.conda/envs/TRACE_env
#PYLIB = $(PY)/lib/libpython3.7m.a
PY = /home/shengchd/.conda/envs/TRACE_env_3.8
#PYLIB = /home/nouyang/.conda/pkgs/python-3.8.2-h191fe78_0/lib/libpython3.8.a
PYINC = -I$(PY)/include -I$(PY)/lib -I$(PY)/bin -I$(PY) -I./ -I$(PY)/include/python3.8 \
        -I/data/software/GSL/2.1-foss-2017a/include -I/data/software/GSL/2.1-foss-2017a/lib
LIBS = -L/usr/lib -L./ -L/$(PY)/lib/ -L/$(PY)/lib/pkgconfig/ \
       -L/data/software/GSL/2.1-foss-2017a/lib \
	   -lX11 -ldl -lrt -lutil -lquadmath -lcrypt \
       -Xcompiler -fopenmp -lgsl -lgslcblas -lm\
       -lpython3.8 $(PY)/lib/libpython3.8.so

#CC=gcc
#-ccbin $(PY)/bin/x86_64-conda_cos6-linux-gnu-gcc
CC = /usr/local/cuda-10.1/bin/nvcc
CFLAGS = -rdc=true -arch=sm_70 -lstdc++

all:	TRACE_gpu

TRACE_gpu: src/esthmm_gpu.c src/dataprocessing.c src/BaumWelch.c src/hmmutils.c src/nrutil.c src/emutils_2.c src/fwd_bwd.c src/logmath.c src/fileutils.c src/viterbi.c src/TRACE_gpu_emission_viterbi.cu
	$(CC) $(CFLAGS) $(LIBS) $(PYLIB) $(PYINC) src/esthmm_gpu.c src/dataprocessing.c src/BaumWelch.c src/hmmutils.c src/nrutil.c src/emutils_2.c src/fwd_bwd.c src/logmath.c src/fileutils.c src/viterbi.c src/TRACE_gpu_emission_viterbi.cu -g -o TRACE_gpu

#TRACE_gpu_driver.o dataprocessing.o hmmutils.o nrutil.o emutils.o logmath.o fileutils.o TRACE_gpu_driver.o

#data_processing_toC: dataprocessing.o nrutil.o
#	$(CC) dataprocessing.o nrutil.o $(PYLIB) $(LIBS) -g -export-dynamic -o data_processing_toC

#dataprocessing.o: $(SRCS)
#	$(CC) $(SRCS) -c -g $(PYINC)

clean:
	rm -f *.o *.pyc core
	rm -r __pycache__/
