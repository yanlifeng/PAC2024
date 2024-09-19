#!/bin/bash

make clean

make

#export OMP_NUM_THREADS=144
#export OMP_PLACES="{0:144},{144:144}"
#export OMP_PROC_BIND=spread
time -p ./polynomial_stencil test.conf
