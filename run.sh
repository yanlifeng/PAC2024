#!/bin/bash

make clean

make

export OMP_NUM_THREADS=576
export OMP_PLACES="{0:576}"
#export OMP_PROC_BIND=spread
#./polynomial_stencil ~/finaldata/Polynomial_Stencil/test2.conf
./polynomial_stencil test.conf
