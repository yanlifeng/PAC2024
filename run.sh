#!/bin/bash

make clean

make

time -p ./polynomial_stencil test.conf
