#!/bin/bash
make clean
make
./unit_test > now.txt
python3 plot_gflops.py
