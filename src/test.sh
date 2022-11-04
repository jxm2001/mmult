#!/bin/bash
file="test_matrix_multiply.cpp"
allCase=$(ls mmult*)
for case in $allCase
do
	echo "#include <cstdio>" > $file
	echo "#include <cstdlib>" >> $file
	echo "#include <cstring>" >> $file
	echo "#include <iostream>" >> $file
	echo "#include \"MMult0.h\"" >> $file
	echo "#include \"dclock.h\"" >> $file
	echo "#include \"$case\"" >> $file
	cat test_matrix_multiply_gen.cpp >> $file
	make clean
	make -j
	./unit_test > "res/$case.txt"
	echo "$case over"
done
