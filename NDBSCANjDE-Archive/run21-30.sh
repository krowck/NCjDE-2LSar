#!/bin/bash
declare -a func=(21 30)
for j in {0..10}
do
	python3 ndbjde.py -acc 0.001 -a 1 -flag 0 -r 5 -p 150 -f "${func[$j]}" -hj 1 -nm 1 &
done
