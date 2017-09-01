#!/bin/bash

DATASETS="nytimes" 
METHODS="aliasHDP"
NUM_ITER="1000"
NUM_TOPICS="1000"
NT="64"

for DATASET in $DATASETS
do
for METHOD in $METHODS
do
	#Create the directory structure
	time_stamp=`date "+%b_%d_%Y_%H.%M.%S"`
	DIR_NAME='out'/$DATASET/$METHOD/$time_stamp/
	mkdir -p $DIR_NAME

	#save details about experiments in an about file
	echo Running LDA inference using $METHOD | tee -a $DIR_NAME/log.txt
	echo For dataset $DATASET | tee -a $DIR_NAME/log.txt
	echo For number of iterations $NUM_ITER | tee -a $DIR_NAME/log.txt
	echo For number of topics $NUM_TOPICS | tee -a $DIR_NAME/log.txt
	echo with results being stored in $DIR_NAME
	echo Using $NT threads total on Amazon EC2 c4.8xlarge | tee -a $DIR_NAME/log.txt

	#run
	#valgrind --leak-check=full --show-leak-kinds=all 
	dist/hdp --method "$METHOD" --num-threads $NT --num-topics $NUM_TOPICS --num-iterations $NUM_ITER --output-state-interval 50 --output-model $DIR_NAME --num-top-words 15 --dataset data/"$DATASET" | tee -a $DIR_NAME/log.txt

	#git add "$DIR_NAME"
	#git pull
	#git commit -m "Experiments on Amazon EC2"
	#git push

done
done
echo 'done'

# dist/hdp --method stcHDP --num-threads 64 --num-topics 50 --num-iterations 100 --output-state-interval 50 --output-model out_1 --num-top-words 15 --dataset data/nips
