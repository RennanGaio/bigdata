#!/bin/bash

DATASET=$1
SRC_DIR=src
DATA_DIR=data
OUT_DIR=out/$DATASET
STRATEGIES="Only_0 All_median All_randon media"

for STRATEGIE in $STRATEGIES;
do
  python trabson_finale.py $STRATEGIE
done
     
