#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

models=$base/models
configs=$base/configs

mkdir -p $models

num_threads=10
#device=0

# measure time

SECONDS=0

logs=$base/logs

model_name=deen_transformer_prenorm

mkdir -p $logs

mkdir -p $logs/$model_name

OMP_NUM_THREADS=$num_threads C:/Users/peapo/AppData/Local/Programs/Python/Python310/python.exe -m joeynmt train $configs/$model_name.yaml > $logs/$model_name/out 2> $logs/$model_name/err

echo "time taken:"
echo "$SECONDS seconds"
