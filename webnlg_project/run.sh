#!/bin/bash
filepath=$(realpath "$0")
dir=$(dirname "$filepath")

model_dir=experiments/T5
data_dir=data

python $dir/train.py --model_dir $model_dir --data_dir $data_dir

