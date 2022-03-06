#!/bin/bash
#SBATCH -A irel
#SBATCH --gres gpu:3
#SBATCH -c 30

python3 train.py \
--train_path 'data/train.csv' \
--val_path 'data/val.csv' \
--test_path 'data/test.csv' \
--tokenizer_name_or_path 't5-base' \
--max_source_length 128 \
--max_target_length 128 \
--train_batch_size 4 \
--val_batch_size 4 \
--test_batch_size 4 \
--model_name_or_path 't5-base' \
--learning_rate 3e-5 \
--eval_beams 4 \
--tgt_max_seq_len 128 \
--gpus 3 \
--max_epochs 5 \
--strategy 'ddp' \
--log_dir '/scratch/shivprasad.sagare/experiments' \
--project_name 'zen' \
--run_name 't5-base-ddp' 

