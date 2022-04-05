root_dir='basic_plm_finetuner'

mkdir -p $root_dir/{notebooks,data,model,experiments}

touch $root_dir/{train.py,evaluate.py,run.sh,model/{model.py,dataloader.py}}