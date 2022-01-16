#!/bin/bash
#SBATCH -A research
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -n 10
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=END
#SBATCH -w gnode73


source /home2/shivprasad.sagare/miniconda3/etc/profile.d/conda.sh
conda activate py36

# module load cuda/10.1
mkdir -p /scratch/shivprasad.sagare
rm -rf /scratch/shivprasad.sagare/*
cp -r /home2/shivprasad.sagare/indic_wikibot/copernicus/stage-3/indicBART /scratch/shivprasad.sagare
cp -r /home2/shivprasad.sagare/indic_wikibot/copernicus/stage-3/data/* /scratch/shivprasad.sagare/indicBART
cd /scratch/shivprasad.sagare/indicBART

language_list=('hi' 'te' 'bn' 'pa' 'or' 'as' 'gu' 'mr' 'kn' 'ta' 'ml')

for lang in ${language_list[@]};
do
    fact_file=$(python -c "print('%s_facts_train.txt' % '$lang')")
    cp $fact_file train.en-$lang.en
    sent_file=$(python -c "print('%s_sentence_train.txt' % '$lang')")
    cp $sent_file train.en-$lang.$lang

    fact_file=$(python -c "print('%s_facts_val.txt' % '$lang')")
    cp $fact_file dev.en-$lang.en
    sent_file=$(python -c "print('%s_sentence_val.txt' % '$lang')")
    cp $sent_file dev.en-$lang.$lang
done
python /scratch/shivprasad.sagare/indicBART/yanmtt/train_nmt.py --train_slang en,en,en,en,en,en,en,en,en,en,en --train_tlang as,bn,gu,hi,kn,ml,mr,or,pa,ta,te \
    --dev_slang en,en,en,en,en,en,en,en,en,en,en --dev_tlang as,bn,gu,hi,kn,ml,mr,or,pa,ta,te --train_src train.en-as.en,train.en-bn.en,train.en-gu.en,train.en-hi.en,train.en-kn.en,train.en-ml.en,train.en-mr.en,train.en-or.en,train.en-pa.en,train.en-ta.en,train.en-te.en \
    --train_tgt train.en-as.as,train.en-bn.bn,train.en-gu.gu,train.en-hi.hi,train.en-kn.kn,train.en-ml.ml,train.en-mr.mr,train.en-or.or,train.en-pa.pa,train.en-ta.ta,train.en-te.te\
    --dev_src dev.en-as.en,dev.en-bn.en,dev.en-gu.en,dev.en-hi.en,dev.en-kn.en,dev.en-ml.en,dev.en-mr.en,dev.en-or.en,dev.en-pa.en,dev.en-ta.en,dev.en-te.en\
    --dev_tgt dev.en-as.as,dev.en-bn.bn,dev.en-gu.gu,dev.en-hi.hi,dev.en-kn.kn,dev.en-ml.ml,dev.en-mr.mr,dev.en-or.or,dev.en-pa.pa,dev.en-ta.ta,dev.en-te.te\
    --model_path model.ft --encoder_layers 6 --decoder_layers 6 --label_smoothing 0.1 \
    --dropout 0.1 --attention_dropout 0.1 --activation_dropout 0.1 --encoder_attention_heads 16 \
    --decoder_attention_heads 16 --encoder_ffn_dim 4096 --decoder_ffn_dim 4096 \
    --d_model 1024 --tokenizer_name_or_path albert-indicunified64k --warmup_steps 16000 \
    --weight_decay 0.00001 --lr 0.001 --max_gradient_clip_value 1.0 --batch_size 1 --batch_size_indicates_lines --dev_batch_size 1 \
    --port 22222 --shard_files --hard_truncate_length 256 --pretrained_model indicbart_model.ckpt --gpus 1 \
    --max_src_length 128 --max_tgt_length 128 &> log
