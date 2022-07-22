#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1081
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=/data/Datasets/OFA/caption_data/caption_test.tsv
# path=../../checkpoints/caption_large_best_clean.pt  # 149.5488
path=/media/hdd1/caption/large_s1_ofa_checkpoints/2_0.06_2500/checkpoint_2_6500.pt  # 2_6500: 141.6590
# path=/media/hdd1/caption/large_s2_ofa_checkpoints/8e-6_3/checkpoint_3_5000.pt  # 3_5000: 147.7489
# path=/media/hdd1/caption/large_s1_mofa_checkpoints/2_0.06_2500/checkpoint_2_5500.pt  # 2_6500: 142.2594
# path=/media/hdd1/caption/large_s2_mofa_checkpoints/8e-6_3/checkpoint_2_2500.pt  # 2_6500: 141.6590
result_path=../../results/caption
selected_cols=1,4,2
split='test'

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=caption \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=16 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"

python coco_eval.py ../../results/caption/test_predict.json /data/Datasets/OFA/caption_data/test_caption_coco_format.json
