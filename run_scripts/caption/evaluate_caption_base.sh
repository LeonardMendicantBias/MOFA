#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1091

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=/data/Datasets/OFA/caption_data/caption_test.tsv
# path=../../checkpoints/caption_base_best.pt  # 146.4

path=./base_s1_ofa_checkpoints/5_0.06_6000/checkpoint_4_16000.pt  # 5_20500: 138.1251
# path=./base_s1_ofa_checkpoints/5_0.06_6000/checkpoint_5_19000.pt  # 5_18000: 138.1058 | 5_21000: 137.6718
# path=./base_s2_ofa_checkpoints/8e-6_3/checkpoint_3_4000.pt  # 2_4000: 146.4353
# path=./base_s1_mofa_checkpoints/5_0.06_6000/checkpoint_5_21000.pt  # 5_21000: 138.3716
# path=./base_s2_mofa_checkpoints/8e-6_3/checkpoint_3_5000.pt  # 3_5000: 147.1244

# path=./medium_s1_ofa_checkpoints/5_0.06_6000/checkpoint_4_13500.pt  # 4_17500: 130.1408 
# path=./medium_s2_ofa_checkpoints/8e-6_3/checkpoint_500.pt  # 2_2500: 137.0154
# path=./medium_s1_mofa_checkpoints/5_0.06_6000/checkpoint_4_13500.pt  # 5_20500: 129.8560
# path=./medium_s2_mofa_checkpoints/8e-6_3/checkpoint_1_1500.pt  # 2_2500: 137.3044

result_path=../../results/caption
selected_cols=1,4,2
split='test'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=${MASTER_PORT} ../../evaluate.py \
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
