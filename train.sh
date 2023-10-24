#!/bin/bash

train_ID="Fine tuning by UBI dataset"

dataset_type="ubi"
datasets="../Data/train/"
validation_dataset="../Data/val/"
net="mb2-ssd-lite"
mb2_width_mult=1.0
lr=0.001
scheduler="cosine" 
t_max=100
pretrained_ssd="models/mb2-ssd-lite-net.pth"
batch_size=64
num_epochs=200

function train(){
    # echo $train_ID
    python train.py \
    --dataset_type $dataset_type \
    --datasets $datasets \
    --validation_dataset $validation_dataset \
    --net $net \
    --mb2_width_mult $mb2_width_mult \
    --lr $lr \
    --scheduler $scheduler \
    --t_max $t_max \
    --pretrained_ssd $pretrained_ssd \
    --batch_size $batch_size \
    --num_epochs $num_epochs
}

train