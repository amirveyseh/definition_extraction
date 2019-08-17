#!/bin/bash

SAVE_ID=$1
CUDA_VISIBLE_DEVICES=1 python3 train.py --id $SAVE_ID --seed 0 --prune_k 1 --lr 0.03 --optim adamax --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --data_dir dataset/definition/textbook/withoutqualifier
