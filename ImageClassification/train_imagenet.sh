#!/usr/bin/env bash 

set -e 

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet.py \
	--data $imagenet \
	--arch resnet18 \
	--lr 0.2 --lr-mode cosine --epoch 120 --batch-size 512  -j 30 \
	--warmup-epochs 5  \
	--weight-decay 0.0001
	
