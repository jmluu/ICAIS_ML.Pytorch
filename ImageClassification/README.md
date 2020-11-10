# Configurations
## ImageNet 
**AlexNet:**
```
CUDA_VISIBLE_DEVICES=1 python imagenet.py \
	--data /imagenet-dir \
	--arch alexnet \
	--lr 0.01 --lr-mode step --lr-decay-period 40 \
	--epoch 160 --batch-size 256  -j 8 \
	--weight-decay 0.00005 
```


**ResNet-18:**  
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet.py \
	--data /imagenet-dir \
	--arch resnet18 \
	--lr 0.2 --lr-mode cosine --epoch 120 --batch-size 512  -j 32 \
	--warmup-epochs 5  --weight-decay 0.0001 


```
**ResNet-50:** 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet.py \
	--data /imagenet-dir \
	--arch resnet50 \
	--lr 0.2 --lr-mode cosine --epoch 120 --batch-size 512  -j 60 \
	--warmup-epochs 5  --weight-decay 0.0001 \
	--no-wd --label-smoothing --last-gamma

```

**MobileNet-V2-1.0:** 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet.py \
	--data /imagenet-dir \
	--arch mobilenet_v2 \
	--lr 0.05 --lr-mode cosine --epoch 150 --batch-size 256  -j 32 \
	--warmup-epochs 5  --weight-decay 0.00004 \
	--no-wd --label-smoothing 
```

## Cifar 