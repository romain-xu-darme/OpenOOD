#!/bin/bash

python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/fnrd.yml \
configs/pipelines/train/train_fnrd.yml \
configs/preprocessors/base_preprocessor.yml \
--optimizer.num_epochs 1 \
--network.backbone.name resnet18_32x32 \
--network.backbone.pretrained true \
--merge_option merge \
--network.backbone.checkpoint 'results/checkpoints/cifar100_res18_acc78.20.ckpt' \
