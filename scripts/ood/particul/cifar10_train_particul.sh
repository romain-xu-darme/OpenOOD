#!/bin/bash

python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/particul.yml \
configs/pipelines/train/train_particul.yml \
configs/preprocessors/base_preprocessor.yml \
--optimizer.num_epochs 30 \
--optimizer.lr 5.0e-4 \
--optimizer.weight_decay 1.0e-5 \
--network.backbone.name resnet18_32x32 \
--network.backbone.checkpoint 'results/checkpoints/cifar10_res18_acc94.30.ckpt' \
--network.num_patterns 2 \
--network.pretrained False \
--trainer.loc_ksize 1 \
--trainer.unq_ratio 1.0
