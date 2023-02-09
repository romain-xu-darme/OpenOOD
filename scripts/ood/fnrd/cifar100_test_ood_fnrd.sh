#!/bin/bash

python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/fnrd.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/fnrd.yml \
--network.backbone.name resnet18_32x32 \
--network.backbone.checkpoint 'results/checkpoints/cifar100_res18_acc78.20.ckpt' \
--network.backbone.pretrained True \
--network.name fnrd_net \
--network.checkpoint 'results/cifar100_fnrd_net_fnrd_e1_lr0.1/best.ckpt' \
--network.pretrained True \
--merge_option merge \
