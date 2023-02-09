#!/bin/bash

python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/fnrd.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/fnrd.yml \
--network.backbone.name resnet18_32x32 \
--network.backbone.checkpoint 'results/checkpoints/cifar10_res18_acc94.30.ckpt' \
--network.backbone.pretrained True \
--network.name fnrd_net \
--network.checkpoint 'results/cifar10_fnrd_net_fnrd_e1_lr0.1/best.ckpt' \
--network.pretrained True \
--merge_option merge \
