#!/bin/bash

python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/fnrd.yml \
configs/pipelines/train/train_fnrd.yml \
configs/preprocessors/base_preprocessor.yml \
--optimizer.num_epochs 1 \
--network.backbone.name lenet \
--network.backbone.pretrained true \
--merge_option merge \
--network.backbone.checkpoint 'results/checkpoints/mnist_lenet_acc99.60.ckpt' \
