#!/bin/bash
# sh scripts/ood/particul/cifar100_test_particul.sh

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/particul.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/particul.yml \
--network.backbone.name resnet18_32x32 \
--network.backbone.checkpoint 'results/checkpoints/cifar10_res18_acc94.30.ckpt' \
--network.backbone.pretrained True \
--network.pretrained True \
--network.checkpoint "${1}/best.ckpt" \
--network.num_patterns 4 \
--mark "${2}"
