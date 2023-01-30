#!/bin/bash
# sh scripts/ood/particul/cifar100_train_particul.sh

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/particul.yml \
configs/pipelines/train/train_particul.yml \
configs/preprocessors/base_preprocessor.yml \
--optimizer.num_epochs 200 \
--optimizer.lr 5.0e-4 \
--optimizer.weight_decay 1.0e-5 \
--network.backbone.name resnet18_32x32 \
--network.backbone.checkpoint 'results/checkpoints/cifar100_res18_acc78.20.ckpt' \
--network.num_patterns 4 \
--network.pretrained False \
--trainer.loc_ksize 1 \
--trainer.unq_ratio 1.0
