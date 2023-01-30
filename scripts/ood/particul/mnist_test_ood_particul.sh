#!/bin/bash
# sh scripts/ood/particul/cifar10_test_ood_particul.sh
PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/particul.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/particul.yml \
--network.backbone.name lenet \
--network.backbone.checkpoint 'results/checkpoints/mnist_lenet_acc99.60.ckpt' \
--network.backbone.pretrained True \
--network.pretrained True \
--network.num_patterns 4 \
--network.checkpoint "${1}/best.ckpt" \
--mark "${2}"
