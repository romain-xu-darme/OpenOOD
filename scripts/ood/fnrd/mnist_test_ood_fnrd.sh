python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/fnrd.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/fnrd.yml \
--network.backbone.name lenet \
--network.backbone.checkpoint 'results/checkpoints/mnist_lenet_acc99.60.ckpt' \
--network.backbone.pretrained True \
--network.name fnrd_net \
--network.checkpoint 'results/mnist_fnrd_net_fnrd_e1_lr0.1/best.ckpt' \
--network.pretrained True \
--merge_option merge \
