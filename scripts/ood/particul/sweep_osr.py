# python scripts/ood/particul/sweep_osr.py
import os

osr_configs = [
    ['mnist6', 'lenet'],
    ['cifar6', 'resnet18_32x32'],
    ['cifar50', 'resnet18_32x32'],
    ['tin20', 'resnet18_64x64'],
]
lr = "0.0005"
num_epochs = 200
mark = 'p6k3'

# Merge results
output = 'results/osr_ood.csv'
output_dir = 'results/osr/p6'

for name, arch in osr_configs:
    for seed in range(1, 6):
        # Training Particul detectors
        command = f"PYTHONPATH='.':$PYTHONPATH \
            python main.py \
            --config configs/datasets/osr_{name}/{name}_seed{seed}.yml \
            configs/networks/particul.yml \
            configs/pipelines/train/train_particul.yml \
            configs/preprocessors/base_preprocessor.yml \
            --optimizer.num_epochs {num_epochs} \
            --optimizer.lr {lr} \
            --optimizer.weight_decay 1.0e-5 \
            --network.backbone.name {arch} \
            --network.backbone.checkpoint 'results/checkpoints/osr/{name}_seed{seed}.ckpt' \
            --network.num_patterns 4 \
            --network.pretrained False \
            --trainer.loc_ksize 3 \
            --trainer.unq_ratio 1.0 \
            --merge_option merge "
        os.system(command)

        # OSR test
        command = f"PYTHONPATH='.':$PYTHONPATH \
            python main.py \
            --config configs/datasets/osr_{name}/{name}_seed{seed}.yml \ \
            configs/datasets/osr_{name}/{name}_seed{seed}_osr.yml \
            configs/networks/particul.yml \
            configs/pipelines/test/test_osr.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/particul.yml \
            --network.backbone.name {arch} \
            --network.backbone.checkpoint 'results/checkpoints/osr/{name}_seed{seed}.ckpt' \
            --network.backbone.pretrained True \
            --network.pretrained True \
            --network.checkpoint './results/{name}_seed{seed}_particul_net_particul_e{num_epochs}_lr{lr}/best.ckpt' \
            --network.num_patterns 6 \
            --output_dir {output_dir} \
            --mark {mark} \
            --merge_option merge"
        os.system(command)

        with open(output, 'a') as fout:
            fname = os.path.join(output_dir, f'{name}_seed{seed}_particul_net_test_ood_osr_particul_{mark}/ood.csv')
            with open(fname, 'r') as fin:
                lines = fin.readlines()
                fout.write(f"{fname},{seed},{lines[1]}")

    # Add additional line to insert average values
    with open(output, 'a') as fout:
        fout.write('\n')
