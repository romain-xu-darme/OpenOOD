import os

dataset_configs = {
	'cifar10': {
		'yaml': "--config configs/datasets/cifar10/cifar10.yml configs/datasets/cifar10/cifar10_ood.yml",
		'networks': {
			'standard': {
				'arch': 'configs/networks/resnet18_32x32.yml',
				'ckpt': '--network.checkpoint "results/checkpoints/cifar10_res18_acc94.30.ckpt" '
						'--network.pretrained True ',
			},
			'particul': {
				'arch': 'configs/networks/particul.yml',
				'ckpt': '--network.checkpoint "results/cifar10_particul_net_particul_e200_lr0.0005_p4k3/best.ckpt" '
						'--network.pretrained True '
						'--network.backbone.name resnet18_32x32 '
						'--network.backbone.checkpoint results/checkpoints/cifar10_res18_acc94.30.ckpt '
						'--network.backbone.pretrained True '
						'--network.num_patterns 4'
			},
			'fnrd': {
				'arch': 'configs/networks/fnrd.yml',
				'ckpt': '--network.checkpoint "results/cifar10_fnrd_net_fnrd_e1_lr0.1/best.ckpt" '
						'--network.pretrained True '
						'--network.backbone.name resnet18_32x32 '
                        '--network.backbone.checkpoint "results/checkpoints/cifar10_res18_acc94.30.ckpt" '
						'--network.backbone.pretrained True '
			},
			'vim': {
				'arch': 'configs/networks/resnet18_32x32.yml',
				'ckpt': '--network.checkpoint "results/checkpoints/cifar10_res18_acc94.30.ckpt" '
						'--network.pretrained True --postprocessor.postprocessor_args.dim 256 ',
			},
			'react': {
				'arch': 'configs/networks/react_net.yml',
				'ckpt': '--network.pretrained False '
						'--network.backbone.name resnet18_32x32 '
						'--network.backbone.pretrained True '
						'--network.backbone.checkpoint "results/checkpoints/cifar10_res18_acc94.30.ckpt" ',
			}
		},
	},
	'cifar100': {
		'yaml': "--config configs/datasets/cifar100/cifar100.yml configs/datasets/cifar100/cifar100_ood.yml",
		'networks': {
			'standard': {
				'arch': 'configs/networks/resnet18_32x32.yml',
				'ckpt': '--network.checkpoint "results/checkpoints/cifar100_res18_acc78.20.ckpt" '
						'--network.pretrained True ',
			},
			'particul': {
				'arch': 'configs/networks/particul.yml',
				'ckpt': '--network.checkpoint "results/cifar100_particul_net_particul_e200_lr0.0005_p4k3/best.ckpt" '
						'--network.pretrained True '
						'--network.backbone.name resnet18_32x32 '
						'--network.backbone.checkpoint results/checkpoints/cifar100_res18_acc78.20.ckpt '
						'--network.backbone.pretrained True '
						'--network.num_patterns 4'
			},
			'fnrd': {
				'arch': 'configs/networks/fnrd.yml',
				'ckpt': '--network.checkpoint "results/cifar100_fnrd_net_fnrd_e1_lr0.1/best.ckpt" '
						'--network.pretrained True '
						'--network.backbone.name resnet18_32x32 '
                        '--network.backbone.checkpoint "results/checkpoints/cifar100_res18_acc78.20.ckpt" '
						'--network.backbone.pretrained True '
			},
			'vim': {
				'arch': 'configs/networks/resnet18_32x32.yml',
				'ckpt': '--network.checkpoint "results/checkpoints/cifar100_res18_acc78.20.ckpt" '
						'--network.pretrained True --postprocessor.postprocessor_args.dim 256 ',
			},
			'react': {
				'arch': 'configs/networks/react_net.yml',
				'ckpt': '--network.pretrained False '
						'--network.backbone.name resnet18_32x32 '
						'--network.backbone.pretrained True '
						'--network.backbone.checkpoint "results/checkpoints/cifar100_res18_acc78.20.ckpt" ',
			}
		},
	},
	'mnist': {
		'yaml': "--config configs/datasets/mnist/mnist.yml configs/datasets/mnist/mnist_ood.yml",
		'networks': {
			'standard': {
				'arch': 'configs/networks/lenet.yml',
				'ckpt': '--network.checkpoint "results/checkpoints/mnist_lenet_acc99.60.ckpt" '
						'--network.pretrained True ',
			},
			'particul': {
				'arch': 'configs/networks/particul.yml',
				'ckpt': '--network.checkpoint "results/mnist_particul_net_particul_e200_lr0.0005_p4k3/best.ckpt" '
						'--network.pretrained True '
						'--network.backbone.name lenet '
						'--network.backbone.checkpoint results/checkpoints/mnist_lenet_acc99.60.ckpt '
						'--network.backbone.pretrained True '
						'--network.num_patterns 4'
			},
			'fnrd': {
				'arch': 'configs/networks/fnrd.yml',
				'ckpt': '--network.checkpoint "results/mnist_fnrd_net_fnrd_e1_lr0.1/best.ckpt" '
						'--network.pretrained True '
						'--network.backbone.name resnet18_32x32 '
                        '--network.backbone.checkpoint "results/checkpoints/mnist_lenet_acc99.60.ckpt" '
						'--network.backbone.pretrained True '
			},
			'vim': {
				'arch': 'configs/networks/lenet.yml',
				'ckpt': '--network.checkpoint "results/checkpoints/mnist_lenet_acc99.60.ckpt" '
						'--network.pretrained True --postprocessor.postprocessor_args.dim 42 ',
			},
			'react': {
				'arch': 'configs/networks/react_net.yml',
				'ckpt': '--network.pretrained False '
						'--network.backbone.name lenet '
						'--network.backbone.pretrained True '
						'--network.backbone.checkpoint "results/checkpoints/mnist_lenet_acc99.60.ckpt" ',
			}
		},
	},
}

pipeline = 'configs/pipelines/test/test_pood.yml configs/preprocessors/base_preprocessor.yml '

method_config = {
	'odin': {'network': 'standard', 'postprocessor': 'configs/postprocessors/odin.yml'},
	'msp': {'network': 'standard', 'postprocessor': 'configs/postprocessors/msp.yml'},
	'mds': {'network': 'standard', 'postprocessor': 'configs/postprocessors/mds.yml'},
	'gram': {'network': 'standard', 'postprocessor': 'configs/postprocessors/gram.yml'},
	'ebo': {'network': 'standard', 'postprocessor': 'configs/postprocessors/ebo.yml'},
	'gradnorm': {'network': 'standard', 'postprocessor': 'configs/postprocessors/gradnorm.yml'},
	'mls': {'network': 'standard', 'postprocessor': 'configs/postprocessors/mls.yml'},
	'klm': {'network': 'standard', 'postprocessor': 'configs/postprocessors/klm.yml'},
	'knn': {'network': 'standard', 'postprocessor': 'configs/postprocessors/knn.yml'},
	'dice': {'network': 'standard', 'postprocessor': 'configs/postprocessors/dice.yml'},
	'react': {'network': 'react', 'postprocessor': 'configs/postprocessors/react.yml'},
	'vim': {'network': 'vim', 'postprocessor': 'configs/postprocessors/vim.yml'},
	'particul': {'network': 'particul', 'postprocessor': 'configs/postprocessors/particul.yml'}
}

benchmark_scenari = {
	'noise': {
		'perturbation': 'noise',
		'values': "'[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]'",
	},
	'blur': {
		'perturbation': 'blur',
		'values': "'[0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0]'",
	},
	'rotate_forth': {
		'perturbation': 'rotate',
		'values': "'[0, 20, 30, 40, 50, 70, 90, 110, 130, 150, 180]'",
	},
	'rotate_back': {
		'perturbation': 'rotate',
		'values': "'[180, 210, 240, 270, 300, 320, 340, 350, 360]'",
	},
	'brightness': {
		'perturbation': 'brightness',
		'values': "'[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]'",
	},
}

methods = method_config.keys()
datasets = ['mnist', 'cifar10', 'cifar100']
scenari = benchmark_scenari.keys()

for dataset in datasets:
	for scenario in scenari:
		for method in methods:
			network_config = dataset_configs[dataset]['networks'][method_config[method]['network']]
			cmd_line = f"python main.py {dataset_configs[dataset]['yaml']} {network_config['arch']} {pipeline} "
			cmd_line += f"{method_config[method]['postprocessor']} "
			cmd_line += f"{network_config['ckpt']} "
			cmd_line += f"--evaluator.perturbation {benchmark_scenari[scenario]['perturbation']} "
			cmd_line += f"--evaluator.magnitudes={benchmark_scenari[scenario]['values']} "
			cmd_line += f"--num_workers 8 --merge_option merge --mark {scenario} --method {method}"
			os.system(cmd_line)
