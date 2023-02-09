from openood.datasets import get_dataloader
from openood.utils import setup_config
from openood.networks.utils import get_network
from openood.explainers.particul_explainer import dummy_preprocessing, ParticulExplainer
import os
from tqdm import tqdm

config = setup_config()
dataloader = get_dataloader(config)
net = get_network(config.network).eval()

trainloader = dataloader['train']
traindataset = trainloader.dataset

num_classes = traindataset.num_classes
num_patterns = net.num_patterns
best_samples = [[{'index': None, 'conf': 0} for _ in range(num_patterns)] for _ in range(num_classes)]

# First pass, find best samples per class and per pattern detector
train_dataiter = iter(trainloader)
for batch in tqdm(train_dataiter, desc='Finding best samples: ', position=0, leave=False):
	images = batch['data_aux'].cuda()
	preds, confs, _ = net(images, return_xai_data=True)
	for sidx in range(confs.size(0)):
		gt = batch['label'][sidx]
		for pidx in range(num_patterns):
			if confs[sidx, gt, pidx] > best_samples[gt][pidx]['conf']:
				best_samples[gt][pidx]['conf'] = confs[sidx, gt, pidx].item()
				best_samples[gt][pidx]['index'] = batch['index'][sidx].item()

# Disable dataset preprocessing
preprocessing = traindataset.transform_aux_image
traindataset.transform_aux_image = dummy_preprocessing
explainer = ParticulExplainer(net=net, preprocessing=preprocessing)
os.makedirs(config.output_dir, exist_ok=True)
for cidx in range(num_classes):
	for pidx in range(num_patterns):
		# Sanity check
		if best_samples[cidx][pidx]['conf'] < 0.8:
			print(cidx, pidx, best_samples[cidx][pidx]['conf'])
		# Recover image
		sample = traindataset[best_samples[cidx][pidx]['index']]
		imgs = explainer.explain(sample['data_aux'])
		imgs[pidx].save(os.path.join(config.output_dir, f"class_{sample['label']:03d}_p{pidx}.png"))

