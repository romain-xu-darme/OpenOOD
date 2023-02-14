from openood.datasets import get_dataloader
from openood.utils import setup_config
from openood.preprocessors.resize_preprocessor import ResizePreProcessor
from openood.networks.utils import get_network
from openood.preprocessors.transform import Convert
from openood.explainers.particul_explainer import dummy_preprocessing, ParticulExplainer
import torchvision.transforms as transforms
import os
from tqdm import tqdm

config = setup_config()
dataloader = get_dataloader(config)
net = get_network(config.network).eval()
resize = ResizePreProcessor(config)

trainloader = dataloader['train']
traindataset = trainloader.dataset

num_classes = traindataset.num_classes
num_patterns = net.num_patterns
best_samples = [[{'index': None, 'conf': 0} for _ in range(num_patterns)] for _ in range(num_classes)]

# First pass, find best samples per class and per pattern detector
target = 0.5
train_dataiter = iter(trainloader)
for batch in tqdm(train_dataiter, desc='Finding best samples: ', position=0, leave=False):
	images = batch['data_aux'].cuda()
	preds, confs, _ = net(images, return_xai_data=True)
	for sidx in range(confs.size(0)):
		gt = batch['label'][sidx]
		for pidx in range(num_patterns):
			if abs(confs[sidx, gt, pidx]-target) < abs(best_samples[gt][pidx]['conf']-target):
				best_samples[gt][pidx]['conf'] = confs[sidx, gt, pidx].item()
				best_samples[gt][pidx]['index'] = batch['index'][sidx].item()

resizing = transforms.Compose([
	Convert('RGB'),
	transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
])
preprocessing = transforms.Compose([
	resizing,
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Disable dataset preprocessing
traindataset.transform_aux_image = dummy_preprocessing
explainer = ParticulExplainer(net=net, preprocessing=preprocessing, resize=resizing)
os.makedirs(config.output_dir, exist_ok=True)
for cidx in range(num_classes):
	for pidx in range(num_patterns):
		if best_samples[cidx][pidx]['conf'] == 0.0:
			continue
		# Recover image
		sample = traindataset[best_samples[cidx][pidx]['index']]
		_, _, imgs = explainer.explain(sample['data_aux'])
		imgs[pidx].resize((150, 150)).save(os.path.join(config.output_dir, f"class_{sample['label']:03d}_p{pidx}.png"))

