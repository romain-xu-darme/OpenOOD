from openood.datasets import get_dataloader, get_ood_dataloader
from openood.utils import setup_config
from openood.networks.utils import get_network
from openood.preprocessors.resize_preprocessor import ResizePreProcessor
from openood.explainers.particul_explainer import dummy_preprocessing, ParticulExplainer
from typing import Callable
import pydot
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import random
from subprocess import check_call


def explain_dataset(
		dataset: Dataset,
		name: str,
		status: str,
		resize: Callable,
		output_dir: str,
		percentage: float = None
):
	# Disable dataset preprocessing
	preprocessing = dataset.transform_aux_image
	dataset.transform_aux_image = dummy_preprocessing
	explainer = ParticulExplainer(net=net, preprocessing=preprocessing, resize=resize)
	os.makedirs(os.path.join(output_dir, status, name), exist_ok=True)

	for sample in tqdm(dataset, desc=f'Processing {name}'):
		if percentage is not None:
			if random.random() > percentage:
				continue

		# Compute part visualisations
		pred, conf, imgs = explainer.explain(sample['data_aux'])
		num_patterns = len(imgs)
		# Resize and save images
		dir_path = os.path.join(output_dir, status, name, str(sample['index']))
		os.makedirs(dir_path, exist_ok=True)
		for pidx, img in enumerate(imgs):
			img = img.resize((150, 150))
			img.save(os.path.join(dir_path, f"p{pidx}.png"))
		sample['data_aux'].resize((150, 150)).save(os.path.join(dir_path, f"source.png"))

		# Create explanation
		graph = pydot.Dot('explanation', graph_type='digraph', compound=True)
		inference = pydot.Subgraph(rank="same")
		inference.add_node(pydot.Node(
						name='Source',
						shape="box",
						label=f"Predicted class: {pred}\nAvg conf.: {sum(conf)/num_patterns:.2f}",
						height=2.6, width=2.1, imagepos="tc", labelloc="b",
					    image=os.path.join(dir_path, 'source.png')
		))

		for pidx in range(num_patterns):
			if conf[pidx] < 0.3:
				inference.add_node(pydot.Node(
					name=f"Pattern{pidx}",
					shape="box",
					#label=f"<<B>NOT FOUND</B> <br/>(conf.:{conf[pidx]:.2f})>",
					label=f"NOT FOUND\n(conf.:{conf[pidx]:.2f}",
					fontcolor="red",
					height=2.6, width=2.1, imagepos="tc", labelloc="b",
					image=os.path.join(dir_path, 'source.png')
				))
			else:
				inference.add_node(pydot.Node(
					name=f"Pattern{pidx}",
					shape="box",
					#label=f"<<B>FOUND</B><br/>(conf.:{conf[pidx]:.2f})>",
					label=f"FOUND\n(conf.:{conf[pidx]:.2f}",
					fontcolor="black",
					height=2.6, width=2.1, imagepos="tc", labelloc="b",
					image=os.path.join(dir_path, f'p{pidx}.png')
				))

		references = pydot.Cluster(name="cluster_reference",
								   label=f"Training examples for class {pred}",
								   labelloc="t"
		)
		for pidx in range(num_patterns):
			# Fetch most correlated sample from training set
			references.add_node(pydot.Node(
				name=f"Pattern{pidx}Ref",
				label="",
				shape="box",
				height=2.1, width=2.1, imagepos="tc",
				image=os.path.join(output_dir, f'class_{pred:03d}_p{pidx}.png')
			))
		graph.add_subgraph(inference)
		graph.add_subgraph(references)
		# Connect
		for pidx in range(num_patterns):
			graph.add_edge(pydot.Edge(
				src=f"Pattern{pidx}Ref", dst=f"Pattern{pidx}",
				dir="none",
			))
		graph.add_edge(pydot.Edge(
			src="Source", dst="Pattern0",
			dir="none",
		))
		graph.write(os.path.join(dir_path, 'explanation.dot'))
		check_call(f"dot -Tpdf -Gmargin=0 {os.path.join(dir_path, 'explanation.dot')} "
				   f"-o {os.path.join(dir_path, 'explanation.pdf')}", shell=True)


config = setup_config()
net = get_network(config.network).eval()
iod_dataloaders = get_dataloader(config)
ood_dataloaders = get_ood_dataloader(config)
resize = ResizePreProcessor(config)

dataloaders = [(iod_dataloaders['test'], 'test', iod_dataloaders['test'].dataset.name)]
for status in ['nearood', 'farood']:
	for dataset_name in ood_dataloaders[status]:
		dataloaders.append((ood_dataloaders[status][dataset_name], status, dataset_name))

for dataloader, status, name in dataloaders:
	explain_dataset(
		dataset=dataloader.dataset,
		name=name,
		status=status,
		output_dir=config.output_dir,
		resize=resize,
		percentage=0.01,
	)
