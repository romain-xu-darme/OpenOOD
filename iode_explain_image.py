import torch
import torchvision.transforms as transforms
from openood.networks.resnet50 import ResNet50
from openood.networks.particul_net import ParticulNet
from openood.explainers.particul_explainer import ParticulExplainer
from openood.preprocessors.transform import Convert
from PIL import Image
import pydot
import os
from subprocess import check_call
import argparse


def get_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser('Explain single image using IODE pretrained on ImageNet')
	parser.add_argument('--image',
						required=True,
						type=str,
						metavar='<path>',
						help='Path to image')
	parser.add_argument('--ref_dir',
						required=True,
						type=str,
						metavar='<path>',
						help='Path to reference directory')
	parser.add_argument('--output_dir',
						required=True,
						type=str,
						metavar='<path>',
						help='Path to output directory')
	parsed_args = parser.parse_args()
	return parsed_args


def explain_image(
		image: Image.Image,
		explainer: ParticulExplainer,
		ref_dir: str,
		output_dir: str,
):
	# Compute prediction
	pred, conf, imgs = explainer.explain(image, device='cuda:0')

	# Resize and save images
	for pidx, img in enumerate(imgs):
		img = img.resize((150, 150))
		img.save(os.path.join(output_dir, f"p{pidx}.png"))
	explainer.resize(image).resize((150, 150)).save(os.path.join(output_dir, f"source.png"))

	# Create explanation
	num_patterns = len(imgs)
	graph = pydot.Dot('explanation', graph_type='digraph', compound=True)
	inference = pydot.Subgraph(rank="same")
	fontsize=18
	inference.add_node(pydot.Node(
					name='Source',
					shape="box",
					label=f"<<B>Pred. class: {pred}<br/>Avg conf.: {sum(conf)/num_patterns:.2f}</B>>",
					fontsize=fontsize,
					height=2.7, width=2.1, imagepos="tc", labelloc="b",
					image=os.path.join(output_dir, 'source.png')
	))

	for pidx in range(num_patterns):
		if conf[pidx] < 0.3:
			inference.add_node(pydot.Node(
				name=f"Pattern{pidx}",
				shape="box",
				fontsize=fontsize,
				label=f"<<B>NOT FOUND <br/>(conf.:{conf[pidx]:.2f})</B>>",
				fontcolor="red",
				height=2.7, width=2.1, imagepos="tc", labelloc="b",
				image=os.path.join(output_dir, 'source.png')
			))
		else:
			inference.add_node(pydot.Node(
				name=f"Pattern{pidx}",
				shape="box",
				fontsize=fontsize,
				label=f"<<B>FOUND<br/>(conf.:{conf[pidx]:.2f})</B>>",
				fontcolor="black",
				height=2.7, width=2.1, imagepos="tc", labelloc="b",
				image=os.path.join(output_dir, f'p{pidx}.png')
			))

	references = pydot.Cluster(name="cluster_reference",
							   fontsize=fontsize,
							   label=f"<<B>Most activate samples for class {pred}</B>>",
							   labelloc="t"
	)
	for pidx in range(num_patterns):
		# Fetch most correlated sample from training set
		references.add_node(pydot.Node(
			name=f"Pattern{pidx}Ref",
			label="",
			shape="box",
			height=2.1, width=2.1, imagepos="tc",
			image=os.path.join(ref_dir, f'class_{pred:03d}_p{pidx}.png')
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
	graph.write(os.path.join(output_dir, 'explanation.dot'))
	check_call(f"dot -Tpdf -Gmargin=0 {os.path.join(output_dir, 'explanation.dot')} "
			   f"-o {os.path.join(output_dir, 'explanation.pdf')}", shell=True)


if __name__ == '__main__':
	args = get_args()

	# Load network
	backbone = ResNet50(num_classes=1000)
	backbone.load_state_dict(torch.load('results/checkpoints/imagenet_res50_acc76.10.pth'))
	net = ParticulNet(backbone=backbone, num_classes=1000, num_patterns=4)
	net.load_state_dict(torch.load('results/imagenet_particul_p4k3/imagenet_merged.ckpt'))
	net.cuda()

	# Preprocessing
	resizing = transforms.Compose([
		Convert('RGB'),
		transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
	])
	preprocessing = transforms.Compose([
		resizing,
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	img = Image.open(args.image)

	# Set up explainer
	explainer = ParticulExplainer(net=net, preprocessing=preprocessing, resize=resizing)
	# Create directory
	os.makedirs(args.output_dir, exist_ok=True)

	explain_image(image=img, explainer=explainer, ref_dir=args.ref_dir, output_dir=args.output_dir)