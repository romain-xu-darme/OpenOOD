import argparse
import pydot
import os
from subprocess import check_call

def get_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser('Show class patterns')
	parser.add_argument('--dir',
						required=True,
						type=str,
						metavar='<path>',
						help='Directory of class pattern images')
	parser.add_argument('--index',
						type=int, nargs=2,
						metavar=('<min>', '<max>'),
						help='Class index range')
	parser.add_argument('--output',
						required=True,
						type=str,
						metavar='<path>',
						help='Path to output file')
	parsed_args = parser.parse_args()
	return parsed_args


if __name__ == '__main__':
	args = get_args()
	graph = pydot.Dot('samples', graph_type='digraph', compound=True)
	for index in range(args.index[0], args.index[1]):
		pidx = 0
		samples = pydot.Cluster(graph_name=f"cluster_class{index}",
								label=f"<<B>Training examples for class {index+1}</B>>",
								fontsize=20,
								labelloc="t"
								)
		while os.path.exists(os.path.join(args.dir, f'class_{index:03d}_p{pidx}.png')):
			samples.add_node(pydot.Node(
				name=f'pattern_{index}_{pidx}',
				fontsize=20,
				shape="box",
				label=f"<Pattern {3 - pidx}>",
				height=2.4, width=2.1, imagepos="tc", labelloc="b",
				image=os.path.join(args.dir, f'class_{index:03d}_p{pidx}.png')
			))
			pidx += 1
		graph.add_subgraph(samples)
	# Invisible connection to stack clusters
	for index in range(args.index[0], args.index[1] - 2):
		graph.add_edge(pydot.Edge(
				src=f"pattern_{index}_0", dst=f"pattern_{index+2}_0",
				dir="none",
				style='invis',
		))

	graph.write(os.path.join(args.dir, 'class_samples.dot'))
	check_call(f"dot -Tpdf -Gmargin=0 {os.path.join(args.dir, 'class_samples.dot')} "
			   f"-o {args.output}", shell=True)