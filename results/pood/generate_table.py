import csv
import os
import argparse
import matplotlib.pyplot as plt


def get_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser('Generate LaTeX table of Perturbation benchmarks and all curves')
	parser.add_argument('--curve-dir',
						required=True,
						type=str,
						metavar='<path>',
						help='Directory of to curve files')
	parser.add_argument('--table',
						required=True,
						type=str,
						metavar='<path>',
						help='Path to LaTeX file')
	parsed_args = parser.parse_args()
	return parsed_args


methods = ['msp', 'odin', 'mds', 'gram', 'ebo', 'gradnorm',
		   'react', 'mls', 'klm', 'vim', 'knn', 'dice', 'particul']
methods_nice = ['MSP', 'ODIN', 'MDS', 'Gram', 'EBO', 'GradNorm',
				'ReAct', 'MaxLogit', 'KLM', 'ViM', 'KNN', 'DICE', 'Particul']
scenari = ['noise', 'blur', 'rotate_forth', 'rotate_back', 'brightness']
datasets = ['MNIST', 'CIFAR10', 'CIFAR100']

scenari_short = {'noise': 'Noise $\\downarrow$',
				 'blur': 'Blur $\\downarrow$',
				 'rotate_forth': 'R+ $\\downarrow$',
				 'rotate_back': 'R- $\\uparrow$',
				 'brightness': 'Bright. $\\uparrow$'
				 }
scenari_polarity = {'noise': 'negative',
					'blur': 'negative',
					'rotate_forth': 'negative',
					'rotate_back': 'positive',
					'brightness': 'positive'
					}
meta_scenari = ['Noise', 'Blur', 'Brightness', 'Rotate']
scenari_magnitude_labels = {'noise': 'Noise ratio', 'blur': '\u03C3', 'brightness': 'Ratio', 'rotate': 'Angle'}


def generate_table(output: str):
	# Prepare table header
	table = "\\begin{table*}\n" \
			"\t\\caption{\\textbf{Comparison of OoD methods on our perturbation benchmark.}}\n" \
			"\t\\label{table:pood}\n" \
			"\t\\centering \n" \
			"\t{\\renewcommand\\baselinestretch{1.3}\\selectfont \\resizebox{\\textwidth}{!}{ \n" \
			"\t\\begin{tabular}{@{\\hskip 8pt}l@{\\hskip 6pt}"
	for _ in datasets:
		table += f"|@{{\\hskip 6pt}}{'c' * len(scenari)}@{{\\hskip 6pt}}"
	table += "}\n\t\\toprule\n\t"
	for dataset in datasets[:-1]:
		table += f"& \\multicolumn{{{len(scenari)}}}{{c@{{\\hskip 6pt}}|@{{\\hskip 6pt}}}}{{{dataset}}}\n"
	table += f"& \\multicolumn{{{len(scenari)}}}{{c}}{{{datasets[-1]}}}\n"
	table += "\\\\ \n\t  "
	for _ in datasets:
		for scenario in scenari:
			table += f"& {scenari_short[scenario]} "
	table += "\\\\ \n\t  "

	for method, mname in zip(methods, methods_nice):
		if method == 'particul':
			table += f"\t\\rowcolor{{NEED_TRAIN}} \\sun~ {mname}"
		else:
			table += f"\t\\rowcolor{{NO_TRAIN}} \\SnowflakeChevron~ {mname}"
		for dataset in datasets:
			for scenario in scenari:
				# Update Spearman table
				found = False
				with open('pood_results.csv', 'r') as csvfile:
					reader = csv.DictReader(csvfile)
					for row in reader:
						if row['dataset'] == dataset.lower() \
								and row['method'] == method and row['perturbation'] == scenario:
							sr = float(row['Spearman'])
							if (sr > -0.3 and scenari_polarity[scenario] == 'negative') or \
									(sr < 0.3 and scenari_polarity[scenario] == 'positive'):
								table += f" & \\textcolor{{red}}{{{sr}}}"
							else:
								table += f" & {sr}"
							found = True
				assert found, f"Could not find entry {scenario}, {dataset.lower()}, {method}"

		table += "\\\\ \n"
	table += "\\bottomrule \n\t\\end{tabular}}\\par}\n\\end{table*}"
	with open(output, 'w') as fout:
		fout.write(table)


def generate_graphs(dir: str):
	for dataset in datasets:
		fig, axs = plt.subplots(nrows=len(methods), ncols=len(meta_scenari), sharex='col')
		for sc_idx, scenario in enumerate(meta_scenari):
			if scenario != 'Rotate':
				for midx, (method, mname) in enumerate(zip(methods, methods_nice)):
					path = os.path.join(scenario.lower(), dataset.lower(), method, 'pood.csv')
					with open(path, 'r') as csvfile:
						reader = csv.DictReader(csvfile)
						magnitudes, avg_confidences = [], []
						for row in reader:
							magnitudes.append(float(row['magnitude']))
							avg_confidences.append(float(row['avg_confidence']))

					# Check correctness
					with open('pood_results.csv', 'r') as csvfile:
						reader = csv.DictReader(csvfile)
						for row in reader:
							if row['dataset'] == dataset.lower() \
									and row['method'] == method and row['perturbation'] == scenario.lower():
								sr = float(row['Spearman'])
								if (sr > -0.3 and scenari_polarity[scenario.lower()] == 'negative') or \
										(sr < 0.3 and scenari_polarity[scenario.lower()] == 'positive'):
									color = 'tab:red'
								else:
									color = 'tab:blue'
					axs[midx][sc_idx].plot(magnitudes, avg_confidences, label=mname, color=color)
					axs[midx][sc_idx].set_yticklabels([])
					axs[midx][sc_idx].tick_params(left=False)
					axs[midx][sc_idx].legend(loc=1)
				axs[-1][sc_idx].set_xlabel(scenari_magnitude_labels[scenario.lower()])
				axs[0][sc_idx].set_title(f'{scenario} on {dataset}')
			else:
				# For rotation, we aggregate results from rotate_forth and rotate_back
				for midx, (method, mname) in enumerate(zip(methods, methods_nice)):
					for direction in ['rotate_forth', 'rotate_back']:
						magnitudes, avg_confidences = [], []
						path = os.path.join(direction, dataset.lower(), method, 'pood.csv')
						with open(path, 'r') as csvfile:
							reader = csv.DictReader(csvfile)
							for row in reader:
								magnitudes.append(float(row['magnitude']))
								avg_confidences.append(float(row['avg_confidence']))

						# Check correctness
						with open('pood_results.csv', 'r') as csvfile:
							reader = csv.DictReader(csvfile)
							for row in reader:
								if row['dataset'] == dataset.lower() \
										and row['method'] == method and row['perturbation'] == direction:
									sr = float(row['Spearman'])
									if (sr > -0.3 and scenari_polarity[direction] == 'negative') or \
											(sr < 0.3 and scenari_polarity[direction] == 'positive'):
										color = 'tab:red'
									else:
										color = 'tab:blue'
						if direction == 'rotate_back':
							axs[midx][sc_idx].plot(magnitudes, avg_confidences, label=mname, color=color)
							axs[midx][sc_idx].legend(loc=1)
						else:
							axs[midx][sc_idx].plot(magnitudes, avg_confidences, color=color)
						axs[midx][sc_idx].set_yticklabels([])
						axs[midx][sc_idx].tick_params(left=False)

				axs[-1][sc_idx].set_xlabel(scenari_magnitude_labels[scenario.lower()])
				axs[0][sc_idx].set_title(f'{scenario} on {dataset}')
		fig = plt.gcf()
		fig.set_size_inches(12, 10)
		os.makedirs(dir, exist_ok=True)
		fig.tight_layout()
		fig.savefig(os.path.join(dir, f'{dataset}_pood.png'), dpi=100)


if __name__ == '__main__':
	args = get_args()
	generate_table(output=args.table)
	generate_graphs(dir=args.curve_dir)
