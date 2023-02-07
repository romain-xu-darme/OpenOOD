import time
import os
import csv
from openood.datasets import get_dataloader, get_pood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger


class TestPOODPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_pood_dataloader(self.config)

        # init network
        net = get_network(self.config.network)

        # init ood evaluator
        evaluator = get_evaluator(self.config)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict, ood_loader_dict)
        print('\n', flush=True)
        print(u'\u2500' * 70, flush=True)

        # start evaluating ood detection method on perturbed dataset
        timer = time.time()
        score = evaluator.eval_pood(net, ood_loader_dict, postprocessor)
        print('Time used for eval_pood: {:.0f}s'.format(time.time() - timer))
        print('Completed!', flush=True)

        csv_path = os.path.join('results/pood/', 'pood_results.csv')
        write_content = {
            'dataset': self.config.dataset.name,
            'method': self.config.method,
            'perturbation': self.config.mark,
            'Spearman': round(score[0], 2),
            'p-value': round(score[1], 3)
        }
        field_names = write_content.keys()
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writerow(write_content)
