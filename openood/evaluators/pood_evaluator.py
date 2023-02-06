import csv
import os
from typing import Dict

import numpy as np
import torch.nn as nn
from scipy.stats import spearmanr
from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator


class POODEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        """Perturbation-based OOD Evaluator.

        Args:
            config (Config): Config file from
        """
        super(POODEvaluator, self).__init__(config)
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None

    def eval_pood(self, net: nn.Module, ood_data_loaders: Dict, postprocessor: BasePostprocessor):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()

        result = []
        for magnitude in ood_data_loaders:
            ood_data_loader = ood_data_loaders[magnitude]
            print(f"Compute average confidence for magnitude {magnitude}")
            _, conf, _ = postprocessor.inference(net, ood_data_loader)
            conf = np.average(conf)
            result.append((magnitude, conf))

        csv_path = os.path.join(self.config.output_dir, 'pood.csv')
        fieldnames = ['perturbation', 'magnitude', 'avg_confidence']
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for magnitude, conf in result:
                write_content = {'perturbation': self.config.evaluator.perturbation,
                                 'magnitude': magnitude,
                                 'avg_confidence': conf
                                 }
                writer.writerow(write_content)
        result = np.array(result)
        score = spearmanr(result[:, 0], result[:, 1])
        print('Spearman: ', score)
        return score
