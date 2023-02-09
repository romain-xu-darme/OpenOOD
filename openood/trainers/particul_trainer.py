import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from particul.od2.loss import ParticulOD2Loss
from scipy.stats import logistic
from openood.utils import Config


class ParticulTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader, config: Config) -> None:
        self.train_loader = train_loader
        self.config = config
        self.net = net
        self.prediction_criterion = ParticulOD2Loss(
            npatterns=net.num_patterns,
            loc_ksize=config.trainer.loc_ksize,
            unq_ratio=config.trainer.unq_ratio,
            unq_thres=1.0,
            cls_ratio=0.0,
            cls_thres=0.0,
        ).cuda()
        self.optimizer = torch.optim.RMSprop(
            net.particuls.parameters(),
            lr=config.optimizer['lr'],
            weight_decay=config.optimizer['weight_decay'])

    def train_epoch(self, epoch_idx: int):
        self.net.train()
        self.net.backbone.eval()

        train_dataiter = iter(self.train_loader)
        global_metrics = {key: 0 for key in self.prediction_criterion.metrics}

        for _ in tqdm(range(1, len(train_dataiter) + 1), desc='Epoch {:03d}'.format(epoch_idx), position=0, leave=False):
            batch = next(train_dataiter)
            images = Variable(batch['data']).cuda()
            labels = Variable(batch['label']).cuda()

            pred_original, amaps = self.net(images, return_activation=True)
            loss, metrics = self.prediction_criterion(labels[:, None], (pred_original, amaps))

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            for val, key in zip(metrics, global_metrics):
                global_metrics[key] += val

        num_batch = len(train_dataiter)
        for key in global_metrics:
            global_metrics[key] /= num_batch
        global_metrics['epoch_idx'] = epoch_idx

        return self.net, global_metrics

    def calibrate(self):
        self.net.eval()
        # Disable normalization on all Particul detectors (mandatory to avoid calibration on normalized values!!)
        for particul in self.net.particuls:
            particul.enable_normalization = False
        train_dataiter = iter(self.train_loader)

        # Get max correlation scores for each image from the training set
        max_scores = [[[] for _ in range(self.net.num_patterns)] for _ in range(self.net.num_classes)]
        for batch in tqdm(train_dataiter,  desc='Calib: ', position=0, leave=False):
            # Use weak augmentation
            images = Variable(batch['data_aux']).cuda()
            labels = Variable(batch['label'])

            # Compute prediction and correlation scores. Shape (N x C), (N x C x H x W x P)
            _, amaps = self.net(images, return_activation=True)
            amaps = amaps.detach().cpu().numpy()  # Shape N x C x P x H x W

            # Recover maximum value of pattern correlations. Shape N x C x P
            vmax = np.max(amaps, axis=(3, 4))

            for v, label in zip(vmax, labels):
                for p in range(self.net.num_patterns):
                    max_scores[label][p].append(v[label, p])

        for c in range(self.net.num_classes):
            for p in range(self.net.num_patterns):
                # Get mean and standard deviation of distribution
                # of max correlation scores for a given pattern detector
                mu, sigma = logistic.fit(max_scores[c][p])
                assert sigma > 0, "Something went really wrong here"
                # Update model distribution parameters
                self.net.particuls[c].detectors[p].calibrate(mean=mu, std=sigma)

        # Restore normalization
        for particul in self.net.particuls:
            particul.enable_normalization = True
        return self.net.cuda()

