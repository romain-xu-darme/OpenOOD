import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from openood.utils import Config


class FNRDTrainer:
    def __init__(self, net, train_loader, config: Config) -> None:
        self.train_loader = train_loader
        self.config = config
        self.net = net
        self.prediction_criterion = nn.NLLLoss().cuda()
        self.optimizer = torch.optim.SGD(
            net.parameters(),
            lr=config.optimizer["lr"],
            momentum=config.optimizer["momentum"],
            nesterov=config.optimizer["nesterov"],
            weight_decay=config.optimizer["weight_decay"],
        )
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=config.scheduler["milestones"],
            gamma=config.scheduler["gamma"],
        )
        self.lmbda = self.config.trainer["lmbda"]

    def train_epoch(self, epoch_idx):
        """Compute the Maximum Function Region (MFR)
        for a system. The MFR of a neuron is the min and max
        post-activation values for the whole
        training dataset."""
        self.net.eval()
        correct_count = 0.0
        train_dataiter = iter(self.train_loader)
        assert epoch_idx == 1
        with torch.no_grad():
            for train_step in tqdm(
                range(1, len(train_dataiter) + 1),
                desc="Epoch {:03d}".format(epoch_idx),
                position=0,
                leave=True,
            ):
                batch = next(train_dataiter)
                images = Variable(batch["data"]).cuda()
                labels = Variable(batch["label"]).cuda()

                pred_original, feature_list = self.net.backbone(
                    images, return_feature_list=True
                )
                n_batch = feature_list[0].size(0)
                if self.net.last_features_only:
                    activations = feature_list[-1].view(n_batch,-1)
                else:
                    activations = [f.view(n_batch, -1) for f in feature_list]
                    activations = torch.cat(activations, dim=1)
                for activation, label in zip(activations, labels):
                    self.net.min_mask[label.item()] = self.net.min_mask[label.item()].minimum(activation)
                    self.net.max_mask[label.item()] = self.net.max_mask[label.item()].maximum(activation)
                pred = pred_original.data.max(1)[1]
                correct_count += pred.eq(labels.data).sum().item()
        acc = correct_count / len(self.train_loader.dataset)
        metrics = {}
        metrics["train_acc"] = acc
        metrics["epoch_idx"] = epoch_idx
        metrics["loss"] = 0.0
        return self.net, metrics
