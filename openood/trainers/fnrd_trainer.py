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
        self.net.train()
        correct_count = 0.0
        total = 0.0
        accuracy = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(
            range(1, len(train_dataiter) + 1),
            desc="Epoch {:03d}".format(epoch_idx),
            position=0,
            leave=True,
        ):
            batch = next(train_dataiter)
            images = Variable(batch["data"]).cuda()
            labels = Variable(batch["label"]).cuda()
            self.net.zero_grad()

            pred_original, feature_list = self.net(images, return_feature_list=True)
            activations = torch.Tensor(*feature_list)
            if train_step == 0:
                self.net.min_mask = activations.min(dim=0)[0]
                self.net.max_mask = activations.max(dim=0)[0]
            else:
                self.net.min_mask = torch.minimum(activations, self.net.min_mask)[0]
                self.net.max_mask = torch.maximum(activations, self.net.max_mask)[0]
            pred_original = F.softmax(pred_original, dim=-1)
            eps = self.config.trainer["eps"]
            pred_original = torch.clamp(pred_original, 0.0 + eps, 1.0 - eps)
            pred_idx = torch.max(pred_original.data, 1)[1]
            total += labels.size(0)
            correct_count += (pred_idx == labels.data).sum()
            accuracy = correct_count / total
        metrics = {}
        metrics["train_acc"] = accuracy
        metrics["epoch_idx"] = epoch_idx
        return self.net, metrics


def encode_onehot(labels, n_classes):
    onehot = torch.FloatTensor(labels.size()[0], n_classes)  # batchsize * num of class
    labels = labels.data
    if labels.is_cuda:
        onehot = onehot.cuda()
    onehot.zero_()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot
