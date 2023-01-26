import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from particul.od2.loss import ParticulOD2Loss
from openood.utils import Config


class ParticulEvaluator:
    def __init__(self, config: Config):
        self.config = config

    def eval_loss(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  epoch_idx: int = -1):
        # Set model output mode
        net.eval()

        # Loss function
        loss_fn = ParticulOD2Loss(
            npatterns=net.num_patterns,
            loc_ksize=self.config.trainer.loc_ksize,
            unq_ratio=self.config.trainer.unq_ratio,
            unq_thres=1.0,
            cls_ratio=0.0,
            cls_thres=0.0,
        ).cuda()

        # Keep only global Particul loss
        global_metrics = {'loss': 0, 'acc': 0, 'epoch_idx': epoch_idx}

        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Eval: ', position=0, leave=False):
                # Process data
                data = batch['data'].cuda()
                target = batch['label'].cuda()
                pred, amaps = net(data, return_activation=True)

                # Update metrics
                loss, _ = loss_fn(target[:, None], (pred, amaps))
                global_metrics['loss'] += loss.item()
                pred = pred.data.max(1)[1]
                global_metrics['acc'] += pred.eq(target.data).sum().item()/target.shape[0]

        # Normalize
        num_batch = len(data_loader)
        global_metrics['loss'] /= num_batch
        global_metrics['acc'] /= num_batch

        return global_metrics

    @staticmethod
    def eval_conf(net: nn.Module, data_loader: DataLoader):
        net.eval()
        global_metrics = {'conf': 0, 'acc': 0}

        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Eval: ', position=0, leave=False):
                data = batch['data'].cuda()
                target = batch['label'].cuda()
                pred, conf = net(data, return_confidence=True)
                global_metrics['conf'] += conf.mean().item()
                pred = pred.data.max(1)[1]
                global_metrics['acc'] += pred.eq(target.data).sum().item()/target.shape[0]

        # Normalize
        num_batch = len(data_loader)
        global_metrics['conf'] /= num_batch
        global_metrics['acc'] /= num_batch
        return global_metrics

    def extract(self, net: nn.Module, data_loader: DataLoader):
        raise NotImplemented

    def save_metrics(self, value):
        raise NotImplemented
