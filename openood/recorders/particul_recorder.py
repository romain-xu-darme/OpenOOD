import os
import time
import torch.nn as nn
from openood.utils.config import Config
from pathlib import Path
from typing import Dict
import torch


class ParticulRecorder:
    def __init__(self, config: Config) -> None:
        self.config = config

        self.best_loss = 100.0
        self.best_epoch_idx = 0

        self.begin_time = time.time()
        self.output_dir = config.output_dir

    def report(self, train_metrics: Dict, val_metrics: Dict):
        print('Epoch {:03d} | Time {:5d}s | Train Loss {:.3f} (Loc: {:.3f}, Unq {:.3f}, Cls {:.3f})| '
              'Val Loss {:.3f} | Val Acc {:.2f}'.format(
                  (train_metrics['epoch_idx']),
                  int(time.time() - self.begin_time), train_metrics['Detection'],
                  train_metrics['Locality'], train_metrics['Unicity'], train_metrics['Clustering'],
                  val_metrics['loss'], 100.0 * val_metrics['acc']),
              flush=True)

    def save_model_state(self, net: nn.Module, path: str):
        save_path = os.path.join(self.output_dir, path)
        torch.save(net.state_dict(), save_path)

    def load_model_state(self, net: nn.Module, path: str):
        load_pth = os.path.join(self.output_dir, path)
        net.load_state_dict(torch.load(load_pth))
        return net.cuda()

    def save_model(self, net: nn.Module, val_metrics: Dict):
        if self.config.recorder.save_all_models:
            self.save_model_state(net, f"model_epoch{val_metrics['epoch_idx']}.ckpt")

        if val_metrics['loss'] <= self.best_loss:
            # Delete former best model (if any)
            old_path = os.path.join(self.output_dir, f'best_epoch{self.best_epoch_idx}_loss{self.best_loss:.4f}.ckpt')
            Path(old_path).unlink(missing_ok=True)

            self.best_epoch_idx = val_metrics['epoch_idx']
            self.best_loss = val_metrics['loss']
            best_path = f'best_epoch{self.best_epoch_idx}_loss{self.best_loss:.4f}.ckpt'
            self.save_model_state(net, best_path)
            self.save_model_state(net, 'best.ckpt')

    def summary(self):
        print('Training Completed! '
              'Best loss: {:.2f} '
              'at epoch {:d}'.format(self.best_loss, self.best_epoch_idx),
              flush=True)
