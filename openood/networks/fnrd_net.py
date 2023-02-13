from openood.networks.lenet import LeNet
from openood.networks.resnet18_32x32 import ResNet18_32x32

import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union
import torch


class FNRDNet(nn.Module):
    def __init__(self, backbone: nn.Module):
        """
        Build a FNRDNet
        Args:
            backbone: Target classifier
        """
        super(FNRDNet, self).__init__()

        self.backbone = backbone

        if isinstance(backbone, LeNet):
            mask_size = 1780
        elif isinstance(backbone, ResNet18_32x32):
            mask_size = 180736
        else:
            mask_size = -1
        self.max_mask = nn.Parameter(
            torch.empty(mask_size, dtype=torch.float, device="cuda")
        )
        self.min_mask = nn.Parameter(
            torch.empty(mask_size, dtype=torch.float, device="cuda")
        )


    def forward(
        self,
        x: Tensor,
        return_confidence: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass
        Args:
            x: Tensor (N x 3 x H x W)
            return_confidence: Return confidence scores
        Returns:
            Prediction and confidence scores if return_confidence is True, prediction only otherwise
        """
        pred, feature_list = self.backbone(x, return_feature_list=True)

        outliers = torch.Tensor()
        n_batch = feature_list[0].size(0)
        activations = [f.view(n_batch, -1) for f in feature_list]
        activations = torch.cat(activations, dim=1)
        # For each element of the batch
        for act in activations:
            max_outliers = torch.numel(act[act > self.max_mask])
            min_outliers = torch.numel(act[act < self.min_mask])
            outliers = torch.cat(
                (outliers, torch.Tensor([max_outliers + min_outliers]))
            )
        cfd = 1-(outliers/len(self.min_mask))

        if return_confidence:
            return pred, cfd
        else:
            return pred
