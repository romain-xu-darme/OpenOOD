import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union
import torch


class FNRDNet(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        """
        Build a FNRDNet
        Args:
            backbone: Target classifier
            num_classes: Number of classes
        """
        super(FNRDNet, self).__init__()

        self.backbone = backbone
        self.max_mask : torch.Tensor
        self.min_mask : torch.Tensor

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
        num_neurons = len(self.min_mask)
        outliers = []
        for act in zip(*feature_list):
            max_outliers = torch.numel(act[act > self.max_mask])
            min_outliers = torch.numel(act[act < self.min_mask])
            outliers.append((max_outliers + min_outliers) / num_neurons)
        cfd = sum(outliers) / len(outliers)
        if return_confidence:
            return pred, cfd
        else:
            return pred
