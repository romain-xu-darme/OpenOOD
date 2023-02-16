from openood.networks.lenet import LeNet
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.networks.resnet18_64x64 import ResNet18_64x64
from openood.networks.resnet50 import ResNet50


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
        """
        super(FNRDNet, self).__init__()

        self.backbone = backbone
        self.last_features_only = False
        if isinstance(backbone, LeNet):
            mask_size = 1780
        elif isinstance(backbone, ResNet18_32x32):
            mask_size = 180736
        elif isinstance(backbone, ResNet18_64x64):
            mask_size = 721408
        elif isinstance(backbone, ResNet50):
            mask_size = 2048
            self.last_features_only = True
        else:
            mask_size = -1
        self.max_mask = nn.Parameter(
            torch.zeros((num_classes, mask_size), dtype=torch.float, device="cuda")
        )
        self.min_mask = nn.Parameter(
            100000*torch.ones((num_classes, mask_size), dtype=torch.float, device="cuda")
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

        n_batch = feature_list[0].size(0)
        if self.last_features_only:
            activations = feature_list[-1].view(n_batch, -1)
        else:
            activations = [f.view(n_batch, -1) for f in feature_list]
            activations = torch.cat(activations, dim=1)

        # Select min and max ranges corresponding to the predicted class
        class_idx = torch.argmax(pred, dim=1, keepdim=True).expand((n_batch, self.max_mask.size(1)))
        max_mask = torch.gather(self.max_mask, dim=0, index=class_idx)
        min_mask = torch.gather(self.min_mask, dim=0, index=class_idx)
        outliers = torch.sum(activations > max_mask, dim=1)+torch.sum(activations < min_mask, dim=1)
        cfd = 1-(outliers/self.min_mask.size(1))

        if return_confidence:
            return pred, cfd
        else:
            return pred
