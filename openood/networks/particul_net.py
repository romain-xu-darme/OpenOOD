import torch.nn as nn
from particul.od2.layers import ClassWiseParticul
from torch import Tensor
from typing import Tuple, Union
import torch
import openood.networks.lenet

class ParticulNet(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, num_patterns: int):
        """
        Build a ParticulNet
        Args:
            backbone: Target classifier
            num_classes: Number of classes
            num_patterns: Number of patterns per class
        """
        super(ParticulNet, self).__init__()

        self.backbone = backbone
        # Small hack to get correct feature size for LeNet architecture
        feature_size = 16 if isinstance(backbone, openood.networks.lenet.LeNet) else backbone.feature_size
        self.num_patterns = num_patterns
        self.num_classes = num_classes
        self.particuls = nn.ModuleList([ClassWiseParticul(feature_size, num_patterns)
                                       for _ in range(num_classes)])

        # Freeze classifier
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self,
                x: Tensor,
                return_confidence: bool = False,
                return_activation: bool = False,
                ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass
        Args:
            x: Tensor (N x 3 x H x W)
            return_confidence: Return confidence scores
            return_activation: Return pattern activation maps

        Returns:
            Prediction and confidence scores if return_confidence is True
            Prediction and pattern activation maps if return_activation is True
            Prediction only otherwise
        """
        logits, features = self.backbone(x, return_feature_map=True)

        # Aggregate results from all class-wise detectors
        amaps, conf = zip(*[p(features) for p in self.particuls])
        amaps = torch.stack(amaps, dim=1)  # Shape N x C x P x H x W
        conf = torch.stack(conf, dim=1)  # Shape N x C x P
        N, C, P, H, W = amaps.shape

        if return_confidence:
            # For each element of the batch, find index of most probable class
            class_idx = logits.argmax(dim=1, keepdim=True)  # Shape N x 1
            class_idx = class_idx[:, :, None].expand([N, 1, P])

            # Confidence scores of most probable classes
            conf = conf.gather(dim=1, index=class_idx).squeeze(dim=1)  # Shape N x P
            # Average confidence of all class detectors
            conf = conf.mean(dim=1, keepdim=True)
            return logits, conf
        elif return_activation:
            return logits, amaps
        else:
            return logits
