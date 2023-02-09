import torch.nn as nn
from particul.od2.layers import ClassWiseParticul
from torch import Tensor
from typing import Tuple, Union
import torch


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
        self.particuls = nn.ModuleList([ClassWiseParticul(backbone.feature_size, num_patterns)
                                       for _ in range(num_classes)])

    # test conf
    # def forward(self, x, return_confidence=False):
    #
    #     _, feature = self.backbone(x, return_feature=True)
    #
    #     pred = self.fc(feature)
    #     confidence = self.confidence(feature)
    #
    #     if return_confidence:
    #         return pred, confidence
    #     else:
    #         return pred

    def forward(self,
                x: Tensor,
                return_confidence: bool = False,
                return_patterns: bool = False
                ) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Forward pass
        Args:
            x: Tensor (N x 3 x H x W)
            return_confidence: Return confidence scores
            return_patterns: Return pattern activation maps

        Returns:
            Prediction and confidence scores if return_confidence is True, prediction only otherwise
        """
        logits, feature_list = self.backbone(x, return_feature_list=True)

        print('backbone:', logits.shape, features.shape)
        # For each element of the batch, find most probable class
        class_idx = logits.argmax(dim=1, keepdim=True)  # Shape N x 1

        # Aggregate results from all class-wise detectors
        amaps, conf = zip(*[p(features) for p in self.particuls])
        amaps = torch.stack(amaps, dim=1)
        conf = torch.stack(conf, dim=1)
        print(amaps.shape, conf.shape)
        return amaps, conf

    # @property
    # def calibrated(self) -> bool:
    #     """ Return true if and only if all detectors have been calibrated
	# 	"""
    #     return all([particul.calibrated for particul in self.particuls])
    #
    # @property
    # def enable_normalization(self) -> bool:
    #     """ True if and only if normalization is enabled on all detectors
	# 	"""
    #     return all([p.enable_normalization for p in self.particuls])
    #
    # @enable_normalization.setter
    # def enable_normalization(self, val: bool) -> None:
    #     """ Enable/disable normalization layer on each detector
	# 	"""
    #     for p in self.particuls:
    #         p.enable_normalization = val
    #
    # def __repr__(self):
    #     """ Overwrite __repr__ to display a single detector """
    #     main_str = self._get_name() + ' [connected to ' + self.source + '] ('
    #     child_lines = [f'(particuls): {self.nclasses} x ' +
    #                    _addindent(repr(self.particuls[0]), 2)]
    #     main_str += '\n  ' + '\n  '.join(child_lines) + '\n)'
    #     return main_str
    #
    # def save(self, path: str) -> None:
    #     """ Save model
    #
	# 	:param path: Path to destination file
	# 	"""
    #     torch.save({
    #         'module': 'particul.od2.model',
    #         'class': 'ParticulOD2',
    #         'loader': 'load',
    #         'nclasses': self.nclasses,
    #         'npatterns': self.npatterns,
    #         'nchans': self.nchans,
    #         'source': self.source,
    #         'state_dict': self.state_dict(),
    #     }, path)
    #
    # @staticmethod
    # def load(path: str, map_location='cpu') -> nn.Module:
    #     """ Load model
    #
	# 	:param path: Path to source file
	# 	:param map_location: Target device
	# 	"""
    #     infos = torch.load(path, map_location='cpu')
    #     model = ParticulOD2(
    #         infos['nclasses'],
    #         infos['npatterns'],
    #         infos['nchans'],
    #         infos['source'],
    #     )
    #     model.load_state_dict(infos['state_dict'])
    #     return model.to(map_location)