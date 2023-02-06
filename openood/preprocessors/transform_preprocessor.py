import torchvision.transforms as tvs_trans
from torch import Tensor
import torch
from openood.utils.config import Config

from .transform import Convert, interpolation_modes, normalization_dict

supported_perturbations = ['noise', 'blur']

class GaussianNoise:
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, X: Tensor):
        X = torch.stack([x + torch.randn(x.size()) * self.ratio * (torch.max(x) - torch.min(x)) for x in X])
        return X


class TransformPreprocessor():
    """ Perturbation-based transformation."""
    def __init__(self, config: Config, magnitude: float):
        self.pre_size = config.dataset.pre_size
        self.image_size = config.dataset.image_size
        self.interpolation = interpolation_modes[config.dataset.interpolation]
        normalization_type = config.dataset.normalization_type
        if normalization_type in normalization_dict.keys():
            self.mean = normalization_dict[normalization_type][0]
            self.std = normalization_dict[normalization_type][1]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

        pert_name = config.evaluator.perturbation
        assert pert_name in supported_perturbations, f'Unsupported perturbation {pert_name}'

        # Build list of operations
        ops = [Convert('RGB'), tvs_trans.Resize(self.pre_size, interpolation=self.interpolation)]
        if pert_name == 'blur' and magnitude > 0.0:
            ops.append(tvs_trans.GaussianBlur(kernel_size=3, sigma=magnitude))
        ops += [
            tvs_trans.CenterCrop(self.image_size),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=self.mean, std=self.std)]
        # Post-tensor operations
        if pert_name == 'noise':
            ops.append(GaussianNoise(magnitude))

        self.transform = tvs_trans.Compose(ops)

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        return self.transform(image)
