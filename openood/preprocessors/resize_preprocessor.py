import torchvision.transforms as tvs_trans

from openood.utils.config import Config

from .base_preprocessor import BasePreprocessor
from .transform import Convert


class ResizePreProcessor(BasePreprocessor):
    """For test and validation dataset standard image transformation."""
    def __init__(self, config: Config):
        super(ResizePreProcessor, self).__init__(config)
        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(self.pre_size, interpolation=self.interpolation),
            tvs_trans.CenterCrop(self.image_size),
        ])
