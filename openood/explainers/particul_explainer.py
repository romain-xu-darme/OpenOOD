import torch
from openood.networks.particul_net import ParticulNet
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Callable, List, Optional
from torch import Tensor


def dummy_preprocessing(data: Any):
	return data


def migrate(tensor: Tensor) -> np.array:
	return tensor.detach().cpu().numpy()


def polarity_and_collapse(
		array: np.array,
		polarity: Optional[str] = None,
		avg_chan: Optional[int] = None,
) -> np.array:
	""" Apply polarity filter (optional) followed by average over channels (optional)

	:param array: Target
	:param polarity: Polarity (positive, negative, absolute)
	:param avg_chan: Dimension across which channels are averaged
	"""
	assert polarity in [None, 'positive', 'negative', 'absolute'], f'Invalid polarity {polarity}'

	# Polarity first
	if polarity == 'positive':
		array = np.maximum(0, array)
	elif polarity == 'negative':
		array = np.abs(np.minimum(0, array))
	elif polarity == 'absolute':
		array = np.abs(array)

	# Channel average
	if avg_chan is not None:
		array = np.average(array, axis=avg_chan)
	return array


def normalize_min_max(array: np.array) -> np.array:
	""" Perform min-max normalization of a numpy array

	:param array: Target
	"""
	vmin = np.amin(array)
	vmax = np.amax(array)
	# Avoid division by zero
	return (array - vmin) / (vmax - vmin + np.finfo(np.float32).eps)


def apply_mask(
		img: Image.Image,
		mask: np.array,
		value: Optional[List[float]] = None,
		alpha: Optional[float] = 0.5,
		keep_largest: Optional[bool] = True,
) -> Image.Image:
	""" Return image with colored mask cropped to Otsu's threshold

	:param img: Source image
	:param mask: Image mask
	:param value: Mask color
	:param alpha: Overlay intensity
	:param keep_largest: Keep largest connected component only
	"""
	if value is None:
		value = [255.0, 0, 0]
	# Apply overlay intensity
	value = [v * alpha for v in value]
	# Apply threshold
	mask = mask > threshold_otsu(mask)
	# if keep_largest:
	# 	# Keep only one connected component
	# 	mask = get_largest_component(mask)

	M = (np.array(img) + value) * np.expand_dims(mask, axis=-1)
	vmax = np.amax(M)
	if vmax > 255:
		M = np.round(M / vmax * 255)
	M += np.array(img) * np.expand_dims(1 - mask, axis=-1)
	return Image.fromarray(M.astype(np.uint8))


default_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
				  [255, 0, 255], [0, 255, 255], [128, 0, 255], [255, 0, 128],
				  [128, 128, 255], [255, 255, 255]]


class ParticulExplainer(nn.Module):
	def __init__(self,
				 net: ParticulNet,
				 preprocessing: Callable,
				 noise_ratio: float = 0.2,
				 nsamples: int = 10,
				 polarity: Optional[str] = 'absolute',
				 gaussian_ksize: Optional[int] = 5,
				 normalize: Optional[bool] = True,
				 ):
		"""	 Create ParticulExplainer

		:param net: ParticulNet
		:param preprocessing: Preprocessing function
		:param noise_ratio: Noise level
		:param nsamples: Number of noisy samples
		:param polarity: Polarity filter applied on gradients
		:param gaussian_ksize: Size of Gaussian filter kernel
		:param normalize: Perform min-max normalization on gradients
		"""
		super().__init__()
		self.net = net.eval()
		self.preprocessing = preprocessing
		self.ratio = noise_ratio
		self.nsamples = nsamples
		self.polarity = polarity
		self.gaussian_ksize = gaussian_ksize
		self.normalize = normalize

	def explain(self,
				img: Image.Image,
				show_confidence: bool = False,
				part_index: List = None,
				device='cuda:0'
				) -> List[Image.Image]:
		""" Explain confidence

		:param img: Input image
		:param show_confidence: Display confidence level
		:param part_index: Indices of detectors (default: all)
		:param device: Target device (default: cuda:0)
		:returns: List of images
		"""
		tensor = img
		tensor = self.preprocessing(tensor)
		if tensor.dim() != 4:
			tensor = tensor[None]
		tensor = tensor.to(device, non_blocking=True)

		# Compute activation maps and copy to numpy array
		logits, confs, amaps = self.net(tensor, return_xai_data=True)

		# Find most probable class
		pred = torch.argmax(logits[0]).item()

		# Move to CPU and select activation maps from most probable class only
		amaps = migrate(amaps)[0, pred]
		P, H, W = amaps.shape

		# Find interesting locations
		part_index = range(P) if part_index is None else part_index
		locs = []
		coeffs = []
		for pidx in part_index:
			vmax = np.amax(amaps[pidx])
			vmax_loc = amaps[pidx].argmax()
			assert amaps[pidx, vmax_loc // W, vmax_loc % W] == vmax
			locs.append((pidx, vmax_loc // W, vmax_loc % W))
			coeffs.append(vmax)

		# Disable normalization layer to compute gradients on correlation scores directly
		for particul in self.net.particuls:
			particul.enable_normalization = False

		# Compute variance from noise ratio
		img_array = np.array(img)
		sigma = (np.max(img_array) - np.min(img_array)) * self.ratio

		# Generate noisy images around original img_array
		noisy_images = [
			img_array + np.random.normal(loc=0, scale=sigma, size=img_array.shape)
			for _ in range(self.nsamples)
		]
		# Caution!!! Noisy images must be rounded to uint8 in order to ensure normalization by
		# torchvision.transforms.ToTensor preprocessing function
		noisy_images = [np.clip(noisy_image, 0, 255).astype(np.uint8) for noisy_image in noisy_images]
		noisy_images = [Image.fromarray(noisy_image) for noisy_image in noisy_images]

		# Compute gradients
		grads = []
		for x in noisy_images:
			# Preprocess input
			T = self.preprocessing(x)[None]
			# Map to device
			T = T.to(device, dtype=torch.float, non_blocking=True)
			T.requires_grad_()

			# Single forward pass
			_, _, amaps = self.net(T, return_xai_data=True)
			amaps = amaps[0, pred]
			noisy_sample_grads = []
			for p, h, w in locs:
				output = amaps[p, h, w]
				output.backward(retain_graph=True)
				noisy_sample_grads.append(migrate(T.grad.data[0]))
				# Reset gradients
				T.grad.data.zero_()
			grads.append(noisy_sample_grads)

		# grads has shape (nsamples) x len(locs) x 3 x H x W
		grads = np.array(grads)

		# Restore model behavior
		for particul in self.net.particuls:
			particul.enable_normalization = True

		# Average results
		grads = np.mean(grads, axis=0)

		# Initialize heatmaps
		input_size = tensor.size()[1:]
		res = [np.zeros(input_size) for _ in range(P)]

		# Merge results
		for (pidx, _, _), coeff, grad in zip(locs, coeffs, grads):
			res[pidx] += coeff * grad
		res = [res[pidx] for pidx in part_index]

		# Post-processing
		# Apply polarity filter and channel average
		res = [polarity_and_collapse(heatmap, polarity=self.polarity, avg_chan=0) for heatmap in res]
		# Gaussian filter
		if self.gaussian_ksize:
			res = [gaussian_filter(heatmap, sigma=self.gaussian_ksize) for heatmap in res]
		# Normalize
		if self.normalize:
			res = [normalize_min_max(heatmap) for heatmap in res]

		# Apply gradient mask to image
		processed = [apply_mask(img, pattern, default_colors[0]) for pidx, pattern in enumerate(res)]

		# Add confidence score
		if show_confidence:
			imgs_w_conf = []
			myFont = ImageFont.truetype('OpenSans-Semibold.ttf', 30)
			for img, conf in zip(processed, confs[0, pred]):
				width, height = img.size
				res = Image.new(img.mode, (width, height+30), (255, 255, 255))
				res.paste(img, (0, 0))
				draw = ImageDraw.Draw(res)
				draw.text((width/2 - 5, height), f"{int(conf*100)}%", font=myFont, fill=(0, 0, 0))
				imgs_w_conf.append(res)
			processed = imgs_w_conf

		return processed
