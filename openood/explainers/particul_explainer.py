import torch
from openood.networks.particul_net import ParticulNet
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Callable, List, Optional, Tuple
from torch import Tensor


def dummy_preprocessing(data: Any):
	return data


def migrate(tensor: Tensor) -> np.array:
	return tensor.detach().cpu().numpy().copy()


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


def get_largest_component(
		array: np.array
) -> np.array:
	""" Given a boolean array, return an array containing the largest region of
	positive values.

	:param array: Input array
	"""
	# Early abort
	if (array == False).all():
		return array

	# Find largest connected component in a single pass
	ncols = array.shape[1]
	cmax = None
	smax = 0
	active = []
	for y in range(array.shape[0]):
		marked = array[y]
		processed = []
		for comp in active:
			# Recover last row activation
			cprev = comp[y - 1]
			# Check possible connections with current column
			cexp = np.convolve(cprev, np.array([True, True, True]))[1:-1]
			# Is there a match?
			match = np.logical_and(array[y], cexp)
			if (match != False).any():
				# Update marked (untick elements in match)
				marked = np.logical_and(marked, np.logical_not(match))
				comp[y] = match
				# Check merge condition
				merged = False
				for pidx in range(len(processed)):
					if (np.logical_and(processed[pidx][y], match) != False).any():
						merged = True
						processed[pidx] = np.logical_or(processed[pidx], comp)
						break
				if not merged:
					processed.append(comp)
			else:
				# End of component
				size = np.sum(comp)
				if size > smax:
					smax = size
					cmax = comp
		active = processed

		# Init new components using unmarked elements
		i = 0
		while i < ncols:
			if marked[i]:
				# New component (all False)
				comp = np.zeros(array.shape) > 1.0
				# Extend component inside the current row
				while i < ncols and marked[i]:
					comp[y, i] = True
					i += 1
				active.append(comp)
			else:
				i += 1
	# Check last active components
	for comp in active:
		size = np.sum(comp)
		if size > smax:
			smax = size
			cmax = comp
	return cmax


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
	if keep_largest:
		# Keep only one connected component
		mask = get_largest_component(mask)

	M = (np.array(img) + value) * np.expand_dims(mask, axis=-1)
	vmax = np.amax(M)
	if vmax > 255:
		M = np.round(M / vmax * 255)
	M += np.array(img) * np.expand_dims(1 - mask, axis=-1)
	return Image.fromarray(M.astype(np.uint8))


def apply_bounding_box (
		img: Image.Image,
		mask: np.array,
		keep_largest: Optional[bool] = True,
) -> Image.Image:
	""" Return image with colored bounding box

	:param img: Source image
	:param mask: Image mask
	:param keep_largest: Keep largest connected component only
	"""
	# Apply threshold
	mask = mask > threshold_otsu(mask)
	if keep_largest:
		# Keep only one connected component
		mask = get_largest_component(mask)
	xmin, xmax, ymin, ymax = 0, mask.shape[0]-1, 0, mask.shape[1]-1
	while xmin < xmax:
		if mask[xmin, :].any():
			break
		xmin += 1
	while xmin < xmax:
		if mask[xmax, :].any():
			break
		xmax -= 1
	while ymin < ymax:
		if mask[:, ymin].any():
			break
		ymin += 1
	while ymin < ymax:
		if mask[:, ymax].any():
			break
		ymax -= 1
	draw = ImageDraw.Draw(img)
	draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
	return img


default_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
				  [255, 0, 255], [0, 255, 255], [128, 0, 255], [255, 0, 128],
				  [128, 128, 255], [255, 255, 255]]


class ParticulExplainer(nn.Module):
	def __init__(self,
				 net: ParticulNet,
				 preprocessing: Callable,
				 resize: Callable,
				 noise_ratio: float = 0.2,
				 nsamples: int = 10,
				 polarity: Optional[str] = 'absolute',
				 gaussian_ksize: Optional[int] = 5,
				 normalize: Optional[bool] = False,
				 ):
		"""	 Create ParticulExplainer

		:param net: ParticulNet
		:param preprocessing: Preprocessing function
		:param resize: Resizing function included in preprocessing
		:param noise_ratio: Noise level
		:param nsamples: Number of noisy samples
		:param polarity: Polarity filter applied on gradients
		:param gaussian_ksize: Size of Gaussian filter kernel
		:param normalize: Perform min-max normalization on gradients
		"""
		super().__init__()
		self.net = net.eval()
		self.preprocessing = preprocessing
		self.resize = resize
		self.ratio = noise_ratio
		self.nsamples = nsamples
		self.polarity = polarity
		self.gaussian_ksize = gaussian_ksize
		self.normalize = normalize

	def explain(self,
				img: Image.Image,
				part_index: List = None,
				device='cuda:0'
				) -> Tuple[int, List[float], List[Image.Image]]:
		""" Explain confidence

		:param img: Input image
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
		confs = migrate(confs)[0, pred]
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
		if self.normalize:
			res = [normalize_min_max(heatmap) for heatmap in res]

		# Apply gradient mask to image
		if self.resize:
			img = self.resize(img)
		processed = [apply_mask(img.copy(), pattern) for pattern in res]

		return pred, list(confs), processed
