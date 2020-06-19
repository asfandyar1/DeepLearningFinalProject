import torchvision.transforms as tf
import torch

class SetBoxLabel:
	def __init__(self, box, clss):
		self.box = box
		self.clss = clss
	def __call__(self, image, label):
		if self.box not in label: raise RuntimeWarning('Invalid label name {}'.fromat(self.box))
		if self.clss not in label: raise RuntimeWarning('Invalid label name {}'.fromat(self.clss))

		return image, {'boxes' : label[self.box], 'labels' : label[self.clss]}

class Resize:
	def __init__(self, width, height):
		self.width = width
		self.height = height

	def __call__(self, image, label):
		if self.width is None and self.height is None:
			return image, label

		orig_w, orig_h = image.size
		width = self.width if self.width != None else orig_w
		height = self.height if self.height != None else orig_h
		
		image = image.resize((width, height))

		def resizepos(tensor):
			return torch.cat((
				(tensor[:, 0] * width / orig_w).unsqueeze(0),
				(tensor[:, 1] * height / orig_h).unsqueeze(0),
				(tensor[:, 2] * width / orig_w).unsqueeze(0),
				(tensor[:, 3] * height / orig_h).unsqueeze(0)
			), dim=0).transpose(0,1)
		label['vehicle_position'] = resizepos(label['vehicle_position'])
		label['plate_position'] = resizepos(label['plate_position'])
		label['char_positions'] = resizepos(label['char_positions'])
		return image, label

def collate_fn(batch):
	images, labels = tuple(zip(*batch))
	labels = [t for t in labels]
	return images, labels

class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, image, target):
		for t in self.transforms:
			image, target = t(image, target)
		return image, target

class ToTensor(object):
	def __call__(self, image, target):
		image = tf.ToTensor()(image)
		return image, target