import torch
import pickle
import os
from PIL import Image
import argparse
from transforms import Compose, Resize


def to_box(coord):
	return [coord[0], coord[1], coord[0] + coord[2], coord[1] + coord[3]]

def getLabeledData(image_file, text_file, transform=None):
	#print(image_file, text_file)
	image = Image.open(image_file)
	labels = {}

	with open(text_file) as label_file:
		for line in label_file:
			if 'position_vehicle:' in line:
				line = line.split(' ')
				l = [line[-4], line[-3], line[-2], line[-1].split('\n')[0]]
				l = to_box(list(map(float, l)))
				labels['vehicle_position'] = torch.as_tensor([l], dtype=torch.float32)
			if '\t' and 'type:' in line:
				line = line.split(' ')
				if (line[-1].split('\n')[0] == "car"):
					labels['vehicle_type'] = torch.as_tensor([2], dtype=torch.int64)
				else:
					labels['vehicle_type'] = torch.as_tensor([1], dtype=torch.int64)
			if 'plate:' in line:
				aux_plate = line.split(' ')[-1].split('\n')[0]
				aux_plate = list(map(int, map(ord, aux_plate)))
				if len(aux_plate) > 3:
					labels['plate'] = torch.as_tensor(aux_plate, dtype=torch.int64)
			if 'position_plate:' in line:
				line = line.split(' ')
				l = [line[-4], line[-3], line[-2], line[-1].split('\n')[0]]
				l = to_box(list(map(float, l)))
				labels['plate_position'] = torch.as_tensor([l], dtype=torch.float32)
			if '\t' and 'char' in line:
				line = line.split(' ')
				char_aux = [line[-4], line[-3], line[-2], line[-1].split('\n')[0]]
				char_aux = to_box(list(map(float,char_aux)))
				if 'char_positions' not in labels:
					labels['char_positions'] = []
				labels['char_positions'].append(char_aux)
		if len(labels) != 5:
			raise RuntimeWarning('Labels has incomplete data')
		labels['char_positions'] = torch.as_tensor(labels['char_positions'], dtype=torch.float32)
		if transform != None:
			return transform(image, labels)
		return image, labels


class CarLicensePlatesFiles(torch.utils.data.Dataset):
	def __init__(self, dataDir, transform=None):
		folders = [os.path.join(dataDir, i) for i in os.listdir(dataDir)]
		files = []
		for i in folders:
			files += [os.path.join(i, file) for file in os.listdir(i)]
		self.image_files = list(filter(lambda x: '.png' in x, files))
		self.text_files = list(filter(lambda x: '.txt' in x, files))
		self.image_files.sort()
		self.text_files.sort()
		self.transform = transform

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, i):
		return getLabeledData(self.image_files[i], self.text_files[i], transform=self.transform)

class CarLicensePlatesPickle(torch.utils.data.Dataset):
	# Initialization method for the dataset
	def __init__(self, dataDir, transform = None):
		with open(dataDir, 'rb') as fp:
			self.data = pickle.load(fp)
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		image, labels = self.data[i]
		if self.transform is not None:
			return self.transform(image, labels)
		return image, labels


if __name__ == "__main__":
	#When this script is called, it is used for loading all files from folders in some path
	#and storing the images/labels in a single file that can be used for loading with
	#CarLicensePlatesPickle class. Also gives the option to resize the data before storing it.
	parser = argparse.ArgumentParser()
	parser.add_argument('paths', help='Path from where to get data')
	parser.add_argument('output', help='Output pickle file')
	parser.add_argument('--width', type=int, default=None, help='Desired image width')
	parser.add_argument('--height', type=int, default=None, help='Desired image height')
	args = parser.parse_args()
	transform = Compose([
		Resize(args.width, args.height)]
	)
	dataset = CarLicensePlatesFiles(args.paths, transform=transform)
	data = []
	for i in range(len(dataset)):
		data.append(dataset[i])
		if i % 100 == 0:
			print('{}/{}'.format(i+1, len(dataset)))
	print('Complete')
	with open(args.output, 'wb') as fp:
		pickle.dump(data, fp)




