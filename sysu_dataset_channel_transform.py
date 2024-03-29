import os
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing, ChannelExchange

class SYSUData(data.Dataset):
	def __init__(self, data_root, transform=None, visibleIndex=None, thermalIndex=None):
		# Load training images and labels
		self.train_visible_image = np.load(os.path.join(data_root, 'train_rgb_resized_img_128_256.npy'))
		self.train_visible_label = np.load(os.path.join(data_root, 'train_rgb_label_128_256.npy'))

		self.train_thermal_image = np.load(os.path.join(data_root, 'train_ir_resized_img_128_256.npy'))
		self.train_thermal_label = np.load(os.path.join(data_root, 'train_ir_label_128_256.npy'))

		self.transform = transform
		self.vIndex   = visibleIndex
		self.tIndex   = thermalIndex

		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

		self.transform_thermal = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Pad(10),
			transforms.RandomCrop((256, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
			ChannelRandomErasing(probability=0.5),
			#ChannelAdapGray(probability=0.5)
		])

		self.transform_color = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Pad(10),
			transforms.RandomCrop((256, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
			ChannelRandomErasing(probability=0.5),
			# transforms.RandomGrayscale(p = 0.1),
		])

		self.transform_color1 = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Pad(10),
			transforms.RandomCrop((256, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
			ChannelRandomErasing(probability=0.5),
			# ChannelExchange(gray=2)
		])

	def __getitem__(self, index):
		img_v, target_v = self.train_visible_image[self.vIndex[index]], self.train_visible_label[self.vIndex[index]]
		img_t, target_t = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

		img_v_0 = self.transform_color(img_v)
		img_v_1 = self.transform_color1(img_v)
		img_t = self.transform_thermal(img_t)

		return img_v_0, img_v_1, img_t, target_v, target_t

	def __len__(self):
		# return len(self.train_visible_label)
		return len(self.vIndex)


def process_query_sysu(data_root, mode='all'):
	if mode== 'all':
		ir_cameras = ['cam3','cam6']
	elif mode =='indoor':
		ir_cameras = ['cam3','cam6']

	file_path = os.path.join(data_root,'exp/test_id.txt')
	img_paths_ir = []
	with open(file_path, 'r') as file:
		ids = file.read().splitlines()
		ids = [int(y) for y in ids[0].split(',')]
		ids = ["%04d" % x for x in ids]

	for pid in sorted(ids):
		for cam in ir_cameras:
			img_dir = os.path.join(data_root, cam, pid)
			if os.path.isdir(img_dir):
				new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
				img_paths_ir.extend(new_files)

	query_img = []
	query_pid = []
	query_camid = []
	for img_path in img_paths_ir:
		camid, pid = int(img_path[-15]), int(img_path[-13:-9])
		query_img.append(img_path)
		query_pid.append(pid)
		query_camid.append(camid)

	return query_img, np.array(query_pid), np.array(query_camid)


def process_gallery_sysu(data_root, mode='all', shot=1, trial=0):
	random.seed(trial)

	if mode== 'all':
		rgb_cameras = ['cam1','cam2','cam4','cam5']
	elif mode =='indoor':
		rgb_cameras = ['cam1','cam2']

	file_path = os.path.join(data_root,'exp/test_id.txt')
	img_paths_rgb = []
	with open(file_path, 'r') as file:
		ids = file.read().splitlines()
		ids = [int(y) for y in ids[0].split(',')]
		ids = ["%04d" % x for x in ids]

	for pid in sorted(ids):
		for cam in rgb_cameras:
			img_dir = os.path.join(data_root, cam, pid)
			if os.path.isdir(img_dir):
				new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
				img_paths_rgb += random.sample(new_files, shot)
				# img_paths_rgb.append(random.choice(new_files))

	gallery_img = []
	gallery_pid = []
	gallery_camid = []
	for img_path in img_paths_rgb:
		camid, pid = int(img_path[-15]), int(img_path[-13:-9])
		gallery_img.append(img_path)
		gallery_pid.append(pid)
		gallery_camid.append(camid)
	return gallery_img, np.array(gallery_pid), np.array(gallery_camid)
