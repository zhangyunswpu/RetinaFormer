import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape #获得原始图片的高宽
	Hc, Wc = [size, size] #裁剪的高和宽

	# simple re-weight for the edge 裁剪
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc) #随机生成0到H-Hc的整数

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc) #随机生成0到W-Wc的整数

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :] #裁剪高HS-HS+hC 宽Ws-Ws+Wc

	# 水平翻转 对OTS不友好
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)
	#随机旋转
	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1)) #旋转
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2 #直接取中间的256*256
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs

#一对/一对地加载数据
class PairLoader(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'test','testoh','testdh']

		self.mode = mode #是train 还是验证 还是test
		self.size = size #裁剪的大小
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip #是否仅水平翻转

		self.root_dir = os.path.join(data_dir, sub_dir)
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT'))) #存储GT数据集下的数据名称
		self.img_num = len(self.img_names) #存储GT数据集下的数据集数量

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1] 将像素值从0-1 到-1到1
		img_name = self.img_names[idx]
		source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
		target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1


		#数据增强处理
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip) #都切到256*256的大小
		#裁剪处理
		if self.mode == 'testoh':
			source_img = source_img[ 100:2048, 100:1950,:]  #
			target_img = target_img[ 100:2048, 100:1950,:]  #
		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}
