import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os


class UBI_Dataset:

	def __init__(self, root, transform=None, target_transform=None, dataset_type='train', keep_difficult=False, label_file=None):
		"""Dataset for VOC data.
		Args:
			root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
				Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
		"""
		# self.root = self.root / "ssd_project/ssd-pytorch-leanh"
		# self.root = './data/VOCdevkit/'

		self.datapath = root
  
		self.transform = transform
		self.target_transform = target_transform

		self.dataset_type = dataset_type
		self.image_sets_file = os.path.join(self.datapath, f'{dataset_type}.txt')
		
  
		self.ids = UBI_Dataset._read_image_ids(self.image_sets_file)
		self.keep_difficult = keep_difficult

		# if the labels file exists, read in the class names
		# label_file_name = self.root + "labels.txt"
  
		label_file_name = os.path.join('/', *self.datapath.split('/')[0:-3], 'models/labels.txt')
		if os.path.exists(label_file_name):
      
			# prepend BACKGROUND as first class
			classes = ['BACKGROUND']
   
			with open(label_file_name, 'r') as infile:
				for line in infile:
					classes.append(line.rstrip())  
			classes = [ elem.replace(" ", "") for elem in classes]
			self.class_names = tuple(classes)
			logging.info("VOC Labels read from file: " + str(self.class_names))
		else:
			logging.info("No labels file, using default VOC classes.")
			# self.class_names = ('BACKGROUND',
			# 'aeroplane', 'bicycle', 'bird', 'boat',
			# 'bottle', 'bus', 'car', 'cat', 'chair',
			# 'cow', 'diningtable', 'dog', 'horse',
			# 'motorbike', 'person', 'pottedplant',
			# 'sheep', 'sofa', 'train', 'tvmonitor')

			self.class_names = ('BACKGROUND', "vehicle", "rider", "pedestrian") 


		self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

	def __getitem__(self, index):
		image_id = self.ids[index]
		boxes, labels, is_difficult = self._get_annotation(image_id)
		if not self.keep_difficult:
			boxes = boxes[is_difficult == 0]
			labels = labels[is_difficult == 0]
		image = self._read_image(image_id)
		if self.transform:
			image, boxes, labels = self.transform(image, boxes, labels)
		if self.target_transform:
			boxes, labels = self.target_transform(boxes, labels)
		return image, boxes, labels

	def get_image(self, index):
		image_id = self.ids[index]
		image = self._read_image(image_id)
		if self.transform:
			image, _ = self.transform(image)
		return image

	def get_annotation(self, index):
		image_id = self.ids[index]
		return image_id, self._get_annotation(image_id)

	def __len__(self):
		return len(self.ids)

	@staticmethod
	def _read_image_ids(image_sets_file):
		ids = []
		with open(image_sets_file) as f:
			for line in f:
				ids.append(line.rstrip())
		return ids

	def _get_annotation(self, image_id):

		annotation_file = os.path.join(self.datapath, "labels", f"{image_id}.xml")
   
		objects = ET.parse(annotation_file).findall("object")

		boxes = []
		labels = []
		is_difficult = []
		for object in objects:
			class_name = object.find('name').text.lower().strip()
			# we're only concerned with clases in our list
			if class_name in self.class_dict:
				bbox = object.find('bndbox')

				# VOC dataset format follows Matlab, in which indexes start from 0
				x1 = float(bbox.find('xmin').text) - 1
				y1 = float(bbox.find('ymin').text) - 1
				x2 = float(bbox.find('xmax').text) - 1
				y2 = float(bbox.find('ymax').text) - 1

				if x1 > x2:
					x1, x2 = x2, x1
				if y1 > y2:
					y1, y2 = y2, y1
     
				boxes.append([x1, y1, x2, y2])

				labels.append(self.class_dict[class_name])

				try:
					is_difficult_str = object.find('difficult').text
				except:
					# is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
					is_difficult.append(0)

		return (np.array(boxes, dtype=np.float32),
				np.array(labels, dtype=np.int64),
				np.array(is_difficult, dtype=np.uint8))

	def _read_image(self, image_id):


		image_file = os.path.join(self.datapath, "images", f"{image_id}.jpg")

		image = cv2.imread(image_file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		return image