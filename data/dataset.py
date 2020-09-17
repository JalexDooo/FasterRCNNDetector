import glob
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import Dataset

from .utils import *
from data.voc_dataset import VOCBboxDataset


class Transform:
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = resize_box(bbox, (H, W), (o_H, o_W))

        img, params = random_flip(img, x_random=True, return_param=True)
        bbox = flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])
        return img, bbox, label, scale


class VOC2007Detect(Dataset):
    def __init__(self, path, is_train=True, min_size=600, max_size=1000):
        self.path = path
        self.is_train = is_train
        if self.is_train:
            self.db = VOCBboxDataset(path)
            self.tsf = Transform(min_size, max_size)
        else:
            self.db = VOCBboxDataset(path, split='val', use_difficult=True)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        if self.is_train:
            img, bbox, label, scale = self.tsf((ori_img, bbox, label))
            return img, bbox, label, scale
        else:
            img = preprocess(ori_img)
            return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
