import os
import torch
import random
import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt
from torch import Tensor
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Sequence, Union, Tuple
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler


def load_image(path, crop_scale, resize = (512, 512)): #resize of image_size*crop_scale crop_scale defines how much of frame we want to cut
    x = plt.imread(path)
    if crop_scale != 0:
        resize = (int(512*crop_scale), int(512*crop_scale))
    if resize != None and len(resize) == 2:
        x = cv2.resize(x, resize)
    return x


def prepare_filenames(data_path: str):
    file_paths = [os.path.join(data_path, img) for img in os.listdir(data_path)]
    return file_paths

def prepare_data(healthy_paths, sick_paths, crop_scale, as_dict = False):
    def equalize_class(healthy_paths, sick_paths):
        X = np.array([*healthy_paths, *sick_paths]).reshape(-1, 1)
        y = [*np.full(len(healthy_paths), 0), *np.full(len(sick_paths), 1)]
        rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        X_resampled = X_resampled.reshape(-1)
        
        healthy_paths_resampled = [_x for _x, _class in zip(X_resampled, y_resampled) if _class == 0]
        sick_paths_resampled = [_x for _x, _class in zip(X_resampled, y_resampled) if _class == 1]
        return healthy_paths_resampled, sick_paths_resampled
        
    def prepare_data_by_class(paths: list, labels_value, split_part = 0.9):
        images = [load_image(path, crop_scale)for path in paths]
        images = np.asarray(images)
        # images = minMaxScaler(images)
        np.random.shuffle(images)

        split_index = int(images.shape[0] * split_part)
        images_train = images[:split_index]  
        images_test = images[split_index:]

        labels_train = np.full(shape=images_train.shape[0], fill_value=labels_value, dtype=np.uint8)
        labels_test = np.full(shape=images_test.shape[0], fill_value=labels_value, dtype=np.uint8)

        return images_train, images_test, labels_train, labels_test

    def concatenate_data_part(h_images, s_images, h_labels, s_labels):
        images = np.concatenate((h_images, s_images))
        #images = np.stack((images,) * 3, axis=-1)
        labels = np.concatenate((h_labels, s_labels))

        bundle = list(zip(images, labels))
        random.shuffle(bundle)
        bundle = list(zip(*bundle))
        images = np.asarray(bundle[0])
        labels = np.asarray(bundle[1])

        return images, labels
        
    healthy_paths, sick_paths = equalize_class(healthy_paths, sick_paths)
    h_images_train, h_images_test, h_labels_train, h_labels_test = prepare_data_by_class(healthy_paths, labels_value=0)
    s_images_train, s_images_test, s_labels_train, s_labels_test = prepare_data_by_class(sick_paths, labels_value=1)

    images_train, labels_train = concatenate_data_part(h_images_train, s_images_train, h_labels_train, s_labels_train)
    images_test, labels_test = concatenate_data_part(h_images_test, s_images_test, h_labels_test, s_labels_test)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=images_test.shape[0])
    sss.get_n_splits(images_train, labels_train)
    train_ids, val_ids = next(sss.split(images_train, labels_train))

    images_val, labels_val = images_train[val_ids], labels_train[val_ids]
    images_train, labels_train = images_train[train_ids], labels_train[train_ids]
    
    if as_dict:
        return {'train': {'images': images_train, 'labels': labels_train}, 
                'val': {'images': images_val, 'labels': labels_val}, 
                'test': {'images': images_test, 'labels': labels_test}}

    return images_train, images_val, images_test, labels_train, labels_val, labels_test

# def prepare_data_distinct(root_path, healthy_paths, sick_paths, crop_scale):
#     def prepare_data_by_class(paths: list, labels_value):
#         images = parallel_load_data(paths, crop_scale)
#         images = np.asarray(images)
#         images = minMaxScaler(images)
#         np.random.shuffle(images)

#         labels = np.full(shape=images.shape[0], fill_value=labels_value, dtype=np.uint8)

#         return images, labels

#     def concatenate_data_part(h_images, s_images, h_labels, s_labels):
#         images = np.concatenate((h_images, s_images))
#         #images = np.stack((images,) * 3, axis=-1)
#         labels = np.concatenate((h_labels, s_labels))

#         bundle = list(zip(images, labels))
#         random.shuffle(bundle)
#         bundle = list(zip(*bundle))
#         images = np.asarray(bundle[0])
#         labels = np.asarray(bundle[1])

#         return images, labels
    
#     healthy_paths = load_data_paths(root_path, healthy_paths)
#     sick_paths = load_data_paths(root_path, sick_paths)

#     h_images, h_labels = prepare_data_by_class(healthy_paths, labels_value=0)
#     s_images, s_labels = prepare_data_by_class(sick_paths, labels_value=1)

#     images, labels = concatenate_data_part(h_images, s_images, h_labels, s_labels)

#     return images, labels


class MyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.image_data, self.label_data = dataset        
        self.transform = transform
    
    def __len__(self):
        return len(self.label_data)
    
    def __getitem__(self, idx):
        image = self.image_data[idx]
        label = self.label_data[idx]
        image = image.astype(np.uint8)
        if self.transform:
            image = self.transform(image)
        return image, label
            
     