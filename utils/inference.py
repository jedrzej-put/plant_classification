import sys
sys.path.append('/srv/container/')
from segmentation.unet import Unet
from classification.helpers.model import CovidModel, SmallCNNModel

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, io
from torch.utils.data.dataloader import default_collate
from torchmetrics.classification import BinaryRecall, BinaryPrecision,BinaryJaccardIndex

import pandas as pd
from skimage import io, transform
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import yaml

from typing import List

def load_model(path:str, device, Model_Class = Unet, *args):
    model = Model_Class(*args)
    if issubclass(Model_Class, Unet):
        model.load_state_dict(torch.load(path))
    elif issubclass(Model_Class, SmallCNNModel):
        model.load_state_dict(torch.load(path)['model_state_dict'])
    else:
        raise ValueError("Invalid model")
    model.eval()
    model.to(device)
    return model

def get_paths(img_dir):
  img_files = os.listdir(img_dir)
  img_files = [os.path.join(img_dir, img) for img in img_files]
  return img_files

class CustomImageDataset(Dataset):
  def __init__(self, paths_img_files, transform=None):
    self.paths_img_files = paths_img_files
    self.transform = transform

  def __len__(self):
    return len(self.paths_img_files)

  def __getitem__(self, idx):
    img_path = self.paths_img_files[idx]
    image = Image.open(img_path)
    if self.transform:
        image = self.transform(image)
    return image, img_path


@torch.inference_mode()
def inference_model(model, images_dataloader, output_path, device):
    results = []
    for batch in images_dataloader:
        images, paths = batch
        images = images.to(device)
        # images = torch.stack(batch, 0)
        outputs = model(images)
        predicted_masks = (outputs >= 0.5).float() * 255

        for img, mask,path in zip(images, predicted_masks,paths):
            mask = mask.squeeze()
            mask = mask.numpy(force=True).astype(np.uint8)
            img = img.permute(1, 2, 0)
            img = (img.numpy(force=True) * 255).astype(np.uint8)
            res = cv2.bitwise_and(img,img, mask=mask)
            
            res_pil = transforms.functional.to_pil_image(res)
            res_pil.save(f"/{output_path}/{path.split('/')[-1]}")
            results.extend(res)
    return results

def get_intersection(input_dir:str, output_dir:str, device: str, model_path: str) -> List[np.ndarray]:
    model = load_model(model_path, device)
    paths_img_files = get_paths(input_dir)
    images_dataset = CustomImageDataset(
        paths_img_files=paths_img_files,
        transform=transforms.Compose([transforms.Resize((592, 592)), transforms.ToTensor()])
    )
    images_dataloader = DataLoader(images_dataset, batch_size=4, shuffle=True, pin_memory=True)
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    inference_model(model, images_dataloader, output_path, device)

@torch.inference_mode()
def _inference_model_batch(model, images_on_device: torch.Tensor, device, target_transformation) -> torch.Tensor:
    
    """ Return a Tensor of transformed intersections on gpu """
    results = torch.Tensor()
    outputs = model(images_on_device)
    predicted_masks = (outputs >= 0.5).float() * 255
    for img, mask in zip(images_on_device, predicted_masks):
        mask = mask.squeeze()
        mask = mask.numpy(force=True).astype(np.uint8)
        img = img.permute(1, 2, 0)
        img = (img.numpy(force=True) * 255).astype(np.uint8)
        res = cv2.bitwise_and(img,img, mask=mask)
        
        res_pil = transforms.functional.to_pil_image(res)
        transformed_res = target_transformation(res_pil)
        results= torch.cat((results, torch.unsqueeze(transformed_res, 0)),dim=0)

    results = results.to(device)
    return results

@torch.inference_mode()
def _inference_model_batch_optimal(model, images_on_device: torch.Tensor, device, target_transformation) -> torch.Tensor:
    
    """ Return Tensor of transformed intersections on gpu """
    results = torch.Tensor().to(device)
    outputs = model(images)
    predicted_masks = (outputs >= 0.5).float() * 255
    for img, mask in zip(images_on_device, predicted_masks):
        mask = mask.squeeze()
        mask = mask.unsqueeze(0).repeat(3,1,1)
        res = torch.where(mask == 255, img, 0)
        results= torch.cat((results, torch.unsqueeze(res, 0)),dim=0)
    return results
    
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('config', help='determine path to config yaml file')
    args = parser.parse_args() 
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)
    device = torch.device(config["DEVICE"])
    path_to_model = config["MODEL_PATH"]
    model = load_model(path_to_model, device)
    
    paths_img_files = get_paths(img_dir=config["IMAGES_PATH"])
    
    images_dataset = CustomImageDataset(
        paths_img_files=paths_img_files,
        transform=transforms.Compose([transforms.Resize((592, 592)), transforms.ToTensor()])
    )
    images_dataloader = DataLoader(images_dataset, batch_size=4, shuffle=True, pin_memory=True)

    path = Path(config["OUTPUT_PATH"])
    path.mkdir(parents=True, exist_ok=True)
    inference_model(model, images_dataloader, config["OUTPUT_PATH"], device)

def main_for_batch():
    device = torch.device("cuda")
    path_to_model = '/srv/container/segmentation/rye-1/models/model-epoch-85.pth'
    unet_model = load_model(path_to_model, device)
    paths_img_files = get_paths(img_dir="/srv/data/rye/leaves-1/sick/images")
    images_dataset = CustomImageDataset(
        paths_img_files=paths_img_files,
        transform=transforms.Compose([transforms.Resize((592, 592)), transforms.ToTensor()])
    )
    images_dataloader = DataLoader(images_dataset, batch_size=4, shuffle=True, pin_memory=True)
    small_model = load_model('/srv/container/classification/logs/01_rye_clf_aw_1_small/models/model_0024.pt',
                            device,
                            SmallCNNModel,
                            [32, 64, 128, 256]
    )
    target_transform=transforms.Compose([transforms.Resize((592, 592)), transforms.ToTensor()])
    with torch.no_grad():
        for batch in images_dataloader:
            images, paths = batch
            images = images.to(device)
            intersection_images = _inference_model_batch(unet_model, images, device,target_transform)
            print(np.shape(intersection_images))
            outputs = small_model(intersection_images)
            print(np.shape(outputs))
    
if __name__ == '__main__':
    main()
    # main_for_batch()