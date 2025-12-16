import os

import torch
from PIL import Image
import pandas as pd
import re
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.functional import (
    hflip,
    vflip,
    rotate,
    resized_crop,
)
import random


class Loader:
    def __init__(self, proj_path):
        self.proj_path = proj_path

    def load(self, folds=None):
        paths = self.get_paths_for_folds(folds)
        rgb_imgs = []
        ir_imgs = []
        for path in paths:
            rgb_list, ir_list = self.load_images_for_case(path)

            rgb_imgs.extend(rgb_list)
            ir_imgs.extend(ir_list)

        rgb_imgs = np.array(rgb_imgs, dtype=np.float32)
        ir_imgs = np.array(ir_imgs, dtype=np.float32)

        ir_imgs = np.expand_dims(ir_imgs, axis=3)

        return rgb_imgs, ir_imgs

    def load_images_for_case(self, path):
        rgb_paths = [
            f"{path}_rgb.png",
            f"{path}_rgb_crop_1.png",
            f"{path}_rgb_crop_2.png",
        ]

        ir_paths = [
            f"{path}_ir.png",
            f"{path}_ir_crop_1.png",
            f"{path}_ir_crop_2.png",
        ]

        rgb_images = []
        ir_images = []

        for rgb_p, ir_p in zip(rgb_paths, ir_paths):
            if not os.path.exists(rgb_p) or not os.path.exists(ir_p):
                continue

            rgb = Image.open(rgb_p)
            ir = Image.open(ir_p).convert("L")

            rgb, ir = self.process_image(rgb, ir)

            rgb_images.append(rgb)
            ir_images.append(ir)

        return rgb_images, ir_images

    def get_paths_for_folds(self, folds=None):
        ds_path = self.proj_path + "wound_description_selected_fold.xlsx"
        patients_path = self.proj_path + "patients\\patients\\"
        df = pd.read_excel(ds_path, sheet_name="Arkusz1")
        paths = []

        if isinstance(folds, int):
            df = df[df['Fold'] == folds]
        elif isinstance(folds, list):
            df = df[df['Fold'].isin(folds)]

        for _, row in df.iterrows():
            record = re.sub(r"\D", " ", row.iloc[0]).split()
            full_path = patients_path + row.iloc[1] + "\\case_" + "_".join(record)
            paths.append(full_path)
        return paths

    def process_image(self, rgb_img, ir_img):
        rgb_img = rgb_img.resize((256, 256))
        ir_img = ir_img.resize((256, 256))

        rgb_img = np.array(rgb_img)
        ir_img = np.array(ir_img)

        rgb_img, ir_img = self.normalize_from_zero(rgb_img, ir_img)
        return rgb_img, ir_img

    def normalize_from_zero(self, rgb_img, ir_img):
        return rgb_img / 255.0, ir_img / 255.0


class CustomDataset(Dataset):
    def __init__(
        self,
        rgb_images,
        ir_images,
        augment=False,
        output_size=(256, 256)
    ):
        self.rgb_images = torch.tensor(rgb_images, dtype=torch.float32).permute(0, 3, 1, 2)
        self.ir_images = torch.tensor(ir_images, dtype=torch.float32).permute(0, 3, 1, 2)
        self.augment = augment
        self.output_size = output_size

    def __len__(self):
        return len(self.rgb_images)

    def synchronized_transform(self, rgb, ir):
        if random.random() < 0.5:
            rgb = hflip(rgb)
            ir = hflip(ir)

        if random.random() < 0.5:
            rgb = vflip(rgb)
            ir = vflip(ir)

        angle = random.uniform(-15, 15)
        rgb = rotate(rgb, angle)
        ir = rotate(ir, angle)

        i, j, h, w = RandomResizedCrop.get_params(
            rgb,
            scale=[0.8, 1.0],
            ratio=[0.9, 1.1]
        )

        rgb = resized_crop(rgb, i, j, h, w, self.output_size)
        ir = resized_crop(ir, i, j, h, w, self.output_size)

        return rgb, ir

    def __getitem__(self, idx):
        rgb = self.rgb_images[idx]
        ir = self.ir_images[idx]

        if self.augment:
            rgb, ir = self.synchronized_transform(rgb, ir)

        return rgb, ir