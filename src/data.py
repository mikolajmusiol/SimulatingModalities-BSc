import torch
from PIL import Image
import pandas as pd
import re
import numpy as np
from torch.utils.data import Dataset


class Loader:
    def __init__(self, proj_path):
        self.proj_path = proj_path

    def load(self, folds=None):
        paths = self.get_paths_for_folds(folds)
        rgb_imgs = []
        ir_imgs = []
        for path in paths:
            rgb, ir = self.load_image(path)
            rgb_imgs.append(rgb)
            ir_imgs.append(ir)
        return np.array(rgb_imgs, dtype=np.float32), np.array(np.expand_dims(ir_imgs, axis=3), dtype=np.float32)

    def get_paths_for_folds(self, folds=None):
        ds_path = self.proj_path + "wound_description_selected_fold.xlsx"
        patients_path = self.proj_path + "patients\\patients\\"
        df = pd.read_excel(ds_path, sheet_name="Arkusz1")
        paths = []

        if isinstance(folds, int):
            df = df[df['Fold'] == folds]
        elif isinstance(folds, list):
            df = df[df['Fold'].isin(folds)]

        for index, row_values in df.iterrows():
            record = re.sub(r"\D", " ", row_values.iloc[0]).split()
            full_path = patients_path + row_values.iloc[1] + "\\case_" + "_".join(record)
            paths.append(full_path)
        return paths

    def load_image(self, path):
        img_rgb = Image.open(path + "_rgb.png")
        img_ir = Image.open(path + "_ir.png").convert('L')
        img_rgb, img_ir = self.process_image(img_rgb, img_ir)
        return img_rgb, img_ir

    def process_image(self, rgb_img, ir_img):
        rgb_img, ir_img = rgb_img.resize((256, 256)), ir_img.resize((256, 256))
        rgb_img, ir_img = np.array(rgb_img), np.array(ir_img)
        rgb_img, ir_img = self.normalize_from_zero(rgb_img, ir_img)
        return rgb_img, ir_img

    def normalize(self, rgb_img, ir_img):
        rgb_img = (rgb_img / 127.5) - 1
        ir_img = (ir_img / 127.5) - 1
        return rgb_img, ir_img

    def normalize_from_zero(self, rgb_img, ir_img):
        rgb_img = rgb_img / 255.0
        ir_img = ir_img / 255.0
        return rgb_img, ir_img

class CustomDataset(Dataset):
    def __init__(self, rgb_images, ir_images, transform=None):
        self.rgb_images = torch.tensor(rgb_images, dtype=torch.float32).permute(0, 3, 1, 2)
        self.ir_images = torch.tensor(ir_images, dtype=torch.float32).permute(0, 3, 1, 2)
        self.transform = transform

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_image = self.rgb_images[idx]
        ir_image = self.ir_images[idx]
        if self.transform:
            rgb_image = self.transform(rgb_image)
            ir_image = self.transform(ir_image)
        return rgb_image, ir_image