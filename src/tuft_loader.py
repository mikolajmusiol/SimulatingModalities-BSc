import numpy as np
from PIL import Image
from pathlib import Path
import random
from scipy.ndimage import rotate, shift


class Loader:
    def __init__(self, proj_path):
        self.proj_path = proj_path

    def load(self):
        rgb_paths, ir_paths = self.get_paths()
        rgb_imgs = []
        ir_imgs = []

        for rgb_path, ir_path in zip(rgb_paths, ir_paths):
            rgb = self.load_image(rgb_path, False)
            ir = self.load_image(ir_path, True)

            rgb = np.array(rgb, dtype=np.float32)
            ir = np.array(np.expand_dims([ir], axis=3), dtype=np.float32)[0]

            rgb_imgs.append(rgb)
            ir_imgs.append(ir)

            aug_pairs = self.get_augmented_pairs(rgb, ir)
            for rgb_aug, ir_aug in aug_pairs:
                rgb_imgs.append(rgb_aug)
                ir_imgs.append(ir_aug)

        rgb_imgs = np.array(rgb_imgs, dtype=np.float32)
        ir_imgs = np.array(ir_imgs, dtype=np.float32)

        return rgb_imgs, ir_imgs

    def get_paths(self):
        dir_rgb = Path(self.proj_path + "\\tufts\\RGB-faces-128x128")
        dir_ir = Path(self.proj_path + "\\tufts\\thermal-face-128x128")

        rgb_paths = [str(p) for p in dir_rgb.glob("*")]
        ir_paths = [str(p) for p in dir_ir.glob("*")]

        return rgb_paths, ir_paths

    def get_augmented_pairs(self, rgb, ir):
        pairs = []

        pairs.append((np.fliplr(rgb).copy(), np.fliplr(ir).copy()))

        angle = random.uniform(-15, 15)
        rgb_rot = rotate(rgb, angle, axes=(0, 1), reshape=False, mode='nearest')
        ir_rot = rotate(ir, angle, axes=(0, 1), reshape=False, mode='nearest')
        pairs.append((rgb_rot, ir_rot))

        h_shift = random.randint(-20, 20)
        w_shift = random.randint(-20, 20)
        rgb_shift = shift(rgb, (h_shift, w_shift, 0), mode='nearest')
        ir_shift = shift(ir, (h_shift, w_shift, 0), mode='nearest')
        pairs.append((rgb_shift, ir_shift))

        clean_pairs = []
        for r, i in pairs:
            r = np.clip(r, 0.0, 1.0)
            i = np.clip(i, 0.0, 1.0)
            clean_pairs.append((r, i))

        return clean_pairs

    def load_image(self, path, ir):
        if ir:
            img = Image.open(path).convert('L')
        else:
            img = Image.open(path)
        return self.process_image(img)

    def process_image(self, img):
        img = img.resize((256, 256))
        img = np.array(img)
        return img / 255.0