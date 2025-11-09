from PIL import Image
import numpy as np
from pathlib import Path


class Loader():
    def __init__(self, proj_path):
        self.proj_path = proj_path

    def load(self):
        rgb_paths, ir_paths = self.get_paths()
        rgb_imgs = []
        ir_imgs = []
        for rgb_path, ir_path in zip(rgb_paths, ir_paths):
            rgb = self.load_image(rgb_path, False)
            ir = self.load_image(ir_path, True)
            rgb_imgs.append(rgb)
            ir_imgs.append(ir)
        return np.array(rgb_imgs, dtype=np.float32), np.array(np.expand_dims(ir_imgs, axis=3), dtype=np.float32)

    def get_paths(self):
        dir_rgb = Path(self.proj_path + "\\tufts\\RGB-faces-128x128")
        dir_ir = Path(self.proj_path + "\\tufts\\thermal-face-128x128")

        rgb_paths = [str(p) for p in dir_rgb.glob("*")]
        ir_paths = [str(p) for p in dir_ir.glob("*")]

        return rgb_paths, ir_paths

    def load_image(self, path, ir):
        if ir:
            img = Image.open(path).convert('L')
        else:
            img = Image.open(path)
        img = self.process_image(img)
        return img

    def process_image(self, img):
        img = img.resize((256, 256))
        img = np.array(img)
        return img / 255.0
