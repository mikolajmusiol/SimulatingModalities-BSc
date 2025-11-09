import torch
import numpy as np
import torchmetrics
from torchmetrics.audio import SignalNoiseRatio

class Metrics():
    def __init__(self):
        self.names_list = ['mse', 'nmse', 'rmse', 'mae', 'snr', 'psnr', 'ssim']
        self.mse = 0
        self.nmse = 0
        self.rmse = 0
        self.mae = 0
        self.snr = 0
        self.psnr = 0
        self.ssim = 0

    def get_metrics(self):
        return np.array([self.mse, self.nmse, self.rmse, self.mae, self.snr, self.psnr, self.ssim])

    def calculate_metrics(self, generated_image, target_image):
        self.mse = self.compute_mse(generated_image, target_image)
        self.nmse = self.compute_nmse(generated_image, target_image)
        self.rmse = self.compute_rmse(generated_image, target_image)
        self.mae = self.compute_mae(generated_image, target_image)

        snr = SignalNoiseRatio()
        self.snr = snr(generated_image, target_image)

        psnr = torchmetrics.image.PeakSignalNoiseRatio()
        ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

        if len(generated_image.shape) == 3 and len(target_image.shape) == 3:
            generated_image = generated_image.unsqueeze(0).permute(0, 3, 1, 2)
            target_image = target_image.unsqueeze(0).permute(0, 3, 1, 2)

        self.psnr = psnr(generated_image, target_image)
        self.ssim = ssim(generated_image, target_image)

    def compute_mse(self, output, target):
        return torch.mean((output - target) ** 2).item()

    def compute_nmse(self, output, target):
        mse = torch.mean((output - target) ** 2)
        norm = torch.mean(target ** 2)
        return (mse / norm).item()

    def compute_rmse(self, output, target):
        mse = self.compute_mse(output, target)
        return torch.sqrt(torch.tensor(mse)).item()

    def compute_mae(self, output, target):
        return torch.mean(torch.abs(output - target)).item()

