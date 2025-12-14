import torch
import numpy as np
import torchmetrics


class Metrics:
    def __init__(self, device='cuda'):
        self.device = device
        self.names_list = ['mse', 'nmse', 'rmse', 'mae', 'psnr', 'ssim']

        self.psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=2.0).to(device)
        self.ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.results = {}
        self.reset()

    def reset(self):
        self.results = {name: 0.0 for name in self.names_list}

    def get_metrics(self):
        return np.array([self.results[n] for n in self.names_list])

    def calculate_metrics(self, generated_image, target_image):
        if generated_image.device != self.device:
            generated_image = generated_image.to(self.device)
            target_image = target_image.to(self.device)

        if generated_image.ndim == 3:
            generated_image = generated_image.permute(2, 0, 1)
            target_image = target_image.permute(2, 0, 1)

            generated_image = generated_image.unsqueeze(0)
            target_image = target_image.unsqueeze(0)

        gen_flat = generated_image.reshape(-1)
        target_flat = target_image.reshape(-1)

        mse = torch.mean((gen_flat - target_flat) ** 2)
        mae = torch.mean(torch.abs(gen_flat - target_flat))

        norm = torch.mean(target_flat ** 2)
        nmse = mse / (norm + 1e-8)

        psnr = self.psnr_metric(generated_image, target_image)
        ssim = self.ssim_metric(generated_image, target_image)

        self.results['mse'] = mse.item()
        self.results['rmse'] = torch.sqrt(mse).item()
        self.results['mae'] = mae.item()
        self.results['nmse'] = nmse.item()
        self.results['psnr'] = psnr.item()
        self.results['ssim'] = ssim.item()

        return self.get_metrics()
