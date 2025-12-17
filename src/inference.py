import torch
from matplotlib import patches
from matplotlib.widgets import RectangleSelector

from src.metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np

def visualize_image(model, dataset, image=0, error_threshold=None, save_dir=None, metrics=False, roi=None):
    model.cuda()
    model.eval()

    input_tensor = dataset.rgb_images[image:image + 1]
    target_image = dataset.ir_images[image:image + 1][0].detach()

    generated_image = model(input_tensor.cuda())

    generated_image_np = generated_image.cpu().detach().numpy()[0].transpose((1, 2, 0))
    target_image_np = target_image.numpy().transpose((1, 2, 0))
    input_image_np = input_tensor[0].detach().numpy().transpose((1, 2, 0))

    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    ax[0, 0].imshow(input_image_np)
    ax[0, 0].set_title("Input Image", fontsize=10)

    ax[0, 1].imshow(target_image_np)
    ax[0, 1].set_title("Target Image", fontsize=10)

    ax[1, 0].imshow(generated_image_np)
    ax[1, 0].set_title("Generated Image", fontsize=10)

    error_image = np.abs(target_image_np - generated_image_np)

    max_min_diff = error_image.max() - error_image.min()
    if max_min_diff == 0:
        normalized_difference = np.zeros_like(error_image, dtype=np.uint8)
    else:
        normalized_difference = (((error_image - error_image.min()) / max_min_diff) * 255).astype(np.uint8)

    difference = 255 - normalized_difference
    if error_threshold is not None:
        difference = np.where(difference > 255 - error_threshold, 255, 0)

    ax[1, 1].imshow(difference, cmap="Grays")
    ax[1, 1].set_title("Error visualization", fontsize=10)

    if roi is not None:
        x, y, w, h = roi
        rect_target = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        rect_gen = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        rect_input = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')

        ax[0, 1].add_patch(rect_target)
        ax[1, 0].add_patch(rect_gen)
        ax[0, 0].add_patch(rect_input)

    if metrics:
        metrics_obj = Metrics()
        metrics_obj.calculate_metrics(
            torch.from_numpy(generated_image_np),
            torch.from_numpy(target_image_np),
            roi=roi
        )

        metrics_array = []
        for name, value in zip(metrics_obj.names_list, metrics_obj.get_metrics()):
            metrics_array.append(f"{name}: {round(value, 3)}")

        plt.subplots_adjust(bottom=0.1)
        f.tight_layout()

        header = "ROI Metrics:\n" if roi else "Full Image Metrics:\n"
        text = header + str.join('\n', metrics_array)

        f.text(0, 0, text, va='top', ha='left', fontsize=10)

    if save_dir is not None:
        f.savefig(save_dir + '.png', bbox_inches='tight')

    plt.show()
    plt.close(fig=f)
