import torch

from src.metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np

def visualize_image(model, dataset, image=0, error_threshold=None, save_dir=None, metrics=False):
    model.cuda()
    model.eval()

    input_tensor = dataset.rgb_images[image:image + 1]
    target_image = dataset.ir_images[image:image + 1][0].detach()

    generated_image = model(input_tensor.cuda())

    generated_image = generated_image.cpu().detach().numpy()[0].transpose((1, 2, 0))
    target_image = target_image.numpy().transpose((1, 2, 0))

    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8))

    ax[0,0].imshow(input_tensor[0].detach().numpy().transpose((1, 2, 0)))
    ax[0,0].set_title("Input Image", fontsize=10)

    ax[0,1].imshow(target_image)
    ax[0,1].set_title("Target Image", fontsize=10)

    ax[1,0].imshow(generated_image)
    ax[1,0].set_title("Generated Image", fontsize=10)

    error_image = np.abs(target_image - generated_image)
    normalized_difference = (((error_image - error_image.min()) / (error_image.max() - error_image.min())) * 255).astype(np.uint8)
    difference = 255 - normalized_difference
    if error_threshold is not None:
        difference = np.where(difference > 255 - error_threshold, 255, 0)

    ax[1, 1].imshow(difference, cmap="Grays")
    ax[1, 1].set_title("Error visualization", fontsize=10)

    if metrics:
        metrics = Metrics()
        metrics.calculate_metrics(torch.from_numpy(generated_image), torch.from_numpy(target_image))

        metrics_array = []
        for name, value in zip(metrics.names_list, metrics.get_metrics()):
            metrics_array.append(f"{name}: {round(value, 3)}")

        plt.subplots_adjust(bottom=0.1)
        f.tight_layout()
        text = str.join('\n', metrics_array)
        f.text(0, 0, text, va='top', ha='left', fontsize=10)

    if save_dir is not None:
        f.savefig(save_dir+'.png', bbox_inches='tight')

    plt.show()
    plt.close(fig=f)
