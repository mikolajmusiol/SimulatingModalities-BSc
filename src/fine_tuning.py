import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from src.callbacks import EarlyStopper
from src.data import CustomDataset
from src.discriminator import Discriminator
from src.generator import Generator
from src.logger import Logger
from src.metrics import Metrics
from src.data import Loader
from src.inference import visualize_image
from src.loss import discriminator_loss, generator_loss

model_name = 'benchmark_val2_e200_ft_tuft_11'

PATH_TO_OLD_GEN = r"C:\Users\OL4F\Desktop\Inzynierka\SimulatingModalities-BSc\models\tuft11_5\generator.pth"
PATH_TO_OLD_DISC = r"C:\Users\OL4F\Desktop\Inzynierka\SimulatingModalities-BSc\models\tuft11_5\discriminator.pth"

NEW_DATA_PATH = "C:\\Users\\OL4F\\Desktop\\Inzynierka\\SimulatingModalities-BSc\\woundsDB\\data\\"

epochs = 200
batch_size = 16
training_folds = [1,3,4,5,6,7,8,9,10]
validation_folds = [2]

gen_optimizer_lr = 1e-5
disc_optimizer_lr = 1e-5

def load_weights_safe(model, path):
    try:
        pretrained_dict = torch.load(path)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Weights loaded: {len(pretrained_dict)})")
    except Exception as e:
        print(f"Error: {e}")


early_stopper = EarlyStopper(patience=10, min_delta=0)
stop_early = False

generator = Generator().cuda()
discriminator = Discriminator().cuda()

load_weights_safe(generator, PATH_TO_OLD_GEN)
load_weights_safe(discriminator, PATH_TO_OLD_DISC)

for param in generator.parameters():
    param.requires_grad = False

for name, param in generator.named_parameters():
    if 'up_stack' in name or 'last' in name:
        param.requires_grad = True

gen_optimizer = optim.Adam(generator.parameters(), lr=gen_optimizer_lr, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_optimizer_lr, betas=(0.5, 0.999))

metrics = Metrics()
logger = Logger(f'C:\\Users\\OL4F\\Desktop\\Inzynierka\\SimulatingModalities-BSc\\logs\\{model_name}')

loader = Loader(NEW_DATA_PATH)
inference_dir = f'C:\\Users\\OL4F\\Desktop\\Inzynierka\\SimulatingModalities-BSc\\visualizations\\{model_name}\\'
Path(inference_dir).mkdir(parents=True, exist_ok=True)

train_rgb_images, train_ir_images = loader.load(folds=training_folds)
train_dataset = CustomDataset(train_rgb_images, train_ir_images, augment=False)

validation_rgb_images, validation_ir_images = loader.load(folds=validation_folds)
validation_dataset = CustomDataset(validation_rgb_images, validation_ir_images, augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


def train_step(input_image, target):
    gen_optimizer.zero_grad()
    disc_optimizer.zero_grad()

    # Generate image
    gen_output = generator(input_image)

    # Discriminator loss
    disc_real_output = discriminator(input_image, target)
    disc_generated_output = discriminator(input_image, gen_output.detach())
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # Generator loss
    disc_generated_output = discriminator(input_image, gen_output)
    gen_total_loss, gan_loss, l1_loss = generator_loss(disc_generated_output, gen_output, target)

    # Backprop
    gen_total_loss.backward()
    gen_optimizer.step()

    disc_loss.backward()
    disc_optimizer.step()

    return gen_output, gen_total_loss, disc_loss


def train_epoch(data_loader, epoch):
    generator.train()
    discriminator.train()

    for step, (input_image, target) in enumerate(data_loader):
        input_image = input_image.cuda()
        target = target.cuda()
        gen_output, gen_total_loss, disc_loss = train_step(input_image, target)

        print(f"Step: {step + 1} | Gen Loss: {gen_total_loss.item():.4f} | Disc Loss: {disc_loss.item():.4f}")

        metrics.calculate_metrics(gen_output.detach().cpu(), target.cpu())
        logger.log_train(gen_total_loss.item(), disc_loss.item(), metrics, epoch * len(data_loader) + step + 1)


def validate_epoch(data_loader, epoch):
    generator.eval()
    discriminator.eval()

    validation_gen_loss = 0
    with torch.no_grad():
        for step, (input_image, target) in enumerate(data_loader):
            input_image = input_image.cuda()
            target = target.cuda()

            gen_output = generator(input_image)

            disc_real_output = discriminator(input_image, target)
            disc_generated_output = discriminator(input_image, gen_output.detach())
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            disc_generated_output = discriminator(input_image, gen_output)
            gen_total_loss, gan_loss, l1_loss = generator_loss(disc_generated_output, gen_output, target)

            print(f"Validation step: {step + 1}")
            metrics.calculate_metrics(gen_output.detach().cpu(), target.cpu())
            logger.log_validation(gen_total_loss.item(), disc_loss.item(), metrics, epoch * len(data_loader) + step + 1)
            validation_gen_loss += gen_total_loss.item()
    return validation_gen_loss


def training_loop(stop_early, output_dir):
    for epoch in range(epochs):
        train_epoch(train_loader, epoch)
        validation_gen_loss = validate_epoch(validation_loader, epoch)
        print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {validation_gen_loss:.4f}")

        if stop_early and early_stopper.early_stop(validation_gen_loss):
            print("Early stopping triggered!")
            break

        if epoch % 10 == 0:
            visualize_image(generator.cuda(), validation_dataset, save_dir=f'{inference_dir}\\{epoch}', metrics=True)

    final_output_path = output_dir + model_name
    Path(final_output_path).mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), f'{final_output_path}\\generator.pth')
    torch.save(discriminator.state_dict(), f'{final_output_path}\\discriminator.pth')


if __name__ == '__main__':
    training_loop(stop_early, "C:\\Users\\OL4F\\Desktop\\Inzynierka\\SimulatingModalities-BSc\\models\\")