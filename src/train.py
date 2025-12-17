from torch.utils.data import DataLoader

from src.callbacks import EarlyStopper
from src.discriminator import Discriminator
from src.generator import Generator
import torch.optim as optim

from src.inference import visualize_image
from src.loss import discriminator_loss, generator_loss
from src.metrics import Metrics
from src.logger import Logger
from src.data import Loader, CustomDataset
import torch
from pathlib import Path

#Hyperparameters
model_name = 'benchmark_val2_e200'
epochs = 200
batch_size = 16
training_folds = [1, 3, 4, 5, 6, 7, 8, 9, 10]
validation_folds = [2]

gen_optimizer_lr = 2e-4
disc_optimizer_lr = 2e-4

early_stopper = EarlyStopper(patience=10, min_delta=0)
stop_early = False
#######################################

generator = Generator().cuda()
discriminator = Discriminator().cuda()
gen_optimizer = optim.Adam(generator.parameters(), lr=gen_optimizer_lr, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_optimizer_lr, betas=(0.5, 0.999))

metrics = Metrics()
logger = Logger(f'C:\\Users\\OL4F\\Desktop\\Inzynierka\\SimulatingModalities-BSc\\logs\\{model_name}')
loader = Loader("C:\\Users\\OL4F\\Desktop\\Inzynierka\\SimulatingModalities-BSc\\woundsDB\\data\\")
inference_dir = f'C:\\Users\\OL4F\\Desktop\\Inzynierka\\SimulatingModalities-BSc\\visualizations\\{model_name}\\'
Path(f'C:\\Users\\OL4F\\Desktop\\Inzynierka\\SimulatingModalities-BSc\\visualizations\\{model_name}').mkdir(parents=True, exist_ok=True)

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
        print(f"Step: {step + 1}")
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
            # Generate image
            gen_output = generator(input_image)

            # Discriminator loss
            disc_real_output = discriminator(input_image, target)
            disc_generated_output = discriminator(input_image, gen_output.detach())
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            # Generator loss
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
        print(validation_gen_loss)
        if stop_early and early_stopper.early_stop(validation_gen_loss):
            break

        if epoch % 10 == 0:
            visualize_image(generator.cuda(), validation_dataset, save_dir=f'{inference_dir}\\{epoch}', metrics=True)
        print(f"Completed epoch {epoch + 1}/{epochs}")

    Path(output_dir + model_name).mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), f'{output_dir + model_name}\\generator.pth')
    torch.save(discriminator.state_dict(), f'{output_dir + model_name}\\discriminator.pth')

if __name__ == "__main__":
    training_loop(stop_early, "C:\\Users\\OL4F\\Desktop\\Inzynierka\\SimulatingModalities-BSc\\models\\")