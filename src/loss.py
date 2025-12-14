import torch
import torch.nn as nn


criterion_GAN = nn.MSELoss()
criterion_L1 = nn.L1Loss()


def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
    gan_loss = criterion_GAN(disc_generated_output, torch.ones_like(disc_generated_output))
    l1_loss = criterion_L1(gen_output, target)
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = criterion_GAN(disc_real_output, torch.ones_like(disc_real_output))
    generated_loss = criterion_GAN(disc_generated_output, torch.zeros_like(disc_generated_output))
    total_disc_loss = (real_loss + generated_loss) * 0.5
    return total_disc_loss