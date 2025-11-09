import torch
import torch.nn as nn

bce_loss = nn.BCEWithLogitsLoss()

def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
    gan_loss = bce_loss(disc_generated_output, torch.ones_like(disc_generated_output))
    l1_loss = torch.mean(torch.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = bce_loss(disc_real_output, torch.ones_like(disc_real_output))
    generated_loss = bce_loss(disc_generated_output, torch.zeros_like(disc_generated_output))
    return real_loss + generated_loss