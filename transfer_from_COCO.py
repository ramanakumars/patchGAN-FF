import os
import numpy as np
import torch
from torch import optim
from torch import nn
from torchinfo import summary
import tqdm
from patchgan.trainer import Trainer, device
from patchgan.unet import UnetGenerator, get_norm_layer, weights_init, Discriminator
from patchgan.io import MmapDataGenerator

mmap_imgs = '../shuffled_data_b_cropped/train_aug_imgs.npy'
mmap_mask = '../shuffled_data_b_cropped/train_aug_mask.npy'
batch_size= 16
traindata = MmapDataGenerator(mmap_imgs, mmap_mask, batch_size)

mmap_imgs_val = '../shuffled_data_b_cropped/valid_aug_imgs.npy'
mmap_mask_val = '../shuffled_data_b_cropped/valid_aug_mask.npy'
batch_size= 16
val_dl = MmapDataGenerator(mmap_imgs_val, mmap_mask_val, batch_size)

GEN_FILTS  = 32
DISC_FILTS = 16
ACTIV      = 'relu'

# create the generator
norm_layer = get_norm_layer()
generator = UnetGenerator(4, 1, GEN_FILTS, norm_layer=norm_layer, 
                          use_dropout=True, activation=ACTIV)
generator.apply(weights_init)
generator.load_transfer_data(
    torch.load('checkpoints/generator_COCO.pth', map_location=device))
generator = generator.to(device)

# create the discriminator
discriminator = Discriminator(5, DISC_FILTS, n_layers=3,
                              norm_layer=norm_layer).to(device)
generator.load_transfer_data(
    torch.load('checkpoints/discriminator_COCO.pth', map_location=device))

summary(generator, [1, 4, 256, 256])

# create the training object and start training
trainer = Trainer(generator, discriminator, 
                  f'checkpoints/checkpoints-COCO/')

trainer.fc_beta = 0.7
trainer.fc_gamma = 0.75

G_loss_plot, D_loss_plot = trainer.train(traindata, val_dl, 200, gen_learning_rate=5.e-4, 
                                        dsc_learning_rate=5.e-4, lr_decay=0.95)
        
# save the loss history
np.savez('loss_history.npz', D_loss = D_loss_plot ,G_loss = G_loss_plot)

