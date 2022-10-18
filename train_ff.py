import numpy as np
import torch
from torch import optim
from torch import nn
from torchinfo import summary
import tqdm
from patchgan import *

mmap_imgs = '../shuffled_data_b_cropped/train_aug_imgs.npy'
mmap_mask = '../shuffled_data_b_cropped/train_aug_mask.npy'
batch_size= 64
traindata = MmapDataGenerator(mmap_imgs, mmap_mask, batch_size)

mmap_imgs_val = '../shuffled_data_b_cropped/valid_aug_imgs.npy'
mmap_mask_val = '../shuffled_data_b_cropped/valid_aug_mask.npy'
batch_size= 64
val_dl = MmapDataGenerator(mmap_imgs_val, mmap_mask_val, batch_size)

GEN_FILTS  = 32
DISC_FILTS = 16
ACTIV      = 'relu'

# create the generator
norm_layer = get_norm_layer()
generator = UnetGenerator(4, 1, GEN_FILTS, norm_layer=norm_layer, 
                          use_dropout=True, activation=ACTIV)
generator.apply(weights_init)
generator = generator.cuda()

# create the discriminator
discriminator = Discriminator(5, DISC_FILTS, n_layers=3, norm_layer=norm_layer).cuda()
discriminator.apply(weights_init)

summary(generator, [1, 4, 256, 256])

# create the training object and start training
trainer = Trainer(generator, discriminator, 
                  f'checkpoints/checkpoints-{GEN_FILTS}-{DISC_FILTS}-{ACTIV}/')

trainer.fc_beta = 0.75
trainer.fc_gamma = 0.7

G_loss_plot, D_loss_plot = trainer.train(traindata, val_dl, 200, gen_learning_rate=5.e-4, 
                                        dsc_learning_rate=1.e-4, lr_decay=0.95)
        
# save the loss history
np.savez('loss_history.npz', D_loss = D_loss_plot ,G_loss = G_loss_plot)
