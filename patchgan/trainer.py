import torch
import os
import tqdm
import glob
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from .losses import generator_loss, discriminator_loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer:
    '''
        Trainer module which contains both the full training driver
        which calls the train_batch method
    '''

    def __init__(self, generator, discriminator, savefolder):
        '''
            Store the generator and discriminator info
        '''
        self.generator = generator
        self.discriminator = discriminator

        self.savefolder = savefolder
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)

        self.start = 1

    def train_batch(self, x, y):
        '''
            Train the generator and discriminator on a single batch
        '''

        # crop the batch randomly to 256x256
        input_img, target_img = x, y

        # conver the input image and mask to tensors
        input_img = torch.as_tensor(input_img, dtype=torch.float).to(device)
        target_img = torch.as_tensor(target_img, dtype=torch.float).to(device)

        # generate the image mask
        generated_image = self.generator(input_img)

        # get the generator (tversky) loss
        G_loss = generator_loss(generated_image, target_img,
                                beta=self.fc_beta, gamma=self.fc_gamma)

        # train the generator one batch
        self.gen_optimizer.zero_grad()
        G_loss.backward()
        self.gen_optimizer.step()

        # Train the discriminator
        disc_inp_fake = torch.cat((input_img, generated_image), 1)
        disc_inp_real = torch.cat((input_img, target_img), 1)

        D_real = self.discriminator(disc_inp_real)
        D_fake = self.discriminator(disc_inp_fake.detach())

        try:
            D_fake_loss = discriminator_loss(D_fake, self.fake_target_train)
            D_real_loss = discriminator_loss(D_real, self.real_target_train)
        except Exception:
            D_fake_loss = discriminator_loss(
                D_fake, self.fake_target_train[:input_img.shape[0]])
            D_real_loss = discriminator_loss(
                D_real, self.real_target_train[:input_img.shape[0]])

        D_total_loss = D_real_loss + D_fake_loss

        self.disc_optimizer.zero_grad()
        D_total_loss.backward()
        self.disc_optimizer.step()

        return G_loss, D_total_loss

    def test_batch(self, x, y):
        '''
            Train the generator and discriminator on a single batch
        '''

        # crop the batch randomly to 256x256
        input_img, target_img = x, y

        # conver the input image and mask to tensors
        input_img = torch.Tensor(input_img).to(device)
        target_img = torch.Tensor(target_img).to(device)

        # generate the image mask
        generated_image = self.generator(input_img)

        # get the generator (tversky) loss
        G_loss = generator_loss(generated_image, target_img,
                                beta=self.fc_beta, gamma=self.fc_gamma)

        # Train the discriminator
        disc_inp_fake = torch.cat((input_img, generated_image), 1)
        disc_inp_real = torch.cat((input_img, target_img), 1)

        D_real = self.discriminator(disc_inp_real)
        D_fake = self.discriminator(disc_inp_fake.detach())

        try:
            D_fake_loss = discriminator_loss(D_fake, self.fake_target_val)
            D_real_loss = discriminator_loss(D_real, self.real_target_val)
        except Exception:
            D_fake_loss = discriminator_loss(
                D_fake, self.fake_target_val[:input_img.shape[0]])
            D_real_loss = discriminator_loss(
                D_real, self.real_target_val[:input_img.shape[0]])

        D_total_loss = D_real_loss + D_fake_loss

        return G_loss.cpu().item(), D_total_loss.cpu().item()

    def train(self, train_data, val_data, epochs, dsc_learning_rate=1.e-4,
              gen_learning_rate=1.e-3,
              validation_freq=5, save_freq=10, lr_decay=None, decay_freq=5):
        '''
            Training driver which loads the optimizer and calls the
            `train_batch` method. Also handles checkpoint saving
            Inputs
            ------
            train_data : DataLoader object
                Training data that is mapped using the DataLoader or
                MmapDataLoader object defined in patchgan/io.py
            val_data : DataLoader object
                Validation data loaded in using the DataLoader or
                MmapDataLoader object
            epochs : int
                Number of epochs to run the model
            dsc_learning_rate : float [default: 1e-4]
                Initial learning rate for the discriminator
            gen_learning_rate : float [default: 1e-3]
                Initial learning rate for the generator
            validation_freq : int [default: 5]
                Frequency at which to validate the model using the
                validation data
            save_freq : int [default: 10]
                Frequency at which to save checkpoints to the save folder
            lr_decay : float [default: None]
                Learning rate decay rate (ratio of new learning rate
                to previous). A value of 0.95, for example, would set the
                new LR to 95% of the previous value
            decay_freq : int [default: 5]
                Frequency at which to decay the learning rate. For example,
                a value of for decay_freq and 0.95 for lr_decay would decay
                the learning to 95% of the current value every 5 epochs.
            Outputs
            -------
            G_loss_plot : numpy.ndarray
                Generator loss history as a function of the epochs
            D_loss_plot : numpy.ndarray
                Discriminator loss history as a function of the epochs
        '''

        # create the Adam optimzers
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(), lr=gen_learning_rate)
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=dsc_learning_rate)

        # create the output data for the discriminator
        self.real_target_train = torch.ones(
            train_data.batch_size, 1, 30, 30).to(device)
        self.fake_target_train = torch.zeros(
            train_data.batch_size, 1, 30, 30).to(device)

        self.real_target_val = torch.ones(
            val_data.batch_size, 1, 30, 30).to(device)
        self.fake_target_val = torch.zeros(
            val_data.batch_size, 1, 30, 30).to(device)

        # set up the learning rate scheduler with exponential lr decay
        if lr_decay is not None:
            gen_scheduler = ExponentialLR(self.gen_optimizer, gamma=lr_decay)
            dsc_scheduler = ExponentialLR(self.disc_optimizer, gamma=lr_decay)
        else:
            gen_scheduler = None
            dsc_scheduler = None

        gen_lr = gen_learning_rate
        dsc_lr = dsc_learning_rate

        # empty lists for storing epoch loss data
        D_loss_ep, G_loss_ep = [], []
        for epoch in range(self.start, epochs + 1):
            if (gen_scheduler is not None) & (dsc_scheduler is not None):
                gen_lr = gen_scheduler.get_last_lr()[0]
                dsc_lr = dsc_scheduler.get_last_lr()[0]
            else:
                gen_lr = gen_learning_rate
                dsc_lr = dsc_learning_rate

            print(f"Epoch {epoch} -- lr: {gen_lr:5.3e}, {dsc_lr:5.3e}")
            print("-------------------------------------------------------")

            # batch loss data
            pbar = tqdm.tqdm(train_data, desc='Training: ')

            train_data.shuffle()

            # set to training mode
            self.generator.train()
            self.discriminator.train()

            D_loss = torch.zeros(len(train_data) + 1).to(device)
            G_loss = torch.zeros(len(train_data) + 1).to(device)
            # loop through the training data
            for i, (input_img, target_img) in enumerate(pbar):

                # train on this batch
                gen_loss, disc_loss = self.train_batch(input_img, target_img)

                # append the current batch loss
                D_loss[i] = disc_loss.item()
                G_loss[i] = gen_loss.item()

                mean_Gloss = torch.mean(G_loss[:i])
                mean_Dloss = torch.mean(D_loss[:i])

                pbar.set_postfix_str(
                    f"gen: {mean_Gloss:.3e} disc {mean_Dloss:.3e}")

            # update the epoch loss
            D_loss_ep.append(torch.mean(D_loss).cpu().item())
            G_loss_ep.append(torch.mean(G_loss).cpu().item())

            if epoch % validation_freq == 0:
                # validate every `validation_freq` epochs
                self.discriminator.eval()
                self.generator.eval()
                pbar = tqdm.tqdm(val_data, desc='Validation: ')

                val_data.shuffle()

                D_loss = torch.zeros(len(val_data) + 1).to(device)
                G_loss = torch.zeros(len(val_data) + 1).to(device)
                # loop through the training data
                for i, (input_img, target_img) in enumerate(pbar):

                    # train on this batch
                    gen_loss, disc_loss = self.test_batch(
                        input_img, target_img)

                    # append the current batch loss
                    D_loss[i] = disc_loss
                    G_loss[i] = gen_loss

                    mean_Gloss = torch.mean(G_loss[:i])
                    mean_Dloss = torch.mean(D_loss[:i])

                    pbar.set_postfix_str(
                        f'gen: {mean_Gloss:.3e} disc {mean_Dloss:.3e}')

            # apply learning rate decay
            if (gen_scheduler is not None) & (dsc_scheduler is not None):
                if epoch % decay_freq == 0:
                    gen_scheduler.step()
                    dsc_scheduler.step()

            # save checkpoints
            if epoch % save_freq == 0:
                self.save(epoch)

        return G_loss_ep, D_loss_ep

    def save(self, epoch):
        gen_savefile = f'{self.savefolder}/generator_ep_{epoch:03d}.pth'
        disc_savefile = f'{self.savefolder}/discriminator_ep_{epoch:03d}.pth'

        print(f"Saving to {gen_savefile} and {disc_savefile}")
        torch.save(self.generator.state_dict(), gen_savefile)
        torch.save(self.discriminator.state_dict(), disc_savefile)

    def load_last_checkpoint(self):
        gen_checkpoints = sorted(
            glob.glob(self.savefolder + "generator_ep*.pth"))
        disc_checkpoints = sorted(
            glob.glob(self.savefolder + "discriminator_ep*.pth"))

        gen_epochs = set([int(ch.split(
            '/')[-1].replace('generator_ep_', '')[:-4]) for
            ch in gen_checkpoints])
        dsc_epochs = set([int(ch.split(
            '/')[-1].replace('discriminator_ep_', '')[:-4]) for
            ch in disc_checkpoints])

        self.start = max(gen_epochs.union(dsc_epochs))

        assert len(gen_epochs) > 0, "No checkpoints found!"

        self.load(f"{self.savefolder}/generator_ep_{self.start:03d}.pth",
                  f"{self.savefolder}/discriminator_ep_{self.start:03d}.pth")

    def load(self, generator_save, discriminator_save):
        print(generator_save, discriminator_save)
        self.generator.load_state_dict(torch.load(generator_save))
        self.discriminator.load_state_dict(torch.load(discriminator_save))

        gfname = generator_save.split('/')[-1]
        dfname = discriminator_save.split('/')[-1]
        print(
            f"Loaded checkpoints from {gfname} and {dfname}")
