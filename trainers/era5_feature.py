import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils import data

from misc.ops import *
from models.model_spatial import NN_embedding

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

class ERA5_feature():

    def __init__(self, feat_name, coordinates, params):

        self.feat_name = feat_name
        self.latitude = coordinates['latitude']
        self.longitude = coordinates['longitude']
        self.Lon_grid, self.Lat_grid = np.meshgrid (
            self.longitude,
            self.latitude
        )
        self.grid_dim = self.Lon_grid.shape

        # define parameters
        self.dataloader_params, self.trainer_params, self.model_params = params

        # initialize model, optimizer and lr schedulers
        self._init_model()
        self._init_optimizer()
        self._init_lr_scheduler()

        # initialize embedding and reconstruction loss
        self.var_batch_loss = {'train': [], 'val': [], 'test': []}
        self.rec_batch_loss = {'train': [], 'val': [], 'test': []}
        self.rec_epoch_loss_history = {'train': [], 'val': [], 'test': []}
        self.var_epoch_loss_history = {'train': [], 'val': [], 'test': []}

        self.SAVE_DIR = os.path.join(self.dataloader_params.root_path, 'loss', feat_name)
        os.makedirs(self.SAVE_DIR, exist_ok=True)

    def _init_model(self):
        self.model = NN_embedding (
            params = self.model_params,
            grid_dim = self.grid_dim,
            seq_len = self.dataloader_params.len_inp_time_series
        )
        if self.dataloader_params.use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()

    def _init_optimizer(self):
        self.optimizer = optim.Adam (
            self.model.parameters(), 
            lr=self.trainer_params.learning_rate
        )

    def _init_lr_scheduler(self):
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=self.trainer_params.lr_patience, 
            factor=self.trainer_params.lr_reduce_factor,
            verbose=True,
            mode=self.trainer_params.lr_schedule_mode,
            cooldown=self.trainer_params.lr_cooldown, 
            min_lr=self.trainer_params.min_lr
        )

    def step_lr_scheduler(self, split='val'):
        self.lr_scheduler.step(self.rec_epoch_loss_history[split][-1])

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        self.optimizer.step()

    def train(self):
        self.model.train()
        self.init_loss(split='train')

    def eval(self, split=None):
        self.model.eval()
        if split is not None:
            self.init_loss(split)

    def encode(self, x):
        return self.model(x)

    def decode(self, z):
        out_len = z.shape[1]
        return self.model.decode(z, out_len)

    def init_loss(self, split):
        self.var_batch_loss[split] = []
        self.rec_batch_loss[split] = []

    def update_loss(self, loss, split):
        self.rec_batch_loss[split].append(loss['rec'])
        self.var_batch_loss[split].append(loss['var'])

    def load(self, checkpoint):
        self.model = torch.load(checkpoint)

    def save(self, model_path):
        torch.save(self.model, model_path)

    def append_loss(self):
        for split in ['train', 'val', 'test']:
            loss_rec = sum(self.rec_batch_loss[split]) / len(self.rec_batch_loss[split])
            self.rec_epoch_loss_history[split].append(loss_rec)
            if self.model_params.use_VAE:
                loss_var = sum(self.var_batch_loss[split]) / len(self.var_batch_loss[split])
                self.var_epoch_loss_history[split].append(loss_var)
            self.init_loss(split)

    def plot(self, x, coords):
        self.plot_loss()
        self.plot_random_reconstruction(x, coords)

    def plot_loss(self):
        tags = ['rec', 'var'] if self.model_params.use_VAE else ['rec']
        for tag in tags:
            for split in ['train', 'val', 'test']:
                if tag == 'rec':
                    plt.plot(self.rec_epoch_loss_history[split], label=split)
                else:
                    plt.plot(self.var_epoch_loss_history[split], label=split)
            plt.ylabel('{} loss'.format(tag))
            plt.xlabel('epoch')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(self.SAVE_DIR, 'loss_{}.png'.format(tag)))
            plt.close()

    def plot_random_reconstruction(self, x, time_seq):
        self.eval()
        perm1 = torch.randperm(x.size(0))
        perm2 = torch.randperm(x.size(1))

        idx1 = perm1[0]
        idx2 = perm2[0]

        if self.dataloader_params.use_cuda and torch.cuda.is_available():
            x = x.cuda()

        y = self.model.output(x)

        x = x.detach().cpu().numpy()[idx1,idx2,:,:].squeeze()
        y = y.detach().cpu().numpy()[idx1,idx2,:,:].squeeze()
        time = time_seq[idx1, idx2]

        fig = plt.figure(figsize=(10,20))

        gs = fig.add_gridspec(1,2)

        for i in range(2):
            fig.add_subplot(gs[0,i])
            m = Basemap(projection='merc', llcrnrlat=np.min(self.latitude),\
                        urcrnrlat=min(np.max(self.latitude),85), llcrnrlon=np.min(self.longitude),\
                        urcrnrlon=np.max(self.longitude),resolution='c')
            m.drawcoastlines()
            m.drawstates()
            m.drawcountries(linewidth=1, linestyle='solid', color='blue')
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            lon, lat = m(self.Lon_grid, self.Lat_grid)
            if i == 0:
                plt.contourf(lon, lat, x, cmap='jet', levels=np.linspace(-1,1, 41), extend='both')
            else:
                plt.contourf(lon, lat, y, cmap='jet', levels=np.linspace(-1,1, 41), extend='both')
            plt.colorbar(shrink=0.1)
            plt.title(time)
        plt.tight_layout()
        plt.savefig(os.path.join(self.SAVE_DIR, 'reconstruction.png'))
        plt.close()