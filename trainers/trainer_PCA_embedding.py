import os
import numpy as np
import torch
import torch.optim as optim
import wandb
from models.model_spatial  import PCA_embedding
from models.model_temporal import TCN
from dataloaders.dataloader import ERA5_dataloader
from misc.ops import loss_function, get_log_paths
from trainers.utils import *

class Trainer(object):
    """Trainer class for S2S Forecasting with PCA embeddings
    Attributes
    ----------
    dataloader_params : argparse params
    trainer_params : argparse params
    model_params : argparse params
    root_path : str
        root directory for storing checkpoints, scale and predictions
    log_path : str
        directory for storing checkpoints "root_path/checkpoints/"
    wandb_name : str
        name of WANDB logger
    ERA5_data : Instance of <ERA5_Dataloader> class
        contains dictionary of PyTorch dataloaders for <ERA5_Dataset> 
        PyTorch Dataset class
    ERA5_featues_list : list of str
        list of ERA5 features in the dataset
    spatial_embedding_dim : int
        dimension of spatial embeddings to be used
        for PCA embeddings, it is equal to num_pca_modes
    scale_dict : dict of {str: dict of {str: xarray dataset}}
        keys: feature
        vals: dict
            keys: 'mean', 'std'
            vals: xarray dataset
    spatial_embedding_model_dict : dict of {str: custom PyTorch Models}
        keys: feature name
        vals: PyTorch Models for learning spatial embeddings 
        (inherits torch.nn.module)
    temporal_model : PyTorch Model (inherits torch.nn.module)
        model for temporal S2S modelling using ERA5 data
    optimizer : PyTorch optimizer <torch.optim>
        optimizer for temporal model
    Parameters
    ----------
    params : list of argparse params
        (dataloader_params, trainer_params, model_params)
    Methods
    -------
    fit()
        fits/trains the model on the training data and save the best model
    predict(x)
        returns predictions from current state of the model
    load()
        loads the model from specified checkpoint
    save_predictions(split)
        generates prediction for provided split
        calculates anomaly, prints skill and save forecasts
    Note
    ----
    All other methods are assumed private and are meant for internal processing only
    """
    def __init__(self, params):
        self.dataloader_params, self.trainer_params, self.model_params = params

        # Define logger
        if self.trainer_params.operating_mode == 'train':
            self.root_path, self.log_path, self.wandb_name = get_log_paths(self.trainer_params.run,\
                                                                      'TCN',\
                                                                      'PCA_embedding')
        elif self.trainer_params.operating_mode == 'load':
            if self.trainer_params.ckpt_timestamp == None:
                raise Exception("please provide trainer_params.ckpt_timestamp")
            else:
                self.root_path, self.log_path, _ = get_log_paths(self.trainer_params.run, \
                                                            'TCN',\
                                                            'PCA_embedding',\
                                                            self.trainer_params.ckpt_timestamp)
        print('-'*90)
        print('ROOT_PATH = {}'.format(self.root_path))
        print('-'*90)
        # get DataLoaders for train, val, test splits
        self.dataloader_params.root_path = self.root_path
        self.ERA5_data = ERA5_dataloader(self.dataloader_params)

        # Define embedding model_dict
        self.ERA5_features_list = self.ERA5_data.ERA5_features_list
        self.spatial_embedding_dim = self.model_params.spatial_embedding_dim
        self.scale_dict = self.ERA5_data.scale_dict

        self.spatial_embedding_model_dict = {} # to be defined in fit method
        # NOTE: for all other trainers with NN based embeddings, the embedding
        # models should be defined in the __init__ itself

        self.indices_list = self.ERA5_data.indices_list
        inp_dim = len(self.ERA5_features_list)*self.spatial_embedding_dim + len(self.indices_list)
        self.temporal_model = TCN (
                params  = self.model_params,
                inp_dim = inp_dim,
                out_dim = self.spatial_embedding_dim,
                out_len = self.dataloader_params.len_out_time_series
            )

        if self.dataloader_params.use_cuda and torch.cuda.is_available():
            self.temporal_model = self.temporal_model.cuda()

        # Define optimizer and learning rate scheduler
        self.optimizer = optim.Adam(self.temporal_model.parameters(), lr=self.trainer_params.learning_rate)

    def fit(self):
        """fits/trains PyTorch model on the training data and save checkpoints"""
        print('-'*90)
        print('Generating PCA embeddings ...')
        self.spatial_embedding_model_dict = {}
        os.makedirs(os.path.join(self.log_path, 'spatial'), exist_ok=True)
        for feat in self.ERA5_features_list:
            normalized_feat = self.ERA5_data.dataloader_dict['train'].dataset.ds_dict[feat].values
            self.spatial_embedding_model_dict[feat] = PCA_embedding (
                feature=normalized_feat,
                n_modes=self.spatial_embedding_dim
            )
            explained_var= self.spatial_embedding_model_dict[feat].explained_variance_.numpy()*100
            print('{}: explained variance = {:2.2f}%'.format(feat, explained_var))
            model_path = os.path.join(self.log_path, 'spatial', '{}_pca.pth'.format(feat))
            torch.save(self.spatial_embedding_model_dict[feat], model_path)        
        
        print('-'*90)
        print('Training temporal model ...')
        #print(self.temporal_model)
        print('-'*90)

        best_loss = float('inf')
        config = {'dataloader_params': vars(self.dataloader_params),
                 'model_params': vars(self.model_params),
                 'trainer_params': vars(self.trainer_params)}

        wandb.init(name=self.wandb_name, entity=self.trainer_params.wandb_entity, project='s2s_forecasting', config=config)
        wandb.watch(self.temporal_model)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,\
                            patience=self.trainer_params.lr_patience,\
                            factor=self.trainer_params.lr_reduce_factor,\
                            verbose=True, mode=self.trainer_params.lr_schedule_mode,\
                            cooldown=self.trainer_params.lr_cooldown,\
                            min_lr=self.trainer_params.min_lr)
        print('-'*90)
        for epoch in range(self.trainer_params.epochs):
            train_loss = self._train()
            val_loss   = self._eval(self.ERA5_data.dataloader_dict['val'])
            test_loss  = self._eval(self.ERA5_data.dataloader_dict['test'])
            print("Epoch: {}, train_loss: {}, val_loss: {}, test_loss: {}"\
                    .format(epoch, train_loss, val_loss, test_loss))
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss
            })
            if self.trainer_params.schedule_lr:
                lr_scheduler.step(val_loss)

            if epoch % self.trainer_params.save_interval == 0:
                model_path = os.path.join(self.log_path, "net_{}.pth".format(epoch))
                torch.save(self.temporal_model.state_dict(), model_path)
                if val_loss < best_loss:
                    model_path = os.path.join(self.log_path, "net_best_so_far.pth")
                    torch.save(self.temporal_model.state_dict(), model_path)
                    best_loss = val_loss

    def _train(self):
        """Training pass for a single epoch
        Returns
        -------
        loss : float
            training loss for single epoch
        """
        self.temporal_model.train()
        n_batches = len(self.ERA5_data.dataloader_dict['train'])
        epoch_loss = 0.0
        for idx, sample in enumerate(self.ERA5_data.dataloader_dict['train']):
            print('batch: {}/{}'.format(idx+1,n_batches), end='\r')

            x_dict, y_true = sample[0], sample[1]
            if self.trainer_params.use_cuda and torch.cuda.is_available():
                for key in x_dict.keys():
                    x_dict[key] = x_dict[key].cuda()
                y_true = y_true.cuda()

            self.optimizer.zero_grad()

            y_pred = self.predict(x_dict)

            loss = loss_function(y_true, y_pred, self.trainer_params.loss_type)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.detach().cpu().item() 

        epoch_loss /= n_batches
        return epoch_loss

    def _eval(self, dloader):
        """Evaluation pass for a single epoch
        Returns
        -------
        loss : float
            validation/test loss for single epoch
        """
        self.temporal_model.eval()
        n_batches = len(dloader)
        epoch_loss = 0.0
        for idx, sample in enumerate(dloader):
            x_dict, y_true = sample[0], sample[1]
            if self.trainer_params.use_cuda and torch.cuda.is_available():
                for key in x_dict.keys():
                    x_dict[key] = x_dict[key].cuda()
                y_true = y_true.cuda()

            y_pred = self.predict(x_dict)

            loss = loss_function(y_true, y_pred, self.trainer_params.loss_type)
            epoch_loss += loss.detach().cpu().item() 

        epoch_loss /= n_batches
        return epoch_loss

    def predict(self, x):
        """Returns predictions from current state of model
        Parameters
        ----------
        x : Pytorch Tensor
        Returns
        -------
        Pytorch Tensor
            output from the model
        """
        x_list = [self.spatial_embedding_model_dict[feat](x[feat]) for feat in self.ERA5_features_list]
        x = x_list + [x['indices']]
        x = torch.cat(x, axis=-1)
        x = self.temporal_model(x)
        return self.spatial_embedding_model_dict['t2m'].inverse(x)

    def load(self):
        """Loads the best/latest checkpoint from log_path"""
        if self.log_path is None:
            raise Exception("Invalid ckpt_timestamp: no checkpoints exist")
        if self.trainer_params.ckpt == 'best':
            checkpoint_path = os.path.join(self.log_path, 'net_best_so_far.pth')
        elif self.trainer_params.ckpt == 'latest':
            filename_list = os.listdir(self.log_path)
            filepath_list = [os.path.join(self.log_path, fname) for fname in filename_list]
            checkpoint_path = sorted(filepath_list, key=os.path.getmtime)[-1]
        elif self.trainer_params.ckpt_epoch is not None:
            checkpoint_path = os.path.join(self.log_path, 'net_{}.pth'.format(self.trainer_params.ckpt_epoch))
        else:
            raise Exception("please provide trainer_params.ckpt_epoch")

        print('-'*90)
        print('Loading model from: <{}>'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.temporal_model.load_state_dict(checkpoint)
        for feat in self.ERA5_features_list:
            checkpoint = os.path.join(self.log_path, 'spatial', '{}_pca.pth'.format(feat))
            self.spatial_embedding_model_dict[feat] = torch.load(checkpoint)

        print('-'*90)


    def save_predictions(self, split='test'):
        """Saves predictions and prints metrics
        Generates predictions for provided train/val/test split
        Saved forecast as a netcdf file
        Parameters
        ----------
        split : str
            train/val/test split
        """
        self.temporal_model.eval()
        print('Getting model predictions for {}-set ...'.format(split))
        dataloader = self.ERA5_data.dataloader_dict[split]
        with torch.no_grad():
            y_pred_series = []
            y_true_series = []
            inp_dates_seq = []
            out_dates_seq = []
            for idx, sample in enumerate(dataloader):
                x, y_true, dates = sample[0], sample[1], sample[2]
                if self.trainer_params.use_cuda and torch.cuda.is_available():
                    for key in x.keys():
                        x[key] = x[key].cuda()
                    y_true = y_true.cuda()
                #y_true = self.spatial_embedding_model_dict['t2m'](y_true)
                y_true_series.append(y_true)
                y_pred_series.append(self.predict(x))
                inp_dates_seq.append(dates[0])
                out_dates_seq.append(dates[1])
        y_true_series = torch.cat(y_true_series, dim=0).cpu().detach().numpy()
        y_pred_series = torch.cat(y_pred_series, dim=0).cpu().detach().numpy()
        inp_dates_seq = torch.cat(inp_dates_seq, dim=0).cpu().detach().numpy()
        out_dates_seq = torch.cat(out_dates_seq, dim=0).cpu().detach().numpy()
        save_preds (
            y_true_series, y_pred_series,
            inp_dates_seq, out_dates_seq,
            self.scale_dict['t2m'],
            self.dataloader_params,
            split
        )