import os
import torch
import torch.optim as optim
import wandb
from models.model_temporal import TCN
from dataloaders.dataloader import ERA5_dataloader
from misc.ops import loss_function, get_log_paths
from trainers.utils import *
from trainers.era5_feature import ERA5_feature

class Trainer(object):
    """Trainer class for S2S Forecasting with autoencoder embeddings
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
    scale_dict : dict of {str: dict of {str: xarray dataset}}
        keys: feature
        vals: dict
            keys: 'mean', 'std'
            vals: xarray dataset
    spatial_embedding_dict : dict of {str: class <ERA5_feature> instance}
        keys: feature name
        vals: class <ERA5_feature> instance
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
    predict(x, loss=False)
        returns predictions from current state of the model
        also returns embedding and reconstruction loss if loss == True
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
                                                                      'NN_embedding')
        elif self.trainer_params.operating_mode == 'load':
            if self.trainer_params.ckpt_timestamp == None:
                raise Exception("please provide trainer_params.ckpt_timestamp")
            else:
                self.root_path, self.log_path, _ = get_log_paths(self.trainer_params.run, \
                                                            'TCN',\
                                                            'NN_embedding',\
                                                            self.trainer_params.ckpt_timestamp)

        os.makedirs(os.path.join(self.log_path, 'spatial'), exist_ok=True)
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

        self.spatial_embedding_dict = {}
        split = list(self.ERA5_data.dataloader_dict.keys())[0]
        for feat in self.ERA5_features_list:
            coords = {
                'latitude': self.ERA5_data.dataloader_dict[split].dataset.ds_dict[feat].latitude.values,
                'longitude': self.ERA5_data.dataloader_dict[split].dataset.ds_dict[feat].longitude.values
            }
            self.spatial_embedding_dict[feat] = ERA5_feature (
                feat_name = feat,
                coordinates = coords,
                params = params
            )

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
        self.optimizer = optim.Adam (
                self.temporal_model.parameters(), 
                lr=self.trainer_params.learning_rate
            )

        self.epoch = 0

    def fit(self):
        """fits/trains PyTorch model on the training data and save checkpoints"""
        print('-'*90)
        print('-'*90)
        print('Training temporal and spatial model (end to end) ...')
        print('-'*90)

        best_loss = float('inf')
        config = {'dataloader_params': vars(self.dataloader_params),
                 'model_params': vars(self.model_params),
                 'trainer_params': vars(self.trainer_params)}

        wandb.init(name=self.wandb_name, entity='nishant-parashar', project='s2s_forecasting', config=config)
        wandb.watch(self.temporal_model)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau (
                self.optimizer,
                patience=self.trainer_params.lr_patience,
                factor=self.trainer_params.lr_reduce_factor,
                verbose=True, mode=self.trainer_params.lr_schedule_mode,
                cooldown=self.trainer_params.lr_cooldown,
                min_lr=self.trainer_params.min_lr
            )

        print('-'*90)
        for epoch in range(self.trainer_params.epochs):
            self.epoch = epoch
            train_loss = self._train()
            val_loss   = self._eval(split='val')
            test_loss  = self._eval(split='test')

            self._append_embedding_loss()

            self._display_epoch_loss(train_loss, val_loss, test_loss)

            loss_dict = {
                'train_loss': train_loss['tcn'],
                'val_loss': val_loss['tcn'],
                'test_loss': test_loss['tcn']
            }
            wandb.log(loss_dict)

            for feat in self.ERA5_features_list:
                self.spatial_embedding_dict[feat].step_lr_scheduler(split='val')

            if self.trainer_params.schedule_lr:
                lr_scheduler.step(val_loss['tcn'])

            if epoch % self.trainer_params.save_interval == 0:
                self._plot_embeddings(split='test')
                self._save_model(epoch)
                if val_loss['tcn'] < best_loss:
                    self._save_model(best=True)
                    best_loss = val_loss['tcn']

    def _train(self):
        """Training pass for a single epoch
        Returns
        -------
        loss : dict of {str: float}
            key: 'tcn', 'rec', 'var'
            vals: training loss for single epoch
        """
        self.temporal_model.train()
        for era5_feat in self.spatial_embedding_dict.values():
            era5_feat.train()

        dloader = self.ERA5_data.dataloader_dict['train']
        n_batches = len(dloader)

        epoch_loss_tcn = 0.0
        epoch_loss_rec = 0.0
        epoch_loss_var = 0.0
        for idx, sample in enumerate(dloader):
            print('batch: {}/{}'.format(idx+1,n_batches), end='\r')

            x_dict, y_true = sample[0], sample[1]
            if self.trainer_params.use_cuda and torch.cuda.is_available():
                for key in x_dict.keys():
                    x_dict[key] = x_dict[key].cuda()
                y_true = y_true.cuda()

            for feat in self.spatial_embedding_dict.values():
                feat.optimizer_zero_grad()
            self.optimizer.zero_grad()

            y_pred, loss_rec, loss_var = self.predict(x_dict, loss=True, split='train')

            loss_tcn = loss_function(y_true, y_pred, self.trainer_params.loss_type)
            loss = loss_rec + loss_var + loss_tcn
            loss.backward()

            for era5_feat in self.spatial_embedding_dict.values():
                era5_feat.optimizer_step()
            self.optimizer.step()
            epoch_loss_tcn += loss_tcn.detach().cpu().item()
            epoch_loss_rec += loss_rec.detach().cpu().item()
            epoch_loss_var += loss_var.detach().cpu().item()

        epoch_loss_tcn /= n_batches
        epoch_loss_rec /= n_batches
        epoch_loss_var /= n_batches
        return {
            'tcn': epoch_loss_tcn,
            'rec': epoch_loss_rec,
            'var': epoch_loss_var
        }

    def _eval(self, split):
        """Evaluation pass for a single epoch
        Returns
        -------
        loss : dict of {str: float}
            key: 'tcn', 'rec', 'var'
            vals: training loss for single epoch
        """
        self.temporal_model.eval()
        for era5_feat in self.spatial_embedding_dict.values():
            era5_feat.eval(split)

        dloader = self.ERA5_data.dataloader_dict[split]
        n_batches = len(dloader)

        epoch_loss_tcn = 0.0
        epoch_loss_rec = 0.0
        epoch_loss_var = 0.0
        for idx, sample in enumerate(dloader):

            x_dict, y_true = sample[0], sample[1]
            if self.trainer_params.use_cuda and torch.cuda.is_available():
                for key in x_dict.keys():
                    x_dict[key] = x_dict[key].cuda()
                y_true = y_true.cuda()

            y_pred, loss_rec, loss_var = self.predict(x_dict, loss=True, split=split)

            loss_tcn = loss_function(y_true, y_pred, self.trainer_params.loss_type)

            epoch_loss_tcn += loss_tcn.detach().cpu().item()
            epoch_loss_rec += loss_rec.detach().cpu().item()
            epoch_loss_var += loss_var.detach().cpu().item()

        epoch_loss_tcn /= n_batches
        epoch_loss_rec /= n_batches
        epoch_loss_var /= n_batches
        return {
            'tcn': epoch_loss_tcn,
            'rec': epoch_loss_rec,
            'var': epoch_loss_var
        }

    def predict(self, x, loss=False, split='test'):
        """Returns predictions from current state of model
        Parameters
        ----------
        x : dict of {str: Pytorch Tensor}
            key:  feature name
            vals: PyTorch tensor of feature embeddings
        loss : bool
            if loss == True, embedding and reconstruction
            loss are also returned along with model predictions
        split : str
            train/val/test split
        Returns
        -------
        tuple of Pytorch Tensors
            if loss == True: returns (y_pred, loss_rec, loss_vae)
            else: returns y_pred
            y_pred: prediction from spatiotemporal model
            loss_rec: reconstruction loss for autoencoder
            loss_vae: embedding loss (latent layer loss)
        """
        emb_dict = {
            feat: self.spatial_embedding_dict[feat].encode(x[feat])
            for feat in self.ERA5_features_list
        }
        x_emb = [emb['emb'] for emb in emb_dict.values()] + [x['indices']]
        x_emb = torch.cat(x_emb, axis=-1)
        y_pred = self.temporal_model(x_emb)
        y_pred = self.spatial_embedding_dict['t2m'].decode(y_pred)

        if loss == False:
            return y_pred

        loss_var = [emb['var_loss'] for emb in emb_dict.values()]
        loss_rec = [emb['rec_loss'] for emb in emb_dict.values()]

        loss_var = torch.stack(loss_var).mean()
        loss_rec = torch.stack(loss_rec).mean()

        for feat in self.ERA5_features_list:
            feat_loss = {
                'rec': emb_dict[feat]['rec_loss'].detach().cpu().item(), 
                'var': emb_dict[feat]['var_loss'].detach().cpu().item()
            }
            self.spatial_embedding_dict[feat].update_loss(feat_loss, split)

        return y_pred, loss_rec, loss_var

    def _display_epoch_loss(self, train_loss, val_loss, test_loss):
        if self.model_params.use_VAE:
            print (
                "Epoch: {:03d}, loss (TCN, AE_rec, AE_var): train: {:1.3e}, {:1.3e}, {:1.3e} | "
                .format(self.epoch, train_loss['tcn'], train_loss['rec'], train_loss['var']) +
                "val: {:1.3e}, {:1.3e}, {:1.3e} | "
                .format(val_loss['tcn'], val_loss['rec'], val_loss['var']) +
                "test: {:1.3e}, {:1.3e}, {:1.3e}"
                .format(test_loss['tcn'], test_loss['rec'], test_loss['var'])
            )
        else:
            print (
                "Epoch: {:03d}, loss (TCN, AE_rec): train: {:1.3e}, {:1.3e} | "
                .format(self.epoch, train_loss['tcn'], train_loss['rec']) +
                "val: {:1.3e}, {:1.3e} | "
                .format(val_loss['tcn'], val_loss['rec']) +
                "test: {:1.3e}, {:1.3e}"
                .format(test_loss['tcn'], test_loss['rec'])
            )

    def _save_model(self, epoch=None, best=False):
        """saves model checkpoints
        Parameters
        ----------
        epoch : int
            #epoch of training loop
        best : str
            tag for saing the best model
        """
        if best == True:
            model_path = os.path.join(self.log_path, "net_best_so_far.pth")
            torch.save(self.temporal_model.state_dict(), model_path)
            for feat in self.ERA5_features_list:
                model_path = os.path.join(self.log_path, 'spatial', '{}_emb_best.pth'.format(feat))
                self.spatial_embedding_dict[feat].save(model_path)
        else:
            if epoch is None:
               raise Exception("Please provide #epoch for saving current model state")
            model_path = os.path.join(self.log_path, "net_{}.pth".format(epoch))
            torch.save(self.temporal_model.state_dict(), model_path)
            for feat in self.ERA5_features_list:
                model_path = os.path.join(self.log_path, 'spatial', '{}_emb_{}.pth'.format(feat, epoch))
                self.spatial_embedding_dict[feat].save(model_path)

    def load(self):
        """Loads the best/latest checkpoint from log_path"""
        spatial_ckeckpoint_dict = {}
        if self.log_path is None:
            raise Exception("Invalid ckpt_timestamp: no checkpoints exist")

        if self.trainer_params.ckpt == 'best':
            checkpoint_path = os.path.join(self.log_path, 'net_best_so_far.pth')
            for feat in self.ERA5_features_list:
                spatial_ckeckpoint_dict[feat] = os.path.join(self.log_path, 'spatial', '{}_emb_best.pth'.format(feat))

        elif self.trainer_params.ckpt == 'latest':
            filename_list = os.listdir(self.log_path)
            filepath_list = [os.path.join(self.log_path, fname) for fname in filename_list]
            filepath_list = [fpath for fpath in filepath_list if os.path.isfile(fpath)]
            checkpoint_path = sorted(filepath_list, key=os.path.getmtime)[-1]

            filename_list = os.listdir(os.path.join(self.log_path, 'spatial'))
            for feat in self.ERA5_features_list:
                filepath_list = [
                    os.path.join(self.log_path, 'spatial', fname) 
                    for fname in filename_list
                    if fname.startswith(feat)
                ]
                spatial_ckeckpoint_dict[feat] = sorted(filepath_list, key=os.path.getmtime)[-1]

        elif self.trainer_params.ckpt_epoch is not None:
            epoch = self.trainer_params.ckpt_epoch
            checkpoint_path = os.path.join(self.log_path, 'net_{}.pth'.format(epoch))
            for feat in self.ERA5_features_list:
                spatial_ckeckpoint_dict[feat] = os.path.join(self.log_path, 'spatial', '{}_emb_{}.pth'.format(feat, epoch))
        else:
            raise Exception("please provide trainer_params.ckpt_epoch")

        print('-'*90)
        print('Loading model from: <{}>'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.temporal_model.load_state_dict(checkpoint)
        for feat in self.ERA5_features_list:
            self.spatial_embedding_dict[feat].load(spatial_ckeckpoint_dict[feat])
        print('-'*90)

    def _append_embedding_loss(self):
        for feat in self.ERA5_features_list:
            self.spatial_embedding_dict[feat].append_loss()

    def _plot_embeddings(self, split):
        dloader = self.ERA5_data.dataloader_dict[split]
        sample = iter(dloader).next()
        x_dict, inp_time_seq = sample[0], sample[2][0]
        for feat in self.ERA5_features_list:
            self.spatial_embedding_dict[feat].plot (
                x_dict[feat],
                inp_time_seq.numpy()
            )

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