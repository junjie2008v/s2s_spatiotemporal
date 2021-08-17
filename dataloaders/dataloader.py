import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
from misc.ops import tic, toc
from dataloaders.utils import *


class ERA5_dataset(Dataset):
    """PyTorch Dataset class for data obtained from ERA5
    Generates spatio-temporal samples on the fly diectly
    from netcdf ERA5 climate data files
    Attributes
    ----------
    params : argparse parameters
    split : str
        train/val/test split
    years_list : list of int
        list of years in train/val/test split
    ds_dict : dict of {str: xarray dataset}
        keys: feature name
        vals: dataset
            Index: latitude, longitude, time
            variable: feature
    df_indices : pandas dataframe
        columns: climate indices & time-based features
        index: date
    scale_dict : dict of {str: dict of {str: xarray dataset}}
        keys: feature
        vals: dict
            keys: 'mean', 'std'
            vals: xarray dataset
    transform : list of PyTorch transforms
    Methods
    -------
    __len__
        returns length of dataset
        usage: len(datset<class ERA5_dataset>)
    __item__
        dataset iterator to be used by PyTorch DataLoader
    Note
    ----
    All other methods are assumed private and are meant for internal processing only
    """
    def __init__(self, params, split='train', transform=None, scale_dict=None):
        self.params = params
        self.split  = split
        self.years_list = self._get_split_years()
        self.ds_dict = self._get_gridded_datasets()
        self.df_indices = self._get_indices_df()
        self._apply_rolling_mean(window=self.params.rolling_mean_window)
        self.scale_dict = self._get_scale() if scale_dict is None else scale_dict
        self._normalize()
        self.transform = transform

    def _get_split_years(self):
        """Returns list of years in split
        Returns
        -------
            list of int
                list of years in split (train/val/test)
        """
        if self.split == 'val':
            return np.arange(self.params.val_years[0], self.params.val_years[1]+1)
        elif self.split == 'test':
            return np.arange(self.params.test_years[0], self.params.test_years[1]+1)
        elif self.split == 'train':
            return np.arange(self.params.train_years[0], self.params.train_years[1]+1)

    def _get_gridded_datasets(self):
        """Returns dictionary of datasets after splitting based on split
        Returns
        -------
            dict: {str: xarray dataset}
                keys: feature name
                vals: dataset (NaNs replaced by 0.)
                splitted dataset based on split train/val/test
        """
        t1 = tic(message = 'fetching ERA5 netcdf gridded data for {}-split ...'.format(self.split))
        INP_DIR = os.path.join(self.params.data_dir, 'cdf_files')
        resolution = self.params.resolution

        ds_dict_single_level = {feat: get_ERA5_single_level_dataset(INP_DIR, feat, self.years_list, resolution)
                                for feat in self.params.ERA5_features_single_level}

        ds_dict_pressure_levels = {feat: get_ERA5_pressure_levels_dataset(INP_DIR, feat, self.years_list, resolution)
                                for feat in self.params.ERA5_features_pressure_levels}

        ds_dict = {**ds_dict_single_level, **ds_dict_pressure_levels}

        print('ERA5 features list: {}'.format(list(ds_dict.keys())))
        toc(t1, 'successfully fetched ERA5 netcdf data for {}-split'.format(self.split))
        return ds_dict

    def _get_indices_df(self):
        """Returns pandas dataframe containing indices and other scalars
        Returns
        -------
        df : pandas dataframe
            columns: climate indices & time-based features
            index: date
        """
        t1 = tic(message = 'fetching indices {}-split ...'.format(self.split))
        INP_DIR = os.path.join(self.params.data_dir, 'cdf_files')
        df = get_indices_dataframe(INP_DIR, self.years_list)
        toc(t1, 'successfully fetched indices data for {}-split'.format(self.split))
        return df

    def _apply_rolling_mean(self, window):
        """Applies rolling mean to the dataset
        Applied moving average of length: window
        to all the features in self.ds_dict
        Parameters
        ----------
        window : int
            number of time steps for moving average
        """
        t1 = tic(message = 'applying rolling mean of window size = {} to raw netcdf data for {}-split ...'.format(window, self.split))
        for key in self.ds_dict.keys():
            self.ds_dict[key] = self.ds_dict[key].rolling(time=window, center=False).mean().dropna(dim='time')
        self.df_indices = self.df_indices.rolling(window, center=False).mean().dropna()
        toc(t1, message = 'successfully applied rolling mean to raw netcdf data for {}-split'.format(self.split))

    def _get_scale(self):
        """Returns standard normalization scale for all features
        Returns
        -------
        scale_dict : dict of {str: dict of {str: xarray dataset}}
            keys: feature
            vals: dict
                keys: 'mean', 'std'
                vals: xarray dataset
        """
        scale_dict = {
            key: calculate_scale (
                self.ds_dict[key],
                coordinate='time',
                strategy=self.params.normalizing_strategy
            )
            for key in self.ds_dict.keys()
        }
        return scale_dict

    def _normalize(self):
        """normalizes all ERA5 xarray datasets in ds_dict"""
        for key in self.ds_dict.keys():
            self.ds_dict[key] = normalize (
                self.ds_dict[key],
                self.scale_dict[key],
                coordinate='time',
                strategy=self.params.normalizing_strategy
            )

    def _get_encoded_time_seq(self, idtI, idtO, delta_t):
        """returns encoded inp & out time seq
        Parameters
        ----------
            idtI : tuple of (int, int)
                initial and final input time series indexes
            idtO : tuple of (int, int)
                initial and final output time series indexes
            delta_t : int
                delta time series
        Returns
        -------
            tuple of (ndarray, ndarray)
                returns endoded time seq (inp_seq, out_seq) 
                in YYYYMMDD integer format
        """
        inp_seq = self.ds_dict['t2m'].time[idtI[0] : idtI[1] : delta_t]
        YYYY, MM, DD = inp_seq.dt.year.values, inp_seq.dt.month.values, inp_seq.dt.day.values
        inp_seq = YYYY*10000 + MM*100 + DD
        out_seq = self.ds_dict['t2m'].time[idtO[0] : idtO[1] : delta_t]
        YYYY, MM, DD = out_seq.dt.year.values, out_seq.dt.month.values, out_seq.dt.day.values
        out_seq = YYYY*10000 + MM*100 + DD
        return inp_seq, out_seq

    def _get_sample_time_series_ids(self, idx):
        """returns input/output sample time series indexes
        Returns
        -------
            tuple of (tuple of (int, int), tuple of (int, int), int)
                (idtI, idtO, delta_t)
                idtI : input indexes  (inital time index, final time index)
                idtO : output indexes (inital time index, final time index)
                delta_t : delta time series
        """
        delta_t = self.params.delta_time_series
        idtI_i = idx * self.params.stride_time_series
        idtI_f = idtI_i + self.params.len_inp_time_series * delta_t
        idtO_i = idtI_f
        idtO_f = idtO_i + self.params.len_out_time_series * delta_t
        return (idtI_i, idtI_f), (idtO_i, idtO_f), delta_t

    def __len__(self):
        """Mandatory len function for PyTorch Dataset class
        Returns
        -------
            int
                length of PyTorch dataset equal to the number of samples
        """ 
        return ( self.ds_dict['t2m'].time.shape[0] - (
                                self.params.len_inp_time_series
                              + self.params.len_out_time_series
                            ) * self.params.delta_time_series
                ) // self.params.stride_time_series + 1

    def __getitem__(self, idx):
        """Mandatory getitem function for PyTorch Dataset class
        Returns
        -------
            list of PyTorch tensors
                list [input_sample, output_sample, time_seq] of single PyTorch Dataset sample
                    time_seq: tuple of (encoded_inp_time_seq, encoded_out_time_seq)
                    date encoding: YYYYMMDD
        """
        idtI, idtO, Dt = self._get_sample_time_series_ids(idx)

        input_dict = {
            key:
                torch.tensor (
                    self.ds_dict[key][idtI[0] : idtI[1] : Dt, :, :].values.astype(self.params.precision)
            )
            for key in self.ds_dict.keys()
        }

        input_dict['indices'] = torch.tensor (
                self.df_indices[idtI[0] : idtI[1] : Dt].values.astype(self.params.precision)
            )

        output = torch.tensor (
                self.ds_dict['t2m'][idtO[0] : idtO[1] : Dt, :, :].values.astype(self.params.precision)
            )

        time_seq = self._get_encoded_time_seq(idtI, idtO, Dt)

        return input_dict, output, time_seq


class ERA5_dataloader(object):
    """Dataloader class consisting of dataloaders for train/val/test splits 
    Attributes
    ----------
    scale_dict : dict of {str: dict of {str: xarray dataset}}
        keys: feature
        vals: dict
            keys: 'mean', 'std'
            vals: xarray dataset
    ERA5_features_list: list of str
        list of features extracted from ERA5 climate netcdf files
    indices_list: list of str
        list of climate indices & time-based features
    dataloader_dict : dict of {str: Pytorch Dataloader}
        dictionary of dataloaders 
        operating_mode=train: dataloaders for train, val & test splits
        operating_mode=load:  dataloaders for splits in save_splits
    Methods
    -------
    No public methods
    """
    def __init__(self, params):
        self.params = params
        self.scale_dict = None
        self.dataloader_dict = {}

        if self.params.operating_mode == 'train':
            for split in ['train', 'val', 'test']:
                self.dataloader_dict[split] = self._dataloader(split=split)
            self.ERA5_features_list = list(self.dataloader_dict['train'].dataset.ds_dict.keys())
            self.indices_list = self.dataloader_dict['train'].dataset.df_indices.columns
        else:
            with open(os.path.join(self.params.root_path, 'scale_dict.pkl'), 'rb') as handle:
                self.scale_dict = pickle.load(handle)
            self.dataloader_dict = {split: self._dataloader(split=split) for split in self.params.save_splits}
            self.ERA5_features_list = list(self.dataloader_dict[self.params.save_splits[0]].dataset.ds_dict.keys())
            self.indices_list = self.dataloader_dict[self.params.save_splits[0]].dataset.df_indices.columns

    def _dataloader(self, split):
        """returns dataloader for the given split
        Parameters
        ----------
            split : str
                train/val/test split
        Returns
        -------
            dataloader : Pytorch Dataloader for <ERA5_Dataset> dataset class
        """
        trans = []
        kwargs = {'num_workers': self.params.num_workers, 'pin_memory': self.params.use_cuda}
        dset = ERA5_dataset(self.params, split, transform=trans, scale_dict=self.scale_dict)

        shuffle = False
        if self.params.operating_mode == 'train':
            shuffle = True if split=='train' else False
        dloader = DataLoader(dataset=dset, batch_size=self.params.batch_size, shuffle=shuffle, **kwargs)

        if split=='train':
            self.scale_dict = dset.scale_dict
            with open(os.path.join(self.params.root_path, 'scale_dict.pkl'), 'wb') as handle:
                pickle.dump(self.scale_dict, handle)

        return dloader