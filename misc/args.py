import argparse

def dataloader_params():
    parser = argparse.ArgumentParser(description="Dataloader arguments")

    parser.add_argument('--operating_mode', type=str, default='train', choices=['train','load'],
                        help='Train model or load pre-trained model')

    parser.add_argument('--save_splits', nargs="+", default=['test'],# 'val', 'train'],
                        help="Splits for which predictions to be saved")

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/opt/nautilus_dataset/',
                        help='Data directory containing all the features')

    parser.add_argument('--ERA5_features_single_level', nargs="+", \
                        default=['t2m', 'sp', 'd2m', 'sst_pac', 'sst_atl'],
                        help="ERA5 single level features to be used for training \
                        https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-level")

    parser.add_argument('--ERA5_features_pressure_levels', nargs="+", \
                        default=['z_50', 'z_300'],
                        help="ERA5 single level features to be used for training \
                        https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels")

    parser.add_argument('--resolution', type=float, default='1', choices=[0.25,0.5,1,2],
                        help="latitude x longitude resolution in degrees")

    parser.add_argument('--normalizing_strategy', type=str, default='time',
                        choices=['time', 'month', 'season', 'dayofyear'],
                        help="type of normalization to be performed on ERA5 datasets")

    parser.add_argument('--precision', type=str, default='float32',
                        help="operating precision, defaults to float32")
    parser.add_argument('--rolling_mean_window', type=int, default=7,
                        help="window size for applying moving average on raw data")
    parser.add_argument('--len_inp_time_series', type=int, default=1,
                        help="length of inp time-series to be used for temporal models")
    parser.add_argument('--len_out_time_series', type=int, default=1,
                        help="length of out time-series to be used for temporal models")
    parser.add_argument('--stride_time_series', type=int, default=1,
                        help="#days striding of time-series to be used for temporal models")
    parser.add_argument('--delta_time_series', type=int, default=1,
                        help="delta_T of time-series to be used for temporal models")

    # Generic arguments
    parser.add_argument('--batch_size', type=int, default=1,
                        help="batch size for training")
    parser.add_argument('--use_cuda',    dest='use_cuda', action='store_true')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false')
    parser.set_defaults(use_cuda=True)
    parser.add_argument('--num_workers', type=int, default=8,
                        help="number of parallel workers")

    # Dataset partition
    parser.add_argument('--val_years', nargs="+", default=[2016, 2018],
                        help="range of years of validation data")
    parser.add_argument('--test_years', nargs="+", default=[2019, 2021],
                        help="range of years of testing data")
    parser.add_argument('--train_years', nargs="+", default=[1979, 2015],
                        help="range of years of testing data")

    opt,_ = parser.parse_known_args()

    return opt


def model_params():
    parser = argparse.ArgumentParser(description='Arguments for  models')

    # tcn model args
    parser.add_argument('--dropout', type=float, default=0.2,
                    help="dropout rate")
    parser.add_argument('--network_depth', type=int, default=4,
                        help="depth of TCN residual blocks")
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help="number of nodes in the hidden layer")
    parser.add_argument('--filter_width', type=int, default=2,
                        help="width of 1D conv filters")

    # embedding model args
    parser.add_argument('--spatial_embedding_model', type=str, default='NN', choices=['PCA', 'NN'],
                        help="model for generating spatial embedding for ERA5 features")
    parser.add_argument('--spatial_embedding_dim', type=int, default=10,
                        help="dimension of spatial embeddings for ERA5 features")
    parser.add_argument('--use_VAE',    dest='use_VAE', action='store_true')
    parser.add_argument('--no-use_VAE', dest='use_VAE', action='store_false')
    parser.set_defaults(use_VAE=False)

    opt,_ = parser.parse_known_args()

    return opt

def trainer_params():
    parser = argparse.ArgumentParser(description="Arguments for model trainer")

    parser.add_argument('--run', type=str, default='NN',
                        help="name for the current run", choices=['NN', 'autoreg'])
    parser.add_argument('--operating_mode', type=str, default='train',
                        help="weather to train or load existing ", choices=['train', 'load'])
    parser.add_argument('--ckpt_timestamp', type=str, default=None, help="timestamp of model to be loaded")
    parser.add_argument('--ckpt', type=str, default='best',
                        help="which checkpoint to load", choices=['best', 'latest', 'epoch'])
    parser.add_argument('--ckpt_epoch', type=int, default=None, help="ckpt epoch to load")
    parser.add_argument('--seed', type=int, default=1234,
                        help="seeding things")

    # Learning rate arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="learning rate for training")
    parser.add_argument('--lr_schedule_mode', type=str, default='min',
                        help="mode for lr scheduler. min | max")
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5,
                        help="factor by which to reduce the learning rate")
    parser.add_argument('--lr_cooldown', type=int, default=5,
                        help="number of epochs to wait before resuming normal operations after lr reduction")
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help="minimun lr for learning rate scheduler")

    # Training loop arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help="number of training epochs")

    parser.add_argument('--schedule_lr', dest='schedule_lr', action='store_true',
                        help="whether to schedule decay of learning rate")
    parser.add_argument('--no-schedule_lr', dest='schedule_lr', action='store_true')
    parser.set_defaults(schedule_lr=True)

    parser.add_argument('--loss_type', type=str, default='mse',
                        help="type of loss to be used: mae | mse | mnacc | smooth_mae | hybrid")
    parser.add_argument('--loss_delay', type=int, default=100,
                        help="starting index of output sequence to use for loss calculation")

    parser.add_argument('--use_cuda',    dest='use_cuda', action='store_true')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false')
    parser.set_defaults(use_cuda=True)

    parser.add_argument('--num_workers', type=int, default=8,
                        help="number of parallel workers")

    # Logging specific arguments
    parser.add_argument('--save_interval', type=int, default=10,
                        help="running test loop after every test_interval epochs")
    parser.add_argument('--wandb_entity', type=str, default='nishant.parashar',
                        help="individual experimental logging")

    opt,_ = parser.parse_known_args()

    return opt