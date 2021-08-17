from torch.utils import data
import xarray as xr
import numpy as np
import pandas as pd
import os
from dataloaders.utils import *

def save_preds(y_true, y_pred, x_dates, y_dates, ds_scale_dict, params, split):

    PREDS_DIR = os.path.join(params.root_path, 'predictions', split)
    os.makedirs(PREDS_DIR, exist_ok=True)

    PREDS_FILE = os.path.join(PREDS_DIR, 'forecast.nc')
    if os.path.isfile(PREDS_FILE):
        os.remove(PREDS_FILE)

    CDF_DIR = os.path.join(params.data_dir, 'cdf_files')
    resolution = params.resolution
    window = params.rolling_mean_window
    ds_t2m = get_ERA5_single_level_dataset(CDF_DIR, 't2m', range(1981, 2011), resolution)
    ds_t2m = ds_t2m.rolling(time=window, center=False).mean()

    issue_dates, forecast_dates = get_datetime_index(x_dates, y_dates)

    ds_anom = get_anomaly_dataset (
        ds_t2m, ds_scale_dict,
        y_true, y_pred,
        issue_dates, forecast_dates,
        params.normalizing_strategy
    )

    pred_skill = np.zeros((y_dates.shape[0], y_dates.shape[1]))
    for j in range(y_dates.shape[1]):
        for i in range(y_dates.shape[0]):
            pred_skill[i,j] = skill(
                y_true = ds_anom['t2ma_true'][:,:,i,j].values.reshape(-1),
                y_pred = ds_anom['t2ma_pred'][:,:,i,j].values.reshape(-1)
            )
        print('mean skill {}-split for forecast #{} = {}'.format(split, j+1, pred_skill[:,j].mean()))

    ds_anom.to_netcdf(PREDS_FILE)
    ds_anom.close()
    ds_t2m.close()


def get_datetime_index(x_dates, y_dates):
    issue_dates = ['{}-{:02d}-{:02d}'.format(d//10000, d%10000//100, d%100) for d in x_dates[:,-1]]
    issue_dates = pd.to_datetime(issue_dates)
    forecast_dates_list = []
    for j in range(y_dates.shape[0]):
        forecast_dates = ['{}-{:02d}-{:02d}'.format(d//10000, d%10000//100, d%100) for d in y_dates[j,:]]
        forecast_dates_list.append(pd.to_datetime(forecast_dates))
    return issue_dates, forecast_dates_list

def skill(y_true, y_pred):
    return np.dot(y_true, y_pred)/(np.linalg.norm(y_true, 2)*np.linalg.norm(y_pred, 2))


def get_anomaly_dataset(ds_t2m, ds_scale_dict, t2m_true, t2m_pred, issue_dates, forecast_dates, norm_strategy):

    ds = xr.Dataset({
        't2m': (['time','forecast','latitude','longitude'], t2m_true),
        },
        coords = {
            'time': (['time'], issue_dates),
            'forecast_date': (['time', 'forecast'], forecast_dates),
            'latitude': ('latitude', ds_t2m.latitude),
            'longitude': ('longitude', ds_t2m.longitude)
        }
    )
    ds2 = xr.Dataset({
        't2m': (['time','forecast','latitude','longitude'], t2m_pred),
        },
        coords = {
            'time': (['time'], issue_dates),
            'forecast_date': (['time', 'forecast'], forecast_dates),
            'latitude': ('latitude', ds_t2m.latitude),
            'longitude': ('longitude', ds_t2m.longitude)
        }
    )
    ds = unnormalize(ds, ds_scale_dict, 'forecast_date', norm_strategy)
    ds = ds.groupby(ds.forecast_date.dt.dayofyear) - ds_t2m.groupby(ds_t2m.time.dt.dayofyear).mean()
    ds = ds.rename({'t2m': 't2ma_true'})

    ds2 = unnormalize(ds2, ds_scale_dict, 'forecast_date', norm_strategy)
    ds2 = ds2.groupby(ds2.forecast_date.dt.dayofyear) - ds_t2m.groupby(ds_t2m.time.dt.dayofyear).mean()
    ds = ds.assign(t2ma_pred = ds2['t2m'])

    return ds