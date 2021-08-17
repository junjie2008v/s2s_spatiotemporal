import xarray as xr
import os
import pandas as pd

NETCDF_DICT = {'t2m': 'tmp2m.nc', 'sp': 'pressure_RH.nc', 'd2m': 'pressure_RH.nc',
               'sst_pac': 'SST_pacific_ocean.nc', 'sst_atl': 'SST_atlantic_ocean.nc',
               'z': 'geopotential.nc'}

def get_ERA5_single_level_dataset(INP_DIR, feat, years_list, resolution):
    """returns ERA5 single level feature as xarray datasets
    Parameters
    ----------
    INP_DIR : str
        Root data directory
    feat : str
        feature name with suffixes for regions
        may be dictinct from era5 feature names
    years_list : list of int
        list of years for which data is to be extracted
    resolution: int
        spatial resolution of latitude x longitude grid to be extracted
    Returns
    -------
    ds : xarray dataset
        coordinates: latitude, longitude, time
    """
    era5_feat = feat.split("_")[0]

    DATA_DIR = os.path.join(INP_DIR, 'latest')

    ds = xr.open_dataset(os.path.join(DATA_DIR, NETCDF_DICT[feat]))
    skip = int(resolution * 4) # ERA5 data originally at 0.25 deg x 0.25 deg resolution

    #idt = ds.time.dt.year.isin(years_list)
    #ds = ds[era5_feat][idt,::skip,::skip].fillna(0.0)
    ds = ds[era5_feat].sel(time = ds.time.dt.year.isin(years_list),
                        latitude  = ds.latitude[::skip],
                        longitude = ds.longitude[::skip]) #.fillna(0.0)
    return ds.load()


def get_ERA5_pressure_levels_dataset(INP_DIR, feat, years_list, resolution):
    """returns ERA5 pressure levels feature as xarray datasets
    Parameters
    ----------
    INP_DIR: str
        Root data directory
    feat: str
        feature name with suffixes for vertical levels
        may be dictinct from era5 feature names
    years_list: list of int
        list of years for which data is to be extracted
    resolution: int
        spatial resolution of latitude x longitude grid to be extracted
    Returns
    -------
    ds : xarray dataset
        coordinates: latitude, longitude, time
    """
    era5_feat = feat.split("_")[0]
    level = int(feat.split("_")[1])

    DATA_DIR = os.path.join(INP_DIR, 'latest')
    ds = xr.open_dataset(os.path.join(DATA_DIR, NETCDF_DICT[era5_feat]))

    skip = int(resolution * 4) # ERA5 data originally at 0.25 deg x 0.25 deg resolution
    skip = 2*skip # taking a 2x coarser grid for atm features

    ds = ds[era5_feat].sel(time = ds.time.dt.year.isin(years_list),
                        level = level,
                        latitude  = ds.latitude[::skip],
                        longitude = ds.longitude[::skip])

    id_lat = (ds.latitude  <= 80)
    id_lon = (ds.longitude <= -25)

    ds = ds.sel(latitude  = ds.latitude[id_lat], longitude = ds.longitude[id_lon]) #.fillna(0.0)

    return ds.load()

def get_indices_dataframe(INP_DIR, years_list):
    """returns pandas dataframe containing indices and other scalars
    Parameters
    ----------
    INP_DIR: str
        Root data directory
    feat: str
        feature name with suffixes for vertical levels
        may be dictinct from era5 feature names
    years_list: list of int
        list of years for which data is to be extracted
    Returns
    -------
    df : pandas dataframe
        columns: climate indices & time-based features
        index: date
    """
    df = pd.read_hdf(os.path.join(INP_DIR, 'indices', 'indices.h5'))
    df = df[df.index.year.isin(years_list)]
    df['month'] = df.index.month / 12.0
    df['dayofyear'] = df.index.dayofyear / 366.0
    return df

def calculate_scale(ds, coordinate='time', strategy='time'):
    """returns mean and std-dev of ERA5 dataset
    Parameters
    ----------
    ds: xarray dataset
        coordinates: latitude, longitude, time / forecast_date
    coordinate: str
        coordinate for calculating scale (pandas date_time based coordinates)
    strategy: str
        time, season, dayofyear, month
    Returns
    -------
    dict {str: xarray dataset}
        keys : 'mean', 'val'
        vals : xarray dataset
    """
    if strategy == 'time':
        ds_mean = ds.mean(coordinate)
        ds_std = ds.std(coordinate)
    else:
        group = get_group(coordinate, strategy)
        ds_mean = ds.groupby(group).mean()
        # calculating std-dev using all time (strategy='time')
        ds_std  = ds.std(coordinate)
        #ds_std = ds.groupby(group).std()
    return {'mean': ds_mean, 'std': ds_std}

def normalize(ds, ds_scale_dict, coordinate='time', strategy='time'):
    """returns normalized ERA5 dataset
    Parameters
    ----------
    ds: xarray dataset
        un-normalized dataset
        coordinates: latitude, longitude, time / forecast_date
    ds_scale_dict:  dict {str: xarray dataset}
        keys : 'mean', 'val'
        vals : xarray dataset
    coordinate: str
        coordinate for calculating scale (pandas date_time based coordinates)
    strategy: str
        time, season, dayofyear, month
    Returns
    -------
    ds: xarray dataset
        normalized xarray dataset
    """
    epsilon = 1e-8
    if strategy == 'time':
        ds = ds - ds_scale_dict['mean']
        ds = ds / (ds_scale_dict['std'] + epsilon)
    else:
        group = get_group(coordinate, strategy)
        ds = ds.groupby(group) - ds_scale_dict['mean']
        # std-dev normalization using all time (strategy='time')
        ds = ds / (ds_scale_dict['std'] + epsilon)
        #ds = ds.groupby(group) / (ds_scale_dict['std'] + epsilon)
    return ds

def unnormalize(ds, ds_scale_dict, coordinate='time', strategy='time'):
    """returns un-normalized ERA5 dataset
    Parameters
    ----------
    ds: xarray dataset
        normalized dataset
        coordinates: latitude, longitude, time / forecast_date
    ds_scale_dict:  dict {str: xarray dataset}
        keys : 'mean', 'val'
        vals : xarray dataset
    coordinate: str
        coordinate for calculating scale (pandas date_time based coordinates)
    strategy: str
        time, season, dayofyear, month
    Returns
    -------
    ds: xarray dataset
        un-normalized xarray dataset
    """
    epsilon = 1e-8
    if strategy == 'time':
        ds = ds * (ds_scale_dict['std'] + epsilon)
        ds = ds + ds_scale_dict['mean']
    else:
        group = get_group(coordinate, strategy)
        ds = ds * (ds_scale_dict['std'] + epsilon)
        #ds = ds.groupby(group) * (ds_scale_dict['std'] + epsilon)
        ds = ds.groupby(group) + ds_scale_dict['mean']
    return ds

def get_group(coordinate, strategy):
    """returns grouping for xr_dataset.groupby
    Parameters
    ----------
    coordinate: str
        pd datetime based time index
    strategy: str
        normalizing strategy (season, month, dayofyear)
    """
    return coordinate+'.'+strategy