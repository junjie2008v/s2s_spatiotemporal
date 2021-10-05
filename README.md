# s2s_spatiotemporal

Spatiotemporal modeling for Subseasonal Forecasting

## Conda Installation
```
conda create -n s2s python=3.6
conda activate s2s
conda install numpy
conda install matplotlib basemap
conda install seaborn
conda install pandas
conda install pytables
conda install xarray
conda install scipy
conda install scikit-learn
conda install -c conda-forge xgboost
conda install -c conda-forge wandb
conda install jupyter notebook
conda update --all
```

##### LINUX

`conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=10.1 -c pytorch`

##### MAC

`conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch`


#### Fix basemap issue:
In the following files:
```
/opt/anaconda3/envs/s2s/lib/python3.6/site-packages/mpl_toolkits/basemap/__init__.py
/opt/anaconda3/envs/s2s/lib/python3.6/site-packages/mpl_toolkits/basemap/proj.py
```

comment out `from matplotlib.cbook import dedent` and replace it by the following code-block

```
try:
    from inspect import cleandoc as dedent
except ImportError:
    # Deprecated as of version 3.1. Not quite the same
    # as textwrap.dedent.
    from matplotlib.cbook import dedent
```
## Instructions
### Running the code
python main.py --len_inp_time_series 12 --len_out_time_series 4 --stride_time_series 1 --delta_time_series 7 --batch_size 32 --spatial_embedding_dim 10

NOTE: refer to misc/args.py for list of acceptable arguments
