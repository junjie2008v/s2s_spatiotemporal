import torch
import torch.nn as nn
import numpy as np
import random
import os
from datetime import datetime
import time


def loss_function(y_true, y_pred, loss_type):
    if loss_type == 'smooth_mae':
        loss = smooth_mae_loss(y_true, y_pred)
    elif loss_type == 'mae':
        loss = mae_loss(y_true, y_pred)
    elif loss_type == 'mse':
        loss = mse_loss(y_true, y_pred)
    elif loss_type == 'mnacc':
        loss = mnacc_loss(y_true, y_pred)
    return loss

def mse_loss(y_true, y_pred):
    loss = nn.MSELoss()
    out = loss(y_pred, y_true)
    return out

def mae_loss(y_true, y_pred):
    loss = nn.L1Loss()
    out = loss(y_pred, y_true)
    return out

def smooth_mae_loss(y_true, y_pred):
    loss = nn.SmoothL1Loss()
    out = loss(y_pred, y_true)
    return out

def mnacc_loss(y_true, y_pred):
    f, a = y_pred, y_true
    favg = torch.mean(f, dim=2, keepdims=True)
    aavg = torch.mean(a, dim=2, keepdims=True)

    ff, aa = f - favg, a - aavg
    fstd = torch.sqrt(torch.mean(torch.square(ff), dim=2))
    astd = torch.sqrt(torch.mean(torch.square(aa), dim=2))

    cov = torch.mean(ff*aa, dim=2)
    corr = cov/(fstd*astd)

    out = -1 * torch.mean(corr, dim=1)
    return torch.mean(out)

def seed(trainer_params):
    torch.manual_seed(trainer_params.seed)
    torch.cuda.manual_seed(trainer_params.seed)
    np.random.seed(trainer_params.seed)
    random.seed(trainer_params.seed)

def get_log_paths(run, model_name, feature_type, ckpt_timestamp=None):
    init_time = ckpt_timestamp
    if init_time is None:
        time = datetime.now()
        init_time = time.strftime('%Y_%m_%d_%H_%M')
    root_path = os.path.join('runs', run, model_name, feature_type, init_time)
    log_path = os.path.join(root_path, 'checkpoints')
    if ckpt_timestamp is None:
        os.makedirs(root_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
    else:
        if os.path.isdir(root_path) == False:
            raise Exception("Invalid ckpt_timestamp: {}".format(ckpt_timestamp)) 
    wandb_name = model_name  + '_' + feature_type + '_' + init_time
    return root_path, log_path, wandb_name

def tic(message=''):
    """Returns current time for timing a task
    Parameters
    ----------
    message : str
        message to be printed
    Returns
    -------
        time.time()
            time at which task is initiated
    """
    print(message)
    return time.time()

def toc(t1, message=''):
    """Returns time taken for execution of a task
    Parameters
    ----------
    t1 : time.time()
        time at which task was initiated
    message : str
        message to be printed
    Returns
    -------
        time.time()
            time taken for execution of task initiated at t1
    """
    print(message + ' | time taken =  {:1.4f}s'.format(time.time()-t1))
    print('-'*90)