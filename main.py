import os
from datetime import datetime

from misc.args import dataloader_params, trainer_params, model_params
from misc.ops import seed

if model_params().spatial_embedding_model == 'PCA':
    from trainers.trainer_PCA_embedding import Trainer
elif model_params().spatial_embedding_model == 'NN':
    from trainers.trainer_NN_embedding import Trainer

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    dataloader_params = dataloader_params()
    trainer_params = trainer_params()
    model_params = model_params()
    params = (dataloader_params, trainer_params, model_params)
    # Fix Random Seeding
    seed(trainer_params)
    
    trainer = Trainer(params)

    if trainer_params.operating_mode == 'train':
        trainer.fit()
    trainer.load()
    
    for split in dataloader_params.save_splits:
        trainer.save_predictions(split)