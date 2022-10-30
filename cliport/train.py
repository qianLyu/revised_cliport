"""Main training script."""

import os
from pathlib import Path

import torch
from cliport import agents
from cliport.dataset import RavensDataset, RavensMultiTaskDataset, WeightTrainerDataset, SSPretrainDataset

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from easydict import EasyDict

import numpy as np
import torch
import clip


@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):
    # Logger
    wandb_logger = WandbLogger(name=cfg['tag']) if cfg['train']['log'] else None

    # Checkpoint saver
    hydra_dir = Path(os.getcwd())
    checkpoint_path = os.path.join(cfg['train']['train_dir'], 'checkpoints')
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        filepath=os.path.join(checkpoint_path, 'best'),
        save_top_k=1,
        save_last=True,
    )

    # Trainer
    max_epochs = cfg['train']['n_steps'] // cfg['train']['n_demos']
    trainer = Trainer(
        gpus=cfg['train']['gpu'],
        fast_dev_run=cfg['debug'],
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=max_epochs,
        automatic_optimization=False,
        check_val_every_n_epoch=1, #max_epochs // 200,
        resume_from_checkpoint=last_checkpoint,
    )

    # Resume epoch and global_steps
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint)
        trainer.current_epoch = last_ckpt['epoch']
        trainer.global_step = last_ckpt['global_step']
        del last_ckpt

    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    agent_type = cfg['train']['agent']
    n_demos = cfg['train']['n_demos']
    n_val = cfg['train']['n_val']
    name = '{}-{}-{}'.format(task, agent_type, n_demos)

    # # Datasets
    # dataset_type = cfg['dataset']['type']
    # if 'multi' in dataset_type:
    #     train_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=True)
    #     val_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False)
    # else:
    #     train_ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True)
    #     val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)


    # # Weight Trainer

    # checkpoint_path = os.path.join(cfg['train']['train_dir'], 'weight_checkpoints')
    # last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    # last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None

    # w_max_epochs = 100
    # w_trainer = Trainer(
    #     gpus=cfg['train']['gpu'],
    #     fast_dev_run=cfg['debug'],
    #     logger=wandb_logger,
    #     # checkpoint_callback=checkpoint_callback,
    #     max_epochs=w_max_epochs,
    #     automatic_optimization=False,
    #     check_val_every_n_epoch=1,
    #     resume_from_checkpoint=last_checkpoint,
    # )
    # w_train_ds = WeightTrainerDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True)
    # w_val_ds = WeightTrainerDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)
    # w_agent = agents.names['weight_trainer'](name, cfg, w_train_ds, w_val_ds)
    # w_trainer.fit(w_agent)


    # Semantic-Spatial Pre-trainer Datasets

    train_ds = SSPretrainDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True)
    val_ds = SSPretrainDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)

    # Initialize agent
    agent = agents.names[agent_type](name, cfg, train_ds, val_ds)

    # Main training loop
    trainer.fit(agent)

if __name__ == '__main__':
    main()
