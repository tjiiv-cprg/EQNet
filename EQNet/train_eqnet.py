from shared_libs.env_config_parse import extend_extra_sys_path
extend_extra_sys_path("EQNet/env_config.yml")

import torch
torch.set_float32_matmul_precision('medium')

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger

from EQNet.argoverse_v2_datamodule import ArgoverseV2DataModule
from eqnet import EQNet
from EQNet import data
from shared_libs.env_config_parse import check_git_status_and_get_branch

BRANCH = check_git_status_and_get_branch() or "DIRTY"
tblogger = TensorBoardLogger(save_dir="results", name=BRANCH)
LOGDIR = tblogger.log_dir

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--machine', type=str)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--grad_acc', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)
    # parser.add_argument('--load_pretrained_qcnet', type=str, default=None)
    parser.add_argument('--load_pretrained', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    EQNet.add_model_specific_args(parser)
    args = parser.parse_args()

    args.submission_dir = LOGDIR
    
    if args.dataset == 'argoverse_v2':
        args.root = data.get_argoverse_v2_root(args.machine)
        datamodule = ArgoverseV2DataModule(**vars(args))
    else:
        print(f"[WARNING]: unknown dataset {args.dataset}")
    
    print(args)

    if args.load_pretrained:
        model = EQNet.load_from_checkpoint(args.load_pretrained, **vars(args), strict=False)
    else:
        model = EQNet(**vars(args))
        # if args.load_pretrained_qcnet:
        #     model.load_pretrained_qcnet(args.load_pretrained_qcnet)

    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    if args.devices > 1:
        strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
    else:
        strategy = "auto"
    print(f"[INFO]: {strategy=}")
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy=strategy,
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs,
                         logger=tblogger)
    trainer.fit(model, datamodule, ckpt_path=args.resume)
