import argparse
import os
import copy
import torch
import torch.backends.cudnn as cudnn
import optuna
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr_spectra import SimCLR
from datasets.spectra_dataset import spectral_collate_fn, MixedDataset
from preprocessing import PreprocessNIR
from finetuning import finetuning_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for SimCLR")
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--study-name', type=str, default=None, help='Study name for Optuna')
    parser.add_argument('--storage', type=str, default=None, help='RDB storage URL for Optuna')
    parser.add_argument('--data', type=str, default='./datasets/spectra', help='Path to spectral dataset')
    parser.add_argument('--dataset-name', default='spectra', choices=['stl10','cifar10','spectra'], help='Dataset name')
    parser.add_argument('--arch', default='transformer', choices=['transformer'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=200, help='Number of pretraining epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for pretraining')
    parser.add_argument('--workers', type=int, default=12, help='Data loader worker count')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--gpu-index', type=int, default=0, help='GPU index to use')
    parser.add_argument('--n-views', type=int, default=2, help='Number of views for contrastive training')
    parser.add_argument('--epochs-adapt', type=int, default=10, help='Adaptation epochs for finetuning')
    parser.add_argument('--batch-size-adapt', type=int, default=32, help='Batch size for adaptation')
    parser.add_argument('--k-spt', type=int, default=25, help='Support set size')
    parser.add_argument('--k-qry', type=int, default=25, help='Query set size')
    parser.add_argument('--emb-size', type=int, default=128, help='Embedding size for regression')
    # Architecture parameters
    parser.add_argument('--nhead', type=int, default=4, help='Number of heads in the transformer')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--dim-feedforward', type=int, default=256, help='Dimension of the feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in the transformer')
    parser.add_argument('--use-mean-pool', action='store_true', help='Use mean pooling in the transformer')
    parser.add_argument('--wav-min', type=float, default=190.0, help='Minimum wavelength for spectral data')
    parser.add_argument('--wav-max', type=float, default=25000.0, help='Maximum wavelength for spectral data')
    parser.add_argument('--out-dir', type=str, default='./optuna_results', help='Directory to save Optuna results')
    parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')

    parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
    return parser.parse_args()


def set_device(args):
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1


def objective(trial, args):
    # Sample hyperparameters
    # Architecture parameters
    args.emb_size = trial.suggest_int('emb_size', 64, 256, step=64)
    args.nhead = trial.suggest_int('nhead', 1, 8)
    args.num_layers = trial.suggest_int('num_layers', 1, 4)
    args.dim_feedforward = trial.suggest_int('dim_feedforward', 128, 512, step=128)
    args.use_mean_pool = trial.suggest_categorical('use_mean_pool', [True, False])
    # Ensure compatibility: embedding size must be divisible by number of heads
    if args.emb_size % args.nhead != 0:
        raise optuna.exceptions.TrialPruned("Incompatible architecture: emb_size must be divisible by nhead")
    # Training hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    temperature = trial.suggest_float('temperature', 0.01, 0.5)
    out_dim = trial.suggest_int('out_dim', 64, 256, step=64)
    lr_adapt = 0.0001

    # Update args for training
    args.lr = lr
    args.temperature = temperature
    args.out_dim = out_dim
    args.lr_adapt = lr_adapt

    args.name = os.path.join(args.out_dir, f"trial_{trial.number}")
    os.makedirs(args.name, exist_ok=True)

    # Prepare data
    dataset = ContrastiveLearningDataset(args.data)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        collate_fn=spectral_collate_fn)

    # Build model
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, args=args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0)

    # Pretrain SimCLR
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader, trial=trial)

    # Finetuning evaluation
    results = finetuning_eval(
        simclr.model,
        epochs=args.epochs_adapt,
        lr=args.lr_adapt,
        batch_size=args.batch_size_adapt,
        emb_size=args.emb_size,
        repeats=1,
        k_spt=args.k_spt,
        k_qry=args.k_qry,
        device=args.device)
    # Maximize R2 score
    r2 = results.loc['mean', 'r2_test']
    return float(r2)


def main():
    args = parse_args()
    set_device(args)
    os.makedirs(args.out_dir, exist_ok=True)

    # Create or load Optuna study
    if args.storage:
        study = optuna.create_study(
            direction='maximize',
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True)
    else:
        study = optuna.create_study(direction='maximize')

    # Optimize
    if args.timeout:
        study.optimize(lambda trial: objective(trial, args), timeout=args.timeout, n_trials=args.n_trials, catch=(Exception,))
    else:
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, catch=(Exception,))

    # Save results
    print("Best trial parameters:", study.best_trial.params)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(args.out_dir, 'trials.csv'), index=False)


if __name__ == '__main__':
    main()
