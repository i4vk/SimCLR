import argparse
import copy
import torch
import torch.backends.cudnn as cudnn
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr_spectra import SimCLR
from datasets.spectra_dataset import spectral_collate_fn, MixedDataset
from preprocessing import PreprocessNIR
from utils import finetuning
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
import os
from models.transformer import RegressionModel
import torch.nn as nn

from finetuning import finetuning_eval

model_names = ['transformer']

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--name', type=str, default='exp', help='experiment name')
parser.add_argument('--data', metavar='DIR', default='./datasets/spectra',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='spectra',
                    help='dataset name', choices=['stl10', 'cifar10', 'spectra'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='transformer',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: transformer)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

parser.add_argument("--lr_adapt", type=float, default=0.01, help="learning rate for inner loop")
parser.add_argument("--k_spt", type=int, default=25, help="number of support samples per class")
parser.add_argument("--k_qry", type=int, default=25, help="number of query samples per class")
parser.add_argument("--epochs_adapt", type=int, default=10, help="number of epochs for adaptation")
parser.add_argument("--batch_size_adapt", type=int, default=32, help="batch size for adaptation")

parser.add_argument("--emb_size", type=int, default=128, 
                    help="embedding size for tokens")
parser.add_argument("--nhead", type=int, default=4, 
                    help="number of heads in the transformer")
parser.add_argument("--num_layers", type=int, default=1, 
                    help="number of transformer layers")
parser.add_argument("--dim_feedforward", type=int, default=256, 
                    help="dimension of the feedforward network in the transformer")
parser.add_argument("--dropout", type=float, default=0.1, 
                    help="dropout rate in the transformer")
parser.add_argument("--use_mean_pool", action='store_true', 
                    help="Use mean pooling in the transformer")
parser.add_argument("--wav_min", type=float, default=190, 
                    help="Minimum wavelength for spectral data")
parser.add_argument("--wav_max", type=float, default=25000, 
                    help="Maximum wavelength for spectral data")
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        collate_fn=spectral_collate_fn)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, args=args)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)

    print("Training completed.")

    output_dir = simclr.writer.log_dir

    results_test = finetuning_eval(simclr.model, epochs=args.epochs_adapt, lr=args.lr_adapt,
                                   batch_size=args.batch_size_adapt, emb_size=args.emb_size,
                                   repeats=5, k_spt=args.k_spt, k_qry=args.k_qry,
                                   device=args.device, output_dir=output_dir)
    results_test.to_csv(os.path.join(output_dir, "results.csv"), index=True)


if __name__ == "__main__":
    main()
