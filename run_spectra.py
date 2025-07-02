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

model_names = ['transformer']

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--name', type=str, default='exp', help='experiment name')
parser.add_argument('-data', metavar='DIR', default='./datasets/spectra',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='spectra',
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

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
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

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)

    print("Training completed.")

    output_dir = simclr.writer.log_dir

    results_df = []
    for rep in range(5):
        print(f"Running evaluation for repetition {rep + 1}...")
        
        # Test model
        preprocessor = PreprocessNIR(savgol=False, scale=False, scale_y=True, window_length=15, polyorder=2, deriv=1)
        test_data = MixedDataset(path="datasets/MixedDataset", split='test', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)

        transformer_adapt = copy.deepcopy(simclr.model)
        model_adapt = RegressionModel(base_model=transformer_adapt, emb_size=args.out_dim, output_dim=1)
        model_adapt.to(args.device)

        # Test model
        mses_test = []
        maes_test = []
        rmses_test = []
        r2s_test = []
        full_mses_test = []
        full_maes_test = []
        full_rmses_test = []
        full_r2s_test = []

        preds_df_test = []
        for test_task in test_data:
            data = test_task.sample_fixed(args.k_spt, args.k_qry)
            x_supp = data["support_features"]
            y_supp = data["support_targets"]
            wl_supp = data["support_wl"]
            supp_pad_mask = data["support_pad_mask"]
            x_query = data["query_features"]
            y_query = data["query_targets"]
            wl_query = data["query_wl"]
            query_pad_mask = data["query_pad_mask"]

            fast_weights, mse, mae, rmse, r2 = finetuning(model_adapt, x_supp, y_supp, wl_supp, supp_pad_mask, 
                                                          x_query, y_query, wl_query, query_pad_mask,
                                                        epochs=args.epochs_adapt, lr=args.lr_adapt, 
                                                        batch_size=args.batch_size_adapt)
            
            mses_test.append(mse)
            maes_test.append(mae)
            rmses_test.append(rmse)
            r2s_test.append(r2)

            y_true = []
            y_pred = []
            y_idx = []

            full_mse = 0.0
            full_mae = 0.0
            full_rmse = 0.0
            full_r2 = 0.0
            num_instances = 0
            test_dl = test_task.query_dataloader()
            for x, y, idx, wl, mask in test_dl:
                y_true.extend(y.detach().numpy().squeeze(1).tolist())
                x, y = x.to(args.device), y.to(args.device)
                wl, mask = wl.to(args.device), mask.to(args.device)
                logits = fast_weights(x, wl, mask)
                full_mse += ((logits - y) ** 2).sum().item()
                full_mae += torch.abs(logits - y).sum().detach().cpu()
                num_instances += y.size(0)
                
                y_pred.extend(logits.detach().cpu().numpy().squeeze(1).tolist())
                y_idx.extend(idx.detach().numpy().tolist())

            full_mse /= num_instances
            full_mae /= num_instances
            full_rmse = np.sqrt(full_mse)
            full_r2 = r2_score(y_true, y_pred)
            full_mses_test.append(full_mse)
            full_maes_test.append(full_mae)
            full_rmses_test.append(full_rmse)
            full_r2s_test.append(full_r2)

            preds_df_test.append(pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "idx": y_idx, "name": test_task.name}))
        
        # Save predictions
        preds_df_test = pd.concat(preds_df_test, axis=0).set_index("idx").sort_index()
        preds_df_test.to_csv(os.path.join(output_dir, f"predictions_test_{rep}.csv"), index=True)

        mses_test = np.array(mses_test).mean(axis=0).astype(np.float16)
        maes_test = np.array(maes_test).mean(axis=0).astype(np.float16)
        rmses_test = np.array(rmses_test).mean(axis=0).astype(np.float16)
        r2s_test = np.array(r2s_test).mean(axis=0).astype(np.float16)
        full_mses_test = np.array(full_mses_test).mean().astype(np.float16)
        full_maes_test = np.array(full_maes_test).mean().astype(np.float16)
        full_rmses_test = np.array(full_rmses_test).mean().astype(np.float16)
        full_r2s_test = np.array(full_r2s_test).mean().astype(np.float16)

        print(f"Test | MSE: {mses_test:.4f} | MAE: {maes_test:.4f} | Full MSE: {full_mses_test:.4f} | Full MAE: {full_maes_test:.4f}", flush=True)
        print(f"Test | RMSE: {rmses_test:.4f} | R2: {r2s_test:.4f} | Full RMSE: {full_rmses_test:.4f} | Full R2: {full_r2s_test:.4f}", flush=True)

        results_df_rep = pd.DataFrame({"mse_test": [full_mses_test], "mae_test": [full_maes_test], 
                                    "rmse_test": [full_rmses_test], "r2_test": [full_r2s_test]}, index=[rep])
        results_df.append(results_df_rep)

    results_df = pd.concat(results_df, axis=0)
    results_df.loc["mean"] = results_df.mean(axis=0)
    results_df.to_csv(os.path.join(output_dir, "results.csv"), index=True)


if __name__ == "__main__":
    main()
