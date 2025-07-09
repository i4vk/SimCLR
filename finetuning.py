import argparse
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from models.resnet_simclr import ResNetSimCLR
from datasets.spectra_dataset import MixedDataset
from preprocessing import PreprocessNIR
from utils import finetuning
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
import os
from models.transformer import RegressionModel

model_names = ['transformer']

parser = argparse.ArgumentParser(description='PyTorch SimCLR Finetuning Evaluation')
parser.add_argument('--checkpoint', type=str, required=True, 
                    help='path to pretrained model checkpoint')
parser.add_argument('--name', type=str, default='finetuning_exp', 
                    help='experiment name')
parser.add_argument('-data', metavar='DIR', default='./datasets/spectra',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='spectra',
                    help='dataset name', choices=['stl10', 'cifar10', 'spectra'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='transformer',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: transformer)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')

parser.add_argument("--lr_adapt", type=float, default=0.01, 
                    help="learning rate for inner loop")
parser.add_argument("--k_spt", type=int, default=25, 
                    help="number of support samples per class")
parser.add_argument("--k_qry", type=int, default=25, 
                    help="number of query samples per class")
parser.add_argument("--epochs_adapt", type=int, default=10, 
                    help="number of epochs for adaptation")
parser.add_argument("--batch_size_adapt", type=int, default=32, 
                    help="batch size for adaptation")
parser.add_argument("--repetitions", type=int, default=5, 
                    help="number of evaluation repetitions")

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


def load_checkpoint(checkpoint_path, model, device):
    """Load a pretrained model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the state dict
    model.load_state_dict(checkpoint['state_dict'])
    
    # Print checkpoint info
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Model architecture: {checkpoint['arch']}")
    
    return model

def finetuning_eval(model, epochs=10, lr=0.01, batch_size=32, emb_size=128,
                    repeats=1, k_spt=25, k_qry=25, device="cuda", output_dir=None):
    """Finetune the model on the support set and evaluate on the query set."""
    results_df = []
    model = copy.deepcopy(model)
    model.backbone.fc = nn.Sequential(nn.Linear(emb_size, emb_size), nn.ReLU(), nn.Linear(emb_size, 1))
    model.to(device)
    model.eval()

    for rep in range(repeats):        
        # Load test data
        preprocessor = PreprocessNIR(savgol=False, scale=False, scale_y=True, 
                                   window_length=15, polyorder=2, deriv=1)
        test_data = MixedDataset(path="datasets/MangoDataset_by_year", split='test', 
                               supp_sz=k_spt, query_sz=k_qry, 
                               preprocessor=preprocessor)

        # Create adaptation model
        model_adapt = copy.deepcopy(model)
        # model_adapt = RegressionModel(base_model=transformer_adapt, 
        #                             emb_size=args.out_dim, output_dim=1)
        # model_adapt.to(args.device)

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
        
        for task_idx, test_task in enumerate(test_data):                
            data = test_task.sample_fixed(k_spt, k_qry)
            x_supp = data["support_features"]
            y_supp = data["support_targets"]
            wl_supp = data["support_wl"]
            supp_pad_mask = data["support_pad_mask"]
            x_query = data["query_features"]
            y_query = data["query_targets"]
            wl_query = data["query_wl"]
            query_pad_mask = data["query_pad_mask"]

            fast_weights, mse, mae, rmse, r2 = finetuning(
                model_adapt, x_supp, y_supp, wl_supp, supp_pad_mask, 
                x_query, y_query, wl_query, query_pad_mask,
                epochs=epochs, lr=lr, 
                batch_size=batch_size
            )
            
            mses_test.append(mse)
            maes_test.append(mae)
            rmses_test.append(rmse)
            r2s_test.append(r2)

            # Evaluate on full query set
            y_true = []
            y_pred = []
            y_idx = []

            full_mse = 0.0
            full_mae = 0.0
            num_instances = 0

            fast_weights.eval()           
            test_dl = test_task.query_dataloader()
            for x, y, idx, wl, mask in test_dl:
                y_true.extend(y.detach().numpy().squeeze(1).tolist())
                x, y = x.to(device), y.to(device)
                wl, mask = wl.to(device), mask.to(device)
                
                with torch.no_grad():
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

            preds_df_test.append(pd.DataFrame({
                "y_true": y_true, 
                "y_pred": y_pred, 
                "idx": y_idx, 
                "name": test_task.name
            }))
        
        # Save predictions
        preds_df_test = pd.concat(preds_df_test, axis=0).set_index("idx").sort_index()
        if output_dir is not None:
            preds_df_test.to_csv(os.path.join(output_dir, f"predictions_test_{rep}.csv"),
                                 index=True)

        # Calculate mean metrics
        mses_test = np.array(mses_test).mean(axis=0).astype(np.float16)
        maes_test = np.array(maes_test).mean(axis=0).astype(np.float16)
        rmses_test = np.array(rmses_test).mean(axis=0).astype(np.float16)
        r2s_test = np.array(r2s_test).mean(axis=0).astype(np.float16)
        full_mses_test = np.array(full_mses_test).mean().astype(np.float16)
        full_maes_test = np.array(full_maes_test).mean().astype(np.float16)
        full_rmses_test = np.array(full_rmses_test).mean().astype(np.float16)
        full_r2s_test = np.array(full_r2s_test).mean().astype(np.float16)

        results_df_rep = pd.DataFrame({
            "mse_test": [full_mses_test], 
            "mae_test": [full_maes_test], 
            "rmse_test": [full_rmses_test], 
            "r2_test": [full_r2s_test]
        }, index=[rep])
        results_df.append(results_df_rep)

    # Combine and save final results
    results_df = pd.concat(results_df, axis=0)
    results_df.loc["mean"] = results_df.mean(axis=0)
    results_df.loc["std"] = results_df.iloc[:-1].std(axis=0)  # Exclude mean row from std calculation
    return results_df


def main():
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # Create output directory for results
    checkpoint_dir = os.path.dirname(args.checkpoint)
    output_dir = os.path.join(checkpoint_dir, f"finetuning_eval_{args.name}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")

    # Load pretrained model
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, args=args)
    model = load_checkpoint(args.checkpoint, model, args.device)
    model.backbone.fc = nn.Sequential(nn.Linear(args.emb_size, args.emb_size), nn.ReLU(), nn.Linear(args.emb_size, 1))
    model.to(args.device)
    model.eval()

    print("Model loaded successfully. Starting evaluation...")

    results_df = []
    for rep in range(args.repetitions):
        print(f"Running evaluation for repetition {rep + 1}/{args.repetitions}...")
        
        # Load test data
        preprocessor = PreprocessNIR(savgol=False, scale=False, scale_y=True, 
                                   window_length=15, polyorder=2, deriv=1)
        test_data = MixedDataset(path="datasets/MixedDataset", split='test', 
                               supp_sz=args.k_spt, query_sz=args.k_qry, 
                               preprocessor=preprocessor)

        # Create adaptation model
        model_adapt = copy.deepcopy(model)
        # model_adapt = RegressionModel(base_model=transformer_adapt, 
        #                             emb_size=args.out_dim, output_dim=1)
        # model_adapt.to(args.device)

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
        
        for task_idx, test_task in enumerate(test_data):
            if task_idx % 5 == 0:
                print(f"  Processing task {task_idx + 1}/{len(test_data)}")
                
            data = test_task.sample_fixed(args.k_spt, args.k_qry)
            x_supp = data["support_features"]
            y_supp = data["support_targets"]
            wl_supp = data["support_wl"]
            supp_pad_mask = data["support_pad_mask"]
            x_query = data["query_features"]
            y_query = data["query_targets"]
            wl_query = data["query_wl"]
            query_pad_mask = data["query_pad_mask"]

            fast_weights, mse, mae, rmse, r2 = finetuning(
                model_adapt, x_supp, y_supp, wl_supp, supp_pad_mask, 
                x_query, y_query, wl_query, query_pad_mask,
                epochs=args.epochs_adapt, lr=args.lr_adapt, 
                batch_size=args.batch_size_adapt
            )
            
            mses_test.append(mse)
            maes_test.append(mae)
            rmses_test.append(rmse)
            r2s_test.append(r2)

            # Evaluate on full query set
            y_true = []
            y_pred = []
            y_idx = []

            full_mse = 0.0
            full_mae = 0.0
            num_instances = 0

            fast_weights.eval()           
            test_dl = test_task.query_dataloader()
            for x, y, idx, wl, mask in test_dl:
                y_true.extend(y.detach().numpy().squeeze(1).tolist())
                x, y = x.to(args.device), y.to(args.device)
                wl, mask = wl.to(args.device), mask.to(args.device)
                
                with torch.no_grad():
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

            preds_df_test.append(pd.DataFrame({
                "y_true": y_true, 
                "y_pred": y_pred, 
                "idx": y_idx, 
                "name": test_task.name
            }))
        
        # Save predictions
        preds_df_test = pd.concat(preds_df_test, axis=0).set_index("idx").sort_index()
        preds_df_test.to_csv(os.path.join(output_dir, f"predictions_test_{rep}.csv"), 
                           index=True)

        # Calculate mean metrics
        mses_test = np.array(mses_test).mean(axis=0).astype(np.float16)
        maes_test = np.array(maes_test).mean(axis=0).astype(np.float16)
        rmses_test = np.array(rmses_test).mean(axis=0).astype(np.float16)
        r2s_test = np.array(r2s_test).mean(axis=0).astype(np.float16)
        full_mses_test = np.array(full_mses_test).mean().astype(np.float16)
        full_maes_test = np.array(full_maes_test).mean().astype(np.float16)
        full_rmses_test = np.array(full_rmses_test).mean().astype(np.float16)
        full_r2s_test = np.array(full_r2s_test).mean().astype(np.float16)

        print(f"  Rep {rep + 1} Results:")
        print(f"  Test | MSE: {mses_test:.4f} | MAE: {maes_test:.4f} | Full MSE: {full_mses_test:.4f} | Full MAE: {full_maes_test:.4f}")
        print(f"  Test | RMSE: {rmses_test:.4f} | R2: {r2s_test:.4f} | Full RMSE: {full_rmses_test:.4f} | Full R2: {full_r2s_test:.4f}")

        results_df_rep = pd.DataFrame({
            "mse_test": [full_mses_test], 
            "mae_test": [full_maes_test], 
            "rmse_test": [full_rmses_test], 
            "r2_test": [full_r2s_test]
        }, index=[rep])
        results_df.append(results_df_rep)

    # Combine and save final results
    results_df = pd.concat(results_df, axis=0)
    results_df.loc["mean"] = results_df.mean(axis=0)
    results_df.loc["std"] = results_df.iloc[:-1].std(axis=0)  # Exclude mean row from std calculation
    
    results_df.to_csv(os.path.join(output_dir, "results.csv"), index=True)
    
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Mean MSE:  {results_df.loc['mean', 'mse_test']:.4f} ± {results_df.loc['std', 'mse_test']:.4f}")
    print(f"Mean MAE:  {results_df.loc['mean', 'mae_test']:.4f} ± {results_df.loc['std', 'mae_test']:.4f}")
    print(f"Mean RMSE: {results_df.loc['mean', 'rmse_test']:.4f} ± {results_df.loc['std', 'rmse_test']:.4f}")
    print(f"Mean R2:   {results_df.loc['mean', 'r2_test']:.4f} ± {results_df.loc['std', 'r2_test']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
