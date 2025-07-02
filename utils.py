import os
import shutil

import torch
import yaml


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

import copy
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

def finetuning(model, x_supp, y_supp, wl_supp, supp_pad_mask, x_query, y_query, wl_query, query_pad_mask, epochs=10, lr=0.0001, batch_size=32):
    fast_weights = copy.deepcopy(model)
    fast_weights.train()
    optimizer = torch.optim.Adam(fast_weights.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    support_dataset = torch.utils.data.TensorDataset(x_supp, y_supp, wl_supp, supp_pad_mask)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=batch_size, shuffle=True)
    
    for i in range(epochs):
        for x_batch, y_batch, wl_batch, pad_mask_batch in support_loader:
            logits = fast_weights(x_batch, wl_batch, pad_mask_batch)
            loss = loss_fn(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            fast_weights.eval()
            logits = fast_weights(x_query, wl_query, query_pad_mask)
            mse = loss_fn(logits, y_query).item()
            mae = torch.abs(logits - y_query).mean().item()
            rmse = torch.sqrt(loss_fn(logits, y_query)).item()
            r2 = r2_score(y_query.detach().cpu().numpy(), logits.detach().cpu().numpy())
    return fast_weights, mse, mae, rmse, r2