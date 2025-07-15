import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from copy import deepcopy
    
import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Any, Callable, cast, Optional, Tuple, Union

# def spectral_collate_fn(batch):
#     """
#     batch: lista de tuples (xi, yi, wli)
#       - xi: Tensor (C, L_i)
#       - yi: Tensor (…)
#       - wli: Tensor (L_i,)
#     Devuelve:
#       x_padded: (B, C, L_max)
#       wl_padded: (B, L_max)
#       pad_mask:  (B, L_max)  True = padding
#       y:         Tensor con todos los yi
#     """
#     xs, wls, ys, mask = zip(*batch)
#     B = len(xs)

#     # Transponer xi a (L_i, C) para pad_sequence
#     xs_seq = [x.permute(1,0) for x in xs]  # lista de (L_i, C)
#     x_padded = pad_sequence(xs_seq, batch_first=True, padding_value=0.)  # (B, L_max, C)
#     x_padded = x_padded.permute(0,2,1)  # (B, C, L_max)

#     # Pad wavelengths
#     wl_padded = pad_sequence(wls, batch_first=True, padding_value=0.)  # (B, L_max)

#     # Máscara: True donde es padding (longitud = 0)
#     pad_mask = (wl_padded == 0.)  # (B, L_max)

#     # Targets
#     y = torch.stack(ys, dim=0)

#     return x_padded, wl_padded, pad_mask, y    

from torch.nn.utils.rnn import pad_sequence

def spectral_collate_fn(batch):
    """
    batch: lista de tuplas (views, wl, y, mask_y)
      - views: lista de n_views tensores 1D (L_i,)
      - wl:     tensor 1D (L_i,)
      - y:      tensor target
      - mask_y: tensor bool 1D (L_i,)
    Devuelve:
      views_padded: lista de n_views tensores (B, 1, L_max) con idéntico padding
      wl_padded:    tensor (B, L_max)
      pad_mask:     tensor bool (B, L_max)
      y_batch:      tensor targets (B, …)
    """
    from torch.nn.utils.rnn import pad_sequence

    B = len(batch)
    n_views = len(batch[0][0])

    # 1) Flatten todas las vistas en una sola lista
    flat_views = []
    for views, _, _, _, _ in batch:
        flat_views.extend(views)

    # 2) Hacer pad de todos juntos para garantizar el mismo L_max
    #    Cada vista v: Tensor (L_i,) → (L_i, 1)
    seq = [v.unsqueeze(1) for v in flat_views]                 # List[(L_i,1)]
    pad_all = pad_sequence(seq, batch_first=True, padding_value=0.)   # (B*n_views, L_max, 1)
    pad_all = pad_all.permute(0, 2, 1)                           # (B*n_views, 1, L_max)

    # 3) Volver a agrupar en (B, n_views, 1, L_max)
    L_max = pad_all.size(-1)
    pad_all = pad_all.view(B, n_views, 1, L_max)                # (B, n_views, 1, L_max)

    # 4) Separar por vista
    views_padded = [pad_all[:, i] for i in range(n_views)]      # lista de n_views tensores (B,1,L_max)

    # 5) Padding de wl para cada vista y máscaras
    flat_wls = []
    for _, wls_sample, _, _, _ in batch:
        # wls_sample: lista de n_views tensores 1D
        flat_wls.extend(wls_sample)
    # Todas las vistas juntas: (B*n_views, L_i) → pad a (B*n_views, L_max)
    pad_all_wl = pad_sequence(flat_wls, batch_first=True, padding_value=0.)  # (B*n_views, L_max)
    # Reagrupar a (B, n_views, L_max)
    pad_all_wl = pad_all_wl.view(B, n_views, -1)
    # Separar por vista
    wls_padded = [pad_all_wl[:, i] for i in range(n_views)]               # lista de n_views tensores (B, L_max)
    # Máscaras de padding por vista
    pad_mask = [(wli == 0.) for wli in wls_padded]

    # 6) Targets
    ys = [y for _, _, y, _, _ in batch]
    y_batch = torch.stack(ys, dim=0)                            # (B, …)

    # 7) datasets
    # Si hay dataset, extraerlo
    if batch[0][-1] is not None:
        datasets = [dataset for _, _, _, _, dataset in batch]
    else:
        datasets = None


    if datasets is not None:
        return views_padded, wls_padded, pad_mask, y_batch, datasets
    else:
        return views_padded, wls_padded, pad_mask, y_batch

class SpectraDataset(Dataset):
    def __init__(self, root_folder="datasets/spectra", mode="supp", transform=None):
        assert mode in ['supp', 'query'], "Mode must be either 'supp' or 'query'"
        self.mode = mode
        self.transform = transform

        # Carga de datos
        self.X = pd.read_csv(os.path.join(root_folder, f'X_{mode}.csv'))
        self.y = pd.read_excel(os.path.join(root_folder, f'y_{mode}.xlsx'))

        if "dataset" in self.y.columns:
            self.dataset = self.y['dataset']
            self.y = self.y.drop(columns=['dataset'])
        else:
            self.dataset = None

        # Wavelength grid
        self.wl_all = self.X.columns.astype(float).to_numpy()

        # Estadísticas de y (ignorando NaN)
        y_vals = self.y.values.astype(float)
        self.y_mean = np.nanmean(y_vals, axis=0)
        self.y_std  = np.nanstd(y_vals,  axis=0)
        self.y_std[self.y_std == 0] = 1.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Espectro
        x_row = self.X.iloc[idx]
        mask_spec = x_row.notna().to_numpy()
        wl = self.wl_all[mask_spec]
        intensities = x_row.values[mask_spec].astype(float)

        if self.transform:
            # Allow transform to adjust both intensities and wl (e.g., cropping)
            transformed = self.transform(intensities, wl)
            if isinstance(transformed, tuple) and len(transformed) == 2:
                intensities, wl = transformed
            else:
                intensities = transformed

        # Etiquetas y
        y_row = self.y.iloc[idx].values.astype(float)
        mask_y = ~np.isnan(y_row)
        # Normalizar y, preservando NaN
        y_scaled = (y_row - self.y_mean) / self.y_std

        # Tensores
        if type(wl) == np.ndarray:
            wl_t   = torch.tensor(wl, dtype=torch.float32)
        else:
            wl_t   = wl
        # I_t    = torch.tensor(intensities, dtype=torch.float32)
        if type(intensities) == np.ndarray:
            I_t = torch.tensor(intensities, dtype=torch.float32).unsqueeze(0)
        else:
            I_t    = intensities
        y_t    = torch.tensor(y_scaled, dtype=torch.float32)
        mask_t = torch.tensor(mask_y, dtype=torch.bool)

        return I_t, wl_t, y_t, mask_t, self.dataset.iloc[idx] if self.dataset is not None else None


class BatchDataset(Dataset):
    def __init__(self, tasks, shots, mode='support'):
        self.tasks = tasks
        self.shots = shots

        x = []
        y = []
        for region in self.tasks.keys():
            if mode == 'support':
                x.append(self.tasks[region].support_x)
                y.append(self.tasks[region].support_y)
            elif mode == 'query':
                x.append(self.tasks[region].query_x)
                y.append(self.tasks[region].query_y)
            else:
                raise ValueError("Mode not found")

        shapes = [t.shape[-1] for t in x]
        if len(set(shapes)) > 1:
            max_shape = max(shapes)
            for i in range(len(x)):
                if x[i].shape[-1] < max_shape:
                    x[i] = torch.cat([x[i], torch.zeros(x[i].shape[0], 1, max_shape - x[i].shape[-1])], dim=-1)
        
        self.x = torch.cat(x, dim=0)
        self.y = torch.cat(y, dim=0)

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

class MixedTask(Dataset):
    def __init__(self, support, query, name, device='cuda', path='data/MixedDataset'):
        self.support_x, self.support_y, self.support_idx, self.support_wl = support
        self.query_x, self.query_y, self.query_idx, self.query_wl = query
        self.name = name
        self.path = path

        if type(self.support_x) == pd.DataFrame:
            self.support_x = torch.tensor(self.support_x.values, dtype=torch.float32)
            self.support_y = torch.tensor(self.support_y.values, dtype=torch.float32)
            self.support_wl = torch.tensor(self.support_wl, dtype=torch.float32)
            self.query_x = torch.tensor(self.query_x.values, dtype=torch.float32)
            self.query_y = torch.tensor(self.query_y.values, dtype=torch.float32)
            self.query_wl = torch.tensor(self.query_wl, dtype=torch.float32)
        elif type(self.support_x) == np.ndarray:
            self.support_x = torch.tensor(self.support_x, dtype=torch.float32)
            self.support_y = torch.tensor(self.support_y, dtype=torch.float32)
            self.query_x = torch.tensor(self.query_x, dtype=torch.float32)
            self.query_y = torch.tensor(self.query_y, dtype=torch.float32)

        if self.support_x.ndim == 2:
            self.support_x = self.support_x.unsqueeze(1)
        if self.query_x.ndim == 2:
            self.query_x = self.query_x.unsqueeze(1)
        if self.support_y.ndim == 1:
            self.support_y = self.support_y.unsqueeze(1)
        if self.query_y.ndim == 1:
            self.query_y = self.query_y.unsqueeze(1)

        self.device = device

    def __len__(self):
        return self.query_x.shape[0]
    
    def __getitem__(self, idx):
        # return self.query_x[idx], self.query_y[idx], self.query_idx[idx], self.query_wl[idx], 
        x = self.query_x[idx]
        y = self.query_y[idx]
        idx_orig = self.query_idx[idx]
        wl = self.query_wl
        # máscara de padding: True donde wl==0, aquí será todo False si no hay ceros
        pad_mask = (wl == 0)
        return x, y, idx_orig, wl, pad_mask
    
    def sample(self, shots, queries):
        if self.support_x.shape[0] >= shots:
            support_idx = np.random.choice(self.support_x.shape[0], shots, replace=False)
        else:
            support_idx = np.random.choice(self.support_x.shape[0], self.support_x.shape[0], replace=False)

        if self.query_x.shape[0] >= queries:
            query_idx = np.random.choice(self.query_x.shape[0], queries, replace=False)
        else:
            query_idx = np.random.choice(self.query_x.shape[0], self.query_x.shape[0], replace=False)

        # tensor 1D con shape (L_max,)
        wl_vec = self.support_wl.to(self.device)
        # convertir en (batch_size, L_max)
        support_wl       = wl_vec.unsqueeze(0).repeat(len(support_idx), 1)
        support_pad_mask = (wl_vec == 0.).unsqueeze(0).repeat(len(support_idx), 1)

        q_wl_vec = self.query_wl.to(self.device)
        query_wl       = q_wl_vec.unsqueeze(0).repeat(len(query_idx), 1)
        query_pad_mask = (q_wl_vec == 0.).unsqueeze(0).repeat(len(query_idx), 1)

        return {'support_features': self.support_x[support_idx].to(self.device),
                'support_targets': self.support_y[support_idx].to(self.device),
                'support_wl'         : support_wl,
                'support_pad_mask'   : support_pad_mask,
                'query_features': self.query_x[query_idx].to(self.device),
                'query_targets': self.query_y[query_idx].to(self.device),
                'query_wl'         : query_wl,
                'query_pad_mask'   : query_pad_mask}

    def sample_fixed(self, shots, queries):
        if not os.path.exists(os.path.join(self.path, f"{self.name}/fixed_val_support_{shots}shots.csv")) or not os.path.exists(os.path.join(self.path, f"{self.name}/fixed_val_support_{shots}shots.csv")):
            if self.support_x.shape[0] >= shots:
                support_idx = np.random.choice(self.support_x.shape[0], shots, replace=False)
            else:
                support_idx = np.random.choice(self.support_x.shape[0], self.support_x.shape[0], replace=False)

            if self.query_x.shape[0] >= queries:
                query_idx = np.random.choice(self.query_x.shape[0], queries, replace=False)
            else:
                query_idx = np.random.choice(self.query_x.shape[0], self.query_x.shape[0], replace=False)

            # val = pd.DataFrame({'support_idx': support_idx, 'query_idx': query_idx})
            # val.to_csv(os.path.join(self.path, f"{self.name}/fixed_val_{shots}shots.csv"))
            pd.DataFrame({'support_idx': support_idx}).to_csv(os.path.join(self.path, f"{self.name}/fixed_val_support_{shots}shots.csv"))
            pd.DataFrame({'query_idx': query_idx}).to_csv(os.path.join(self.path, f"{self.name}/fixed_val_query_{shots}shots.csv"))
        else:
            support_df = pd.read_csv(os.path.join(self.path, f"{self.name}/fixed_val_support_{shots}shots.csv"), index_col=0)
            support_idx = support_df['support_idx'].values

            query_df = pd.read_csv(os.path.join(self.path, f"{self.name}/fixed_val_query_{shots}shots.csv"), index_col=0)
            query_idx = query_df['query_idx'].values

        # build pad masks
        support_pad_mask = (self.support_wl == 0.)
        query_pad_mask   = (self.query_wl   == 0.)

        # …existing code para cargar support_idx, query_idx…
        wl_vec = self.support_wl.to(self.device)
        support_wl       = wl_vec.unsqueeze(0).repeat(len(support_idx), 1)
        support_pad_mask = (wl_vec == 0.).unsqueeze(0).repeat(len(support_idx), 1)

        q_wl_vec = self.query_wl.to(self.device)
        query_wl       = q_wl_vec.unsqueeze(0).repeat(len(query_idx), 1)
        query_pad_mask = (q_wl_vec == 0.).unsqueeze(0).repeat(len(query_idx), 1)

        return {'support_features': self.support_x[support_idx].to(self.device),
                'support_targets': self.support_y[support_idx].to(self.device),
                'support_wl': support_wl,
                'support_pad_mask': support_pad_mask,
                'query_features': self.query_x[query_idx].to(self.device),
                'query_targets': self.query_y[query_idx].to(self.device),
                'query_wl': query_wl,
                'query_pad_mask': query_pad_mask}
    
    def query_dataloader(self):
        dataset = MixedTask(support=(self.query_x, self.query_y, self.query_idx, self.query_wl), query=(self.query_x, self.query_y, self.query_idx, self.query_wl), name=self.name)
        return DataLoader(dataset, batch_size=32, shuffle=False)

    def get_shape(self, idx):
        return self.query_x[idx].shape
    

class MixedDataset(Dataset):
    def __init__(self, path='data/MixedDataset', split='train', supp_sz=5, query_sz=5, preprocessor=None, device='cuda', max_tasks=None):
        self.supp_sz = supp_sz
        self.query_sz = query_sz
        self.device = device
        self.split = split        
        self.dataset_path = path
        self.max_tasks = max_tasks

        split_df = pd.read_csv(os.path.join(self.dataset_path, "splits.csv"), index_col=0)
        tasks_names_split = split_df.query(f"split == '{split}'")['task'].values.tolist()

        if self.max_tasks is not None:
            if os.path.exists(os.path.join(self.dataset_path, "choices.json")):
                import json
                with open(os.path.join(self.dataset_path, "choices.json"), 'r') as f:
                    choices = json.load(f)
                tasks_names_split = choices[str(self.max_tasks)]
            else:
                if len(tasks_names_split) > self.max_tasks:
                    tasks_names_split = np.random.choice(tasks_names_split, self.max_tasks, replace=False).tolist()
                else:
                    tasks_names_split = np.random.choice(tasks_names_split, len(tasks_names_split), replace=False).tolist()

        self.tasks, self.task_names = self.__create_tasks(preprocessor=preprocessor, path_data=self.dataset_path, tasks_names_split=tasks_names_split)

        # assert(len(self.tasks) == len(tasks_names_split))

    def __len__(self):
        return len(self.tasks)
    
    def __create_tasks(self, preprocessor=None, path_data=None, tasks_names_split=None):
        tasks = {}
        for dataset in os.listdir(path_data):
            if dataset.strip() in tasks_names_split:
                support_x = pd.read_csv(os.path.join(path_data, dataset, "X_supp.csv"), index_col=0)
                support_y = pd.read_csv(os.path.join(path_data, dataset, "y_supp.csv"), index_col=0)
                query_x = pd.read_csv(os.path.join(path_data, dataset, "X_query.csv"), index_col=0)
                query_y = pd.read_csv(os.path.join(path_data, dataset, "y_query.csv"), index_col=0)
                wl_supp = torch.tensor([float(w) for w in support_x.columns.tolist()])
                wl_query = torch.tensor([float(w) for w in query_x.columns.tolist()])
                support_idx = support_x.index
                query_idx = query_x.index

                if preprocessor is not None:
                    support_x, support_y = preprocessor.fit_transform(support_x, support_y)
                    query_x, query_y = preprocessor.transform(query_x, query_y)

                tasks[dataset] = MixedTask(support=(support_x, support_y, support_idx, wl_supp), query=(query_x, query_y, query_idx, wl_query), name=dataset, path=self.dataset_path)
        return tasks, list(tasks.keys())
    
    def __getitem__(self, idx):
        name = self.task_names[idx]
        return self.tasks[name]
    
    def sample_task(self):
        task = np.random.choice(self.tasks)
        return task
    
    def sample(self):
        task = self.sample_task()
        return task.sample(self.supp_sz, self.query_sz)

    def batch_dataloader(self, mode='support'):
        return DataLoader(BatchDataset(self.tasks, self.supp_sz, mode), batch_size=32, shuffle=True)