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
    for views, _, _, _ in batch:
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

    # 5) Padding de wl (una sola por muestra) y máscara
    wls = [wl for _, wl, _, _ in batch]
    wl_padded = pad_sequence(wls, batch_first=True, padding_value=0.)  # (B, L_max)
    pad_mask  = (wl_padded == 0.)                                      # (B, L_max)

    # 6) Targets
    ys = [y for _, _, y, _ in batch]
    y_batch = torch.stack(ys, dim=0)                            # (B, …)

    return views_padded, wl_padded, pad_mask, y_batch

# def spectral_collate_fn(batch):
#     """
#     batch: lista de tuplas (views, wl, y, mask)
#       - views: lista de n_views tensores 1D (L_i,)
#       - wl:     tensor 1D (L_i,)
#       - y:      tensor target
#       - mask:   tensor bool 1D (L_i,)
#     Devuelve:
#       views_padded: lista de n_views tensores (B, 1, L_max)
#       wl_padded:    tensor (B, L_max)
#       pad_mask:     tensor bool (B, L_max)
#       y_batch:      tensor targets (B, …)
#       mask_batch:   tensor masks y (B, …)
#     """
#     B = len(batch)
#     n_views = len(batch[0][0])
#     # Inicializar listas por vista
#     views_batches = [[] for _ in range(n_views)]
#     wls, ys, mask_y = [], [], []

#     # Desempaquetar elementos
#     for views, wl, y, m in batch:
#         for i, v in enumerate(views):
#             views_batches[i].append(v)
#         wls.append(wl)
#         ys.append(y)
#         mask_y.append(m)

#     # Padding de cada vista por separado
#     views_padded = []
#     for view_list in views_batches:
#         # pasar de (L_i,) a (L_i,1) para pad_sequence
#         seq = [v.unsqueeze(1) for v in view_list]        # List[(L_i,1)]
#         pad = pad_sequence(seq, batch_first=True, padding_value=0.)  # (B, L_max,1)
#         pad = pad.permute(0,2,1)                          # (B, 1, L_max)
#         views_padded.append(pad)

#     # Padding de longitudes y máscara de padding
#     wl_padded = pad_sequence(wls, batch_first=True, padding_value=0.)  # (B, L_max)
#     pad_mask = (wl_padded == 0.)                                      # (B, L_max)

#     # Batch de targets y masks
#     y_batch    = torch.stack(ys,     dim=0)  # (B, …)
#     mask_batch = torch.stack(mask_y, dim=0)  # (B, …)

#     return views_padded, wl_padded, pad_mask, y_batch, mask_batch

class SpectraDataset(Dataset):
    def __init__(self, root_folder="datasets/spectra", mode="supp", transform=None):
        assert mode in ['supp', 'query'], "Mode must be either 'supp' or 'query'"
        self.mode = mode
        self.transform = transform

        # Carga de datos
        self.X = pd.read_csv(os.path.join(root_folder, f'X_{mode}.csv'))
        self.y = pd.read_excel(os.path.join(root_folder, f'y_{mode}.xlsx'))

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
            intensities = self.transform(intensities)

        # Etiquetas y
        y_row = self.y.iloc[idx].values.astype(float)
        mask_y = ~np.isnan(y_row)
        # Normalizar y, preservando NaN
        y_scaled = (y_row - self.y_mean) / self.y_std

        # Tensores
        wl_t   = torch.tensor(wl, dtype=torch.float32)
        # I_t    = torch.tensor(intensities, dtype=torch.float32)
        I_t    = intensities
        y_t    = torch.tensor(y_scaled, dtype=torch.float32)
        mask_t = torch.tensor(mask_y, dtype=torch.bool)

        return I_t, wl_t, y_t, mask_t
    
    def dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=agg_collate_fn)