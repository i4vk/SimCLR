import math
import torch
from torch import nn
from torch.nn import functional as F

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SpectralPositionalEncoding(nn.Module):
    def __init__(
        self,
        wav_min: float,
        wav_max: float,
        emb_size: int,
        factor: float = 1000.0,
        base: float = 10000.0,
    ):
        """
        Positional Encoding para longitudes de onda espectrales.

        Args:
            wav_min: λ_min (mínima longitud de onda esperada).
            wav_max: λ_max (máxima longitud de onda esperada).
            emb_size: dimensión del embedding (X).
            factor: constante multiplicativa (1000 en el artículo).
            base: base del exponencial en el denominador (10000 en el artículo).
        """
        super().__init__()
        self.wav_min = wav_min
        self.wav_max = wav_max
        self.emb_size = emb_size
        self.factor = factor
        self.base = base

        # Precomputamos denom[k] = base^(k/emb_size) y lo registramos como buffer
        k = torch.arange(emb_size).float()           # (emb_size,)
        denom = base ** (k / emb_size)               # (emb_size,)
        self.register_buffer('denom', denom)

    def forward(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wavelengths: Tensor (B, L) con longitudes de onda por posición.

        Returns:
            pe: Tensor (B, L, emb_size) con el positional encoding.
        """
        # Normalizamos a [0,1]: (B, L)
        lambda_hat = (wavelengths - self.wav_min) / (self.wav_max - self.wav_min)

        # Expandimos para calcular ángulos: (B, L, 1) / (emb_size,) -> (B, L, emb_size)
        angles = self.factor * lambda_hat.unsqueeze(-1) / self.denom.unsqueeze(0).unsqueeze(0)

        # Intercalamos sin y cos
        pe = torch.zeros_like(angles)
        pe[:,:,0::2] = torch.sin(angles[:,:,0::2])
        pe[:,:,1::2] = torch.cos(angles[:,:,1::2])

        return pe

    
import torch
import torch.nn as nn

class SpectralTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        emb_size: int,
        wav_min: float,
        wav_max: float,
        factor: float = 1000.0,
        base: float = 10000.0,
    ):
        super().__init__()
        self.intensity_proj = nn.Linear(in_channels, emb_size)
        self.pos_enc = SpectralPositionalEncoding(
            wav_min=wav_min,
            wav_max=wav_max,
            emb_size=emb_size,
            factor=factor,
            base=base,
        )
        # Embedding especial para padding
        self.pad_embedding = nn.Parameter(torch.zeros(1, emb_size))
        nn.init.trunc_normal_(self.pad_embedding, std=.02)

    def forward(
        self,
        intensities: torch.Tensor,      # (B, 1, L)
        wavelengths: torch.Tensor,      # (B, L)
        padding_mask: torch.Tensor        # (B, L), False=válido, True=padding
    ):
        B, _, L = intensities.shape

        # 1) Intensidad → emb_size
        x = intensities.permute(0, 2, 1)      # (B, L, C)
        x = self.intensity_proj(x)           # (B, L, D)

        # 2) Positional encoding
        pos = self.pos_enc(wavelengths)      # (B, L, D)

        # 3) Tokens “crudos”
        tokens = x + pos                     # (B, L, D)

        # 4) Reemplazar los tokens no válidos por pad_embedding
        #    padding_mask: True = dato real, False = padding
        pad_emb = self.pad_embedding.view(1, 1, -1)   # (1,1,D)
        pad_emb = pad_emb.expand(B, L, -1)            # (B,L,D)
        tokens = torch.where(
            padding_mask.unsqueeze(-1),
            pad_emb,    # si padding
            tokens    # si valido
        )

        return tokens, padding_mask

import torch
import torch.nn as nn

class SpectralTransformerEncoder(nn.Module):
    def __init__(
        self,
        emb_size: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_mean_pool: bool = False,
    ):
        """
        Encoder Transformer para espectros, que recibe ya sus tokens y padding_mask.

        Args:
            emb_size: dimensión interna del Transformer.
            nhead, num_layers, dim_feedforward, dropout: parámetros del encoder.
            use_mean_pool: si True, hará mean-pooling sobre tokens; 
                          si False, devuelve el token [CLS].
        """
        super().__init__()

        # Token [CLS]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        nn.init.trunc_normal_(self.cls_token, std=.02)

        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.use_mean_pool = use_mean_pool

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Tensor (B, N, emb_size) — embeddings + positional + pad_embedding.
            padding_mask: Tensor (B, N) boolean, False=dato real, True=padding.

        Returns:
            latents: Tensor (B, emb_size)
        """
        B, N, D = tokens.shape

        # Insertar CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)     # (B,1,D)
        x = torch.cat([cls_tokens, tokens], dim=1)       # (B, N+1, D)

        # Construir máscara para el Transformer: True=ignorar
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        src_key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, N+1)

        # Pasar por el encoder (requiere seq_len, B, D)
        x = x.transpose(0, 1)                            # (N+1, B, D)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0, 1)                     # (B, N+1, D)

        if self.use_mean_pool:
            # x_tokens: (B, N, D)
            x_tokens = x[:, 1:, :]                       

            # valid_mask: True = dato real, False = padding
            valid_mask = ~padding_mask                    # (B, N)
            valid_mask = valid_mask.unsqueeze(-1)         # (B, N, 1)

            # Suma solo donde valid_mask es True
            sum_tokens = (x_tokens * valid_mask).sum(dim=1)             # (B, D)
            count_tokens = valid_mask.sum(dim=1).clamp(min=1)           # (B, 1) evita división por 0
            latents = sum_tokens / count_tokens                         # (B, D)
        else:
            latents = x[:, 0, :]                                        # (B, D)


        return latents


class SpectralModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        wav_min: float,
        wav_max: float,
        emb_size: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_mean_pool: bool = False,
        output_dim: int = None
    ):
        """
        Wrapper que une tokenizer + encoder en un flujo único:
          (intensities, wavelengths, mask) -> latents
        """
        super().__init__()
        self.tokenizer = SpectralTokenizer(
            in_channels=in_channels,
            emb_size=emb_size,
            wav_min=wav_min,
            wav_max=wav_max,
            factor=1000.0,
            base=10000.0,
        )
        self.encoder = SpectralTransformerEncoder(
            emb_size=emb_size,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_mean_pool=use_mean_pool,
        )
        self.fc = nn.Linear(emb_size, output_dim) if output_dim is not None else None

    def forward(self, intensities: torch.Tensor, wavelengths: torch.Tensor, valid_mask: torch.Tensor):
        """
        intensities: (B, 1, L)
        wavelengths: (B, L)
        valid_mask: (B, L) True=dato real, False=padding
        """
        tokens, padding_mask = self.tokenizer(intensities, wavelengths, valid_mask)
        latents = self.encoder(tokens, padding_mask)
        if self.fc is not None:
            # Si se especifica una capa final, la aplicamos
            # latents: (B, emb_size)
            latents = self.fc(latents)
        return latents


class RegressionModel(torch.nn.Module):
    def __init__(self, base_model, emb_size, output_dim):
        super(RegressionModel, self).__init__()
        self.base_model = base_model
        self.cls = nn.Linear(emb_size, output_dim)

    def forward(self, x, wl, mask):
        feats = self.base_model(x, wl, mask)
        return self.cls(feats)  # (B, output_dim)