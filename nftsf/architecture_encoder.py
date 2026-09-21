"""
NFTSF architecture with a shared GRU context encoder.

Rationale
---------
The original `create_nfm` feeds the raw (standardized) x_past as `context`
to EVERY AutoregressiveRationalQuadraticSpline layer. Those K * len(hidden_layers_list)
spline conditioners each re-encode the past independently, and during
sampling this re-encoding happens D times sequentially (once per
autoregressive dimension step).

This module adds a single shared encoder:

    standardized x_past  ->  encoder g(.)  ->  h  ->  flow conditioner
                             (runs ONCE)          (h is reused by every
                                                   spline layer and every
                                                   autoregressive step)

The flow still has O(D) sequential autoregressive sampling steps — that
is structural to A-RQS — but the per-step cost drops because the
conditioner input is compact and already processed.
"""

from __future__ import annotations
from typing import Sequence, Literal

import torch
import torch.nn as nn
import normflows as nf


# -----------------------------------------------------------------------------
# Encoder variants
# -----------------------------------------------------------------------------
class _MLPEncoder(nn.Module):
    def __init__(self, n_past: int, past_dim: int, hidden: int, layers: int,
                 context_dim: int):
        super().__init__()
        sizes = [n_past * past_dim] + [hidden] * layers + [context_dim]
        mods = []
        for i in range(len(sizes) - 1):
            mods.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x_past: torch.Tensor) -> torch.Tensor:
        if x_past.dim() == 2:
            x_past = x_past.unsqueeze(-1)
        return self.net(x_past.flatten(1))


class _CNNEncoder(nn.Module):
    def __init__(self, past_dim: int, hidden: int, layers: int, context_dim: int):
        super().__init__()
        chans = [past_dim] + [hidden] * layers
        convs = []
        for i in range(len(chans) - 1):
            convs.append(nn.Conv1d(chans[i], chans[i + 1], kernel_size=3, padding=1))
            convs.append(nn.ReLU())
        self.convs = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden, context_dim)

    def forward(self, x_past: torch.Tensor) -> torch.Tensor:
        if x_past.dim() == 2:
            x_past = x_past.unsqueeze(-1)
        x = x_past.transpose(1, 2)
        x = self.convs(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class _GRUEncoder(nn.Module):
    """Bidirectional GRU + small MLP head. Default encoder."""
    def __init__(self, past_dim: int, hidden: int, layers: int, context_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=past_dim, hidden_size=hidden,
            num_layers=layers, batch_first=True, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, context_dim),
        )

    def forward(self, x_past: torch.Tensor) -> torch.Tensor:
        if x_past.dim() == 2:
            x_past = x_past.unsqueeze(-1)
        _, h = self.gru(x_past)
        h_fwd = h[-2]
        h_bwd = h[-1]
        h_final = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.head(h_final)


class _TransformerEncoder(nn.Module):
    def __init__(self, past_dim: int, hidden: int, layers: int, context_dim: int,
                 nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(past_dim, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=2 * hidden,
            batch_first=True, dropout=0.0,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(hidden, context_dim)

    def forward(self, x_past: torch.Tensor) -> torch.Tensor:
        if x_past.dim() == 2:
            x_past = x_past.unsqueeze(-1)
        x = self.proj(x_past)
        x = self.tr(x)
        x = x.mean(dim=1)
        return self.head(x)


_ENCODERS = {
    "mlp": _MLPEncoder,
    "cnn": _CNNEncoder,
    "gru": _GRUEncoder,
    "transformer": _TransformerEncoder,
}


# -----------------------------------------------------------------------------
# Wrapper module
# -----------------------------------------------------------------------------
class NFTSFEncoded(nn.Module):
    """Encoder + conditional RQS flow, trained end-to-end.

    Interface mirrors normflows.ConditionalNormalizingFlow but takes raw
    (standardized) x_past instead of a pre-computed context vector.
    """
    def __init__(self, encoder: nn.Module, flow: nf.ConditionalNormalizingFlow):
        super().__init__()
        self.encoder = encoder
        self.flow = flow

    def forward_kld(self, x_future: torch.Tensor, x_past: torch.Tensor
                    ) -> torch.Tensor:
        context = self.encoder(x_past)
        return self.flow.forward_kld(x_future, context=context)

    def log_prob(self, x_future: torch.Tensor, x_past: torch.Tensor
                 ) -> torch.Tensor:
        context = self.encoder(x_past)
        return self.flow.log_prob(x_future, context=context)

    @torch.no_grad()
    def sample(self, num_samples: int, x_past: torch.Tensor):
        context = self.encoder(x_past)
        return self.flow.sample(num_samples, context=context)


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def create_nfm_encoder(
    device: torch.device,
    n_past: int,
    n_future: int,
    past_dim: int = 1,
    encoder: Literal["gru", "mlp", "cnn", "transformer"] = "gru",
    encoder_hidden: int = 128,
    encoder_layers: int = 2,
    context_dim: int = 64,
    K: int = 6,
    hidden_units: int = 64,
    hidden_layers_list: Sequence[int] = (1, 2),
    tail_bound: float = 30.0,
) -> NFTSFEncoded:
    """Create an encoder + conditional A-RQS flow, trained jointly."""
    if encoder not in _ENCODERS:
        raise ValueError(f"encoder must be one of {list(_ENCODERS)}, got {encoder!r}")

    EncCls = _ENCODERS[encoder]
    if encoder == "mlp":
        enc = EncCls(n_past=n_past, past_dim=past_dim,
                     hidden=encoder_hidden, layers=encoder_layers,
                     context_dim=context_dim)
    else:
        enc = EncCls(past_dim=past_dim,
                     hidden=encoder_hidden, layers=encoder_layers,
                     context_dim=context_dim)

    prior = nf.distributions.DiagGaussian(n_future, trainable=False)
    flows = []
    for _ in range(K):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                n_future, hidden_layers, hidden_units,
                num_context_channels=context_dim, tail_bound=tail_bound,
            )
            for hidden_layers in hidden_layers_list
        ]
        flows += [nf.flows.LULinearPermute(n_future)]
    cflow = nf.ConditionalNormalizingFlow(q0=prior, flows=flows)

    return NFTSFEncoded(encoder=enc, flow=cflow).to(device)


# -----------------------------------------------------------------------------
# Presets for ablation
# -----------------------------------------------------------------------------
def preset_stage1(device, n_past: int, n_future: int, past_dim: int = 1, K: int = 6, encoder: str = "gru",
                  encoder_hidden: int = 128, encoder_layers: int = 2, hidden_units: int = 64, hidden_layers_list: tuple[int] = (1, 2)):
    """Stage 1: Encoder + FULL flow. Tests 'encoder alone was the missing piece'."""
    return create_nfm_encoder(
        device, n_past=n_past, n_future=n_future, past_dim=past_dim,
        encoder=encoder, encoder_hidden=encoder_hidden, encoder_layers=encoder_layers,
        context_dim=64,
        K=K, hidden_units=hidden_units, hidden_layers_list=hidden_layers_list,
    )


def preset_stage2(device, n_past: int, n_future: int, past_dim: int = 1, encoder: str = "gru",
        encoder_hidden : int = 128, encoder_layers: int = 2, K :int = 3, hidden_units: int = 32):
    """Stage 2: Encoder + LIGHTER flow. Tests 'encoder lets us shrink the flow'."""
    return create_nfm_encoder(
        device, n_past=n_past, n_future=n_future, past_dim=past_dim,
        encoder=encoder, encoder_hidden= encoder_hidden, encoder_layers=encoder_layers,
        context_dim=64,
        K=K, hidden_units=hidden_units, hidden_layers_list=(1,),
    )
