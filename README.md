# NFTSF

NFTSF (Normalizing-Flow Time Series Forecasting) is a conditional normalizing
flow that forecasts the **distribution** of future states of a stochastic
trajectory given its past — rather than a single point forecast. It is the
model proposed in *Forecast Distributions, Not Trajectories: Rethinking Time
Series Forecasting Benchmarks for Stochastic Dynamics*.

This repository packages **just the model** (architecture + train/forecast
scripts + one runnable example) for easy reuse. For the full benchmark suite
— baselines (CSDI, TSDiff, ARIMA), all four datasets, and figure
reproduction — see the main repository:
[PessoaP/TSF_for_stochastic_dynamics](https://github.com/PessoaP/TSF_for_stochastic_dynamics).

## How it works

NFTSF learns a conditional probability density `p(x_future | x_past)` by transforming a
diagonal Gaussian prior through a stack of autoregressive rational-quadratic
spline (A-RQS) flow layers interleaved with LU-decomposed linear
permutations, conditioned on the past.

```python
from nftsf import build_nftsf

model = build_nftsf(device, n_past=25, n_future=25)
```

`build_nftsf` builds a model sized for whatever `n_past` / `n_future` you
give it. `train.py` and `forecast.py` both use this same function.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and (optionally) a CUDA-capable GPU.

## Layout

```
NFTSF/
├── nftsf/                          ← the model
│   ├── architecture.py             ← base flow (create_nfm)
│   ├── architecture_encoder.py     ← flow with an encoder for longer inputs (create_nfm_encoder)
│   └── model.py                    ← build_nftsf(): sized from n_past/n_future
├── train.py                   ← train a model on a single dataset
├── forecast.py                ← sample forecasts from a trained model
└── example/
    ├── data/                  ← single_well_{train,test}.npz (from Zenodo)
    └── plot_forecast.py       ← visualize a forecast .npz
```

## Quickstart: `single_well` example

`example/data/` already contains a ready-to-use dataset: 1D position
trajectories from an overdamped single-well SDE (1000 training / 250 test
trajectories, length 1000 each), taken from the Zenodo archive of the main
repository. Each `.npz` has a `positions` array of shape `(N_traj, T)`.

**1. Train** (past = future = 25 steps):

```bash
python train.py \
    --data_path example/data/single_well_train.npz \
    --output_dir example/checkpoints/single_well \
    --n_past 25 --n_future 25 \
    --epochs 1000 --batch_size 4096 --val_fraction 0.1 --device auto
```

This saves `model.pth` and `config.json` to `example/checkpoints/single_well/`.
`--epochs 1000` matches the paper's setting (at ~9s/epoch on a single GPU
with the included data, expect well under an hour thanks to early stopping,
enabled by default with `--early_stopping_patience 50`); for a quick smoke
test that the pipeline works, use e.g. `--epochs 50` (a few minutes) instead.

**2. Forecast** on held-out trajectories:

```bash
python forecast.py \
    --model_path example/checkpoints/single_well/model.pth \
    --config     example/checkpoints/single_well/config.json \
    --data_path  example/data/single_well_test.npz \
    --out        example/forecasts/single_well_25_25.npz \
    --n_samples 500 --seed 42
```

This draws an ensemble of samples per trajectory and saves them, together
with the ground truth, the past window, and 50%/90% credible intervals, to
a single `.npz` bundle.

**3. Plot**:

```bash
python example/plot_forecast.py \
    --forecast_npz example/forecasts/single_well_25_25.npz \
    --out example/forecasts/single_well_25_25.png
```

This whole pipeline (train → forecast → plot) has been run end-to-end
against the included data to confirm it works out of the box.

## Citation

If you use NFTSF, please cite the paper:

> *Forecast Distributions, Not Trajectories: Rethinking Time Series
> Forecasting Benchmarks for Stochastic Dynamics.*

## License

CC BY 4.0 — see [LICENSE](LICENSE).
