#!/usr/bin/env python3
"""
Forecast with a trained NFTSF model and save a .npz bundle of samples.

Usage:
    python forecast.py \
        --model_path example/checkpoints/single_well/model.pth \
        --config     example/checkpoints/single_well/config.json \
        --data_path  example/data/single_well_test.npz \
        --out        example/forecasts/single_well_25_25.npz \
        --n_samples 1000 --seed 42
"""

import os
import json
import argparse
import time as timelib

import numpy as np
import torch
from tqdm import tqdm

from nftsf import build_nftsf

torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description="NFTSF forecast script")
    p.add_argument("--model_path", required=True, help="Path to .pth checkpoint")
    p.add_argument("--config", required=True, help="Path to training config.json")
    p.add_argument("--data_path", required=True, help="Path to test .npz file (with 'positions')")
    p.add_argument("--out", "-o", required=True, help="Output .npz path")
    p.add_argument("--n_samples", type=int, default=500, help="Number of ensemble samples")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Number of trajectories per forward pass")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_test_split", type=int, default=None,
                   help="Override train_test_split if not present in data")
    p.add_argument("--test_size", type=int, default=None,
                   help="Number of trajectories to evaluate (default: all)")
    return p.parse_args()


def setup_device(device_arg):
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    print(f"Device: {device}")
    return device


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def build_model(device, cfg, n_past, n_future):
    """Construct the model architecture matching the training config."""
    flow_blocks = cfg.get("flow_blocks", 6 if n_past < 100 else 3)
    hidden_units = cfg.get("hidden_units", 64)
    hidden_layers = cfg.get("hidden_layers", "1,2")
    tail_bound = cfg.get("tail_bound", 30.0)
    hidden_layers_list = tuple(int(x) for x in str(hidden_layers).split(","))
    encoder_type = cfg.get("encoder_type", "cnn")

    model_type = "basic_nf" if n_past < 100 else f"encoder_nf ({encoder_type})"
    print(f"Building model for n_past={n_past}, n_future={n_future}: {model_type}")
    return build_nftsf(
        device, n_past=n_past, n_future=n_future,
        flow_blocks=flow_blocks, hidden_units=hidden_units,
        hidden_layers_list=hidden_layers_list, tail_bound=tail_bound,
        encoder_type=encoder_type,
    )


def load_test_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    positions = data["positions"].astype(np.float32)
    tts = int(data["train_test_split"]) if "train_test_split" in data else None
    print(f"Test data: {positions.shape} (N_traj, T)")
    return positions, tts


def run_forecast_batched(model, positions, context_length, prediction_length,
                         n_samples, train_test_split, test_size,
                         batch_size, device):
    test_size = min(test_size or positions.shape[0], positions.shape[0])
    positions = positions[:test_size]
    N = positions.shape[0]

    ctx_np = positions[:, train_test_split - context_length: train_test_split]
    gt_np = positions[:, train_test_split: train_test_split + prediction_length]

    ctx_tensor = torch.tensor(ctx_np, dtype=torch.float32, device=device)

    samples_out = np.zeros((N, prediction_length, n_samples), dtype=np.float32)
    time_elapsed = 0.0

    n_batches = (N + batch_size - 1) // batch_size
    for b in tqdm(range(n_batches), desc="Forecasting"):
        b_start = b * batch_size
        b_end = min(b_start + batch_size, N)
        B = b_end - b_start

        ctx_b = ctx_tensor[b_start:b_end]
        ctx_tiled = ctx_b.repeat_interleave(n_samples, dim=0)

        with torch.no_grad():
            start_t = timelib.time()
            out = model.sample(B * n_samples, ctx_tiled)
            samp = out[0] if isinstance(out, tuple) else out
            time_elapsed += timelib.time() - start_t

        samp_np = samp.cpu().numpy().reshape(B, n_samples, prediction_length)
        samples_out[b_start:b_end] = samp_np.transpose(0, 2, 1)

    print(f"Inference time: {time_elapsed:.2f}s  "
          f"({time_elapsed/N*1000:.1f} ms/traj, "
          f"{time_elapsed/(N*n_samples)*1000:.3f} ms/sample)")

    return samples_out, gt_np, ctx_np, time_elapsed


def main():
    args = parse_args()
    set_seed(args.seed)
    device = setup_device(args.device)

    with open(args.config) as f:
        cfg = json.load(f)

    n_past = cfg.get("n_past", 100)
    n_future = cfg.get("n_future", 100)
    print(f"n_past={n_past}, n_future={n_future}")

    positions, tts_npz = load_test_data(args.data_path)
    N, T = positions.shape

    if args.train_test_split is not None:
        train_test_split = args.train_test_split
    elif tts_npz is not None:
        train_test_split = tts_npz
    else:
        train_test_split = n_past
    print(f"train_test_split = {train_test_split}")

    model = build_model(device, cfg, n_past, n_future)

    state = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"Loaded model from {args.model_path}")

    test_size = args.test_size if args.test_size is not None else N
    samples, ground_truth, contexts, time_elapsed = run_forecast_batched(
        model=model,
        positions=positions,
        context_length=n_past,
        prediction_length=n_future,
        n_samples=args.n_samples,
        train_test_split=train_test_split,
        test_size=test_size,
        batch_size=args.batch_size,
        device=device,
    )

    ci90_lower = np.percentile(samples, 5, axis=2)
    ci90_upper = np.percentile(samples, 95, axis=2)
    ci50_lower = np.percentile(samples, 25, axis=2)
    ci50_upper = np.percentile(samples, 75, axis=2)

    out_path = args.out
    if not out_path.endswith(".npz"):
        out_path += ".npz"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    np.savez_compressed(
        out_path,
        samples=samples,
        ground_truth=ground_truth,
        contexts=contexts,
        ci90_lower=ci90_lower,
        ci90_upper=ci90_upper,
        ci50_lower=ci50_lower,
        ci50_upper=ci50_upper,
        train_test_split=train_test_split,
        prediction_length=n_future,
        context_length=n_past,
        num_of_samples=args.n_samples,
        time_elapsed=time_elapsed,
    )
    print(f"\nSaved: {out_path}")
    print(f"samples : {samples.shape} (N, H, S)")
    print(f"ground_truth : {ground_truth.shape} (N, H)")
    print(f"contexts : {contexts.shape} (N, L)")

    assert not np.isnan(samples).any(), "NaN in samples"
    assert not np.isinf(samples).any(), "Inf in samples"
    assert np.all(ci90_lower <= ci90_upper), "CI90 ordering violated"
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
