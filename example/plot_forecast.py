#!/usr/bin/env python3
"""Quick visual check for a forecast .npz produced by forecast.py."""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast_npz", required=True)
    parser.add_argument("--out", default="example/forecasts/single_well_25_25.png")
    parser.add_argument("--n_traj", type=int, default=4)
    args = parser.parse_args()

    data = np.load(args.forecast_npz)
    ground_truth = data["ground_truth"]
    contexts = data["contexts"]
    samples = data["samples"]
    ci90_lower, ci90_upper = data["ci90_lower"], data["ci90_upper"]
    ci50_lower, ci50_upper = data["ci50_lower"], data["ci50_upper"]
    median = np.median(samples, axis=2)

    n_traj = min(args.n_traj, ground_truth.shape[0])
    n_past = contexts.shape[1]
    n_future = ground_truth.shape[1]
    past_steps = np.arange(n_past)
    future_steps = np.arange(n_past, n_past + n_future)

    fig, axes = plt.subplots(1, n_traj, figsize=(4 * n_traj, 4), sharey=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.plot(past_steps, contexts[i], "k-", label="Context")
        ax.plot(future_steps, ground_truth[i], "k--", label="Truth")
        ax.plot(future_steps, median[i], color="tab:orange", label="Pred median")
        ax.fill_between(future_steps, ci90_lower[i], ci90_upper[i],
                         color="tab:orange", alpha=0.2, label="90% CI")
        ax.fill_between(future_steps, ci50_lower[i], ci50_upper[i],
                         color="tab:orange", alpha=0.35, label="50% CI")
        ax.axvline(n_past, color="gray", linestyle=":")
        ax.set_title(f"Trajectory {i}")
        ax.set_xlabel("Step")
    axes[0].set_ylabel("Position, x")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
