#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from dataclasses import dataclass

from model import Optimizer, nihgcn
from myutils import *
from load_data import load_data
from sampler import NewSampler


@dataclass
class Args:
    device: str
    data: str
    n_jobs: int 
    lr: float = 0.001
    wd: float = 1e-5
    layer_size: tuple = (1024, 1024)
    alpha: float = 0.25
    gamma: float = 8
    epochs: int = 1000


def run_single_model(cell_exprs, drug_finger, res, null_mask, target_dim, target_index, args, evaluate_fun):
    sampler = NewSampler(res.values, null_mask, target_dim, target_index)
    val_labels = sampler.test_data[sampler.test_mask]

    if len(np.unique(val_labels)) < 2:
        print(f"Target {target_index} skipped: Validation set has only one class.")
        return None

    model = nihgcn(
        adj_mat=sampler.train_data,
        cell_exprs=cell_exprs,
        drug_finger=drug_finger,
        layer_size=args.layer_size,
        alpha=args.alpha,
        gamma=args.gamma,
        device=args.device,
    )
    opt = Optimizer(
        model=model,
        train_data=sampler.train_data,
        test_data=sampler.test_data,
        test_mask=sampler.test_mask,
        train_mask=sampler.train_mask,
        evaluate_fun=evaluate_fun,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        device=args.device,
    )
    true_data, pred_data = opt()
    return true_data.detach().cpu().numpy(), pred_data.detach().cpu().numpy()


def process_iteration(i, target_dim, args, res, exprs, drug_finger, null_mask, cell_sum, drug_sum):
    if target_dim == 0 and cell_sum.iloc[i] < 10:
        return None
    if target_dim == 1 and drug_sum.iloc[i] < 10:
        return None

    return run_single_model(
        cell_exprs=exprs,
        drug_finger=drug_finger,
        res=res,
        null_mask=null_mask,
        target_dim=target_dim,
        target_index=i,
        args=args,
        evaluate_fun=roc_auc,
    )


def save_results(results, true_path, pred_path):
    trues, preds = zip(*[r for r in results if r is not None])
    pd.DataFrame(trues).to_csv(true_path, index=False)
    pd.DataFrame(preds).to_csv(pred_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="nci", help="Dataset name")
    parser.add_argument("--target", type=str, choices=["cell", "drug"], default="cell", help="Prediction target")
    parser.add_argument("--n-jobs", type=int, default=30, help="Number of parallel jobs")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    target_dim = 0 if cli_args.target == "cell" else 1
    args = Args(
        device="cuda" if torch.cuda.is_available() else "cpu",
        data=cli_args.data,
        n_jobs=cli_args.n_jobs,
    )

    res, drug_finger, exprs, null_mask = load_data(args)
    exprs = exprs.copy()
    samples = res.shape[target_dim]
    cell_sum = np.sum(res, axis=1)
    drug_sum = np.sum(res, axis=0)

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_iteration)(i, target_dim, args, res, exprs, drug_finger, null_mask, cell_sum, drug_sum)
        for i in tqdm(range(samples), desc=f"NIHGCN ({cli_args.data} - {cli_args.target})")
    )

    save_results(results, f"true_{cli_args.data}_{cli_args.target}.csv", f"pred_{cli_args.data}_{cli_args.target}.csv")
    print("Done!")


if __name__ == "__main__":
    main()
