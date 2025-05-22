#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dataclasses import dataclass
from joblib import Parallel, delayed

from model import GModel
from myutils import roc_auc, translate_result, filter_target
from load_data import load_data
from optimizer import Optimizer
from sampler import NewSampler


@dataclass
class Args:
    device: str
    data: str
    n_jobs: int
    lr: float = 5e-4
    epochs: int = 1000


def run_single_model(exprs, cna, mut, drug_feature, res_mat, null_mask, target_dim, target_index, args, seed):
    sampler = NewSampler(res_mat, null_mask, target_dim, target_index)
    val_labels = sampler.test_data[sampler.test_mask]

    model = GModel(
        adj_mat=sampler.train_data.float(),
        gene=exprs,
        cna=cna,
        mutation=mut,
        sigma=2,
        k=11,
        iterates=3,
        feature_drug=drug_feature,
        n_hid1=192,
        n_hid2=36,
        alpha=5.74,
        device=args.device,
    )
    opt = Optimizer(
        model=model,
        train_data=sampler.train_data,
        test_data=sampler.test_data,
        test_mask=sampler.test_mask,
        train_mask=sampler.train_mask,
        evaluate_fun=roc_auc,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    ).to(args.device)
    _, true_data, predict_data = opt()
    return true_data, predict_data


def process_target(seed, target_index, exprs, cna, mut, drug_feature, res, null_mask, target_dim, args):
    try:
        return run_single_model(
            exprs=exprs,
            cna=cna,
            mut=mut,
            drug_feature=drug_feature,
            res_mat=res.values,
            null_mask=null_mask,
            target_dim=target_dim,
            target_index=target_index,
            args=args,
            seed=seed,
        )
    except Exception as e:
        print(f"❌ Failed at target {target_index}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ctrp", help="Dataset name (e.g., ctrp)")
    parser.add_argument("--target", type=str, default="cell", choices=["cell", "drug"], help="Target dimension")
    parser.add_argument("--n-jobs", type=int, default=20, help="Number of parallel jobs")
    args_cli = parser.parse_args()

    args = Args(
        device="cuda" if torch.cuda.is_available() else "cpu",
        data=args_cli.data,
        n_jobs=args_cli.n_jobs,
    )

    target_option = args_cli.target
    target_dim = 0 if target_option == "cell" else 1

    print(f"📦 Loading data for dataset: {args.data}")
    res, drug_feature, exprs, mut, cna, null_mask = load_data(args)

    samples = res.shape[target_dim]
    cell_sum = np.sum(res.values, axis=1)
    drug_sum = np.sum(res.values, axis=0)

    true_data_s = pd.DataFrame()
    predict_data_s = pd.DataFrame()
    skipped_targets = []
    passed_targets = []

    for target_index in range(samples):
        label_vec = res.iloc[target_index] if target_dim == 0 else res.iloc[:, target_index]
        passed, reason, pos, neg, total = filter_target(label_vec)

        if passed:
            passed_targets.append(target_index)
        else:
            skipped_targets.append((target_index, reason, pos, neg, total))

    print(f"\n🚫 Skipped Targets: {len(skipped_targets)}")
    for idx, reason, pos, neg, total in skipped_targets:
        print(f"Target {idx}: skipped because {reason} (total={total}, pos={pos}, neg={neg})")

    print(f"\n🚀 Running MOFGCN on {len(passed_targets)} targets with {args.n_jobs} jobs...")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_target)(seed, target_index, exprs, cna, mut, drug_feature, res, null_mask, target_dim, args)
        for seed, target_index in enumerate(tqdm(passed_targets, desc=f"MOFGCN ({args.data} - {target_option})"))
    )

    for r in results:
        if r is not None:
            true_data, pred_data = r
            true_data_s = pd.concat([true_data_s, translate_result(true_data)], ignore_index=True)
            predict_data_s = pd.concat([predict_data_s, translate_result(pred_data)], ignore_index=True)

    # Save
    true_path = f"mofgcn_true_{args.data}_{target_option}.csv"
    pred_path = f"mofgcn_pred_{args.data}_{target_option}.csv"
    true_data_s.to_csv(true_path, index=False)
    predict_data_s.to_csv(pred_path, index=False)

    print(f"\n✅ Done. Results saved to:\n  - {true_path}\n  - {pred_path}")


if __name__ == "__main__":
    main()
