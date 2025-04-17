import argparse

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from load_data import load_data
from model import Optimizer, nihgcn
from myutils import *
from sampler import NewSampler
from sklearn.model_selection import KFold
from tqdm import tqdm


class Args:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = "ctrp"  # Dataset{gdsc or ccle}
        self.lr = 0.001  # the learning rate
        self.wd = 1e-5  # the weight decay for l2 normalizaton
        self.layer_size = [1024, 1024]  # Output sizes of every layer
        self.alpha = 0.25  # the scale for balance gcn and ni
        self.gamma = 8  # the scale for sigmod
        self.epochs = 1000  # the epochs for model


args = Args()

# Load data
res, drug_finger, exprs, null_mask, pos_num = load_data(args)
cell_sum = np.sum(res, axis=1)
drug_sum = np.sum(res, axis=0)

target_dim = [
    0,  # Cell
    # 1  # Drug
]


def nihgcn_new(
    cell_exprs,
    drug_finger,
    res_mat,
    null_mask,
    target_dim,
    target_index,
    evaluate_fun,
    args,
    seed,
):
    sampler = NewSampler(res_mat, null_mask, target_dim, target_index, seed)

    val_labels = sampler.test_data[sampler.test_mask]

    if len(np.unique(val_labels)) < 2:
        print(f"Target {target_index} skipped: Validation set has only one class.")
        return None, None

    model = nihgcn(
        sampler.train_data,
        cell_exprs=cell_exprs,
        drug_finger=drug_finger,
        layer_size=args.layer_size,
        alpha=args.alpha,
        gamma=args.gamma,
        device=args.device,
    )
    opt = Optimizer(
        model,
        sampler.train_data,
        sampler.test_data,
        sampler.test_mask,
        sampler.train_mask,
        evaluate_fun,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        device=args.device,
    )
    true_data, predict_data = opt()
    return true_data, predict_data


n_kfold = 1
n_jobs = 2  # Number of parallel jobs


def process_iteration(dim, target_index, seed, args):
    """Function to encapsulate each iteration"""
    if dim:
        if drug_sum[target_index] < 10:
            return None
    else:
        if cell_sum[target_index] < 10:
            return None

    fold_results = []
    for fold in range(n_kfold):
        true_data, predict_data = nihgcn_new(
            cell_exprs=exprs,
            drug_finger=drug_finger,
            res_mat=res,
            null_mask=null_mask,
            target_dim=dim,
            target_index=target_index,
            evaluate_fun=roc_auc,
            args=args,
            seed=seed,
        )
        fold_results.append((true_data, predict_data))

    return fold_results


# Execute parallel processing
true_data_s = pd.DataFrame()
predict_data_s = pd.DataFrame()

for dim in target_dim:
    # Generate all tasks beforehand
    tasks = [
        (dim, target_index, seed, args)
        for seed, target_index in enumerate(np.arange(res.shape[dim]))
    ]

    # Execute in parallel with progress bar
    results = Parallel(n_jobs=n_jobs, verbose=0, prefer="threads")(
        delayed(process_iteration)(*task)
        for task in tqdm(tasks, desc=f"Processing dim {dim}")
    )

    # Combine results
    for fold_results in results:
        if fold_results is None:
            continue
        for true_data, predict_data in fold_results:
            true_data_s = pd.concat(
                [true_data_s, translate_result(true_data)],
                ignore_index=True,
                copy=False,  # Save memory
            )
            predict_data_s = pd.concat(
                [predict_data_s, translate_result(predict_data)],
                ignore_index=True,
                copy=False,
            )

# Save results
true_data_s.to_csv(f"new_cell_true_{args.data}.csv")
predict_data_s.to_csv(f"new_cell_pred_{args.data}.csv")
