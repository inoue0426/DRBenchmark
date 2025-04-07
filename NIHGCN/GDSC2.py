import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from load_data import load_data
from model import Optimizer, nihgcn
from myutils import translate_result, roc_auc
from sampler import RandomSampler

class Args:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.data = "gdsc2"  # Dataset{gdsc or ccle}
        self.lr = 0.001  # Learning rate
        self.wd = 1e-5  # Weight decay for L2 normalization
        self.layer_size = [1024, 1024]  # Output sizes of every layer
        self.alpha = 0.25  # Scale for balance GCN and NI
        self.gamma = 8  # Scale for sigmoid
        self.epochs = 1000  # Number of epochs

args = Args()

# Load data
res, drug_finger, exprs, null_mask, pos_num = load_data(args)

# K-fold cross validation
k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=42)

true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()

for train_index, test_index in kfold.split(np.arange(pos_num)):
    # Initialize sampler and model
    sampler = RandomSampler(res, train_index, test_index, null_mask)
    model = nihgcn(
        adj_mat=sampler.train_data,
        cell_exprs=exprs,
        drug_finger=drug_finger,
        layer_size=args.layer_size,
        alpha=args.alpha,
        gamma=args.gamma,
        device=args.device
    ).to(args.device)

    # Initialize optimizer
    opt = Optimizer(
        model,
        sampler.train_data,
        sampler.test_data,
        sampler.test_mask,
        sampler.train_mask,
        roc_auc,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        device=args.device
    ).to(args.device)

    # Train and get predictions
    true_data, predict_data = opt()
    true_datas = pd.concat([true_datas, translate_result(true_data)], ignore_index=True)
    predict_datas = pd.concat([predict_datas, translate_result(predict_data)], ignore_index=True)

# Save results
true_datas.to_csv("true_gdsc2.csv")
predict_datas.to_csv("pred_gdsc2.csv")
