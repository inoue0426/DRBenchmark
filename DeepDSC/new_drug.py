import argparse
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from load_data import load_data
from sampler import NewSampler
from DeepDSC.DeepDSC import (
    AE,
    DF,
    GeneExpressionDataset,
    calculate_morgan_fingerprints,
    prepare_data,
    prepare_drug_data,
    prepare_train_val_test_data,
    train_autoencoder,
    train_df_model
)

data = "gdsc2"
PATH = "../gdsc2_data/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Args:
    def __init__(self):
        self.device = device  # cuda:number or cpu
        self.data = "gdsc2"  # Dataset{gdsc or ccle}

args = Args()
res, drug_feature, exprs, mut, cna, null_mask, pos_num = load_data(args)
cells = {i: j for i, j in enumerate(res.index)}
drugs = {i: j for i, j in enumerate(res.columns)}

cell_sum = np.sum(res, axis=1)
drug_sum = np.sum(res, axis=0)

target_dim = [
    0,  # Drug
    # 1  # Cell
]

def main(PATH, train, val):
    normalized_gene_exp_tensor, gene_exp = prepare_data(
        data1=PATH + "/gene_exp_part1.csv.gz", data2=PATH + "gene_exp_part2.csv.gz"
    )
    normalized_gene_exp_dataset = GeneExpressionDataset(normalized_gene_exp_tensor)
    normalized_gene_exp_dataloader = DataLoader(
        normalized_gene_exp_dataset, batch_size=10000, shuffle=True
    )

    # オートエンコーダーのトレーニング
    autoencoder = AE(normalized_gene_exp_tensor.shape[1]).to(device)
    train_autoencoder(autoencoder, normalized_gene_exp_dataloader)

    # 圧縮特徴の抽出
    compressed_features_tensor = autoencoder.encoder(normalized_gene_exp_tensor)
    compressed_features = pd.DataFrame(
        compressed_features_tensor.cpu().detach().numpy(), index=gene_exp.columns
    )

    # 薬物応答データの準備
    drug_response, nsc_sm = prepare_drug_data(is_nsc=False, is_gdsc=True, is_1=False)
    mfp = calculate_morgan_fingerprints(drug_response, nsc_sm)
    print(f"Morgan fingerprints shape: {mfp.shape}")

    train_labels = train[2]
    val_labels = val[2]
    train_data = train[[0, 1]]
    val_data = val[[0, 1]]
    val_data.columns = [0, 1]

    print(
        f"Training data size: {len(train_data)}, Validation data size: {len(val_data)}"
    )
    train_data, val_data = prepare_train_val_test_data(
        train_data, val_data, compressed_features, mfp
    )
    df_model = DF().to(device)
    val_labels, best_val_out = train_df_model(
        df_model,
        train_data,
        val_data,
        torch.tensor(train_labels).double().to(device),
        torch.tensor(val_labels).double().to(device),
    )
    print("DF model training completed.")
    return val_labels, best_val_out

def DeepDSC(res_mat, null_mask, target_dim, target_index, seed):
    sampler = NewSampler(res_mat, null_mask, target_dim, target_index, seed)

    train_data = pd.DataFrame(sampler.train_data, index=res.index, columns=res.columns)
    test_data = pd.DataFrame(sampler.test_data, index=res.index, columns=res.columns)

    train_mask = pd.DataFrame(sampler.train_mask, index=res.index, columns=res.columns)
    test_mask = pd.DataFrame(sampler.test_mask, index=res.index, columns=res.columns)

    train = pd.DataFrame(train_mask.values.nonzero()).T
    train[2] = train_data.values[train_mask.values.nonzero()].astype(int)

    test = pd.DataFrame(test_mask.values.nonzero()).T
    test[2] = test_data.values[test_mask.values.nonzero()].astype(int)

    train[0] = [cells[i] for i in train[0]]
    train[1] = [drugs[i] for i in train[1]]
    test[0] = [cells[i] for i in test[0]]
    test[1] = [drugs[i] for i in test[1]]

    val_labels, best_val_out = main(PATH, train, test)
    return val_labels, best_val_out

if __name__ == "__main__":
    n_kfold = 1
    true_data_s = pd.DataFrame()
    predict_data_s = pd.DataFrame()

    for dim in target_dim:
        for seed, target_index in enumerate(tqdm(np.arange(res.shape[dim]))):
            if dim:
                if drug_sum[target_index] < 10:
                    continue
            else:
                if cell_sum[target_index] < 10:
                    continue
            epochs = []
            for fold in range(n_kfold):
                val_labels, best_val_out = DeepDSC(
                    res.values, null_mask.T.values, dim, target_index, seed
                )

            true_data_s = pd.concat([true_data_s, pd.DataFrame(val_labels.cpu().numpy())], axis=1)
            predict_data_s = pd.concat(
                [predict_data_s, pd.DataFrame(best_val_out.cpu().numpy())], axis=1
            )

    true_data_s.to_csv(f"new_drug_true_{args.data}.csv")
    predict_data_s.to_csv(f"new_drug_pred_{args.data}.csv")
