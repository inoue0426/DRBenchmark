import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .DeepDSC import (AE, DF, GeneExpressionDataset,
                      calculate_morgan_fingerprints, prepare_data,
                      prepare_drug_data, prepare_train_val_test_data,
                      train_autoencoder, train_df_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def main(PATH, train, val):
    print("Loading gene expression data...")
    normalized_gene_exp_tensor, gene_exp = prepare_data(
        data1=PATH + "/gene_exp_part1.csv.gz", data2=PATH + "gene_exp_part2.csv.gz"
    )
    normalized_gene_exp_dataset = GeneExpressionDataset(normalized_gene_exp_tensor)
    normalized_gene_exp_dataloader = DataLoader(
        normalized_gene_exp_dataset, batch_size=10000, shuffle=True
    )

    # オートエンコーダーのトレーニング
    print("Training autoencoder...")
    autoencoder = AE(normalized_gene_exp_tensor.shape[1]).to(device)
    train_autoencoder(autoencoder, normalized_gene_exp_dataloader)
    print("Autoencoder training completed.")

    # 圧縮特徴の抽出
    print("Extracting compressed features...")
    compressed_features_tensor = autoencoder.encoder(normalized_gene_exp_tensor)
    compressed_features = pd.DataFrame(
        compressed_features_tensor.cpu().detach().numpy(), index=gene_exp.columns
    )
    print(f"Compressed features shape: {compressed_features.shape}")

    # 薬物応答データの準備
    print("Preparing drug response data...")
    drug_response, nsc_sm = prepare_drug_data(is_nsc=True)
    print(f"Drug response data shape: {drug_response.shape}")

    print("Calculating Morgan fingerprints...")
    mfp = calculate_morgan_fingerprints(drug_response, nsc_sm)
    print(f"Morgan fingerprints shape: {mfp.shape}")

    train_labels = train[2]
    val_labels = val[2]
    train_data = train[[0, 1]]
    val_data = val[[0, 1]]
    print(
        f"Training data size: {len(train_data)}, Validation data size: {len(val_data)}"
    )

    # トレーニング、検証、テストデータの準備
    print("Preparing training, validation, and test data...")
    train_data, val_data = prepare_train_val_test_data(
        train_data, val_data, compressed_features, mfp
    )
    print("Data preparation completed.")

    # DFモデルのトレーニング
    print("Training DF model...")
    df_model = DF().to(device)
    train_df_model(df_model, train_data, val_data, train_labels, val_labels)
    print("DF model training completed.")
