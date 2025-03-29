import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(
    data1="../nci_data/gene_exp_part1.csv.gz", data2="../nci_data/gene_exp_part2.csv.gz"
):
    gene_exp = pd.concat(
        [
            pd.read_csv(data1, index_col=0).T,
            pd.read_csv(data2, index_col=0).T,
        ],
        axis=1,
    ).T
    gene_exp = gene_exp.fillna(0)
    normalized_gene_exp = gene_exp.subtract(gene_exp.min(1), axis="rows")
    normalized_gene_exp = normalized_gene_exp.div(gene_exp.max(1) + 1e-8, axis="rows")
    return torch.tensor(normalized_gene_exp.values).t().to(device), gene_exp


def prepare_train_val_test_data(
    train_data, val_data, test_data, compressed_features, mfp
):
    def get_data(X):
        return pd.concat(
            [
                mfp.loc[X.values[:, 0]].reset_index(drop=True),
                compressed_features.loc[X.values[:, 1]].reset_index(drop=True),
            ],
            axis=1,
        )

    train_data = get_data(train_data)
    val_data = get_data(val_data)
    test_data = get_data(test_data)

    train_tensor = torch.tensor(train_data.values).to(device)
    val_tensor = torch.tensor(val_data.values).to(device)
    test_tensor = torch.tensor(test_data.values).to(device)

    return (train_tensor, val_tensor, test_tensor)


def calculate_morgan_fingerprints(drug_response, nsc_sm):
    SMILES = [nsc_sm[i] for i in drug_response.columns]
    params = Chem.SmilesParserParams()
    params.useChirality = True
    params.radicalElectrons = 2
    params.removeHs = False
    params.replacements = {}

    mfp = []

    for i in SMILES:
        mol = Chem.MolFromSmiles(i, params=params)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
        mfp.append(np.array(fp))

    mfp = pd.DataFrame(mfp, dtype=np.float32, index=drug_response.columns)

    return mfp


def prepare_drug_data(is_nsc=True, is_gdsc=True, is_1=True):
    if is_nsc:
        convert = pd.read_csv("nsc_cid_smiles_class_name.csv", index_col=0)
        nsc_class = dict(convert[["NSC", "MECHANISM"]].values)
        nsc_sm = dict(convert[["NSC", "SMILES"]].values)
        drug_response = pd.read_csv("../nci_data/drugAct.csv", index_col=0)
        drug_response = drug_response.loc[
            sorted(list(set(nsc_class.keys()) & set(drug_response.index)))
        ].fillna(0)
        drug_response = drug_response.loc[
            [nsc_class[i] != "Other" for i in drug_response.index]
        ]
        drug_response = drug_response.apply(
            lambda x: (x - np.nanmean(x)) / np.nanstd(x)
        ).T
        return drug_response, nsc_sm
    else:
        if is_gdsc:
            convert = dict(
                pd.concat(
                    [
                        pd.read_csv("gdsc1_drug2smiles.csv", index_col=0),
                        pd.read_csv("gdsc2_drug2smiles.csv", index_col=0),
                    ]
                )[["drugs", "SMILES"]]
                .drop_duplicates()
                .values
            )
            if is_1:
                drug_response = pd.read_csv("../gdsc1_data/drugAct.csv", index_col=0)

            else:
                drug_response = pd.read_csv("../gdsc2_data/drugAct.csv", index_col=0)
        else:
            convert = dict(
                pd.concat([pd.read_csv("ctrp_drug2smiles.csv", index_col=0)])[
                    ["drugs", "SMILES"]
                ]
                .drop_duplicates()
                .values
            )

            drug_response = pd.read_csv("../ctrp_data/drugAct.csv", index_col=0)

        drug_response = drug_response.loc[
            sorted(set(convert.keys()) & set(drug_response.index))
        ]
        drug_response = drug_response.fillna(0).T
        return drug_response, convert


class GeneExpressionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class AE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        torch.manual_seed(0)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2000).double(),
            nn.SELU(),
            nn.Linear(2000, 1000).double(),
            nn.SELU(),
            nn.Linear(1000, 500).double(),
        )

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[2].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)

        self.decoder = nn.Sequential(
            nn.Linear(500, 1000).double(),
            nn.SELU(),
            nn.Linear(1000, 2000).double(),
            nn.SELU(),
            nn.Linear(2000, input_dim).double(),
            nn.Sigmoid(),
        )

        torch.nn.init.xavier_uniform_(self.decoder[0].weight)
        torch.nn.init.xavier_uniform_(self.decoder[2].weight)
        torch.nn.init.xavier_uniform_(self.decoder[4].weight)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DF(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)

        self.layers = nn.Sequential(
            nn.Linear(756, 1000).double(),
            nn.ELU(),
            nn.Linear(1000, 800).double(),
            nn.ELU(),
            nn.Linear(800, 500).double(),
            nn.ELU(),
            nn.Linear(500, 100).double(),
            nn.ELU(),
            nn.Linear(100, 1).double(),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(0.1)

        torch.nn.init.kaiming_uniform_(self.layers[0].weight)
        torch.nn.init.kaiming_uniform_(self.layers[2].weight)
        torch.nn.init.kaiming_uniform_(self.layers[4].weight)
        torch.nn.init.kaiming_uniform_(self.layers[6].weight)
        torch.nn.init.kaiming_uniform_(self.layers[8].weight)

    def forward(self, x):
        return self.layers(x)


def train_autoencoder(autoencoder, dataloader, num_epochs=800):
    optimizer = torch.optim.Adamax(autoencoder.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    l1_lambda = 0.1
    for epoch in range(num_epochs):
        for data in dataloader:
            optimizer.zero_grad()
            train_out = autoencoder(data)
            train_loss = criterion(train_out, data)
            l1_norm = sum(p.abs().sum() for p in autoencoder.encoder.parameters())
            train_loss = train_loss + l1_lambda * l1_norm
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1)
            optimizer.step()
        # if (epoch + 1) % 10 == 0:
        #     print(f"Epoch {epoch+1} \t\t Training Loss: {train_loss.item()}")


def train_df_model(
    model, train_tensor, val_tensor, train_labels, val_labels, num_epochs=100
):
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.0004)
    criterion = nn.BCELoss()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        train_out = model(train_tensor)
        train_loss = criterion(train_out.squeeze(), train_labels)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_out = model(val_tensor)
            val_loss = criterion(val_out.squeeze(), val_labels)
            val_out = val_out.squeeze()
            val_acc = torch.sum((val_out >= 0.5) == val_labels) / len(val_labels)

        # print(f"Epoch {epoch+1} Loss: {train_loss.item()} Val Loss: {val_loss.item()}")
        # print(f"Accuracy: {val_acc}")


def evaluate_model(model, test_data, test_labels):
    model.eval()
    val_out = model(test_data)
    return val_out.squeeze().detach().cpu().numpy()


def print_binary_classification_metrics(y_true, y_prob):
    y_true = y_true.detach().cpu().numpy()

    # Calculate standard metrics using 0.5 threshold
    y_pred = (y_prob >= 0.5).astype(int)

    metrics_data = {
        "Accuracy": [accuracy_score(y_true, y_pred)],
        "Precision": [precision_score(y_true, y_pred)],
        "Recall": [recall_score(y_true, y_pred)],
        "F1 Score": [f1_score(y_true, y_pred)],
        "AUROC": [roc_auc_score(y_true, y_prob)],
        "AUPR": [average_precision_score(y_true, y_prob)],
    }

    return pd.DataFrame(metrics_data)
