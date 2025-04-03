import argparse
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from molFrags import *
from sklearn.model_selection import KFold
from torch_dataset import *

warnings.filterwarnings("ignore")

from data_process import data_process
from load_data import load_data
from main_classify import *
from MF import *
from models_classify import *
from sampler import Sampler
from utils import *

tmp = "nci"


class Args:
    def __init__(self):
        self.lr = 0.0001  # Learning rate
        self.bs = 5000  # Batch size
        self.ep = 100  # Number of epochs
        self.o = f"./{tmp}_output_dir/"  # Output directory
        self.data = tmp


# Create args object
args = Args()

os.makedirs(args.o, exist_ok=True)
# Data processing
start_time = time.time()
seed = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

drug_subfeat, cline_subfeat, drug_dim, drug_compo_elem, cline_compos_elem = (
    data_process(args)
)


def prepare_matrix_factorization(train_set):
    """Matrix factorization preparation and execution"""
    print("Building known matrix...")
    CDR_known = train_set.set_index(["Cline", "Drug"]).unstack("Cline")
    CDR_known.columns = CDR_known.columns.droplevel()

    CDR_matrix = np.array(CDR_known)
    CDR_mask = 1 - np.float32(np.isnan(CDR_matrix))
    CDR_matrix[np.isnan(CDR_matrix)] = 0

    print("Performing matrix factorization...")
    drug_glofeat, cline_glofeat = svt_solve(A=CDR_matrix, mask=CDR_mask)
    drug_glofeat = pd.DataFrame(drug_glofeat, index=list(CDR_known.index))
    cline_glofeat = pd.DataFrame(cline_glofeat, index=list(CDR_known.columns))

    return drug_glofeat, cline_glofeat


def prepare_data_loaders(
    train_set, validation_set, drug_glofeat, cline_glofeat, batch_sizes
):
    """Prepare train and validation data loaders"""
    print("Preparing data loaders...")
    # Shuffle data
    train_set = train_set.sample(frac=1, random_state=seed)
    validation_set = validation_set.sample(frac=1, random_state=seed)

    # Create train loaders
    print("Preparing train data loaders...")
    drug_loader_train, cline_loader_train, glo_loader_train, _, _, label_train = (
        BatchGenerate(
            train_set,
            drug_subfeat,
            cline_subfeat,
            drug_glofeat,
            cline_glofeat,
            drug_compo_elem,
            cline_compos_elem,
            bs=batch_sizes,
        )
    )

    # Create validation loaders
    print("Preparing test data loaders...")
    (
        drug_loader_valid,
        cline_loader_valid,
        glo_loader_valid,
        dc_valid,
        cc_valid,
        label_valid,
    ) = BatchGenerate(
        validation_set,
        drug_subfeat,
        cline_subfeat,
        drug_glofeat,
        cline_glofeat,
        drug_compo_elem,
        cline_compos_elem,
        bs=batch_sizes,
    )

    return (
        drug_loader_train,
        cline_loader_train,
        glo_loader_train,
        label_train,
        drug_loader_valid,
        cline_loader_valid,
        glo_loader_valid,
        label_valid,
        dc_valid,
        cc_valid,
    )


def setup_model(drug_dim, glo_dim, device, args):
    """Initialize model and optimizer"""
    print("Initializing model and optimizer...")
    model = SubCDR(
        SubEncoder(in_drug=drug_dim, in_cline=8, out=82),
        GraphEncoder(in_channels=32, out_channels=16),
        GloEncoder(in_channels=glo_dim, out_channels=128),
        Decoder(in_channels=160),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    myloss = torch.nn.BCELoss()

    return model, optimizer, myloss


def train_epoch(model, loaders, optimizer, myloss):
    """Train for one epoch"""
    model.train()
    drug_loader_train, cline_loader_train, glo_loader_train, label_train = loaders
    train(
        model,
        optimizer,
        myloss,
        drug_loader_train,
        cline_loader_train,
        glo_loader_train,
        label_train,
    )


def validate(model, loaders, myloss):
    """Perform validation"""
    drug_loader_valid, cline_loader_valid, glo_loader_valid, label_valid = loaders
    auc, aupr, y_true, y_pred = test(
        model,
        myloss,
        drug_loader_valid,
        cline_loader_valid,
        glo_loader_valid,
        label_valid,
    )
    return auc, aupr, y_true, y_pred


def train_and_validate_fold(train_set, validation_set, args):
    """Main training and validation function for one fold"""
    print(
        f"Train set size: {len(train_set)}, Validation set size: {len(validation_set)}"
    )

    # Matrix factorization
    drug_glofeat, cline_glofeat = prepare_matrix_factorization(train_set)
    glo_dim = 2 * drug_glofeat.shape[1]

    # Prepare data
    batch_sizes = args.bs
    loaders = prepare_data_loaders(
        train_set, validation_set, drug_glofeat, cline_glofeat, batch_sizes
    )
    train_loaders = loaders[:4]
    valid_loaders = loaders[4:8]

    # Setup model
    model, optimizer, myloss = setup_model(drug_dim, glo_dim, device, args)

    # Training loop
    print("\nStarting training...")
    start = time.time()
    best_auc = 0
    best_aupr = 0

    for epoch in range(args.ep):
        print(f"\nEpoch {epoch + 1}/{args.ep}")

        # Train
        print("Training...")
        train_epoch(model, train_loaders, optimizer, myloss)

        # Validate
        print("Validating...")
        auc, aupr, y_true, y_pred = validate(model, valid_loaders, myloss)
        print(f"Test AUC: {auc:.4f}, Test AUPR: {aupr:.4f}")

        # Save best model
        if auc > best_auc:
            print("New best model found! Saving...")
            best_auc = auc
            best_aupr = aupr
            best_pred = y_pred
            torch.save(model.state_dict(), f"{args.o}classification_model.pkl")

    training_time = time.time() - start
    print(f"Best AUC: {best_auc:.4f}, Best AUPR: {best_aupr:.4f}")
    return best_pred, y_true


def run_cross_validation(args):
    """Run k-fold cross validation"""
    print("\nStarting 5-fold cross validation...")
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    best_preds = []
    y_trues = []

    for train_index, test_index in kfold.split(np.arange(pos_num)):
        sampler = Sampler(res, train_index, test_index, null_mask)

        train_data = pd.DataFrame(
            sampler.train_data, index=res.index, columns=res.columns
        )
        test_data = pd.DataFrame(
            sampler.test_data, index=res.index, columns=res.columns
        )

        train_mask = pd.DataFrame(
            sampler.train_mask, index=res.index, columns=res.columns
        )
        test_mask = pd.DataFrame(
            sampler.test_mask, index=res.index, columns=res.columns
        )

        train = pd.DataFrame(train_mask.values.nonzero()).T
        train[2] = train_data.values[train_mask.values.nonzero()].astype(int)

        test = pd.DataFrame(test_mask.values.nonzero()).T
        test[2] = test_data.values[test_mask.values.nonzero()].astype(int)

        train[0] = [cells[i] for i in train[0]]
        train[1] = [drugs[i] for i in train[1]]

        test[0] = [cells[i] for i in test[0]]
        test[1] = [drugs[i] for i in test[1]]

        cols = ["Cline", "Drug", "Values"]

        train.columns = cols
        test.columns = cols

        train_set = train
        validation_set = test
        best_pred, y_true = train_and_validate_fold(train_set, validation_set, args)
        best_preds.append(best_pred)
        y_trues.append(y_true)

    pd.DataFrame(best_preds).to_csv(f"pred_{tmp}.csv")
    pd.DataFrame(y_trues).to_csv(f"true_{tmp}.csv")

    return pd.DataFrame(best_preds), pd.DataFrame(y_trues)


res, exprs, null_mask, pos_num = load_data(args)
cells = {i: j for i, j in enumerate(res.index)}
drugs = {i: j for i, j in enumerate(res.columns)}
k = 5
best, true = run_cross_validation(args)
