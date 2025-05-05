import glob
import re

import numpy as np
import pandas as pd
from myutils import *
from rdkit import Chem
from rdkit.Chem import AllChem


def load_data(args):
    """Load data based on the specified dataset."""
    if args.data == "gdsc1":
        print("load gdsc1")
        PATH = "gdsc1_data/"
        return _load_data(PATH)
    elif args.data == "gdsc2":
        print("load gdsc2")
        PATH = "gdsc2_data/"
        return _load_data(PATH)
    elif args.data == "ctrp":
        PATH = "ctrp_data/"
        return _load_data(PATH)
    elif args.data == "nci":
        print("load nci")
        PATH = "nci_data/"
        return _load_nci(PATH)
    else:
        raise NotImplementedError


def _get_base_data(PATH):
    """Load and prepare base data common to all datasets."""
    # Load original drug response data
    drugAct = pd.read_csv(PATH + "drugAct.csv", index_col=0)

    # Load and concatenate gene expression data
    gene_exp_files = sorted(
        glob.glob(PATH + "gene_exp_part*.csv.gz")
    )
    exprs = pd.concat([
        pd.read_csv(f, index_col=0).T for f in gene_exp_files
    ], axis=1)

    # Fill missing values with 0
    exprs = exprs.fillna(0)

    return drugAct, exprs


def _load_data(PATH):
    data_dir = dir_path(k=1) + PATH
    # 加载细胞系-药物矩阵

    drugAct, exprs = _get_base_data(data_dir)
    SMILES = pd.read_csv(data_dir + "drug2smiles.csv", index_col=0)

    cells = sorted(set(drugAct.columns) & set(exprs.index))
    drugs = sorted(set(drugAct.index) & set(SMILES['Drug']))
    exprs = exprs.loc[cells]
    exprs = np.array(exprs, dtype=np.float32)
    SMILES = SMILES[SMILES['Drug'].isin(drugs)]
    drugAct = drugAct.loc[sorted(drugs), cells]

    # Convert drug activity to binary response matrix
    res = np.array(drugAct, dtype=np.float32).T

    # 加载药物-指纹特征矩阵
    drug_feature = pd.read_csv(
        data_dir + "nih_drug_feature.csv", index_col=0, header=0
    ).loc[sorted(SMILES['Drug'])]
    drug_feature = np.array(drug_feature, dtype=np.float32)

    null_mask = (drugAct.isna()).astype(int).T
    null_mask = np.array(null_mask, dtype=np.float32)
    return res, drug_feature, exprs, null_mask


def _load_nci(PATH):
    data_dir = dir_path(k=1) + PATH
    # 加载细胞系-药物矩阵

    drugAct, exprs = _get_base_data(data_dir)
    mut = pd.read_csv(data_dir + "mut.csv", index_col=0).T
    cells = sorted(set(drugAct.columns) & set(exprs.index) & set(mut.index))

    # Filter and align data
    exprs = exprs.loc[cells]
    exprs = np.array(exprs, dtype=np.float32)

    drugAct = drugAct.loc[:, cells]

    # 加载药物-指纹特征矩阵
    drug_feature = pd.read_csv(data_dir + "nih_drug_feature.csv", index_col=0, header=0)

    drugs = sorted(set(drugAct.index) & set(drug_feature.index))

    drug_feature = drug_feature.loc[drugs]
    drug_feature = np.array(drug_feature, dtype=np.float32)

    # Convert drug activity to binary response matrix
    drugAct = drugAct.loc[drugs]
    res = drugAct
    res = np.array(res, dtype=np.float32).T

    null_mask = (drugAct.isna()).astype(int).T
    null_mask = np.array(null_mask, dtype=np.float32)
    return res, drug_feature, exprs, null_mask
