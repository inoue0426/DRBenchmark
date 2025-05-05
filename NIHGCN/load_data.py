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
    elif args.data == "gdsc2":
        print("load gdsc2")
        PATH = "gdsc2_data/"
    elif args.data == "ctrp":
        PATH = "ctrp_data/"
    elif args.data == "nci":
        print("load nci")
        PATH = "nci_data/"
    else:
        raise NotImplementedError

    return _load_common(PATH, is_nci=(args.data == "nci"))


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


def _load_common(PATH, is_nci=False):
    data_dir = dir_path(k=1) + PATH
    drugAct, exprs = _get_base_data(data_dir)

    if is_nci:
        mut = pd.read_csv(data_dir + "mut.csv", index_col=0).T
        cells = sorted(set(drugAct.columns) & set(exprs.index) & set(mut.index))
    else:
        SMILES = pd.read_csv(data_dir + "drug2smiles.csv", index_col=0)
        cells = sorted(set(drugAct.columns) & set(exprs.index))

    # Filter and align data
    exprs = exprs.loc[cells]
    exprs = np.array(exprs, dtype=np.float32)

    # 加载药物-指纹特征矩阵
    drug_feature = pd.read_csv(data_dir + "nih_drug_feature.csv", index_col=0, header=0)

    if is_nci:
        drugs = sorted(set(drugAct.index) & set(drug_feature.index))
    else:
        drugs = sorted(set(drugAct.index) & set(SMILES['Drug']))
        SMILES = SMILES[SMILES['Drug'].isin(drugs)]

    drug_feature = drug_feature.loc[drugs]
    drug_feature = np.array(drug_feature, dtype=np.float32)

    # Convert drug activity to binary response matrix
    drugAct = drugAct.loc[drugs, cells]
    res = drugAct.T

    null_mask = (drugAct.isna()).astype(int).T
    null_mask = np.array(null_mask, dtype=np.float32)

    return res, drug_feature, exprs, null_mask
