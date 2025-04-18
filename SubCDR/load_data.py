import glob
import re

import numpy as np
import pandas as pd
from myutils import *


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
        return _load_data(PATH, is_ctrp=True)
    elif args.data == "nci":
        print("load nci")
        PATH = "../nci_data/"
        return _load_nci(PATH)
    else:
        raise NotImplementedError


def _get_base_data(PATH):
    """Load and prepare base data common to all datasets."""
    # Load original drug response data
    drugAct = pd.read_csv(PATH + "drugAct.csv", index_col=0)

    # Load and concatenate gene expression data
    exprs = pd.concat(
        [
            pd.read_csv(PATH + "gene_exp_part1.csv.gz", index_col=0),
            pd.read_csv(PATH + "gene_exp_part2.csv.gz", index_col=0),
        ]
    ).T.dropna()
    return drugAct, exprs


def _load_nci(data_dir):
    # 加载细胞系-药物矩阵

    drugAct, exprs = _get_base_data(data_dir)
    drugAct.columns = exprs.index
    cells = sorted(set(drugAct.columns) & set(exprs.index))

    # Load mechanism of action (moa) data
    moa = pd.read_csv("../data/nsc_cid_smiles_class_name.csv", index_col=0)

    # Filter drugs that have SMILES information
    drugAct = drugAct[drugAct.index.isin(moa.NSC)]

    # Load drug synonyms and filter based on availability in other datasets
    tmp = pd.read_csv("../data/drugSynonym.csv")
    tmp = tmp[
        (~tmp.nci60.isna() & ~tmp.ctrp.isna())
        | (~tmp.nci60.isna() & ~tmp.gdsc1.isna())
        | (~tmp.nci60.isna() & ~tmp.gdsc2.isna())
    ]
    tmp = [int(i) for i in set(tmp["nci60"].str.split("|").explode())]

    # Select drugs not classified as 'Other' in MOA and included in other datasets
    drugAct = drugAct.loc[
        sorted(
            set(drugAct.index)
            & (set(moa[moa["MECHANISM"] != "Other"]["NSC"]) | set(tmp))
        )
    ]

    # SMILES = pd.read_csv(data_dir + "drug2smiles.csv", index_col=0)
    exprs = exprs.loc[cells]
    drugAct = drugAct.loc[:, cells]

    # Convert drug activity to binary response matrix
    res = (drugAct > 0).astype(int).T

    pos_num = sp.coo_matrix(res).data.shape[0]
    null_mask = (drugAct.isna()).astype(int).T

    return res, exprs, null_mask, pos_num


def _load_data(PATH, is_ctrp=False):
    data_dir = dir_path(k=1) + PATH
    # 加载细胞系-药物矩阵

    drugAct, exprs = _get_base_data(data_dir)

    cells = sorted(set(drugAct.columns) & set(exprs.index))

    SMILES = pd.read_csv(data_dir + "drug2smiles.csv", index_col=0)
    exprs = exprs.loc[cells]
    drugAct = drugAct.loc[sorted(SMILES.drugs), cells]

    if is_ctrp:
        drugAct = drugAct.apply(lambda x: (x - np.nanmean(x)) / np.nanstd(x))

    # Convert drug activity to binary response matrix
    res = (drugAct > 0).astype(int).T

    pos_num = sp.coo_matrix(res).data.shape[0]

    null_mask = (drugAct.isna()).astype(int).T
    null_mask = np.array(null_mask, dtype=np.float32)
    return res, exprs, null_mask, pos_num
