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
        return _load_data(PATH, is_ctrp=True)
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
    exprs = pd.concat(
        [
            pd.read_csv(PATH + "gene_exp_part1.csv.gz", index_col=0),
            pd.read_csv(PATH + "gene_exp_part2.csv.gz", index_col=0),
        ]
    ).T.dropna()
    mut = pd.read_csv(PATH + "mut.csv", index_col=0).T.dropna()
    if "nci_data/" in PATH:
        cna = pd.read_csv(PATH + "cop.csv", index_col=0).fillna(0).T
    else:
        cna = (
            pd.concat(
                [
                    pd.read_csv(PATH + "cop_part1.csv.gz", index_col=0),
                    pd.read_csv(PATH + "cop_part2.csv.gz", index_col=0),
                ]
            )
            .fillna(0)
            .T
        )

    return drugAct, exprs, mut, cna


def _load_data(PATH, is_ctrp=False):
    data_dir = dir_path(k=1) + PATH
    # 加载细胞系-药物矩阵

    drugAct, exprs, mut, cna = _get_base_data(data_dir)

    cells = sorted(
        set(drugAct.columns) & set(exprs.index) & set(mut.index) & set(cna.index)
    )

    SMILES = pd.read_csv(data_dir + "drug2smiles.csv", index_col=0)
    exprs = exprs.loc[cells]
    drugAct = drugAct.loc[sorted(SMILES.drugs), cells]
    exprs = np.array(exprs, dtype=np.float32)
    mut = mut.loc[cells]
    mut = np.array(mut, dtype=np.float32)
    cna = cna.loc[cells]
    cna = np.array(cna, dtype=np.float32)

    if is_ctrp:
        drugAct = drugAct.apply(lambda x: (x - np.nanmean(x)) / np.nanstd(x))

    # Convert drug activity to binary response matrix
    res = (drugAct > 0).astype(int)
    res = np.array(res, dtype=np.float32).T

    pos_num = sp.coo_matrix(res).data.shape[0]

    # 加载药物-指纹特征矩阵
    drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
    drug_feature = np.array(drug_feature, dtype=np.float32)

    null_mask = (drugAct.isna()).astype(int).T
    null_mask = np.array(null_mask, dtype=np.float32)
    return res, drug_feature, exprs, mut, cna, null_mask, pos_num


def _load_nci(PATH):
    data_dir = dir_path(k=1) + PATH
    # 加载细胞系-药物矩阵

    drugAct, exprs, mut, cna = _get_base_data(data_dir)
    drugAct.columns = exprs.index
    cells = sorted(
        set(drugAct.columns) & set(exprs.index) & set(mut.index) & set(cna.index)
    )

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
    exprs = np.array(exprs, dtype=np.float32)
    mut = mut.loc[cells]
    mut = np.array(mut, dtype=np.float32)
    cna = cna.loc[cells]
    cna = np.array(cna, dtype=np.float32)

    # Convert drug activity to binary response matrix
    res = (drugAct > 0).astype(int)
    res = np.array(res, dtype=np.float32).T

    pos_num = sp.coo_matrix(res).data.shape[0]
    # 加载药物-指纹特征矩阵
    drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
    drug_feature = np.array(drug_feature, dtype=np.float32)

    null_mask = (drugAct.isna()).astype(int).T
    null_mask = np.array(null_mask, dtype=np.float32)
    return res, drug_feature, exprs, mut, cna, null_mask, pos_num
