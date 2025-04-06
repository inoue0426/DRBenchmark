import os

import hickle as hkl
import pandas as pd


def data_load(data="nci"):
    if data == "nci":
        return _data_load()
    elif data == "gdsc1":
        return _data_load("../../gdsc1_data/")
    elif data == "gdsc2":
        return _data_load("../../gdsc2_data/")
    elif data == "ctrp":
        return _data_load("../../ctrp_data/")
    else:
        NotImplementedError


def _data_load(PATH="../../nci_data/"):
    Drug_feature_file = PATH + "drug_graph_feat"
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        feat_mat, adj_list, degree_list = hkl.load("%s/%s" % (Drug_feature_file, each))
        drug_feature[each.split(".")[0]] = [feat_mat, adj_list, degree_list]

    exp = pd.concat(
        [
            pd.read_csv(PATH + "gene_exp_part1.csv.gz", index_col=0).T,
            pd.read_csv(PATH + "gene_exp_part2.csv.gz", index_col=0).T,
        ],
        axis=1,
    ).dropna()

    mutation = pd.read_csv(PATH + "mut.csv", index_col=0).T
    mutation = mutation.dropna()
    # mutation.index = exp.index

    if os.path.exists(PATH + "met.csv"):
        methylation = pd.read_csv(PATH + "met.csv", index_col=0).T.fillna(0)
    else:
        methylation = pd.concat(
            [
                pd.read_csv(PATH + "met_part1.csv.gz", index_col=0).T,
                pd.read_csv(PATH + "met_part2.csv.gz", index_col=0).T,
            ],
            axis=1,
        )

    cells = sorted(set(exp.index) & set(mutation.index) & set(methylation.index))

    exp = exp.loc[cells]
    mutation = mutation.loc[cells]
    methylation = methylation.loc[cells]
    methylation = methylation.fillna(0)

    nb_celllines = exp.shape[0]
    nb_drugs = len(drug_feature)

    print("Drug feature dimension:", len(drug_feature))
    print("Gene expression dimension:", exp.shape)
    print("Mutation dimension:", mutation.shape)
    print("Methylation dimension:", methylation.shape)
    print("Number of cell lines:", nb_celllines)
    print("Number of drugs:", nb_drugs)

    return (drug_feature, exp, mutation, methylation, nb_celllines, nb_drugs)
