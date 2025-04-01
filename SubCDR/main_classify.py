import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import KFold
from torch_dataset import *

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from data_process import data_process
from MF import *
from models_classify import *
from sklearn.model_selection import KFold
from torch_dataset import *
from utils import *

# ---Binarization of the IC50 values
def getBinary(Tensors, thresh=0.5):
    ones = torch.ones_like(Tensors)
    zeros = torch.zeros_like(Tensors)
    return torch.where(Tensors < thresh, ones, zeros)

# ---data batchsize
def PairFeatures(
    pairs, drug_subfeat, cline_subfeat, drug_glofeat, cline_glofeat, drug_compo_elem, cline_compos_elem
):
    drug_subs = []
    cline_subs = []
    drug_glos = []
    cline_glos = []
    drug_compos = []
    cline_compos = []
    label = []
    for _, row in pairs.iterrows():
        cline_subs.append(cline_subfeat[str(row[0])])
        drug_subs.append(drug_subfeat[str(row[1])])
        cline_glos.append(np.array(cline_glofeat.loc[row[0]]))
        drug_glos.append(np.array(drug_glofeat.loc[row[1]]))
        drug_compos.append([row[1], drug_compo_elem[str(row[1])]])
        cline_compos.append([row[0], cline_compos_elem])
        label.append(row[2])
    return (
        drug_subs,
        cline_subs,
        drug_glos,
        cline_glos,
        drug_compos,
        cline_compos,
        label,
    )


def BatchGenerate(
    pairs, drug_subfeat, cline_subfeat, drug_glofeat, cline_glofeat, drug_compo_elem, cline_compos_elem, bs
):
    drug_subs, cline_subs, drug_glos, cline_glos, drug_compos, cline_compos, label = (
        PairFeatures(
            pairs,
            drug_subfeat,
            cline_subfeat,
            drug_glofeat,
            cline_glofeat,
            drug_compo_elem,
            cline_compos_elem
        )
    )
    ds_loader = Data.DataLoader(
        BatchData(drug_subs), batch_size=bs, shuffle=False, collate_fn=collate_seq
    )
    cs_loader = Data.DataLoader(BatchData(cline_subs), batch_size=bs, shuffle=False)
    glo_loader = Data.DataLoader(
        PairsData(drug_glos, cline_glos), batch_size=bs, shuffle=False
    )
    label = torch.from_numpy(np.array(label, dtype="float32")).to(device)
    label = Data.DataLoader(
        dataset=Data.TensorDataset(label), batch_size=bs, shuffle=False
    )
    return ds_loader, cs_loader, glo_loader, drug_compos, cline_compos, label


def train(model, optimizer, myloss, drug_loader_train, cline_loader_train, glo_loader_train, label_train):
    loss_train = 0
    Y_true, Y_pred = [], []
    for batch, (drug, cline, glo_feat, label) in enumerate(
        zip(drug_loader_train, cline_loader_train, glo_loader_train, label_train)
    ):
        label = getBinary(label[0])
        pred, _ = model(drug.to(device), cline.to(device), glo_feat.to(device))
        optimizer.zero_grad()
        loss = myloss(pred, label)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        Y_true += label.cpu().detach().numpy().tolist()
        Y_pred += pred.cpu().detach().numpy().tolist()
    auc, aupr = classification_metric(Y_true, Y_pred)
    print("train-loss=", loss_train / len(drug_loader_train.dataset))
    print("train-AUC:" + str(round(auc, 4)) + " train-AUPR:" + str(round(aupr, 4)))


def test(model, myloss, drug_loader_test, cline_loader_test, glo_loader_test, label_test):
    loss_test = 0
    Y_true, Y_pred = [], []
    all_maps = []
    model.eval()
    with torch.no_grad():
        for batch, (drug, cline, glo_feat, label) in enumerate(
            zip(drug_loader_test, cline_loader_test, glo_loader_test, label_test)
        ):
            label = getBinary(label[0])
            pred, maps = model(drug.to(device), cline.to(device), glo_feat.to(device))
            loss = myloss(pred, label)
            loss_test += loss.item()
            Y_true += label.cpu().detach().numpy().tolist()
            Y_pred += pred.cpu().detach().numpy().tolist()
    print("test-loss=", loss.item() / len(drug_loader_test.dataset))
    auc, aupr = classification_metric(Y_true, Y_pred)
    return auc, aupr, Y_true, Y_pred
