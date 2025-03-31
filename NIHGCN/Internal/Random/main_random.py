# coding: utf-8
import argparse

from load_data import load_data
from model import Optimizer, nihgcn
from myutils import *
from sampler import RandomSampler
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser(description="Run NIHGCN")
parser.add_argument("-device", type=str, default="cuda:0", help="cuda:number or cpu")
parser.add_argument("-data", type=str, default="gdsc", help="Dataset{gdsc or ccle}")
parser.add_argument("--lr", type=float, default=0.001, help="the learning rate")
parser.add_argument(
    "--wd", type=float, default=1e-5, help="the weight decay for l2 normalizaton"
)
parser.add_argument(
    "--layer_size", nargs="?", default=[1024, 1024], help="Output sizes of every layer"
)
parser.add_argument(
    "--alpha", type=float, default=0.25, help="the scale for balance gcn and ni"
)
parser.add_argument("--gamma", type=float, default=8, help="the scale for sigmod")
parser.add_argument("--epochs", type=float, default=1000, help="the epochs for model")
args = parser.parse_args()


# load data
res, drug_finger, exprs, null_mask, pos_num, args = load_data(args)

true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()
k = 5
n_kfolds = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=42)

res = pd.DataFrame()
for train_index, test_index in kfold.split(np.arange(pos_num)):
    sampler = RandomSampler(res, train_index, test_index, null_mask)
    model = nihgcn(
        adj_mat=sampler.train_data,
        cell_exprs=exprs,
        drug_finger=drug_finger,
        layer_size=args.layer_size,
        alpha=args.alpha,
        gamma=args.gamma,
        device=args.device,
    ).to(args.device)
    opt = Optimizer(
        model,
        sampler.train_data,
        sampler.test_data,
        sampler.test_mask,
        sampler.train_mask,
        roc_auc,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        device=args.device,
    ).to(args.device)
    true_data, predict_data = opt()
    true_datas = pd.concat([true_datas, pd.DataFrame(true_data)], ignore_index=True)
    predict_datas = pd.concat(
        [predict_datas, pd.DataFrame(predict_data)], ignore_index=True
    )
