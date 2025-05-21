import numpy as np
import scipy.sparse as sp
import torch
from myutils import mask, to_coo_matrix, to_tensor


class BalancedSampler(object):
    def __init__(self, edge_train, label_train, edge_test, label_test, adj_shape):
        """
        edge_train/test: np.ndarray of shape (N, 2)
        label_train/test: np.ndarray of shape (N,), values 0 or 1
        adj_shape: tuple, (num_rows, num_cols)
        """
        self.adj_shape = adj_shape

        # 分割されたデータからcoo_matrixを作成
        self.train_pos = self.edge_list_to_coo(edge_train[label_train == 1])
        self.train_neg = self.edge_list_to_coo(edge_train[label_train == 0])
        self.test_pos = self.edge_list_to_coo(edge_test[label_test == 1])
        self.test_neg = self.edge_list_to_coo(edge_test[label_test == 0])

        # トレーニング・テスト用マスクを作成
        self.train_mask = mask(self.train_pos, self.train_neg, dtype=int)
        self.test_mask = mask(self.test_pos, self.test_neg, dtype=bool)

        # モデル用入力（正例だけ）
        self.train_data = to_tensor(self.train_pos)
        self.test_data = to_tensor(self.test_pos)

    def edge_list_to_coo(self, edge_list):
        data = np.ones(edge_list.shape[0])
        return sp.coo_matrix((data, (edge_list[:, 0], edge_list[:, 1])), shape=self.adj_shape)



class NewSampler(object):
    def __init__(self, original_adj_mat, null_mask, target_dim, target_index, seed):
        super().__init__()
        self.seed = seed
        self.set_seed()
        self.adj_mat = original_adj_mat
        self.null_mask = null_mask
        self.dim = target_dim
        self.target_index = target_index
        self.train_data, self.test_data = self.sample_train_test_data()
        self.train_mask, self.test_mask = self.sample_train_test_mask()

    def set_seed(self):
        np.random.seed(self.seed)  # NumPyのシードを設定
        torch.manual_seed(self.seed)  # PyTorchのシードを設定

    def sample_target_test_index(self):
        if self.dim:
            target_pos_index = np.where(self.adj_mat[:, self.target_index] == 1)[0]
        else:
            target_pos_index = np.where(self.adj_mat[self.target_index, :] == 1)[0]
        return target_pos_index

    def sample_train_test_data(self):
        test_data = np.zeros(self.adj_mat.shape, dtype=np.float32)
        test_index = self.sample_target_test_index()
        if self.dim:
            test_data[test_index, self.target_index] = 1
        else:
            test_data[self.target_index, test_index] = 1
        train_data = self.adj_mat - test_data
        train_data = torch.from_numpy(train_data)
        test_data = torch.from_numpy(test_data)
        return train_data, test_data

    def sample_train_test_mask(self):
        test_index = self.sample_target_test_index()
        neg_value = np.ones(self.adj_mat.shape, dtype=np.float32)
        neg_value = neg_value - self.adj_mat - self.null_mask
        neg_test_mask = np.zeros(self.adj_mat.shape, dtype=np.float32)
        if self.dim:
            target_neg_index = np.where(neg_value[:, self.target_index] == 1)[0]
            if test_index.shape[0] < target_neg_index.shape[0]:
                target_neg_test_index = np.random.choice(
                    target_neg_index, test_index.shape[0], replace=False
                )
            else:
                target_neg_test_index = target_neg_index
            neg_test_mask[target_neg_test_index, self.target_index] = 1
            neg_value[:, self.target_index] = 0
        else:
            target_neg_index = np.where(neg_value[self.target_index, :] == 1)[0]
            if test_index.shape[0] < target_neg_index.shape[0]:
                target_neg_test_index = np.random.choice(
                    target_neg_index, test_index.shape[0], replace=False
                )
            else:
                target_neg_test_index = target_neg_index
            neg_test_mask[self.target_index, target_neg_test_index] = 1
            neg_value[self.target_index, :] = 0
        train_mask = (self.train_data.numpy() + neg_value).astype(bool)
        test_mask = (self.test_data.numpy() + neg_test_mask).astype(bool)
        train_mask = torch.from_numpy(train_mask)
        test_mask = torch.from_numpy(test_mask)
        return train_mask, test_mask
