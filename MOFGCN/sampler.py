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



class NewSampler:
    def __init__(self, original_adj_mat, null_mask, target_dim, target_index):
        self.adj_mat = original_adj_mat
        self.null_mask = null_mask
        self.dim = target_dim
        self.target_index = target_index
        self.train_data, self.test_data = self._sample_train_test_data()
        self.train_mask, self.test_mask = self._sample_train_test_mask()

    def _get_target_indices(self, matrix, value):
        if self.dim == 0:  # dim=0 → 行（cell）
            return np.where(matrix[self.target_index, :] == value)[0]
        return np.where(matrix[:, self.target_index] == value)[0]

    def _sample_target_test_index(self):
        return self._get_target_indices(self.adj_mat, 1)

    def _sample_train_test_data(self):
        test_data = np.zeros(self.adj_mat.shape, dtype=np.float32)
        test_index = self._sample_target_test_index()

        if self.dim == 0:  # Cell（行）をターゲット
            test_data[self.target_index, test_index] = 1
        else:  # Drug（列）をターゲット
            test_data[test_index, self.target_index] = 1

        train_data = self.adj_mat - test_data
        # Null Maskを適用
        train_data[self.null_mask == 1] = 0
        return torch.from_numpy(train_data), torch.from_numpy(test_data)

    def _sample_train_test_mask(self):
        neg_value = np.ones(self.adj_mat.shape, dtype=np.float32) - self.adj_mat - self.null_mask
        neg_test_mask = np.zeros(self.adj_mat.shape, dtype=np.float32)

        target_neg_index = self._get_target_indices(neg_value, 1)

        if self.dim == 0:  # Cell（行）をターゲット
            neg_test_mask[self.target_index, target_neg_index] = 1
            neg_value[self.target_index, :] = 0
        else:  # Drug（列）をターゲット
            neg_test_mask[target_neg_index, self.target_index] = 1
            neg_value[:, self.target_index] = 0

        train_mask = (self.train_data.numpy() + neg_value).astype(bool)
        test_mask = (self.test_data.numpy() + neg_test_mask).astype(bool)
        # Null Maskを適用
        train_mask[self.null_mask == 1] = False
        return torch.from_numpy(train_mask), torch.from_numpy(test_mask)
