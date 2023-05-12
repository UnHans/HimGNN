import numpy as np
import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_score, recall_score, f1_score
import dgl
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score, average_precision_score


def AUC(tesYAll, tesPredictAll):
    tesAUC = roc_auc_score(tesYAll, tesPredictAll)
    tesAUPR = average_precision_score(tesYAll, tesPredictAll)
    return tesAUC,tesAUPR
def RMSE(tesYAll,tesPredictAll):
    return mean_squared_error(tesYAll, tesPredictAll,squared=False),0


class GraphDataset_Classification(Dataset):
    def __init__(self, g_list, y_tensor):
        self.g_list = g_list
        self.y_tensor = y_tensor
        self.len = len(g_list)

    def __getitem__(self, idx):
        return self.g_list[idx], self.y_tensor[idx]

    def __len__(self):
        return self.len


class GraphDataLoader_Classification(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(GraphDataLoader_Classification, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        batched_gs = dgl.batch([item[0] for item in batch])
        batched_ys = torch.stack([item[1] for item in batch])
        #batched_ws = torch.stack([item[2] for item in batch])
        return (batched_gs, batched_ys)

class GraphDataset_Regression(Dataset):
    def __init__(self, g_list, y_tensor):
        self.g_list = g_list
        self.y_tensor = y_tensor
        self.len = len(g_list)

    def __getitem__(self, idx):
        return self.g_list[idx], self.y_tensor[idx]
    def __len__(self):
        return self.len

class GraphDataLoader_Regression(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(GraphDataLoader_Regression, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        batched_gs = dgl.batch([item[0] for item in batch])
        batched_ys = torch.stack([item[1] for item in batch])
        return (batched_gs, batched_ys)


