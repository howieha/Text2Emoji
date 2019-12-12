import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import numpy as np


class EmojiDataset(Dataset):
    """Emoji Classification Dataset"""

    def __init__(self, np_X, csr_Y):
        self.X = np_X
        self.Y = csr_Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        tempX = self.X[idx, :]
        tempY = self.Y.getrow(idx).tocoo()
        tempX = torch.LongTensor(tempX)
        tempY = torch.sparse.FloatTensor(torch.FloatTensor([tempY.row.tolist(), tempY.col.tolist()]),
                                         torch.FloatTensor(tempY.data))#.astype(np.float)))
        sample = {'X': tempX, 'Y': tempY}
        return sample








