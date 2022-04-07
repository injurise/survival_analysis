
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np



class standard_dataset(Dataset):

    def __init__(self, X, y, scale_data=False):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class surv_dataset(Dataset):

    def __init__(self, X, tb, e):
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.tb = torch.from_numpy(tb).float()
        self.e = torch.from_numpy(e).float()

    def __getitem__(self, i):
        return self.X[i] , {'tb':self.tb[i], 'e':self.e[i]}

    def __len__(self):
        return len(self.X)

class cpath_dataset(Dataset):

    def __init__(self, X,clinical_vars, tb, e):
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.clinical_vars = torch.from_numpy(clinical_vars).float()
        self.tb = torch.from_numpy(tb).float()
        self.e = torch.from_numpy(e).float()

    def __getitem__(self, i):
        return {"X":self.X[i], 'clinical_vars':self.clinical_vars[i]} , \
               {'tb':self.tb[i], 'e':self.e[i]}

    def __len__(self):
        return len(self.X)