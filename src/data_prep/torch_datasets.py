
import torch
from sklearn.preprocessing import StandardScaler

class Dataset(torch.utils.data.Dataset):

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