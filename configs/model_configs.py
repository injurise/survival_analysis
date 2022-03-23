
from torch import nn

class BDNN_Zhang(nn.Module):
  '''
    DNN from the paper
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(4, 8),
      nn.Tanh(),
      nn.Linear(8, 4),
      nn.Tanh(),
      nn.Linear(4, 1),
      nn.Sigmoid(),
      nn.Linear(1, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)