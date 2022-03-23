
import pandas as pd
import torch
from torch import nn

from src.data_prep.torch_datasets import Dataset
from src.models.train_torch_net import train_torch_net
from configs.model_configs import BDNN_Zhang


if __name__ == '__main__':

    survival_pseudo_data = pd.read_csv("../data/surv_pseudo_data.csv")
    X = survival_pseudo_data[["X1","X2","X3","tpseudo"]].values
    y = survival_pseudo_data[["pseudo"]].values

    dataset = Dataset(X, y)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    nn_zhang = BDNN_Zhang()

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    train_torch_net(dataset,neural_net=nn_zhang,loss_function=loss_function,n_epochs=2)
