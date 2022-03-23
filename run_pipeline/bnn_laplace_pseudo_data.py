
from torch import nn
import pandas as pd


import numpy as np
import torch

from src.data_prep.torch_datasets import Dataset
from src.models.train_torch_net import train_torch_net
from configs.model_configs import BDNN_Zhang
from laplace import Laplace

if __name__ == '__main__':
    survival_pseudo_data = pd.read_csv("data/surv_pseudo_data.csv")
    X = survival_pseudo_data[["X1", "X2", "X3", "tpseudo"]].values
    y = survival_pseudo_data[["pseudo"]].values

    dataset = Dataset(X, y)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    nn_zhang = BDNN_Zhang()

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    map_nn,batch_losses = train_torch_net(dataset, neural_net=nn_zhang, loss_function=loss_function, n_epochs=2)

    la = Laplace(map_nn, 'regression', subset_of_weights='all', hessian_structure='full')
    la.fit(trainloader)
    log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    for i in range(2):
        hyper_optimizer.zero_grad()
        neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()

    f_mu, f_var = la(dataset.X)

    f_mu = f_mu.squeeze().detach().cpu().numpy()
    f_sigma = f_var.squeeze().sqrt().cpu().numpy()
    pred_std = np.sqrt(f_sigma ** 2 + la.sigma_noise.item() ** 2)

