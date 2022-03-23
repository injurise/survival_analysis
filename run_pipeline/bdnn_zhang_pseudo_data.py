
import pandas as pd
from torch import nn

import bnn
from src.data_prep.torch_datasets import Dataset
from src.models.train_bnn_vi import train_bayesenized_nn
from configs.model_configs import BDNN_Zhang


if __name__ == '__main__':

    survival_pseudo_data = pd.read_csv("../data/surv_pseudo_data.csv")
    X = survival_pseudo_data[["X1","X2","X3","tpseudo"]].values
    y = survival_pseudo_data[["pseudo"]].values

    dataset = Dataset(X, y)
    nn_zhang = BDNN_Zhang()
    bnn.bayesianize_(nn_zhang, inference="inducing", inducing_rows=64, inducing_cols=64)


    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    train_bayesenized_nn(dataset,dataset.__len__(),neural_net=nn_zhang,loss_function=loss_function,n_epochs=2)
