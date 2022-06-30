import torch
import pandas as pd
from src.data_prep.torch_datasets import cpath_dataset


def load_cpath_data(cuda):
    pathway_mask = pd.read_csv("../data/pathway_mask.csv", index_col=0).values
    pathway_mask = torch.from_numpy(pathway_mask).type(torch.FloatTensor)
    if cuda:
        pathway_mask = pathway_mask.to(device='cuda')

    train_data = pd.read_csv("../data/train.csv")
    X_train_np = train_data.drop(["SAMPLE_ID", "OS_MONTHS", "OS_EVENT", "AGE"], axis=1).values
    tb_train = train_data.loc[:, ["OS_MONTHS"]].values
    e_train = train_data.loc[:, ["OS_EVENT"]].values
    clinical_vars_train = train_data.loc[:, ["AGE"]].values

    val_data = pd.read_csv("../data/validation.csv")
    X_val_np = val_data.drop(["SAMPLE_ID", "OS_MONTHS", "OS_EVENT", "AGE"], axis=1).values
    tb_val = val_data.loc[:, ["OS_MONTHS"]].values
    e_val = val_data.loc[:, ["OS_EVENT"]].values
    clinical_vars_val = val_data.loc[:, ["AGE"]].values

    test_data = pd.read_csv("../data/test.csv")
    X_test_np = test_data.drop(["SAMPLE_ID", "OS_MONTHS", "OS_EVENT", "AGE"], axis=1).values
    tb_test = test_data.loc[:, ["OS_MONTHS"]].values
    e_test = test_data.loc[:, ["OS_EVENT"]].values
    clinical_vars_test = test_data.loc[:, ["AGE"]].values

    cpath_train_dataset = cpath_dataset(X_train_np,
                                        clinical_vars_train,
                                        tb_train,
                                        e_train)

    cpath_val_dataset = cpath_dataset(X_val_np,
                                      clinical_vars_val,
                                      tb_val,
                                      e_val)
    cpath_test_dataset = cpath_dataset(X_test_np,
                                       clinical_vars_test,
                                       tb_test,
                                       e_test)

    # import data
    cpath_train_loader = torch.utils.data.DataLoader(cpath_train_dataset,
                                                     batch_size=len(cpath_train_dataset),
                                                     shuffle=True,
                                                     num_workers=0)

    cpath_val_loader = torch.utils.data.DataLoader(cpath_val_dataset,
                                                   batch_size=len(cpath_val_dataset),
                                                   shuffle=False,
                                                   num_workers=0)
    cpath_test_loader = torch.utils.data.DataLoader(cpath_test_dataset,
                                                    batch_size=len(cpath_test_dataset),
                                                    shuffle=False,
                                                    num_workers=0)

    return cpath_train_loader,cpath_test_loader,cpath_val_loader,pathway_mask