import numpy as np
import os
import torch
from torch import nn
import pandas as pd
from src.data_prep.torch_datasets import cpath_dataset
from src.models.model_configs.model_architectures import Bay_TestNet
from src.models.bnn_cust_cpath import train_cpath_bnn,validate_cpath_model,evaluate_cpath_model
from src.models.model_utils import AverageMeter,save_checkpoint
from src.models.variational_layers.linear_reparam import LinearReparam
from src.data_prep.torch_datasets import cpath_dataset




def run_cpath_bnn(cpath_train_dataset,cpath_val_dataset,model,args,mode):
    model.cpu()

    best_pred = 0

    tb_writer = None

    cpath_train_loader = torch.utils.data.DataLoader(cpath_train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.workers)

    cpath_val_loader = torch.utils.data.DataLoader(cpath_val_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.workers)

    if args.mode == 'train':

        for epoch in range(args.epochs):

            lr = args.lr
            if (epoch >= 80 and epoch < 120):
                lr = 0.1 * args.lr
            elif (epoch >= 120 and epoch < 160):
                lr = 0.01 * args.lr
            elif (epoch >= 160 and epoch < 180):
                lr = 0.001 * args.lr
            elif (epoch >= 180):
                lr = 0.0005 * args.lr

            optimizer = torch.optim.Adam(model.parameters(), lr)

            # train for one epoch
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            train_cpath_bnn(args, cpath_train_loader, model, optimizer, epoch,
              tb_writer)

            val_score = validate_cpath_model(args, cpath_val_loader, model, epoch,
                             tb_writer)

            is_best = val_score >= best_pred
            best_val = max(val_score, best_pred)

            if is_best:
                save_checkpoint(
                    {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val': best_val,
                    },
                    is_best,
                    filename=os.path.join(
                        args.save_dir,
                    '   bayesian_{}.pth'.format(args.model_name)))

    elif args.mode == 'test':
        checkpoint_file = args.save_dir + '/bayesian_{}.pth'.format(
            args.model_name)

        checkpoint = torch.load(checkpoint_file,
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        evaluate_cpath_model(args, model ,cpath_val_loader)


class arguments():
    def __init__(self, num_mc, batch_size, print_freq, epochs, mode, lr, workers, model_name, save_dir):
        self.num_mc = num_mc
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.epochs = epochs
        self.mode = mode
        self.lr = lr
        self.workers = workers
        self.model_name = model_name
        self.save_dir = save_dir



if __name__ == '__main__':
    class Bay_CPathNet(nn.Module):

        def __init__(self, In_Nodes, Hidden_Nodes, Last_Nodes, mean=0., variance=0.1):
            super(Bay_CPathNet, self).__init__()
            self.tanh = nn.Tanh()
            self.l1 = LinearReparam(in_features=In_Nodes,
                                    out_features=Hidden_Nodes,
                                    prior_means=np.full((Hidden_Nodes, In_Nodes), mean),
                                    prior_variances=np.full((Hidden_Nodes, In_Nodes), variance),
                                    posterior_mu_init=np.full((Hidden_Nodes, In_Nodes), 0.5),
                                    posterior_rho_init=np.full((Hidden_Nodes, In_Nodes), -3.),
                                    bias=False,
                                    )

            self.l2 = LinearReparam(in_features=Hidden_Nodes,
                                    out_features=Hidden_Nodes,
                                    prior_means=np.full((Hidden_Nodes, Hidden_Nodes), mean),
                                    prior_variances=np.full((Hidden_Nodes, Hidden_Nodes), variance),
                                    posterior_mu_init=np.full((Hidden_Nodes, Hidden_Nodes), 0.5),
                                    posterior_rho_init=np.full((Hidden_Nodes, Hidden_Nodes), -3.),
                                    bias=False,
                                    )

            self.l3 = LinearReparam(in_features=Hidden_Nodes,
                                    out_features=Last_Nodes,
                                    prior_means=np.full((Last_Nodes, Hidden_Nodes), mean),
                                    prior_variances=np.full((Last_Nodes, Hidden_Nodes), variance),
                                    posterior_mu_init=np.full((Last_Nodes, Hidden_Nodes), 0.5),
                                    posterior_rho_init=np.full((Last_Nodes, Hidden_Nodes), -3.),
                                    bias=False,
                                    )

            self.l4 = LinearReparam(in_features=Last_Nodes + 1,
                                    out_features=1,
                                    prior_means=np.full((1, Last_Nodes + 1), mean),
                                    prior_variances=np.full((1, Last_Nodes + 1), variance),
                                    posterior_mu_init=np.full((1, Last_Nodes + 1), 0.5),
                                    posterior_rho_init=np.full((1, Last_Nodes + 1), -3.),
                                    bias=False,
                                    )

        def forward(self, x, clinical_vars):
            pred = self.tanh(self.l1(x,return_kl = False))
            pred = self.tanh(self.l2(pred, return_kl=False))
            pred = self.tanh(self.l3(pred, return_kl=False))
            x_cat = torch.cat((pred, clinical_vars), 1)
            lin_pred = self.l4(x_cat, return_kl=True)
            return lin_pred

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

    cpath_train_dataset = cpath_dataset(X_train_np,
                                        clinical_vars_train,
                                        tb_train,
                                        e_train)

    cpath_val_dataset = cpath_dataset(X_val_np,
                                      clinical_vars_val,
                                      tb_val,
                                      e_val)

    # run model

    args = arguments(200, 84, 50, 10, "train", 0.01, 0, "test_model_cpath", "model_checkpoints")

    model = Bay_CPathNet(5567, 100, 30, variance=200)

    run_cpath_bnn(cpath_train_dataset, cpath_val_dataset, model, args, "train")

