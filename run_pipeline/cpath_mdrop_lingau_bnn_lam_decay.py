import sys
sys.path.append('/cluster/home/amollers/Github/survival_analysis')
sys.path.append('/Users/alexandermollers/Documents/GitHub/survival_analysis')
import numpy as np
import torch
from torch import nn
import pandas as pd
from src.models.model_utils import AverageMeter,save_checkpoint
from src.models.variational_layers.linear_reparam import LinearReparam, LinearGroupNJ_Pathways
from src.data_prep.torch_datasets import cpath_dataset

import os                              # until reworked
import torch.optim as optim
from src.models.cpath_bnn_functions import train,test,validate

############# Model Definition ##################

class cpath_md_lg(nn.Module):
    def __init__(self, In_Nodes, Pathway_Nodes, Hidden_Nodes,Last_layer_Nodes, args,mask):
        super(cpath_md_lg, self).__init__()
        # activation
        self.tanh = nn.Tanh()
        # layers
        self.fc1 = LinearGroupNJ_Pathways(In_Nodes, Pathway_Nodes,mask= mask, cuda=args.cuda)
        self.fc2 = LinearReparam(in_features=Pathway_Nodes,
                                out_features=Hidden_Nodes,
                                prior_means=np.full((Hidden_Nodes, Pathway_Nodes), args.gp_mean),
                                prior_variances=np.full((Hidden_Nodes, Pathway_Nodes), args.gp_var),
                                posterior_mu_init=np.full((Hidden_Nodes, Pathway_Nodes), 0.5),
                                posterior_rho_init=np.full((Hidden_Nodes, Pathway_Nodes), -3.),
                                bias=False,
                                )
        self.fc3 = LinearReparam(in_features=Hidden_Nodes,
                                out_features=Last_layer_Nodes,
                                prior_means=np.full((Last_layer_Nodes, Hidden_Nodes), args.gp_mean),
                                prior_variances=np.full((Last_layer_Nodes, Hidden_Nodes), args.gp_var),
                                posterior_mu_init=np.full((Last_layer_Nodes, Hidden_Nodes), 0.5),
                                posterior_rho_init=np.full((Last_layer_Nodes, Hidden_Nodes), -3.),
                                bias=False,
                                )

        self.fc4 = LinearReparam(in_features=Last_layer_Nodes + 1,
                                out_features=1,
                                prior_means=np.full((1, Last_layer_Nodes + 1), args.gp_mean),
                                prior_variances=np.full((1, Last_layer_Nodes + 1), args.gp_var),
                                posterior_mu_init=np.full((1, Last_layer_Nodes + 1), 0.5),
                                posterior_rho_init=np.full((1, Last_layer_Nodes + 1), -3.),
                                bias=False
                                )

        # layers including kl_divergence
        self.kl_list = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, x, clinical_vars):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x,return_kl = False))
        x = self.tanh(self.fc3(x,return_kl = False))
        x_cat = torch.cat((x, clinical_vars), 1)
        lin_pred = self.fc4(x_cat,return_kl = False)

        return lin_pred

    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD


def main():
    global best_loss_val_score

    pathway_mask = pd.read_csv("../data/pathway_mask.csv", index_col=0).values
    pathway_mask = torch.from_numpy(pathway_mask).type(torch.FloatTensor)
    if args.cuda:
        pathway_mask=pathway_mask.to(device = 'cuda')

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


    # init model
    model = cpath_md_lg(5567,860, 100, 30,args,mask = pathway_mask)
    if args.cuda:
        model.cuda()
        # init optimizer
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    lmbda = lambda epoch: 0.995 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

    for epoch in range(1, args.epochs + 1):
        loss_train_score, c_index_train_score,pll_train = train(args, model, cpath_train_loader, epoch, optimizer)
        loss_val_score,c_index_val_score,pll_val = validate(args, cpath_val_loader, model, epoch)
        test_conc_metric,test_pll = test(args, model, cpath_test_loader)

        epoch_dict = {

            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]["lr"],
            'gp_mean': args.gp_mean,
            'gp_var': args.gp_var,
            'loss_train_score': loss_train_score,
            'train_pll': pll_train,
            'ctrain_score': c_index_train_score,
            'cval_score': c_index_val_score,
            'loss_val_score': loss_val_score,
            'val_pll': pll_val,
            'test_conc_metric': test_conc_metric,
            'test_pll': test_pll,
            'state_dict': model.state_dict()
        }
        is_best = loss_val_score < best_loss_val_score
        best_loss_val_score = min(loss_val_score, best_loss_val_score)
        if args.save_best_model & is_best:
                save_checkpoint(
                    epoch_dict,
                    is_best,
                    filename=os.path.join(
                        args.save_dir,
                        'bayesian_{}.pth'.format(args.arch)))

        epoch_dict.pop("state_dict", None)
        epoch_log = {k: [v] for k, v in epoch_dict.items()}
        epoch_log_df = pd.DataFrame.from_dict(epoch_log, orient="columns")
        log_file_name = args.arch + '_logs.csv'
        log_path = os.path.join(args.log_dir, log_file_name)
        epoch_log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path),index = False)
        scheduler.step()

    return epoch_log_df.to_string(header=False, index=False)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--num-mc',dest = "num_mc", type=int, default=200)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gp-mean', dest="gp_mean", type=float, default=0)
    parser.add_argument('--gp-var', dest="gp_var", type=float, default=0.01)
    parser.add_argument('--save-best-model', dest = "save_best_model",choices=('True','False'), default='True')
    parser.add_argument('--save-dir',dest="save_dir", type=str, default='/Users/alexandermollers/Documents/GitHub/survival_analysis/run_pipeline/model_checkpoints')
    parser.add_argument('--arch',default='cpath_model')
    parser.add_argument('--log-dir',dest = "log_dir", default='/Users/alexandermollers/Documents/GitHub/survival_analysis/run_pipeline/model_checkpoints')

    args = parser.parse_args()
    model_bool = args.save_best_model == 'True'
    args.save_best_model = model_bool
    args.cuda = torch.cuda.is_available() # check if we can put the net on the GPU
    best_loss_val_score = 0
    main()


