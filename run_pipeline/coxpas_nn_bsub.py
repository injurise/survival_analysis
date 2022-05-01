import os
from CoxPASNet.coxpasnet.DataLoader import load_data, load_pathway
import torch
import torch.nn as nn
import torch.optim as optim
from sksurv.metrics import concordance_index_censored

from CoxPASNet.coxpasnet.Model import Cox_PASNet
from CoxPASNet.coxpasnet.SubNetwork_SparseCoding import dropout_mask, s_mask
from CoxPASNet.coxpasnet.Survival_CostFunc_CIndex import neg_par_log_likelihood

import copy
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

#These functions are from them, I just adjust them a tiny little bit to store some results differently

def trainCoxPASNet_GS(train_x, train_age, train_ytime, train_yevent,
                      eval_x, eval_age, eval_ytime, eval_yevent,test_x,test_age,
                      test_ytime, test_yevent, pathway_mask, In_Nodes,
                      Pathway_Nodes, Hidden_Nodes, Out_Nodes, Dropout_Rate):

    dtype = torch.FloatTensor
    net = Cox_PASNet(In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, pathway_mask)
    ###if gpu is being used
    if torch.cuda.is_available():
        net.cuda()
    ###
    ###optimizer
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.lr)

    for epoch in range(args.epochs):
        net.train()
        opt.zero_grad()  ###reset gradients to zeros
        ###Randomize dropout masks
        net.do_m1 = dropout_mask(Pathway_Nodes, Dropout_Rate[0])
        net.do_m2 = dropout_mask(Hidden_Nodes, Dropout_Rate[1])

        pred = net(train_x, train_age)  ###Forward
        loss = neg_par_log_likelihood(pred, train_ytime, train_yevent)  ###calculate loss
        loss.backward()  ###calculate gradients
        opt.step()  ###update weights and biases

        net.sc1.weight.data = net.sc1.weight.data.mul(
            net.pathway_mask)  ###force the connections between gene layer and pathway layer

        ###obtain the small sub-network's connections
        do_m1_grad = copy.deepcopy(net.sc2.weight._grad.data)
        do_m2_grad = copy.deepcopy(net.sc3.weight._grad.data)
        do_m1_grad_mask = torch.where(do_m1_grad == 0, do_m1_grad, torch.ones_like(do_m1_grad))
        do_m2_grad_mask = torch.where(do_m2_grad == 0, do_m2_grad, torch.ones_like(do_m2_grad))
        ###copy the weights
        net_sc2_weight = copy.deepcopy(net.sc2.weight.data)
        net_sc3_weight = copy.deepcopy(net.sc3.weight.data)

        ###serializing net
        net_state_dict = net.state_dict()

        ###Sparse Coding
        ###make a copy for net, and then optimize sparsity level via copied net
        copy_net = copy.deepcopy(net)
        copy_state_dict = copy_net.state_dict()
        for name, param in copy_state_dict.items():
            ###omit the param if it is not a weight matrix
            if not "weight" in name:
                continue
            ###omit gene layer
            if "sc1" in name:
                continue
            ###stop sparse coding
            if "sc4" in name:
                break
            ###sparse coding between the current two consecutive layers is in the trained small sub-network
            if "sc2" in name:
                active_param = net_sc2_weight.mul(do_m1_grad_mask)
            if "sc3" in name:
                active_param = net_sc3_weight.mul(do_m2_grad_mask)
            nonzero_param_1d = active_param[active_param != 0]
            if nonzero_param_1d.size(
                    0) == 0:  ###stop sparse coding between the current two consecutive layers if there are no valid weights
                break
            copy_param_1d = copy.deepcopy(nonzero_param_1d)
            ###set up potential sparsity level in [0, 100)
            S_set = torch.arange(100, -1, -1)[1:]
            copy_param = copy.deepcopy(active_param)
            S_loss = []
            for S in S_set:
                param_mask = s_mask(sparse_level=S.item(), param_matrix=copy_param, nonzero_param_1D=copy_param_1d,
                                    dtype=dtype)
                transformed_param = copy_param.mul(param_mask)
                copy_state_dict[name].copy_(transformed_param)
                copy_net.train()
                y_tmp = copy_net(train_x, train_age)
                loss_tmp = neg_par_log_likelihood(y_tmp, train_ytime, train_yevent)
                S_loss.append(loss_tmp.item())
            ###apply cubic interpolation
            interp_S_loss = interp1d(S_set, S_loss, kind='cubic')
            interp_S_set = torch.linspace(min(S_set), max(S_set), steps=100)
            interp_loss = interp_S_loss(interp_S_set)
            optimal_S = interp_S_set[np.argmin(interp_loss)]
            optimal_param_mask = s_mask(sparse_level=optimal_S.item(), param_matrix=copy_param,
                                        nonzero_param_1D=copy_param_1d, dtype=dtype)
            if "sc2" in name:
                final_optimal_param_mask = torch.where(do_m1_grad_mask == 0, torch.ones_like(do_m1_grad_mask),
                                                       optimal_param_mask)
                optimal_transformed_param = net_sc2_weight.mul(final_optimal_param_mask)
            if "sc3" in name:
                final_optimal_param_mask = torch.where(do_m2_grad_mask == 0, torch.ones_like(do_m2_grad_mask),
                                                       optimal_param_mask)
                optimal_transformed_param = net_sc3_weight.mul(final_optimal_param_mask)
            ###update weights in copied net
            copy_state_dict[name].copy_(optimal_transformed_param)
            ###update weights in net
            net_state_dict[name].copy_(optimal_transformed_param)

        net.train()
        train_pred = net(train_x, train_age)
        train_loss = neg_par_log_likelihood(train_pred, train_ytime, train_yevent).view(1, )

        net.eval()
        eval_pred = net(eval_x, eval_age)
        eval_loss = neg_par_log_likelihood(eval_pred, eval_ytime, eval_yevent).view(1, )

        test_pred = net(test_x, test_age)
        test_loss = neg_par_log_likelihood(test_pred, test_ytime, test_yevent).view(1, )


        train_cindex = concordance_index_censored(train_yevent.detach().cpu().numpy().astype(bool).reshape(-1),
                                                 train_ytime.detach().cpu().numpy().reshape(-1),
                                                 train_pred.detach().cpu().numpy().reshape(-1))[0]
        eval_cindex = concordance_index_censored(eval_yevent.detach().cpu().numpy().astype(bool).reshape(-1),
                                                 eval_ytime.detach().cpu().numpy().reshape(-1),
                                                 eval_pred.detach().cpu().numpy().reshape(-1))[0]
        test_cindex = concordance_index_censored(test_yevent.detach().cpu().numpy().astype(bool).reshape(-1),
                                                 test_ytime.detach().cpu().numpy().reshape(-1),
                                                 test_pred.detach().cpu().numpy().reshape(-1))[0]

        epoch_log = {
                'epoch': epoch + 1,
                'lr': args.lr,
                'l2': args.l2,
                'loss_train_score': train_loss.detach().cpu().numpy()[0],
                'ctrain_score': train_cindex,
                'cval_score': eval_cindex,
                'loss_val_score': eval_loss.detach().cpu().numpy()[0],
                'test_conc_metric': test_cindex
            }
        epoch_log = {k: [v] for k, v in epoch_log.items()}
        epoch_log_df = pd.DataFrame.from_dict(epoch_log, orient="columns")
        log_file_name = args.arch + '_logs.csv'
        log_path = os.path.join(args.log_dir, log_file_name)
        epoch_log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path),index=False)


        print("Loss in Train: ", train_loss)

    return (train_loss, eval_loss, train_cindex, eval_cindex)


def main():
    dtype = torch.FloatTensor
    ''' Net Settings'''
    In_Nodes = 5567 ###number of genes
    Pathway_Nodes = 860 ###number of pathways
    Hidden_Nodes = 100 ###number of hidden nodes
    Out_Nodes = 30 ###number of hidden nodes in the last hidden layer
    ''' Initialize '''
    Initial_Learning_Rate = [0.03] #[0.03, 0.01, 0.001, 0.00075]
    L2_Lambda = [0.01]  #[0.1, 0.01, 0.005, 0.001]
    num_epochs = 10 #3000 ###for grid search
    Num_EPOCHS = 15 #20000 ###for training
    ###sub-network setup
    Dropout_Rate = [0.7,0.5]


    ''' load data and pathway '''
    pathway_mask = load_pathway("../data/pathway_mask.csv", dtype)

    x_train, ytime_train, yevent_train, age_train = load_data("../data/train.csv", dtype)
    x_valid, ytime_valid, yevent_valid, age_valid = load_data("../data/validation.csv", dtype)
    x_test, ytime_test, yevent_test, age_test = load_data("../data/test.csv", dtype)

    opt_l2_loss = 0
    opt_lr_loss = 0
    opt_loss = torch.Tensor([float("Inf")])
    ###if gpu is being used
    if torch.cuda.is_available():
        opt_loss = opt_loss.cuda()
    ###
    opt_c_index_va = 0
    opt_c_index_tr = 0
    ###grid search the optimal hyperparameters using train and validation data

    loss_train, loss_valid, c_index_tr, c_index_va = trainCoxPASNet_GS(x_train, age_train, ytime_train, yevent_train,
                                                                    x_valid, age_valid, ytime_valid, yevent_valid,
                                                                    x_test,age_test,ytime_test,yevent_test,
                                                                    pathway_mask,In_Nodes, Pathway_Nodes, Hidden_Nodes,
                                                                    Out_Nodes,Dropout_Rate)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--arch',default='coxpas')
    parser.add_argument('--log-dir',dest = "log_dir", default='/Users/alexandermollers/Documents/GitHub/survival_analysis/run_pipeline/model_checkpoints')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() # check if we can put the net on the GPU
    best_cval_score = 0
    main()