
import torch
import torch.nn as nn
import numpy as np
from CoxPASNet.coxpasnet.Survival_CostFunc_CIndex import R_set, neg_par_log_likelihood, c_index

from configs.model_configs import Bay_CPASNet
import torch.optim as optim
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss


def trainCoxPASNet(train_x, train_age, train_ytime, train_yevent, \
			eval_x, eval_age, eval_ytime, eval_yevent, pathway_mask, \
			In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, \
			Learning_Rate, L2, Num_Epochs, Dropout_Rate):

	net = Bay_CPASNet(In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, pathway_mask)
	###
	###optimizer
	opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay = L2)

	for epoch in range(Num_Epochs+1):
		net.train()
		opt.zero_grad() ###reset gradients to zeros

		pred = net(train_x, train_age) ###Forward
        ce_loss = neg_par_log_likelihood(pred, train_ytime, train_yevent)
        kl = get_kl_loss(net)
        loss = ce_loss + kl
        loss.backward() ###calculate gradients
        opt.step()
        net.sc1.weight.data = net.sc1.weight.data.mul(net.pathway_mask) ###force the connections between gene layer and pathway layer

        if epoch % 20 == 0:
            with torch.no_grad():
                net.train()
                train_output_mc = []
                for mc_run in range(10):
                    output = net(train_x, train_age)
                    train_output_mc.append(output)
                    outputs = torch.stack(train_output_mc)
                train_pred = outputs.mean(dim=0)
                train_loss = neg_par_log_likelihood(train_pred, train_ytime, train_yevent).view(1,)

                eval_output_mc = []
                for mc_run in range(10):
                    output = net(eval_x, eval_age)
                    eval_output_mc.append(output)
                    eval_outputs = torch.stack(eval_output_mc)
                eval_pred = eval_outputs.mean(dim=0)
                eval_loss = neg_par_log_likelihood(eval_pred, eval_ytime, eval_yevent).view(1,)

                train_cindex = c_index(train_pred, train_ytime, train_yevent)
                eval_cindex = c_index(eval_pred, eval_ytime, eval_yevent)
                print(f"Epoch: {epoch}, Train Loss: {train_loss},Eval Loss: {eval_loss}, "
                      f" Train Cindex: {train_cindex}, Eval Cindex: {eval_cindex}")

	return (train_loss, eval_loss, train_cindex, eval_cindex)