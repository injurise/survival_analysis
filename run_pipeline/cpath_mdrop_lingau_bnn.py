import numpy as np
import torch
from torch import nn
import pandas as pd
from src.models.bnn_cust_cpath import train_cpath_bnn,validate_cpath_model,evaluate_cpath_model
from src.models.model_utils import AverageMeter,save_checkpoint
from src.models.variational_layers.linear_reparam import LinearReparam, LinearGroupNJ_Pathways
from src.data_prep.torch_datasets import cpath_dataset
from src.models.loss_functions.loss_functions import partial_ll_loss
from sksurv.metrics import concordance_index_censored
from torch.autograd import Variable # decpreciated just used in the test method right now, so importing
                                    # until reworked
import torch.optim as optim
import time


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
                                prior_means=np.full((Hidden_Nodes, Pathway_Nodes), 0),
                                prior_variances=np.full((Hidden_Nodes, Pathway_Nodes), 0.01),
                                posterior_mu_init=np.full((Hidden_Nodes, Pathway_Nodes), 0.5),
                                posterior_rho_init=np.full((Hidden_Nodes, Pathway_Nodes), -3.),
                                bias=False,
                                )
        self.fc3 = LinearReparam(in_features=Hidden_Nodes,
                                out_features=Last_layer_Nodes,
                                prior_means=np.full((Last_layer_Nodes, Hidden_Nodes), 0),
                                prior_variances=np.full((Last_layer_Nodes, Hidden_Nodes), 0.01),
                                posterior_mu_init=np.full((Last_layer_Nodes, Hidden_Nodes), 0.5),
                                posterior_rho_init=np.full((Last_layer_Nodes, Hidden_Nodes), -3.),
                                bias=False,
                                )

        self.fc4 = LinearReparam(in_features=Last_layer_Nodes + 1,
                                out_features=1,
                                prior_means=np.full((1, Last_layer_Nodes + 1), 0),
                                prior_variances=np.full((1, Last_layer_Nodes + 1), 0.01),
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

def train(args, model, train_data_loader, epoch, optimizer):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        plls = AverageMeter()
        c_indexs = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            tb = target["tb"].cpu()
            e = target["e"].cpu()
            input_var = input["X"].cpu()
            clinical_var = input["clinical_vars"].cpu()

            output_ = []
            for mc_run in range(args.num_mc):
                output = model(input_var, clinical_var)
                output_.append(output)
            output = torch.mean(torch.stack(output_), dim=0)
            loss_crit_metric = partial_ll_loss(output.reshape(-1), tb.reshape(-1), e.reshape(-1))
            scaled_loss_crit_metric = loss_crit_metric * (len(train_data_loader.dataset) /train_data_loader.batch_size)
            scaled_kl = model.kl_divergence() / train_data_loader.batch_size
            loss = loss_crit_metric + scaled_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            conc_metric = concordance_index_censored(e.detach().numpy().astype(bool).reshape(-1),
                                                     tb.detach().numpy().reshape(-1),
                                                     output.reshape(-1).detach().numpy())[0]
            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            losses.update(loss.item(), input["X"].size(0))
            plls.update(scaled_loss_crit_metric.item(), input["X"].size(0))
            c_indexs.update(conc_metric, input["X"].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'PLL {plls.val:.3f} ({plls.avg:.3f})\t'
                      'C-Index {c_ind.val:.3f} ({c_ind.avg:.3f})'.format(
                    epoch,
                    i,
                    len(train_data_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    plls=plls,
                    c_ind=c_indexs))


def test(args,model,test_data_loader):
    ##Still has to be adjusted
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # commenting this out until test function has been reworked
        #test_loss += discrimination_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_data_loader.dataset)
    print('Test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_data_loader.dataset),
        100. * correct / len(test_data_loader.dataset)))


# train the model and save some visualisations on the way
def validate(args, cpath_val_loader, model, epoch, tb_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    c_indexs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(cpath_val_loader):

            tb = target["tb"].cpu()
            e = target["e"].cpu()
            input_var = input["X"].cpu()
            clinical_var = input["clinical_vars"].cpu()

            # compute output
            output_ = []
            for mc_run in range(args.num_mc):
                output = model(input_var, clinical_var)
                output_.append(output)
            output = torch.mean(torch.stack(output_), dim=0)
            error_metric = partial_ll_loss(output.reshape(-1), tb.reshape(-1), e.reshape(-1))
            scaled_error_metric = error_metric * (len(cpath_val_loader.dataset) /cpath_val_loader.batch_size)
            scaled_kl = model.kl_divergence() / cpath_val_loader.batch_size

            # ELBO loss
            loss = error_metric + scaled_kl

            conc_metric = concordance_index_censored(e.detach().numpy().astype(bool).reshape(-1),
                                                     tb.detach().numpy().reshape(-1),
                                                     output.reshape(-1).detach().numpy())[0]

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            losses.update(loss.item(), input["X"].size(0))
            errors.update(scaled_error_metric.item(), input["X"].size(0))
            c_indexs.update(conc_metric, input["X"].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error {error.val:.3f} ({error.avg:.3f})\t'
                      'C-Index {c_ind.val:.3f} ({c_ind.avg:.3f}) '.format(
                    i,
                    len(cpath_val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    error=errors,
                    c_ind=c_indexs))

    print(' * Error {error.avg:.3f}'.format(error=errors))

    return errors.avg

def main():
    pathway_mask = pd.read_csv("../data/pathway_mask.csv", index_col=0).values
    pathway_mask = torch.from_numpy(pathway_mask).type(torch.FloatTensor)

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

    # import data
    cpath_train_loader = torch.utils.data.DataLoader(cpath_train_dataset,
                                                     batch_size=len(cpath_train_dataset),
                                                     shuffle=True,
                                                     num_workers=0)

    cpath_val_loader = torch.utils.data.DataLoader(cpath_val_dataset,
                                                   batch_size=len(cpath_val_dataset),
                                                   shuffle=False,
                                                   num_workers=0)


    # init model
    model = cpath_md_lg(5567,860, 100, 30,args,mask = pathway_mask)
    if args.cuda:
        model.cuda()
        # init optimizer
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, args.epochs + 1):
        train(args, model, cpath_train_loader, epoch, optimizer)
        # test()
        # visualizations
        val_score = validate(args, cpath_val_loader, model, epoch)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_mc', type=int, default=200)
    parser.add_argument('--print_freq', type=int, default=5)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()  # check if we can put the net on the GPU
    main()
