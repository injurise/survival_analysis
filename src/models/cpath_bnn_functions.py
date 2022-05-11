from src.models.loss_functions.loss_functions import partial_ll_loss
from sksurv.metrics import concordance_index_censored
from src.models.model_utils import AverageMeter,save_checkpoint
import numpy as np
import torch
from torch import nn
import time




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

            if args.cuda:
                tb = target["tb"].cuda()
                e = target["e"].cuda()
                input_var = input["X"].cuda()
                clinical_var = input["clinical_vars"].cuda()

            else:
                tb = target["tb"].cpu()
                e = target["e"].cpu()
                input_var = input["X"].cpu()
                clinical_var = input["clinical_vars"].cpu()

            output_ = []
            for mc_run in range(args.num_mc):
                output = model(input_var, clinical_var)
                output_.append(output)
            output = torch.mean(torch.stack(output_), dim=0)
            loss_crit_metric = partial_ll_loss(output.reshape(-1).cpu(), tb.reshape(-1).cpu(), e.reshape(-1).cpu())
            scaled_loss_crit_metric = loss_crit_metric * (len(train_data_loader.dataset) /train_data_loader.batch_size)
            scaled_kl = model.kl_divergence() / train_data_loader.batch_size
            loss = scaled_loss_crit_metric + scaled_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            conc_metric = concordance_index_censored(e.detach().cpu().numpy().astype(bool).reshape(-1),
                                                     tb.detach().cpu().numpy().reshape(-1),
                                                     output.reshape(-1).detach().cpu().numpy())[0]
            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            losses.update(loss.item())
            plls.update(scaled_loss_crit_metric.item())
            c_indexs.update(conc_metric)

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

            return losses.sum,c_indexs.avg,plls.sum


def test(args,model,test_data_loader):
    ##Still has to be adjusted
    model.eval()
    test_loss = 0
    output_list = []
    tb_list = []
    e_list = []

    model.eval()

    with torch.no_grad():
        for input, target in test_data_loader:

            if args.cuda:
                tb = target["tb"].cuda()
                e = target["e"].cuda()
                input_var = input["X"].cuda()
                clinical_var = input["clinical_vars"].cuda()

            else:
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
            output_list.append(output)
            tb_list.append(tb)
            e_list.append(e)
        output = torch.cat(output_list)
        tb = torch.cat(tb_list)
        e = torch.cat(e_list)

        pll= partial_ll_loss(output.reshape(-1).cpu(), tb.reshape(-1).cpu(), e.reshape(-1).cpu())
        #scaled_error_metric = error_metric * (len(test_data_loader.dataset) / test_data_loader.batch_size)
        #scaled_kl = model.kl_divergence() / test_data_loader.batch_size

        # ELBO loss
        #loss = error_metric + scaled_kl
        conc_metric = concordance_index_censored(e.detach().cpu().numpy().astype(bool).reshape(-1),
                                                     tb.detach().cpu().numpy().reshape(-1),
                                                     output.reshape(-1).detach().cpu().numpy())[0]
        return conc_metric,pll.item()




# train the model and save some visualisations on the way
def validate(args, cpath_val_loader, model, tb_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    c_indexs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(cpath_val_loader):

            if args.cuda:
                tb = target["tb"].cuda()
                e = target["e"].cuda()
                input_var = input["X"].cuda()
                clinical_var = input["clinical_vars"].cuda()

            else:
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
            error_metric = partial_ll_loss(output.reshape(-1).cpu(), tb.reshape(-1).cpu(), e.reshape(-1).cpu())
            scaled_error_metric = error_metric * (len(cpath_val_loader.dataset) /cpath_val_loader.batch_size)
            scaled_kl = model.kl_divergence() / cpath_val_loader.batch_size

            # ELBO loss
            loss = scaled_error_metric + scaled_kl

            conc_metric = concordance_index_censored(e.detach().cpu().numpy().astype(bool).reshape(-1),
                                                     tb.detach().cpu().numpy().reshape(-1),
                                                     output.reshape(-1).detach().cpu().numpy())[0]

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            losses.update(loss.item())
            errors.update(scaled_error_metric.item())
            c_indexs.update(conc_metric)

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

    return losses.sum,c_indexs.avg,errors.sum
