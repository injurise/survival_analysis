import time
import torch
from torch import nn
from src.models.model_utils import AverageMeter
from src.models.loss_functions.loss_functions import partial_ll_loss
from sksurv.metrics import concordance_index_censored

def train_cox_model(args,
          surv_train_loader,
          model,
          optimizer,
          epoch,
          tb_writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    plls = AverageMeter()
    c_indexs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(surv_train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        tb = target["tb"].cpu()
        e = target["e"].cpu()
        input_var = input.cpu()

        # compute output
        output_ = []
        kl_ = []
        for mc_run in range(args.num_mc):
            output, kl = model(input_var)
            output_.append(output)
            kl_.append(kl)
        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        y = output.view(args.batch_size)
        loss_crit_metric = partial_ll_loss(output.view(args.batch_size),tb.reshape(-1),e.reshape(-1))
        scaled_loss_crit_metric = loss_crit_metric / args.batch_size
        scaled_kl = kl / args.batch_size
        #ELBO loss
        loss =  scaled_loss_crit_metric + scaled_kl

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        conc_metric = concordance_index_censored(e.detach().numpy().astype(bool).reshape(-1),
                                                 tb.detach().numpy().reshape(-1),
                                                 output.view(args.batch_size).detach().numpy())[0]
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        plls.update(scaled_loss_crit_metric.item(), input.size(0))
        c_indexs.update(conc_metric, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PLL {plls.val:.3f} ({plls.avg:.3f})\t'
                  'C-Index {c_ind.val:.3f} ({c_ind.avg:.3f})' .format(
                      epoch,
                      i,
                      len(surv_train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      plls=plls,
                      c_ind=c_indexs))

        if tb_writer is not None:
            tb_writer.add_scalar('train/loss_crit_metric',
                                 scaled_loss_crit_metric.item(), epoch)
            tb_writer.add_scalar('train/kl_div', scaled_kl.item(), epoch)
            tb_writer.add_scalar('train/elbo_loss', loss.item(), epoch)
            tb_writer.flush()


def validate_cox_model(args, surv_val_loader, model, epoch, tb_writer=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    c_indexs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(surv_val_loader):

            tb = target["tb"].cpu()
            e = target["e"].cpu()
            input_var = input.cpu()

            # compute output
            output_ = []
            kl_ = []
            for mc_run in range(args.num_mc):
                output, kl = model(input_var)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            error_metric = partial_ll_loss(output.view(args.batch_size), tb.reshape(-1), e.reshape(-1))
            scaled_error_metric = error_metric /  args.batch_size
            scaled_kl = kl / args.batch_size

            #ELBO loss
            loss = scaled_error_metric + scaled_kl

            conc_metric = concordance_index_censored(e.detach().numpy().astype(bool).reshape(-1),
                                                     tb.detach().numpy().reshape(-1),
                                                     output.view(args.batch_size).detach().numpy())[0]

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            errors.update(scaled_error_metric.item(), input.size(0))
            c_indexs.update(conc_metric,input.size(0))

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
                          len(surv_val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          error=errors,
                          c_ind=c_indexs))
        '''
            if tb_writer is not None:
                tb_writer.add_scalar('val/loss_crit_metric',
                                     loss_crit_metric.item(), epoch)
                tb_writer.add_scalar('val/kl_div', scaled_kl.item(), epoch)
                tb_writer.add_scalar('val/elbo_loss', loss.item(), epoch)
                tb_writer.add_scalar('val/accuracy', prec1.item(), epoch)
                tb_writer.flush()
        '''
    print(' * Error {error.avg:.3f}'.format(error=errors))

    return errors.avg


def evaluate_surv_model(args, model, surv_val_loader):

    pred_probs_mc = []
    test_loss = 0
    correct = 0
    output_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        begin = time.time()
        for i, (input, target) in enumerate(surv_val_loader):

            tb = target["tb"].cpu()
            e = target["e"].cpu()
            input_var = input.cpu()

            output_mc = []
            for mc_run in range(args.num_mc):
                output, _ = model.forward(input_var)
                output_mc.append(output)
            output_ = torch.mean(torch.stack(output_mc), dim=0)
            output_list.append(output_)
            print('Test Error:',
                  (partial_ll_loss(output.view(args.batch_size), tb.reshape(-1), e.reshape(-1)) /args.batch_size
                   ))
        end = time.time()


