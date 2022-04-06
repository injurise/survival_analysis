import time
import torch
from torch import nn
from src.models.model_utils import AverageMeter



def train(args,
          train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          tb_writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cpu()
        input_var = input.cpu()
        target_var = target


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
        loss_crit_metric = criterion(output.view(args.batch_size), target_var)
        scaled_kl = kl / args.batch_size
        #ELBO loss
        loss = loss_crit_metric + scaled_kl

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        mses.update(loss_crit_metric.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Mses {mses.val:.3f} ({mses.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      mses=mses))

        if tb_writer is not None:
            tb_writer.add_scalar('train/loss_crit_metric',
                                 loss_crit_metric.item(), epoch)
            tb_writer.add_scalar('train/kl_div', scaled_kl.item(), epoch)
            tb_writer.add_scalar('train/elbo_loss', loss.item(), epoch)
            tb_writer.add_scalar('train/accuracy', prec1.item(), epoch)
            tb_writer.flush()


def validate(args, val_loader, model, criterion, epoch, tb_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cpu()
            input_var = input.cpu()
            target_var = target.cpu()


            # compute output
            output_ = []
            kl_ = []
            for mc_run in range(args.num_mc):
                output, kl = model(input_var)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            error_metric = criterion(output.view(args.batch_size), target_var)
            scaled_kl = kl / args.batch_size
            #ELBO loss
            loss = error_metric + scaled_kl

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            errors.update(error_metric.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error{error.val:.3f} ({error.avg:.3f})'.format(
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          error=errors))
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


def evaluate(args, model, criterion, val_loader):
    pred_probs_mc = []
    test_loss = 0
    correct = 0
    output_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        begin = time.time()
        for data, target in val_loader:
            data, target = data.cpu(), target.cpu()
            output_mc = []
            for mc_run in range(args.num_mc):
                output, _ = model.forward(data)
                output_mc.append(output)
            output_ = torch.mean(torch.stack(output_mc), dim=0)
            output_list.append(output_)
        end = time.time()

        print('Test Error:',
              (criterion(output_list, target)))
