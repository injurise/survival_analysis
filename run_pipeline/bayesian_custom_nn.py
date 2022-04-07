import numpy as np
import os
import torch
from torch import nn
from src.data_prep.torch_datasets import Dataset
from src.models.model_configs.model_architectures import Bay_TestNet
from src.models.bnn_cust import train,validate,evaluate
from src.models.model_utils import AverageMeter,save_checkpoint


def run_custom_bnn(train_dataset,val_dataset,args,model,criterion):
    model.cpu()

    best_pred = 0

    tb_writer = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.workers,
                                            pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.workers,
                                            pin_memory=True)

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
            train(args, train_loader, model, criterion, optimizer, epoch,
              tb_writer)

            val_score = validate(args, val_loader, model, criterion, epoch,
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
        evaluate(args, model, criterion ,val_loader)

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

    # Simulate some Data
    x1 = np.random.normal(size=1000)
    x2 = np.random.normal(size=1000)
    X = np.stack((x1, x2), axis=-1)
    y = x1 + x2

    X_train = X[0:500]
    y_train = y[0:500]
    X_test = X[500:1000]
    y_test = y[500:1000]

    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_test, y_test)

    #run model

    args = arguments(200, 500, 200, 200, "train", 0.01, 0, "test_model_no_noise", "model_checkpoints")

    model = Bay_TestNet(2, 3, 1)

    criterion = nn.MSELoss().cpu()

    run_custom_bnn(train_dataset,val_dataset,args,model,criterion)

    inp = torch.tensor([[0.5, 0.5]], dtype=torch.float)

    print(model(inp))
