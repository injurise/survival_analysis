import numpy as np
import os
import torch
from torch import nn
from src.data_prep.torch_datasets import surv_dataset
from src.models.model_configs.model_architectures import Bay_TestNet
from src.models.bnn_cust_cox import train_cox_model,validate_cox_model,evaluate_surv_model
from src.models.model_utils import AverageMeter,save_checkpoint


def run_survival_bnn(survival_train_dataset,surival_val_dataset,args,mode):
    model.cpu()

    best_pred = 0

    tb_writer = None

    surv_train_loader = torch.utils.data.DataLoader(survival_train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.workers)

    surv_val_loader = torch.utils.data.DataLoader(surival_val_dataset,
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
            train_cox_model(args, surv_train_loader, model, optimizer, epoch,
              tb_writer)

            val_score = validate_cox_model(args, surv_val_loader, model, epoch,
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
        evaluate(args, model ,surv_val_loader)

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
    from pysurvival.models.simulations import SimulationModel

    # Initializing the simulation model
    lin_sim = SimulationModel(survival_distribution='exponential',
                              risk_type='linear',
                              censored_parameter=100.0,
                              risk_parameter=100,
                              alpha=0.01,
                              beta=5., )
    N = 1000
    linear_data_train = lin_sim.generate_data(num_samples=N, num_features=2).astype(float)
    linear_data_test = lin_sim.generate_data(num_samples=int(N), num_features=2).astype(float)

    survival_train_dataset = surv_dataset(linear_data_train[["x_1","x_2"]].values,
                                          linear_data_train["time"].values,
                                          linear_data_train["event"].values)
    surival_val_dataset = surv_dataset(linear_data_test[["x_1","x_2"]].values,
                                          linear_data_test["time"].values,
                                          linear_data_test["event"].values)

    #run model

    args = arguments(200, 1000, 500, 300, "train", 0.01, 0, "test_model_cox", "model_checkpoints")

    from src.models.variational_layers.linear_reparam import LinearReparam


    class Bay_SurvTestNet(nn.Module):

        def __init__(self, In_Nodes, Hidden_Nodes, Out_Nodes, mean=0., variance=0.1):
            super(Bay_SurvTestNet, self).__init__()
            # self.tanh = nn.Tanh()
            self.l1 = LinearReparam(in_features=In_Nodes,
                                    out_features=Hidden_Nodes,
                                    prior_means=np.full((Hidden_Nodes, In_Nodes), mean),
                                    prior_variances=np.full((Hidden_Nodes, In_Nodes), variance),
                                    posterior_mu_init=np.full((Hidden_Nodes, In_Nodes), 0.5),
                                    posterior_rho_init=np.full((Hidden_Nodes, In_Nodes), -3.),
                                    bias=False,
                                    )


            self.l2 = LinearReparam(in_features=Hidden_Nodes,
                                    out_features=Out_Nodes,
                                  prior_means=np.full((Out_Nodes, Hidden_Nodes), mean),
                                    prior_variances=np.full((Out_Nodes, Hidden_Nodes), variance),
                                    posterior_mu_init=np.full((Out_Nodes, Hidden_Nodes), 0.5),
                                    posterior_rho_init=np.full((Out_Nodes, Hidden_Nodes), -3.),
                                    bias=False,
                                    )

        def forward(self, x):
            lin_pred = self.l1(x)
            lin_pred = self.l2(lin_pred[0])

            return lin_pred


    model = Bay_SurvTestNet(2, 3, 1,variance = 1500)

    #criterion = nn.MSELoss().cpu()

    run_survival_bnn(survival_train_dataset,surival_val_dataset,args,model)

    #inp = torch.tensor([[0.5, 0.5]], dtype=torch.float)

    #print(model(inp))
