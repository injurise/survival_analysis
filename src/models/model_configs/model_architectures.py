import numpy as np
import torch
from torch import nn
from src.models.variational_layers.linear_reparam import LinearReparam


class Bay_TestNet(nn.Module):

    def __init__(self, In_Nodes, Hidden_Nodes, Out_Nodes ,mean=0. ,variance=.1):
        super(Bay_TestNet, self).__init__()
        # self.tanh = nn.Tanh()
        self.l1 = LinearReparam(in_features=In_Nodes,
                                out_features=Out_Nodes,
                                prior_means=np.full((Out_Nodes ,In_Nodes) ,mean),
                                prior_variances=np.full((Out_Nodes ,In_Nodes) ,variance),
                                posterior_mu_init=0.5,
                                posterior_rho_init=-3.0,
                                bias=False,
                                )

        '''
        self.l2 = LinearReparam(in_features=Hidden_Nodes,
                                out_features=Hidden_Nodes,
                                prior_means=np.full((Hidden_Nodes,Hidden_Nodes),mean),
                                prior_variances=np.full((Hidden_Nodes,Hidden_Nodes),variance),
                                posterior_mu_init=0.5,
                                posterior_rho_init=-3.0,
                                bias=False,
                                )

        self.l3 = LinearReparam(in_features=Hidden_Nodes,
                                out_features=Hidden_Nodes,
                                prior_means=np.full((Hidden_Nodes,Hidden_Nodes),mean),
                                prior_variances=np.full((Hidden_Nodes,Hidden_Nodes),variance),
                                posterior_mu_init=0.5,
                                posterior_rho_init=-3.0,
                                bias=False,
                                )
        self.l4 = LinearReparam(in_features=Hidden_Nodes,
                                out_features=Out_Nodes,
                                prior_means=np.full((Out_Nodes,Hidden_Nodes),mean),
                                prior_variances=np.full((Out_Nodes,Hidden_Nodes),variance),
                                posterior_mu_init=0.5,
                                posterior_rho_init=-3.0,
                                bias=False,
                                )
        '''
    def forward(self, x):

        lin_pred = self.l1(x)

        return lin_pred


class BDNN_Zhang(nn.Module):
  '''
    DNN from the paper
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(4, 8),
      nn.Tanh(),
      nn.Linear(8, 4),
      nn.Tanh(),
      nn.Linear(4, 1),
      nn.Sigmoid(),
      nn.Linear(1, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)

class Bay_CPASNet(nn.Module):
	def __init__(self, In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, Pathway_Mask):
		super(Bay_CPASNet, self).__init__()
		self.tanh = nn.Tanh()
		self.pathway_mask = Pathway_Mask
		###gene layer --> pathway layer
		self.sc1 = nn.Linear(In_Nodes, Pathway_Nodes)
		###pathway layer --> hidden layer
		self.sc2 = nn.Linear(Pathway_Nodes, Hidden_Nodes)
		###hidden layer --> hidden layer 2
		self.sc3 = nn.Linear(Hidden_Nodes, Out_Nodes, bias=False)
		###hidden layer 2 + age --> Cox layer
		self.sc4 = nn.Linear(Out_Nodes+1, 1, bias = False)
		self.sc4.weight.data.uniform_(-0.001, 0.001)

	def forward(self, x_1, x_2):
		###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
		self.sc1.weight.data = self.sc1.weight.data.mul(self.pathway_mask)
		x_1 = self.tanh(self.sc1(x_1))
		x_1 = self.tanh(self.sc2(x_1))
		x_1 = self.tanh(self.sc3(x_1))
		###combine age with hidden layer 2
		x_cat = torch.cat((x_1, x_2), 1)
		lin_pred = self.sc4(x_cat)

		return lin_pred