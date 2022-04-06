from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from src.models.variational_layers.variational_layer import BaseVariationalLayer_


class LinearReparam(BaseVariationalLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_means,
                 prior_variances,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Linear layer with reparameterization trick.
        Inherits from bayesian_torch.layers.BaseVariationalLayer_
        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(LinearReparam, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_means = prior_means
        self.prior_variances = prior_variances
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.rho_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer(
                'eps_bias',
                torch.Tensor(out_features),
                persistent=False)
            self.register_buffer(
                'prior_bias_mu',
                torch.Tensor(out_features),
                persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_features),
                                 persistent=False)
        else:
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        self.prior_weight_mu = torch.from_numpy(self.prior_means)
        self.prior_weight_sigma = torch.from_numpy(self.prior_variances)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.mu_bias is not None:
            self.prior_bias_mu =torch.from_numpy(np.mean(self.prior_means,axis = 1))
            self.prior_bias_sigma = torch.from_numpy(np.mean(self.prior_variances,axis =1 ))
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)
    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma)
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias,
                              self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        weight = self.mu_weight + \
            (sigma_weight * self.eps_weight.data.normal_())
        if return_kl:
            kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.linear(input, weight, bias)
        if return_kl:
            if self.mu_bias is not None:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight

            return out, kl

        return out