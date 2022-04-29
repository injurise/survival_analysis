from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.autograd import Variable
from torch.nn.modules import utils
from src.models.variational_layers.variational_layer import BaseVariationalLayer_


class LinearReparam(BaseVariationalLayer_):
    #Adjusted Linear Layer Class to make gaussian parameters flexibel per weight
    # and to be able to initialise posterior
    def __init__(self,
                 in_features,
                 out_features,
                 prior_means,
                 prior_variances,
                 posterior_mu_init,
                 posterior_rho_init,
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

        for row in range(self.mu_weight.shape[0]):
            self.mu_weight.detach()[row] = torch.normal(mean=torch.from_numpy(
                self.posterior_mu_init[0][row]), std=0.1, generator=None, out=None)
            self.rho_weight.detach()[row] = torch.normal(mean=torch.from_numpy(
                self.posterior_rho_init[0][row]), std=0.1, generator=None, out=None)

        if self.mu_bias is not None:
            self.prior_bias_mu =torch.from_numpy(np.mean(self.prior_means,axis = 1))
            self.prior_bias_sigma = torch.from_numpy(np.mean(self.prior_variances,axis =1 ))
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)
    def kl_divergence(self):
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




"""
Variational Dropout version of linear and convolutional layers


Karen Ullrich, Christos Louizos, Oct 2017
"""
def reparametrize(mu, logvar, cuda=False, sampling=True):
    if sampling:
        std = logvar.mul(0.5).exp_()
        if cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return mu + eps * std
    else:
        return mu


# -------------------------------------------------------
# LINEAR LAYER
# -------------------------------------------------------

class LinearGroupNJ_Pathways(Module):
    """Fully Connected Group Normal-Jeffrey's layer (aka Group Variational Dropout).
    References:
    [1] Kingma, Diederik P., Tim Salimans, and Max Welling. "Variational dropout and the local reparameterization trick." NIPS (2015).
    [2] Molchanov, Dmitry, Arsenii Ashukha, and Dmitry Vetrov. "Variational Dropout Sparsifies Deep Neural Networks." ICML (2017).
    [3] Louizos, Christos, Karen Ullrich, and Max Welling. "Bayesian Compression for Deep Learning." NIPS (2017).
    """

    def __init__(self, in_features, out_features, mask,cuda=False, init_weight=None, init_bias=None, clip_var=None):

        super(LinearGroupNJ_Pathways, self).__init__()
        self.cuda = cuda
        self.mask = mask
        if cuda:
            self.device = "cuda"
            self.mask =self.mask.cuda()
        else:
            self.device = 'cpu'

        self.in_features = in_features
        self.out_features = out_features

        self.clip_var = clip_var
        self.deterministic = False  # flag is used for compressed inference
        # trainable params according to Eq.(6)
        # dropout params
        self.z_mu = Parameter(torch.empty((1,out_features),device = self.device))
        self.z_logvar = Parameter(torch.empty((1,out_features),device = self.device))  # = z_mu^2 * alpha
        # weight params
        self.weight_mu = Parameter(torch.empty((out_features, in_features),device = self.device))
        self.weight_logvar = Parameter(torch.empty((out_features, in_features),device = self.device))

        self.bias_mu = Parameter(torch.empty((1,out_features),device = self.device))
        self.bias_logvar = Parameter(torch.empty((1,out_features),device = self.device))

        # init params either random or with pretrained net
        self.reset_parameters(init_weight, init_bias)

        # activations for kl
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # numerical stability param
        self.epsilon = 1e-8



    def reset_parameters(self, init_weight, init_bias):
        # init means
        stdv = 1. / math.sqrt(self.weight_mu.size(1))

        self.z_mu.data.normal_(1, 1e-2)

        if init_weight is not None:
            self.weight_mu.data = torch.Tensor(init_weight)
        else:
            self.weight_mu.data.normal_(0, stdv)
            if self.cuda:
                self.weight_mu.data = self.weight_mu * self.mask.cuda()
            else:
                self.weight_mu.data =  self.weight_mu* self.mask

        if init_bias is not None:
            self.bias_mu.data = torch.Tensor(init_bias)
        else:
            self.bias_mu.data.fill_(0)

        # init logvars
        self.z_logvar.data.normal_(-9, 1e-2)
        self.weight_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)

    def clip_variances(self):
        if self.clip_var:
            self.weight_logvar.data.clamp_(max=math.log(self.clip_var))
            self.bias_logvar.data.clamp_(max=math.log(self.clip_var))

    def get_log_dropout_rates(self):
        log_alpha = self.z_logvar - torch.log(self.z_mu.pow(2) + self.epsilon)
        return log_alpha

    def compute_posterior_params(self):
        # not adjusted to new pathway groups yet!
        weight_var, z_var = self.weight_logvar.exp(), self.z_logvar.exp()
        self.post_weight_var = self.z_mu.pow(2) * weight_var + z_var * self.weight_mu.pow(2) + z_var * weight_var
        self.post_weight_mu = self.weight_mu * self.z_mu
        return self.post_weight_mu, self.post_weight_var

    def forward(self, x):
        if self.deterministic:
            assert self.training == False, "Flag deterministic is True. This should not be used in training."
            return F.linear(x, self.post_weight_mu, self.bias_mu)

        batch_size = x.size()[0]

        #Setting means to 0 according to pathways mask
        mu_weights_path = self.weight_mu * self.mask
        weight_mu_activations = F.linear(x,mu_weights_path,self.bias_mu)

        logvar_weights_path = self.weight_logvar.exp() * self.mask
        var_activations = F.linear(x.pow(2), logvar_weights_path,self.bias_logvar.exp()) # pow(2) comes from the local rep trick

        # compute z
        # note that we reparametrise according to [2] Eq. (11) (not [1])
        z = reparametrize(self.z_mu.repeat(batch_size, 1), self.z_logvar.repeat(batch_size, 1), sampling=self.training,
                          cuda=self.cuda)

        # apply local reparametrisation trick see [1] Eq. (6)
        # to the parametrisation given in [3] Eq. (6)
        return reparametrize(weight_mu_activations *z , (var_activations * z.pow(2)).log(),
                             sampling=self.training, cuda=self.cuda)

    def kl_divergence(self):
        # KL(q(z)||p(z))
        # we use the kl divergence approximation given by [2] Eq.(14)
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        log_alpha = self.get_log_dropout_rates()
        KLD = -torch.sum(k1 * self.sigmoid(k2 + k3 * log_alpha) - 0.5 * self.softplus(-log_alpha) - k1)

        # KL(q(w|z)||p(w|z))
        # we use the kl divergence given by [3] Eq.(8)
        KLD_element = self.mask * (-0.5 * self.weight_logvar + 0.5 * (self.weight_logvar.exp() + self.weight_mu.pow(2)) - 0.5)
        KLD += torch.sum(KLD_element)

        # KL bias
        KLD_element = -0.5 * self.bias_logvar + 0.5 * (self.bias_logvar.exp() + self.bias_mu.pow(2)) - 0.5
        KLD += torch.sum(KLD_element)

        return KLD

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


