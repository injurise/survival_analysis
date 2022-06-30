import torch
import torch.nn as nn

import math
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import HalfCauchy
from src.models.variational_layers.dist_var_layers import ReparametrizedGaussian, ScaleMixtureGaussian, InverseGamma
import numpy as np
from scipy.special import loggamma




class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculates kl divergence between two gaussians (Q || P)
        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P
        returns torch.Tensor of shape 0
        """
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 *
                                                              (sigma_p ** 2)) - 0.5
        return kl.mean()

class Horseshoe_params:
        def __init__(self):
            self.horseshoe_scale = None
            self.global_cauchy_scale= 1.
            self.weight_cauchy_scale= 1.
            self.beta_rho_scale= -5.
            self.log_tau_mean = None
            self.log_tau_rho_scale= -5.
            self.bias_rho_scale= -5.
            self.log_v_mean=None
            self.log_v_rho_scale= -5.


class HorseshoeLayer_out_mask(nn.Module):
    """
    Single linear layer of a horseshoe prior for regression
    """
    def __init__(self, in_features, out_features, parameters, cuda=False,mask=None):
        """
        Args:
            in_features: int, number of input features
            out_features: int, number of output features
            parameters: instance of class HorseshoeHyperparameters
            cuda: boolean
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cuda = cuda

        if mask == None:
            self.mask = torch.ones(out_features,in_features)
        else:
            self.mask = mask

        if self.cuda:
            self.mask = self.mask.cuda()

        # Scale to initialize weights, according to Yingzhen's work
        if parameters.horseshoe_scale == None:
            scale = 1. * np.sqrt(6. / (in_features + out_features))
        else:
            scale = parameters.horseshoe_scale

        # Initialization of parameters of prior distribution
        # weight parameters
        self.prior_tau_shape = torch.Tensor([0.5])

        # local shrinkage parameters
        self.prior_lambda_shape = torch.Tensor([0.5])
        self.prior_lambda_rate = torch.Tensor([1 / parameters.weight_cauchy_scale**2])

        # global shrinkage parameters
        self.prior_v_shape = torch.Tensor([0.5])
        self.prior_theta_shape = torch.Tensor([0.5])
        self.prior_theta_rate = torch.Tensor([1 / parameters.global_cauchy_scale**2])

        # Initialization of parameters of variational distribution
        # weight parameters
        self.beta_mean = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-scale, scale) * self.mask)
        self.beta_rho = nn.Parameter(torch.ones([out_features, in_features]) * parameters.beta_rho_scale * self.mask)
        self.beta = ReparametrizedGaussian(self.beta_mean, self.beta_rho,self.mask)

        # local shrinkage parameters
        self.lambda_shape = self.prior_lambda_shape * torch.ones(out_features)
        self.lambda_rate = self.prior_lambda_rate * torch.ones(out_features)
        self.lambda_ = InverseGamma(self.lambda_shape, self.lambda_rate)

        # Sample from half-Cauchy to initialize the mean of log_tau
        # We initialize the parameters using a half-Cauchy because this
        # is the prior distribution over tau
        if parameters.log_tau_mean == None:
            distr = HalfCauchy(1 / np.sqrt(self.prior_lambda_rate))
            sample = distr.sample(torch.Size([out_features])).squeeze()
            self.log_tau_mean = nn.Parameter(torch.log(sample))
        else:
            self.log_tau_mean = parameters.log_tau_mean

        self.log_tau_rho = nn.Parameter(torch.ones(out_features) * parameters.log_tau_rho_scale)
        self.log_tau = ReparametrizedGaussian(self.log_tau_mean, self.log_tau_rho)

        # bias parameters
        self.bias_mean = nn.Parameter(torch.zeros([1, out_features], ))
        self.bias_rho = nn.Parameter(torch.ones([1, out_features]) * parameters.bias_rho_scale)
        self.bias = ReparametrizedGaussian(self.bias_mean, self.bias_rho)

        # global shrinkage parameters
        self.theta_shape = self.prior_theta_shape
        self.theta_rate = self.prior_theta_rate
        self.theta = InverseGamma(self.theta_shape, self.theta_rate)

        # Sample from half-Cauchy to initialize the mean of log_v
        # We initialize the parameters using a half-Cauchy because this
        # is the prior distribution ovev
        if parameters.log_v_mean == None:
            distr = HalfCauchy(1 / np.sqrt(self.prior_theta_rate))
            sample = distr.sample()
            self.log_v_mean = nn.Parameter(torch.log(sample))
        else:
            self.log_v_mean = parameters.log_v_mean

        self.log_v_rho = nn.Parameter(torch.ones([1, 1])  * parameters.log_v_rho_scale)
        self.log_v = ReparametrizedGaussian(self.log_v_mean, self.log_v_rho)


    def log_prior(self):
        """
        Computes the expectation of the log of the prior p under the variational posterior q
        """
        def exp_log_inverse_gamma(shape, exp_rate, exp_log_rate, exp_log_x, exp_x_inverse):
            """
            Calculates the expectation of the log of an inverse gamma distribution p under
            the posterior distribution q
            E_q[log p(x | shape, rate)]
            Args:
            shape: float, the shape parameter of the gamma distribution
            exp_rate: torch tensor, the expectation of the rate parameter under q
            exp_log_rate: torch tensor, the expectation of the log of the rate parameter under q
            exp_log_x: torch tensor, the expectation of the log of the random variable under q
            exp_x_inverse: torch tensor, the expectation of the inverse of the random variable under q
            Returns:
            exp_log: torch tensor, E_q[log p(x | shape, rate)]
            """
            exp_log = - torch.lgamma(shape) + shape * exp_log_rate - (shape + 1) * exp_log_x\
                      -exp_rate * exp_x_inverse

            # We need to sum over all components since this is a vectorized implementation.
            # That is, we compute the sum over the individual expected values. For example,
            # in the horseshoe BLR model we have one local shrinkage parameter for each weight
            # and therefore one expected value for each of these shrinkage parameters.
            return torch.sum(exp_log)

        def exp_log_gaussian(mean, std, dim = False):
            """
            Calculates the expectation of the log of a Gaussian distribution p under the posterior distribution q
            E_q[log p(x)] - see note log_prior_gaussian.pdf
            Args:
            mean: torch tensor, the mean of the posterior distribution
            std: torch tensor, the standard deviation of the posterior distribution
            Returns:
            exp_gaus: torch tensor, E_q[p(x)]
            Comment about how this function is vectorized:
            Every component beta_i follows a univariate Gaussian distribution, and therefore has
            a scalar mean and a scalar variance. We can combine all components of beta into a
            diagonal Gaussian distribution, which has a mean vector of the same length as the
            beta vector, and a standard deviation vector of the same length. By summing over the
            mean vector and over the standard deviations, we therefore sum over all components of beta.
            """
            if dim==False:
                dim = mean.shape[0] * mean.shape[1]
            exp_gaus = - 0.5 * dim * (torch.log(torch.tensor(2 * math.pi))) - 0.5 * (torch.sum(mean **2) + torch.sum(std**2))
            return exp_gaus

        # Calculate E_q[ln p(\tau | \lambda)] + E[ln p(\lambda)]
        # E_q[ln p(\tau | \lambda)] for the weights
        shape = self.prior_tau_shape
        exp_lambda_inverse = self.lambda_.exp_inverse()
        exp_log_lambda = self.lambda_.exp_log()
        exp_log_tau = self.log_tau.mean
        exp_tau_inverse = torch.exp(-self.log_tau.mean + 0.5 * self.log_tau.std_dev **2)
        log_inv_gammas_weight = exp_log_inverse_gamma(shape, exp_lambda_inverse, -exp_log_lambda,
                                exp_log_tau, exp_tau_inverse)

        # E_q[ln p(\lambda)] for the weights
        shape = self.prior_lambda_shape
        rate = self.prior_lambda_rate
        log_inv_gammas_weight += exp_log_inverse_gamma(shape, rate, np.log(rate),
                                                       exp_log_lambda, exp_lambda_inverse)

        # E_q[ln p(v | \theta)] for the global shrinkage parameter
        shape = self.prior_v_shape
        exp_theta_inverse = self.theta.exp_inverse()
        exp_log_theta = self.theta.exp_log()
        exp_log_v = self.log_v.mean
        exp_v_inverse = torch.exp(-self.log_v.mean + 0.5 * self.log_v.std_dev **2)
        log_inv_gammas_global = exp_log_inverse_gamma(shape, exp_theta_inverse, -exp_log_theta,
                                exp_log_v, exp_v_inverse)

        # E_q[ln p(\theta)] for the global shrinkage parameter
        shape = self.prior_theta_shape
        rate = self.prior_theta_rate
        log_inv_gammas_global += exp_log_inverse_gamma(shape, rate, np.log(rate),
                                                       exp_log_theta, exp_theta_inverse)

        # Add all expectations
        log_inv_gammas = log_inv_gammas_weight + log_inv_gammas_global

        # E_q[N(beta)]
        log_gaussian = exp_log_gaussian(self.beta.mean, self.beta.std_dev,dim=self.beta.std_dev.count_nonzero().item())\
                       + exp_log_gaussian(self.bias.mean, self.bias.std_dev)

        return log_gaussian + log_inv_gammas

    def log_variational_posterior(self):
        """
        Computes the log of the variational posterior by computing the entropy.
        The entropy is defined as -integral[q(theta) log(q(theta))]. The log of the
        variational posterior is given by integral[q(theta) log(q(theta))].
        Therefore, we compute the entropy and return -entropy.
        Tau and v follow log-Normal distributions. The entropy of a log normal
        is the entropy of the normal distribution + the mean.
        """

        # calulate beta entropy (entropy for multivariate gaussian)
        #nonzero_idx = self.beta_rho != 0
        #b_rho_log = self.beta_rho.clone()
        #b_rho_log[nonzero_idx] = torch.log(b_rho_log[nonzero_idx])

        #entropy_part1 = (self.in_features * self.out_features) / 2 * (torch.log(torch.tensor([2 * math.pi])) + 1)
        entropy_part1 = (self.mask.count_nonzero().item()) / 2 * (torch.log(torch.tensor([2 * math.pi])) + 1)
        entropy_part2 = torch.sum(self.mask * torch.log(torch.log1p(torch.exp(self.beta_rho))))
        beta_entropy = entropy_part1 + entropy_part2

        entropy = beta_entropy\
                + self.log_tau.entropy() + torch.sum(self.log_tau.mean)\
                + self.lambda_.entropy() + self.bias.entropy()\
                + self.log_v.entropy() + torch.sum(self.log_v.mean)\
                + self.theta.entropy()

        if sum(torch.isnan(entropy)).item() != 0:
            raise Exception("entropy/log_variational_posterior computation ran into nan!")
            print('self.beta.entropy(): ', self.beta.entropy())
            print('beta mean: ', self.beta.mean)
            print('beta std: ', self.beta.std_dev)

        return -entropy

    def kl_divergence(self):
        return self.log_variational_posterior() - self.log_prior()


    def forward(self, input_, sample=True, n_samples=1):
        """
        Performs a forward pass through the layer, that is, computes
        the layer output for a given input batch.
        Args:
            input_: torch Tensor, input data to forward through the net
            sample: bool, whether to samples weights and bias
            n_samples: int, number of samples to draw from the weight and bias distribution
        """
        beta = self.beta.sample(n_samples)
        log_tau = torch.unsqueeze(self.log_tau.sample(n_samples), 2)
        log_v = torch.unsqueeze(self.log_v.sample(n_samples), 1)
        mask = torch.unsqueeze(self.mask, 0)

        weight = mask * beta * log_tau * log_v

        bias = self.bias.sample(n_samples)

        input_ = input_.expand(n_samples, -1, -1)

        if self.cuda:
            input_ = input_.cuda()
            weight = weight.cuda()
            bias = bias.cuda()

        result = torch.einsum('bij,bkj->bik', [input_, weight]) + bias
        return result

    def analytic_update(self):
        """
        Calculates analytic updates of lambda_ and theta
        Lambda and theta follow inverse Gamma distributions and can be updated
        analytically. The update equations are given in the paper in equation 9
        of the appendix: bayesiandeeplearning.org/2017/papers/42.pdf
        """
        new_shape = torch.Tensor([1])
        # new lambda rate is given by E[1/tau_i] + 1/b_0^2
        new_lambda_rate = torch.exp(-self.log_tau.mean + 0.5 * (self.log_tau.std_dev**2)) \
                          + self.prior_lambda_rate

        # new theta rate is given by E[1/v] + 1/b_g^2
        new_theta_rate = torch.exp(-self.log_v.mean + 0.5 * (self.log_v.std_dev**2)) \
                         + self.prior_theta_rate

        self.lambda_.update(new_shape, new_lambda_rate)
        self.theta.update(new_shape, new_theta_rate)
