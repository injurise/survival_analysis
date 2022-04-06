import torch
import torch.nn as nn
import torch.distributions as distributions


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


def bnn_linear_layer_cust(params, d):
    # Get BNN layer
    bnn_layer = LinearReparam(
        in_features=d.in_features,
        out_features=d.out_features,
        prior_mean=params["prior_mu"],
        prior_variance=params["prior_sigma"],
        posterior_mu_init=params["posterior_mu_init"],
        posterior_rho_init=params["posterior_rho_init"],
        bias=d.bias is not None,
    )
    bnn_layer.dnn_to_bnn_flag = True
    return bnn_layer


def dnn_to_bnn_bcoxpas(m, bnn_prior_parameters):
    for name, value in list(m._modules.items()):
        if m._modules[name]._modules:
            dnn_to_bnn_bcoxpas(m._modules[name], bnn_prior_parameters)
        if name == "sc1":
            pass
        elif "Linear" in m._modules[name].__class__.__name__:
            setattr(
                m,
                name,
                bnn_linear_layer_cust(
                    bnn_prior_parameters,
                    m._modules[name]))
        else:
            pass
    return