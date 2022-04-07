import numpy as np
import torch

def partial_ll_loss(lrisks, tb, eb, eps=1e-3):

    tb = tb + eps*np.random.random(len(tb))
    sindex = np.argsort(-tb)

    tb = tb[sindex]
    eb = eb[sindex]

    lrisks = lrisks[sindex]
    lrisksdenom = torch.logcumsumexp(lrisks, dim = 0)

    plls = lrisks - lrisksdenom
    pll = plls[eb == 1]

    pll = torch.sum(pll)

    return -pll