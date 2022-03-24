import numpy as np
import torch
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
import src.models.loss_functions as lf
from tqdm import tqdm


def train_step_bnn_cph(cph_bnn, x, t, e, optimizer, batchsize=256):
    n = x.shape[0]

    batches = (n // batchsize) + 1

    epoch_loss = 0

    for i in range(batches):
        xb = x[i * batchsize:(i + 1) * batchsize]
        tb = t[i * batchsize:(i + 1) * batchsize]
        eb = e[i * batchsize:(i + 1) * batchsize]

        # Training Step

        torch.enable_grad()
        optimizer.zero_grad()

        output = cph_bnn(xb)
        kl = get_kl_loss(cph_bnn)
        ce_loss = lf.partial_ll_loss(output, tb, eb)
        loss = ce_loss + kl / batchsize

        loss.backward()
        optimizer.step()

        epoch_loss += float(loss)
    step_loss = epoch_loss / n
    return step_loss


def train_cph_bnn(cph_bnn, train_data, epochs=50,
                  patience=3, batchsize=256, lr=1e-3, debug=False,
                  random_state=0, return_losses=False):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    xt, tt, et = train_data

    optimizer = torch.optim.Adam(cph_bnn.parameters(), lr=lr)

    valc = np.inf
    patience_ = 0

    losses = []

    for epoch in tqdm(range(epochs)):

        _ = train_step_bnn_cph(cph_bnn, xt, tt, et, optimizer, batchsize)

        # test_step_still has to be implemented
        # valcn = test_step(model, xv, tv_, ev_)
        valcn = _

        losses.append(valcn)

        if valcn > valc:
            patience_ += 1
        else:
            patience_ = 0

        if patience_ == patience:
            break

        valc = valcn

    if return_losses:
        return (cph_bnn), losses
    else:
        return (cph_bnn)
