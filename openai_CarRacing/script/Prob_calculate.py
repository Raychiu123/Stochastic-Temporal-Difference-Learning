import torch
import numpy as np

LOG2PI = np.log(2.0 * np.pi)

def kl_div_gaussian(q_mu, q_logvar, p_mu, p_logvar):
    '''Batched KL divergence D(q||p) computation.'''
    kl= 0.5*torch.sum(((p_mu - q_mu)/torch.exp(p_logvar))**2, dim = -1) + \
        torch.sum(p_logvar, dim = -1) - torch.sum(q_logvar, dim = -1) + \
        0.5*torch.sum((torch.exp(q_logvar)/torch.exp(p_logvar))**2, dim = -1) - 1/2
    return kl


def gaussian_log_prob(mu, logvar, x):
    '''Batched log probability log p(x) computation.'''
    logprob = torch.sum(-0.5*((x-mu)/torch.exp(logvar))**2 - \
            0.5*mu.new_tensor(LOG2PI) - logvar, dim = -1)
    return logprob