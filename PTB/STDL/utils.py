import math
import pickle
import torch
import numpy as np

LOG2PI = np.log(2.0 * np.pi)
criterion = torch.nn.NLLLoss(size_average=False, ignore_index=0)

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def logistic_anneal(step, a=0.0025, x0=2500):
    return 1. / (1. + math.exp(-a * (step - x0)))

def linear_anneal(step, x0, initial=0.01):
    return min(1., initial + step / x0)

def kl_div_gaussian(q_mu, q_logvar, p_mu, p_logvar):
    '''Batched KL divergence D(q||p) computation.'''
    kl= 0.5*torch.sum(((p_mu - q_mu)/torch.exp(p_logvar))**2, dim = -1) + \
        torch.sum(p_logvar, dim = -1) - torch.sum(q_logvar, dim = -1) + \
        0.5*torch.sum((torch.exp(q_logvar)/torch.exp(p_logvar))**2, dim = -1) - 0.5
    return kl

def gaussian_log_prob(mu, logvar, x):
    '''Batched log probability log p(x) computation.'''
    logprob = torch.sum(-0.5*((x-mu)/torch.exp(logvar))**2 - \
            0.5*mu.new_tensor(LOG2PI) - logvar, dim = -1)
    return logprob

def bce_loss(x, x_prob):
	loss = torch.sum(x*torch.log(x_prob) + (1-x)*torch.log(1-x_prob), -1)
	return loss

def NLL(logp, target):
    #target = target[:, :torch.max(length).item()].contiguous().view(-1)
    #print("function:target", target.size())
    logp = logp.view(-1, logp.size(-1))
    #print("function logp", logp.size())
    return criterion(logp, target)