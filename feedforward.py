import sys
import random
import math
import subprocess

import tqdm
import torch

LOG2 = math.log(2)

def identity(x):
    return x

def pairs(xs):
    return zip(xs, xs[1:])

def initialized_linear(a, b, device=None):
    linear = torch.nn.Linear(a, b, device=device)
    torch.nn.init.xavier_uniform_(linear.weight)
    linear.bias.data.fill_(0.01)
    return linear

class FeedForward(torch.nn.Module):
    """ Generate feedforward network with given structure in terms of numbers of hidden units.
    Example: FeedForward([3,4,5,2]) will yield a network with structure:
    3 inputs ->
    ReLU ->
    4 hidden units ->
    ReLU ->
    5 hidden units ->
    ReLU ->
    2 outputs 
    """
    def __init__(self, structure, activation=torch.nn.ReLU(), dropout=0, batch_norm=False, layer_norm=False, transform=None, device=None):
        super().__init__()
        self.device = device
        
        def layers():
            the_structure = list(structure)
            assert len(the_structure) >= 2
            for a, b in pairs(the_structure[:-1]):
                yield initialized_linear(a, b, device=self.device)
                if dropout:
                    yield torch.nn.Dropout(dropout)
                if layer_norm:
                    yield torch.nn.LayerNorm(b, device=self.device)
                if batch_norm:
                    yield torch.nn.BatchNorm1d(b, device=self.device)
                yield activation
            *_, penultimate, last = the_structure
            yield initialized_linear(penultimate, last, device=self.device)

        self.ff = torch.nn.Sequential(*layers())
        self.transform = identity if transform is None else transform

    def forward(self, x):
        return self.transform(self.ff(x))
                
def generate_xor_training_example(n):
    """ Generate n training examples for XOR function. """
    x1 = torch.Tensor([random.choice([0,1]) for _ in range(n)])
    x2 = torch.Tensor([random.choice([0,1]) for _ in range(n)])
    x = torch.stack([x1, x2], -1)
    y = (x1 != x2).float()
    return x,y

def epsilonify(x, eps=10**-5):
    """ Differentiably scale a value from [0,1] to [0+e, 1-e] """
    return (1-2*eps)*x + eps

def logistic(x):
    """ Differentiably squash a value from R to the interval (0,1) """
    return 1/(1+torch.exp(-x))

def logit(x):
    """ Differentiably blow up a value from the interval (0,1) to R """
    return torch.log(x) - torch.log(1-x)

def se_loss(y, yhat):
    """ Squared error loss.
    Appropriate loss for y and yhat \in R.
    Pushes yhat toward the mean of y. """
    return (y-yhat)**2

def bernoulli_loss(y, yhat):
    """ Appropriate loss for y \in {0,1}, yhat \in (0,1).
    But it's common to use this for y \in [0,1] and it still works.
    """
    return -(y*yhat.log() + (1-y)*(1-yhat).log())

def continuous_bernoulli_loss(x, lam):
    """ Appropriate loss for y \in [0,1], yhat \in (0,1).
    Technically more correct than Bernoulli loss for that case, 
    but more complex/annoying and potentially numerically unstable.
    See https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution
    """
    logZ = LOG2 + torch.log(torch.atanh(1-2*lam) / (1 - 2*lam))
    return logZ + bernoulli_loss(x, lam)

def continuous_bernoulli_mean(lam):
    """ Expectation of a Continuous Bernoulli distribution.
    See https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution 
    """
    return lam/(2*lam - 1) + 1/(2 * torch.atanh(1-2*lam))

def beta_loss(x, alpha, beta):
    unnorm = (alpha - 1)*torch.log(x) + (beta - 1)*torch.log(1-x)
    logZ = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return unnorm - logZ

def train_xor_example(batch_size=10, num_epochs=1000, print_every=100, structure=[2,3,1], **kwds):
    """ Example: Train a network to reproduce the XOR function. """
    net = FeedForward(structure)
    opt = torch.optim.Adam(params=net.parameters(), **kwds)
    for i in range(num_epochs):
        opt.zero_grad()
        x, y = generate_xor_training_example(batch_size)
        yhat = net(x).squeeze(-1)
        loss = se_loss(y, yhat).mean()
        if i % print_every == 0:
            print("epoch %d, loss = %s" % (i, str(loss.item())))
        loss.backward()
        opt.step()
    return net


