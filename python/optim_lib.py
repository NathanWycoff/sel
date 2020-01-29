#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/optim_lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.25.2020

import numpy as np

#def sgd(x0, cf, gf, steps, stepsize):
#    """
#    x0 - Initial vector
#    cf - function which takes something like x0 and returns the scalar objective.
#    gf - function which takes something like x0 and returns the stochastic gradient; of the same shape as x0.
#    stepsize - how much to multiply gf(x) by when subtracting it from x0.
#    steps - how many times to do all that.
#    """
#    x = x0
#    costs = np.empty(steps)
#    for s in range(steps):
#        g = gf(x)
#        x = x - stepsize * g
#        costs[s] = cf(x)
#    return x, costs
#
#def adam(x0, cf, gf, steps, alpha = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
#    """
#    x0 - Initial vector
#    cf - function which takes something like x0 and returns the scalar objective.
#    gf - function which takes something like x0 and returns the stochastic gradient; of the same shape as x0.
#    stepsize - how much to multiply gf(x) by when subtracting it from x0.
#    steps - how many times to do all that.
#    """
#    x0 = x0.flatten()
#    N = len(x0)
#
#    # Initialize moment vectors
#    m = np.zeros(N)
#    v = np.zeros(N)
#
#    x = x0
#    costs = np.empty(steps)
#    for s in range(steps):
#        g = gf(x).flatten()
#        m = beta1 * m + (1 - beta1) * g
#        v = beta2 * v + (1-beta2) * np.square(g)
#
#        mhat = m / (1-beta1)
#        vhat = v / (1-beta2)
#
#        x = x - alpha * mhat / (np.sqrt(vhat) + eps)
#        costs[s] = cf(x)
#    return x, costs

class Adam(object):
    """
    An implementation of the Adam optimizer, as described in https://arxiv.org/pdf/1412.6980.pdf

    learn_rate - The step size.
    beta1 - Momentum parameter for first moment exponential moving average.
    beta2 - Momentum parameter for raw second moment exponential moving average.
    eps - Softens division by the raw second moment; if eps = 0 we will get a divide by 0 error in the event that we get a zero gradient first. 
    """
    def __init__(self, learn_rate = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-6):
        self.learn_rate = learn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def apply_gradients(self, params, dirs):
        """
        params - A list or dictionary of parameters to update
        dirs - A list of descent directions to move along. This should be of the same length as params, and each entry should be of the same size as the corresponding params entry.
        """

        # Deal with dictionaries.
        if isinstance(params, dict):
            isdict = True
            names = params.keys()
            params = list(params.values())
        else:
            isdict = False

        # Get shape of input.
        pshapes = [p.shape for p in params]
        dshapes = [d.shape for d in dirs]
        assert pshapes == dshapes
        shapes = pshapes

        # Convert to just a vector.
        x = np.concatenate([p.flatten() for p in params])
        g = np.concatenate([d.flatten() for d in dirs])

        if not hasattr(self, 'm'):
            self.m = np.zeros(x.shape)
            self.v = np.zeros(x.shape)

        # Update moment vectors
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1-self.beta2) * np.square(g)

        mhat = self.m / (1-self.beta1)
        vhat = self.v / (1-self.beta2)

        # Update solution vector.
        x = x - self.learn_rate * mhat / (np.sqrt(vhat) + self.eps)

        # Reshape to input shape.
        sprod = [0] + list(np.cumsum([np.prod(s) for s in shapes]))
        ret = [x[sprod[i]:sprod[i+1]].reshape(shapes[i]) for i in range(len(shapes))]

        if isdict:
            ret = dict(zip(names, ret))

        return ret

class GD(object):
    """
    Implements a simple gradient descent algorithm.

    alpha - The step size.
    beta1 - Momentum parameter for first moment exponential moving average.
    beta2 - Momentum parameter for raw second moment exponential moving average.
    eps - Softens division by the raw second moment; if eps = 0 we will get a divide by 0 error in the event that we get a zero gradient first. 
    """
    def __init__(self, learn_rate = 1e-3):
        self.learn_rate = learn_rate

    def apply_gradients(self, params, dirs):
        """
        For now, only params may be a dictionary; dirs should just be a list.
        params - A list of parameters to update
        dirs - A list of descent directions to move along. This should be of the same length as params, and each entry should be of the same size as the corresponding params entry.
        """
        # Deal with dictionaries.
        if isinstance(params, dict):
            isdict = True
            names = params.keys()
            params = list(params.values())
        else:
            isdict = False

        # Get shape of input.
        pshapes = [p.shape for p in params]
        dshapes = [d.shape for d in dirs]
        assert pshapes == dshapes
        shapes = pshapes

        # Convert to just a vector.
        x = np.concatenate([p.flatten() for p in params])
        g = np.concatenate([d.flatten() for d in dirs])

        # Update solution vector.
        x = x - self.learn_rate * g

        # Reshape to input shape.
        sprod = [0] + list(np.cumsum([np.prod(s) for s in shapes]))
        ret = [x[sprod[i]:sprod[i+1]].reshape(shapes[i]) for i in range(len(shapes))]

        if isdict:
            ret = dict(zip(names, ret))

        return ret
