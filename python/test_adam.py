#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/adam.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.25.2020

import numpy as np
import matplotlib.pyplot as plt
exec(open("./python/optim_lib.py").read())

np.random.seed(123)

Ns = [[2,2], [3,4]]
N = np.sum([np.prod(ni) for ni in Ns])
steps = 100
stepsize = 1e-3

L = np.random.normal(size=[N,N])
A = L.T @ L

sigma = 1e-1

obj = lambda x: np.square(np.sum(L @ x))
def grad(x):
    g = 2 * A @ x + np.random.normal(scale = sigma, size=x.shape)
    gs = [g[:(np.prod(Ns[0]))].reshape(Ns[0]), g[(np.prod(Ns[0])):].reshape(Ns[1])]
    return gs

x0 = np.random.normal(size=[N,1])

#opt = Adam(learn_rate = stepsize)
opt = GD(learn_rate = stepsize)

costs = np.empty(steps)
x = x0
xl = [x[:(np.prod(Ns[0]))].reshape(Ns[0]), x[(np.prod(Ns[0])):].reshape(Ns[1])]
xs = {'a' : xl[0], 'b' : xl[1]}
for step in range(steps):
    gs = grad(x)
    xs = opt.apply_gradients(xs, gs)
    x = np.concatenate([xi.flatten() for xi in xs.values()])
    costs[step] = obj(x)
    print(gs)

fig = plt.figure()
plt.plot(costs)
plt.savefig("optemp.pdf")
plt.close()
