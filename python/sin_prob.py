#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/sin_prob.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 05.12.2020

## Use SEL ane LIFs to solve Fangfang's sine problem.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
exec(open("python/lib.py").read())
exec(open("python/rnn3.py").read())

# Define problem in the sel language
steps, x, y = sine_signal(seqlen=100, size=4, clock=True, tensor=False)
t_steps = len(steps)-1
every = 3
in_spikes = [[] if i % 3 != 0 else [0] for i in range(t_steps)]
target = y

np.random.seed(123)

R = 2
H = 100
P = 1
Q = 1

learn_rate = 5e-4
mb_size = 1
#epochs = 100 * mb_size
epochs = 1000 * mb_size

t_eps = 0.01
betas = np.array([2 for i in range(H)])

opt = Adam(learn_rate = learn_rate)
#opt = GD(learn_rate = learn_rate)
snn = ALifNeuralNetwork(R, H, P, Q, t_eps = 0.05, mb_size = mb_size, costfunc = mse, optimizer = opt)

costs = np.zeros(epochs)
decision = np.zeros(epochs, dtype = np.bool)
dirs = np.zeros(epochs, dtype = np.bool)
for epoch in tqdm(range(epochs)):

    # Run the snn
    ret = snn.run(t_steps = t_steps, in_spikes = in_spikes, target = target, train = True, save_states = True, save_traces = True)

    costs[epoch] = ret['cost']

plt.figure()
plt.subplot(1,2,1)
plt.plot(costs)
plt.subplot(1,2,2)
plt.plot(ret['y'].flatten(), label = 'Predictions')
plt.plot(y, label = 'Target')
plt.legend()
plt.savefig("temp.pdf")
plt.close()
