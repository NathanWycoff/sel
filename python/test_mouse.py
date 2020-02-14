#!/usr/bin/env python4
# -*- coding: utf-8 -*-
#  python/test_mouse.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.14.2020

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
exec(open("python/lib.py").read())
exec(open("python/mouse_lib.py").read())

np.random.seed(123)

# For the  mouse problem, groups of neurons fire to indicate the task parameters. We specify their properties here.
signal_duration = 0.05
break_duration = 0.05
cue_duration = 0.15
spikes_per_signal = 1
neur_per_group = 10
n_signals = 1

R = 2
H = 100
P = 4*neur_per_group
Q = 1

learn_rate = 5e-3
mb_size = 100
epochs = 100 * mb_size

t_eps = 0.05
betas = np.array([2 for i in range(H)])

#opt = Adam(learn_rate = learn_rate)
opt = GD(learn_rate = learn_rate)
snn = ALifNeuralNetwork(R, H, P, Q, t_eps = 0.05, mb_size = mb_size, cost = crossentropy, optimizer = opt)

costs = np.zeros(epochs)
decision = np.zeros(epochs, dtype = np.bool)
dirs = np.zeros(epochs, dtype = np.bool)
for epoch in tqdm(range(epochs)):

    # Sample a new problem!
    coinflip, inlist, in_spikes, target, t_steps, cue_time = make_mouse_prob(t_eps, n_signals, signal_duration, break_duration, cue_duration, spikes_per_signal, neur_per_group, noise = False)

    # Run the snn
    ret = snn.run(t_steps = t_steps, in_spikes = in_spikes, target = target, train = True, save_states = True, save_traces = True)

    # record some results. 
    costs[epoch] = ret['cost']
    decision[epoch] = (np.mean(ret['y'][:,int(cue_time / t_eps):]) > 0)
    dirs[epoch] = coinflip

# Gets accuracy for the last tenth of runs. 
on_last = epochs//10
print("Accuracy: %f"%(np.mean(decision[-on_last:]==dirs[-on_last:])))

plot_allinone(snn, ret, inlist, costs, path = "allinone.pdf")
