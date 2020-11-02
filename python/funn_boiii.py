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
epochs = 10 * mb_size

t_eps = 0.05
betas = np.array([2 for i in range(H)])

#TODO: Change initialization to Xavier? Or maybe not?
isig = [1.0,0.001,1.0]
LAMBDA_in = np.zeros([H,P]) + np.random.normal(size=[H,P], scale = isig[0])
LAMBDA_rec = np.zeros([H,H]) + np.random.normal(size=[H,H], scale = isig[1])
LAMBDA_out = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = isig[2])

#opt = Adam(learn_rate = learn_rate)
opt = GD(learn_rate = learn_rate)
vb_opt = GD(learn_rate = learn_rate)
snn = ALifNeuralNetwork(R, H, P, Q, t_eps = 0.05, mb_size = 1, cost = crossentropy, optimizer = opt)

costs = np.zeros(epochs)
decision = np.zeros(epochs, dtype = np.bool)
dirs = np.zeros(epochs, dtype = np.bool)
LAMI_grad = np.zeros_like(LAMBDA_in)
LAMR_grad = np.zeros_like(LAMBDA_rec)
LAMO_grad = np.zeros_like(LAMBDA_out)
for epoch in tqdm(range(epochs)):

    # Sample a new problem!
    #TODO: Seeded
    np.random.seed(epoch)
    coinflip, inlist, in_spikes, target, t_steps, cue_time = make_mouse_prob(t_eps, n_signals, signal_duration, break_duration, cue_duration, spikes_per_signal, neur_per_group, noise = False)

    # Run the snn
    THETA_in = LAMBDA_in
    THETA_rec = LAMBDA_rec
    THETA_out = LAMBDA_out
    snn.trainable_params = {'in' : THETA_in, 'rec' : THETA_rec, 'out' : THETA_out}

    ret = snn.run(t_steps = t_steps, in_spikes = in_spikes, target = target, train = True, save_states = True, save_traces = True)

    # Record mean gradient update.
    LAMI_grad += snn.last_grads[0]
    LAMR_grad += snn.last_grads[1]
    LAMO_grad += snn.last_grads[2]

    # Apply gradient update. 
    if epoch % mb_size == 0:
        # Feed the accumulated gradients to our optimizer. 
        cur_LAMBDAs = [LAMBDA_in, LAMBDA_rec, LAMBDA_out]
        grads_LAMBDAs = [LAMI_grad, LAMR_grad, LAMO_grad]
        LAMBDA_in, LAMBDA_rec, LAMBDA_out = vb_opt.apply_gradients(cur_LAMBDAs, grads_LAMBDAs)

        LAMI_grad = np.zeros_like(LAMI_grad)
        LAMR_grad = np.zeros_like(LAMR_grad)
        LAMO_grad = np.zeros_like(LAMO_grad)

    # record some results. 
    costs[epoch] = ret['cost']
    decision[epoch] = (np.mean(ret['y'][:,int(cue_time / t_eps):]) > 0)
    dirs[epoch] = coinflip

# Gets accuracy for the last tenth of runs. 
on_last = epochs//10
print("Accuracy: %f"%(np.mean(decision[-on_last:]==dirs[-on_last:])))

snn.mb_size = mb_size
plot_allinone(snn, ret, inlist, costs, path = "allinone.pdf")
snn.mb_size = 1
