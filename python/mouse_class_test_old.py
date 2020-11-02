#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/mouse_class_test.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.23.2020

## How does my network do when given an aberrant observation?

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
exec(open("python/lib.py").read())
exec(open("python/mouse_lib.py").read())

np.random.seed(123)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
BIGGEST_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGEST_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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

feed_align = False

# Variational Inference Params
prior_var = 1e-0 * 1/H
#prior_var = 0
data_var = 1e-3
V = 1000
#V = 1

# Optimizer 
learn_rate = 5e-4
mb_size = 100
#epochs = 10 * mb_size
epochs = 100 * mb_size
t_eps = 0.05

#VP = np.load('./data/mvfb_class_varpar2.npz')
VP = np.load('./data/mvfb_class_varpar3.npz')

LAMBDA_in = VP['arr_0']
LAMBDA_rec = VP['arr_1']
LAMBDA_out = VP['arr_2']
PHI_in = VP['arr_3']
PHI_rec = VP['arr_4']
PHI_out = VP['arr_5']

###### Create a new problem. 
# We're going to get a down problem and an up problem and merge them.
#TODO: Should we only take half of the signal ones?
np.random.seed(123)
coinflip, inlist, in_spikes, target, t_steps, cue_time = make_mouse_prob(t_eps, n_signals, signal_duration, break_duration, cue_duration, spikes_per_signal, neur_per_group, noise = False)
# NOTE: Flipped channels for illustration
inlist[0] = inlist[1]
inlist[1] = []
#inlist.pop(3)

# NOTE: mb_size is set to 1 on purpose; otherwise last_gradients won't be updated every iter. 
snn = ALifNeuralNetwork(R, H, P, Q, t_eps = t_eps, mb_size = 1, cost = crossentropy)

ys = np.empty([V,t_steps])
costs = np.empty(V)
for v in range(V):
    # Sample our weights from the variational distribution.
    THETA_in = np.random.normal(loc = LAMBDA_in, scale = np.abs(PHI_in))
    THETA_rec = np.random.normal(loc = LAMBDA_rec, scale = np.abs(PHI_rec))
    THETA_out = np.random.normal(loc = LAMBDA_out, scale = np.abs(PHI_out))
    snn.trainable_params = {'in' : THETA_in, 'rec' : THETA_rec, 'out' : THETA_out}

    #TODO: Improve the way states and spikes are reset, as well as RAND_ID
    snn.net_params['spikes'] = []
    snn.RAND_ID = v
    ret = snn.run(t_steps = t_steps, in_spikes = in_spikes, target = target, train = False, save_states = True, save_traces = True)
    snn.reset_states()

    costs[v] = ret['cost']
    ys[v,:] = ret['y']

#NOTE: What if we averaged after the zero comparison? May be more robust. 
ind_decs = np.mean(ys[:,int(cue_time / t_eps):], axis = 1) > 0
est_prob = np.mean(ind_decs)

q = [0.1, 0.5, 0.9]
post_pred = np.quantile(ys, q, axis = 0)

#TODO: Better chart to show increasing accuracy would be to plot the probabilities given for each of the two solutions over time, and we could watch them diverge to 0 and 1. 
fig = plt.figure(figsize=[8,8])
#plt.subplot(2,2,3)
plt.plot(post_pred[0,:], c = 'orange')
plt.plot(post_pred[1,:], c = 'red')
plt.plot(post_pred[2,:], c = 'orange')
plt.title("Output")
plt.savefig("temp_old.png")
plt.close()

#TODO: Larger mb_size on this problem?
snn.mb_size = mb_size
mean_costs = np.zeros(epochs // mb_size)
plot_allinone(snn, ret, inlist, mean_costs, path = "aio_old.png")
snn.mb_size = 1

MY_TICKS = np.linspace(t_eps,t_steps * t_eps, t_steps)

plt.figure(figsize=[5,2.5])
plt.subplot(1,2,2)
for v in range(10):
    plt.plot(MY_TICKS, ys[v,:], color = 'blue', alpha = 0.25)
plt.plot(MY_TICKS, post_pred[0,:], c = 'orange')
plt.plot(MY_TICKS, post_pred[1,:], c = 'red')
plt.plot(MY_TICKS, post_pred[2,:], c = 'orange')
plt.title("SNN Output")
plt.xlabel("Time (s)")
plt.ylabel("Log-Odds")
plt.subplot(1,2,1)
G = len(inlist)
color = plt.cm.rainbow(np.linspace(0,1,G))
nplotted = 0
for g in range(G):
    for gi in range(len(inlist[g])):
        nplotted += 1
        plt.eventplot(inlist[g][gi], color=color[g], linelengths = 0.5, lineoffsets=nplotted, linewidths = 10)
plt.ylim(0,31)
plt.title("Input Channels")
plt.xlabel("Time (s)")
plt.subplots_adjust(bottom = 0.2, hspace = 10.)
plt.savefig("pred_insamp.pdf")
plt.close()
