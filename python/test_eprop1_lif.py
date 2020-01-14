#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/test_eprop1_lif.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.12.2020

## Test learning with e-prop 1 on a simple problem with LIF neurons.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
exec(open("python/lib.py").read())

np.random.seed(123)

H = 5
P = 2
Q = 1
R = 1
mult = 100. / np.sqrt(H)

thresh = 1.0
t_eps = 0.01
alpha = np.exp(-t_eps)
kappa = np.exp(-t_eps)

THETA_in = mult * np.abs(np.random.normal(size=[H,P]) * t_eps)
THETA_rec = mult * np.random.normal(size=[H,H]) * t_eps
THETA_rec -= np.diag(np.diag(THETA_rec)) # No recurrent connectivity.
THETA_out = np.random.normal(size=[Q,H])
THETA_out_rand = np.random.normal(size=[Q,H]) #Random feedback weights, see how that works.

t_steps = 2000
gamma = 0.3 # Called gamma in the article
learn_rate = 1e-6
EPOCHS = 10

# Make up some input spikes.
seq = np.linspace(t_eps * 10, t_eps * t_steps, t_steps / 10)
#spikes1 = seq[np.floor(seq) % 2 == 0]
#spikes2 = seq[np.floor(seq) % 2 != 0]
spikes1 = seq[seq <= 10.]
spikes2 = seq[seq > 10.]
in_spikes = [[] for _ in range(t_steps+1)]
for spike in spikes1:
    in_spikes[int(spike/t_eps)] = [0]
for spike in spikes2:
    in_spikes[int(spike/t_eps)] = [1]

# Create a random objective
#target = np.ones([t_steps]) - 2 * np.array(np.floor(np.linspace(0,t_steps*t_eps, t_steps)) % 2 == 0).astype(np.float64)
target = np.ones([t_steps]) - 2 * np.array(np.linspace(0,t_steps*t_eps,t_steps) <= 10).astype(np.float64)

inlist = [spikes1, spikes2]

nn_params = {'thresh' : thresh, 'alpha' : alpha, 'kappa' : kappa, 't_eps' : t_eps, 'gamma' : gamma}
trainable_params = {'in' : THETA_in, 'rec' : THETA_rec, 'out' : THETA_out}
in_params = {'in_spikes' : in_spikes}

def L(nn, yt, target):
    return nn.trainable_params['out'] * (yt - target)

# Give the jacobian of the state vector with respect to the trainable parameters.
# TODO: Think harder about the timing for this: we have st1 here but st in lib.py
def d_st_d_tp(nn, st1, zt1):
    return np.tile(zt1, (nn.H,1)).reshape([1,nn.H,nn.H])

# Gives the jacobian of the state vector at time t wrt the state vector at time t-1.
# Should return a tensor of shape RxRxH, element i,j,k represents the derivative of neuron state dimension i at time t wrt neuron state dimension j at time t-1 for neuron k.
def D(nn, st, st1):
    return np.repeat(nn.net_params['alpha'], nn.H).reshape([1,1,nn.H])

# This is an H by R matrix to be returned, each element tells us the pseudo-derivative of the observable state of neuron h wrt to dimension r of the hidden state.
def h(nn, st, zt):
    diff = 1. - np.abs((st - nn.net_params['thresh']) / nn.net_params['thresh'])
    diff *= nn.net_params['gamma'] 
    diff *= (diff > 0).astype(np.float)
    return diff

eprop_funcs = {'L' : L, 'd_st_d_tp' : d_st_d_tp, 'D' : D, 'h' : h}

def f(nn, st):
    zt = st[0,:] > nn.net_params['thresh']
    return zt

def g(nn, st1, zt1, xt):
    # Initialize at previous potential.
    st = np.copy(st1)

    # Reset neurons that spiked.
    st *= (1-zt1)

    # Decay Potential.
    st = alpha * st

    # Integrate incoming spikes
    #st += zt1 @ nn.trainable_params['rec'].T
    st += nn.trainable_params['rec'] @ zt1

    # Integrate external stimuli
    st += nn.trainable_params['in'] @ xt

    ## Ensure nonnegativity
    #st *= st > 0

    return st

def get_xt(nn, ip, ti):
    xt = np.zeros([nn.P])
    xt[ip['in_spikes'][ti]] = 1
    return xt

snn = NeuralNetwork(f = f, g = g, get_xt = get_xt, R = R, H = H, P = P, Q = Q, net_params = nn_params, trainable_params = trainable_params, eprop_funcs = eprop_funcs, learn_rate = learn_rate)

costs = np.empty([EPOCHS])
for epoch in range(EPOCHS):
    ret = snn.run(t_steps = t_steps, ip = in_params, target = target, train = True, save_states = True)
    costs[epoch] = ret['cost']

y = ret['y']
S = ret['S']

fig = plt.figure()
plt.subplot(3,1,1)
plt.plot(S.T)
plt.subplot(3,1,2)
plt.plot(y.flatten())
plt.subplot(3,1,3)
plt.plot(costs)
plt.savefig("temp.pdf")
plt.close()

#costs = np.zeros(EPOCHS)
#for epoch in tqdm(range(EPOCHS)):
#    GRADS_rec = np.zeros([H,H])
#    GRADS_out = np.zeros([Q,H])
#
#    # Forward Simulation!
#    cost = 0
#    for t in range(t_steps):
#        # Decay Potential
#        Vs[:,t+1] = alpha * Vs[:,t]
#
#        # Integrate incoming spikes
#        Vs[:,t+1] += np.sum(THETA_rec[:,spikes[t]], axis = 1)
#
#        # Integrate external stimuli
#        Vs[:,t+1] += np.sum(THETA_in[:,in_spikes[t]], axis = 1)
#
#        # Add outgoing spikes to the stack and reset potentials as necessary.
#        spikevec = Vs[:,t+1] >= THRESH
#        spikes[t+1] = list(np.where(spikevec)[0])
#        Vs[:,t+1] *= (1-spikevec)
#
#        # Record network output
#        y[:,t+1] = np.sum(THETA_out[:,spikes[t]]) + kappa * y[:,t]
#
#        # Record eligibility trace
#        #TODO: Is this definitely t+1 on the next line?
#        h = ETA * np.fmax(0, 1 - np.abs(Vs[:,t+1] - THRESH) / THRESH)
#        zhat[:,t+1] = alpha * zhat[:,t] + spikevec
#        #elig[:,t+1] = h * zhat[:,t+1]
#        
#        # Compute gradient contribution...
#        # ...For hidden layers
#        #TODO: Is this definitely t and not t-1 in a few lines?
#        #TODO: Assumes Q = 1
#        delta = y[:,t+1] - target[t]
#        cost += np.square(delta)
#        Ls = delta * THETA_out_rand[0,:]
#        GRADS_rec += np.outer(Ls * h, zhat[:,t])
#
#        # ...For output read weights
#        #TODO: Is this definitely t and not t-1 in a few lines?
#        #TODO: Assumes Q = 1
#        dydg[:,t+1] = alpha * dydg[:,t] + spikevec.astype(np.float64)
#        GRADS_out[0,:] += delta * dydg[:,t+1]
#
#    # Apply grads
#    THETA_rec = THETA_rec - LEARN_RATE * GRADS_rec
#    THETA_out = THETA_out - LEARN_RATE * GRADS_out
#
#    costs[epoch] = cost
#
#
## Check cost over time
#fig = plt.figure()
#plt.plot(np.log(costs[:epoch]))
#plt.savefig("cost.pdf")
#
#plot_res(y, inlist)
#
#print(max([len(x) for x in spikes]))
