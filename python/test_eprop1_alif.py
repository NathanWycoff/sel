#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/test_eprop1_lif.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.12.2020

## Test learning with e-prop 1 on a simple problem with LIF neurons.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
exec(open("python/lib.py").read())

np.random.seed(1234)

H = 5
P = 3
Q = 1
R = 2

t_eps = 0.01
t_end = 20
t_steps = int(np.ceil(t_end/t_eps))

thresh = 1.0
refractory_period = 0.05
#refractory_period = 0.2
refractory_steps = int(np.ceil(refractory_period / t_eps))
tau_m = 0.2
tau_o = tau_m
tau_a = 2.
alpha = np.exp(-t_eps/tau_m)
kappa = np.exp(-t_eps/tau_o)
rho = np.exp(-t_eps/tau_a)
#betas = np.array([1.74e-2 if i < H/2 else 0. for i in range(H)])
#betas = np.array([1. if i < H/2 else 0. for i in range(H)])
betas = np.array([0.03 for i in range(H)])

#THETA_in = np.array([[1., 0., 0., 0.], [0., 1.0, 0., 0.0], [0, 0, 0, 1], [0, 0, 0, 1]])
mult = 0.1
#THETA_in = mult * np.array([[1., 0., 0., 1.1/mult],[0., 1., 0., 1.1/mult]])
isig = 1.0
THETA_in = np.zeros([H,P]) + np.random.normal(size=[H,P], scale = isig)
THETA_rec = np.zeros([H,H]) + np.random.normal(size=[H,H], scale = 0.001)
THETA_out = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = isig)
#THETA_rec = np.zeros([H,H])
#THETA_out = np.array([-1, 1.1]).reshape([Q, H])

# Make up some input spikes.
seq = np.linspace(t_eps * 10, t_eps * t_steps, t_steps / 10)
#spikes1 = seq[np.floor(seq) % 2 == 0]
#spikes2 = seq[np.floor(seq) % 2 != 0]
spikes1 = seq[seq <= 10.]
spikes2 = seq[seq > 10.]
# Add a random noise channel
n_spikes = len(spikes1)
spikes3 = np.sort(np.random.choice(t_steps, n_spikes, replace = False) * t_eps)
in_spikes = [[] for _ in range(t_steps+1)]
for spike in spikes1:
    in_spikes[int(spike/t_eps)] = [0]
for spike in spikes2:
    in_spikes[int(spike/t_eps)] = [1]
for spike in spikes3:
    in_spikes[int(spike/t_eps)] = [2]

# Create a random objective
#target = np.ones([t_steps]) - 2 * np.array(np.floor(np.linspace(0,t_steps*t_eps, t_steps)) % 2 == 0).astype(np.float64)
target = np.ones([t_steps]) - 2 * np.array(np.linspace(0,t_steps*t_eps,t_steps) <= 10).astype(np.float64)

gamma = 0.3 # Called gamma in the article
learn_rate = 0.1
epochs = 100

nn_params = {'thresh' : thresh, 'alpha' : alpha, 'kappa' : kappa, 'betas' : betas,  'rho' : rho,  't_eps' : t_eps, 'gamma' : gamma, 'spikes' : [], 'ref_steps' : refractory_steps, \
        'rc' : np.zeros(H).astype(np.int)}
trainable_params = {'in' : THETA_in, 'rec' : THETA_rec, 'out' : THETA_out}

def dEdy(nn, yt, target_t):

    if np.isnan(target_t):
        return 0.

    return (yt - target_t).flatten()

def L(nn, yt, target_t):
    d = nn.eprop_funcs['dEdy'](nn, yt, target_t)
    return (d * nn.trainable_params['out']).flatten()

# Give the jacobian of the state vector with respect to the trainable parameters.
# TODO: Think harder about the timing for this: we have st1 here but st in lib.py
# TODO: Think harder about naming the third param: it will sometimes be input spikes.
# First index - to neuron; second index - from neuron
def d_st_d_tp(nn, st1, zt1):
    wrt1 = np.tile(zt1, (nn.H,1))
    wrt2 = np.zeros_like(wrt1) #TODO: Is this really zero?
    #dzdv = nn.eprop_funcs['dzds'](nn, st1, zt1)[:,0]
    #wrt2 = np.diag(dzdv) @ wrt1.copy()
    ret = np.stack([wrt1, wrt2]).astype(np.float)
    ret = np.transpose(ret, [1,2,0])
    ret = np.expand_dims(ret, -1)
    return ret

# Gives the jacobian of the state vector at time t wrt the state vector at time t-1.
# Should return a tensor of shape RxRxH, element i,j,k represents the derivative of neuron state dimension i at time t wrt neuron state dimension j at time t-1 for neuron k.
def D(nn, st, st1):
    hs = (1. - np.abs(st1[0,:] - nn.net_params['thresh']) / nn.net_params['thresh'])
    hs *= nn.net_params['gamma'] 
    hs *= (hs > 0).astype(np.float)

    # Perhaps it should be the case that this matrix should only have a nonzero 1,1 entry if beta is 0? Or perhaps not?
    diag1 = np.repeat(nn.net_params['alpha'], nn.H)
    diag2 = nn.net_params['rho'] - hs * nn.net_params['betas']
    superdiag = np.zeros([nn.H])
    subdiag = hs

    ret = np.stack([diag1, superdiag, subdiag, diag2], axis = -1).reshape([nn.H,2,2])
    return ret

# This is an H by R matrix to be returned, each element tells us the pseudo-derivative of the observable state of neuron h wrt to dimension r of the hidden state.
def dzds(nn, st, zt):
    state1 = (1. - np.abs(st[0,:] - nn.net_params['thresh'])) / nn.net_params['thresh']
    state1 *= nn.net_params['gamma'] 
    state1 *= (state1 > 0).astype(np.float)
    state2 = (-1) * state1.copy() * nn.net_params['betas']
    ret = np.stack([state1, state2])
    ret = ret.T
    return ret

eprop_funcs = {'L' : L, 'd_st_d_tp' : d_st_d_tp, 'D' : D, 'dzds' : dzds, 'dEdy' : dEdy}

def f(nn, st):
    betas = nn.net_params['betas']
    thresh = nn.net_params['thresh']
    zt = (st[0,:] - betas * st[1,:]) > thresh
    our_spikes = list(np.where(zt)[0])
    nn.net_params['spikes'].append(our_spikes)
    return zt

def g(nn, st1, zt1, xt):
    # Lower refractory counters (some may go strongly negative, but this shouldn't be a problem).
    nn.net_params['rc'] -= 1

    # Initialize at previous potential.
    st = np.copy(st1)

    # Reset neurons that spiked.
    st *= (1-zt1)

    # Update refractory counter.
    nn.net_params['rc'][zt1.astype(np.bool)] = nn.net_params['ref_steps']

    # Decay Potential.
    st[0,:] = alpha * st[0,:]

    # Update adaptive excess threshold
    st[1,:] = rho * st[1,:] + zt1

    # Integrate incoming spikes
    #st += zt1 @ nn.trainable_params['rec'].T
    st[0,:] += nn.trainable_params['rec'] @ zt1

    # Integrate external stimuli
    st[0,:] += nn.trainable_params['in'] @ xt

    ## Ensure nonnegativity
    #st *= st > 0

    # Reset any neurons still in their refractory period.
    isref = (nn.net_params['rc'] > 0)
    st[0,:] *= (1. - isref)

    return st

def get_xt(nn, ip, ti):
    xt = np.zeros([nn.P])
    xt[ip['in_spikes'][ti]] = 1
    return xt


opt = Adam(learn_rate = learn_rate)
#opt = GD(learn_rate = learn_rate)
snn = NeuralNetwork(f = f, g = g, get_xt = get_xt, R = R, H = H, P = P, Q = Q, net_params = nn_params, trainable_params = trainable_params, optimizer = opt, eprop_funcs = eprop_funcs, costfunc = mse)

costs = np.zeros(epochs)

for epoch in tqdm(range(epochs)):
    in_params = {'in_spikes' : in_spikes}

    #TODO: Betterrrr
    snn.net_params['spikes'] = []
    ret = snn.run(t_steps = t_steps, ip = in_params, target = target, train = True, save_states = True, save_traces = True)
    snn.reset_states()

    costs[epoch] = ret['cost']

y = ret['y']
S = ret['S']

fig = plt.figure(figsize=[8,8])
plt.subplot(2,2,1)
plt.plot(S[0,:,:].T)
plt.title("Potential")
plt.subplot(2,2,2)
plt.plot(S[1,:,:].T)
plt.title("Adaptive Threshold")
plt.subplot(2,2,3)
plt.plot(y.flatten())
plt.title("Output")
plt.subplot(2,2,4)
plt.plot(np.log10(costs))
plt.title("Costs")
plt.savefig("temp.pdf")
plt.close()

print("Col Norms: ")
print(np.sqrt(np.sum(np.square(snn.trainable_params['in']), axis = 0)))
