#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/mouse_adapt_bd.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.14.2019

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
exec(open("python/lib.py").read())
exec(open("python/mouse_lib.py").read())

np.random.seed(123)

# For the  mouse problem, groups of neurons fire to indicate the task parameters. We specify their properties here.
signal_duration = 0.05
#signal_duration = 0.1
break_duration = 0.05
cue_duration = 0.15
spikes_per_signal = 1
neur_per_group = 10
n_signals = 1

R = 2
H = 100
P = 4*neur_per_group
Q = 1

t_eps = 0.05

thresh = 1.0
refractory_period = 0.05
#refractory_period = 0.2
refractory_steps = int(np.ceil(refractory_period / t_eps))
tau_m = 0.02
tau_o = tau_m
tau_a = 2.
alpha = np.exp(-t_eps/tau_m)
kappa = np.exp(-t_eps/tau_o)
rho = np.exp(-t_eps/tau_a)
#betas = np.array([1.74e-2 if i < H/2 else 0. for i in range(H)])
#betas = np.array([1. if i < H/2 else 0. for i in range(H)])
betas = np.array([2 for i in range(H)])

#THETA_in = np.array([[1., 0., 0., 0.], [0., 1.0, 0., 0.0], [0, 0, 0, 1], [0, 0, 0, 1]])
mult = 0.1
#THETA_in = mult * np.array([[1., 0., 0., 1.1/mult],[0., 1., 0., 1.1/mult]])
isig = 1.0
THETA_in = np.zeros([H,P]) + np.random.normal(size=[H,P], scale = isig)
THETA_rec = np.zeros([H,H]) + np.random.normal(size=[H,H], scale = 0.001)
THETA_out = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = isig)
#THETA_rec = np.zeros([H,H])
#THETA_out = np.array([-1, 1.1]).reshape([Q, H])

#mult = 5.0
##THETA_in = mult / np.sqrt(P) * np.abs()
##THETA_in = 2*np.ones([H,P]) + np.random.normal(size=[H,P], scale = 0.1)
#THETA_rec = np.sqrt(H) * np.random.normal(size=[H,H]) 
#THETA_rec -= np.diag(np.diag(THETA_rec)) # No recurrent connectivity.
#THETA_out = np.random.normal(size=[Q,H])

gamma = 0.3 # Called gamma in the article
mb_size = 100
#learn_rate = 5e-3 / mb_size
# This kind of normalization is not needed for ADAM.
learn_rate = 5e-3
epochs = 200 * mb_size

nn_params = {'thresh' : thresh, 'alpha' : alpha, 'kappa' : kappa, 'betas' : betas,  'rho' : rho,  't_eps' : t_eps, 'gamma' : gamma, 'spikes' : [], 'ref_steps' : refractory_steps, \
        'rc' : np.zeros(H).astype(np.int)}
trainable_params = {'in' : THETA_in, 'rec' : THETA_rec, 'out' : THETA_out}

#def L(nn, yt, target_t):
#    if np.isnan(target_t) :
#        return np.zeros(nn.H)
#    else:
#        return (nn.trainable_params['out'] * (yt - target_t)).flatten()

sigmoid = lambda x: 1. / (1. + np.exp(-x))
def dEdy(nn, yt, target_t):

    if np.isnan(target_t):
        return 0.

    prob = sigmoid(yt)
    if target_t == 1:
        #return ((prob-1.) * nn.trainable_params['out']).flatten()
        return (prob-1.).flatten()
    elif target_t == 0:
        #return (prob * nn.trainable_params['out']).flatten()
        return (prob).flatten()
    else:
        raise AttributeError("for entropic loss, the target should be either None, 0, or 1. Instead, it was %s"%target_t)

def L(nn, yt, target_t):
    d = nn.eprop_funcs['dEdy'](nn, yt, target_t)
    return (d * nn.trainable_params['out']).flatten()

def crossentropy(logodds, label):
    prob = sigmoid(logodds)
    if label == 1:
        return -np.log(prob)
    elif label == 0:
        return -np.log(1-prob)
    else:
        raise AttributeError("for entropic loss, the target should be either None, 0, or 1. Instead, it was %s"%label)

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


#opt = Adam(learn_rate = learn_rate)
opt = GD(learn_rate = learn_rate)
snn = NeuralNetwork(f = f, g = g, get_xt = get_xt, R = R, H = H, P = P, Q = Q, net_params = nn_params, trainable_params = trainable_params, optimizer = opt, eprop_funcs = eprop_funcs, update_every = mb_size, costfunc = crossentropy)

costs = np.zeros(epochs)
decision = np.zeros(epochs, dtype = np.bool)
dirs = np.zeros(epochs, dtype = np.bool)

for epoch in tqdm(range(epochs)):

    ### Sample a new problem!
    coinflip, inlist, in_spikes, target, t_steps, cue_time = make_mouse_prob(t_eps, n_signals, signal_duration, break_duration, cue_duration, spikes_per_signal, neur_per_group)
    dirs[epoch] = coinflip

    in_params = {'in_spikes' : in_spikes}

    #TODO: Betterrrr
    snn.net_params['spikes'] = []
    ret = snn.run(t_steps = t_steps, ip = in_params, target = target, train = True, save_states = True, save_traces = True)
    snn.reset_states()

    costs[epoch] = ret['cost']
    decision[epoch] = (np.mean(ret['y'][:,int(cue_time / t_eps):]) > 0)

    on_last = 100
    acc = np.mean(decision[-on_last:]==dirs[-on_last:])
    #if acc > 0.95:
    #    print("Finished with %s cues"%n_signals)

y = ret['y']
S = ret['S']

fig = plt.figure(figsize=[8,8])

plt.subplot(3,3,1)
plt.plot(S[0,:,:].T)
plt.title("Potential")

plt.subplot(3,3,2)
plt.plot(S[1,:,:].T)
plt.title("Adaptive Threshold")

plt.subplot(3,3,3)
n_plot = min([H*H,1000])
toplot = np.random.choice(H*H,n_plot,replace = False)
plt.plot(ret['EPS'][:,:,1,:].reshape([H*H,t_steps]).T[:,toplot])
plt.title("Slow Recurrent ET Component.")

plt.subplot(3,3,4)
n_plot = min([H*H,1000])
toplot = np.random.choice(H*H,n_plot,replace = False)
plt.plot(ret['EPS'][:,:,0,:].reshape([H*H,t_steps]).T[:,toplot])
plt.title("Fast Recurrent ET Component.")

plt.subplot(3,3,5)
n_plot = min([H*P,1000])
toplot = np.random.choice(H*P,n_plot,replace = False)
plt.plot(ret['EPS_in'][:,:,1,:].reshape([H*P,t_steps]).T[:,toplot])
plt.title("Slow Input ET Component.")

plt.subplot(3,3,6)
n_plot = min([H*P,1000])
toplot = np.random.choice(H*P,n_plot,replace = False)
plt.plot(ret['EPS_in'][:,:,0,:].reshape([H*P,t_steps]).T[:,toplot])
plt.title("Fast Input ET Component.")

plt.subplot(3,3,7)
plt.plot(y.flatten())
plt.title("Output")

plt.subplot(3,3,8)
batch_costs = np.mean(costs.reshape(-1,mb_size), axis = 1)
if np.min(batch_costs) > 0:
    plt.plot(np.log10(batch_costs))
else:
    plt.plot(batch_costs)
plt.title("Costs")
plt.savefig("temp.png")
plt.close()


on_last = epochs//10
print("Accuracy: %f"%(np.mean(decision[-on_last:]==dirs[-on_last:])))

N = 100
ma = np.convolve(costs, np.ones((N,))/N, mode='valid')
fig = plt.figure()
plt.plot(ma)
plt.savefig("costs_ma.pdf")
plt.close()

spike_times = [[] for _ in range(H)]
for t in range(t_steps):
    for neur in snn.net_params['spikes'][t]:
        spike_times[neur].append(t)

plot_allinone(y, inlist, spike_times, costs, path = "allinone.pdf")
