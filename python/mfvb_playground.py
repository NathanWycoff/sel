#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/test_eprop1_lif.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.12.2020

## Test learning with e-prop 1 on a simple problem with LIF neurons.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#exec(open("python/lib.py").read())
exec(open("python/new_new_lib.py").read())

np.random.seed(1234)

H = 50
P = 3
Q = 1
R = 2

feed_align = True
prior_var = 1e0 * 1/H
data_var = 1e-3
#prior_var = 1e-5
#V = 100
V = 50
V = 1

#learn_rate = 0.1
#learn_rate = 5e-4
learn_rate = 1e-3
#epochs = 2000
epochs = 100

t_eps = 0.1
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
#THETA_in = np.zeros([H,P]) + np.random.normal(size=[H,P], scale = isig)
#THETA_rec = np.zeros([H,H]) + np.random.normal(size=[H,H], scale = 0.001)
#THETA_out = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = isig)

stdi = np.sqrt(2 / (H + P))
#LAMBDA_in = np.zeros([H,P]) + np.random.normal(size=[H,P], scale = isig)
#LAMBDA_rec = np.zeros([H,H]) + np.random.normal(size=[H,H], scale = 0.001)
#LAMBDA_out = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = isig)
LAMBDA_in = np.zeros([H,P]) + np.random.normal(size=[H,P], scale = stdi)
LAMBDA_rec = np.zeros([H,H]) + np.random.normal(size=[H,H], scale = stdi)
LAMBDA_out = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = stdi)

PHI_in = np.sqrt(prior_var) * np.ones([H,P]) + np.random.normal(size=[H,P], scale = 1e-4)
PHI_rec = np.sqrt(prior_var) * np.ones([H,H]) + np.random.normal(size=[H,H], scale = 1e-4)
PHI_out = np.sqrt(prior_var) * np.ones([Q,H]) + np.random.normal(size=[Q,H], scale = 1e-4)

#THETA_rec = np.zeros([H,H])
#THETA_out = np.array([-1, 1.1]).reshape([Q, H])
LAMBDA_out = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = stdi)
#RAND_MAT = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = isig)
RAND_MAT = np.zeros([Q,H,V]) + np.random.normal(size=[Q,H,V], scale = isig)

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

nn_params = {'thresh' : thresh, 'alpha' : alpha, 'kappa' : kappa, 'betas' : betas,  'rho' : rho,  't_eps' : t_eps, 'gamma' : gamma, 'spikes' : [], 'ref_steps' : refractory_steps, \
        'rc' : np.zeros(H).astype(np.int)}

def dEdy(nn, yt, target_t):

    if np.isnan(target_t):
        return 0.

    return (yt - target_t).flatten()

def L(nn, yt, target_t):
    d = nn.eprop_funcs['dEdy'](nn, yt, target_t)
    if feed_align:
        return (d * RAND_MAT[:,:,nn.RAND_ID]).flatten()
    else:
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

#TODO: The use of opt1 here is due to the fact that I am updating the weights in the snn as well as out here. A short terms solution would be to have a flag that allows us to compute gradients and a separate flag that allows us to do the updating in the snn.run call; right now "train" does both or neither of these.
opt = Adam(learn_rate = learn_rate)
opt1 = Adam(learn_rate = learn_rate)
#opt = GD(learn_rate = learn_rate)
trainable_params = {'in' : None, 'rec' : None, 'out' : None}
snn = NeuralNetwork(f = f, g = g, R = R, H = H, P = P, Q = Q, net_params = nn_params, trainable_params = trainable_params, optimizer = opt, eprop_funcs = eprop_funcs, costfunc = mse)

mean_costs = np.zeros(epochs)

in_params = {'in_spikes' : in_spikes}
in_params['get_xt'] = get_xt

for epoch in tqdm(range(epochs)):

    LAMI_grad = np.zeros_like(LAMBDA_in)
    LAMR_grad = np.zeros_like(LAMBDA_rec)
    LAMO_grad = np.zeros_like(LAMBDA_out)

    PHII_grad = np.zeros_like(LAMBDA_in)
    PHIR_grad = np.zeros_like(LAMBDA_rec)
    PHIO_grad = np.zeros_like(LAMBDA_out)

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
        ret = snn.run(t_steps = t_steps, ip = in_params, target = target, train = True, save_states = True, save_traces = True)
        snn.reset_states()

        costs[v] = ret['cost']
        ys[v,:] = ret['y']

        # Propogate and accumulate gradients to variational parameters.
        # Just variational mean for now.
        LAMI_grad += snn.last_grads[0]
        LAMR_grad += snn.last_grads[1]
        LAMO_grad += snn.last_grads[2]

        PHII_grad += snn.last_grads[0] * (THETA_in - LAMBDA_in) / PHI_in
        PHIR_grad += snn.last_grads[1] * (THETA_rec - LAMBDA_rec) / PHI_rec
        PHIO_grad += snn.last_grads[2] * (THETA_out - LAMBDA_out) / PHI_out

    #TODO: Fold in V mean

    # Fold in data variance.
    LAMI_grad *= epochs / data_var
    LAMR_grad *= epochs / data_var
    LAMO_grad *= epochs / data_var
    PHII_grad *= epochs / data_var
    PHIR_grad *= epochs / data_var
    PHIO_grad *= epochs / data_var

    # Add in prior grads
    LAMI_grad += (LAMBDA_in / np.square(PHI_in))
    LAMR_grad += (LAMBDA_rec / np.square(PHI_rec))
    LAMO_grad += (LAMBDA_out / np.square(PHI_out))
    PHII_grad += (PHI_in * (1/prior_var - 1/np.square(PHI_in)))
    PHIR_grad += (PHI_rec * (1/prior_var - 1/np.square(PHI_rec)))
    PHIO_grad += (PHI_out * (1/prior_var - 1/np.square(PHI_out)))

    # Feed the accumulated gradients to our optimizer. 
    cur_LAMBDAs = [LAMBDA_in, LAMBDA_rec, LAMBDA_out]
    cur_PHIs = [PHI_in, PHI_rec, PHI_out]
    cur_params = cur_LAMBDAs + cur_PHIs
    grads_LAMBDAs = [LAMI_grad, LAMR_grad, LAMO_grad]
    grads_PHIs = [LAMI_grad, LAMR_grad, LAMO_grad]
    grads_all = grads_LAMBDAs + grads_PHIs
    #TODO: The use of opt1 here is due to the fact that I am updating the weights in the snn as well as out here. A short terms solution would be to have a flag that allows us to compute gradients and a separate flag that allows us to do the updating in the snn.run call; right now "train" does both or neither of these.
    LAMBDA_in, LAMBDA_rec, LAMBDA_out, PHI_in, PHI_rec, PHI_out = opt1.apply_gradients(cur_params, grads_all)

    mean_costs[epoch] = np.mean(costs)

y = ret['y']
S = ret['S']

q = [0.1, 0.5, 0.9]
post_pred = np.quantile(ys, q, axis = 0)

fig = plt.figure(figsize=[8,8])
plt.subplot(2,2,1)
plt.plot(S[0,:,:].T)
plt.title("Potential")
plt.subplot(2,2,2)
plt.plot(S[1,:,:].T)
plt.title("Adaptive Threshold")
plt.subplot(2,2,3)
#plt.plot(y.flatten())
plt.plot(post_pred[0,:], c = 'orange')
plt.plot(post_pred[1,:], c = 'red')
plt.plot(post_pred[2,:], c = 'orange')
plt.title("Output")
plt.subplot(2,2,4)
plt.plot(np.log10(mean_costs))
plt.title("Costs")
plt.savefig("temp.pdf")
plt.close()

print("Col Norms: ")
print(np.sqrt(np.sum(np.square(snn.trainable_params['in']), axis = 0)))

# A new input spike sequence that it has never seen before.
seq = np.linspace(t_eps * 10, t_eps * t_steps, t_steps / 10)
n_spikes = len(spikes1)
spikes1 = spikes2 = seq
spikes3 = np.sort(np.random.choice(t_steps, n_spikes, replace = False) * t_eps)
in_spikes = [[] for _ in range(t_steps+1)]
for spike in spikes1:
    in_spikes[int(spike/t_eps)] = [0]
for spike in spikes2:
    in_spikes[int(spike/t_eps)] = [1]
for spike in spikes3:
    in_spikes[int(spike/t_eps)] = [2]

ys = np.empty([V,t_steps])
costs = np.empty(V)
for v in range(V):
    # Sample our weights from the variational distribution.
    THETA_in = np.random.normal(loc = LAMBDA_in, scale = np.abs(PHI_in))
    THETA_rec = np.random.normal(loc = LAMBDA_rec, scale = np.abs(PHI_rec))
    THETA_out = np.random.normal(loc = LAMBDA_out, scale = np.abs(PHI_out))
    snn.trainable_params = {'in' : THETA_in, 'rec' : THETA_rec, 'out' : THETA_out}

    #TODO: Improve the way states and spikes are reset, as well as the way thta the random feedback matrices are accessed.
    snn.net_params['spikes'] = []
    snn.RAND_ID = v
    ret = snn.run(t_steps = t_steps, ip = in_params, target = target, train = True, save_states = True, save_traces = True)
    snn.reset_states()

    costs[v] = ret['cost']
    ys[v,:] = ret['y']

q = [0.1, 0.5, 0.9]
post_pred = np.quantile(ys, q, axis = 0)

fig = plt.figure(figsize=[8,8])
plt.subplot(2,2,1)
plt.plot(S[0,:,:].T)
plt.title("Potential")
plt.subplot(2,2,2)
plt.plot(S[1,:,:].T)
plt.title("Adaptive Threshold")
plt.subplot(2,2,3)
#plt.plot(y.flatten())
plt.plot(post_pred[0,:], c = 'orange')
plt.plot(post_pred[1,:], c = 'red')
plt.plot(post_pred[2,:], c = 'orange')
plt.title("Output")
plt.subplot(2,2,4)
plt.plot(np.log10(mean_costs))
plt.title("Costs")
plt.savefig("new_prob.pdf")
plt.close()

