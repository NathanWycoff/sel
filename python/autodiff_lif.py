#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/test_eprop1_lif.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.12.2020

## Test learning with e-prop 1 on a simple problem with LIF neurons.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
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

THETA_in = tf.Variable(2*mult * np.abs(np.random.normal(size=[H,P]) * t_eps))
THETA_rec = mult * np.random.normal(size=[H,H]) * t_eps
THETA_rec -= np.diag(np.diag(THETA_rec)) # No recurrent connectivity.
THETA_rec = tf.Variable(2*THETA_rec)
THETA_out = tf.Variable(np.random.normal(size=[Q,H]))

t_steps = 2000
gamma = 0.3 # Called gamma in the article
learn_rate = 1e-6
EPOCHS = 20

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

# TODO: Disappear this section
THRESH = thresh
GAMMA = gamma

@tf.custom_gradient
def snl(V):
    """
    The threshold activation function typical in LIF models, but with a smoothed gradient.
    """
    def grad(dy):
        return dy * GAMMA * tf.keras.activations.relu(1. - tf.abs(THRESH-V)) / THRESH
    val = tf.cast(V >= THRESH, tf.float64)
    return val, grad

def f(nn, st):
    #zt = st[0,:] > nn.net_params['thresh']
    zt = snl(st[0,:])
    return zt

def g(nn, st1, zt1, xt):
    # Initialize at previous potential.
    #st = np.copy(st1)
    st = tf.identity(st1)

    # Reset neurons that spiked.
    st *= (1-zt1)

    # Decay Potential.
    st = alpha * st

    # Integrate incoming spikes
    #st += zt1 @ nn.trainable_params['rec'].T
    st += tf.transpose(nn.trainable_params['rec'] @ zt1.reshape([nn.H,1]))

    # Integrate external stimuli
    st += tf.transpose(nn.trainable_params['in'] @ xt.reshape([nn.P,1]))

    ## Ensure nonnegativity
    #st *= st > 0

    return st

def get_xt(nn, ip, ti):
    xt = np.zeros([nn.P])
    xt[ip['in_spikes'][ti]] = 1
    return xt

# Get the required gradients, one by one.
# Set up an environment
class nothing(object):
    pass
nn = nothing()
nn.trainable_params = trainable_params
nn.net_params = nn_params
nn.H = H
nn.P = P
nn.R = R

st1 = tf.Variable(np.random.normal(size=[R,H]))
st = np.random.normal(size=[R,H])
zt1 = np.random.binomial(1,0.5,size=[H])
zt = np.random.binomial(1,0.5,size=[H])
xt = np.random.binomial(1,0.5,size=[P])
yt1 = np.zeros([Q])
target = 0.5

# Derivative of hidden state with respect to itself 1 time step ago.
with tf.GradientTape(persistent = True) as gt:
    st = g(nn, st1, zt1, xt)
    zt = f(nn, st)
    yt = nn.net_params['kappa'] * yt1 + nn.trainable_params['out'] @ tf.reshape(zt, [nn.H,1])
    err = tf.square(yt - target)
Dt = gt.gradient(st, st1)
#TODO: This is a factor of H bigger than I would have expected.
dsdt_in = gt.gradient(st, THETA_in)
dsdt_rec = gt.gradient(st, THETA_rec)
dzds = gt.gradient(zt, st)
dmsedz = gt.gradient(err, zt)
del gt

#epst1_in = np.zeros([nn.H, nn.P, nn.R, 1])
#
#Dts = np.expand_dims(np.expand_dims(tf.transpose(Dt), 1), -1)
#Dts = np.tile(Dts, (1,nn.P,1,1))
#epst_in = Dts @ epst1_in + dsdt_in[:,:,np.newaxis,np.newaxis]
#dzds = tf.transpose(dzds)
#dzds = np.expand_dims(dzds, 1)
#dzds = np.expand_dims(dzds, 1)
#dzds = np.tile(dzds, (1,nn.P,1,1))
#et_in = tf.reshape((dzds @ epst_in), [nn.H,nn.P])
