#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/neurmorphic_turing.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.04.2020

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
import tensorflow as tf
exec(open("python/lib.py").read())
exec(open("python/mouse_lib.py").read())

link = tf.math.sigmoid
delink = lambda x: tf.math.log(x / (1-x))

np.random.seed(1234)

# For the  mouse problem, groups of neurons fire to indicate the task parameters. We specify their properties here.
signal_duration = 0.05
#signal_duration = 0.1
break_duration = 0.15
cue_duration = 0.15
spikes_per_signal = 4
neur_per_group = 1
n_signals = 1
GAMMA = 0.3

epochs = 20
learn_rate = 1e-6

R = 2
Hlif = 5
Hmem = 2
H = Hlif + Hmem
P = 4*neur_per_group
Q = 1
threshold = 0.9# I think my link function assumes this is 1.
mithresh = 0.5

t_eps = 0.05

tau_m = 0.02
tau_o = tau_m
alpha = np.exp(-t_eps/tau_m)
kappa = np.exp(-t_eps/tau_o)

#THETA_in = tf.Variable(np.array([[1.,0,0,0], [0,1.,0,0], [0,0,0,1], [0,0,0,0], [0,0,0,0]]))
#THETA_write = tf.Variable(0.5 * np.array([[1., -1., 0., 0., 0.], [-1., 1., 0., 0., 0.]]))
#THETA_readin = tf.Variable(np.array([[0., 0., 0.4, 0., 0.], [0., 0., 0.4, 0., 0.]]))
#THETA_readout = tf.Variable(np.array([[0., 0.], [0., 0.], [0., 0.], [1.0, 0.], [0., 1.0]]))
#THETA_rec = tf.Variable(np.zeros([Hlif,Hlif]))
#THETA_out = tf.Variable(np.array([0., 0., 0., -1.0, 1.0]).reshape([Q,Hlif]))

THETA_in = tf.Variable(np.random.normal(size=[Hlif,P]))
THETA_write = tf.Variable(np.random.normal(size=[Hmem,Hlif]))
THETA_readin = tf.Variable(np.random.normal(size=[Hmem,Hlif]))
THETA_readout = tf.Variable(np.random.normal(size=[Hlif,Hmem]))
THETA_rec = tf.Variable(np.random.normal(size=[Hlif,Hlif]))
THETA_out = tf.Variable(np.random.normal(size=[Q,Hlif]))

trainable_params = [THETA_in, THETA_write, THETA_readin, THETA_readout, THETA_rec, THETA_out]

coinflip, inlist, in_spikes, target, t_steps, cue_time = make_mouse_prob(t_eps, n_signals, signal_duration, break_duration, cue_duration, spikes_per_signal, neur_per_group)

def get_xt(ti):
    xt = np.zeros([P,1])
    xt[in_spikes[ti],0] = 1
    return xt

# Init state storage
V = tf.Variable(np.zeros([Hlif, 1]))
U = tf.Variable(np.zeros([Hmem, 1]))
A = tf.Variable(np.zeros([Hmem, 1]) + mithresh)
V_last = tf.Variable(np.zeros([Hlif, 1]))
U_last = tf.Variable(np.zeros([Hmem, 1]))
A_last = tf.Variable(np.zeros([Hmem, 1]) + mithresh)

V_trace = np.zeros([t_steps+1, Hlif])
U_trace = np.zeros([t_steps+1, Hmem])
A_trace = np.zeros([t_steps+1, Hmem])

# Record spikes.
spikes = [[] for _ in range(H)]
ys = np.zeros(t_steps+1)

@tf.custom_gradient
def snl(V):
    """
    The thresholdold activation function typical in LIF models, but with a smoothed gradient.
    """
    def grad(dy):
        return dy * GAMMA * tf.keras.activations.relu(1. - tf.abs(threshold-V)) / threshold
    val = tf.cast(V >= threshold, tf.float64)
    return val, grad

opt = tf.keras.optimizers.Adam(learning_rate = learn_rate)

costs = np.empty(epochs)
for epoch in tqdm(range(epochs)):
    with tf.GradientTape() as gt:
        mse = 0
        xt1 = tf.Variable(np.zeros([P,1]))
        zl1 = tf.Variable(np.zeros([Hlif,1]))
        zm1 = tf.Variable(np.zeros([Hmem,1]))
        for ti in range(t_steps):
            xt = get_xt(ti)

            # Update LIF neurons
            V = (1-zl1) * alpha * V_last + tf.matmul(THETA_in, xt1) + tf.matmul(THETA_rec, zl1) + tf.matmul(THETA_readout, zm1)

            # Update memory neurons
            U = (1-zm1) * alpha * U_last + tf.matmul(THETA_readin, zl1)
            A = link(delink(A_last) + tf.matmul(THETA_write, zl1))

            # Record spiking.
            zl = snl(V)
            #TODO: not sure that the gradient will work for this.
            zm = snl(U - A + threshold)

            if np.sum(zl) > 0:
                for i in range(Hlif):
                    if zl[i]:
                        spikes[i].append(ti)

            if np.sum(zm) > 0:
                for i in range(Hlif,H):
                    if zm[i-Hlif]:
                        spikes[i].append(ti)

            new_y = kappa * ys[ti] + tf.matmul(THETA_out, zl)
            ys[ti+1] = new_y

            if not np.isnan(target[ti]):
                mse += tf.square(target[ti] - new_y)

            xt1 = xt
            zl1 = zl
            zm1 = zm

            V_last = V
            U_last = U
            A_last = A

            V_trace[ti+1,:] = V.numpy().flatten()
            U_trace[ti+1,:] = U.numpy().flatten()
            A_trace[ti+1,:] = A.numpy().flatten()
    grad = gt.gradient(mse, trainable_params)

    grads_n_vars = [(grad[i], trainable_params[i]) for i in range(len(trainable_params))]
    opt.apply_gradients(grads_n_vars)

    costs[epoch] = mse.numpy()

fig = plt.figure()

fig = plt.subplot(2,3,1)
plt.plot(V_trace)
plt.title("LIF Potentials")

fig = plt.subplot(2,3,2)
plt.plot(U_trace)
plt.title("mLIF Potentials")

fig = plt.subplot(2,3,3)
plt.plot(A_trace)
plt.title("mLIF Thresholds")

fig = plt.subplot(2,3,4)
plt.eventplot(spikes)

fig = plt.subplot(2,3,5)
inspikes = [np.array(x).flatten() for x in inlist]
plt.eventplot(inspikes, colors = ['red', 'orange', 'blue', 'blue'])

fig = plt.subplot(2,3,6)
plt.plot(ys)
plt.title("Output")

plt.savefig("turing.pdf")
plt.close()
