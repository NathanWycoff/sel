#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/neurmorphic_turing.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.04.2020

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
exec(open("python/lib.py").read())
exec(open("python/mouse_lib.py").read())

link = norm.cdf
delink = norm.ppf

np.random.seed(1234)

# For the  mouse problem, groups of neurons fire to indicate the task parameters. We specify their properties here.
signal_duration = 0.05
#signal_duration = 0.1
break_duration = 0.15
cue_duration = 0.15
spikes_per_signal = 4
neur_per_group = 1
n_signals = 11

R = 2
Hlif = 5
Hmem = 2
H = Hlif + Hmem
P = 4*neur_per_group
Q = 1
threshold = 0.9# I think my link function assumes this is 1.
mthresh = 0.5

t_eps = 0.05

tau_m = 0.02
tau_o = tau_m
alpha = np.exp(-t_eps/tau_m)
kappa = np.exp(-t_eps/tau_o)

THETA_in = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,0,0], [0,0,0,0]])
THETA_write = 0.5 * np.array([[1., -1., 0., 0., 0.], [-1., 1., 0., 0., 0.]])
THETA_readin = np.array([[0., 0., 0.4, 0., 0.], [0., 0., 0.4, 0., 0.]])
THETA_readout = np.array([[0., 0.], [0., 0.], [0., 0.], [1.0, 0.], [0., 1.0]])
THETA_rec = np.zeros([Hlif,Hlif])
THETA_out = np.array([0., 0., 0., -1.0, 1.0])

coinflip, inlist, in_spikes, target, t_steps, cue_time = make_mouse_prob(t_eps, n_signals, signal_duration, break_duration, cue_duration, spikes_per_signal, neur_per_group)

def get_xt(ti):
    xt = np.zeros([P])
    xt[in_spikes[ti]] = 1
    return xt

# Init state storage
V = np.zeros([t_steps+1, Hlif])
U = np.zeros([t_steps+1, Hmem])
A = np.zeros([t_steps+1, Hmem]) + mthresh

# Record spikes.
spikes = [[] for _ in range(H)]

ys = np.zeros(t_steps+1)

xt1 = np.zeros(P)
zl1 = np.zeros(Hlif)
zm1 = np.zeros(Hmem)
for ti in range(t_steps):
    xt = get_xt(ti)

    # Update LIF neurons
    V[ti+1,:] = (1-zl1) * alpha * V[ti,:] + THETA_in @ xt1 + THETA_rec @ zl1 + THETA_readout @ zm1

    # Update memory neurons
    U[ti+1,:] = (1-zm1) * alpha * U[ti,:] + THETA_readin @ zl1
    A[ti+1,:] = link(delink(A[ti,:]) + THETA_write @ zl1) 

    # Record spiking.
    zl = (V[ti+1,:] >= threshold)
    zm = U[ti+1,:] >= A[ti+1,:] 

    if np.sum(zl) > 0:
        for i in range(Hlif):
            if zl[i]:
                spikes[i].append(ti)

    if np.sum(zm) > 0:
        for i in range(Hlif,H):
            if zm[i-Hlif]:
                spikes[i].append(ti)

    ys[ti+1] = kappa * ys[ti] + THETA_out @ zl

    xt1 = xt
    zl1 = zl
    zm1 = zm

fig = plt.figure()

fig = plt.subplot(2,3,1)
plt.plot(V)
plt.title("LIF Potentials")

fig = plt.subplot(2,3,2)
plt.plot(U)
plt.title("mLIF Potentials")

fig = plt.subplot(2,3,3)
plt.plot(A)
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
