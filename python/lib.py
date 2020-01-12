#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.12.2020

from tqdm import tqdm

class NeuralNetwork(object):
    """
    A very general neural network as defined in Bellec et al. 

    f - Latent to visible function
    g - State transition function
    get_xt - Input function

    H - Number of recurrent neurons.
    P - Number of input channels.
    Q - Number of output channels.

    """
    def __init__(self, f, g, get_xt, H, P, Q, params):
        self.f = f
        self.g = g
        self.get_xt = get_xt
        self.H = H
        self.P = P
        self.Q = Q
        self.params = params

        # Initialize state to zeros.
        self.st1 = np.zeros([self.H])
        self.zt1 = f(self, self.st1)

    def run(self, t_steps, ip, save_states = False):
        """
        Run the simulation for the specified number of steps with input parameters given in the dictionary ip.

        if save_states is true, will return the values of the latent and observable states for all timesteps of the simulation.
        """

        if save_states:
            S = np.zeros([self.H, t_steps])
            Z = np.zeros([self.H, t_steps])

        st1 = self.st1
        zt1 = self.zt1
        for ti in tqdm(range(t_steps)):
            xt = self.get_xt(self, ip, ti)
            st = self.g(self, st1, zt1, xt)
            zt = self.f(self, st)

            st1 = st
            zt1 = zt

            if save_states:
                S[:,ti] = st
                Z[:,ti] = zt

        self.st1 = st1
        self.zt1 = zt1

        if save_states:
            return([S, Z])


class SpikingNeuralNetwork(NeuralNetwork):
    """
    Slightly more specific, will have weight initializers as well as the opportunity for additional parameters.

    """
    pass

class LifNeuralNetwork(SpikingNeuralNetwork):
    """
    Should be very quick to use and instantiate.

    """
    pass
