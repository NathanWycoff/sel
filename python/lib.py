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

    R - Dimension of state space for a single neuron.
    H - Number of recurrent neurons.
    P - Number of input channels.
    Q - Number of output channels.

    net_params - Additional, fixed network parameters, such as membrane time constants. Should be a dictionary of arbitrary objects.
    trainable_params - Parameters to be learned via eprop, such as connection weights. Should be a dictionary of np.arrays.
    eprop_funcs - A dictionary of functions needed to do eprop.

    """

    #TODO: learn rate should be a full optmier.
    def __init__(self, f, g, get_xt, R, H, P, Q, net_params, trainable_params, eprop_funcs = None, learn_rate = 0.001):
        self.f = f
        self.g = g
        self.get_xt = get_xt
        self.H = H
        self.P = P
        self.Q = Q
        self.R = R
        self.net_params = net_params
        self.trainable_params = trainable_params
        self.eprop_funcs = eprop_funcs
        self.learn_rate = learn_rate

        assert Q == 1

        # Initialize state to zeros.
        self.st1 = np.zeros([self.R, self.H])
        self.zt1 = f(self, self.st1)

    def run(self, t_steps, ip, target = None, train = False, save_states = False):
        """
        Run the simulation for the specified number of steps with input parameters given in the dictionary ip.

        if save_states is true, will return the values of the latent and observable states for all timesteps of the simulation.

        target - An indexible collection of length t_steps; gives the target at a given time point (1D for now)
        """


        if train:
            if target is None:
                raise ValueError("Need a target to do training; unsupervised training is not available.")
            if self.eprop_funcs is None:
                raise ValueError("Need to specify eprop_funcs to do training; audo-diff is not available.")

        if train:
            epst1 = np.zeros([self.R, self.H, self.H])
            filt_et = np.zeros([self.H, self.H])
            grad_rec = np.zeros([self.H, self.H])
            grad_out = np.zeros([self.Q, self.H])
            dydout = np.zeros([self.Q,self.H])

        if target is not None:
            cost = 0

        if save_states:
            S = np.zeros([self.H, t_steps])
            Z = np.zeros([self.H, t_steps])

        ys = np.zeros([self.Q, t_steps])
        yt1 = np.zeros([self.Q,1])

        st1 = self.st1
        zt1 = self.zt1
        for ti in tqdm(range(t_steps)):
            #if ti == 20:
            #    break
            xt = self.get_xt(self, ip, ti)
            st = self.g(self, st1, zt1, xt)
            zt = self.f(self, st)

            yt = self.net_params['kappa'] * yt1 + self.trainable_params['out'] @ zt
            ys[:,ti] = yt

            if target is not None:
                cost += np.sum(np.square(yt - target[ti]))

            if train:
                # E-prop 1 for Recurrent connectivity.
                Dt = self.eprop_funcs['D'](self, st, st1)
                dsdt = self.eprop_funcs['d_st_d_tp'](self, st, zt)
                epst = Dt * (epst1 + dsdt)
                h = self.eprop_funcs['h'](self, st, zt)
                dzds = (h.T).reshape([self.H, self.R, 1])
                et = (np.transpose(epst, [2,1,0]) @ dzds).reshape([self.H, self.H])

                filt_et = kappa * filt_et + et
                
                Lt = self.eprop_funcs['L'](self, yt, target[ti])

                gradt_rec = filt_et * Lt
                grad_rec += gradt_rec

                # Readout weights are easier.
                assert self.Q == 1
                delta = -float(target[ti] - yt) 
                dydout = self.net_params['kappa'] * dydout + zt
                gradt_out = delta * dydout
                grad_out += gradt_out

            if save_states:
                S[:,ti] = st
                Z[:,ti] = zt

            st1 = st
            zt1 = zt
            yt1 = yt

        # Just gradient descent for now, and just on recurrent connections.
        grad_rec = grad_rec - np.diag(np.diag(grad_rec))
        self.trainable_params['rec'] = self.trainable_params['rec'] - self.learn_rate * grad_rec
        self.trainable_params['out'] = self.trainable_params['out'] - self.learn_rate * grad_out

        self.st1 = st1
        self.zt1 = zt1

        ret = {}
        ret['y'] = ys

        if save_states:
            ret['S'] = S
            ret['Z'] = Z
        if train:
            ret['grad_rec'] = grad_rec
        if target is not None:
            ret['cost'] = cost

        return ret


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
