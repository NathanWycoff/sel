#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.12.2020

exec(open("./python/optim_lib.py").read())

mse = lambda x,y: np.sum(np.square(x-y))

def crossentropy(logodds, label):
    prob = sigmoid(logodds)
    if label == 1:
        return -np.log(prob)
    elif label == 0:
        return -np.log(1-prob)
    else:
        raise AttributeError("for entropic loss, the target should be either None, 0, or 1. Instead, it was %s"%label)

class NeuralNetwork(object):
    """
    A very general neural network as defined in Bellec et al. 

    f - Latent to visible function
    g - State transition function

    R - Dimension of state space for a single neuron.
    H - Number of recurrent neurons.
    P - Number of input channels.
    Q - Number of output channels.

    net_params - Additional, fixed network parameters, such as membrane time constants. Should be a dictionary of arbitrary objects.
    trainable_params - Parameters to be learned via eprop, such as connection weights. Should be a dictionary of np.arrays.
    eprop_funcs - A dictionary of functions needed to do eprop.
    costfunc - A scalar function; it's first argument is the prediction and second the truth.

    """

    #TODO: learn rate should be a full optmier.
    def __init__(self, f, g, R, H, P, Q, net_params, trainable_params, optimizer, eprop_funcs = None, update_every = 1, costfunc = mse):
        self.f = f
        self.g = g
        self.H = H
        self.P = P
        self.Q = Q
        self.R = R
        self.net_params = net_params
        self.trainable_params = trainable_params
        self.eprop_funcs = eprop_funcs
        self.optimizer = optimizer
        self.update_every = update_every
        self.n_runs = 0
        self.costfunc = costfunc

        self.grad_in = np.zeros([self.H, self.P])
        self.grad_rec = np.zeros([self.H, self.H])
        self.grad_out = np.zeros([self.Q, self.H])

        assert Q == 1

        # Initialize state to zeros.
        self.st1 = np.zeros([self.R, self.H])
        self.zt1 = f(self, self.st1)
        self.xt1 = np.zeros([self.P])

    def reset_states(self):
        """
        Reset hidden states.
        """
        # Initialize state to zeros.
        self.st1 = np.zeros([self.R, self.H])
        self.zt1 = f(self, self.st1)
        self.net_params['rc'] = np.zeros(H).astype(np.int)

    def run(self, t_steps, ip, target = None, train = False, save_states = False, save_traces = False):
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
            epst1 = np.zeros([self.H, self.H, self.R, 1])
            epst1_in = np.zeros([self.H, self.P, self.R, 1])
            filt_et = np.zeros([self.H, self.H])
            filt_et_in = np.zeros([self.H, self.P])
            dydout = np.zeros([self.Q,self.H])

        self.n_runs += 1

        if target is not None:
            cost = 0

        if save_states:
            S = np.zeros([self.R, self.H, t_steps])
            Z = np.zeros([self.H, t_steps])

        if save_traces:
            ET = np.zeros([self.H, self.H, t_steps])
            ET_in = np.zeros([self.H, self.P, t_steps])
            EPS = np.zeros([self.H, self.H, self.R, t_steps])
            EPS_in = np.zeros([self.H, self.P, self.R, t_steps])

        ys = np.zeros([self.Q, t_steps])
        yt1 = np.zeros([self.Q,1])

        st1 = self.st1
        zt1 = self.zt1
        xt1 = self.xt1
        for ti in range(t_steps):
            xt = self.ip['get_xt'](self, ip, ti)
            st = self.g(self, st1, zt1, xt1)
            zt = self.f(self, st)

            yt = self.net_params['kappa'] * yt1 + self.trainable_params['out'] @ zt
            ys[:,ti] = yt

            if (target is not None) and (not np.isnan(target[ti])):
                nc = self.costfunc(yt, target[ti])
                cost += nc

            if train:

                # TODO: There is much code reuse between these next few blocks: we should write a function. The logic is very similar as it is, but I think there's a smart way to simultaneously get rid of tiling and make the code more similar.

                ### E-prop 1 for input connectivity.
                ##TODO: The tiling is highly inefficient, I'm sure we can figure out how to do it without it.
                #Dt = self.eprop_funcs['D'](self, st, st1)
                #Dts = np.expand_dims(Dt, 1)
                #Dts = np.tile(Dts, (1,self.P,1,1))
                #dsdt = self.eprop_funcs['d_st_d_tp'](self, st, xt1)
                #epst_in = Dts @ epst1_in + dsdt
                #dzds = self.eprop_funcs['dzds'](self, st, zt)
                #dzds = np.expand_dims(dzds, 1)
                #dzds = np.expand_dims(dzds, 1)
                #dzds = np.tile(dzds, (1,self.P,1,1))
                #et_in = (dzds @ epst_in).reshape([self.H,self.P])

                ## E-prop 1 for input connectivity.
                #TODO: The tiling is highly inefficient, I'm sure we can figure out how to do it without it.
                Dt = self.eprop_funcs['D'](self, st, st1)
                Dts = np.expand_dims(Dt, 1)
                Dts = np.tile(Dts, (1,self.P,1,1))
                dsdt = self.eprop_funcs['d_st_d_tp'](self, st, xt1)
                epst_in = Dts @ epst1_in + dsdt
                dzds = self.eprop_funcs['dzds'](self, st, zt)
                dzds = np.expand_dims(dzds, 1)
                dzds = np.expand_dims(dzds, 1)
                dzds = np.tile(dzds, (1,self.P,1,1))
                et_in = (dzds @ epst_in).reshape([self.H,self.P])

                filt_et_in = kappa * filt_et_in + et_in
                
                Lt = self.eprop_funcs['L'](self, yt, target[ti])

                # The commented out line seems to be equivalent (and hence, more efficient), although less readable, as array multiplication with different dimensions is unintuitive.
                #gradt_in = filt_et_in * Lt
                gradt_in = np.diag(Lt) @ filt_et_in
                self.grad_in += gradt_in

                # E-prop 1 for Recurrent connectivity.
                #TODO: The tiling is highly inefficient, I'm sure we can figure out how to do it without it.
                Dt = self.eprop_funcs['D'](self, st, st1)
                Dts = np.expand_dims(Dt, 1)
                Dts = np.tile(Dts, (1,self.H,1,1))
                #TODO: Make sure it is indeed st and not st1 in the next line.
                dsdt = self.eprop_funcs['d_st_d_tp'](self, st, zt1)
                epst = Dts @ epst1 + dsdt
                dzds = self.eprop_funcs['dzds'](self, st, zt)
                dzds = np.expand_dims(dzds, 1)
                dzds = np.expand_dims(dzds, 1)
                dzds = np.tile(dzds, (1,self.H,1,1))
                et = (dzds @ epst).reshape([self.H,self.H])

                filt_et = kappa * filt_et + et
                
                Lt = self.eprop_funcs['L'](self, yt, target[ti])

                # The commented out line seems to be equivalent (and hence, more efficient)
                #gradt_rec = filt_et * Lt
                gradt_rec = np.diag(Lt) @ filt_et
                self.grad_rec += gradt_rec

                # Readout weights are easier.
                assert self.Q == 1
                delta = self.eprop_funcs['dEdy'](self, yt, target[ti])
                dydout = self.net_params['kappa'] * dydout + zt
                gradt_out = delta * dydout
                self.grad_out += gradt_out

                # Update vars
                epst1_in = epst_in
                epst1 = epst

                if save_traces:
                    ET[:,:,ti] = et
                    ET_in[:,:,ti] = et_in
                    EPS[:,:,:,ti] = epst.reshape([self.H,self.H,self.R])
                    EPS_in[:,:,:,ti] = epst_in.reshape([self.H,self.P,self.R])

            if save_states:
                S[:,:,ti] = st
                Z[:,ti] = zt

            st1 = st
            zt1 = zt
            yt1 = yt
            xt1 = xt

        # Just gradient descent for now, and just on recurrent connections.
        if train and (self.n_runs % self.update_every == 0):
            # We choose to disallow self-connections, though it might be interesting to explore in the future.
            self.grad_rec = self.grad_rec - np.diag(np.diag(self.grad_rec))
            grads = [self.grad_in, self.grad_rec, self.grad_out]
            self.trainable_params = self.optimizer.apply_gradients(self.trainable_params, grads)
            #self.trainable_params['in'] = self.trainable_params['in'] - self.learn_rate * self.grad_in
            #self.trainable_params['rec'] = self.trainable_params['rec'] - self.learn_rate * self.grad_rec
            #self.trainable_params['out'] = self.trainable_params['out'] - self.learn_rate * self.grad_out
            self.last_grads = [self.grad_in, self.grad_rec, self.grad_out]
            self.grad_in = np.zeros([self.H, self.P])
            self.grad_rec = np.zeros([self.H, self.H])
            self.grad_out = np.zeros([self.Q, self.H])

        self.st1 = st1
        self.zt1 = zt1

        ret = {}
        ret['y'] = ys

        if save_states:
            ret['S'] = S
            ret['Z'] = Z
        if target is not None:
            ret['cost'] = cost
        if save_traces:
            ret['ET'] = ET
            ret['ET_in'] = ET_in
            ret['EPS'] = EPS
            ret['EPS_in'] = EPS_in

        return ret


class SpikingNeuralNetwork(NeuralNetwork):
    """
    Slightly more specific, will have weight initializers as well as the opportunity for additional parameters.

    """
    pass

class ALifNeuralNetwork(SpikingNeuralNetwork):
    """
    Should be very quick to use and instantiate.

    """
    def __init__(self, R, H, P, Q, t_eps = 0.01, thresh = 1., \
            refractory_period = 0.05, tau_m = 0.02, tau_o = 0.02, \
            tau_a = 2., betas = np.array([1 for i in range(H)]), \
            gamma = 0.3, optimizer = None, isig = [1.0,0.001,1.0]):
        # Initialize weights and store model parameters
        self.refractory_steps = int(np.ceil(refractory_period / t_eps))

        THETA_in = np.zeros([H,P]) + np.random.normal(size=[H,P], scale = isig[0])
        THETA_rec = np.zeros([H,H]) + np.random.normal(size=[H,H], scale = isig[1])
        THETA_out = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = isig[2])

        alpha = np.exp(-t_eps/tau_m)
        kappa = np.exp(-t_eps/tau_o)
        rho = np.exp(-t_eps/tau_a)

        nn_params = {'thresh' : thresh, 'alpha' : alpha, 'kappa' : kappa, 'betas' : betas,  'rho' : rho,  't_eps' : t_eps, 'gamma' : gamma, 'spikes' : [], 'ref_steps' : refractory_steps, \
        'rc' : np.zeros(H).astype(np.int)}

        trainable_params = {'in' : THETA_in, 'rec' : THETA_rec, 'out' : THETA_out}

        # Define neural dynamics
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



def plot_allinone(y, inlist, hiddenlist, costs, path = "allinone.pdf"):
    """
    y is a matrix giving one output in each row over time (time is represented by columns)
    inlist a list of lists. Each sublist represents a "group" of input neurons, which will have the same color. Each sublist should contain np.arrays giving neuron firing times.
    inlist A list of arrays, giving the firing times of each neuron
    """
    fig = plt.figure(figsize=[8,8])

    plt.subplot(2,2,1)
    plt.plot(y.T)
    plt.title("Output")

    G = len(inlist)
    color = plt.cm.rainbow(np.linspace(0,1,G))
    plt.subplot(2,2,2)
    nplotted = 0

    for g in range(G):
        for gi in range(len(inlist[g])):
            nplotted += 1
            plt.eventplot(inlist[g][gi], color=color[g], linelengths = 0.5, lineoffsets=nplotted)

    plt.title("Input Spikes")

    G = len(hiddenlist)
    plt.subplot(2,2,3)
    nplotted = 0

    for g in range(G):
        nplotted += 1
        plt.eventplot(hiddenlist[g], color='blue', linelengths = 0.5, lineoffsets=nplotted)

    plt.title("Hidden Spikes")

    plt.subplot(2,2,4)
    plt.title("Cost:")
    plt.plot(costs)

    plt.savefig(path)
    plt.close()
