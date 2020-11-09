#!/usr/bin/env python4
# -*- coding: utf-8 -*-
#  python/test_mouse.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.14.2020

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
exec(open("python/lib.py").read())

np.random.seed(123)

# For the  mouse problem, groups of neurons fire to indicate the task parameters. We specify their properties here.
signal_duration = 0.05
break_duration = 0.05
cue_duration = 0.15
spikes_per_signal = 1
neur_per_group = 10
n_signals = 1

R = 2
H = 40
#P = 100
P = 2
Q = 1
N = 100 # Keep even
NN = 100

feed_align = False
#feed_align = True
t_steps = 50
burn_in = t_steps//2

# Variational Inference Params
prior_var = 1e0 * 1/H
#prior_var = 0
data_var = 1e-2#Should be 1 when doing classification.
#data_var = 1e-0#Should be 1 when doing classification.
#V = 10
V = 2
V_test = 10

# Optimizer 
#learn_rate = 5e-4
#learn_rate = 0#5e-5
learn_rate = 1e-3
#mb_size = 50
mb_size = 50
#epochs = 500 * mb_size
epochs = 100 * mb_size
# Keep multiple of mb_size
#epochs = 100 * mb_size

#epochs = 5000 * mb_size

## Make problem
#X_vec = np.concatenate([np.random.uniform(low = -1, high = -0.75, size = N//2),  np.random.uniform(low = 0.75, high = 1, size = N//2)])
#x_val = np.random.uniform(low = -1, high = 1, size = N)

x_val = np.random.uniform(low = -1, high = 1, size = N)
X_vec = np.copy(x_val)
X_vec = np.stack([X_vec, np.ones(N)], axis = 1)

#x_test = np.random.uniform(low = -10, high = 10, size = NN)
x_test = np.random.uniform(low = -3, high = 3, size = N)
X_test = np.copy(x_test)
X_test = np.stack([X_test, np.ones(NN)], axis = 1)

#x_val  = np.random.uniform(low = 0, high = 1, size = N)
#x_val = np.random.uniform(low = 0.1, high = 10, size = N)
#X_vec = np.stack([np.sin(np.ceil((p+0.1)/2)*2*np.pi*x_val) if p % 2 == 0 else  np.cos(np.ceil((p+0.1)/2)*2*np.pi*x_val) for p in range(P-1)], axis = 1)
#X_vec = np.append(X_vec, np.ones([N,1]), axis = 1)

#y_vec = np.repeat(0, N)
#y_vec = (x_val > 0)
a = 10
y_vec = np.random.uniform(size=N) < (np.exp(a*x_val) / (np.exp(a*x_val)+1))
#y_vec = np.sqrt(np.cos(2*x_val + np.pi) + 1.2) 
#y_vec -= np.mean(y_vec)

t_eps = 0.05
betas = np.array([2 for i in range(H)])

#TODO: No Xavier? Also, make sure it aligns for FA. 
sf = 1
LAMBDA_in = np.zeros([H,P]) + np.abs(np.random.normal(size=[H,P], scale = sf*np.sqrt(2/P)))
LAMBDA_rec = np.zeros([H,H]) + np.random.normal(size=[H,H], scale = sf*np.sqrt(2 / H))
LAMBDA_out = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = sf*np.sqrt(2/H))

#isig = [1.0,0.001,1.0]
#isig = [1.0,0.1,1.0]
#LAMBDA_in = np.zeros([H,P]) + np.random.normal(size=[H,P], scale = isig[0])
#LAMBDA_rec = np.zeros([H,H]) + np.random.normal(size=[H,H], scale = isig[1])
#LAMBDA_out = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = isig[2])

#phi_std = 1e-8
phi_std = prior_var
PHI_in = np.sqrt(prior_var) * np.ones([H,P]) + np.random.normal(size=[H,P], scale = phi_std)
PHI_rec = np.sqrt(prior_var) * np.ones([H,H]) + np.random.normal(size=[H,H], scale = phi_std)
PHI_out = np.sqrt(prior_var) * np.ones([Q,H]) + np.random.normal(size=[Q,H], scale = phi_std)

#RAND_MAT = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = isig)
#RAND_MAT = np.zeros([Q,H,V]) + np.random.normal(size=[Q,H,V], scale = stdi)

isig = 1.
RAND_MAT = np.zeros([Q,H,V]) + np.random.normal(size=[Q,H,V], scale = isig)
#isig = 1.
#RAND_MAT = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = isig)

#opt = Adam(learn_rate = learn_rate)
opt = GD(learn_rate = learn_rate)
# NOTE: mb_size is set to 1 on purpose; otherwise last_gradients won't be updated every iter. 
snn = ALifNeuralNetwork(R, H, P, Q, t_eps = t_eps, mb_size = 1, costfunc = crossentropy, optimizer = opt, thresh = 1)

# Rewrite the L function so that it avoids weight transport using FA.
def fa_L(nn, yt, target_t):
    d = nn.eprop_funcs['dEdy'](nn, yt, target_t)
    if feed_align:
        return (d * RAND_MAT[:,:,nn.RAND_ID]).flatten()
        #return (d * RAND_MAT).flatten()
    else:
        return (d * nn.trainable_params['out']).flatten()

# Overwrite existing L function
snn.eprop_funcs['L'] = fa_L

# Want our own optimizer that is not interferred with anywhere else.
#vb_opt = Adam(learn_rate = learn_rate)
vb_opt = GD(learn_rate = learn_rate)

guess = np.empty(epochs, dtype = float)
truths = np.empty(epochs, dtype = float)
mean_costs = np.zeros(epochs)
mb_costs = np.zeros(epochs//mb_size)
est_probs = np.zeros(epochs) - 1.
LAMI_grad = np.zeros_like(LAMBDA_in)
LAMR_grad = np.zeros_like(LAMBDA_rec)
LAMO_grad = np.zeros_like(LAMBDA_out)

PHII_grad = np.zeros_like(LAMBDA_in)
PHIR_grad = np.zeros_like(LAMBDA_rec)
PHIO_grad = np.zeros_like(LAMBDA_out)
for epoch in tqdm(range(epochs)):
    n_of_epoch = np.random.choice(N)

    y_out = np.empty([V,t_steps])
    costs = np.empty(V)

    # Define inputs for this datum
    def get_xt(self, ip, ti):
        xt = X_vec[n_of_epoch].reshape([P])
        return xt
    target = np.repeat(y_vec[n_of_epoch], t_steps)

    in_params = {'get_xt' : get_xt}

    for v in range(V):
        # Sample our weights from the variational distribution.
        THETA_in = np.random.normal(loc = LAMBDA_in, scale = np.abs(PHI_in))
        THETA_rec = np.random.normal(loc = LAMBDA_rec, scale = np.abs(PHI_rec))
        THETA_out = np.random.normal(loc = LAMBDA_out, scale = np.abs(PHI_out))
        snn.trainable_params = {'in' : THETA_in, 'rec' : THETA_rec, 'out' : THETA_out}

        #TODO: Improve the way states and spikes are reset, as well as RAND_ID
        snn.net_params['spikes'] = []
        snn.RAND_ID = v

        ret = snn.run(t_steps = t_steps, in_params = in_params, in_spikes = None, target = target, train = True, save_states = True, save_traces = True)

        #ret = snn.run(t_steps = t_steps, in_spikes = in_spikes, target = target, train = True, save_states = True, save_traces = True)
        snn.reset_states()

        costs[v] = ret['cost']
        y_out[v,:] = ret['y']

        # Propogate and accumulate gradients to variational parameters.
        # Just variational mean for now.
        LAMI_grad += snn.last_grads[0] / V
        LAMR_grad += snn.last_grads[1] / V
        LAMO_grad += snn.last_grads[2] / V

        PHII_grad += (snn.last_grads[0] * (THETA_in - LAMBDA_in) / PHI_in) / V
        PHIR_grad += (snn.last_grads[1] * (THETA_rec - LAMBDA_rec) / PHI_rec) / V
        PHIO_grad += (snn.last_grads[2] * (THETA_out - LAMBDA_out) / PHI_out) / V

    #NOTE: What if we averaged after the zero comparison? May be more robust. 
    #ind_decs = np.mean(y_out[:,int(cue_time / t_eps):], axis = 1) > 0
    ev_y_out = np.mean(y_out, axis = 0)
    e_y_out = np.mean(ev_y_out[burn_in:])

    guess[epoch] = e_y_out
    truths[epoch] = target[-1]

    ## Fold in data variance.
    #TODO: If we re-use this block, we may need to account for mb_size. 
    #LAMI_grad *= epochs / data_var
    #LAMR_grad *= epochs / data_var
    #LAMO_grad *= epochs / data_var
    #PHII_grad *= epochs / data_var
    #PHIR_grad *= epochs / data_var
    #PHIO_grad *= epochs / data_var

    # Add in prior grads
    if (epoch+1) % mb_size == 0:
        # Because this code is being ran epoch / mb_size times, we need to account for that
        const = (N / mb_size) / data_var
        constinv = 1/const
        LAMI_grad += constinv * (LAMBDA_in / np.square(PHI_in))
        LAMR_grad += constinv * (LAMBDA_rec / np.square(PHI_rec))
        LAMO_grad += constinv * (LAMBDA_out / np.square(PHI_out))
        PHII_grad += constinv * (PHI_in * (1/prior_var - 1/np.square(PHI_in)))
        PHIR_grad += constinv * (PHI_rec * (1/prior_var - 1/np.square(PHI_rec)))
        PHIO_grad += constinv * (PHI_out * (1/prior_var - 1/np.square(PHI_out)))

        # Feed the accumulated gradients to our optimizer. 
        cur_LAMBDAs = [LAMBDA_in, LAMBDA_rec, LAMBDA_out]
        cur_PHIs = [PHI_in, PHI_rec, PHI_out]
        cur_params = cur_LAMBDAs + cur_PHIs
        grads_LAMBDAs = [LAMI_grad, LAMR_grad, LAMO_grad]
        grads_PHIs = [LAMI_grad, LAMR_grad, LAMO_grad]
        grads_all = grads_LAMBDAs + grads_PHIs
        #TODO: The use of vb_opt here is due to the fact that I am updating the weights in the snn as well as out here. A short terms solution would be to have a flag that allows us to compute gradients and a separate flag that allows us to do the updating in the snn.run call; right now "train" does both or neither of these.
        LAMBDA_in, LAMBDA_rec, LAMBDA_out, PHI_in, PHI_rec, PHI_out = vb_opt.apply_gradients(cur_params, grads_all)
        #LAMBDA_in, LAMBDA_rec, LAMBDA_out = vb_opt.apply_gradients(cur_LAMBDAs, grads_LAMBDAs)
        LAMI_grad = np.zeros_like(LAMBDA_in)
        LAMR_grad = np.zeros_like(LAMBDA_rec)
        LAMO_grad = np.zeros_like(LAMBDA_out)

        PHII_grad = np.zeros_like(LAMBDA_in)
        PHIR_grad = np.zeros_like(LAMBDA_rec)
        PHIO_grad = np.zeros_like(LAMBDA_out)

        mb_costs[epoch // mb_size] = np.mean(mean_costs[(epoch-mb_size+1):(epoch+1)])

    mean_costs[epoch] = np.mean(costs)

model_preds = np.empty(N)
n_of_epoch = np.random.choice(N)

y_out = np.empty([V,t_steps])
costs = np.empty(V)

THETA_in_draws = [np.random.normal(loc = LAMBDA_in, scale = np.abs(PHI_in)) for _ in range(V_test)]
THETA_rec_draws  = [np.random.normal(loc = LAMBDA_rec, scale = np.abs(PHI_rec)) for _ in range(V_test)]
THETA_out_draws  = [np.random.normal(loc = LAMBDA_out, scale = np.abs(PHI_out)) for _ in range(V_test)]

yv = np.empty([V_test,NN])
for n in tqdm(range(NN)):
    # Define inputs for this datum
    def get_xt(self, ip, ti):
        xt = X_test[n].reshape([P])
        return xt
    target = np.repeat(y_vec[n_of_epoch], t_steps)

    in_params = {'get_xt' : get_xt}

    for v in range(V_test):
        # Sample our weights from the variational distribution.
        THETA_in = THETA_in_draws[v]
        THETA_rec = THETA_rec_draws[v]
        THETA_out = THETA_out_draws[v]
        snn.trainable_params = {'in' : THETA_in, 'rec' : THETA_rec, 'out' : THETA_out}

        #TODO: Improve the way states and spikes are reset, as well as RAND_ID
        snn.net_params['spikes'] = []
        snn.RAND_ID = v

        ret = snn.run(t_steps = t_steps, in_params = in_params, in_spikes = None, target = target, train = True, save_states = True, save_traces = True)

        yv[v,n] = np.mean(ret['y'][:,burn_in:])




fig = plt.figure()
plt.subplot(2,1,1)
plt.scatter(x_val, y_vec)
for v in range(V_test):
    plt.scatter(x_test, yv[v,:], color = 'orange', alpha = 0.5)
plt.title("Data")
plt.subplot(2,1,2)
plt.title("Data")
plt.plot(mb_costs)
plt.savefig("temp.pdf")
plt.close()
