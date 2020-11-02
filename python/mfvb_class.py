#!/usr/bin/env python4
# -*- coding: utf-8 -*-
#  python/test_mouse.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.14.2020

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
exec(open("python/lib.py").read())
exec(open("python/mouse_lib.py").read())

np.random.seed(123)

# For the  mouse problem, groups of neurons fire to indicate the task parameters. We specify their properties here.
signal_duration = 0.05
break_duration = 0.05
cue_duration = 0.15
spikes_per_signal = 1
neur_per_group = 10
n_signals = 1

R = 2
H = 100
P = 4*neur_per_group
Q = 1

#feed_align = False
feed_align = True

# Variational Inference Params
prior_var = 1e-0 * 1/H
#prior_var = 0
data_var = 1e-3
V = 5
#V = 1

# Optimizer 
learn_rate = 5e-4
mb_size = 20
#epochs = 10 * mb_size
epochs = 5000 * mb_size
#epochs = 100 * mb_size

t_eps = 0.05
betas = np.array([2 for i in range(H)])

#TODO: No Xavier? Also, make sure it aligns for FA. 
stdi = np.sqrt(2 / (H + P))
#LAMBDA_in = np.zeros([H,P]) + np.random.normal(size=[H,P], scale = stdi)
#LAMBDA_rec = np.zeros([H,H]) + np.random.normal(size=[H,H], scale = stdi)
#LAMBDA_out = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = stdi)

isig = [1.0,0.001,1.0]
LAMBDA_in = np.zeros([H,P]) + np.random.normal(size=[H,P], scale = isig[0])
LAMBDA_rec = np.zeros([H,H]) + np.random.normal(size=[H,H], scale = isig[1])
LAMBDA_out = np.zeros([Q,H]) + np.random.normal(size=[Q,H], scale = isig[2])

phi_std = 1e-8
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
snn = ALifNeuralNetwork(R, H, P, Q, t_eps = t_eps, mb_size = 1, cost = crossentropy, optimizer = opt)

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

ans = np.empty(epochs, dtype = bool)
guess = np.empty(epochs, dtype = bool)
mean_costs = np.zeros(epochs)
est_probs = np.zeros(epochs) - 1.
LAMI_grad = np.zeros_like(LAMBDA_in)
LAMR_grad = np.zeros_like(LAMBDA_rec)
LAMO_grad = np.zeros_like(LAMBDA_out)

PHII_grad = np.zeros_like(LAMBDA_in)
PHIR_grad = np.zeros_like(LAMBDA_rec)
PHIO_grad = np.zeros_like(LAMBDA_out)
for epoch in tqdm(range(epochs)):

    # Sample a new problem!
    np.random.seed(epoch)
    coinflip, inlist, in_spikes, target, t_steps, cue_time = make_mouse_prob(t_eps, n_signals, signal_duration, break_duration, cue_duration, spikes_per_signal, neur_per_group, noise = False)

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
        ret = snn.run(t_steps = t_steps, in_spikes = in_spikes, target = target, train = True, save_states = True, save_traces = True)
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

    #NOTE: What if we averaged after the zero comparison? May be more robust. 
    ind_decs = np.mean(ys[:,int(cue_time / t_eps):], axis = 1) > 0
    est_probs[epoch]= np.mean(ind_decs)

    ans[epoch] = coinflip
    guess[epoch] = np.mean(ind_decs) > 0

    ## Fold in data variance.
    #TODO: If we re-use this block, we may need to account for mb_size. 
    #LAMI_grad *= epochs / data_var
    #LAMR_grad *= epochs / data_var
    #LAMO_grad *= epochs / data_var
    #PHII_grad *= epochs / data_var
    #PHIR_grad *= epochs / data_var
    #PHIO_grad *= epochs / data_var

    # Add in prior grads
    if epoch % mb_size == 0:
        # Because this code is being ran epoch / mb_size times, we need to account for that
        const = epochs // mb_size
        constinv = data_var / epochs / const
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

    mean_costs[epoch] = np.mean(costs)

y = ret['y']
S = ret['S']

q = [0.1, 0.5, 0.9]
post_pred = np.quantile(ys, q, axis = 0)

def get_ema(x, ll = 0.5):
    """
    ll - The convex combination parameter; ll = 1 gives the identity operation. 
    """
    z = np.copy(x)
    for i in range(1,len(x)):
        z[i] = ll * x[i] + (1-ll) * z[i-1]
    return(z)

#TODO: Better chart to show increasing accuracy would be to plot the probabilities given for each of the two solutions over time, and we could watch them diverge to 0 and 1. 
fig = plt.figure(figsize=[8,8])
plt.subplot(2,2,1)
plt.plot(S[0,:,:].T)
plt.title("Potential")
#plt.subplot(2,2,2)
#plt.plot(S[1,:,:].T)
#plt.title("Adaptive Threshold")
plt.subplot(2,2,2)
ema_ll = 0.01
plt.plot(get_ema(est_probs[ans], ll = ema_ll), label = "Up")
plt.plot(get_ema(est_probs[np.logical_not(ans)], ll = ema_ll), label = "Down")
plt.title("Variational Probability of Up")
plt.legend()
plt.subplot(2,2,3)
#plt.plot(y.flatten())
plt.plot(post_pred[0,:], c = 'orange')
plt.plot(post_pred[1,:], c = 'red')
plt.plot(post_pred[2,:], c = 'orange')
plt.title("Output")
plt.subplot(2,2,4)
plt.plot(np.log10(mean_costs))
plt.title("Costs")
plt.savefig("temp.png")
plt.close()

#TODO: Larger mb_size on this problem?
snn.mb_size = mb_size
plot_allinone(snn, ret, inlist, mean_costs, path = "aio.png")
snn.mb_size = 1

last_prop = 0.1
last_num = int(np.ceil(last_prop * epochs))
acc = np.mean(guess[(-last_num):] == ans[(-last_num):])
print(acc)

# Save our weights!
cur_LAMBDAs = [LAMBDA_in, LAMBDA_rec, LAMBDA_out]
cur_PHIs = [PHI_in, PHI_rec, PHI_out]
cur_params = cur_LAMBDAs + cur_PHIs
np.savez('./data/mvfb_class_varpar4', LAMBDA_in, LAMBDA_rec, LAMBDA_out, PHI_in, PHI_rec, PHI_out)


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
BIGGEST_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGEST_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Make a plot for our short paper. 
fig = plt.figure(figsize=[5,2.5])
plt.subplot(1,2,1)
batch_costs = np.mean(mean_costs.reshape(-1,mb_size), axis = 1)
if np.min(batch_costs) > 0:
    plt.plot(np.log10(batch_costs))
else:
    plt.plot(batch_costs)
plt.xlabel("Training Iteration (epochs)")
plt.ylabel("Divergence (nats)")
plt.title("KL Loss")
plt.subplot(1,2,2)
ema_ll = 0.01
plt.plot(get_ema(est_probs[ans], ll = ema_ll), label = "Up")
plt.plot(get_ema(est_probs[np.logical_not(ans)], ll = ema_ll), label = "Down")
plt.xlabel("Training Observations")
plt.ylabel("Probability")
plt.title("P(Cheese=Up)")
plt.legend(title = "Cheese is really...")
plt.subplots_adjust(bottom = 0.2, wspace = 0.3, hspace = 10.)
plt.savefig("training.pdf")
plt.close()
