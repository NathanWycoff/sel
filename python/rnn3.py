import math
import numpy as np
#import torch
#import torch.nn as nn
#import torch.nn.functional as F

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt


def simple_signal(seqlen=20):
    timesteps = np.linspace(1, 10, seqlen+1)
    signal = np.sin(timesteps)
    signal.resize((seqlen+1, 1))
    #x = torch.Tensor(signal[:-1])
    #x = torch.Tensor(signal[:-1])
    y = signal[1:]
    x = signal[:-1]
    return timesteps, x, y

def sine_signal(seqlen=1000, size=4, clock=True, tensor=True):
    steps = np.linspace(0, 1, seqlen+1)
    freqs = np.arange(1, 5)
    phases = np.random.uniform(size=size)
    amps = np.random.uniform(0.2, 1, size=size)
    signals = np.array([a * np.sin(f * (steps + p) * 2 * np.pi)
                        for p, f, a in zip(phases, freqs, amps)])
    signal = signals.sum(axis=0)
    signal.resize((seqlen+1, 1))
    x = steps[1:].reshape(seqlen, 1) if clock else signal[:-1]
    y = signal[1:]
    if tensor:
        x = torch.Tensor(x)
        y = torch.Tensor(y)
    return steps, x, y


#def plot_results(steps, outs, y, save='RNN3.pdf', info={}):
#    x = steps[1:] * 1000
#    plt.scatter(x, y, s=80, label="Actual")
#    plt.scatter(x, outs, label="Predicted")
#    for k in info:
#        plt.plot([], [], ' ', label=f"{k} = {info[k]}")
#    plt.legend()
#    plt.tight_layout()
#    plt.savefig(save, dpi=300)
#
#
#class RNN3(nn.Module):
#    def __init__(self, input_size, hidden_size, output_size):
#        super(RNN3, self).__init__()
#        self.hidden_size = hidden_size
#        self.w_ih = torch.zeros(input_size, hidden_size, requires_grad=True)
#        self.w_hh = torch.zeros(hidden_size, hidden_size, requires_grad=True)
#        self.w_ho = torch.zeros(hidden_size, output_size, requires_grad=True)
#        self.reset_parameters()
#        self.init_hidden()
#
#    def forward(self, inp):
#        self.hidden = torch.tanh(inp @ self.w_ih + self.hidden @ self.w_hh)
#        out = self.hidden @ self.w_ho
#        self.hidden.detach_()
#        return out
#
#    def reset_parameters(self):
#        stdv = 1.0 / math.sqrt(self.hidden_size)
#        for weight in self.parameters():
#            nn.init.uniform_(weight, -stdv, stdv)
#
#    def parameters(self):
#        return [self.w_ih, self.w_hh, self.w_ho]
#
#    def init_hidden(self):
#        self.hidden = torch.zeros(1, self.hidden_size, requires_grad=True)
#
#
#def test_RNN3_1():
#    """Incremental updates to weights"""
#    # steps, x, y = simple_signal(20)
#    steps, x, y = sine_signal(100, clock=True)
#    epochs = 300
#    lr = 0.1
#    net = RNN3(input_size=1, hidden_size=60, output_size=1)
#    # opt = torch.optim.SGD(net.parameters(), lr=lr)
#    # opt = torch.optim.Adam(net.parameters(), lr=0.001)
#    opt = torch.optim.Adam(net.parameters(), lr=0.01)
#    for i in range(epochs):
#        total_loss = 0
#        for j in range(x.size(0)):
#            opt.zero_grad()
#            inp = x[j:j+1]
#            tgt = y[j:j+1]
#            out = net.forward(inp)
#            loss = (out - tgt).pow(2).sum() / out.size(0)
#            total_loss += loss
#            loss.backward()
#            opt.step()
#        if i % 10 == 0:
#            mse = total_loss / x.size(0)
#            print("Epoch: {} loss {}".format(i, mse))
#    outs = []
#    for i in range(x.size(0)):
#       inp = x[i:i+1]
#       out = net.forward(inp)
#       outs.append(out.data.numpy().ravel()[0])
#
#    info = {'epochs': epochs, 'mse': f'{mse:.5g}'}
#    plot_results(steps, outs, y.data.numpy(), info=info)
#
#
#def test_RNN3_2():
#    """Backpropagation through time"""
#    steps, x, y = sine_signal(100, clock=False)
#    # steps, x, y = sine_signal(100, clock=True)  # E-prop experiment
#    epochs = 300
#    lr = 0.1
#    net = RNN3(input_size=1, hidden_size=6, output_size=1)
#    opt = torch.optim.SGD(net.parameters(), lr=lr)
#    # opt = torch.optim.Adam(net.parameters(), lr=0.01)
#    for i in range(epochs):
#        total_loss = 0
#        outs = torch.zeros_like(y)
#        for j in range(x.size(0)):
#            opt.zero_grad()
#            inp = x[j:j+1]
#            out = net.forward(inp)
#            outs[j] = out
#        total_loss = F.mse_loss(y, outs)
#        total_loss.backward()
#        opt.step()
#        if i % 10 == 0:
#            print("Epoch: {} loss {}".format(i, total_loss))
#    outs = []
#    for i in range(x.size(0)):
#       inp = x[i:i+1]
#       out = net.forward(inp)
#       outs.append(out.data.numpy().ravel()[0])
#    info = {'epochs': epochs, 'mse': f'{total_loss:.3g}'}
#    plot_results(steps, outs, y.data.numpy(), info=info)
#
#
#def main():
#    # test_RNN3_1()
#    test_RNN3_2()
#
#
#if __name__ == '__main__':
#    main()
