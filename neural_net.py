# neural_net.py
# Defines the CNN+LSTM “brain,” plus an ε-Greedy policy with slower decay (10 000 steps).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN(nn.Module):
    """
    A tiny CNN → LSTM → 2-FC architecture that maps a 1×128×128 preprocessed frame
    to Q-values over 5 actions.
    """
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.out_neurons = self.count_neurons((1, 128, 128))
        self.lstm = nn.LSTMCell(self.out_neurons, 256)
        self.fc1 = nn.Linear(in_features=256, out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)

    def forward(self, x, hidden=None):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(-1, self.out_neurons)
        hx, cx = self.lstm(x, hidden)
        x = hx
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, (hx, cx)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)

class AI:
    """
    Wraps the CNN “brain” and a policy “body” that picks actions.
    The body’s __call__ must return a Python int.
    """
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs, hidden):
        output, (hx, cx) = self.brain(inputs, hidden)
        action = self.body(output)  # Python int
        return action, (hx, cx)

class EpsilonGreedyBody:
    """
    ε-Greedy policy (returns a Python int).  
    • ε starts at 1.0, decays linearly to 0.05 over 10 000 steps  
    • After that, ε stays at 0.05
    """
    def __init__(self, initial_epsilon=1.0, min_epsilon=0.05, decay_steps=10000):
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = float(decay_steps)
        self.steps_done = 0

    def __call__(self, qvalues):
        self.steps_done += 1
        # Anneal ε linearly
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon - (1.0 - self.min_epsilon) / self.decay_steps
        )
        if np.random.rand() < self.epsilon:
            # Explore: pick a random action in [0, num_actions)
            return int(np.random.randint(0, qvalues.shape[-1]))
        else:
            # Exploit: pick argmax_a Q(s,a)
            return int(qvalues.data.cpu().numpy().argmax(axis=-1).item())

# (We keep SoftmaxBody around for comparison, but we won’t use it in ai.py)
class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T, dim=1)
        action = probs.multinomial(num_samples=1)
        return int(action.item())
