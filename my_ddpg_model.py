# Implementation based on UDACITY DRLND course

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor_QNetwork(nn.Module):
    """Actor (Policy) Model. DuelingQNetwork"""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc1_duelling=128, fc2_duelling=128, fcout_duelling=128, init_weights=3e-3):
        """Initialize parameters and build model.
        Normalization of Layers improves network perofmance significantly.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor_QNetwork, self).__init__()
        # Random seed
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        # Input layer
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)

        # create hidden layers according to HIDDEN_SIZES
        self.bn2 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn3 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc1_duelling)

        # create duelling layers according to DUELLING_SIZES
        self.bn_adv_fc1 = nn.BatchNorm1d(fc1_duelling)
        self.adv_fc1 = nn.Linear(fc1_duelling, fc2_duelling)
        self.bn_val_fc1 = nn.BatchNorm1d(fc2_duelling)
        self.val_fc1 = nn.Linear(fc2_duelling, fcout_duelling)

        # Output layer
        self.bn_adv_out = nn.BatchNorm1d(fcout_duelling)
        self.adv_out = nn.Linear(fcout_duelling, action_size)
        self.bn_val_out = nn.BatchNorm1d(fcout_duelling)
        self.val_out = nn.Linear(fcout_duelling, 1)

        self.reset_parameters(init_weights)

        # Initialize Parameter
    def reset_parameters(self, init_weights):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.adv_fc1.weight.data.uniform_(*hidden_init(self.adv_fc1))
        self.val_fc1.weight.data.uniform_(*hidden_init(self.val_fc1))
        self.adv_out.weight.data.uniform_(-init_weights, init_weights)
        self.val_out.weight.data.uniform_(-init_weights, init_weights)


    def forward(self, state):
        """Build a network that maps state -> action values. Forward propagation"""

         # classical network with relu activation function
        x = F.relu(self.fc1(self.bn1(state)))
        x = F.relu(self.fc2(self.bn2(x)))
        x = F.relu(self.fc3(self.bn3(x)))

         # duelling strams adv and val
        adv, val = None, None
        adv = F.relu(self.adv_fc1(self.bn_adv_fc1(x)))
        val = F.relu(self.val_fc1(self.bn_val_fc1(x)))
        adv = self.adv_out(self.bn_adv_out(adv))
        val = self.val_out(self.bn_val_out(val)).expand(x.size(0), self.action_size)

        # combing result of adv and val stream
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)

        return F.tanh(x)


class Critic_QNetwork(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, init_weights=3e-3):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic_QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        # Input Layer
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn_s = nn.BatchNorm1d(fc1_units)

        # Hidden Layer
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters(init_weights)

        # Initialize Parameter
    def reset_parameters(self, init_weights):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-init_weights, init_weights)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fc1(state))
        x = torch.cat((self.bn_s(xs), action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
