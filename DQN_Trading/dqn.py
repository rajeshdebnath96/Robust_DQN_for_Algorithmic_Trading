import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # input layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # hidden layer
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # output layer

    def forward(self, x):
        # activation function
        # print(f'x: {x.shape}')
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # capacity of buffer
        self.buffer = []  # replay buffer
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        ''' replay buffer is a queue (LIFO)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.frame_idx = 0
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size

        self.policy_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)

        # Robust reward estimation buffers: for each (state, action) pair.
        self.Z = {}         # Buffer for quantile estimation
        self.Z_tilde = {}   # Buffer for averaging
        self.robust_buffer_len = 10   # fixed number of reward samples to use
        self.robust_epsilon = 0.1     # contamination parameter for robust estimator
        self.robust_delta = 0.1       # delta parameter for robust estimator
        self.R_threshold = cfg.R_threshold  # reward scale parameter (maximum absolute reward)
        self.C_threshold = cfg.C_threshold  # constant multiplier for threshold
        self.delta_1 = cfg.delta_1          # delta parameter for threshold function

    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                # Convert tuple state to list if necessary
                if isinstance(state, tuple):
                    state = list(state)
                state_tensor = torch.tensor([state], device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        
        # Convert tuple states to lists if necessary
        if isinstance(state_batch[0], tuple):
            state_batch = [list(state) for state in state_batch]
        if isinstance(next_state_batch[0], tuple):
            next_state_batch = [list(state) for state in next_state_batch]
            
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # gradient clipping
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), os.path.join(path, 'dqn_checkpoint.pth'))

    def load(self, path):
        self.target_net.load_state_dict(torch.load(os.path.join(path, 'dqn_checkpoint.pth')))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


# ======================== Threshold Function ========================
def compute_G_t(t, R, C, epsilon, delta_1):
    """
    Computes a threshold value for the robust reward estimator.

    Parameters:
    - t: current number of reward observations in the buffer
    - R: reward scale (typically maximum absolute reward)
    - C: constant multiplier
    - epsilon: contamination parameter (used in robust estimation)
    - delta_1: probability parameter

    Returns:
    - G_t: threshold value
    """
    T_lim = 2 * np.log(4 / delta_1) + 1
    if t <= T_lim:
        return 2 * R
    else:
        return C * R * (np.sqrt(np.log(4 / delta_1) / t) + np.sqrt(epsilon)) + R

# ======================== Robust Trimmed Mean Estimator ========================
def univariate_trimmed_mean(Z, Z_tilde, epsilon, delta):
    """
    Robust reward estimator based on trimmed mean.

    Parameters:
    - Z: list of reward observations used for quantile estimation (length M/2)
    - Z_tilde: list of reward observations for averaging (length M/2)
    - epsilon: contamination parameter
    - delta: probability parameter

    Returns:
    - hat_mu_T: robust mean estimate
    """
    M = len(Z) + len(Z_tilde)
    assert len(Z) == len(Z_tilde) == M // 2, "Z and Z_tilde must both have M/2 elements."

    # Step 1: Set ζ = 8ε + 24 log(4/δ) / M
    zeta = 8 * epsilon + 24 * np.log(4 / delta) / M
    zeta = min(zeta, 1)  # Ensure ζ does not exceed 1

    # Step 2: Compute quantiles γ and β
    Z_sorted = np.sort(Z)
    gamma_index = int(np.floor(zeta * (M // 2)))
    beta_index = int(np.floor((1 - zeta) * (M // 2)))
    gamma_index = min(gamma_index, len(Z_sorted) - 1)
    beta_index = min(beta_index, len(Z_sorted) - 1)
    gamma = Z_sorted[gamma_index]
    beta = Z_sorted[beta_index]

    # Step 3: Compute robust mean estimate
    def phi_gamma_beta(x, gamma, beta):
        return min(max(x, gamma), beta)

    trimmed_sum = sum(phi_gamma_beta(x, gamma, beta) for x in Z_tilde)
    hat_mu_T = (2 / M) * trimmed_sum
    return hat_mu_T
