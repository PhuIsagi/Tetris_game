import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCriticCNN(nn.Module):
    def __init__(self, height, width, n_rotations, n_positions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        conv_out_size = 64 * height * width

        self.actor_rot = nn.Linear(conv_out_size, n_rotations)
        self.actor_x = nn.Linear(conv_out_size, n_positions)
        self.critic = nn.Linear(conv_out_size, 1)

    def forward(self, x):
        conv_out = self.features(x).view(x.size(0), -1)
        rot_logits = self.actor_rot(conv_out)
        x_logits = self.actor_x(conv_out)
        value = self.critic(conv_out)
        return rot_logits, x_logits, value


class PPOAgent:
    def __init__(self, state_shape, n_rotations=4, n_positions=10, lr=1e-4, gamma=0.99, eps_clip=0.2):
        self.n_rotations = n_rotations
        self.n_positions = n_positions
        self.gamma = gamma
        self.eps_clip = eps_clip

        self.model = ActorCriticCNN(state_shape[1], state_shape[2], n_rotations, n_positions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        rot_logits, x_logits, _ = self.model(state)

        rot_prob = F.softmax(rot_logits, dim=-1)
        x_prob = F.softmax(x_logits, dim=-1)

        rot_dist = torch.distributions.Categorical(rot_prob)
        x_dist = torch.distributions.Categorical(x_prob)

        rot_action = rot_dist.sample()
        x_action = x_dist.sample()

        self.states.append(state)
        self.actions.append((rot_action.item(), x_action.item()))
        self.logprobs.append((rot_dist.log_prob(rot_action), x_dist.log_prob(x_action)))

        return rot_action.item(), x_action.item()

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.is_terminals.append(done)

    def clear_memory(self):
        self.states, self.actions, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], []

    def compute_returns(self):
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        return (returns - returns.mean()) / (returns.std() + 1e-7)

    def update(self, epochs=4, batch_size=64):
        returns = self.compute_returns()
        states = torch.cat(self.states, dim=0)
        actions_rot = torch.tensor([a[0] for a in self.actions], dtype=torch.long).to(device)
        actions_x = torch.tensor([a[1] for a in self.actions], dtype=torch.long).to(device)
        old_logprobs_rot = torch.cat([lp[0] for lp in self.logprobs], dim=0)
        old_logprobs_x = torch.cat([lp[1] for lp in self.logprobs], dim=0)

        for _ in range(epochs):
            rot_logits, x_logits, values = self.model(states)
            rot_probs = F.softmax(rot_logits, dim=-1)
            x_probs = F.softmax(x_logits, dim=-1)

            rot_dist = torch.distributions.Categorical(rot_probs)
            x_dist = torch.distributions.Categorical(x_probs)

            logprobs_rot = rot_dist.log_prob(actions_rot)
            logprobs_x = x_dist.log_prob(actions_x)

            ratios_rot = torch.exp(logprobs_rot - old_logprobs_rot.detach())
            ratios_x = torch.exp(logprobs_x - old_logprobs_x.detach())

            advantages = returns - values.squeeze().detach()

            surr1_rot = ratios_rot * advantages
            surr2_rot = torch.clamp(ratios_rot, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            surr1_x = ratios_x * advantages
            surr2_x = torch.clamp(ratios_x, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1_rot, surr2_rot).mean() \
                   -torch.min(surr1_x, surr2_x).mean() \
                   + F.mse_loss(values.squeeze(), returns)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.clear_memory()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=device))
