# #!/usr/bin/python3
# # -*- coding: utf-8 -*-

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# import random

# class TetrisDQN(nn.Module):
#     def __init__(self, input_shape, rotation_size, width):
#         super(TetrisDQN, self).__init__()
#         self.input_shape = input_shape
#         self.rotation_size = rotation_size
#         self.width = width

#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU()
#         )

#         self.shape_embedding = nn.Embedding(8, 16)
#         conv_out_size = self._get_conv_out(input_shape)

#         self.fc = nn.Sequential(
#             nn.Linear(conv_out_size + 32, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#         )

#         self.rotation_head = nn.Linear(256, rotation_size)
#         self.x_head = nn.Linear(256, width)

#     def _get_conv_out(self, shape):
#         o = self.features(torch.zeros(1, *shape))
#         return int(np.prod(o.size()))

#     def forward(self, x, current_shape, next_shape):
#         conv_out = self.features(x).view(x.size(0), -1)
#         current_emb = self.shape_embedding(current_shape)
#         next_emb = self.shape_embedding(next_shape)
#         shape_emb = torch.cat([current_emb, next_emb], dim=1)
#         combined = torch.cat([conv_out, shape_emb], dim=1)

#         fc_out = self.fc(combined)
#         rotation_q = self.rotation_head(fc_out)
#         x_q = self.x_head(fc_out)
#         return rotation_q, x_q

# class DQNAgent:
#     def __init__(self, state_shape, rotation_size=4, width=10):
#         self.state_shape = state_shape
#         self.rotation_size = rotation_size
#         self.width = width
#         self.memory = deque(maxlen=20000)
#         self.gamma = 0.95
#         self.epsilon = 1.0
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.batch_size = 64
#         self.update_target_every = 1000
#         self.steps_since_target_update = 0

#         self.model = TetrisDQN(state_shape, rotation_size, width)
#         self.target_model = TetrisDQN(state_shape, rotation_size, width)
#         self.update_target_network()

#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025)
#         self.criterion = nn.SmoothL1Loss()

#     def update_target_network(self):
#         self.target_model.load_state_dict(self.model.state_dict())

#     def remember(self, state, action, reward, next_state, done, current_shape, next_shape):
#         self.memory.append((state, action, reward, next_state, done, current_shape, next_shape))

#     # ✅ Chỉnh sửa: luôn trả tuple (rotation, x)
#     def act(self, state, current_shape, next_shape, epsilon=None):
#         if epsilon is None:
#             epsilon = self.epsilon

#         if np.random.rand() <= epsilon:
#             rotation = random.randrange(self.rotation_size)
#             x = random.randrange(self.width)
#             return (rotation, x)  # trả tuple

#         state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
#         current_shape_t = torch.LongTensor([current_shape])
#         next_shape_t = torch.LongTensor([next_shape])

#         with torch.no_grad():
#             rotation_q, x_q = self.model(state_t, current_shape_t, next_shape_t)
#         rotation_action = torch.argmax(rotation_q[0]).item()
#         x_action = torch.argmax(x_q[0]).item()
#         return (rotation_action, x_action)  # trả tuple

#     def replay(self):
#         if len(self.memory) < self.batch_size:
#             return

#         minibatch = random.sample(self.memory, self.batch_size)
#         states, actions, rewards, next_states, dones, current_shapes, next_shapes = zip(*minibatch)

#         states = torch.FloatTensor(np.array(states)).unsqueeze(1)
#         next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(1)
#         current_shapes = torch.LongTensor(current_shapes)
#         next_shapes = torch.LongTensor(next_shapes)
#         actions = torch.LongTensor(np.array(actions))
#         rewards = torch.FloatTensor(rewards)
#         dones = torch.FloatTensor(dones)

#         rotation_q, x_q = self.model(states, current_shapes, next_shapes)
#         current_q_rot = rotation_q.gather(1, actions[:, 0].unsqueeze(1)).squeeze()
#         current_q_x = x_q.gather(1, actions[:, 1].unsqueeze(1)).squeeze()

#         with torch.no_grad():
#             next_rotation_q, next_x_q = self.target_model(next_states, current_shapes, next_shapes)
#             next_q_rot = next_rotation_q.max(1)[0]
#             next_q_x = next_x_q.max(1)[0]

#         target_q_rot = rewards + (1 - dones) * self.gamma * next_q_rot
#         target_q_x = rewards + (1 - dones) * self.gamma * next_q_x

#         loss = self.criterion(current_q_rot, target_q_rot) + self.criterion(current_q_x, target_q_x)

#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#         self.optimizer.step()

#         self.steps_since_target_update += 1
#         if self.steps_since_target_update >= self.update_target_every:
#             self.update_target_network()
#             self.steps_since_target_update = 0

#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

#     def save(self, filename):
#         torch.save({
#             'model_state_dict': self.model.state_dict(),
#             'target_model_state_dict': self.target_model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'epsilon': self.epsilon,
#             'memory': self.memory
#         }, filename)

#     def load(self, filename):
#         import os
#         if not os.path.exists(filename):
#             raise FileNotFoundError(f"Model file {filename} not found.")
#         checkpoint = torch.load(filename, map_location=torch.device('cpu'))
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.epsilon = checkpoint.get('epsilon', 1.0)
#         self.memory = checkpoint.get('memory', deque(maxlen=20000))
#         print(f"Loaded model from {filename}, epsilon={self.epsilon:.4f}")
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNCNN(nn.Module):
    def __init__(self, height, width, n_rotations, n_positions):
        super(DQNCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # input [batch, 1, H, W]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        conv_out_size = 64 * height * width
        self.rotation_head = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_rotations)
        )
        self.x_head = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_positions)
        )

    def forward(self, x):
        # x: [batch, 1, H, W] hoặc [batch, 1, 1, H, W]
        if x.dim() == 5:
            x = x.squeeze(2)  # bỏ chiều thừa nếu có
        conv_out = self.features(x).view(x.size(0), -1)
        rotation_q = self.rotation_head(conv_out)
        x_q = self.x_head(conv_out)
        return rotation_q, x_q

class DQNAgent:
    def __init__(self, state_shape, n_rotations=4, n_positions=10):
        self.state_shape = state_shape
        self.n_rotations = n_rotations
        self.n_positions = n_positions

        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        self.epsilon = 1.0

        self.model = DQNCNN(state_shape[1], state_shape[2], n_rotations, n_positions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def act(self, state, current_shape=None, next_shape=None):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.n_rotations - 1), random.randint(0, self.n_positions - 1)
        state = torch.tensor(np.array([state], dtype=np.float32)).to(device)  # [1,1,H,W]
        with torch.no_grad():
            rotation_q, x_q = self.model(state)
        rotation_action = torch.argmax(rotation_q).item()
        x_action = torch.argmax(x_q).item()
        return rotation_action, x_action

    def remember(self, state, action, reward, next_state, done, current_shape=None, next_shape=None):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Chuyển sang tensor
        states = torch.tensor(np.array(states, dtype=np.float32)).to(device)
        next_states = torch.tensor(np.array(next_states, dtype=np.float32)).to(device)

        # squeeze nếu shape [batch,1,1,H,W]
        if states.dim() == 5:
            states = states.squeeze(2)
            next_states = next_states.squeeze(2)

        rotation_q, x_q = self.model(states)
        with torch.no_grad():
            next_rotation_q, next_x_q = self.model(next_states)

        target_rotation = rotation_q.clone()
        target_x = x_q.clone()

        for i in range(self.batch_size):
            rot, x_pos = actions[i]
            target_rotation[i, rot] = rewards[i] + (0 if dones[i] else 0.99 * torch.max(next_rotation_q[i]))
            target_x[i, x_pos] = rewards[i] + (0 if dones[i] else 0.99 * torch.max(next_x_q[i]))

        loss = self.loss_fn(rotation_q, target_rotation) + self.loss_fn(x_q, target_x)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': self.memory,
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = checkpoint.get('memory', deque(maxlen=20000))
        self.epsilon = checkpoint.get('epsilon', 1.0)
