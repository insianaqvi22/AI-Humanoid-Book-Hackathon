---
title: AI Training Integration
sidebar_position: 7
---

import CodeRunner from '@site/src/components/CodeRunner';

# AI Training Integration in Python

This section covers how to integrate AI training with humanoid robot simulation using Python.

## Reinforcement Learning Setup

Setting up reinforcement learning for robot training:

```python
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim

class RobotEnv(gym.Env):
    """Custom environment for humanoid robot training"""
    def __init__(self):
        super(RobotEnv, self).__init__()

        # Define action and observation space
        # Actions: joint torques for each actuator
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Observations: joint positions, velocities, IMU data
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
        )

        self.current_state = np.zeros(36)
        self.step_count = 0
        self.max_steps = 1000

    def reset(self):
        """Reset the environment to initial state"""
        self.current_state = np.random.normal(0, 0.1, 36)
        self.step_count = 0
        return self.current_state

    def step(self, action):
        """Execute one step in the environment"""
        # Apply action to robot simulation
        self.current_state = self._update_state(action)

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        done = self.step_count >= self.max_steps
        self.step_count += 1

        return self.current_state, reward, done, {}

    def _update_state(self, action):
        """Update robot state based on action"""
        # Simplified state update
        new_state = self.current_state + action * 0.1
        # Add some noise to simulate real-world uncertainty
        new_state += np.random.normal(0, 0.01, len(new_state))
        return new_state

    def _calculate_reward(self):
        """Calculate reward based on current state"""
        # Simplified reward function
        # Encourage stable, upright position
        upright_reward = np.exp(-abs(self.current_state[0]))  # Keep upright
        velocity_penalty = -np.sum(np.abs(self.current_state[18:24]))  # Minimize velocity
        return upright_reward + velocity_penalty
```

## Neural Network Architecture

Deep neural network for robot control:

```python
class RobotPolicyNetwork(nn.Module):
    """Neural network for robot policy learning"""
    def __init__(self, input_size=36, output_size=12):
        super(RobotPolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, x):
        return self.network(x)

class RobotValueNetwork(nn.Module):
    """Neural network for value estimation"""
    def __init__(self, input_size=36):
        super(RobotValueNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)
```

## Training Loop

Implementation of the training loop:

```python
import random
from collections import deque

class RobotTrainer:
    """Trainer for robot control policies"""
    def __init__(self, env, policy_lr=1e-4, value_lr=3e-4):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = RobotPolicyNetwork().to(self.device)
        self.value_net = RobotValueNetwork().to(self.device)

        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=policy_lr
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=value_lr
        )

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # Discount factor

    def select_action(self, state):
        """Select action using the policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy_net(state_tensor)
        return action.cpu().numpy()[0]

    def train_step(self, batch_size=64):
        """Perform one training step"""
        if len(self.memory) < batch_size:
            return

        # Sample random batch from memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Update value network
        current_values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        target_values = rewards + (self.gamma * next_values * ~dones)

        value_loss = nn.MSELoss()(current_values, target_values.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy network
        advantages = target_values - current_values
        current_actions = self.policy_net(states)
        policy_loss = -(advantages * current_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
```

## Integration with Simulation

Connecting AI training to simulation:

```python
def train_robot_policy():
    """Main training function"""
    env = RobotEnv()
    trainer = RobotTrainer(env)

    num_episodes = 1000

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = trainer.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store experience in memory
            trainer.memory.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            # Perform training step
            trainer.train_step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {total_reward}")

    return trainer

# Example usage
# trainer = train_robot_policy()
```

## Interactive AI Training Demo

<CodeRunner language="python" title="AI Training Simulation">

```python
import numpy as np
import random

class SimpleRobotEnv:
    def __init__(self):
        self.state = np.array([0.0, 0.0, 0.0])  # position, velocity, balance
        self.target = np.array([1.0, 0.0, 0.0])
        self.steps = 0
        self.max_steps = 50

    def reset(self):
        self.state = np.random.normal(0, 0.1, 3)
        self.steps = 0
        return self.state

    def step(self, action):
        # Simplified physics simulation
        self.state[1] += action[0] * 0.1  # Update velocity
        self.state[0] += self.state[1] * 0.1  # Update position
        self.state[2] += action[1] * 0.05  # Update balance

        # Apply gravity and damping
        self.state[1] -= 0.01  # Gravity effect
        self.state[1] *= 0.99  # Damping

        # Calculate reward
        pos_error = abs(self.state[0] - self.target[0])
        vel_penalty = abs(self.state[1]) * 0.1
        balance_penalty = abs(self.state[2]) * 0.2
        reward = -pos_error - vel_penalty - balance_penalty

        self.steps += 1
        done = self.steps >= self.max_steps

        return self.state, reward, done, {}

class SimplePolicy:
    def __init__(self):
        self.weights = np.random.normal(0, 0.1, (2, 3))  # 2 actions, 3 state inputs

    def get_action(self, state):
        action = np.dot(self.weights, state)
        return np.clip(action, -1, 1)  # Clamp to [-1, 1]

def train_episode(env, policy):
    state = env.reset()
    total_reward = 0
    actions_taken = []

    for step in range(env.max_steps):
        action = policy.get_action(state)
        actions_taken.append(action.copy())
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward, actions_taken

# Initialize environment and policy
env = SimpleRobotEnv()
policy = SimplePolicy()

print("Starting AI Training Simulation...")
print(f"Initial weights: {policy.weights.flatten()[:4]}...")  # Show first 4 weights

# Run a few training episodes
for episode in range(5):
    reward, actions = train_episode(env, policy)
    print(f"Episode {episode + 1}: Total reward = {reward:.2f}")

    # Simple policy update (random walk for demonstration)
    policy.weights += np.random.normal(0, 0.01, policy.weights.shape)

print(f"Final weights: {policy.weights.flatten()[:4]}...")  # Show first 4 weights
print("Training completed!")
```

</CodeRunner>