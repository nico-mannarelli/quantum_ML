"""
Deep Q-Learning (DQN) Implementation for CartPole-v1

This module implements a Deep Q-Network (DQN) agent using PyTorch
to solve the CartPole-v1 environment from OpenAI Gym.
"""

import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

# Hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
BATCH_SIZE = 64
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000
NUM_EPISODES = 1000
MAX_STEPS = 1000

# Experience tuple
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Add an experience to memory."""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """Sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """Deep Q-Network: A neural network that approximates Q-values."""
    
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    """DQN Agent with epsilon-greedy exploration and experience replay."""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.epsilon = EPSILON_START
        
        # Policy network and target network
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            return random.randrange(self.action_size)

    def optimize_model(self):
        """Optimize the policy network using a batch of experiences."""
        if len(self.memory) < BATCH_SIZE:
            return
        
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), 
            dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train_agent():
    """Train the DQN agent on CartPole-v1 environment."""
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    print("Starting DQN Training on CartPole-v1")
    print("=" * 50)
    
    # Training loop
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = torch.tensor([state], dtype=torch.float32)
        
        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            reward = torch.tensor([reward], dtype=torch.float32)
            next_state = torch.tensor([next_state], dtype=torch.float32) if not done else None
            
            agent.memory.push(state, torch.tensor([action]), reward, next_state)
            state = next_state
            agent.optimize_model()

            if done or truncated:
                break

        # Update target network periodically
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        # Decay epsilon
        if agent.epsilon > EPSILON_END:
            agent.epsilon *= EPSILON_DECAY

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{NUM_EPISODES}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent


def evaluate_agent(agent, num_episodes=10):
    """Evaluate the trained agent."""
    env = gym.make('CartPole-v1')
    total_rewards = []

    print(f"\nEvaluating agent over {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor([state], dtype=torch.float32)
        total_reward = 0
        
        for t in range(MAX_STEPS):
            with torch.no_grad():
                action = agent.policy_net(state).argmax().item()
            state, reward, done, truncated, _ = env.step(action)
            state = torch.tensor([state], dtype=torch.float32)
            total_reward += reward
            
            if done or truncated:
                break
        
        total_rewards.append(total_reward)
        print(f"Episode {episode+1}: Reward = {total_reward}")

    env.close()
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward


if __name__ == "__main__":
    # Train the agent
    agent = train_agent()
    
    # Evaluate the agent
    evaluate_agent(agent, num_episodes=10)

