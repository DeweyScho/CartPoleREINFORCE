#Dewey Schoenfelder
import sys
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
from Network import Network
import torch.optim as optim
import torch.nn as nn
import os
from time import sleep
import numpy as np
import torch.nn.functional as F
import torch.distributions as distributions

env = gym.make("CartPole-v1")
test_env = gym.make('CartPole-v1')
plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPISODES = 500
GAMMA = 0.99  # discount factor
N_TRIALS = 25
REWARD_THRESHOLD = 400
PRINT_EVERY = 10
LR = 0.001

train_rewards = []
test_rewards = []

def main():
    test_categorial_dist()

    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    policy_net = Network(n_observations, n_actions).to(device)
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    res = train(policy_net, optimizer)  # Uncomment to train

    model = load_model('checkpoint/CartPoleREINFORCE_model.pt', device)
    play(model)

def load_model(path: str, device: torch.device):
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)
    model = Network(n_observations, n_actions).to(device)
    checkpoint_data = torch.load(path)
    model.load_state_dict(checkpoint_data['state_dict'])
    model.eval()
    return model

def play(model):
    env = gym.make('CartPole-v1', render_mode="human")
    env.reset()
    state = env.reset()[0]
    for _ in range(200):
        action = select_action_play(model, state)
        state, reward, done, _, _ = env.step(action)
        env.render()
        sleep(0.1)
        if done:
            break
    env.close()

def train(policy_net, optimizer):
    for episode in range(1, MAX_EPISODES + 1):
        loss, train_reward = train_one_episode(policy_net, optimizer, GAMMA)
        test_reward = compute_reward_one_episode(policy_net)

        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')

        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break

    print("----------saving model in file-----------------")
    checkpoint_data = {
        'epoch': MAX_EPISODES,
        'state_dict': policy_net.state_dict(),
    }
    ckpt_path = os.path.join("checkpoint/CartPoleREINFORCE_model.pt")
    torch.save(checkpoint_data, ckpt_path)
    plot_rewards(train_rewards, test_rewards)

def compute_reward_one_episode(policy_net):
    policy_net.eval()
    done = False
    episode_reward = 0
    state = test_env.reset()[0]
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_pred = policy_net(state)
            action_prob = F.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
        state, reward, done, _, _ = test_env.step(action.item())
        episode_reward += reward
    return episode_reward

def train_one_episode(policy_net, optimizer, discount_factor):
    policy_net.train()
    log_prob_actions = []
    rewards = []
    done = False
    episode_reward = 0
    state = env.reset()[0]
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_pred = policy_net(state)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        state, reward, done, _, _ = env.step(action.item())

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        episode_reward += reward

    log_prob_actions = torch.cat(log_prob_actions)
    discounted_rewards = calculate_discounted_rewards(rewards, discount_factor)
    loss = update_policy(discounted_rewards, log_prob_actions, optimizer)
    return loss, episode_reward

def calculate_discounted_rewards(rewards, discount_factor, normalize=True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns

def update_policy(discounted_rewards, log_prob_actions, optimizer):
    loss = - (discounted_rewards.to(device) * log_prob_actions.to(device)).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def plot_rewards(train_rewards, test_rewards):
    import matplotlib
    matplotlib.use("Agg")  # avoid crash
    import matplotlib.pyplot as plt
    import os

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Test Reward')
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r', label='Reward Threshold')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/rewards_plot.png")
    plt.close()


def select_action_play(model, state):
    if len(state) != 4:
        state = state[0]
    state = torch.Tensor(state).to(device)
    with torch.no_grad():
        values = model(state)
        action = np.argmax(values.cpu().numpy())
    return action

def test_categorial_dist():
    action_logits = torch.rand(5)
    action_probs = F.softmax(action_logits, dim=-1)
    print(action_probs)
    dist = distributions.Categorical(action_probs)
    action = dist.sample()
    print(action)
    print(dist.log_prob(action), torch.log(action_probs[action]))

if __name__ == "__main__":
    sys.exit(int(main() or 0))

