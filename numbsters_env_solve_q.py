import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

import numbsters_env  # noqa


def run_q(episodes, is_training=True, render=False, checkpoint=0, continue_with_pkl=False):
    env = gym.make("numbsters-v0", render_mode="human" if render else None)
    env = TimeLimit(env, max_episode_steps=20)

    if is_training and continue_with_pkl:
        f = open("numbsters_solution.pkl", "rb")
        q = pickle.load(f)
        f.close()
    elif is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open("numbsters_solution.pkl", "rb")
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        if not checkpoint or (checkpoint and i % checkpoint == 0):
            print(f"Episode {i}")

        rewards = 0

        state, _ = env.reset()
        terminated = truncated = False
        while not terminated and not truncated:
            if is_training and random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

            rewards += reward

        # epsilon = max(epsilon - 1 / episodes, 0)
        epsilon = 0.0001

        rewards_per_episode[i] = rewards

        if is_training and checkpoint and i % checkpoint == 0:
            save_png(episodes, rewards_per_episode)
            save_q(q)

    env.close()

    if is_training:
        save_png(episodes, rewards_per_episode)
        save_q(q)


def save_png(episodes, rewards_per_episode):
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100) : (t + 1)])
    plt.plot(sum_rewards)
    plt.savefig("numbsters_solution.png")


def save_q(q):
    f = open("numbsters_solution.pkl", "wb")
    pickle.dump(q, f)
    f.close()


if __name__ == "__main__":
    run_q(1_000_000, is_training=True, render=False, checkpoint=20_000, continue_with_pkl=False)
    run_q(1, is_training=False, render=True)
