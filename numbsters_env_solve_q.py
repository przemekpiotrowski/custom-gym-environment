import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import numpy as np
import random
import matplotlib.pyplot as plt

import numbsters_env  # noqa


def run_q(episodes, is_training=True, stack_size=8, deck_size=18, render=False, checkpoint=0, continue_with_pkl=False):
    env = gym.make("numbsters-v0", stack_size=stack_size, deck_size=deck_size, render_mode="human" if render else None)
    env = TimeLimit(env, max_episode_steps=20)

    solution_filename = f"q_solutions/numbsters-{stack_size}x{deck_size}"
    if is_training and continue_with_pkl:
        q = load_q(solution_filename)
    elif is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float16)
    else:
        q = load_q(solution_filename)

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1

    rewards_per_episode = np.zeros(episodes, dtype=np.int16)

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
            save_png(episodes, rewards_per_episode, solution_filename)
            # save_q(q, solution_filename)

    env.close()

    if is_training:
        save_png(episodes, rewards_per_episode, solution_filename)
        save_q(q, solution_filename)


def save_png(episodes, rewards_per_episode, cache_filename):
    sum_rewards = np.zeros(episodes, dtype=np.int16)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100) : (t + 1)])
    plt.plot(sum_rewards)
    plt.savefig(cache_filename + ".png")


def save_q(q, cache_filename):
    print("Saving the Q-table...")
    np.save(cache_filename + ".npy", q, allow_pickle=False)
    # np.savetxt(cache_filename + ".txt.gz", q)


def load_q(cache_filename):
    print("Loading the Q-table...")
    return np.load(cache_filename + ".npy", allow_pickle=False)
    # return np.loadtxt(cache_filename + ".txt.gz", dtype=np.float16)


if __name__ == "__main__":
    STACK_SIZE = 6
    DECK_SIZE = 18
    run_q(
        5_000_000,
        is_training=True,
        stack_size=STACK_SIZE,
        deck_size=DECK_SIZE,
        render=False,
        checkpoint=100_000,
        continue_with_pkl=True,
    )
    run_q(1, is_training=False, stack_size=STACK_SIZE, deck_size=DECK_SIZE, render=True)
