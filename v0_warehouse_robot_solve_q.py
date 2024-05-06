import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

import v0_warehouse_robot_env  # noqa


def run_q(episodes, is_training=True, grid_rows=4, grid_cols=5, render=False):
    env = gym.make(
        "warehouse-robot-v0", grid_rows=grid_rows, grid_cols=grid_cols, render_mode="human" if render else None
    )
    env = TimeLimit(env, max_episode_steps=100)

    if is_training:
        q = np.zeros(
            (
                env.unwrapped.grid_rows,
                env.unwrapped.grid_cols,
                env.unwrapped.grid_rows,
                env.unwrapped.grid_cols,
                env.action_space.n,
            )
        )
    else:
        f = open("v0_warehouse_robot_solution.pkl", "rb")
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1

    steps_per_episode = np.zeros(episodes)

    step_count = 0
    for i in range(episodes):
        print(f"Episode {i}")

        state, _ = env.reset()
        terminated = truncated = False
        while not terminated and not truncated:
            if is_training and random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_state_idx = tuple(state)
                action = np.argmax(q[q_state_idx])

            new_state, reward, terminated, truncated, _ = env.step(action)

            q_state_action_idx = tuple(state) + (action,)
            q_new_state_idx = tuple(new_state)

            if is_training:
                q[q_state_action_idx] = q[q_state_action_idx] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[q_new_state_idx]) - q[q_state_action_idx]
                )

            state = new_state

            step_count += 1
            if terminated or truncated:
                steps_per_episode[i] = step_count
                step_count = 0

        epsilon = max(epsilon - 1 / episodes, 0)

    env.close()

    if is_training:
        sum_steps = np.zeros(episodes)
        for t in range(episodes):
            sum_steps[t] = np.mean(steps_per_episode[max(0, t - 100) : (t + 1)])
        plt.plot(sum_steps)
        plt.savefig("v0_warehouse_robot_solution.png")

    if is_training:
        f = open("v0_warehouse_robot_solution.pkl", "wb")
        pickle.dump(q, f)
        f.close()


if __name__ == "__main__":
    GRID_ROWS = 5
    GRID_COLS = 6
    run_q(2000, is_training=True, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, render=False)
    run_q(1, is_training=False, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, render=True)
