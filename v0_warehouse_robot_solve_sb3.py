import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import A2C

import v0_warehouse_robot_env  # noqa


def train_sb3(episodes, grid_rows=4, grid_cols=5, render=False):
    env = gym.make(
        "warehouse-robot-v0", grid_rows=grid_rows, grid_cols=grid_cols, render_mode="human" if render else None
    )
    env = TimeLimit(env, max_episode_steps=100)

    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="sb3_logs")

    TIMESTEPS = 1_000
    for i in range(episodes):
        print(f"Episode {i*TIMESTEPS}")

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"sb3_models/a2c_{i*TIMESTEPS}")


def test_sb3(episodes, grid_rows=4, grid_cols=5, render=True):
    env = gym.make(
        "warehouse-robot-v0", grid_rows=grid_rows, grid_cols=grid_cols, render_mode="human" if render else None
    )
    env = TimeLimit(env, max_episode_steps=100)

    model = A2C.load("sb3_models/a2c_2000", env=env)

    obs, _ = env.reset()
    terminated = truncated = False
    while not terminated and not truncated:
        action, _ = model.predict(observation=obs, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(action)


if __name__ == "__main__":
    GRID_ROWS = 4
    GRID_COLS = 5
    train_sb3(5, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, render=False)
    test_sb3(1, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, render=True)
