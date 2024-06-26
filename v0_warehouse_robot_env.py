import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env  # noqa
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

import numpy as np

import v0_warehouse_robot as wr

register(
    id="warehouse-robot-v0",
    entry_point="v0_warehouse_robot_env:WarehouseRobotEnv",
)


class WarehouseRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, grid_rows=4, grid_cols=5, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(wr.RobotAction))
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.grid_rows - 1, self.grid_cols - 1, self.grid_rows - 1, self.grid_cols - 1]),
            shape=(4,),
            dtype=np.int32,
        )

        self.wr = wr.WarehouseRobot(grid_rows=grid_rows, grid_cols=grid_cols)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.wr.reset(seed=seed)

        obs = np.concatenate((self.wr.robot_pos, self.wr.target_pos), dtype=np.int32)
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def render(self):
        self.wr.render()

    def step(self, action):
        reward = -1
        terminated = False
        truncated = False
        obs = np.concatenate((self.wr.robot_pos, self.wr.target_pos), dtype=np.int32)
        info = {}

        target_reached = self.wr.perform_action(wr.RobotAction(action))
        if target_reached:
            reward = 20
            terminated = True

        if self.render_mode == "human":
            print(wr.RobotAction(action))
            self.render()

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = gym.make("warehouse-robot-v0", grid_rows=10, grid_cols=10, render_mode="human")
    env = TimeLimit(env, max_episode_steps=20)

    # check_env(env.unwrapped)

    env.reset()
    terminated = truncated = False
    while not terminated and not truncated:
        rand_action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(rand_action)

    print(f"{terminated=}, {truncated=}")
