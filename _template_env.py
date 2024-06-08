import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env  # noqa
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

import numpy as np


register(
    id="template-v0",
    entry_point="_template_env:TemplateEnv",
)


class TemplateEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 1, shape=(2,), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs = np.array([0, 1], dtype=np.uint8)
        info = {}

        return obs, info

    def step(self, action):
        obs = np.array([0, 1], dtype=np.uint8)
        reward = 0
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass


if __name__ == "__main__":
    env = gym.make("template-v0", render_mode="human")
    env = TimeLimit(env, max_episode_steps=5)

    check_env(env.unwrapped)
