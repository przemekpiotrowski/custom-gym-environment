import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env  # noqa
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

import numpy as np

import src.numbsters.game as numbsters

register(
    id="numbsters-v0",
    entry_point="numbsters_env:NumbstersEnv",
)


class NumbstersEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([1, 1, 1, 1, 1, 1, 1]),
            high=np.array([18, 18, 18, 18, 18, 18, 18]),
            shape=(7,),
            dtype=np.uint8,
        )

        self.game = numbsters.Game()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game.setup(seed=seed)

        obs = np.array(self.game.stack, dtype=np.uint8)
        info = {}

        return obs, info

    def step(self, action):
        obs = np.array(self.game.stack, dtype=np.uint8)
        reward = 0
        terminated = False

        if self.game.ends():
            terminated = True
            if self.game.winning_condition():
                reward = 10
            else:
                reward = -10

        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass


if __name__ == "__main__":
    env = gym.make("numbsters-v0", render_mode="human")
    env = TimeLimit(env, max_episode_steps=5)

    # check_env(env.unwrapped)

    obs, _ = env.reset()
    print(f"{obs=}")
    terminated = truncated = False
    while not terminated and not truncated:
        # print(".", end="")
        rand_action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(rand_action)
        print(f"{obs=}")

    print(f"{terminated=}, {truncated=}")
