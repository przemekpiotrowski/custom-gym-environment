import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env  # noqa
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

import numpy as np

import src.numbsters.game as numbsters
from src.numbsters.cards import create_deck

register(
    id="numbsters-v0",
    entry_point="numbsters_env:NumbstersEnv",
)


class NumbstersEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.game_deck = create_deck()
        self.game_stack_len = min(len(self.game_deck), 8)
        self.game_actions = []
        for pos_from in range(self.game_stack_len):
            for pos_to in range(self.game_stack_len + 1):
                self.game_actions.append(f"move_{pos_from+1}_{pos_to}")

        self.action_space = spaces.Discrete(len(self.game_actions))
        self.observation_space = spaces.Box(
            low=np.ones(self.game_stack_len, dtype=np.uint8),
            high=np.ones(self.game_stack_len, dtype=np.uint8) * 18,
            shape=(self.game_stack_len,),
            dtype=np.uint8,
        )

        self.game = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game = numbsters.Game(deck=self.game_deck.copy())
        self.game.setup()
        self.game.draw()

        obs = np.array(self.game.stack, dtype=np.uint8)
        info = {}

        return obs, info

    def step(self, action):
        game_action_fn, pos_from, pos_to = self.game_actions[action].split("_")
        game_action_ok = getattr(self.game, game_action_fn)(int(pos_from), int(pos_to))

        reward = 0
        terminated = False
        truncated = False

        debug = []
        if game_action_ok:
            debug.append("action ok")

            no_eating = False
            if self.game.eat():
                debug.append("eating ok")
            else:
                debug.append("no eating")
                no_eating = True

            if no_eating or self.game.ends():
                debug.append("game ends")

                terminated = True
                if self.game.winning_condition():
                    reward = 10
                else:
                    reward = -10
            else:
                debug.append("game continues")
                self.game.draw()
        else:
            debug.append("invalid action")
            reward = -1

        info = {"debug": debug}
        obs = np.array(self.game.stack, dtype=np.uint8)

        return obs, reward, terminated, truncated, info

    def render(self):
        pass


if __name__ == "__main__":
    env = gym.make("numbsters-v0", render_mode="human")
    env = TimeLimit(env, max_episode_steps=5)

    check_env(env.unwrapped)
    print("*** env check end ***")

    obs, _ = env.reset()
    print(obs)
    terminated = truncated = False
    while not terminated and not truncated:
        # print(".", end="")
        rand_action = env.action_space.sample()
        print(env.unwrapped.game_actions[rand_action], end=", ")
        obs, reward, terminated, truncated, info = env.step(rand_action)
        print(info["debug"], f"{reward=}", obs)

    print(f"{terminated=}, {truncated=}")
