from enum import Enum
import random


class RobotAction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


class GridTile(Enum):
    _FLOOR = 0
    ROBOT = 1
    TARGET = 2

    def __str__(self):
        return self.name[:1]


class WarehouseRobot:
    def __init__(self, grid_rows=4, grid_cols=5):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.reset()

    def reset(self, seed=None):
        self.robot_pos = [0, 0]

        random.seed(seed)
        self.target_pos = [random.randint(1, self.grid_rows - 1), random.randint(1, self.grid_cols - 1)]

    def render(self):
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if [r, c] == self.robot_pos:
                    print(GridTile.ROBOT, end=" ")
                elif [r, c] == self.target_pos:
                    print(GridTile.TARGET, end=" ")
                else:
                    print(GridTile._FLOOR, end=" ")

            print()
        print()


if __name__ == "__main__":
    wr = WarehouseRobot()
    wr.render()
