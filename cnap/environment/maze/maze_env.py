import numpy as np
import gym
from gym import spaces


def flatten_action(action):
    if not isinstance(action, np.int64):
        assert len(action) == 1
        return action[0]
    else:
        return action


class MazeEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self,
                 maze_size,
                 train,
                 maze_indices_list=None,
                 active_index=None):
        self.maze_size = maze_size
        self.maze_indices_list = maze_indices_list
        self.active_index = active_index
        self.finished = False

        self.dx = [0, 1, 1, 1, 0, -1, -1, -1]
        self.dy = [1, 1, 0, -1, -1, -1, 0, 1]
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=maze_size, shape=(3, maze_size, maze_size), dtype=np.float32)

        if maze_size == 8:
            file = 'cnap/environment/maze/dataset/gridworld_8x8.npz'
        else:
            raise NotImplementedError

        with np.load(file, mmap_mode='r') as f:
            if train:
                self.images = f['arr_0']
                self.start_x = f['arr_1']
                self.start_y = f['arr_2']
            else:
                self.images = f['arr_4']
                self.start_x = f['arr_5']
                self.start_y = f['arr_6']

    def step(self, action):
        if self.active_index is not None and self.finished:
            reward = -3.
            return self.state, reward, False, self.state

        newx = self.x + self.dx[flatten_action(action)]
        newy = self.y + self.dy[flatten_action(action)]
        solved_pos = None

        if 0 <= newx < self.maze_size and 0 <= newy < self.maze_size:
            if False:
                done = True
            # if self.visited[newx, newy]:
                # Failed (visited repeated position)
                # done = True
                # reward = -1.0
                # solved_pos = (self.x, self.y)
                # self.reset()
            else:
                self.visited[newx, newy] = 1
                if self.grid[newx, newy] == self.grid[self.x, self.y]:
                    # Move a step
                    self.x = newx
                    self.y = newy
                    if self.x == self.goal_x and self.y == self.goal_y:
                        # Done (reach goal)
                        done = True
                        reward = 1.0
                        solved_pos = (self.x, self.y)
                        self.reset()
                    else:
                        # Continue
                        done = False
                        reward = -0.01
                else:
                    # Failed (reach obstacle)
                    done = True
                    reward = -1.0
                    solved_pos = (self.x, self.y)
                    self.reset()
        else:
            # Failed (outside grid boundary)
            done = True
            reward = -1.0
            solved_pos = (self.x, self.y)
            self.reset()

        self.position_map = np.zeros_like(self.grid)
        self.position_map[self.x, self.y] = 1
        self.goal_map = np.zeros_like(self.position_map)
        self.goal_map[self.goal_x, self.goal_y] = 10
        self.state = np.stack((self.grid, self.position_map, self.goal_map), axis=0)

        if solved_pos:
            position_map = np.zeros_like(self.grid)
            position_map[solved_pos[0], solved_pos[1]] = 1
            solved_state = {"solved": np.stack((self.grid, position_map, self.goal_map))}
        else:
            solved_state = {"solved": np.zeros_like(self.state)}

        return self.state, reward, done, solved_state

    def reset(self, maze_index=None):
        if maze_index is None and self.maze_indices_list is not None:
            if self.active_index is not None:
                if self.active_index >= len(self.maze_indices_list):
                    id = 0
                    self.finished = True
                else:
                    id = self.active_index
                self.active_index += 1
            else:
                id = np.random.randint(len(self.maze_indices_list))
            if len(self.maze_indices_list) == 0:
                maze_index = 0
            else:
                maze_index = self.maze_indices_list[id]
        elif maze_index is None:
            maze_index = np.random.randint(self.images.shape[0])

        # grid
        self.grid = np.copy(self.images[maze_index][0])

        # initial position
        self.x = self.start_x[maze_index]
        self.y = self.start_y[maze_index]
        self.position_map = np.zeros_like(self.grid)
        self.position_map[self.x, self.y] = 1

        # goal
        self.goal_map = self.images[maze_index][1]
        self.goal_x = np.argmax(self.goal_map) // self.maze_size
        self.goal_y = np.argmax(self.goal_map) % self.maze_size

        # state = (grid, position, goal)
        self.state = np.stack((self.grid, self.position_map, self.goal_map), axis=0)

        # fail if visited
        self.visited = np.zeros_like(self.grid)
        self.visited[self.x, self.y] = 1

        self.done = False

        return self.state

    def render(self, close=False):
        print(f"Current position at ({self.x}, {self.y})")
        print(f"Goal at ({self.goal_x}, {self.y})")
