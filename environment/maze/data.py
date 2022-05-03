import numpy as np
import pickle
from collections import deque


def sort_maze_by_level(file, train=False):
    with np.load(file, mmap_mode='r') as f:
        if train:
            images = f['arr_0']
            start_x = f['arr_1']
            start_y = f['arr_2']
        else:
            images = f['arr_4']
            start_x = f['arr_5']
            start_y = f['arr_6']

    images = images.astype(np.float32)
    # images.shape: (34944, 2, 8, 8)
    # 2 => (grid, reward)

    num_images = images.shape[0]
    maze_size = images.shape[2]

    level_dict = {}  # level: [maze_indices], where level is min distance from start to goal
    dx = [0, 1, 1, 1, 0, -1, -1, -1]
    dy = [1, 1, 0, -1, -1, -1, 0, 1]

    for i in range(num_images):
        grid, reward = images[i]
        goal_x = np.argmax(reward) // maze_size
        goal_y = np.argmax(reward) % maze_size

        # compute length from (start_x, start_y) to (goal_x, goal_y)
        queue = deque()
        queue.append((start_x[i][0], start_y[i][0], 0))
        visited = np.zeros_like(grid)
        while len(queue):
            x, y, d = queue.popleft()
            for a in range(8):
                newx = x + dx[a]
                newy = y + dy[a]
                dist = d + 1
                if newx == goal_x and newy == goal_y:
                    if dist in level_dict:
                        level_dict[dist].append(i)
                    else:
                        level_dict[dist] = [i]
                    queue.clear()
                    break
                if 0 <= newx < maze_size and 0 <= newy < maze_size and \
                        not visited[newx][newy] and grid[newx][newy] == 0:
                    visited[newx][newy] = 1
                    queue.append((newx, newy, dist))

    if train:
        pickle.dump(level_dict, open(f"dataset/train_{maze_size}*{maze_size}_maze_by_level.pkl", "wb"))
    else:
        pickle.dump(level_dict, open(f"dataset/test_{maze_size}*{maze_size}_maze_by_level.pkl", "wb"))


if __name__ == "__main__":
    sort_maze_by_level("dataset/gridworld_8x8.npz", True)
    sort_maze_by_level("dataset/gridworld_8x8.npz", False)
