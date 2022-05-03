import torch
import numpy as np
import pickle
from gym import spaces
from sklearn.linear_model import LinearRegression

from models.transe.transe_maze import TransE
from models.executor.gnn import Executor
from models.ppo.policy import Policy
from environment.maze.maze_env import MazeEnv


def generate_mdp(grid, goal_map):
    # States
    states = []
    index_map = {}  # (x, y) -> i, where states[i] = (x, y)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 0.0:
                index_map[(x, y)] = len(states)
                states.append((x, y))
    num_states = len(states) + 1  # plus 1 done state

    # Actions
    num_actions = 8
    dx = [1, 0, -1, 0, 1, 1, -1, -1]
    dy = [0, 1, 0, -1, 1, -1, 1, -1]

    # Transition and Reward matrices
    P = torch.zeros((num_actions, num_states, num_states))
    R = torch.zeros((num_states, num_actions))

    done = len(states)
    for a in range(num_actions):
        P[a][done][done] = 1.0
        R[done][a] = -1.0

    for s, (x, y) in enumerate(states):
        for a in range(num_actions):
            if goal_map[x][y] > 0.0:
                P[a][s][s] = 1.0
                R[s][a] = 1.0
            else:
                newx = x + dx[a]
                newy = y + dy[a]
                if (newx, newy) not in index_map:
                    news = done
                    newr = -1.0
                else:
                    news = index_map[(newx, newy)]
                    if goal_map[newx, newy] > 0.0:
                        newr = 1.0
                    else:
                        newr = -0.01
                P[a][s][news] = 1.0
                R[s][a] = newr
    return P, R, index_map


def value_iteration(P, R, discount, epsilon=1e-8):
    num_states = P.shape[-1]
    value = torch.zeros(num_states)
    while True:
        state_values = torch.add(R, discount * torch.einsum('ijk,k->ji', P, value))
        next_value, _ = torch.max(state_values, dim=1)
        # stop when converge
        diff = torch.linalg.vector_norm(value - next_value)
        if diff < epsilon:
            break
        value = next_value
    return next_value


def v_star_predictability(transe, executor, policy, test_indices, test_maze_size=8):
    all_test_indices = []
    for level in sorted(test_indices.keys()):
        all_test_indices.extend(test_indices[level])
    print("In total: ", len(all_test_indices))

    for level in sorted(test_indices.keys()):
        print("Level ", level, "has ", len(test_indices[level]), " mazes.")

        # Load pre-trained agent using training mazes up to current level
        transe.load_state_dict(torch.load(f"trained/trained_model_level{level}.pt", map_location=torch.device('cpu'))['transe'])
        executor.load_state_dict(torch.load(f"trained/trained_model_level{level}.pt", map_location=torch.device('cpu'))['executor'])
        policy.load_state_dict(torch.load(f"trained/trained_model_level{level}.pt", map_location=torch.device('cpu'))['policy'])

        test_env = MazeEnv(maze_size=test_maze_size, maze_indices_list=all_test_indices, train=False)

        for i in range(len(all_test_indices)):
            # print("Testing on maze: ", i)
            maze_index = all_test_indices[i]
            grid, position_map, goal_map = test_env.reset(maze_index)
            P, R, state_index_map = generate_mdp(grid=grid, goal_map=goal_map)
            true_value = value_iteration(P=P, R=R, discount=0.9)

            all_pre_gnn_latents = []
            all_post_gnn_latents = []
            timestep_values = []

            with torch.no_grad():
                for key in state_index_map:
                    x, y = key
                    state_index = state_index_map[key]
                    position_map = np.zeros_like(position_map)
                    position_map[x][y] = 1
                    obs = np.stack((grid, position_map, goal_map), axis=0)

                    pre_gnn_latents = policy.encoder(torch.Tensor(obs).unsqueeze(dim=0))
                    post_gnn_latents = policy.executor_layer(pre_gnn_latents)

                    all_pre_gnn_latents.append(np.asarray(pre_gnn_latents.squeeze_()))
                    all_post_gnn_latents.append(np.asarray(post_gnn_latents.squeeze_()))
                    timestep_values.append(float(true_value[state_index]))

        pre_reg = LinearRegression().fit(all_pre_gnn_latents, timestep_values)
        print("Before gnn: ", pre_reg.score(all_pre_gnn_latents, timestep_values))

        post_reg = LinearRegression().fit(all_post_gnn_latents, timestep_values)
        print("After gnn: ", post_reg.score(all_post_gnn_latents, timestep_values))


if __name__ == "__main__":
    " Parameters "
    # TransE
    env_input_dimensions = (3, 8, 8)
    action_dimension = 8
    transe_hidden_dimension = 128
    state_embedding_dimension = 10
    # Executor
    gnn_steps = 3
    activation = False
    layernorm = True
    neighbour_aggregation = 'max'
    executor_hidden_dimension = 10
    # Policy
    gamma = 0.99
    freeze_encoder = False
    freeze_executor = True
    transe2gnn = 1
    gnn_decoder = 1
    graph_detach = False
    # Env
    env_action_dim = 8
    action_space = spaces.Discrete(env_action_dim)
    num_processes = 15
    test_indices = pickle.load(open('dataset/test_8*8_maze_by_level.pkl', 'rb'))
    " End of parameters"

    include_executor = True

    transe = TransE(state_embedding_dimension=state_embedding_dimension,
                    input_dimensions=env_input_dimensions,
                    action_dimension=action_dimension,
                    hidden_dimension=transe_hidden_dimension)

    executor = Executor(node_dimension=2,
                        edge_dimension=2,
                        hidden_dimension=executor_hidden_dimension,
                        out_dimension=1,
                        neighbour_aggregation=neighbour_aggregation,
                        activation=activation,
                        layernorm=layernorm,
                        gnn_steps=gnn_steps)

    policy = Policy(action_space=action_space,
                    gamma=gamma,
                    transe=transe,
                    edge_feat=env_action_dim + 1,
                    transe_hidden_dim=state_embedding_dimension,
                    gnn_hidden_dim=executor_hidden_dimension,
                    num_processes=num_processes,
                    include_executor=include_executor,
                    executor=executor.message_passing,
                    freeze_encoder=freeze_encoder,
                    freeze_executor=freeze_executor,
                    transe2gnn=transe2gnn,
                    gnn_decoder=gnn_decoder,
                    gnn_steps=gnn_steps,
                    graph_detach=graph_detach)

    v_star_predictability(transe=transe,
                          executor=executor,
                          policy=policy,
                          test_indices=test_indices)
