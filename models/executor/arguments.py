import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Executor')

    # Executor parameters
    parser.add_argument('--hidden_dimension', type=int, default=50, help='dimension of hidden layers')
    parser.add_argument('--gnn_steps', type=int, default=1, help='depth of gnn')
    parser.add_argument('--activation', type=bool, default=False, help='whether to include message activation')
    parser.add_argument('--layernorm', type=bool, default=True, help='whether to include layer norm')
    parser.add_argument('--neighbour_aggregation', type=str, default='sum', help='neighbour aggregation method: sum/max')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')

    # GraphData parameters
    parser.add_argument('--num_train_graphs', type=int, default=500, help='number of graphs to train')
    parser.add_argument('--train_num_states', type=int, default=20, help='number of states for training')
    parser.add_argument('--train_num_actions', type=int, default=8, help='number of actions for training')
    parser.add_argument('--epsilon', type=float, default=0.0001, help='epsilon')
    parser.add_argument('--graph_type', type=str, default='erdos-renyi', help='type of graphs for training')
    # cartpole graph parameters
    parser.add_argument('--cartpole_depth', type=int, default=10)
    parser.add_argument('--cartpole_delta', type=float, default=0.1)
    parser.add_argument('--cartpole_accel', type=float, default=0.05)
    parser.add_argument('--cartpole_thresh', type=float, default=0.5)

    # Testing parameters
    parser.add_argument('--test', type=bool, default=False, help='whether to include testing stage')
    parser.add_argument('--test_graph_cartpole', type=bool, default=False, help='True: test on cartpole graphs; False: test on synthetic graphs')

    # Device parameters
    parser.add_argument('--device', type=str, default='cpu', help='use "cuda" if GPU is available')

    args = parser.parse_args()
    return args
