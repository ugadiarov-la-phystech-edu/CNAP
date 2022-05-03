import torch
import numpy as np

from models.executor.arguments import get_args
from models.executor.gnn import Executor
from models.executor.data import GenerateData
from models.executor.utils import find_policy


def loss_func(output, target):
    loss = (target.float().squeeze() - output.squeeze()) ** 2
    return loss.mean()


def training(num_graphs, graph_type, gnn_steps=1,
             num_states=20, num_actions=8,
             cartpole_depth=10, cartpole_delta=0.1,
             cartpole_accel=0.05, cartpole_thresh=0.5,
             cartpole_initial_position=0,
             epsilon=0.0001, discount=0.9,
             device=None):
    train_data_iterable = GenerateData(num_states=num_states,
                                       num_actions=num_actions,
                                       graph_type=graph_type,
                                       cartpole_depth=cartpole_depth,
                                       cartpole_delta=cartpole_delta,
                                       cartpole_accel=cartpole_accel,
                                       cartpole_thresh=cartpole_thresh,
                                       cartpole_initial_position=cartpole_initial_position,
                                       epsilon=epsilon,
                                       discount=discount,
                                       device=device)
    train_data_loader = torch.utils.data.DataLoader(train_data_iterable, batch_size=None)
    # loss_func = torch.nn.MSELoss()

    for num in range(num_graphs):
        print("Training graph = ", num)
        executor.train()

        training_data = next(iter(train_data_loader))
        node_features, edge_list, true_values, _ = training_data
        iterations = true_values.shape[0]

        for i in range(iterations - 1):
            optimizer.zero_grad()

            predicted_value_update = executor((node_features[i], edge_list))  # gnn_steps * num_states

            # Compute loss from the ground-truth update of value function between iterations
            # so that the MPNN (executor) learns to model value iteration algorithm
            loss = 0
            for j in range(min(gnn_steps, iterations - 1 - i)):
                loss += loss_func(predicted_value_update[j], true_values[i + j + 1] - true_values[i + j])

            loss.backward()
            optimizer.step()


def testing(num_graphs, graph_type, gnn_steps=1,
            num_states=20, num_actions=8,
            cartpole_depth=10, cartpole_delta=0.1,
            cartpole_accel=0.05, cartpole_thresh=0.5,
            cartpole_initial_position=0,
            epsilon=0.0001, discount=0.9,
            device=None):
    test_data_iterable = GenerateData(num_states=num_states,
                                      num_actions=num_actions,
                                      graph_type=graph_type,
                                      cartpole_depth=cartpole_depth,
                                      cartpole_delta=cartpole_delta,
                                      cartpole_accel=cartpole_accel,
                                      cartpole_thresh=cartpole_thresh,
                                      cartpole_initial_position=cartpole_initial_position,
                                      epsilon=epsilon,
                                      discount=discount,
                                      device=device)
    test_data_loader = torch.utils.data.DataLoader(test_data_iterable, batch_size=None)
    # loss_func = torch.nn.MSELoss()

    value_losses = [[] for _ in range(gnn_steps)]
    policy_accuracies = [[] for _ in range(gnn_steps)]

    for i in range(num_graphs):
        executor.eval()

        testing_data = next(iter(test_data_loader))
        node_features, edge_list, true_values, policy_dict = testing_data

        policy_dict['transition'] = policy_dict['transition'].to(device)
        policy_dict['reward'] = policy_dict['reward'].to(device)
        policy_dict['policy'] = policy_dict['policy'].to(device)

        true_value = true_values[-1]
        predicted_node_feature = node_features[0]

        predicted_value = torch.zeros(gnn_steps, node_features.shape[2], 1, device=device)  # gnn_steps * num_states * 1
        iterations = true_values.shape[0]
        num_actions = node_features.shape[1]

        for j in range(iterations - 1):
            predicted_value_update = executor((predicted_node_feature, edge_list))  # gnn_steps * num_states

            predicted_value[0] += predicted_value_update[0]
            for k in range(gnn_steps - 1):
                predicted_value[k + 1] = predicted_value[k] + predicted_value_update[k + 1]

            predicted_node_feature = torch.cat((predicted_value_update[0].unsqueeze(dim=0).repeat(num_actions, 1, 1) +
                                                predicted_node_feature[:, :, 0:1],
                                                node_features[j + 1, :, :, 1:2]), dim=-1)

            for k in range(min(gnn_steps, iterations - 1 - j)):
                value_losses[k] += [loss_func(predicted_value[k], true_value).detach().item()]
                predicted_policy = find_policy(policy_dict['transition'], policy_dict['reward'],
                                               policy_dict['discount'], predicted_value[k].squeeze())
                policy_accuracies[k] += [100.0 * torch.eq(predicted_policy, policy_dict['policy']).detach().sum()
                                         / len(predicted_value_update[k])]

    for i in range(gnn_steps):
        print("GNN_STEP = ", i)
        print("Last iteration: value loss mean = ", np.mean(np.array(value_losses[i])),
              " std = ", np.std(np.array(value_losses[i])))
        print("Last iteration: policy accuracy mean = ", np.mean(np.array(policy_accuracies[i])),
              " std = ", np.std(np.array(policy_accuracies[i])))


if __name__ == "__main__":
    args = get_args()

    if args.device == 'cuda':
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device("cpu")

    executor = Executor(node_dimension=2,
                        edge_dimension=2,
                        hidden_dimension=args.hidden_dimension,
                        out_dimension=1,
                        gnn_steps=args.gnn_steps,
                        activation=args.activation,
                        layernorm=args.layernorm,
                        neighbour_aggregation=args.neighbour_aggregation).to(args.device)

    train_params = list(filter(lambda p: p.requires_grad, executor.parameters()))
    optimizer = torch.optim.Adam(train_params, lr=args.lr)

    training(num_graphs=args.num_train_graphs,
             num_states=args.train_num_states,
             num_actions=args.train_num_actions,
             graph_type=args.graph_type,
             gnn_steps=args.gnn_steps,
             epsilon=args.epsilon,
             cartpole_depth=args.cartpole_depth,
             cartpole_delta=args.cartpole_delta,
             cartpole_accel=args.cartpole_accel,
             cartpole_thresh=args.cartpole_thresh,
             device=args.device)
    torch.save(executor.state_dict(),
               f"{args.graph_type}-gnnstep{args.gnn_steps}-act{args.activation}-ln{args.layernorm}.pt")

    if args.test:
        if not args.test_graph_cartpole:
            graph_types = ['erdos-renyi', 'barabasi-albert', 'star', 'caveman', 'caterpillar',
                           'lobster', 'tree', 'grid', 'ladder', 'line']
            for graph_type in graph_types:
                print(graph_type)
                print("#1 - 20*5:")
                testing(num_graphs=40, num_states=20, num_actions=5, graph_type=graph_type, gnn_steps=args.gnn_steps,
                        device=args.device)
                print("#2 - 1:")
                testing(num_graphs=40, num_states=50, num_actions=10, graph_type=graph_type, gnn_steps=args.gnn_steps,
                        device=args.device)
                print("#3 - 100*20")
                testing(num_graphs=30, num_states=100, num_actions=20, graph_type=graph_type, gnn_steps=args.gnn_steps,
                        device=args.device)
        else:
            initial_positions = [i for i in np.arange(-args.cartpole_thresh, args.cartpole_thresh, 0.05)]
            for initial_position in initial_positions:
                testing(num_graphs=1, graph_type="cartpole", gnn_steps=args.gnn_steps,
                        cartpole_depth=args.cartpole_depth,
                        cartpole_delta=args.cartpole_delta,
                        cartpole_accel=args.cartpole_accel,
                        cartpole_thresh=args.cartpole_thresh,
                        cartpole_initial_position=initial_position)
