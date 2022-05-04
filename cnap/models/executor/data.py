import torch
import networkx as nx

from cnap.models.executor.utils import GenerateGraph, cartpole_graph, find_policy


class GenerateData(torch.utils.data.IterableDataset):
    def __init__(self,
                 num_states, num_actions,
                 graph_type,
                 cartpole_depth, cartpole_delta,
                 cartpole_accel, cartpole_thresh,
                 cartpole_initial_position,
                 epsilon, discount,
                 device=None):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.cartpole_depth = cartpole_depth
        self.cartpole_delta = cartpole_delta
        self.cartpole_accel = cartpole_accel
        self.cartpole_thresh = cartpole_thresh
        self.cartpole_initial_position = cartpole_initial_position
        self.discount = discount
        self.epsilon = epsilon
        self.device = device

        if graph_type == 'cartpole':
            self.cartpole_graph = True
        else:
            self.cartpole_graph = False
            self.graph_generator = GenerateGraph(graph_type=graph_type,
                                                 size=self.num_states,
                                                 degree=int(self.num_states * 0.5))

    def _transition(self):
        """
            Randomly generate transition probabilities: P(s'|s, a)
            Output size: transition -> action * state * state
        """
        # Generate random edges according to graph type
        adjacency_matrix = []
        for _ in range(self.num_actions):
            adjacency_matrix.append(torch.Tensor(
                nx.adjacency_matrix(self.graph_generator.generate_graph()).todense()
            ))
        adjacency_matrix = torch.stack(adjacency_matrix, dim=0)

        # Generate random probabilities to the edges
        transition = torch.rand(self.num_actions, self.num_states, self.num_states)
        transition *= adjacency_matrix

        # Normalise the probabilities at receiver dimension (s')
        transition /= torch.sum(transition, dim=-1, keepdim=True)
        return transition

    def _reward(self):
        """
            Randomly generate rewards: r(s, a)
            Output size: reward -> state * action
        """
        # Reward range in [0, 1]
        reward = torch.rand(self.num_states, self.num_actions)
        return reward

    def _value_iteration(self, transition, reward):
        """
            Executing value iteration update rule until convergence:
                v[t+1](s) = max(r(s,a) + gamma * sum(p(s'|s,a) * v[t](s)))
            Output size: true_values -> iteration * state
        """
        num_states = transition.shape[-1]
        value = torch.zeros(num_states)
        true_values = [value]

        while True:
            state_values = torch.add(reward,
                                     self.discount * torch.einsum('ijk,k->ji', transition, value))
            next_value, _ = torch.max(state_values, dim=1)

            # stop when converge
            diff = torch.linalg.vector_norm(value - next_value)
            if diff < self.epsilon:
                break

            true_values.append(next_value)
            value = next_value

        return torch.stack(true_values, dim=0)

    def _generate_data(self):
        """
            Generate node_features and edge_features:
                node_features x(s) = (v(s), r(s,a))
                edge_features e(s,s') = (gamma, p(s'|s,a))
            Output size:
                node_features -> iteration * action * state * 2
                edge_list:
                    senders -> edge
                    receivers -> edge
                    edge_features -> edge * 2
                true_values -> iteration * state
        """
        # Generate transition, reward based on graph_type
        if self.cartpole_graph:
            transition, reward = cartpole_graph(depth=self.cartpole_depth,
                                                delta=self.cartpole_delta,
                                                accel=self.cartpole_accel,
                                                thresh=self.cartpole_thresh,
                                                initial_position=self.cartpole_initial_position)
        else:
            valid = False
            while not valid:
                transition = self._transition()
                reward = self._reward()
                valid = not torch.isnan(transition).any()

        # Generate true_values using value iteration
        true_values = self._value_iteration(transition=transition, reward=reward).to(self.device)

        # Recalculate size (to accommodate cartpole_graph)
        num_states = reward.shape[0]
        num_actions = reward.shape[1]

        # Generate policy dict
        policy = find_policy(transition, reward, self.discount, true_values[-1])
        policy_dict = {
            'transition': transition.to(self.device),
            'reward': reward.to(self.device),
            'discount': self.discount,
            'policy': policy.to(self.device)
        }

        # Generate node_features
        resized_true_values = true_values.unsqueeze(dim=1).repeat(1, num_actions, 1).unsqueeze(dim=-1)
        reward = reward.transpose(dim0=0, dim1=1)
        resized_reward = reward.unsqueeze(dim=0).repeat(true_values.shape[0], 1, 1).unsqueeze(dim=-1)
        node_features = torch.cat((resized_true_values, resized_reward), dim=-1).to(self.device)

        # Generate edge_list
        transition = transition.transpose(dim0=-1, dim1=-2)
        senders = []
        receivers = []
        edge_features = []

        for a in range(num_actions):
            for s_i in range(num_states):
                for s_j in range(num_states):
                    if transition[a][s_i][s_j] > 0:
                        senders.append(a * num_states + s_i)
                        receivers.append(a * num_states + s_j)
                        edge_features.append([self.discount, transition[a][s_i][s_j]])

        senders = torch.LongTensor(senders).to(self.device)
        receivers = torch.LongTensor(receivers).to(self.device)
        edge_features = torch.Tensor(edge_features).to(self.device)
        edge_list = (senders, receivers, edge_features)

        yield node_features, edge_list, true_values, policy_dict

    def __iter__(self):
        return self._generate_data()

