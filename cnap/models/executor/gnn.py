import torch
from torch import nn
from torch_scatter import scatter_max, scatter_sum


class MessagePassing(nn.Module):
    def __init__(self,
                 hidden_dimension,
                 neighbour_aggregation,
                 activation, layernorm):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.neighbour_aggregation = neighbour_aggregation
        self.activation = activation
        self.layernorm = layernorm

        # message function
        self.message_proj = nn.Linear(3 * hidden_dimension, hidden_dimension)

        # activation
        if self.activation:
            self.relu = nn.ReLU()

        # layer norm
        if self.layernorm:
            self.ln = nn.LayerNorm(hidden_dimension)

    def forward(self, node_features, senders, receivers, edge_features):
        # message
        message = self.message_proj(
            torch.cat((node_features[senders], node_features[receivers], edge_features), dim=-1))

        # activation
        if self.activation:
            message = self.relu(message)

        # neighbourhood aggregation
        if self.neighbour_aggregation == "sum":
            aggr_messages = torch.zeros((node_features.shape[0], self.hidden_dimension), device=node_features.device)
            aggr_messages.index_add_(0, receivers, message)
        elif self.neighbour_aggregation == "max":
            import warnings
            warnings.filterwarnings("ignore")
            aggr_messages = torch.ones((node_features.shape[0], self.hidden_dimension), device=node_features.device) \
                            * -1e9
            scatter_max(message, receivers, dim=0, out=aggr_messages)
            indegree = scatter_sum(torch.ones_like(message), receivers, dim=0, out=torch.zeros_like(aggr_messages))
            aggr_messages = aggr_messages * (indegree > 0)

        # update
        node_features += aggr_messages

        # layer norm
        if self.layernorm:
            node_features = self.ln(node_features)

        return node_features


class Executor(nn.Module):
    def __init__(self,
                 node_dimension, edge_dimension, hidden_dimension, out_dimension,
                 neighbour_aggregation,
                 activation, layernorm,
                 gnn_steps):
        super().__init__()

        self.hidden_dim = hidden_dimension
        self.gnn_steps = gnn_steps

        self.node_proj = nn.Sequential(nn.Linear(node_dimension, hidden_dimension, bias=False))
        self.edge_proj = nn.Linear(edge_dimension, hidden_dimension)
        self.message_passing = MessagePassing(hidden_dimension=hidden_dimension,
                                              neighbour_aggregation=neighbour_aggregation,
                                              activation=activation,
                                              layernorm=layernorm)
        self.fc = nn.Linear(in_features=hidden_dimension, out_features=out_dimension)

    def forward(self, data):
        node_features, (senders, receivers, edge_features) = data

        num_actions = node_features.shape[0]
        num_states = node_features.shape[1]

        # Projections of node and edge features
        node_features = self.node_proj(node_features)
        edge_features = self.edge_proj(edge_features)

        prev_updates = torch.zeros_like(node_features)
        predicted_value_updates = []

        for i in range(self.gnn_steps):
            x = prev_updates + node_features

            # Perform one step of message passing
            updates = self.message_passing(x.reshape(-1, self.hidden_dim),
                                           senders,
                                           receivers,
                                           edge_features)

            # Reshape to (action, state, 2)
            updates = updates.reshape(num_actions, num_states, -1)
            updates, _ = torch.max(updates, dim=0, keepdim=True)

            # Save the node features for current gnn_step
            prev_updates = updates
            predicted_value_updates += [self.fc(updates.squeeze())]

        return predicted_value_updates
