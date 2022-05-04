import torch
import torch.nn as nn

from itertools import product


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
                .log_prob(actions.squeeze(-1))
                .view(actions.size(0), -1)
                .sum(-1)
                .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Policy(nn.Module):
    def __init__(self, action_space, gamma, transe, edge_feat,
                 transe_hidden_dim, gnn_hidden_dim,
                 executor, num_processes,
                 include_executor,
                 freeze_encoder, freeze_executor,
                 cat_method="encoder_cat_executor",
                 sample_method="uniform",
                 num_neighbours=5,
                 transe2gnn=1, gnn_decoder=1, gnn_steps=1,
                 graph_detach=False):
        super(Policy, self).__init__()

        self.transe = transe
        self.encoder = transe.encoder
        self.graph_detach = graph_detach
        self.num_processes = num_processes
        self.gamma = gamma

        if freeze_encoder:
            print("Encoder: freeze")
            for param in transe.parameters():
                param.requires_grad = False
        else:
            print("Encoder: train")

        self.include_executor = include_executor
        if include_executor:
            print("Executor: included")
            self.transe2gnn = transe2gnn
            self.gnn_steps = gnn_steps
            if self.transe2gnn > 0:
                layers = []
                layers += [nn.Linear(transe_hidden_dim, gnn_hidden_dim)]
                for i in range(self.transe2gnn - 1):
                    layers += [nn.ReLU(), nn.Linear(gnn_hidden_dim, gnn_hidden_dim)]
                self.transe2gnn_fc = nn.Sequential(*layers)
            self.edge_proj = nn.Linear(edge_feat, gnn_hidden_dim)
            self.executor = executor
            self.gnn_decoder = gnn_decoder
            if self.gnn_decoder > 0:
                layers = []
                for i in range(self.gnn_decoder - 1):
                    layers += [nn.Linear(gnn_hidden_dim, gnn_hidden_dim), nn.ReLU()]
                layers += [nn.Linear(gnn_hidden_dim, gnn_hidden_dim)]
                self.gnn_decoder_fc = nn.Sequential(*layers)
            if freeze_executor:
                print("GNN: freeze")
                for param in executor.parameters():
                    param.requires_grad = False
            else:
                print("GNN: train")

            # Method to handle outputs from encoder and executor
            self.cat_method = cat_method
            if self.cat_method in ["executor_only", "encoder_add_executor"]:
                fc_input_dim = gnn_hidden_dim
            elif self.cat_method == "encoder_cat_executor":
                fc_input_dim = transe_hidden_dim + gnn_hidden_dim
            elif self.cat_method == "encoder_cat_executor_decode":
                fc_input_dim = gnn_hidden_dim
                self.cat_decoder_fc = nn.Linear(transe_hidden_dim + gnn_hidden_dim, gnn_hidden_dim)
            else:
                raise NotImplementedError
        else:
            fc_input_dim = transe_hidden_dim
            print("Executor: not included")

        if action_space.__class__.__name__ == 'Discrete':
            self.action_multidim = 1
            self.num_actions = action_space.n
            self.actor_linear = nn.Linear(fc_input_dim, self.num_actions)
        elif action_space.__class__.__name__ == 'MultiDiscrete':
            self.action_multidim = len(action_space.nvec)
            self.num_actions = action_space.nvec[0]
            self.actor_linear = nn.Linear(fc_input_dim, self.action_multidim * self.num_actions)
        else:
            raise NotImplementedError
        self.critic_linear = nn.Linear(fc_input_dim, 1)

        # How to expand the executor graph
        if self.include_executor:
            self.sample_method = sample_method  # choices = ["uniform", "learn_gaussian", "manual_gaussian",
                                                #            "learn_neighbour_policy", "reuse_actor_layer",
                                                #            "expand_all"]
            if self.sample_method == "learn_gaussian":
                self.mean_layer = nn.Linear(transe_hidden_dim, 1)
                self.std_layer = nn.Linear(transe_hidden_dim, 1)
            else:
                self.mean_layer = None
                self.std_layer = None

            if self.sample_method == "learn_neighbour_policy":
                self.neighbour_layer = nn.Linear(transe_hidden_dim, self.action_multidim * self.num_actions)
            elif self.sample_method == "reuse_actor_layer":
                self.neighbour_layer = self.actor_linear
            else:
                self.neighbour_layer = None

            if self.sample_method == "expand_all":
                self.num_neighbours = self.num_actions ** self.action_multidim
            else:
                self.num_neighbours = num_neighbours
        else:
            self.num_neighbours = num_neighbours

    def executor_layer(self, latents):
        num_states = latents.shape[0]
        if self.action_multidim == 1:
            # Expand graph by all possible actions
            node_features, senders, receivers, edge_features = create_graph(current_state=latents,
                                                                            model=self.transe,
                                                                            gamma=self.gamma,
                                                                            num_actions=self.num_actions,
                                                                            num_steps=self.gnn_steps,
                                                                            graph_detach=self.graph_detach)
        else:
            # Expand graph by only num_neighbour actions due to high dimensionality
            node_features, senders, receivers, edge_features = create_graph_selective(current_state=latents,
                                                                                      model=self.transe,
                                                                                      gamma=self.gamma,
                                                                                      action_bins=self.num_actions,
                                                                                      num_steps=self.gnn_steps,
                                                                                      action_multidim=self.action_multidim,
                                                                                      num_neighbours=self.num_neighbours,
                                                                                      graph_detach=self.graph_detach,
                                                                                      sample_method=self.sample_method,
                                                                                      mean_layer=self.mean_layer,
                                                                                      std_layer=self.std_layer,
                                                                                      neighbour_layer=self.neighbour_layer)

        if self.transe2gnn > 0:
            node_features = self.transe2gnn_fc(node_features)
        embedded_edge_features = self.edge_proj(edge_features)
        all_latents = self.executor(node_features, senders, receivers, embedded_edge_features)

        for i in range(self.gnn_steps - 1):
            all_latents = all_latents + node_features
            all_latents = self.executor(all_latents, senders, receivers, embedded_edge_features)

        if self.gnn_decoder > 0:
            all_latents = self.gnn_decoder_fc(all_latents)

        if self.cat_method == "executor_only":
            latents = all_latents[:num_states]
        elif self.cat_method == "encoder_cat_executor":
            latents = torch.cat((latents, all_latents[:num_states]), dim=-1)
        elif self.cat_method == "encoder_cat_executor_decode":
            latents = torch.cat((latents, all_latents[:num_states]), dim=-1)
            latents = self.cat_decoder_fc(latents)
        elif self.cat_method == "encoder_add_executor":
            latents = latents + all_latents[:num_states]
        else:
            raise NotImplementedError
        return latents

    def act(self, observations, deterministic):
        latents = self.encoder(observations)

        if self.include_executor:
            latents = self.executor_layer(latents)

        if self.action_multidim == 1:
            policy = self.actor_linear(latents)
            actor = FixedCategorical(logits=policy)
            if deterministic:
                action = actor.mode()
            else:
                action = actor.sample()
            log_probs = actor.log_probs(action)
            value = self.critic_linear(latents)
            return value, action, log_probs
        else:
            action_all_dims = []
            log_probs_all_dims = []

            # policy
            policy = self.actor_linear(latents)  # shape = (batch_size, action_multidim * num_actions)
            policy_dims = torch.split(policy, self.num_actions.item(), dim=-1)
            for i in range(self.action_multidim):
                actor = FixedCategorical(logits=policy_dims[i])
                if deterministic:
                    action = actor.mode()
                else:
                    action = actor.sample()
                log_probs = actor.log_probs(action)
                action_all_dims += [action]
                log_probs_all_dims += [log_probs]

            action_all_dims = torch.cat(action_all_dims, dim=-1)
            log_probs_all_dims = torch.cat(log_probs_all_dims, dim=-1)
            log_probs_all_dims = torch.sum(log_probs_all_dims, dim=-1).unsqueeze(dim=-1)

            # value
            value = self.critic_linear(latents)

            return value, action_all_dims, log_probs_all_dims

    def get_value(self, observations):
        latents = self.encoder(observations)
        if self.include_executor:
            latents = self.executor_layer(latents)
        return self.critic_linear(latents)

    def evaluate_actions(self, observations, action):
        latents = self.encoder(observations)

        if self.include_executor:
            latents = self.executor_layer(latents)

        if self.action_multidim == 1:
            policy = self.actor_linear(latents)
            actor = FixedCategorical(logits=policy)
            log_probs = actor.log_probs(action)
            value = self.critic_linear(latents)
            entropy = actor.entropy().mean()
            return value, log_probs, entropy
        else:
            log_probs_all_dims = []
            entropy = 0.0

            # policy
            policy = self.actor_linear(latents)  # shape = (batch_size, action_multidim * num_actions)
            policy_dims = torch.split(policy, self.num_actions.item(), dim=-1)
            for i in range(self.action_multidim):
                actor = FixedCategorical(logits=policy_dims[i])
                log_probs = actor.log_probs(action[:, i])
                log_probs_all_dims += [log_probs]
                entropy += actor.entropy().sum()
            log_probs_all_dims = torch.cat(log_probs_all_dims, dim=-1)
            log_probs_all_dims = torch.sum(log_probs_all_dims, dim=-1).unsqueeze(dim=-1)
            entropy = entropy / float(self.action_multidim * action.shape[0])

            # value
            value = self.critic_linear(latents)

            return value, log_probs_all_dims, entropy


def create_graph(current_state, model, gamma, num_actions, num_steps=2, graph_detach=False):
    device = current_state.device
    batch_size = current_state.shape[0]
    embed_dim = current_state.shape[-1]

    # one-hot encoding
    actions = torch.tensor([i for i in range(num_actions)], device=device)
    current_children = current_state.unsqueeze(1)
    children = [current_children]

    node_features = [current_state]
    edge_features = []
    senders = []
    receivers = []

    total_children = batch_size
    for i in range(num_steps):
        num_batches, num_children, _ = current_children.shape

        # stacked_children is (c1, c1, ..., c1, c2, c2, .., c2, ...)
        # each child repeated num_actions times + flattened for batches
        stacked_children = current_children.repeat((1, 1, num_actions)) \
            .reshape((num_batches * num_actions * num_children, embed_dim))

        # stacked action is (0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, ....)
        # repeated enough times for each child to have its action
        stacked_actions = actions.repeat((num_batches, num_children)) \
            .reshape((num_batches * num_children * num_actions))

        transitions = model.transition(stacked_children, stacked_actions)
        next_children = (stacked_children + transitions).reshape((num_batches, num_actions * num_children, embed_dim))
        if graph_detach:
            next_children = next_children.detach()
        children.append(next_children)
        current_children = next_children

        stacked_actions = stacked_actions.to(torch.long)
        one_hot_action = nn.functional.one_hot(stacked_actions, num_classes=num_actions)
        gamma_tens = torch.ones(one_hot_action.shape[0], 1, device=device) * gamma

        edge_features.append(torch.cat([one_hot_action, gamma_tens], -1))
        node_features.append(next_children.view(num_batches * num_actions * num_children, embed_dim))

        senders.append(torch.range(total_children, total_children + batch_size * num_actions * num_children - 1,
                                   dtype=torch.int64))
        receivers.append(torch.range(total_children - num_batches * num_children,
                                     total_children - 1, dtype=torch.int64).repeat((num_actions, 1)).T.flatten())

        total_children += batch_size * num_actions * num_children

    node_features = torch.cat(node_features).to(device)
    edge_features = torch.cat(edge_features).to(device)
    senders = torch.cat(senders).to(device)
    receivers = torch.cat(receivers).to(device)

    return node_features, senders, receivers, edge_features


def create_graph_selective(current_state, model, gamma, action_bins, num_steps,
                           action_multidim, num_neighbours, sample_method, graph_detach=False,
                           mean_layer=None, std_layer=None,
                           neighbour_layer=None):
    device = current_state.device
    batch_size = current_state.shape[0]
    embed_dim = current_state.shape[-1]

    current_children = current_state.unsqueeze(1)
    children = [current_children]

    node_features = [current_state]
    edge_features = []
    senders = []
    receivers = []

    total_children = batch_size
    for i in range(num_steps):
        num_batches, num_children, _ = current_children.shape

        # Repeat each child by num_neighbour times + flattened for batches
        # stacked_children: (c1, c1, ..., c1, c2, c2, .., c2, ...)
        # stacked_children.shape = (num_batches * num_neighbours * num_children, embed_dim)
        stacked_children = current_children.repeat((1, 1, num_neighbours)) \
            .reshape((num_batches * num_neighbours * num_children, embed_dim))

        # Sample an action vector for each stacked_children to expand
        # stacked_action.shape = (num_batches * num_neighbours * num_children, action_multidim)
        if sample_method == "uniform":
            # An action is sampled by sampling in each dimension with uniform distribution
            stacked_action = torch.randint(low=0, high=action_bins, size=(stacked_children.shape[0], action_multidim))
        elif sample_method in ["manual_gaussian", "learn_gaussian"]:
            # An action is sampled by sampling in each dimension with gaussian distribution
            if sample_method == "learn_gaussian":
                mean = mean_layer(stacked_children).repeat(1, action_multidim)
                std = std_layer(stacked_children).repeat(1, action_multidim)
                std = std ** 2
                stacked_action = torch.normal(mean=mean, std=std)
            else:
                mean = action_bins / 2
                std = action_bins / 4
                stacked_action = torch.normal(mean=mean, std=std, size=(stacked_children.shape[0], action_multidim))
            stacked_action = stacked_action.to(torch.long)
            # Make sure the actions are within range [0, action_bins-1]
            stacked_action = torch.where(stacked_action < 0, 0, stacked_action)
            stacked_action = torch.where(stacked_action >= action_bins, action_bins - 1, stacked_action)
        elif sample_method in ["learn_neighbour_policy", "reuse_actor_layer"]:
            # An action is sampled by sampling in each dimension using the actor layer
            policy = neighbour_layer(stacked_children)
            policy_dims = torch.split(policy, action_bins.item(), dim=-1)
            action_all_dims = []
            for d in range(action_multidim):
                actor = FixedCategorical(logits=policy_dims[d])
                action = actor.sample()
                action_all_dims += [action]
            action_all_dims = torch.cat(action_all_dims, dim=-1)
            stacked_action = action_all_dims
        elif sample_method == "expand_all":
            all_actions = list(product(range(action_bins), repeat=action_multidim))
            all_actions = torch.tensor(all_actions)
            stacked_action = all_actions.repeat(num_batches, num_children, 1) \
                .reshape((num_batches * num_children * num_neighbours, action_multidim))
            stacked_action = stacked_action.to(torch.long)
        else:
            raise NotImplementedError
        stacked_action = stacked_action.to(device)

        # Using TransE's transition function to predict next states
        transitions = model.transition(stacked_children, stacked_action)
        next_children = (stacked_children + transitions).reshape((num_batches, num_neighbours * num_children, embed_dim))
        if graph_detach:
            next_children = next_children.detach()
        children.append(next_children)
        current_children = next_children

        # Generate node and edge features
        action_one_hot = nn.functional.one_hot(stacked_action, num_classes=action_bins)
        action_one_hot = action_one_hot.view(stacked_action.shape[0], -1)
        gamma_tens = torch.ones(action_one_hot.shape[0], 1, device=device) * gamma

        edge_features.append(torch.cat([action_one_hot, gamma_tens], -1))
        node_features.append(next_children.view(num_batches * num_neighbours * num_children, embed_dim))

        senders.append(torch.range(total_children, total_children + batch_size * num_neighbours * num_children - 1,
                                   dtype=torch.int64))
        receivers.append(torch.range(total_children - num_batches * num_children,
                                     total_children - 1, dtype=torch.int64).repeat((num_neighbours, 1)).T.flatten())

        total_children += batch_size * num_neighbours * num_children

    node_features = torch.cat(node_features).to(device)
    edge_features = torch.cat(edge_features).to(device)
    senders = torch.cat(senders).to(device)
    receivers = torch.cat(receivers).to(device)

    return node_features, senders, receivers, edge_features