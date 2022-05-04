import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, hidden_dimension, state_embedding_dimension,
                 width_height, num_channels):
        super().__init__()

        # Three-layer CNN
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(num_channels, hidden_dimension, (3, 3), stride=(1,), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dimension),
            nn.ReLU(),
            nn.Conv2d(hidden_dimension, hidden_dimension, (3, 3), stride=(1,), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dimension),
            nn.ReLU(),
            nn.Conv2d(hidden_dimension, 1, (1, 1), stride=(1,)),
            nn.ReLU()
        )

        # Three-layer linear
        self.input_dimension = np.prod(width_height)
        self.linear_layers = nn.Sequential(
            nn.Linear(self.input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, state_embedding_dimension)
        )

    def forward(self, state):
        h1 = self.cnn_layers(state)
        h1 = h1.view(-1, self.input_dimension)
        h2 = self.linear_layers(h1)
        return h2


class Transition(nn.Module):
    def __init__(self, state_embedding_dimension, action_dimension, hidden_dimension):
        super().__init__()

        # Three-layer MLP with layer normalisation after second layer
        self.layers = nn.Sequential(
            nn.Linear(state_embedding_dimension + action_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, state_embedding_dimension)
        )

        self.action_dimension = action_dimension

    def forward(self, state, action):
        # Turn action into an index tensor
        action_int = action.type(torch.LongTensor)
        # Get one-hot encoding
        action_one_hot_encoding = nn.functional.one_hot(action_int, self.action_dimension)
        action_one_hot_encoding = action_one_hot_encoding.type(torch.FloatTensor)
        action_one_hot_encoding = action_one_hot_encoding.to(action.device)

        state_action = torch.cat([state, action_one_hot_encoding], dim=-1)

        return self.layers(state_action)


def energy(predict, target, sigma=0.5):
    dissimilarity = predict - target
    norm = 0.5 / (sigma ** 2)
    energy = norm * dissimilarity.pow(2).sum(1)
    return energy


class TransE(nn.Module):
    def __init__(self, input_dimensions, state_embedding_dimension, action_dimension, hidden_dimension,
                 hinge_loss=1.0, sigma=0.5):
        super().__init__()
        self.hinge_loss = hinge_loss
        self.sigma = sigma

        num_channels = input_dimensions[0]
        width_height = input_dimensions[1:]

        self.encoder = Encoder(hidden_dimension=hidden_dimension,
                               state_embedding_dimension=state_embedding_dimension,
                               width_height=width_height,
                               num_channels=num_channels)
        self.transition = Transition(state_embedding_dimension=state_embedding_dimension,
                                     hidden_dimension=hidden_dimension,
                                     action_dimension=action_dimension)

    def transition_loss(self, state, action, next_state, transe_detach=False):
        enc_state = self.encoder(state)
        enc_next_state = self.encoder(next_state)
        if transe_detach:
            enc_state = enc_state.detach()
            enc_next_state = enc_next_state.detach()
        pred_transition = self.transition(enc_state, action)
        pred_next_state = enc_state + pred_transition
        return energy(pred_next_state, enc_next_state).mean()

    def contrastive_loss(self, state_from, action, state_to):
        # Use Encoder to encode state_from and state_to
        state_from_encoded = self.encoder(state_from)
        state_to_encoded = self.encoder(state_to)

        # Use Transition to predict transitions
        predicted_transition = self.transition(state_from_encoded, action)
        # d(z(s) + t(z(s), a), z(s'))
        positive_energy = energy((state_from_encoded + predicted_transition), state_to_encoded, self.sigma)

        # Negative sample states, sampled uniformly from a replay buffer
        replay_buffer = state_to.clone().detach()
        size = state_to.shape[0]
        negative = replay_buffer[torch.randint(0, size, (size,))]
        negative_encoded = self.encoder(negative)
        # d(z(neg_s), z(s'))
        negative_energy = energy(negative_encoded, state_to_encoded, self.sigma)

        zeros = torch.zeros_like(negative_energy)
        loss = positive_energy.mean() + torch.max(zeros, self.hinge_loss - negative_energy).mean()
        return loss

    def forward(self, state):
        return self.encoder(state)

