import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, state_dimension, hidden_dimension, state_embedding_dimension):
        super().__init__()

        # Three-layer MLP
        self.layers = nn.Sequential(
            nn.Linear(state_dimension, hidden_dimension * 2),
            nn.ReLU(),
            nn.Linear(hidden_dimension * 2, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, state_embedding_dimension)
        )

    def forward(self, state):
        return self.layers(state)


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
    def __init__(self, state_dimension, state_embedding_dimension, action_dimension, hidden_dimension,
                 hinge_loss=1.0, sigma=0.5, action_multidim=1):
        super().__init__()
        self.hinge_loss = hinge_loss
        self.sigma = sigma
        self.encoder = Encoder(state_dimension=state_dimension,
                               hidden_dimension=hidden_dimension,
                               state_embedding_dimension=state_embedding_dimension)
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