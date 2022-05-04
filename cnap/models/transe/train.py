import gym
import torch
import numpy as np

from cnap.models.transe.arguments import get_args
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from cnap.environment.env import make_vec_envs


# Generate training data for discrete action space
def generate_training_data(env_name, episodes):
    env = gym.make(env_name)
    training_data = []

    for _ in range(episodes):
        state_from = env.reset()
        done = False
        while not done:
            # Visualise the environment
            # env.render()

            # Perform a random action
            action = env.action_space.sample()
            state_to, _, done, _ = env.step(action)

            # Store the transition tuple (state_from, action, state_to)
            training_data.append([torch.Tensor(state_from), torch.Tensor([action]), torch.Tensor(state_to)])
            state_from = state_to

    return training_data


# Generate training data for continuous action space
def create_action_table(env, action_bins):
    action_low, action_high = env.action_space.low, env.action_space.high
    action_dim = action_low.size
    action_table = np.reshape([np.linspace(action_low[i], action_high[i], action_bins) for i in range(action_dim)],
                              [action_dim, action_bins])
    return action_table


def discretize_action(action, action_table):
    action_dim = action_table.shape[0]
    action_discretised = []
    for i in range(action_dim):
        action_discretised += [np.searchsorted(action_table[i], action[i]) - 1]
    return action_discretised


def generate_training_data_continuous(env_name, episodes, action_bins):
    env = make_vec_envs(num_processes=1,
                        env_type=env_name,
                        seed=None,
                        gamma=0.99,
                        log_dir=None,
                        continuous=True,
                        bins=action_bins,
                        device=None,
                        normalise=True)

    training_data = []
    action_dim = 2

    for _ in range(episodes):
        state_from = env.reset()
        done = False
        while not done:
            # Visualise the environment
            # env.render()

            # Perform a random action
            action = [np.random.randint(0, action_bins, action_dim)]
            action = torch.LongTensor(action)
            state_to, _, done, _ = env.step(action)

            # Store the transition tuple (state_from, action, state_to)
            training_data.append([state_from, action, state_to])
            state_from = state_to

    return training_data


def training(training_data, transe, epochs, batch_size):
    print("Start training of TransE.")

    optimizer = torch.optim.Adam(transe.parameters())

    for i in range(epochs):
        print("Epoch #", i)
        transe.train()

        sampler = BatchSampler(
            SubsetRandomSampler(range(len(training_data))),
            batch_size,
            drop_last=True)

        batch_num = 0
        for indices in sampler:
            optimizer.zero_grad()
            batch_data = [training_data[i] for i in indices]
            loss = transe.contrastive_loss(torch.stack([row[0] for row in batch_data], 0).squeeze(1),
                                           torch.stack([row[1] for row in batch_data], 0).squeeze(1),
                                           torch.stack([row[2] for row in batch_data], 0).squeeze(1))
            batch_num += 1
            if batch_num % 100 == 0:
                print("Batch #", batch_num, " Loss = ", loss.detach().item())

            loss.backward(retain_graph=True)
            optimizer.step()

    print("End of training.")


if __name__ == "__main__":
    args = get_args()

    if args.env in ['cartpole', 'acrobot', 'mountaincar', 'mountaincar-continuous', 'pendulum']:
        from cnap.models.transe.transe_classic import TransE
    elif args.env in ['walker', 'swimmer']:
        from cnap.models.transe.transe_mujoco import TransE
    else:
        raise NotImplementedError

    # Generate training data
    if not args.continuous:
        training_data = generate_training_data(env_name=args.env_name,
                                               episodes=args.num_episodes)
    else:
        training_data = generate_training_data_continuous(# env_name=args.env_name,
                                                          env_name=args.env,
                                                          episodes=args.num_episodes,
                                                          action_bins=args.action_bins)

    # Training Encoder and Transition
    transe = TransE(state_dimension=args.state_dimension,
                    state_embedding_dimension=args.state_embedding_dimension,
                    action_dimension=args.action_dimension,
                    hidden_dimension=args.hidden_dimension,
                    action_multidim=args.action_multidim)
    training(training_data=training_data, transe=transe, epochs=args.num_epochs, batch_size=args.batch_size)

    # Save the TransE model
    if not args.continuous:
        torch.save(transe.state_dict(), f"trained_transe/transe_{args.env}.pt")
    else:
        torch.save(transe.state_dict(), f"trained_transe/transe_{args.env}_{args.action_bins}bins.pt")
