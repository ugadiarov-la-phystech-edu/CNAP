import argparse


def get_args():
    parser = argparse.ArgumentParser(description='TransE')

    parser.add_argument('--env', type=str, default='cartpole', help='environment to train for (default: "cartpole")')
    parser.add_argument('--batch_size', type=int, default=128, help='size of batch')
    parser.add_argument('--num_episodes', type=int, default=None, help='number of episodes to generate training data')
    parser.add_argument('--num_epochs', type=int, default=None, help='number of epochs')
    parser.add_argument('--state_embedding_dimension', type=int, default=None, help='dimension of state embeddings')
    parser.add_argument('--hidden_dimension', type=int, default=None, help='dimension of hidden layers')
    parser.add_argument('--action_bins', type=int, default=10, help='number of bins to discretize continuous action '
                                                                    'space')

    args = parser.parse_args()
    if args.env == 'cartpole':
        args.env_name = 'CartPole-v0'
        if not args.num_episodes:
            args.num_episodes = 50000
        if not args.num_epochs:
            args.num_epochs = 10
        if not args.state_embedding_dimension:
            args.state_embedding_dimension = 50
        if not args.hidden_dimension:
            args.hidden_dimension = 64
        args.state_dimension = 4
        args.action_dimension = 2
        args.action_multidim = 1
        args.continuous = False
    elif args.env == 'acrobot':
        args.env_name = 'Acrobot-v1'
        if not args.num_episodes:
            args.num_episodes = 4000
        if not args.num_epochs:
            args.num_epochs = 5
        if not args.state_embedding_dimension:
            args.state_embedding_dimension = 50
        if not args.hidden_dimension:
            args.hidden_dimension = 32
        args.state_dimension = 6
        args.action_dimension = 3
        args.action_multidim = 1
        args.continuous = False
    elif args.env == 'mountaincar':
        args.env_name = 'MountainCar-v0'
        if not args.num_episodes:
            args.num_episodes = 5000
        if not args.num_epochs:
            args.num_epochs = 5
        if not args.state_embedding_dimension:
            args.state_embedding_dimension = 50
        if not args.hidden_dimension:
            args.hidden_dimension = 16
        args.state_dimension = 2
        args.action_dimension = 3
        args.action_multidim = 1
        args.continuous = False
    elif args.env == 'mountaincar-continuous':
        args.env_name = 'MountainCarContinuous-v0'
        if not args.num_episodes:
            args.num_episodes = 1000
        if not args.num_epochs:
            args.num_epochs = 5
        if not args.state_embedding_dimension:
            args.state_embedding_dimension = 50
        if not args.hidden_dimension:
            args.hidden_dimension = 16
        args.state_dimension = 2
        args.action_dimension = args.action_bins
        args.action_multidim = 1
        args.continuous = True
    elif args.env == 'pendulum':
        args.env_name = 'Pendulum-v0'
        if not args.num_episodes:
            args.num_episodes = 1000
        if not args.num_epochs:
            args.num_epochs = 5
        if not args.state_embedding_dimension:
            args.state_embedding_dimension = 50
        if not args.hidden_dimension:
            args.hidden_dimension = 16
        args.state_dimension = 3
        args.action_dimension = args.action_bins
        args.action_multidim = 1
        args.continuous = True
    elif args.env == 'walker':
        args.env_name = 'Walker2d-v2'
        if not args.num_episodes:
            args.num_episodes = 50000
        if not args.num_epochs:
            args.num_epochs = 10
        if not args.state_embedding_dimension:
            args.state_embedding_dimension = 64
        if not args.hidden_dimension:
            args.hidden_dimension = 64
        args.state_dimension = 17
        args.action_dimension = args.action_bins
        args.action_multidim = 6
        args.continuous = True
    elif args.env == 'swimmer':
        args.env_name = 'Swimmer-v2'
        if not args.num_episodes:
            args.num_episodes = 1000
        if not args.num_epochs:
            args.num_epochs = 5
        if not args.state_embedding_dimension:
            args.state_embedding_dimension = 64
        if not args.hidden_dimension:
            args.hidden_dimension = 64
        args.state_dimension = 8
        args.action_dimension = args.action_bins
        args.action_multidim = 2
        args.continuous = True

    return args
