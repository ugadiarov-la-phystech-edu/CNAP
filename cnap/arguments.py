import argparse
from gym import spaces


def get_args():
    parser = argparse.ArgumentParser(description='main')

    parser.add_argument('--seed', type=int, default=11111, help='seed')

    # Executor parameters
    parser.add_argument('--graph_type', type=str, default='erdos-renyi', help='graph types that executor trained on')
    parser.add_argument('--gnn_steps', type=int, default=1, help='depth of gnn')
    parser.add_argument('--activation', type=bool, default=False, help='whether to include message activation')
    parser.add_argument('--layernorm', type=bool, default=True, help='whether to include layer norm')
    parser.add_argument('--neighbour_aggregation', type=str, default='max',
                        help='method to aggregate neighbour messages')
    parser.add_argument('--sample_method', type=str, default="expand_all",
                        choices=["uniform", "learn_gaussian", "manual_gaussian", "learn_neighbour_policy",
                                 "reuse_actor_layer", "expand_all"])
    parser.add_argument('--cat_method', type=str, default="encoder_cat_executor",
                        choices=["encoder_cat_executor", "executor_only", "encoder_add_executor",
                                 "encoder_cat_executor_decode"])

    # Policy parameters
    parser.add_argument('--freeze_encoder', type=bool, default=False, help='whether to freeze pretrained encoder')
    parser.add_argument('--freeze_executor', type=bool, default=True, help='whether to freeze pretrained executor')
    parser.add_argument('--transe2gnn', type=int, default=1, help='number of layers between transition and executor')
    parser.add_argument('--gnn_decoder', type=int, default=1, help='number f layers after executor')
    parser.add_argument('--graph_detach', type=bool, default=False)

    # PPO parameters
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--ppo_epoch', type=int, default=None)
    parser.add_argument('--num_mini_batch', type=int, default=32)
    parser.add_argument('--use_gae', type=bool, default=False)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--transe_detach', type=bool, default=False)
    parser.add_argument('--use_clipped_value_loss', type=bool, default=True)
    parser.add_argument('--transe_loss_coef', type=float, default=0.001)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.0003)

    # Control parameters
    parser.add_argument('--env', type=str, default='cartpole', help='environment to run on')
    parser.add_argument('--include_transe', type=bool, default=False, help='whether to include pretrained transe')
    parser.add_argument('--include_executor', type=bool, default=False, help='whether to include pretrained executor')
    parser.add_argument('--gnn_hidden_dim', type=int, default=None, help='dimension of executor hidden layers')
    parser.add_argument('--transe_embedding_dim', type=int, default=None, help='dimension of state embeddings')
    parser.add_argument('--transe_hidden_dim', type=int, default=None, help='dimension of transe hidden layers')
    parser.add_argument('--mini_batch_size', type=int, default=None)
    parser.add_argument('--num_train_episodes', type=int, default=None)
    parser.add_argument('--num_train_steps', type=int, default=2048)
    parser.add_argument('--num_total_train_steps', type=int, default=1e7)
    parser.add_argument('--num_test_episodes', type=int, default=100)
    parser.add_argument('--num_rollouts', type=int, default=None)
    parser.add_argument('--ppo_updates', type=int, default=None)
    parser.add_argument('--num_processes', type=int, default=None)
    parser.add_argument('--action_bins', type=int, default=11)
    parser.add_argument('--num_neighbours', type=int, default=10)
    parser.add_argument('--pass_threshold', type=float, default=0.95)
    parser.add_argument('--save_model', type=bool, default=False, help='whether to save the trained model')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate on test episodes')
    parser.add_argument('--lr_decay', type=bool, default=False)
    parser.add_argument('--save_interval', type=int, default=100, help='update interval to save the trained model')
    parser.add_argument('--save_dir', type=str, default='logs', help='directory to save model')
    parser.add_argument('--log_dir', type=str, default='logs/logs', help='directory to save logs')
    parser.add_argument('--enable_time_limit', type=bool, default=False, help='whether take time limit into acount')
    parser.add_argument('--record_video', type=bool, default=False, help='whether to record the video')
    parser.add_argument('--save_video_interval', type=int, default=204800, help='interval to record video in terms of steps')
    parser.add_argument('--video_length', type=int, default=500, help='length of video recorded')

    args = parser.parse_args()
    if args.env == 'maze':
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 128
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 10
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 10
        if not args.ppo_epoch:
            args.ppo_epoch = 4
        args.env_input_dims = (3, 8, 8)
        args.env_action_dim = 8
        args.action_space = spaces.Discrete(args.env_action_dim)
        if not args.num_processes:
            args.num_processes = 15
        if not args.num_rollouts:
            args.num_rollouts = 1000
        if not args.ppo_updates:
            args.ppo_updates = 1
        if not args.num_train_steps:
            args.num_train_steps = 64
        if not args.num_test_episodes:
            args.num_test_episodes = 1000
    elif args.env == 'cartpole':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 50
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 50
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 64
        args.transe_weights_path = '../models/transe/trained_transe/transe_cartpole.pt'
        if not args.mini_batch_size:
            args.mini_batch_size = 128
        if not args.num_train_episodes:
            args.num_train_episodes = 10
        args.env_input_dims = 4
        args.env_action_dim = 2
        args.action_space = spaces.Discrete(args.env_action_dim)
        if not args.ppo_epoch:
            args.ppo_epoch = 1
        if not args.num_rollouts:
            args.num_rollouts = 1
        if not args.ppo_updates:
            args.ppo_updates = 100
        if not args.num_processes:
            args.num_processes = 15
        args.deterministic = True
    elif args.env == 'acrobot':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 50
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 50
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 32
        args.transe_weights_path = '../models/transe/trained_transe/transe_acrobot.pt'
        if not args.mini_batch_size:
            args.mini_batch_size = 128
        if not args.num_train_episodes:
            args.num_train_episodes = 5
        args.env_input_dims = 6
        args.env_action_dim = 3
        args.action_space = spaces.Discrete(args.env_action_dim)
        if not args.ppo_epoch:
            args.ppo_epoch = 1
        if not args.num_rollouts:
            args.num_rollouts = 20
        if not args.ppo_updates:
            args.ppo_updates = 1
        if not args.num_processes:
            args.num_processes = 5
        args.deterministic = True
    elif args.env == 'mountaincar':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 50
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 50
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 16
        args.transe_weights_path = '../models/transe/trained_transe/transe_mountaincar.pt'
        if not args.mini_batch_size:
            args.mini_batch_size = 128
        if not args.num_train_episodes:
            args.num_train_episodes = 5
        args.env_input_dims = 2
        args.env_action_dim = 3
        args.action_space = spaces.Discrete(args.env_action_dim)
        if not args.ppo_epoch:
            args.ppo_epoch = 1
        if not args.num_rollouts:
            args.num_rollouts = 20
        if not args.ppo_updates:
            args.ppo_updates = 1
        if not args.num_processes:
            args.num_processes = 5
        args.deterministic = True
    elif args.env == 'mountaincar-continuous':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 50
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 50
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 16
        args.transe_weights_path = f'../models/transe/trained_transe/transe_mountaincar_continuous_{args.action_bins}bins.pt'
        if not args.mini_batch_size:
            args.mini_batch_size = 128
        if not args.num_train_episodes:
            args.num_train_episodes = 5
        args.env_input_dims = 2
        args.env_action_dim = args.action_bins
        args.action_space = spaces.Discrete(args.env_action_dim)
        if not args.ppo_epoch:
            args.ppo_epoch = 1
        if not args.num_rollouts:
            args.num_rollouts = 20
        if not args.ppo_updates:
            args.ppo_updates = 1
        if not args.num_processes:
            args.num_processes = 5
        args.deterministic = False
    elif args.env == 'pendulum':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 50
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 50
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 16
        args.transe_weights_path = f'../models/transe/trained_transe/transe_pendulum_{args.action_bins}bins.pt'
        if not args.mini_batch_size:
            args.mini_batch_size = 128
        if not args.num_train_episodes:
            args.num_train_episodes = 5
        args.env_input_dims = 3
        args.env_action_dim = args.action_bins
        args.action_space = spaces.Discrete(args.env_action_dim)
        if not args.ppo_epoch:
            args.ppo_epoch = 1
        if not args.num_rollouts:
            args.num_rollouts = 20
        if not args.ppo_updates:
            args.ppo_updates = 1
        if not args.num_processes:
            args.num_processes = 5
        args.deterministic = False
    elif args.env == 'walker':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 64
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 64
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 64
        args.env_input_dims = 17
        args.action_multidim = 6
        args.env_action_dim = args.action_bins * args.action_multidim
        args.action_space = spaces.MultiDiscrete([args.action_bins for _ in range(args.action_multidim)])
        args.ppo_updates = int(args.num_total_train_steps / args.num_train_steps)
        if not args.ppo_epoch:
            args.ppo_epoch = 10
        if not args.num_processes:
            args.num_processes = 1
        args.deterministic = False
        args.transe_weights_path = f'../models/transe/trained_transe/transe_walker_{args.action_bins}bins.pt'
    elif args.env == 'halfcheetah':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 64
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 64
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 64
        args.env_input_dims = 17
        args.action_multidim = 6
        args.env_action_dim = args.action_bins * args.action_multidim
        args.action_space = spaces.MultiDiscrete([args.action_bins for _ in range(args.action_multidim)])
        args.ppo_updates = int(args.num_total_train_steps / args.num_train_steps)
        if not args.ppo_epoch:
            args.ppo_epoch = 10
        if not args.num_processes:
            args.num_processes = 1
        args.deterministic = False
    elif args.env == 'ant':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 64
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 64
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 64
        args.env_input_dims = 111
        args.action_multidim = 8
        args.env_action_dim = args.action_bins * args.action_multidim
        args.action_space = spaces.MultiDiscrete([args.action_bins for _ in range(args.action_multidim)])
        args.ppo_updates = int(args.num_total_train_steps / args.num_train_steps)
        if not args.ppo_epoch:
            args.ppo_epoch = 10
        if not args.num_processes:
            args.num_processes = 1
        args.deterministic = False
    elif args.env == 'humanoid':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 64
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 64
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 64
        args.env_input_dims = 376
        args.action_multidim = 17
        args.env_action_dim = args.action_bins * args.action_multidim
        args.action_space = spaces.MultiDiscrete([args.action_bins for _ in range(args.action_multidim)])
        args.ppo_updates = int(args.num_total_train_steps / args.num_train_steps)
        if not args.ppo_epoch:
            args.ppo_epoch = 10
        if not args.num_processes:
            args.num_processes = 1
        args.deterministic = False
    elif args.env == 'humanoid-standup':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 64
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 64
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 64
        args.env_input_dims = 376
        args.action_multidim = 17
        args.env_action_dim = args.action_bins * args.action_multidim
        args.action_space = spaces.MultiDiscrete([args.action_bins for _ in range(args.action_multidim)])
        args.ppo_updates = int(args.num_total_train_steps / args.num_train_steps)
        if not args.ppo_epoch:
            args.ppo_epoch = 10
        if not args.num_processes:
            args.num_processes = 1
        args.deterministic = False
    elif args.env == 'hopper':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 64
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 64
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 64
        args.env_input_dims = 11
        args.action_multidim = 3
        args.env_action_dim = args.action_bins * args.action_multidim
        args.action_space = spaces.MultiDiscrete([args.action_bins for _ in range(args.action_multidim)])
        args.ppo_updates = int(args.num_total_train_steps / args.num_train_steps)
        if not args.ppo_epoch:
            args.ppo_epoch = 10
        if not args.num_processes:
            args.num_processes = 1
        args.deterministic = False
    elif args.env == 'inverted-double-pendulum':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 64
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 64
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 64
        args.env_input_dims = 376
        args.action_multidim = 17
        args.env_action_dim = args.action_bins * args.action_multidim
        args.action_space = spaces.MultiDiscrete([args.action_bins for _ in range(args.action_multidim)])
        args.ppo_updates = int(args.num_total_train_steps / args.num_train_steps)
        if not args.ppo_epoch:
            args.ppo_epoch = 10
        if not args.num_processes:
            args.num_processes = 1
        args.deterministic = False
    elif args.env == 'inverted-pendulum':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 64
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 64
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 64
        args.env_input_dims = 11
        args.action_multidim = 1
        args.env_action_dim = args.action_bins * args.action_multidim
        args.action_space = spaces.MultiDiscrete([args.action_bins for _ in range(args.action_multidim)])
        args.ppo_updates = int(args.num_total_train_steps / args.num_train_steps)
        if not args.ppo_epoch:
            args.ppo_epoch = 10
        if not args.num_processes:
            args.num_processes = 1
        args.deterministic = False
    elif args.env == 'reacher':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 64
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 64
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 64
        args.env_input_dims = 11
        args.action_multidim = 2
        args.env_action_dim = args.action_bins * args.action_multidim
        args.action_space = spaces.MultiDiscrete([args.action_bins for _ in range(args.action_multidim)])
        args.ppo_updates = int(args.num_total_train_steps / args.num_train_steps)
        if not args.ppo_epoch:
            args.ppo_epoch = 10
        if not args.num_processes:
            args.num_processes = 1
        args.deterministic = False
    elif args.env == 'swimmer':
        if not args.gnn_hidden_dim:
            args.gnn_hidden_dim = 64
        if not args.transe_embedding_dim:
            args.transe_embedding_dim = 64
        if not args.transe_hidden_dim:
            args.transe_hidden_dim = 64
        args.env_input_dims = 8
        args.action_multidim = 2
        args.env_action_dim = args.action_bins * args.action_multidim
        args.action_space = spaces.MultiDiscrete([args.action_bins for _ in range(args.action_multidim)])
        args.ppo_updates = int(args.num_total_train_steps / args.num_train_steps)
        if not args.ppo_epoch:
            args.ppo_epoch = 10
        if not args.num_processes:
            args.num_processes = 1
        args.deterministic = False

    return args
