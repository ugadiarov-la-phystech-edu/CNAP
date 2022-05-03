import os

import numpy as np
import torch
import torch.optim as optim

from arguments import get_args
from models.ppo.ppo import PPO
from models.ppo.policy import Policy
from models.ppo.rollout import gather_fixed_episodes_rollout
from models.transe.transe_classic import TransE
from models.ppo.utils import set_seed, cleanup_log_dir
from models.executor.gnn import Executor
from environment.env import make_vec_envs


def run_fixed_num_episodes(env, policy, rollout, num_episodes, deterministic, file):
    obs = env.reset()
    episode_rewards = []
    while len(episode_rewards) < num_episodes:
        # env.render()
        with torch.no_grad():
            value, action, log_probs = policy.act(obs, deterministic=deterministic)
        obs, reward, done, infos = env.step(action)

        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

    episode_rewards = np.asarray(episode_rewards)
    average_reward = np.average(episode_rewards)
    reward_total = np.sum(episode_rewards)
    done_episodes = len(episode_rewards)
    print("Rollout {} : Total rewards {:.3f} from total episodes {} (Average: {:.3f})".format(
        rollout, reward_total, done_episodes, average_reward))
    if file:
        file.writelines("Rollout {} : Total rewards {:.3f} from total episodes {} (Average: {:.3f})\n".format(
            rollout, reward_total, done_episodes, average_reward))
    return average_reward


if __name__ == '__main__':
    args = get_args()

    model_name = 'xlvin-r' if args.graph_type == "erdos-renyi" else 'xlvin-cp'
    if args.include_executor:
        subdir = f"{args.env}/{model_name}_{args.action_bins}bins_{args.gnn_steps}gnnsteps_{args.seed}seed"
    else:
        subdir = f"{args.env}/ppo_{args.action_bins}bins_{args.seed}seed"
    log_dir = os.path.join(os.path.expanduser(args.log_dir), subdir)
    eval_log_dir = log_dir + "_eval"
    cleanup_log_dir(log_dir)
    cleanup_log_dir(eval_log_dir)

    if args.save_model:
        save_path = os.path.join(args.save_dir, subdir)
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        if args.include_executor:
            filename = f"{args.env}-{model_name}-gnnstep{args.gnn_steps}-bins{args.action_bins}-seed{args.seed}.txt"
        else:
            filename = f"{args.env}-ppo-bins{args.action_bins}-seed{args.seed}.txt"
        f = open(os.path.join(save_path, filename), "w")
    else:
        f = None

    # Device
    if args.device == 'cuda':
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device("cpu")
    print("Device: ", args.device)

    # Seed
    set_seed(args.seed)
    if args.save_model:
        f.writelines(f"Seed: {args.seed}\n")

    # TransE
    transe = TransE(state_embedding_dimension=args.transe_embedding_dim,
                    state_dimension=args.env_input_dims,
                    hidden_dimension=args.transe_hidden_dim,
                    action_dimension=args.env_action_dim,
                    sigma=0.5,
                    hinge_loss=1.0).to(args.device)
    if args.include_transe:
        transe.load_state_dict(torch.load(args.transe_weights_path, map_location=args.device))

    # Executor
    executor = Executor(node_dimension=2,
                        edge_dimension=2,
                        hidden_dimension=args.gnn_hidden_dim,
                        out_dimension=1,
                        gnn_steps=args.gnn_steps,
                        activation=args.activation,
                        layernorm=args.layernorm,
                        neighbour_aggregation=args.neighbour_aggregation).to(args.device)
    executor_path = f'../models/executor/trained_executor/{args.graph_type}-gnnstep{args.gnn_steps}-'\
                    f'act{args.activation}-ln{args.layernorm}.pt'
    if args.include_executor:
        executor.load_state_dict(torch.load(executor_path, map_location=args.device))

    # Policy
    policy = Policy(action_space=args.action_space,
                    gamma=args.gamma,
                    transe=transe,
                    edge_feat=args.env_action_dim + 1,  # +1 as we add gamma as edge_feature too
                    transe_hidden_dim=args.transe_embedding_dim,
                    gnn_hidden_dim=args.gnn_hidden_dim,
                    num_processes=args.num_processes,
                    include_executor=args.include_executor,
                    executor=executor.message_passing,
                    freeze_encoder=args.freeze_encoder,
                    freeze_executor=args.freeze_executor,
                    transe2gnn=args.transe2gnn,
                    gnn_decoder=args.gnn_decoder,
                    gnn_steps=args.gnn_steps,
                    graph_detach=args.graph_detach).to(args.device)

    # PPO
    params = list(policy.parameters())
    optimizer = optim.Adam(params)
    ppo = PPO(actor_critic=policy,
              clip_param=args.clip_param,
              ppo_epoch=args.ppo_epoch,
              num_mini_batch=args.num_mini_batch,
              value_loss_coef=args.value_loss_coef,
              entropy_coef=args.entropy_coef,
              transe_loss_coef=args.transe_loss_coef,
              optimizer=optimizer,
              max_grad_norm=args.max_grad_norm,
              mini_batch_size=args.mini_batch_size,
              transe_detach=args.transe_detach,
              use_clipped_value_loss=args.use_clipped_value_loss)

    # Create envs
    if args.env in ['cartpole', 'acrobot', 'mountaincar']:
        continuous = False
    elif args.env in ['mountaincar-continuous', 'pendulum']:
        continuous = True

    train_envs = make_vec_envs(num_processes=args.num_processes,
                               env_type=args.env,
                               seed=args.seed,
                               gamma=args.gamma,
                               log_dir=log_dir,
                               continuous=continuous,
                               bins=args.action_bins,
                               device=args.device)
    test_envs = make_vec_envs(num_processes=args.num_processes,
                              env_type=args.env,
                              seed=args.seed,
                              gamma=args.gamma,
                              log_dir=eval_log_dir,
                              continuous=continuous,
                              bins=args.action_bins,
                              device=args.device)

    # Train and Test
    max_average_reward = float('-inf')
    for j in range(args.num_rollouts):
        rollouts = gather_fixed_episodes_rollout(env=train_envs,
                                                 policy=policy,
                                                 num_episodes=args.num_train_episodes,
                                                 gamma=args.gamma,
                                                 num_processes=args.num_processes,
                                                 device=args.device,
                                                 file=f)

        for k in range(args.ppo_updates):
            print(f'PPO update {k}')
            value_loss_epoch, action_loss_epoch, dist_entropy_epoch, transe_loss_epoch = ppo.update(rollouts)

            if args.debug:
                print("value_loss_epoch ", value_loss_epoch, flush=True)
                print("action_loss_epoch ", action_loss_epoch, flush=True)
                print("dist_entropy_epoch ", dist_entropy_epoch, flush=True)
                print("transe_loss_epoch ", transe_loss_epoch, flush=True)

            average_reward = run_fixed_num_episodes(env=test_envs,
                                                    policy=policy,
                                                    rollout=j,
                                                    num_episodes=args.num_test_episodes,
                                                    deterministic=args.deterministic,
                                                    file=f)
            if average_reward > max_average_reward:
                max_average_reward = average_reward

    if args.save_model:
        if args.include_executor:
            model_name = 'xlvin-r' if args.graph_type == 'erdos-renyi' else 'xlvin-cp'
            model_path = f'{args.env}-{model_name}-gnnsteps{args.gnn_steps}-act{args.activation}-ln{args.layernorm}.pt'
            torch.save({
                'transe': transe.state_dict(),
                'executor': executor.state_dict(),
                'policy': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f=os.path.join(save_path, model_path))
        else:
            model_path = f'{args.env}-ppo-gnnsteps{args.gnn_steps}-act{args.activation}-ln{args.layernorm}.pt'
            torch.save({
                'transe': transe.state_dict(),
                'executor': executor.state_dict(),
                'policy': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f=os.path.join(save_path, model_path))

    print("\n Largest average reward = ", max_average_reward)

    if args.save_model:
        f.writelines("\n Largest average reward = " + str(max_average_reward))
        f.close()
