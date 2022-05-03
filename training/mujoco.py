import os
import json
import numpy as np
import torch
import torch.optim as optim
from collections import deque

from arguments import get_args
from models.transe.transe_mujoco import TransE
from models.executor.gnn import Executor
from models.ppo.ppo import PPO
from models.ppo.policy import Policy
from models.ppo.rollout import RolloutStorage
from models.ppo.utils import set_seed, update_linear_schedule, cleanup_log_dir
from environment.env import make_vec_envs


def evaluate(env, policy, rollout, num_episodes, deterministic=False):
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
    print(
        f"Rollout {rollout}: Average episode reward = {np.average(episode_rewards)} from total {num_episodes} episodes")


if __name__ == '__main__':
    args = get_args()

    if args.include_executor:
        subdir = f"{args.env}/xlvin_{args.action_bins}bins_{args.gnn_steps}gnnsteps_{args.num_neighbours}neighbours_" \
                 f"{args.cat_method}_{args.sample_method}_{args.seed}seed"
    else:
        subdir = f"{args.env}/ppo_{args.action_bins}bins_{args.seed}seed"
    log_dir = os.path.join(os.path.expanduser(args.log_dir), subdir)
    eval_log_dir = log_dir + "_eval"
    cleanup_log_dir(log_dir)
    cleanup_log_dir(eval_log_dir)

    # Device
    if args.device == 'cuda':
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device("cpu")
    print("Device: ", args.device)

    # Seed
    set_seed(args.seed)
    print("Seed: ", args.seed)

    # TransE
    transe = TransE(state_embedding_dimension=args.transe_embedding_dim,
                    state_dimension=args.env_input_dims,
                    hidden_dimension=args.transe_hidden_dim,
                    action_dimension=args.action_bins,
                    action_multidim=args.action_multidim,
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
    executor_path = f'../models/executor/trained_executor/{args.graph_type}' \
                    f'-gnnstep{args.gnn_steps}-act{args.activation}-' \
                    f'ln{args.layernorm}-hidden{args.gnn_hidden_dim}.pt'
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
                    num_neighbours=args.num_neighbours,
                    freeze_encoder=args.freeze_encoder,
                    freeze_executor=args.freeze_executor,
                    transe2gnn=args.transe2gnn,
                    gnn_decoder=args.gnn_decoder,
                    gnn_steps=args.gnn_steps,
                    graph_detach=args.graph_detach,
                    sample_method=args.sample_method,
                    cat_method=args.cat_method).to(args.device)

    # PPO
    params = list(policy.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    ppo = PPO(actor_critic=policy,
              clip_param=args.clip_param,
              ppo_epoch=args.ppo_epoch,
              num_mini_batch=args.num_mini_batch,
              value_loss_coef=args.value_loss_coef,
              entropy_coef=0.0,
              transe_loss_coef=0.0 if not args.include_executor else args.transe_loss_coef,
              optimizer=optimizer,
              max_grad_norm=args.max_grad_norm,
              mini_batch_size=args.mini_batch_size,
              transe_detach=args.transe_detach,
              use_clipped_value_loss=args.use_clipped_value_loss)

    # Create envs
    train_envs = make_vec_envs(num_processes=args.num_processes,
                               env_type=args.env,
                               seed=args.seed,
                               gamma=args.gamma,
                               log_dir=log_dir,
                               continuous=True,
                               bins=args.action_bins,
                               device=args.device,
                               normalise=True,
                               record_video=args.record_video,
                               save_video_interval=args.save_video_interval,
                               video_length=args.video_length)
    if args.evaluate:
        test_envs = make_vec_envs(num_processes=args.num_processes,
                                  env_type=args.env,
                                  seed=args.seed,
                                  gamma=args.gamma,
                                  log_dir=eval_log_dir,
                                  continuous=True,
                                  bins=args.action_bins,
                                  device=args.device,
                                  normalise=True)

    # Train & Test
    if args.save_model:
        save_path = os.path.join(args.save_dir, subdir)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

    rollouts = RolloutStorage(num_steps=args.num_train_steps,
                              num_processes=args.num_processes,
                              obs_shape=train_envs.observation_space.shape,
                              action_space=train_envs.action_space,
                              device=args.device)

    obs = train_envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(args.device)
    episode_rewards = deque(maxlen=100)
    reward_record = []  # to draw learning curve: avg episode reward against timesteps

    for j in range(args.ppo_updates):

        if args.lr_decay:
            # decrease learning rate linearly
            update_linear_schedule(
                ppo.optimizer, j, args.ppo_updates,
                args.lr)

        for step in range(args.num_train_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = policy.act(rollouts.obs[step], deterministic=False)

            obs, reward, done, infos = train_envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs=obs, next_obs=obs, actions=action,
                            action_log_probs=action_log_prob,
                            value_preds=value, rewards=reward, masks=masks,
                            bad_masks=bad_masks)

        with torch.no_grad():
            next_value = policy.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value=next_value, use_gae=args.use_gae, gamma=args.gamma,
                                 gae_lambda=args.gae_lambda, enable_time_limit=args.enable_time_limit)

        value_loss, action_loss, dist_entropy, transe_loss = ppo.update(rollouts)
        if args.debug:
            print("value loss = ", value_loss)
            print("action loss = ", action_loss)
            print("dist loss = ", dist_entropy)
            print("transe loss = ", transe_loss)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == args.ppo_updates - 1) and args.save_model:
            if not args.include_executor:
                with open(
                        os.path.join(save_path,
                                     f'{args.env}_ppo_bins{args.action_bins}_episodic_rewards.json'), 'w') as f:
                    json.dump(reward_record, f)
                torch.save({
                    'transe': transe.state_dict(),
                    'executor': executor.state_dict(),
                    'policy': policy.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f=os.path.join(save_path, f"{args.env}_ppo_bins{args.action_bins}.pt"))
            else:
                with open(
                    os.path.join(save_path,
                        f'{args.env}_xlvin_bins{args.action_bins}_neighbour{args.num_neighbours}_'
                        f'step{args.gnn_steps}_{args.cat_method}_{args.sample_method}_episodic_rewards.json'),
                        'w') as f:
                    json.dump(reward_record, f)
                torch.save({
                    'transe': transe.state_dict(),
                    'executor': executor.state_dict(),
                    'policy': policy.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f=os.path.join(save_path,
                    f"{args.env}_xlvin_bins{args.action_bins}_neighbour{args.num_neighbours}_step{args.gnn_steps}.pt"))

        if len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_train_steps
            print(
                "Updates {}, num timesteps {} \n"
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j,
                            total_num_steps,
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))
            reward_record.append(np.mean(episode_rewards))

        if args.evaluate:
            evaluate(env=test_envs, policy=policy, rollout=j, num_episodes=args.num_test_episodes, deterministic=True)
