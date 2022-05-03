import torch
import torch.optim as optim
import numpy as np
import pickle

from arguments import get_args
from models.transe.transe_maze import TransE
from models.executor.gnn import Executor
from models.ppo.ppo import PPO
from models.ppo.policy import Policy
from models.ppo.rollout import gather_maze_rollout
from environment.env import make_vec_maze_envs, test_vec_envs


def test_run_fixed_number_mazes(env, policy, rollout, num_test_mazes, pass_threshold):
    obs = env.reset()
    done_mazes = 0
    passed_mazes = 0
    while done_mazes < num_test_mazes:
        with torch.no_grad():
            value, action, log_probs = policy.act(observations=obs.to(args.device), deterministic=False)
        obs, reward, done, solved_obs = env.step(action)
        reward = torch.tensor(reward, device=args.device)
        passed = torch.sum(torch.eq(reward, torch.ones_like(reward, device=args.device)))
        failed = torch.sum(torch.eq(reward, torch.ones_like(reward, device=args.device) * -1))

        done_mazes += passed + failed
        passed_mazes += passed

    passed_percentage = float(passed_mazes) / float(done_mazes)
    print("Rollout {} : percentage passed {:.3f} from total episodes {}".format(
        rollout, passed_percentage * 100.0, done_mazes))
    if passed_percentage > pass_threshold:
        return True, passed_percentage
    else:
        return False, passed_percentage


def test_hold_out_mazes(test_maze_size, test_indices, policy, num_processes):
    test_passed = np.zeros(len(test_indices.keys()))
    test_total = np.zeros(len(test_indices.keys()))
    test_perc = np.zeros(len(test_indices.keys()))

    for level in sorted(test_indices.keys()):
        test_env = test_vec_envs(maze_size=test_maze_size,
                                 train_maze=False,
                                 maze_indices_list=test_indices[level],
                                 device=args.device,
                                 num_processes=num_processes)
        obs = test_env.reset()
        while True:
            with torch.no_grad():
                value, action, log_probs = policy.act(obs, deterministic=True)
            obs, reward, done, solved_obs = test_env.step(action)
            reward = torch.tensor(reward, device=args.device)
            passed = torch.sum(torch.eq(reward, torch.ones_like(reward)))
            failed = torch.sum(torch.eq(reward, torch.ones_like(reward) * -1))
            done_mazes = passed + failed
            test_passed[int(level) - 1] += passed
            test_total[int(level) - 1] += done_mazes

            if torch.max(reward) < -1.0:
                break

        test_perc[int(level) - 1] = test_passed[int(level) - 1] * 100.0 / test_total[int(level) - 1]
        overall_perc_passed = test_perc[int(level) - 1]
        print("#test# Overall percentage passed for paths of length {} is: {}".format(level, overall_perc_passed))
    overall_perc = np.sum(test_passed) * 100.0 / np.sum(test_total)
    print("#test# Overall percentage passed is: {}".format(overall_perc))

    return overall_perc, (test_passed, test_total, test_perc)


if __name__ == "__main__":
    args = get_args()

    # Device
    if args.device == 'cuda':
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device("cpu")
    print("Device: ", args.device)

    # Load dataset
    train_indices = pickle.load(open('../environment/maze/dataset/train_8*8_maze_by_level.pkl', 'rb'))
    test_indices = pickle.load(open('../environment/maze/dataset/test_8*8_maze_by_level.pkl', 'rb'))

    # TransE
    transe = TransE(state_embedding_dimension=args.transe_embedding_dim,
                    input_dimensions=args.env_input_dims,
                    action_dimension=args.env_action_dim,
                    hidden_dimension=args.transe_hidden_dim).to(args.device)

    # Executor
    executor = Executor(node_dimension=2,
                        edge_dimension=2,
                        hidden_dimension=args.gnn_hidden_dim,
                        out_dimension=1,
                        neighbour_aggregation=args.neighbour_aggregation,
                        activation=args.activation,
                        layernorm=args.layernorm,
                        gnn_steps=args.gnn_steps).to(args.device)
    executor_path = f'../models/executor/trained_executor/{args.graph_type}' \
                    f'-gnnstep{args.gnn_steps}-act{args.activation}-' \
                    f'ln{args.layernorm}-hidden{args.gnn_hidden_dim}.pt'
    if args.include_executor:
        executor.load_state_dict(torch.load(executor_path, map_location=args.device))

    # Policy
    policy = Policy(action_space=args.action_space,
                    gamma=args.gamma,
                    transe=transe,
                    edge_feat=args.env_action_dim + 1,
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
    params = list(policy.parameters()) + list(transe.parameters())
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

    # Train and Test by maze difficulty level
    for level in sorted(train_indices.keys()):
        print("Training on maze level: ", level)

        level_train_indices = list(train_indices[level])
        train_envs = make_vec_maze_envs(num_processes=args.num_processes,
                                        train=True,
                                        indices_list=level_train_indices,
                                        device=args.device)
        test_envs = make_vec_maze_envs(num_processes=args.num_processes,
                                       train=True,
                                       indices_list=level_train_indices,
                                       device=args.device)

        passed = False
        rollout = 0
        while not passed and rollout < args.num_rollouts:
            rollout += 1
            rollouts, num_passed, num_failed = gather_maze_rollout(env=train_envs,
                                                                   policy=policy,
                                                                   num_steps=args.num_train_steps,
                                                                   gamma=args.gamma,
                                                                   num_processes=args.num_processes,
                                                                   have_solved_state=True,
                                                                   device=args.device)
            perc = float(num_passed) / float(num_passed + num_failed)
            print(f"Gathered rollout {rollout}: #passes={num_passed}, #failed={num_failed}, perc={perc:.3f}")

            for k in range(args.ppo_updates):
                print(f'PPO update {k}')
                value_loss_epoch, action_loss_epoch, dist_entropy_epoch, transe_loss_epoch = ppo.update(rollouts)

                if args.debug:
                    print("value_loss_epoch ", value_loss_epoch, flush=True)
                    print("action_loss_epoch ", action_loss_epoch, flush=True)
                    print("dist_entropy_epoch ", dist_entropy_epoch, flush=True)
                    print("transe_loss_epoch ", transe_loss_epoch, flush=True)

                passed, pass_percentage = test_run_fixed_number_mazes(env=test_envs,
                                                                      policy=policy,
                                                                      rollout=rollout,
                                                                      num_test_mazes=args.num_test_episodes,
                                                                      pass_threshold=args.pass_threshold)
                if passed and args.save_model:
                    print("Level {}, rollout {} : percentage passed {:.3f} from total done mazes {}".format(
                        level, rollout, pass_percentage, args.num_test_episodes))
                    torch.save({
                        'level': level,
                        'transe': transe.state_dict(),
                        'executor': executor.state_dict(),
                        'policy': policy.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, f=f"maze/trained_model_level{level}.pt")

                    test_hold_out_mazes(test_maze_size=8, test_indices=test_indices,
                                        policy=policy, num_processes=args.num_processes)
                    break


