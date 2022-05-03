import os
import numpy as np
import torch
import gym
from gym import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from stable_baselines3.common.monitor import Monitor

from environment.maze.maze_env import MazeEnv


def construct_mask(bins):
    a = np.zeros([bins, bins])
    for i in range(bins):
        for j in range(bins):
            if i + j <= bins - 1:
                a[i, j] = 1.0
    return a


def discretizing_wrapper(env, K):
    """
    # discretize each action dimension to K bins
    """
    unwrapped_env = env.unwrapped
    unwrapped_env.orig_step_ = unwrapped_env.step
    unwrapped_env.orig_reset_ = unwrapped_env.reset

    action_low, action_high = env.action_space.low, env.action_space.high
    naction = action_low.size
    action_table = np.reshape([np.linspace(action_low[i], action_high[i], K) for i in range(naction)],
                              [naction, K])
    assert action_table.shape == (naction, K)

    def discretizing_reset():
        obs = unwrapped_env.orig_reset_()
        return obs

    def discretizing_step(action):
        # action is a sequence of discrete indices
        action_cont = action_table[np.arange(naction), action]
        obs, rew, done, info = unwrapped_env.orig_step_(action_cont)

        return (obs, rew, done, info)

    # change action space
    env.action_space = spaces.MultiDiscrete([K for _ in range(naction)])

    unwrapped_env.step = discretizing_step
    unwrapped_env.reset = discretizing_reset

    return env


def make_env(env_type, seed, rank, log_dir, continuous=False, bins=1):
    def _thunk():
        if not continuous:
            if env_type == 'cartpole':
                env = gym.make("CartPole-v0")
            elif env_type == 'mountaincar':
                env = gym.make('MountainCar-v0')
            elif env_type == 'acrobot':
                env = gym.make('Acrobot-v1')
        else:
            if env_type == 'mountaincar-continuous':
                env = gym.make('MountainCarContinuous-v0')
            elif env_type == 'pendulum':
                env = gym.make('Pendulum-v1')
            elif env_type == 'walker':
                env = gym.make('Walker2d-v2')
            elif env_type == 'halfcheetah':
                env = gym.make('HalfCheetah-v2')
            elif env_type == 'ant':
                env = gym.make('Ant-v2')
            elif env_type == 'hopper':
                env = gym.make('Hopper-v2')
            elif env_type == 'humanoid':
                env = gym.make('Humanoid-v2')
            elif env_type == 'humanoid-standup':
                env = gym.make('HumanoidStandup-v2')
            elif env_type == 'inverted-double-pendulum':
                env = gym.make('InvertedDoublePendulum-v2')
            elif env_type == 'inverted-pendulum':
                env = gym.make('InvertedPendulum-v2')
            elif env_type == 'reacher':
                env = gym.make('Reacher-v2')
            elif env_type == 'swimmer':
                env = gym.make('Swimmer-v2')
            env = discretizing_wrapper(env=env, K=bins)

        if seed:
            env.seed(seed + rank)
        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)
        if log_dir:
            env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
        return env
    return _thunk


def make_vec_envs(env_type,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  continuous,
                  bins=11,
                  normalise=False,
                  num_frame_stack=None,
                  record_video=False,
                  save_video_interval=204800,
                  video_length=500):
    envs = [
        make_env(env_type=env_type, seed=seed, rank=i, log_dir=log_dir,
                 continuous=continuous, bins=bins)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # Normalise env
    # note: encoder should also be trained with normalised envs
    if normalise:
        if len(envs.observation_space.shape) == 1:
            if gamma is None:
                envs = VecNormalize(envs, norm_reward=False)
            else:
                envs = VecNormalize(envs, gamma=gamma)

    if record_video:
        envs = VecVideoRecorder(envs, os.path.join(log_dir, "videos"),
                                record_video_trigger=lambda x: (x+1) % save_video_interval == 0,
                                video_length=video_length,
                                name_prefix=f"{env_type}_{seed}")

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


def make_maze_env(indices_list, train, maze_size=8, active_index=None):
    def _thunk():
        return MazeEnv(maze_size=maze_size,
                       maze_indices_list=indices_list,
                       train=train,
                       active_index=active_index)

    return _thunk


def make_vec_maze_envs(num_processes,
                       indices_list,
                       train,
                       num_frame_stack=None,
                       maze_size=8,
                       device=None):
    envs = [make_maze_env(indices_list=indices_list, train=train, maze_size=maze_size)
            for _ in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)  # run multiple environments in parallel
    else:
        envs = DummyVecEnv(envs)  # run multiple environments sequentially

    # if len(envs.observation_space.shape) == 1:
        # envs = VecNormalize(envs)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 1, device)
    return envs


def test_vec_envs(num_processes,
                  maze_size,
                  train_maze,
                  maze_indices_list=None,
                  num_frame_stack=None,
                  device=None):

    maze_subindices = []
    if maze_indices_list is not None:
        per_process = len(maze_indices_list) // num_processes
        rem = len(maze_indices_list) % num_processes
        cur_ind = 0
        for i in range(num_processes):
            maze_subindices.append([])
            if i < rem:
                maze_subindices[i].append(maze_indices_list[cur_ind])
                cur_ind += 1
            maze_subindices[i] += maze_indices_list[cur_ind:cur_ind + per_process]
            cur_ind += per_process
        assert cur_ind == len(maze_indices_list)
    else:
        for _ in range(num_processes):
            maze_subindices.append(None)

    envs = [
        make_maze_env(maze_size=maze_size,
                      indices_list=maze_subindices[i],
                      train=train_maze,
                      active_index=0)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 1, device)
    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or isinstance(actions, torch.IntTensor) \
                or isinstance(actions, torch.cuda.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) /
                          np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs, self.clip_obs)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


