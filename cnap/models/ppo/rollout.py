import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, num_steps, obs_shape, action_space, device, num_processes=1, have_solved_state=True):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_multidim = 1
        elif action_space.__class__.__name__ == 'MultiDiscrete':
            action_multidim = len(action_space.nvec)
        else:
            raise NotImplementedError

        self.actions = torch.zeros(num_steps, num_processes, action_multidim)

        # because continuous actions are discretised as int64
        self.actions = self.actions.long()

        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.have_solved_state = have_solved_state
        # solved obs, for when env is reset and the new maze is in obs
        if self.have_solved_state:
            self.solved_obs = torch.zeros(num_steps, num_processes, *obs_shape)

        self.device = device
        self.num_steps = num_steps
        self.step = 0

    def insert(self, obs, next_obs, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None):
        self.obs[self.step + 1].copy_(obs)
        self.rewards[self.step].copy_(rewards)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.masks[self.step + 1].copy_(masks)
        if bad_masks is not None:
            self.bad_masks[self.step + 1].copy_(bad_masks)
        self.solved_obs[self.step].copy_(next_obs)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.solved_obs = self.solved_obs.to(device)

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        enable_time_limit=False):
        if enable_time_limit:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] *
                                          gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                         + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                                         gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if self.have_solved_state:
                solved_obs_batch = self.solved_obs.view(-1, *self.solved_obs.size()[2:])[indices]
            else:
                solved_obs_batch = None

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]
            adv_targ = adv_targ.to(self.device)

            yield obs_batch, actions_batch, \
                  value_preds_batch, return_batch, \
                  masks_batch, old_action_log_probs_batch, solved_obs_batch, adv_targ


def gather_fixed_episodes_rollout(env, policy, num_episodes, gamma, num_processes, device, file=None, deterministic=False):
    all_obs = []
    all_actions = []
    all_log_probs = []
    all_values = []
    all_rewards = []
    all_masks = []
    all_next_obs = []

    obs = env.reset()
    all_obs += [obs]
    done_ep = 0
    step = 0
    while done_ep < num_episodes:
        with torch.no_grad():
            value, action, log_probs = policy.act(all_obs[-1].to(device), deterministic=deterministic)

        # env.render()
        obs, reward, done, _ = env.step(action)
        done_ep += torch.sum(torch.tensor(done))

        reward = torch.tensor(reward)
        mask = ~ torch.tensor(done).unsqueeze_(-1)

        next_obs = obs

        all_obs += [obs]
        all_actions += [action]
        all_log_probs += [log_probs]
        all_values += [value]
        all_rewards += [reward]
        all_masks += [mask]

        all_next_obs += [next_obs]

        step += 1

    print("Number of steps ", step)
    if file:
        file.writelines("Number of steps " + str(step) + "\n")

    rollouts = RolloutStorage(num_steps=step, obs_shape=env.observation_space.shape,
                              action_space=env.action_space, num_processes=num_processes,
                              device=device,
                              have_solved_state=True)

    rollouts.obs = torch.stack(all_obs, dim=0)
    rollouts.actions = torch.stack(all_actions, dim=0)
    rollouts.action_log_probs = torch.stack(all_log_probs, dim=0)
    rollouts.value_preds = torch.cat((torch.stack(all_values, dim=0), torch.zeros_like(all_values[0]).unsqueeze(0)),
                                     dim=0)
    rollouts.rewards = torch.stack(all_rewards, dim=0)
    rollouts.masks = torch.cat((torch.zeros_like(all_masks[0]).unsqueeze(0), torch.stack(all_masks, dim=0)), dim=0)

    rollouts.solved_obs = torch.stack(all_next_obs, dim=0)

    rollouts.to(device)

    with torch.no_grad():
        next_value = policy.get_value(rollouts.obs[-1].to(device)).detach()
    rollouts.compute_returns(next_value=next_value, gamma=gamma, use_gae=False, gae_lambda=0.01)
    return rollouts


def gather_fixed_steps_rollout(env, policy, num_steps, gamma, num_processes, device, deterministic=False, use_gae=False,
                               gae_lambda=0.01):
    all_obs = []
    all_actions = []
    all_log_probs = []
    all_values = []
    all_rewards = []
    all_masks = []
    all_next_obs = []

    obs = env.reset()
    all_obs += [obs]
    done_ep = 0
    step = 0
    episode_rewards = []

    while step < num_steps:

        with torch.no_grad():
            value, action, log_probs = policy.act(all_obs[-1].to(device), deterministic=deterministic)

        obs, reward, done, infos = env.step(action)
        done_ep += torch.sum(torch.tensor(done))

        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

        reward = torch.tensor(reward)
        mask = ~ torch.tensor(done).unsqueeze_(-1)

        next_obs = obs

        all_obs += [obs]
        all_actions += [action]
        all_log_probs += [log_probs]
        all_values += [value]
        all_rewards += [reward]
        all_masks += [mask]

        all_next_obs += [next_obs]

        step += 1

    print("Number of steps ", step * num_processes)
    print("Number of episodes ", len(episode_rewards))

    rollouts = RolloutStorage(num_steps=step, obs_shape=env.observation_space.shape,
                              action_space=env.action_space, num_processes=num_processes,
                              device=device,
                              have_solved_state=True)

    rollouts.obs = torch.stack(all_obs, dim=0)
    rollouts.actions = torch.stack(all_actions, dim=0)
    rollouts.action_log_probs = torch.stack(all_log_probs, dim=0)
    rollouts.value_preds = torch.cat((torch.stack(all_values, dim=0),
                                      torch.zeros_like(all_values[0]).unsqueeze(0)), dim=0)
    rollouts.rewards = torch.stack(all_rewards, dim=0)
    rollouts.masks = torch.cat((torch.zeros_like(all_masks[0]).unsqueeze(0), torch.stack(all_masks, dim=0)), dim=0)

    rollouts.solved_obs = torch.stack(all_next_obs, dim=0)

    rollouts.to(device)

    with torch.no_grad():
        next_value = policy.get_value(rollouts.obs[-1].to(device)).detach()
    rollouts.compute_returns(next_value=next_value, gamma=gamma, use_gae=use_gae, gae_lambda=gae_lambda)

    return rollouts, done_ep, episode_rewards


def gather_maze_rollout(env, policy, num_steps, gamma, num_processes, device, deterministic=False, have_solved_state=True):
    rollouts = RolloutStorage(num_steps=num_steps,
                              obs_shape=env.observation_space.shape,
                              action_space=env.action_space,
                              num_processes=num_processes,
                              device=device,
                              have_solved_state=have_solved_state)
    num_passed = 0
    num_failed = 0
    obs = env.reset()
    rollouts.obs[0].copy_(obs)

    for step in range(num_steps):
        with torch.no_grad():
            value, action, log_probs = policy.act(rollouts.obs[step].to(device), deterministic=deterministic)
        # env.render()
        obs, reward, done, solved_obs = env.step(action)
        mask = ~ torch.tensor(done).unsqueeze_(-1)
        if have_solved_state:
            solved_obs = torch.tensor([solved["solved"] for solved in solved_obs])
            next_obs = mask.unsqueeze(-1).unsqueeze(-1) * obs + \
                       (~mask.unsqueeze(-1).unsqueeze(-1)) * solved_obs
        else:
            next_obs = None

        num_passed += torch.sum(torch.eq(reward, torch.ones_like(reward)))
        num_failed += torch.sum(torch.eq(reward, torch.ones_like(reward) * -1))

        rollouts.insert(obs=obs, actions=action.squeeze_(0), action_log_probs=log_probs.squeeze_(0),
                        value_preds=value.squeeze_(0), rewards=reward, masks=mask, next_obs=next_obs)

    rollouts.to(device)

    with torch.no_grad():
        next_value = policy.get_value(rollouts.obs[-1].to(device)).detach()

    rollouts.compute_returns(next_value=next_value, gamma=gamma, use_gae=False, gae_lambda=0.01)
    return rollouts, num_passed, num_failed
