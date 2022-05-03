import torch
import torch.nn as nn


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 optimizer,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 transe_loss_coef=0.,
                 mini_batch_size=None,
                 transe_detach=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.transe_loss_coef = transe_loss_coef

        self.transe_detach = transe_detach

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optimizer

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        transe_loss_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch, self.mini_batch_size)
            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
                old_action_log_probs_batch, solved_obs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                if self.transe_loss_coef > 0.:
                    # transe_loss = self.transe_loss_coef * \
                                  # self.actor_critic.transe.transition_loss(obs_batch, actions_batch.squeeze(-1),
                                                                           # solved_obs_batch,
                                                                           # self.transe_detach)
                    transe_loss = self.transe_loss_coef * \
                                  self.actor_critic.transe.contrastive_loss(obs_batch, actions_batch.squeeze(-1),
                                                                            solved_obs_batch)
                    loss += transe_loss
                    transe_loss_epoch += transe_loss.item()

                loss.backward(retain_graph=True)
                """
                total_norm = 0
                for p in self.actor_critic.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                print("actor_critic ", total_norm)
                """
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        transe_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, transe_loss_epoch
