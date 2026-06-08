"""Proximal Policy Optimization (PPO) for continuous control."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rl.sac import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LR,
    _hidden_activation_module,
    _mlp_hidden_layers,
    _resolve_hidden_dims,
    _torch_load,
    resolve_device,
)


def _squashed_log_prob(dist, pre_tanh_action):
    logp = dist.log_prob(pre_tanh_action).sum(dim=-1, keepdim=True)
    logp -= (2 * (np.log(2) - pre_tanh_action - F.softplus(-2 * pre_tanh_action))).sum(
        dim=-1, keepdim=True
    )
    return logp


def _pre_tanh_from_action(action):
    return torch.atanh(torch.clamp(action, -0.999999, 0.999999))


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        hidden_dims=None,
        hidden_activation="relu",
    ):
        super().__init__()
        self.hidden_dims = _resolve_hidden_dims(hidden_dim, hidden_dims)
        self.hidden_activation = str(hidden_activation)
        self.net, last = _mlp_hidden_layers(
            obs_dim, self.hidden_dims, hidden_activation=self.hidden_activation
        )
        self.mean = nn.Linear(last, act_dim)
        self.log_std = nn.Linear(last, act_dim)

    def forward(self, obs, action=None, deterministic=False):
        h = self.net(obs)
        mu = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -20, 2)
        std = log_std.exp()
        dist = Normal(mu, std)

        if action is None:
            pre_tanh = mu if deterministic else dist.rsample()
            action = torch.tanh(pre_tanh)
        else:
            pre_tanh = _pre_tanh_from_action(action)

        logp = _squashed_log_prob(dist, pre_tanh)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action, logp, entropy


class ValueNetwork(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        hidden_dims=None,
        hidden_activation="relu",
    ):
        super().__init__()
        hidden_dims = _resolve_hidden_dims(hidden_dim, hidden_dims)
        body, last = _mlp_hidden_layers(
            obs_dim, hidden_dims, hidden_activation=hidden_activation
        )
        self.net = nn.Sequential(body, nn.Linear(last, 1))

    def forward(self, obs):
        return self.net(obs)


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, last_value, gamma, gae_lambda):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values + [float(last_value)], dtype=np.float32)
        adv = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            adv[t] = last_gae
        returns = adv + values[:-1]
        return adv, returns

    def as_tensors(self, device):
        return (
            torch.tensor(np.stack(self.obs), dtype=torch.float32, device=device),
            torch.tensor(np.stack(self.actions), dtype=torch.float32, device=device),
            torch.tensor(self.log_probs, dtype=torch.float32, device=device).unsqueeze(1),
        )


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        lr=DEFAULT_LR,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        hidden_dims=None,
        hidden_activation="relu",
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        minibatch_size=64,
        device=None,
    ):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dims = _resolve_hidden_dims(hidden_dim, hidden_dims)
        self.hidden_dim = self.hidden_dims[-1]
        self.hidden_activation = str(hidden_activation)
        self.lr = float(lr)
        self.device = resolve_device(device)

        act_kw = dict(hidden_dims=self.hidden_dims, hidden_activation=self.hidden_activation)
        self.actor = GaussianActor(obs_dim, act_dim, **act_kw).to(self.device)
        self.critic = ValueNetwork(obs_dim, **act_kw).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
        )
        self.buffer = RolloutBuffer()

    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, logp, _ = self.actor(obs_t, deterministic=deterministic)
            value = self.critic(obs_t)
        return (
            action.cpu().numpy()[0],
            float(logp.cpu().item()),
            float(value.cpu().item()),
        )

    def remember(self, obs, action, reward, done, log_prob, value):
        self.buffer.add(obs, action, reward, done, log_prob, value)

    def finish_rollout(self, last_obs, done):
        if len(self.buffer) == 0:
            return None
        if done:
            last_value = 0.0
        else:
            obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                last_value = float(self.critic(obs_t).cpu().item())
        adv, returns = self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        obs, actions, old_logp = self.buffer.as_tensors(self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device).unsqueeze(1)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = obs.shape[0]
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch_size):
                idx = indices[start : start + self.minibatch_size]
                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_logp = old_logp[idx]
                batch_adv = adv_t[idx]
                batch_ret = ret_t[idx]

                _, new_logp, entropy = self.actor(batch_obs, action=batch_actions)
                ratio = torch.exp(new_logp - batch_old_logp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(batch_obs)
                value_loss = F.mse_loss(values, batch_ret)

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.mean().item())
                stats["updates"] += 1

        if stats["updates"] > 0:
            for key in ("policy_loss", "value_loss", "entropy"):
                stats[key] /= stats["updates"]

        self.buffer.reset()
        return stats

    def set_learning_rate(self, lr):
        self.lr = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    def set_train_mode(self):
        self.actor.train()
        self.critic.train()

    def save_actor(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_actor(self, path):
        state = _torch_load(path)
        self.actor.load_state_dict(state)
        self.actor.eval()

    def save_checkpoint(self, path):
        payload = {
            "meta": {
                "algorithm": "ppo",
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden_dim": self.hidden_dim,
                "hidden_dims": self.hidden_dims,
                "hidden_activation": self.hidden_activation,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_eps": self.clip_eps,
                "lr": self.lr,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "ppo_epochs": self.ppo_epochs,
                "minibatch_size": self.minibatch_size,
            },
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path, load_optimizer=True):
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.set_train_mode()


def infer_dims_from_actor_file(path):
    """Read obs_dim, act_dim, hidden_dims from a PPO actor or full checkpoint."""
    payload = _torch_load(path, full_checkpoint=True)
    if isinstance(payload, dict) and "actor" in payload:
        meta = payload.get("meta", {})
        if meta.get("hidden_dims") is not None:
            return (
                int(meta["obs_dim"]),
                int(meta["act_dim"]),
                [int(h) for h in meta["hidden_dims"]],
            )
        if meta.get("hidden_dim") is not None:
            h = int(meta["hidden_dim"])
            return int(meta["obs_dim"]), int(meta["act_dim"]), [h, h]
        state = payload["actor"]
    else:
        state = payload

    obs_dim = int(state["net.0.weight"].shape[1])
    dims = [int(state["net.0.weight"].shape[0])]
    idx = 2
    while f"net.{idx}.weight" in state:
        dims.append(int(state[f"net.{idx}.weight"].shape[0]))
        idx += 2
    act_dim = int(state["mean.weight"].shape[0])
    return obs_dim, act_dim, dims
