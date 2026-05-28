import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_load(path, full_checkpoint=False):
    try:
        return torch.load(
            path,
            map_location=DEVICE,
            weights_only=not full_checkpoint,
        )
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def infer_dims_from_actor_file(path):
    """
    Read obs_dim, act_dim, hidden_dim from actor-only or full SAC checkpoint.
    """
    payload = _torch_load(path, full_checkpoint=True)
    if isinstance(payload, dict) and "actor" in payload:
        meta = payload.get("meta", {})
        if meta.get("hidden_dim") is not None:
            return (
                int(meta["obs_dim"]),
                int(meta["act_dim"]),
                int(meta["hidden_dim"]),
            )
        state = payload["actor"]
    else:
        state = payload

    hidden_dim = int(state["net.0.weight"].shape[0])
    obs_dim = int(state["net.0.weight"].shape[1])
    act_dim = int(state["mean.weight"].shape[0])
    return obs_dim, act_dim, hidden_dim


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -20, 2)
        std = log_std.exp()
        dist = Normal(mu, std)

        action = mu if deterministic else dist.rsample()

        logp = None
        if with_logprob:
            logp = dist.log_prob(action).sum(dim=-1, keepdim=True)
            logp -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
                dim=-1, keepdim=True
            )

        return torch.tanh(action), logp


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(s).to(DEVICE),
            torch.FloatTensor(a).to(DEVICE),
            torch.FloatTensor(r, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(ns).to(DEVICE),
            torch.FloatTensor(d, dtype=torch.float32).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        hidden_dim=256,
        buffer_size=100_000,
        batch_size=256,
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.buffer_size = buffer_size

        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_dim).to(DEVICE)
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dim).to(DEVICE)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dim).to(DEVICE)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_dim).to(DEVICE)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_dim).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.buffer = ReplayBuffer(buffer_size)

    def act(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
        return action.cpu().numpy()[0]

    def remember(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, act, r, ns, d = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_action, next_logp = self.actor(ns)
            target_q = r + self.gamma * (1 - d) * (
                torch.min(
                    self.q1_target(ns, next_action), self.q2_target(ns, next_action)
                )
                - self.alpha * next_logp
            )

        q_loss = F.mse_loss(self.q1(s, act), target_q) + F.mse_loss(
            self.q2(s, act), target_q
        )
        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()

        sampled_action, logp = self.actor(s)
        actor_loss = (
            self.alpha * logp - torch.min(self.q1(s, sampled_action), self.q2(s, sampled_action))
        ).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, online_net, target_net):
        for p, pt in zip(online_net.parameters(), target_net.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

    def save_actor(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_actor(self, path):
        state = _torch_load(path)
        self.actor.load_state_dict(state)
        self.actor.eval()

    def load_actor_for_training(self, path):
        """Load pretrained actor weights; keep critics trainable for online SAC."""
        state = _torch_load(path)
        self.actor.load_state_dict(state)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.set_train_mode()

    def set_train_mode(self):
        self.actor.train()
        self.q1.train()
        self.q2.train()

    def save_checkpoint(self, path):
        payload = {
            "meta": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden_dim": self.hidden_dim,
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": self.alpha,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "buffer_size": self.buffer_size,
            },
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path, load_optimizers=True):
        try:
            payload = torch.load(path, map_location=DEVICE, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=DEVICE)
        meta = payload.get("meta", {})
        if meta.get("obs_dim") not in (None, self.obs_dim) or meta.get("act_dim") not in (
            None,
            self.act_dim,
        ):
            raise ValueError(
                f"Checkpoint obs/act dims {meta.get('obs_dim')}/{meta.get('act_dim')} "
                f"do not match agent {self.obs_dim}/{self.act_dim}"
            )
        self.actor.load_state_dict(payload["actor"])
        self.q1.load_state_dict(payload["q1"])
        self.q2.load_state_dict(payload["q2"])
        self.q1_target.load_state_dict(payload["q1_target"])
        self.q2_target.load_state_dict(payload["q2_target"])
        if load_optimizers:
            if "actor_opt" in payload:
                self.actor_opt.load_state_dict(payload["actor_opt"])
            if "critic_opt" in payload:
                self.critic_opt.load_state_dict(payload["critic_opt"])
        self.set_train_mode()

