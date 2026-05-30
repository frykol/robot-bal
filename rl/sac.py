import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _cuda_usable():
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(0)
        # Current PyTorch wheels often require sm_75+ (CC 7.5).
        if major < 7:
            return False
        x = torch.zeros(1, device="cuda")
        x @ x
        return True
    except RuntimeError:
        return False


def resolve_device(requested=None):
    if requested is not None:
        dev = torch.device(requested)
        if dev.type == "cuda" and not _cuda_usable():
            print(
                "CUDA requested but GPU is unavailable or incompatible; using CPU.",
                file=sys.stderr,
            )
            return torch.device("cpu")
        return dev

    if _cuda_usable():
        return torch.device("cuda")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(
            f"GPU {name} (CC {cap[0]}.{cap[1]}) is not supported by this PyTorch build; using CPU.",
            file=sys.stderr,
        )
    return torch.device("cpu")


DEVICE = resolve_device()

DEFAULT_HIDDEN_DIM = 16
DEFAULT_LR = 3e-4


def _torch_load(path, full_checkpoint=False):
    try:
        return torch.load(
            path,
            map_location=DEVICE,
            weights_only=not full_checkpoint,
        )
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def _resolve_hidden_dims(hidden_dim, hidden_dims):
    if hidden_dims is not None:
        dims = [int(h) for h in hidden_dims]
        if len(dims) < 1:
            raise ValueError("hidden_dims must have at least one layer size")
        return dims
    h = int(hidden_dim)
    return [h, h]


def _hidden_activation_module(name):
    key = str(name).lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported hidden_activation: {name!r} (use 'tanh' or 'relu')")


def _mlp_hidden_layers(in_features, hidden_dims, hidden_activation="relu"):
    layers = []
    width = in_features
    act = _hidden_activation_module(hidden_activation)
    for h in hidden_dims:
        layers.append(nn.Linear(width, h))
        layers.append(act)
        width = h
    return nn.Sequential(*layers), width


def _infer_hidden_dims_from_actor_state(state):
    obs_dim = int(state["net.0.weight"].shape[1])
    dims = [int(state["net.0.weight"].shape[0])]
    idx = 2
    while f"net.{idx}.weight" in state:
        dims.append(int(state[f"net.{idx}.weight"].shape[0]))
        idx += 2
    return obs_dim, dims


def infer_dims_from_actor_file(path):
    """
    Read obs_dim, act_dim, hidden_dims (list) from actor-only or full SAC checkpoint.
    """
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

    obs_dim, hidden_dims = _infer_hidden_dims_from_actor_state(state)
    act_dim = int(state["mean.weight"].shape[0])
    return obs_dim, act_dim, hidden_dims


class SquashedGaussianActor(nn.Module):
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
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        hidden_dims=None,
        hidden_activation="relu",
    ):
        super().__init__()
        hidden_dims = _resolve_hidden_dims(hidden_dim, hidden_dims)
        body, last = _mlp_hidden_layers(
            obs_dim + act_dim, hidden_dims, hidden_activation=hidden_activation
        )
        self.net = nn.Sequential(body, nn.Linear(last, 1))

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return (
            torch.tensor(s, dtype=torch.float32).to(device),
            torch.tensor(a, dtype=torch.float32).to(device),
            torch.tensor(r, dtype=torch.float32).to(device).unsqueeze(1),
            torch.tensor(ns, dtype=torch.float32).to(device),
            torch.tensor(d, dtype=torch.float32).to(device).unsqueeze(1),
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
        lr=DEFAULT_LR,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        hidden_dims=None,
        hidden_activation="relu",
        buffer_size=100_000,
        batch_size=256,
        device=None,
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dims = _resolve_hidden_dims(hidden_dim, hidden_dims)
        self.hidden_dim = self.hidden_dims[-1]
        self.hidden_activation = str(hidden_activation)
        self.lr = lr
        self.buffer_size = buffer_size
        self.device = resolve_device(device)

        act_kw = dict(hidden_dims=self.hidden_dims, hidden_activation=self.hidden_activation)
        self.actor = SquashedGaussianActor(obs_dim, act_dim, **act_kw).to(self.device)
        self.q1 = QNetwork(obs_dim, act_dim, **act_kw).to(self.device)
        self.q2 = QNetwork(obs_dim, act_dim, **act_kw).to(self.device)
        self.q1_target = QNetwork(obs_dim, act_dim, **act_kw).to(self.device)
        self.q2_target = QNetwork(obs_dim, act_dim, **act_kw).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.buffer = ReplayBuffer(buffer_size)

    def act(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
        return action.cpu().numpy()[0]

    def remember(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, act, r, ns, d = self.buffer.sample(self.batch_size, self.device)

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

    def set_learning_rate(self, lr):
        """Update Adam LR for actor and both critics (e.g. linear decay per episode)."""
        self.lr = float(lr)
        for group in self.actor_opt.param_groups:
            group["lr"] = self.lr
        for group in self.critic_opt.param_groups:
            group["lr"] = self.lr

    def set_alpha(self, alpha):
        """Entropy coefficient (0 = no entropy bonus in actor/critic targets)."""
        self.alpha = float(alpha)

    def save_checkpoint(self, path):
        payload = {
            "meta": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden_dim": self.hidden_dim,
                "hidden_dims": self.hidden_dims,
                "hidden_activation": self.hidden_activation,
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
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
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

