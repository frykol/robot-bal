import argparse
from pathlib import Path

import torch

from rl.sac import SquashedGaussianActor


def export_torchscript(weights_path, output_path, obs_dim=4, act_dim=1):
    actor = SquashedGaussianActor(obs_dim, act_dim)
    state = torch.load(weights_path, map_location="cpu")
    actor.load_state_dict(state)
    actor.eval()

    class DeterministicPolicy(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, obs):
            action, _ = self.model(obs, deterministic=True, with_logprob=False)
            return action

    policy = DeterministicPolicy(actor)
    traced = torch.jit.trace(policy, torch.randn(1, obs_dim))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"Saved TorchScript policy: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", default="artifacts/actor_sim.pt")
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/policy.ts"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_torchscript(args.weights_path, args.output_path)

