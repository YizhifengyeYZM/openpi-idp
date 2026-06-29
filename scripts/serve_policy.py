import dataclasses
import enum
import logging
import os
import pathlib
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _apply_checkpoint_config(train_config: _config.TrainConfig, checkpoint_dir: str) -> _config.TrainConfig:
    metadata_path = pathlib.Path(checkpoint_dir) / "metadata.pt"
    if not metadata_path.exists():
        return train_config

    try:
        import torch

        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not load checkpoint metadata from %s: %s", metadata_path, exc)
        return train_config

    checkpoint_config = metadata.get("config", {})
    overrides = {
        key: checkpoint_config[key]
        for key in ("pytorch_loss_type", "pytorch_idp_tau", "pytorch_valid_action_dim", "pytorch_freeze_paligemma")
        if key in checkpoint_config
    }
    if overrides:
        logging.info("Applying PyTorch checkpoint config overrides: %s", overrides)
        train_config = dataclasses.replace(train_config, **overrides)
    return train_config


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            train_config = _apply_checkpoint_config(train_config, args.policy.dir)
            if loss_type := os.environ.get("OPENPI_PYTORCH_LOSS_TYPE"):
                if loss_type not in {"flow", "idp_iso", "idp_geo"}:
                    raise ValueError(f"Unsupported OPENPI_PYTORCH_LOSS_TYPE: {loss_type}")
                train_config = dataclasses.replace(train_config, pytorch_loss_type=loss_type)
            sample_kwargs = None
            if num_steps := os.environ.get("OPENPI_SAMPLE_NUM_STEPS"):
                sample_kwargs = {"num_steps": int(num_steps)}
            if os.environ.get("OPENPI_DISABLE_PYTORCH_COMPILE") == "1":
                train_config = dataclasses.replace(
                    train_config,
                    model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
                )
            return _policy_config.create_trained_policy(
                train_config, args.policy.dir, default_prompt=args.default_prompt, sample_kwargs=sample_kwargs
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
