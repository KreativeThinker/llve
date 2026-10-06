from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    name: str
    num_rounds: int
    steps_per_episode: int
    batch_size: int
    learning_rate: float
    gamma: float
    entropy_coef: float
    max_episode_steps: int
    hidden_dim: int
    zfc_target: float

    lol_train_path: Path = None
    lol_test_path: Path = None


PAPER_CONFIG = TrainConfig(
    name="paper",
    num_rounds=1500,
    steps_per_episode=10,
    batch_size=2,
    learning_rate=0.002,
    gamma=0.95,
    entropy_coef=0.01,
    max_episode_steps=10,
    hidden_dim=32,
    zfc_target=2.5e5,
)

TINY_CONFIG = TrainConfig(
    name="tiny",
    num_rounds=10,
    steps_per_episode=5,
    batch_size=1,
    learning_rate=0.001,
    gamma=0.95,
    entropy_coef=0.01,
    max_episode_steps=5,
    hidden_dim=16,
    zfc_target=2.5e5,
)
