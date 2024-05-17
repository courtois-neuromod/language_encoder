"""."""

from dataclasses import dataclass

from encoder import test_ridgeReg, train_ridgeReg
from encoder_dataclass import DataBaseConfig
from utils import (
    build_target,
    scale_embedding,
    split_episodes,
)

data_config = DataBaseConfig()


train_groups, train_runs, val_runs, test_runs, val_season = split_episodes(
    data_config,
)


y_train, run_length, train_groups = build_target(
    data_config,
    train_runs,
    train_groups,
)
scale_embedding(data_config, train_runs, run_length)
