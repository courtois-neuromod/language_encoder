from dataclasses import dataclass

from encoder import test_ridgeReg, train_ridgeReg
from encoder_dataclass import DataBaseConfig
from utils import (
    build_target,
    extract_feature_regressor,
    split_episodes,
)

data_config = DataBaseConfig()


train_groups, train_runs, val_runs, test_runs, val_season = split_episodes(
    data_config,
)


y_train, length_train, train_groups = build_target(
    data_config,
    train_runs,
    train_groups,
)
# breakpoint()
x_train = extract_feature_regressor(data_config, train_runs, length_train)


y_val, length_val, _ = build_target(data_config, val_runs)
x_val = extract_feature_regressor(data_config, val_runs, length_val)


model = train_ridgeReg(
    x_train,
    y_train,
    train_groups,
    data_config,
)


test_ridgeReg(
    data_config,
    model,
    x_train,
    y_train,
    x_val,
    y_val,
)
