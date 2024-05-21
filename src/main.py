from dataclasses import dataclass

from encoder import test_ridgeReg, train_ridgeReg
from encoder_dataclass import DataBaseConfig
from utils import (
    build_target,
    extract_feature_regressor,
    process_embeddings,
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
train_embeddings, scaler = process_embeddings(data_config, train_runs)

print(f"scaler mean: {scaler.mean_}")
print(f"scaler scale: {scaler.scale_}")

print(f"length of train_embeddings: {len(train_embeddings)}")


x_train = extract_feature_regressor(data_config, train_embeddings, train_runs, length_train)
print(f"length of x_train: {len(x_train)}")
print(f"length of y_train: {len(y_train)}")


y_val, length_val, _ = build_target(data_config, val_runs)
print(f"length of val_runs: {len(val_runs)}")

print(f"length of y_val: {len(y_val)}")
val_embeddings, _ = process_embeddings(data_config, val_runs, scaler)
print(f"length of val_embeddings: {len(val_embeddings)}")


x_val = extract_feature_regressor(data_config, val_embeddings, val_runs, length_val)
print(f"length of x_val: {len(x_val)}")


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
