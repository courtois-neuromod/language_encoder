"""."""
from dataclasses import dataclass

from encoder import DataBaseConfig, test_ridgeReg, train_ridgeReg
from utils import (
    build_target,
    extract_feature_regressor,
    split_episodes,
)


@dataclass
class DataConfig(DataBaseConfig):
    bold_dir: str = "/scratch/ibilgin/friends.timeseries/"
    stimuli_dir: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/gpt2"
    )
    output_dir: str = "/scratch/ibilgin/Dropbox/friends_encoder/data/ridge_regression"
    tr_tsv_path: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/word_alignment/"
    )
    hrf_model = "spm"


data_config = DataConfig()


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
