import csv
import pickle
from dataclasses import dataclass

import numpy as np
from encoder import Encoder
from encoder_m import DataBaseConfig, test_ridgeReg, train_ridgeReg
from ridge import CustomRidge
from sklearn.model_selection import LeavePOut, train_test_split
from tqdm import tqdm
from utils import (
    build_output,
    build_text,
    build_text_input,
    get_groups,
    list_seasons,
    load_stmuli,
    preprocess_stimuli_data,
    split_episodes,
)
from utils_a import get_groups
from visualize import create_map


@dataclass
class DataConfig(DataBaseConfig):
    bold_dir: str = "/scratch/ibilgin/friends.timeseries/"
    stimuli_dir: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/gpt2"
    )
    output_dir: str = "/scratch/ibilgin/Dropbox/language_encoder/data/ridge_regression"
    tsv_path: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/"
    )


data_length = 10

data_config = DataConfig()
seasons = list_seasons(data_config.tsv_path)


# get train and test fmri and stimuli sets.
train_groups, train_runs, val_runs, test_runs, val_season = split_episodes(
    data_config,
)
training_seasons = list(
    filter(lambda x: x not in [data_config.test_season, val_season], seasons),
)  # name of the training seasons
feature_train = build_text(data_config, training_seasons[:1])  # list of list
# print(f"len of feature train: {len(feature_train)}")
# print(f"type of feature_train: {type(feature_train)}")
# print(
#     len(feature_train[1])
# )  # this is all the episodes [dim1] and words in each episode [dim2] each are a vector of size of 768
feature_train = feature_train[:data_length]
pad_rows = 3
delta_time = 1.49
gentles_train_set = build_text_input(
    data_config,
    training_seasons[:1],
    pad_rows,
    delta_time,
)
