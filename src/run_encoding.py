"""."""

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


data_config = DataConfig()
seasons = list_seasons(data_config.tsv_path)

# get layer embeddings
# get_layer_embeddding(data_config)


# get train and test fmri and stimuli sets.
train_groups, train_runs, val_runs, test_runs, val_season = split_episodes(
    data_config,
)

print(
    test_runs
)  # ['ses-027_task-s03e01a_timeseries', 'ses-027_task-s03e01b_timeseries', 'ses-028_task-s03e02a_timeseries'] name of the fmri files

print(train_runs)
print(len(train_runs))  # 194

print(
    train_groups
)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]


training_seasons = list(
    filter(lambda x: x not in [data_config.test_season, val_season], seasons),
)  # name of the training seasons

feature_train = build_text(data_config, training_seasons)  # list of list
print(f"len of feature train: {len(feature_train)}")
print(f"type of feature_train: {type(feature_train)}")
print(
    len(feature_train[1])
)  # this is all the episodes [dim1] and words in each episode [dim2] each are a vector of size of 768


feature_val = build_text(data_config, val_season)
# feature_test = build_text(data_config,data_config.test_season) # we do not use it for now


gentles_train_set = build_text_input(data_config, training_seasons)

print(f"len_gentles_train_set: {gentles_train_set}")
print(f"len_gentles_train_set: {len(gentles_train_set)}")
print(f"len_gentles_train_set: {len(gentles_train_set[0])}")


# gentles_val_set = build_text_input(data_config, val_season)
# # gentles_test = build_text_input(data_config, data_config.test_season)


y_train, length_train, train_groups = build_output(
    data_config,
    train_runs,
    train_groups,
)

# print(f"length_train: {length_train}")
# print(f"y_train: {y_train}")
# print(f"train_groups: {train_groups}")
# print(f"len of y_train: {len(y_train)}")


y_val, length_val, _ = build_output(data_config, val_runs)


y_test, length_test, _ = build_output(data_config, test_runs)

out_per_fold = 40
splits = []
logo = LeavePOut(out_per_fold)
nscans = [f.shape[0] for f in y_train]  # number of scnas per session


counter = 0
limit = 2
for train, test in tqdm(logo.split(feature_train), desc="Processing splits"):
    # Compute the number of rows in each run (= the number of samples extracted from the model for each run)
    print(train)
    print(test)
    if counter >= limit:
        break
    gentles_train = [gentles_train_set[i] for i in train]
    # print(f"len_gentles_train_set: {len(gentles_train_set)}")
    # print(f"len_gentles_train_set[0]: {gentles_train_set[0]}")
    groups_train = get_groups(gentles_train)
    # print(f"gentles_train:{len(gentles_train[0])}")
    # print(f"groups_train:{len(groups_train[0])}")
    gentles_test = [gentles_train_set[i] for i in test]
    groups_test = get_groups(gentles_test)
    print(f"groups_test:{groups_test}")

    # Preparing fMRI data

    my_list = [feature_train[i] for i in train]
    print(f"len of my list = {len(my_list)}")

    splits.append(
        {
            "fmri_train": [y_train[i] for i in train],
            "fmri_test": [y_train[i] for i in test],
            "features_gpt2_train": [feature_train[i] for i in train],
            "features_gpt2_test": [feature_train[i] for i in test],
            "groups_train": groups_train,
            "nscans_train": [nscans[i] for i in train],
            "gentles_train": gentles_train,
            "groups_test": groups_test,
            "nscans_test": [nscans[i] for i in test],
            "gentles_test": gentles_test,
        }
    )
    counter += 1

# print(f"splits: {splits}")


fmri_ndim = None
features_ndim = None
features_reduction_method = None  #'pca'
fmri_reduction_method = None
tr = 1.49
encoding_method = "hrf"
linearmodel = "ridgecv"

encoder = Encoder(
    linearmodel=linearmodel,
    features_reduction_method=features_reduction_method,
    fmri_reduction_method=fmri_reduction_method,
    fmri_ndim=fmri_ndim,
    features_ndim=features_ndim,
    encoding_method=encoding_method,
    tr=tr,
)


def adjust_list_sizes(list1, list2):
    # Find the smaller length
    min_length = min(len(list1), len(list2))

    # Adjust both lists to the smaller length
    list1 = list1[:min_length]
    list2 = list2[:min_length]

    return list1, list2


# HERE THEY concatante the features for that iteration, the number of features in total
should match the number of words hence groups.

Apparently the split does not work as it should be which means split might be taking
the wrong data

1. print the len of features and len of gentles to make sure they match before the out_per_fold
and after the fold.

scores = {"gpt2": []}
for i, split in tqdm(enumerate(splits), desc="Processing encoding"):
    print(i)
    print(split)
    fmri_train = np.vstack(split["fmri_train"])
    fmri_test = np.vstack(split["fmri_test"])
    features_gpt2_train = np.vstack(split["features_gpt2_train"])
    features_gpt2_test = np.vstack(split["features_gpt2_test"])

    groups_train = split["groups_train"]
    nscans_train = split["nscans_train"]
    gentles_train = split["gentles_train"]

    groups_test = split["groups_test"]
    nscans_test = split["nscans_test"]
    gentles_test = split["gentles_test"]

    # Fitting the model with GPT-2

    features_gpt2_train, fmri_train = adjust_list_sizes(features_gpt2_train, fmri_train)
    # features_gpt2_test, fmri_test = adjust_list_sizes(features_gpt2_test, fmri_test)
    print(f"len_features_gpt2_train: {len(features_gpt2_train)}")
    print(f"len_fmri_train: {len(fmri_train)}")
    # print(f"len_features_gpt2_test: {len(features_gpt2_test)}")
    # print(f"len_fmri_test: {len(fmri_test)}")
    encoder.fit(
        features_gpt2_train,
        fmri_train,
        groups=groups_train,
        gentles=gentles_train,
        nscans=nscans_train,
    )
    encoder.set_features_pipe(
        features_gpt2_test, groups_test, gentles_test, nscans_test
    )
    pred = encoder.predict(features_gpt2_test)
    scores_gpt2 = encoder.eval(pred, fmri_test)
    scores["gpt2"].append(scores_gpt2)

print("saving the pickle")
with open(f"{data_config.output_dir}/ridge_results.csv", "wb") as f:
    pickle.dump(scores, f)
print("saving the csv")

with open(
    f"{data_config.output_dir}/ridge_results.csv", "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    for key, value in scores.items():
        writer.writerow([key, value])
