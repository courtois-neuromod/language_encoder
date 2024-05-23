"""``project`` utilities."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from nilearn.glm.first_level import compute_regressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def read_tsv(tsv_path):
    """."""
    file = pd.read_csv(tsv_path, sep="\t")
    return file


def generate_one_hot_vector(gentle):
    """."""

    gentle.loc[gentle["word"] != " ", "word"] = 1
    onsets = gentle["onset"].to_numpy()
    word = gentle["word"].to_numpy()
    duration = np.zeros(len(word))
    regressor_vector = np.stack([onsets, duration, word])
    return regressor_vector


def split_episodes(
    data_config,
) -> tuple:
    """.

    Assigns subject's runs to train, validation and test sets
    """
    sub_h5 = h5py.File(
        f"{data_config.bold_dir}/{data_config.subject_id}/func/"
        f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
        f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
        "r",
    )

    # Season 3 held out for test set
    test_set = []
    for ses in sub_h5:
        test_set += [
            x for x in sub_h5[ses] if x.split("-")[-1][:3] == data_config.test_season
        ]

    # Remaining runs assigned to train and validation sets
    r = np.random.RandomState(
        data_config.random_state,
    )  # select season for validation set

    if data_config.subject_id == "sub-04":
        val_season = r.choice(["s01", "s02", "s04"], 1)[0]
    else:
        val_season = r.choice(["s01", "s02", "s04", "s05", "s06"], 1)[0]
    val_set = []
    for ses in sub_h5:
        val_set += [x for x in sub_h5[ses] if x.split("-")[-1][:3] == val_season]
    train_set = []
    for ses in sub_h5:
        train_set += [
            x
            for x in sub_h5[ses]
            if x.split("-")[-1][:3] not in [data_config.test_season, val_season]
        ]
    train_set = sorted(train_set)

    sub_h5.close()

    # Assign consecutive train set episodes to cross-validation groups
    lts = len(train_set)
    train_groups = (
        np.floor(np.arange(lts) / (lts / data_config.n_splits)).astype(int).tolist()
    )

    return train_groups, train_set, val_set, test_set, val_season


def build_target(
    data_config,
    runs: list,
    run_groups: list = None,
) -> tuple:
    """.

    Concatenates BOLD timeseries into target array.
    """
    y_list = []
    length_list = []
    y_groups = []
    sub_h5 = h5py.File(
        f"{data_config.bold_dir}/{data_config.subject_id}/func/"
        f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
        f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
        "r",
    )

    for i, run in enumerate(runs):
        ses = run.split("_")[0]
        run_ts = np.array(sub_h5[ses][run])
        length_list.append(run_ts.shape[0])
        y_list.append(run_ts)

        if run_groups is not None:
            y_groups.append(np.repeat(run_groups[i], run_ts.shape[0]))

    sub_h5.close()

    y_list = np.concatenate(y_list, axis=0)
    y_groups = (
        np.concatenate(y_groups, axis=0) if run_groups is not None else np.array([])
    )

    return y_list, length_list, y_groups


def extract_one_hot_regressor(data_config, runs, run_length):
    """."""
    x_list = []
    for run_indx, run in enumerate(runs):
        parts = run.split("_")
        task = parts[1].split("_")
        episode_name = task[0].split("_")
        episode = episode_name[0].split("_")[-1].split(".")[0][5:15]
        season = episode_name[0].split("_")[-1].split(".")[0][5:8]

        gentle = read_tsv(f"{data_config.tr_tsv_path}/{season}/friends_{episode}.tsv")
        regressor_vector = generate_one_hot_vector(gentle)

        frame_times = np.arange(run_length[run_indx]) * data_config.TR

        computed_regressor, _ = compute_regressor(
            exp_condition=regressor_vector,
            hrf_model=data_config.hrf_model,
            frame_times=frame_times,
            con_id="word",
        )

        x_list.append(computed_regressor)
    return np.concatenate(x_list, axis=0)


def get_embedding(
    data_config,
    season: str,
    episode: str,
) -> np.array:
    """."""
    h5_path = (
        Path(data_config.stimuli_dir)
        / f"friends_{season}_layer_{data_config.target_layer - 1}_embeddings.h5"
    )
    # print(h5_path)
    with h5py.File(h5_path, "r") as file:
        embedding = np.array(file[episode])

    return embedding


def extract_embedding(data_config, runs):
    """."""
    embedding = []
    embedding_lengths = []
    for run in tqdm(runs, desc="runs", total=len(runs)):
        parts = run.split("_")
        task = parts[1].split("_")
        episode_name = task[0].split("_")
        episode = episode_name[0].split("_")[-1].split(".")[0][5:15]
        season = episode_name[0].split("_")[-1].split(".")[0][5:8]

        emb = get_embedding(data_config, season, episode)
        embedding.append(emb)
        embedding_lengths.append(emb.shape[0])
    features = np.concatenate(embedding, axis=0)
    # print(f"length of features: {len(features)}")
    return features, embedding_lengths

def scale_embeddings(features,embedding_lengths, scaling = None, scaler=None):
    """
    Scales feature embeddings according to the specified scaling method.

    Parameters:
        features (np.array): The input features to scale.
        embedding_lengths (list): List of embedding lengths to split the features after scaling.
        scaling (str, optional): The type of scaling to apply. Options are:
            None - apply no scaling.
            'standard' - apply standardization (z-score normalization).
            'scaler' - apply scaling using a provided scaler or a new StandardScaler.
        scaler (StandardScaler, optional): An instance of a scaler to use if scaling='scaler'. If None and scaling='scaler', a new StandardScaler will be fit.

    Returns:
        tuple: A tuple containing the list of scaled embeddings and the scaler used (if any).
    """
    if scaling is None:
        features_scaled = features
    elif scaling == "standard":
        features_scaled = np.nan_to_num(
            stats.zscore(
                features,
                nan_policy="omit",
                axis=0,
            )
        ).astype("float32")
    elif scaling == "scaler":
        if scaler is None:
            scaler =  StandardScaler().fit(features)
        features_scaled =scaler.transform(features).astype("float32")
    else:
        raise ValueError(f"Unknown scaling type: {scaling}")


    embeddings = np.split(features_scaled, np.cumsum(embedding_lengths[:-1]))

    return embeddings, scaler

def process_embeddings(data_config, runs, scaler=None):
    """Extract and scale embeddings for given runs."""
    features, lengths = extract_embedding(data_config, runs)

    embeddings, scaler = scale_embeddings(features, lengths, scaler, )
    return embeddings, scaler

def build_embedding_regressor(data_config, season, episode, embedding):
    gentle = read_tsv(f"{data_config.tr_tsv_path}/{season}/friends_{episode}.tsv")
    embedding_regressor = []
    for i in range(embedding.shape[1]):
        embedding_regressor.append(
            np.stack(
                (
                    gentle["onset"].values,
                    gentle["duration"].values,
                    embedding[:, i],
                )
            )
        )
    return embedding_regressor

def extract_feature_regressor(data_config,embedding_list, runs, run_length):
    """."""
    x_list = []
    for run_indx, run in tqdm(enumerate(runs), desc="runs", total=len(runs)):
        parts = run.split("_")
        task = parts[1].split("_")
        episode_name = task[0].split("_")
        episode = episode_name[0].split("_")[-1].split(".")[0][5:15]
        season = episode_name[0].split("_")[-1].split(".")[0][5:8]

        regressor_list = build_embedding_regressor(
            data_config, season, episode, embedding_list[run_indx]
        )

        frame_times = np.arange(run_length[run_indx]) * data_config.TR

        computed_regressor = Parallel(n_jobs=-1)(
            delayed(compute_regressor)(
                exp_condition=regressor,
                hrf_model=data_config.hrf_model,
                frame_times=frame_times,
                con_id="word",
            )
            for regressor in regressor_list
        )
        computed_regressor = [c[0] for c in computed_regressor]
        computed_regressor = np.concatenate(computed_regressor, axis=1)

        if len(computed_regressor) != run_length[run_indx]:
            print("ERROR!")
            print(f"computed_regressor: {len(computed_regressor)}")
            print(f"scan: length: {run_length[run_indx]}")

        x_list.append(computed_regressor)

    return np.concatenate(x_list, axis=0)