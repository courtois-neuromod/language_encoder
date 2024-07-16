"""``project`` utilities."""

import glob
import os
from pathlib import Path
import random
import os
import h5py
import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from nilearn import image
from nilearn.glm.first_level import compute_regressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.linear_model import Ridge
from encoder import (
    test_ridgeReg_parcelwise,
    train_ridgeReg_noCV,
    pearson_corr
)
from sklearn.metrics import r2_score, mean_squared_error


def list_seasons(
    idir: str,
) -> list:
    """."""
    season_list = [x.split("/")[-1] for x in sorted(glob.glob(f"{idir}/s0[0-9]"))]

    return season_list


def list_episodes(
    idir: str,
    season: str,
    outfile: str = None,
) -> list:
    """.
    Compile season's list of episodes to process.
    """
    all_epi = [
        x.split("/")[-1].split(".")[0][8:15]
        for x in sorted(glob.glob(f"{idir}/{season}/friends_s*.tsv"))
    ]
    if Path(outfile).exists():
        season_h5_file = h5py.File(outfile, "r")
        processed_epi = list(season_h5_file.keys())
        season_h5_file.close()
    else:
        processed_epi = []

    episode_list = [epi for epi in all_epi if epi not in processed_epi]

    return episode_list


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

def pick_random_datasets(data_list_1, data_list_2):
    train_data_1 = random.choice(data_list_1)
    train_data_2 = random.sample(data_list_1, 3)
    train_data = train_data_1 + train_data_2 
    val_data = data_list_1.remove(train_data_1) + data_list_2.remove(train_data_2) 
    
    if "figures" in train_data:
        val_data = val_data+ ["figures2"]

    if "life" in train_data:
        val_data = val_data+ ["life2"]


    return train_data, val_data




def split_within_dataset_friends(
    data_config,
    within_data,
) -> tuple:
    """Split a friend season into %80 training and
    randomly picked %10 val, and %10 test data.
    """

    if data_config.encoding_level == "parcelwise":
        sub_h5 = h5py.File(
            f"{data_config.parcelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
            f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
            "r",
        )
    elif data_config.encoding_level == "voxelwise":
        sub_h5 = h5py.File(
            f"{data_config.voxelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-T1w_atlas-Freesurfer_label-GM_res-func_timeseries.h5",
            "r",
        )

    test_season = []
    for ses in sub_h5:
        test_season += [
            x for x in sub_h5[ses] if x.split("-")[-1][:3] == data_config.test_season
        ]

    within_data_list = [within_data]  # eg. season 1
    within_data_set = []
    for ses in sub_h5:
        for x in sub_h5[ses]:
            if x.split("-")[-1][:3] in within_data_list:
                within_data_set.append(x)

    # print(within_data_set)

    # Assign consecutive data episodes to cross-validation groups and split one of them as the test set
    data_size = len(within_data_set)
    train_size = int(0.8 * data_size)
    val_size = int(0.1 * data_size)
    test_size = int(0.1 * data_size)  # Ensure all data is used

    indices = list(range(data_size))

    # Randomly select indices for validation and test sets
    val_indices = random.sample(indices, val_size)
    test_indices = random.sample(
        [i for i in indices if i not in val_indices], test_size
    )

    # Use the remaining indices for the training set
    train_indices = [
        i for i in indices if i not in val_indices and i not in test_indices
    ]

    # Split the data based on the indices
    train_data = [within_data_set[i] for i in train_indices]
    val_data = [within_data_set[i] for i in val_indices]
    test_data = [within_data_set[i] for i in test_indices]

    sub_h5.close()
    return train_data, val_data, test_data, test_season


def sliding_windows(data_config, data):
    """Generate sliding windows of 1 hour training data
    for within dataset training."""

    # Generate the sliding windows
    windowed_data = [
        data[i : i + data_config.window_size]
        for i in range(len(data) - data_config.window_size + 1)
    ]

    print(windowed_data)
    return windowed_data


def discrete_windows(data_config, data):
    """Generate the consecutive windows."""
    consecutive_windows = [
        data[i : i + data_config.window_size]
        for i in range(0, len(data), data_config.window_size)
    ]
    return consecutive_windows


def split_data_per_training_season(
    data_config,
    train_season,
) -> tuple:
    """.

    Assigns subject's runs to train, validation and test sets
    - Iterate on seasons
    - Run 8 crossfold for training on that season
    -The rest
    """
    # sub_h5 = h5py.File(
    #     f"{data_config.bold_dir}/{data_config.subject_id}/func/"
    #     f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
    #     f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
    #     "r",
    # )

    if data_config.encoding_level == "parcelwise":
        sub_h5 = h5py.File(
            f"{data_config.parcelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
            f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
            "r",
        )
    elif data_config.encoding_level == "voxelwise":
        sub_h5 = h5py.File(
            f"{data_config.voxelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-T1w_atlas-Freesurfer_label-GM_res-func_timeseries.h5",
            "r",
        )

    # Season 3 held out for test set
    test_set = []
    for ses in sub_h5:
        test_set += [
            x for x in sub_h5[ses] if x.split("-")[-1][:3] == data_config.test_season
        ]
    training_list = [train_season]
    if data_config.subject_id == "sub-04":
        val_seasons = ["s01", "s02", "s03", "s04"]
        val_seasons.remove(training_list[0])
        val_seasons.remove(data_config.test_season)

    else:
        val_seasons = ["s01", "s02", "s03", "s04", "s05", "s06"]
        # print(f"val_seasons: {val_seasons}")
        val_seasons.remove(training_list[0])
        val_seasons.remove(data_config.test_season)
        # print(f"val_seasons: {val_seasons}")

    val_set = []
    for ses in sub_h5:
        for val_season in val_seasons:
            val_set += [x for x in sub_h5[ses] if x.split("-")[-1][:3] == val_season]

    # print(f"val_set: {val_set}")

    train_set = []
    for ses in sub_h5:
        for x in sub_h5[ses]:
            if x.split("-")[-1][:3] in training_list:
                train_set.append(x)

    train_set = sorted(train_set)
    # print(f"train_set: {train_set}")

    # print(f"train_set: {len(train_set)}")
    sub_h5.close()

    # Assign consecutive train set episodes to cross-validation groups
    lts = len(train_set)
    # print(f"lts: {lts}")
    # print(np.arange(lts) / (lts / data_config.n_splits))
    train_groups = (
        np.floor(np.arange(lts) / (lts / data_config.n_splits)).astype(int).tolist()
    )

    return train_groups, train_set, val_set, test_set, val_season


def split_data_randomly(
    data_config,
) -> tuple:
    """.

    Assigns subject's runs to train, validation and test sets
    """
    # sub_h5 = h5py.File(
    #     f"{data_config.bold_dir}/{data_config.subject_id}/func/"
    #     f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
    #     f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
    #     "r",
    # )

    sub_h5 = h5py.File(
        f"{data_config.voxelwise_bold_dir}/{data_config.subject_id}/func/"
        f"{data_config.subject_id}_task-friends_space-T1w_atlas-Freesurfer_label-GM_res-func_timeseries.h5",
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

def build_within_target(
    data_config,
    runs: list,
) -> tuple:
    """.

    Concatenates BOLD timeseries into target arrayn for given list of runs.
    """
    y_list = []
    length_list = []

    if data_config.encoding_level == "parcelwise":
        sub_h5 = h5py.File(
            f"{data_config.parcelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
            f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
            "r",
        )

    elif data_config.encoding_level == "voxelwise":
        sub_h5 = h5py.File(
            f"{data_config.voxelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-T1w_atlas-Freesurfer_label-GM_res-func_timeseries.h5",
            "r",
        )

    for i, run in enumerate(runs):
        ses = run.split("_")[0]
        run_ts = np.array(sub_h5[ses][run])
        length_list.append(run_ts.shape[0])
        y_list.append(run_ts)

    sub_h5.close()

    y_list = np.concatenate(y_list, axis=0)

    print("y_list shape:", y_list.shape)
    print("length_list:", length_list)

    return y_list, length_list



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
    # sub_h5 = h5py.File(
    #     f"{data_config.bold_dir}/{data_config.subject_id}/func/"
    #     f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
    #     f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
    #     "r",
    # )

    if data_config.encoding_level == "parcelwise":
        sub_h5 = h5py.File(
            f"{data_config.parcelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
            f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
            "r",
        )

    elif data_config.encoding_level == "voxelwise":
        sub_h5 = h5py.File(
            f"{data_config.voxelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-T1w_atlas-Freesurfer_label-GM_res-func_timeseries.h5",
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


def build_val_target(
    data_config,
    runs: list,
) -> tuple:
    """.

    Extract the particiular segment's time series for the validation.
    """
    y_val = []
    length_list = []

    if data_config.encoding_level == "parcelwise":
        sub_h5 = h5py.File(
            f"{data_config.parcelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
            f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
            "r",
        )
    elif data_config.encoding_level == "voxelwise":
        sub_h5 = h5py.File(
            f"{data_config.voxelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-T1w_atlas-Freesurfer_label-GM_res-func_timeseries.h5",
            "r",
        )

    # Since we run validation on segment level, there is no data to concat, but only a single segment
    ses = runs[0].split("_")[0]
    y_val = np.array(sub_h5[ses][runs[0]])
    length_list.append(y_val.shape[0])

    sub_h5.close()
    return y_val, length_list


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


def get_embedding(data_config, season: str, episode: str, layer_indx: int) -> np.array:
    """."""
    if data_config.finetuned:
        file_tag = "finetuned"
    else:
        # folder_tag = "base_"
        file_tag = "base"
    outfolder = f"{data_config.stimuli_dir}/{data_config.base_model_name}/{file_tag}/"
    h5_path = f"{outfolder}/friends_{season}_embeddings_{data_config.base_model_name}_{file_tag}.h5"

    with h5py.File(h5_path, "r") as file:
        embedding = np.array(file[episode][f"layer_{layer_indx}"])

    return embedding


def get_layerwise_embedding(
    data_config,
    season: str,
    episode: str,
    layer_indx: int,
) -> np.array:
    """."""
    if data_config.finetuned:
        file_tag = "finetuned"
    else:
        # folder_tag = "base_"
        file_tag = "base"
    outfolder = f"{data_config.stimuli_dir}/{data_config.base_model_name}/{file_tag}/layer_{layer_indx}"
    h5_path = f"{outfolder}/friends_{season}_embeddings_{data_config.base_model_name}_{file_tag}_layer_{layer_indx}.h5"

    with h5py.File(h5_path, "r") as file:
        embedding = np.array(file[episode])

    return embedding


def extract_embedding(data_config, runs, layer_indx):
    """."""
    embedding = []
    embedding_lengths = []
    for run in tqdm(runs, desc="runs", total=len(runs)):
        parts = run.split("_")
        task = parts[1].split("_")
        episode_name = task[0].split("_")
        episode = episode_name[0].split("_")[-1].split(".")[0][5:15]
        season = episode_name[0].split("_")[-1].split(".")[0][5:8]

        emb = get_embedding(data_config, season, episode, layer_indx)

        embedding.append(emb)

        embedding_lengths.append(emb.shape[0])

        features = np.concatenate(embedding, axis=0)

    return features, embedding_lengths


def scale_embeddings(features, embedding_lengths, scaling=None, scaler=None):
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
            scaler = StandardScaler().fit(features)
        features_scaled = scaler.transform(features).astype("float32")
    else:
        raise ValueError(f"Unknown scaling type: {scaling}")

    embeddings = np.split(features_scaled, np.cumsum(embedding_lengths[:-1]))

    return embeddings, scaler


def process_embeddings(data_config, runs, layer_indx, scaler=None):
    """Extract and scale embeddings for given runs."""
    features, lengths = extract_embedding(data_config, runs, layer_indx)

    embeddings, scaler = scale_embeddings(features, lengths, scaler)
    return embeddings, scaler


def build_embedding_regressor(data_config, season, episode, embedding):
    gentle = read_tsv(f"{data_config.tr_tsv_path}/{season}/friends_{episode}.tsv")
    embedding_regressor = []
    for i in range(embedding.shape[1]):
        # print(len(gentle["onset"].values))
        # print(len(gentle["duration"].values))
        # print(len(embedding[:, i]))

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


def extract_feature_regressor(data_config, embedding_list, runs, run_length):
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


def get_season_average_images(data_config, train_seasons):

    if data_config.subject_id == "sub-04":
        val_seasons = ["s01", "s02", "s03", "s04"]
        val_seasons.remove(train_seasons[0])
        val_seasons.remove(data_config.test_season)

    else:
        val_seasons = ["s01", "s02", "s03", "s04", "s05", "s06"]
        val_seasons.remove(train_seasons[0])
        val_seasons.remove(data_config.test_season)

    print(val_seasons)
    for season in val_seasons:
        print(f"season: {season}")

        for layer_index in range(1, 2):  # 13):
            print(f"layer_index: {layer_index}")
            all_epi = []

            pattern = f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}/{train_seasons[0]}/{data_config.subject_id}_{season}e*_RidgeReg_R2_val_{data_config.base_model_name}_layer_{layer_index}.nii.gz"

            print(f"Constructed pattern: {pattern}")
            matching_files = glob.glob(pattern)
            print(f"Matching files: {matching_files}")
            for filename in matching_files:
                episode_name = os.path.basename(filename).split("_")[
                    1
                ]  # Extract episode name
                if (
                    episode_name[:3] == season
                ):  # Check if first 3 characters match "s01"
                    all_epi.append(filename)  # Append filename to all_epi if it matches

            # print(all_epi)

            # Append matching files to the all_epi list
            mean_img = image.mean_img(all_epi)
            print("Saving the mean image")
            mean_img_name = (
                f"{data_config.subject_id}_{season}_mean_R2_layer_{layer_index}.nii.gz"
            )

            mean_img.to_filename(
                f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}/{train_seasons[0]}/mean_img/{mean_img_name}"
            )

def find_best_alpha(data_config, windowed_data, val_runs):
    best_alpha = None

    # Randomly select 3 windows from the training data once
    selected_windows = random.sample(windowed_data, 3)

    # Initialize dictionary to store validation MSE for each alpha
    alpha_performance = {alpha: [] for alpha in data_config.alphas}

    # Iterate over each alpha value
    for alpha in tqdm(data_config.alphas, desc="Fitting Ridge models"):
        print(f"Running the ridge on alpha: {alpha}")

        # Iterate over each selected window
        for window in range(len(selected_windows)):
            print(f"Running the ridge training on window: {window + 1}")
            train_runs = selected_windows[window]

            # Build training targets
            y_train, length_train = build_within_target(data_config, train_runs)
            train_embeddings, scaler = process_embeddings(data_config, train_runs, data_config.embedding_layer)
            x_train = extract_feature_regressor(data_config, train_embeddings, train_runs, length_train)

            print(f"x_train.shape: {x_train.shape}")

            # Train Ridge model
            model = train_ridgeReg_noCV(x_train, y_train, alpha=alpha)

            # Compute validation MSE for each validation run
            for val_run in val_runs:
                parts = val_run.split("_")
                task = parts[1].split("_")
                episode_name = task[0].split("_")
                episode = episode_name[0].split("_")[-1].split(".")[0][5:15]
                season = episode_name[0].split("_")[-1].split(".")[0][5:8]
                print(f"Processed validation run for training on {season}: {episode} ")

                val_run_list = [val_run]
                y_val, length_val = build_val_target(data_config, val_run_list)
                val_embeddings, _ = process_embeddings(data_config, val_run_list, data_config.embedding_layer, scaler)
                x_val = extract_feature_regressor(data_config, val_embeddings, val_run_list, length_val)

                # Predict on validation set and compute MSE
                val_preds = model.predict(x_val)
                val_mse = mean_squared_error(y_val, val_preds)
                print(f"val_mse for {alpha} is {val_mse} for training window {window} of val {val_run}")
                alpha_performance[alpha].append(val_mse)

    # Compute the mean validation MSE for each alpha
    mean_alpha_performance = {alpha: np.mean(mse_list) for alpha, mse_list in alpha_performance.items()}

    # Find the alpha with the lowest mean validation MSE
    best_alpha = min(mean_alpha_performance, key=mean_alpha_performance.get)
    print(f"The best alpha is: {best_alpha} with mean validation MSE: {mean_alpha_performance[best_alpha]}")

    return best_alpha


def create_folder(path):
    # Check if the folder exists
    if not os.path.exists(path):
        # Create the folder
        os.makedirs(path)
        print(f"Folder '{path}' created.")
    else:
        print(f"Folder '{path}' already exists.")
        

def get_within_ridge_average(data_config, train_seasons):

    all_epi = []

    folder_to_save= f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}/{train_seasons[0]}/mean_img"
    
    create_folder(folder_to_save)

    pattern = f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}/{train_seasons[0]}/*_val_*_layer_{data_config.embedding_layer}_*.nii.gz"

    print(f"Constructed pattern: {pattern}")
    matching_files = glob.glob(pattern)
    print(f"Matching files: {matching_files}")
    for filename in matching_files:
        all_epi.append(filename)  # Append filename to all_epi if it matches

    # print(all_epi)

    # Append matching files to the all_epi list
    mean_img = image.mean_img(all_epi)
    print("Saving the mean image")
    mean_img_name = (
        f"{data_config.subject_id}_{train_seasons[0]}_mean_R2_layer_{data_config.embedding_layer}.nii.gz"
    )

    mean_img.to_filename(f"{folder_to_save}/{mean_img_name}")

    