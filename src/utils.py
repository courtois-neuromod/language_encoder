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


def list_data_filenames(
    idir: str,
) -> list:
    """."""
    data_filename_list = [x.split("/")[-1] for x in sorted(glob.glob(f"{idir}/s0[0-9]"))]

    return data_filename_list


def list_segments(
    idir: str,
    data_filename: str,
    outfile: str = None,
) -> list:
    """.
    Compile data_filename's list of segments to process.
    """
    all_epi = [
        x.split("/")[-1].split(".")[0][8:15]
        for x in sorted(glob.glob(f"{idir}/{data_filename}/friends_s*.tsv"))
    ]
    if Path(outfile).exists():
        data_filename_h5_file = h5py.File(outfile, "r")
        processed_epi = list(data_filename_h5_file.keys())
        data_filename_h5_file.close()
    else:
        processed_epi = []

    segment_list = [epi for epi in all_epi if epi not in processed_epi]

    return segment_list


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

def pick_random_movie_datasets(data_list):
    """."""
    val_data = data_list
    train_data = random.sample(data_list, 3)
    print(f" 3 train_data random movie choice: {train_data}")

    for item in train_data:
        val_data.remove(item)

    if "figures" in val_data:
        val_data.remove("figures")

    if "life" in val_data:
        val_data.remove("life")
    
    if "figures" in train_data:
        val_data = val_data + ["figures2"]

    if "life" in train_data:
        val_data = val_data+ ["life2"]


    return train_data, val_data




def split_across_dataset(
    data_config,
    set, 
    training_season,
    friends,
    movie10,
) -> tuple:
    """Split a friend data_filename into %80 training and
    randomly picked %10 val, and %10 test data.
    """
    val_friends_data = friends
    # train_movie10_data_list, val_movie10_data_list = pick_random_movie_datasets(movie10)

    if set == "wolf":
        
        train_movie10_data_list = ["figures", "life", "wolf"]
        val_movie10_data_list = ["figures2", "life2", "bourne"]

    elif set == "bourne":

        train_movie10_data_list = ["figures", "life","bourne"]
        val_movie10_data_list = ["figures2", "life2","wolf"]




    val_friends_data.remove(training_season)
    print(f"val_movie10_data_list: {val_movie10_data_list}")


    if data_config.encoding_level == "parcelwise":
        sub_h5_friends = h5py.File(
            f"{data_config.parcelwise_bold_dir}/friends/parcelwise/friends.timeseries/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
            f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
            "r",
        )

        sub_h5_movie10 = h5py.File(
        f"{data_config.parcelwise_bold_dir}/movie10/parcelwise/movie10.timeseries/{data_config.subject_id}/func/"
        f"{data_config.subject_id}_task-movie10_space-MNI152NLin2009cAsym_"
        f"atlas-{data_config.atlas}_desc-{data_config.parcel}Cropped_timeseries.h5",
        "r",
        )
    elif data_config.encoding_level == "voxelwise":
        sub_h5_friends = h5py.File(
            f"{data_config.voxelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-T1w_atlas-Freesurfer_label-GM_res-func_timeseries.h5",
            "r",
        )


    train_friends_set = []
    for ses in sub_h5_friends:
        train_friends_set += [
            x for x in sub_h5_friends[ses] if x.split("-")[-1][:3] == training_season
        ]

    val_friends_set = []
    for ses in sub_h5_friends:
        for x in sub_h5_friends[ses]:
            if x.split("-")[-1][:3] in val_friends_data:
                val_friends_set.append(x)


    train_movie10_set = []
    val_movie10_set = []

    for ses in sub_h5_movie10:
        for x in sub_h5_movie10[ses]:
            parts = x.split('_')
            task = ''.join([char for char in parts[1].split('-')[1] if not char.isdigit()])
            run = parts[2]
            # print(parts)
            
            if task in train_movie10_data_list:
                if task in ["figures", "life"]:
                    if run == "run-1":
                        train_movie10_set.append(x)
                    elif run == "run-2":
                        val_movie10_set.append(x)
                elif task in ["wolf"]:
                    train_movie10_set.append(x)
                elif task == "bourne":
                    if "bourne09" not in parts[1].split('-'): # exclude the bourne09 because of the inaudable segment
                        train_movie10_set.append(x)

    sub_h5_friends.close()


    for ses in sub_h5_movie10:
        for x in sub_h5_movie10[ses]:
            parts = x.split('_')
            task = ''.join([char for char in parts[1].split('-')[1] if not char.isdigit()])
            run = parts[2]
            # print(parts)
            
            if task in val_movie10_data_list:
                if task in ["wolf"]:
                    val_movie10_set.append(x)
                elif task == "bourne":
                    if "bourne09" not in parts[1].split('-'): # exclude the bourne09 because of the inaudable segment
                        val_movie10_set.append(x)
    sub_h5_movie10.close()



    return train_friends_set, val_friends_set, train_movie10_set, val_movie10_set, train_movie10_data_list, val_movie10_data_list
         



def split_within_dataset(
    data_config,
    within_data,
) -> tuple:
    """Split a friend data_filename into %80 training and
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

    test_segment = []
    for ses in sub_h5:
        test_segment += [
            x for x in sub_h5[ses] if x.split("-")[-1][:3] == data_config.test_segment
        ]

    within_data_list = [within_data]  # eg. data_filename 1
    within_data_set = []
    for ses in sub_h5:
        for x in sub_h5[ses]:
            if x.split("-")[-1][:3] in within_data_list:
                within_data_set.append(x)

    # print(within_data_set)

    # Assign consecutive data segments to cross-validation groups and split one of them as the test set
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
    return train_data, val_data, test_data, test_segment




def split_data_per_training_data_filename(
    data_config,
    train_data_filename,
) -> tuple:
    """.

    Assigns subject's runs to train, validation and test sets
    - Iterate on data_filenames
    - Run 8 crossfold for training on that data_filename
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

    # data_filename 3 held out for test set
    test_set = []
    for ses in sub_h5:
        test_set += [
            x for x in sub_h5[ses] if x.split("-")[-1][:3] == data_config.test_segment
        ]
    training_list = [train_data_filename]
    if data_config.subject_id == "sub-04":
        val_data_filenames = ["s01", "s02", "s03", "s04"]
        val_data_filenames.remove(training_list[0])
        val_data_filenames.remove(data_config.test_segment)

    else:
        val_data_filenames = ["s01", "s02", "s03", "s04", "s05", "s06"]
        # print(f"val_data_filenames: {val_data_filenames}")
        val_data_filenames.remove(training_list[0])
        val_data_filenames.remove(data_config.test_segment)
        # print(f"val_data_filenames: {val_data_filenames}")

    val_set = []
    for ses in sub_h5:
        for val_data_filename in val_data_filenames:
            val_set += [x for x in sub_h5[ses] if x.split("-")[-1][:3] == val_data_filename]

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

    # Assign consecutive train set segments to cross-validation groups
    lts = len(train_set)
    # print(f"lts: {lts}")
    # print(np.arange(lts) / (lts / data_config.n_splits))
    train_groups = (
        np.floor(np.arange(lts) / (lts / data_config.n_splits)).astype(int).tolist()
    )

    return train_groups, train_set, val_set, test_set, val_data_filenames


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

    # data_filename 3 held out for test set
    test_set = []
    for ses in sub_h5:
        test_set += [
            x for x in sub_h5[ses] if x.split("-")[-1][:3] == data_config.test_segment
        ]

    # Remaining runs assigned to train and validation sets
    r = np.random.RandomState(
        data_config.random_state,
    )  # select data_filename for validation set

    if data_config.subject_id == "sub-04":
        val_data_filename = r.choice(["s01", "s02", "s04"], 1)[0]
    else:
        val_data_filename = r.choice(["s01", "s02", "s04", "s05", "s06"], 1)[0]
    val_set = []
    for ses in sub_h5:
        val_set += [x for x in sub_h5[ses] if x.split("-")[-1][:3] == val_data_filename]
    train_set = []
    for ses in sub_h5:
        train_set += [
            x
            for x in sub_h5[ses]
            if x.split("-")[-1][:3] not in [data_config.test_segment, val_data_filename]
        ]
    train_set = sorted(train_set)

    sub_h5.close()

    # Assign consecutive train set segments to cross-validation groups
    lts = len(train_set)
    train_groups = (
        np.floor(np.arange(lts) / (lts / data_config.n_splits)).astype(int).tolist()
    )

    return train_groups, train_set, val_set, test_set, val_data_filename

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



def build_across_target(
    data_config,
    friends_runs: list,
    movie10_runs: list,
) -> tuple:
    """.

    Concatenates BOLD timeseries into target arrayn for given list of runs.
    """
    y_list = []
    length_list = []

    if data_config.encoding_level == "parcelwise":
        sub_h5_friends = h5py.File(
            f"{data_config.parcelwise_bold_dir}/friends/parcelwise/friends.timeseries/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
            f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
            "r",
        )

        sub_h5_movie10 = h5py.File(
        f"{data_config.parcelwise_bold_dir}/movie10/parcelwise/movie10.timeseries/{data_config.subject_id}/func/"
        f"{data_config.subject_id}_task-movie10_space-MNI152NLin2009cAsym_"
        f"atlas-{data_config.atlas}_desc-{data_config.parcel}Cropped_timeseries.h5",
        "r",
        )

    elif data_config.encoding_level == "voxelwise":
        sub_h5 = h5py.File(
            f"{data_config.voxelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-T1w_atlas-Freesurfer_label-GM_res-func_timeseries.h5",
            "r",
        )
    friends_length_list = []
    # ses-065_task-s06e23a_timeseries
    for i, run in enumerate(friends_runs):
        ses = run.split("_")[0]
        run_ts_friends = np.array(sub_h5_friends[ses][run])
        friends_length_list.append(run_ts_friends.shape[0])
        y_list.append(run_ts_friends)

    movie10_length_list = []
    # ses-008_task-figures05_run-1_timeseries
    for i, run in enumerate(movie10_runs):
        ses = run.split("_")[0]
        run_ts_movie10 = np.array(sub_h5_movie10[ses][run])
        movie10_length_list.append(run_ts_movie10.shape[0])
        y_list.append(run_ts_movie10)

    sub_h5_friends.close()
    sub_h5_movie10.close()

    length_list = [friends_length_list, movie10_length_list]
    y_list = np.concatenate(y_list, axis=0)

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
    dataset: str
) -> tuple:
    """.

    Extract the particiular segment's time series for the validation.
    """
    y_val = []
    length_list = []

    if data_config.encoding_level == "parcelwise":
        if dataset=="friends":
            sub_h5 = h5py.File(
                f"{data_config.parcelwise_bold_dir}/{dataset}/parcelwise/{dataset}.timeseries/{data_config.subject_id}/func/"
                f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
                f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
                "r",
            )
        else: 
            sub_h5 = h5py.File(
            f"{data_config.parcelwise_bold_dir}/{dataset}/parcelwise/{dataset}.timeseries/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-movie10_space-MNI152NLin2009cAsym_"
            f"atlas-{data_config.atlas}_desc-{data_config.parcel}Cropped_timeseries.h5",
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
        segment_name = task[0].split("_")
        segment = segment_name[0].split("_")[-1].split(".")[0][5:15]
        data_filename = segment_name[0].split("_")[-1].split(".")[0][5:8]

        gentle = read_tsv(f"{data_config.tr_tsv_path}/{data_filename}/friends_{segment}.tsv")
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


def get_embedding(data_config, dataset_name: str, data_filename: str, segment: str, layer_indx: int) -> np.array:
    """."""
    if data_config.finetuned:
        file_tag = "finetuned"
    else:
        file_tag = "base"
   
    outfolder = f"{data_config.stimuli_dir}/{data_config.base_model_name}/{file_tag}"

    if dataset_name == "friends":
        h5_path = f"{outfolder}/{dataset_name}/{dataset_name}_{data_filename}_embeddings_{data_config.base_model_name}_{file_tag}.h5"
    elif dataset_name == "movie10":
        h5_path = f"{outfolder}/{data_filename}/{data_filename}_embeddings_{data_config.base_model_name}_{file_tag}.h5"

    with h5py.File(h5_path, "r") as file:
        embedding = np.array(file[segment][f"layer_{layer_indx}"])

    return embedding


def get_layerwise_embedding(
    data_config,
    data_filename: str,
    segment: str,
    layer_indx: int,
) -> np.array:
    """."""
    if data_config.finetuned:
        file_tag = "finetuned"
    else:
        # folder_tag = "base_"
        file_tag = "base"
    outfolder = f"{data_config.stimuli_dir}/{data_config.base_model_name}/{file_tag}/layer_{layer_indx}"
    h5_path = f"{outfolder}/friends_{data_filename}_embeddings_{data_config.base_model_name}_{file_tag}_layer_{layer_indx}.h5"

    with h5py.File(h5_path, "r") as file:
        embedding = np.array(file[segment])

    return embedding


def extract_embedding(data_config, dataset_name, runs, layer_indx):
    """."""
    embedding = []
    embedding_lengths = []
    if dataset_name == "friends":
        for run in tqdm(runs, desc="runs", total=len(runs)):
            parts = run.split("_")
            task = parts[1].split("_")
            segment_name = task[0].split("_")
            segment = segment_name[0].split("_")[-1].split(".")[0][5:15]
            data_filename = segment_name[0].split("_")[-1].split(".")[0][5:8]

            emb = get_embedding(data_config, dataset_name, data_filename, segment, layer_indx)
            # print(f"shape of embedding:{emb.shape}")
            embedding.append(emb)

            embedding_lengths.append(emb.shape[0])

        # features = np.concatenate(embedding, axis=0)

    elif dataset_name == "movie10":
        for run in tqdm(runs, desc="runs", total=len(runs)):
            parts = run.split("_")
            # print(f"parts: {parts}")
            data_filename = ''.join([char for char in parts[1].split('-')[1] if not char.isdigit()])
            # print(f"data_filename: {data_filename}")

            segment = parts[1].split('-')[1]
            # print(f"segment: {segment}")

            emb = get_embedding(data_config, dataset_name, data_filename, segment, layer_indx)

            embedding.append(emb)

            embedding_lengths.append(emb.shape[0])

        # features = np.concatenate(embedding, axis=0)



    return embedding, embedding_lengths


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


def process_embeddings(data_config, data_filename, runs, layer_indx, scaler=None):
    """Extract and scale embeddings for given runs."""
    embeddings, lengths = extract_embedding(data_config, data_filename, runs, layer_indx)

    # embeddings, scaler = scale_embeddings(features, lengths, scaler)
    return embeddings, scaler, lengths


def build_embedding_regressor(data_config, dataset_name, data_filename, segment, embedding):
    if dataset_name == "friends":
        filename = f"friends_{segment}.tsv"
    elif dataset_name == "movie10":
        filename = f"tsv_files/{segment}.tsv"
        
    print(f"embedding shape:{embedding.shape}")
    gentle = read_tsv(f"{data_config.tr_tsv_path}/{data_filename}/{filename}")
    words = gentle["onset"].values
    print(f"gentle shape:{len(words)}")
    if data_filename== "bourne":
        onset_values = np.array(gentle["onset"].values)
        embedding_ = embedding[:len(onset_values), :]
        embedding = embedding_


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


def extract_feature_regressor(data_config, dataset_name, embedding_list, runs, run_length):
    """."""
    x_list = []
  

    for run_indx, run in enumerate(runs):
        print(f"name of run: {run}")
        if dataset_name == "friends":
            parts = run.split("_")
            task = parts[1].split("_")
            segment_name = task[0].split("_")
            segment = segment_name[0].split("_")[-1].split(".")[0][5:15]
            data_filename = segment_name[0].split("_")[-1].split(".")[0][5:8]
        elif dataset_name == "movie10":
            parts = run.split("_")
            # print(f"parts: {parts}")
            data_filename = ''.join([char for char in parts[1].split('-')[1] if not char.isdigit()])
            # print(f"data_filename: {data_filename}")
            segment = parts[1].split('-')[1]
            # print(f"segment: {segment}")

        regressor_list = build_embedding_regressor(
        data_config,dataset_name, data_filename, segment, embedding_list[run_indx]
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

    # return np.concatenate(x_list, axis=0)
    return x_list


def get_hrf_convolved_embeddings_accross_dataset(data_config, train_friends_set, train_movie10_set, length_train):
    train_friends_embeddings, _, friends_lengths = process_embeddings(
            data_config,"friends", train_friends_set, data_config.embedding_layer)
    
    sum_length = 0
    for i in train_friends_embeddings:
        sum_length += len(i)


    # Extract and concatenate movie embeddings
    train_movie_embeddings, _ , movie10_lengths= process_embeddings(
            data_config, "movie10", train_movie10_set, data_config.embedding_layer,
        )
    
    sum_length = 0
    for i in train_movie_embeddings:
        sum_length += len(i)

    x_train_friends_convolved = extract_feature_regressor(
        data_config, "friends", train_friends_embeddings, train_friends_set, length_train[0]
    )

    sum_length = 0
    for i in x_train_friends_convolved:
        sum_length += len(i)


    x_train_movie10_convolved = extract_feature_regressor(
        data_config, "movie10", train_movie_embeddings, train_movie10_set, length_train[1]
    )

    
    sum_length = 0
    for i in x_train_movie10_convolved:
        sum_length += len(i)


    all_convolved_embeddings = [x_train_friends_convolved, x_train_movie10_convolved]
    flattened_convolved_embeddings = [item for sublist in all_convolved_embeddings for item in sublist]


    x_train = np.concatenate(flattened_convolved_embeddings, axis=0)

    return x_train


def get_data_filename_average_images(data_config, train_data_filenames):

    if data_config.subject_id == "sub-04":
        val_segments = ["s01", "s02", "s03", "s04"]
        val_segments.remove(train_data_filenames[0])
        val_segments.remove(data_config.test_segment)

    else:
        val_segments = ["s01", "s02", "s03", "s04", "s05", "s06"]
        val_segments.remove(train_data_filenames[0])
        val_segments.remove(data_config.test_segment)

    print(val_segments)
    for data_filename in val_segments:
        print(f"data_filename: {data_filename}")

        for layer_index in range(1, 2):  # 13):
            print(f"layer_index: {layer_index}")
            all_epi = []

            pattern = f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}/{train_data_filenames[0]}/{data_config.subject_id}_{data_filename}e*_RidgeReg_R2_val_{data_config.base_model_name}_layer_{layer_index}.nii.gz"

            print(f"Constructed pattern: {pattern}")
            matching_files = glob.glob(pattern)
            print(f"Matching files: {matching_files}")
            for filename in matching_files:
                segment_name = os.path.basename(filename).split("_")[
                    1
                ]  # Extract segment name
                if (
                    segment_name[:3] == data_filename
                ):  # Check if first 3 characters match "s01"
                    all_epi.append(filename)  # Append filename to all_epi if it matches

            # print(all_epi)

            # Append matching files to the all_epi list
            mean_img = image.mean_img(all_epi)
            print("Saving the mean image")
            mean_img_name = (
                f"{data_config.subject_id}_{data_filename}_mean_R2_layer_{layer_index}.nii.gz"
            )

            mean_img.to_filename(
                f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}/{train_data_filenames[0]}/mean_img/{mean_img_name}"
            )

def find_best_alpha_within_data(data_config, windowed_data, val_runs):
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
            train_embeddings, scaler, _ = process_embeddings(data_config, "friends", train_runs, data_config.embedding_layer)
            x_train_ = extract_feature_regressor(data_config, "friends", train_embeddings, train_runs, length_train)
            x_train = np.concatenate(x_train_, axis=0)

            print(f"x_train.shape: {x_train.shape}")

            # Train Ridge model
            model = train_ridgeReg_noCV(x_train, y_train, alpha=alpha)

            # Compute validation MSE for each validation run
            for val_run in val_runs:
                parts = val_run.split("_")
                task = parts[1].split("_")
                segment_name = task[0].split("_")
                segment = segment_name[0].split("_")[-1].split(".")[0][5:15]
                data_filename = segment_name[0].split("_")[-1].split(".")[0][5:8]
                print(f"Processed validation run for training on {data_filename}: {segment} ")

                val_run_list = [val_run]
                y_val, length_val = build_val_target(data_config, val_run_list, "friends")
                val_embeddings, _, _= process_embeddings(data_config,"friends",  val_run_list, data_config.embedding_layer)
                x_val_ = extract_feature_regressor(data_config, "friends", val_embeddings, val_run_list, length_val)
                x_val = np.concatenate(x_val_, axis=0)


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




def find_best_alpha_across_data(data_config, training_season, train_friends_set, val_friends_set, train_movie10_set, val_movie10_set, train_movie10_data_list, val_movie10_data_list):
    best_alpha = None

    # Randomly select one training movie datasets to train ridge alongside the training season of Friends
    movie_train_name = random.sample(train_movie10_data_list, 1)

    train_movie_set = [run for run in train_movie10_set if ''.join([char for char in run.split("_")[1].split('-')[1] if not char.isdigit()]) == movie_train_name]


    y_train, length_train = build_across_target(
        data_config,
        train_friends_set,
        train_movie_set,
    )

    x_train = get_hrf_convolved_embeddings_accross_dataset(data_config, train_friends_set, train_movie_set, length_train)


    # Initialize dictionary to store validation MSE for each alpha
    alpha_performance = {alpha: [] for alpha in data_config.alphas}

    # Select random validation data across both set

    # Friends
    friends_val_season = ["s01", "s02", "s04"] if data_config.subject_id == "sub-04" else ["s01", "s02", "s04", "s05", "s06"]
    friends_val_season.remove(training_season)
    friends_val_dataset = random.choice(friends_val_season)
    val_friends_set_alpha = [x for x in val_friends_set if x.split("-")[-1][:3] == friends_val_dataset]

    # Movie10
    movie_val_name = random.choice(val_movie10_data_list)
    val_movie_set_alpha = [run for run in val_movie10_set if ''.join([char for char in run.split("_")[1].split('-')[1] if not char.isdigit()]) == movie_val_name]


    # Iterate over each alpha value
    for alpha in tqdm(data_config.alphas, desc="Fitting Ridge models"):
        print(f"Running the ridge on alpha: {alpha}")

        # Train Ridge model
        model = train_ridgeReg_noCV(x_train, y_train, alpha=alpha)
        
        for val_run in val_friends_set_alpha + val_movie_set_alpha:
            val_type = "friends" if val_run in val_friends_set_alpha else "movie10"
            val_run_list = [val_run]

            y_val, length_val = build_val_target(data_config, val_run_list, val_type)
            val_embeddings, _, _ = process_embeddings(data_config, val_type, val_run_list, data_config.embedding_layer)
            x_val_ = extract_feature_regressor(data_config, val_type, val_embeddings, val_run_list, length_val)
            x_val = np.concatenate(x_val_, axis=0)

            # Predict on validation set and compute MSE
            val_preds = model.predict(x_val)
            val_mse = mean_squared_error(y_val, val_preds)
            print(f"val_mse for {alpha} is {val_mse} for {val_type} val {val_run}")
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
        

def get_within_ridge_average(data_config, train_data_filenames):

    all_epi = []

    folder_to_save= f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}/{train_data_filenames[0]}/mean_img"
    
    create_folder(folder_to_save)

    pattern = f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}/{train_data_filenames[0]}/*_val_*_layer_{data_config.embedding_layer}_*.nii.gz"

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
        f"{data_config.subject_id}_{train_data_filenames[0]}_mean_R2_layer_{data_config.embedding_layer}.nii.gz"
    )

    mean_img.to_filename(f"{folder_to_save}/{mean_img_name}")

    


def get_across_ridge_average(data_config, train_data_filenames):

    spare_datasets =  ["wolf", "bourne"]

    for training in ["wolf", "bourne"]:
        


        root_folder_to_save= f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}/{training}/{train_data_filenames}/"
        folder_to_save = f"{root_folder_to_save}/mean_img"
        create_folder(folder_to_save)
        spare_datasets_copy = [ds for ds in spare_datasets if ds != training]
        # Define dataset to iterate
        dataset_to_iterate = ["figures", "life", spare_datasets_copy[0], "s01", "s02", "s04", "s05", "s06"]
        if train_data_filenames in dataset_to_iterate:
            dataset_to_iterate.remove(train_data_filenames)


        for dataset in dataset_to_iterate:
            all_epi = []
            pattern = f"{root_folder_to_save}/*_{dataset}*_RidgeReg_R2_val_gpt2_layer_{data_config.embedding_layer}.nii.gz"

            
            print(f"Constructed pattern: {pattern}")
            matching_files = glob.glob(pattern)
            print(f"Matching files: {matching_files}")
            
            for filename in matching_files:
                all_epi.append(filename)  # Append filename to all_epi if it matches

            if all_epi:
                mean_img = image.mean_img(all_epi)
                print("Saving the mean image")
                mean_img_name = f"{data_config.subject_id}_{dataset}_mean_R2_layer_{data_config.embedding_layer}.nii.gz"
                mean_img.to_filename(f"{folder_to_save}/{mean_img_name}")