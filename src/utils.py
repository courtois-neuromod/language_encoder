"""``project`` utilities."""

import glob
import string
from pathlib import Path
from typing import Any

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

# class Identity(PCA):
#     def __init__(self: "Identity") -> None:
#         """Implement identity operator."""
#         pass

#     def fit(self: "Identity", X: Any, y: Any) -> None:
#         pass

#     def transform(self: "Identity", X: Any) -> Any:
#         return X

#     def fit_transform(self: "Identity", X: Any, y: Any = None) -> Any:
#         self.fit(X, y=y)
#         return self.transform(X)

#     def inverse_transform(self: "Identity", X: Any) -> Any:
#         return X


def get_possible_linear_models() -> list[str]:
    """Fetch possible reduction methods.

    Returns:
        - list
    """
    return ["ridgecv", "glm"]


def get_possible_reduction_methods() -> list[str]:
    """Fetch possible reduction methods.

    Returns:
        - list
    """
    return [None, "pca", "agglomerative_clustering"]


STUDY_PARAMS = {
    "tr": 1.49,
    "max_tokens": 512,
}


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


def set_output(
    season: str,
    output_dir: str,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> tuple:
    """.

    Set compression params and output file name.
    """
    compress_details = ""
    comp_args = {}
    if compression is not None:
        compress_details = f"_{compression}"
        comp_args["compression"] = compression
        if compression == "gzip":
            compress_details += f"_level-{compression_opts}"
            comp_args["compression_opts"] = compression_opts

    out_file = f"{output_dir}/friends_{season}_features_" f"text{compress_details}.h5"

    # Path(f"{args.odir}/temp").mkdir(exist_ok=True, parents=True)

    return comp_args, out_file


def save_features(
    episode: str,
    feature: np.array,
    outfile_name: str,
    comp_args: dict,
) -> None:
    """.

    Save episode's text features into .h5 file.
    """
    flag = "a" if Path(outfile_name).exists() else "w"

    with h5py.File(outfile_name, flag) as f:
        group = f.create_group(episode)

        group.create_dataset(
            "features",
            data=feature,
            **comp_args,
        )


def preprocess_words(tsv_path: str) -> str:
    """Un-punctuate, lower and combine the words like a text.

    Args:
        - tsv_path: path to the episode file
    Returns:
        - list of concatanated words
    """
    data = read_tsv(tsv_path)
    stimuli_data = data["word"].apply(
        lambda x: x.translate(
            str.maketrans("", "", string.punctuation),
        ).lower(),
    )

    return " ".join(stimuli_data)


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


def build_output(
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
        run_ts = np.array(sub_h5[ses][run])[data_config.TR_delay :, :]
        length_list.append(run_ts.shape[0])
        y_list.append(run_ts)

        if run_groups is not None:
            y_groups.append(np.repeat(run_groups[i], run_ts.shape[0]))

    sub_h5.close()
    # y_list = np.concatenate(y_list, axis=0)
    y_groups = (
        np.concatenate(y_groups, axis=0) if run_groups is not None else np.array([])
    )

    return y_list, length_list, y_groups


def build_text(
    data_config,
    seasons: list,
) -> np.array:

    embeddings = []
    if isinstance(seasons, str):
        seasons = [seasons]  # Convert a single season string to a list

    for season in seasons:
        h5_path = (
            Path(data_config.stimuli_dir)
            / f"friends_{season}_layer_{data_config.target_layer - 1}_embeddings.h5"
        )
        # print(h5_path)
        with h5py.File(h5_path, "r") as file:
            for name in file:
                embeddings.append(np.array(file[name]))
    return embeddings


def load_stmuli(path, download=False):
    """Load stimuli data from path.
    Download it if not already done.
    Args:
        - path: str
        - download: bool
        - template:str
    Returns:
        - stimuli_data: list of csv
    """
    if download:
        output = "./data/stimuli_data.zip"
        gdown.download(path, output, quiet=False)
        os.system(f"unzip {output} -d ./data/")

    stimuli_data = sorted(glob.glob(f"{path}/*.tsv"))
    # print(stimuli_data)

    return stimuli_data


def get_groups(gentles):
    """Compute the number of rows in each array
    Args:
        - gentles: list of np.Array
    Returns:
        - groups: list of np.Array
    """
    # We compute the number of rows in each array.
    lengths = [len(f) for f in gentles]
    print(f"lengths: {lengths}")
    start_stop = []
    start = 0
    for l in lengths:
        stop = start + l
        start_stop.append((start, stop))
        start = stop
    groups = [np.arange(start, stop, 1) for (start, stop) in start_stop]
    return groups


def pad_gentle_files(stimuli, pad_rows, delta_time):

    # Create padding for the beginning
    start_padding = pd.DataFrame(
        {
            "word": ["pad"] * pad_rows,
            "onset": [
                stimuli["onset"].iloc[0] - delta_time * (i + 1) for i in range(pad_rows)
            ][::-1],
            "offset": [
                stimuli["offset"].iloc[0] - delta_time * (i + 1)
                for i in range(pad_rows)
            ][::-1],
            "duration": [0]
            * pad_rows,  # assuming duration zero for padding, adjust as needed
        }
    )

    # Create padding for the end
    end_padding = pd.DataFrame(
        {
            "word": ["pad"] * pad_rows,
            "onset": [
                stimuli["onset"].iloc[-1] + delta_time * i
                for i in range(1, pad_rows + 1)
            ],
            "offset": [
                stimuli["offset"].iloc[-1] + delta_time * i
                for i in range(1, pad_rows + 1)
            ],
            "duration": [0]
            * pad_rows,  # assuming duration zero for padding, adjust as needed
        }
    )

    # Concatenate the padding to the original DataFrame
    df_padded = pd.concat([start_padding, stimuli, end_padding]).reset_index(drop=True)
    return df_padded


def preprocess_stimuli_data(stimuli_data, pad_rows, delta_time):
    """Load stimuli data. Preprocess it to lower cases.
    Returns pandas dataframes.
    Args:
        - stimuli_data: list of str
    Returns:
        - stimuli_data: list of np.Array
    """
    stimuli_data_tmp = [pd.read_csv(f, sep="\t") for f in stimuli_data]
    # print(stimuli_data_tmp)

    stimuli_data = []
    # voxels with activation at zero at each time step generate a nan-value pearson correlation => we add a small variation to the first element
    for stimuli in stimuli_data_tmp:
        stimuli["word"] = stimuli["word"].fillna("").astype(str)
        stimuli["word"] = list(map(lambda x: x.lower(), stimuli["word"]))
        stimuli = pad_gentle_files(stimuli, pad_rows, delta_time)
        print(stimuli)

        stimuli_data.append(stimuli)
    return stimuli_data


def build_text_input(data_config, seasons, pad_rows,
delta_time):
    if isinstance(seasons, str):
        seasons = [seasons]  # Convert a single season string to a list
    stimuli = [
        preprocess_stimuli_data(
            load_stmuli(f"{data_config.tsv_path}/{season}", download=False),
            pad_rows,
            delta_time,
        )
        for season in seasons
    ]
    return [
        s["offset"].values for season in range(len(stimuli)) for s in stimuli[season]
    ]


def get_metric(metric_name):
    """Fetch the metric associated with the metric_name.
    Args:
        - metric_name: str
    Returns:
        - metric: built-in function
    """
    metric_dic = {
        "r": r,
        "r_nan": r_nan,
        "r2": lambda x, y: r2_score(x, y, multioutput="raw_values"),
        "r2_nan": r2_nan,
        "mse": lambda x, y: mse(x, y, axis=0),
        "cosine_dist": cosine_dist,
        "mse_dist": mse,
    }
    if metric_name in metric_dic.keys():
        logging.info(f"Loading {metric_name}...")
        return metric_dic[metric_name]
    else:
        logging.info(f"Loading {str(metric_name)} as a custom metric, not a string...")
        return metric_name


def get_linearmodel(name, alpha=1, alpha_min=-3, alpha_max=8, nb_alphas=10):
    """Retrieve the"""
    if name == "ridgecv":
        logging.info(
            f"Loading RidgeCV, with {nb_alphas} alphas varying logarithimicly between {alpha_min} and {alpha_max}..."
        )
        return RidgeCV(
            np.logspace(alpha_min, alpha_max, nb_alphas),
            fit_intercept=True,
            alpha_per_target=True,
            scoring="r2",
        )
    elif name == "glm":
        logging.info(f"Loading LinearRegression...")
        return LinearRegression(fit_intercept=True)
    elif not isinstance(name, str):
        logging.warning(
            "The model seems to be custom.\nUsing it directly for the encoding analysis."
        )
        return name
    else:
        logging.error(
            f"Unrecognized model {name}. Please select among ['ridgecv', 'glm] or a custom encoding model."
        )
