import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
LENGTH_DICT = {
    "bourne": {
        "01": 403,
        "02": 405,
        "03": 405,
        "04": 405,
        "05": 405,
        "06": 405,
        "07": 405,
        "08": 405,
        "09": 405,
        "10": 380,
    },
    "figures": {
        "01": 402,
        "02": 409,
        "03": 410,
        "04": 409,
        "05": 408,
        "06": 408,
        "07": 408,
        "08": 409,
        "09": 409,
        "10": 409,
        "11": 409,
        "12": 373,
    },
    "life": {
        "01": 406,
        "02": 406,
        "03": 406,
        "04": 406,
        "05": 384,  # 390? 2nd shortest is 390... 384 was cut ~9s short
    },
    "wolf": {
        "01": 406,
        "02": 406,
        "03": 406,
        "04": 406,
        "05": 406,
        "06": 406,
        "07": 406,
        "08": 406,
        "09": 406,
        "10": 406,
        "11": 406,
        "12": 406,
        "13": 406,
        "14": 406,
        "15": 406,
        "16": 406,
        "17": 498,
    },
}

COMP_ARGS = {
    "compression": "gzip",
    "compression_opts": 4,
}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--idir",
        type=str,
        required=True,
        help="absolute path to data directory",
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="two-digit subject number",
    )
    parser.add_argument(
        "--space",
        type=str,
        choices=["MNI152NLin2009cAsym", "T1w"],
        help="EPI space. Choose either MNI152NLin2009cAsym or T1w",
    )
    parser.add_argument(
        "--parcel",
        type=str,
        required=True,
        help="parcellation name, e.g., Schaefer18, Ward",
    )
    parser.add_argument(
        "--desc",
        type=str,
        required=True,
        help="parcellation grain or specifics, e.g., 400Parcels7Networks, 10k",
    )
    return parser.parse_args()


def crop_timeseries(args):
    parcel = args.parcel
    desc = args.desc
    snum = args.subject
    space = args.space

    old_ts = h5py.File(
        f"{args.idir}/{parcel}_{desc}/sub-{snum}/func/"
        f"sub-{snum}_task-movie10_space-{space}_atlas-{parcel}_"
        f"desc-{desc}_timeseries.h5", "r",
    )
    new_ts = h5py.File(
        f"{args.idir}/{parcel}_{desc}/sub-{snum}/func/"
        f"sub-{snum}_task-movie10_space-{space}_atlas-{parcel}_"
        f"desc-{desc}Cropped_timeseries.h5", "w",
    )

    for ses in list(old_ts.keys()):
        runs = list(old_ts[ses].keys())
        group = new_ts.create_group(ses)

        for run_id in runs:

            f_specs = {
                x.split("-")[0]:x.split("-")[1] for x in run_id.split("_") if "-" in x
            }
            cutoff = LENGTH_DICT[f_specs["task"][:-2]][f_specs["task"][-2:]]

            group.create_dataset(
                run_id,
                data=np.array(old_ts[ses][run_id])[:cutoff, :],
                **COMP_ARGS,
            )

    old_ts.close()
    new_ts.close()


if __name__ == "__main__":
    """
    Example
    python trim_movie10_segments.py --idir /home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/outputs/movie10 --subject 02 --space MNI152NLin2009cAsym --parcel Schaefer18 --desc 1000Parcels7Networks
    python trim_movie10_segments.py --idir /home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/outputs/movie10 --subject 02 --space MNI152NLin2009cAsym --parcel Ward --desc 10k
    """

    args = get_arguments()
    crop_timeseries(args)