import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiLabelsMasker
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold


def train_ridgeReg(
    X: np.array,
    y: np.array,
    groups: list,
    data_config,
) -> RidgeCV:
    """."""
    alphas = np.logspace(0.1, 3, 10)
    group_kfold = GroupKFold(n_splits=data_config.n_splits)
    cv = group_kfold.split(X, y, groups)
    model = RidgeCV(
        alphas=alphas,
        fit_intercept=True,
        # normalize=False,
        cv=cv,
    )

    return model.fit(X, y)


def pairwise_acc(
    target: np.array,
    predicted: np.array,
    use_distance: bool = False,
) -> float:
    """.

    Computes Pairwise accuracy
    """
    true_count = 0
    total = 0

    for i in range(0, len(target)):
        for j in range(i + 1, len(target)):
            total += 1

            t1 = target[i]
            t2 = target[j]
            p1 = predicted[i]
            p2 = predicted[j]

            if use_distance:
                if cosine(t1, p1) + cosine(t2, p2) < cosine(t1, p2) + cosine(t2, p1):
                    true_count += 1

            else:
                if (
                    pearsonr(t1, p1)[0] + pearsonr(t2, p2)[0]
                    > pearsonr(t1, p2)[0] + pearsonr(t2, p1)[0]
                ):
                    true_count += 1

    return true / total


def pearson_corr(
    target: np.array,
    predicted: np.array,
) -> np.array:
    """.

    Calculates pearson R between predictions and targets.
    """
    r_vals = []
    for i in range(len(target)):
        r_val, _ = pearsonr(target[i], predicted[i])
        r_vals.append(r_val)

    return np.array(r_vals)


def export_images(
    data_config,
    results: dict,
) -> None:
    """.

    Exports RR parcelwise scores as nifti files with
    subject-specific atlas used to extract timeseries.
    """
    atlas_path = Path(
        f"{data_config.bold_dir}/{data_config.subject_id}/func/"
        f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_atlas-{data_config.atlas}_"
        f"desc-{data_config.parcel}_dseg.nii.gz",
    )
    atlas_masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        standardize=False,
    )
    atlas_masker.fit()

    # map Pearson correlations onto brain parcels
    for s in ["train", "val"]:
        nii_file = atlas_masker.inverse_transform(
            np.array(results["parcelwise"][f"{s}_R2"]),
        )
        nib.save(
            nii_file,
            f"{data_config.output_dir}/{data_config.subject_id}_{data_config.atlas}_{data_config.parcel}_RidgeReg_R2_{s}.nii.gz",
        )

    return


def test_ridgeReg(
    data_config,
    R,
    x_train,
    y_train,
    x_val,
    y_val,
) -> None:
    """.

    Exports RR results in .json file.
    """
    res_dict = {}

    # Global R2 score
    res_dict["train_R2"] = R.score(x_train, y_train)
    res_dict["val_R2"] = R.score(x_val, y_val)

    # Parcel-wise predictions
    pred_train = R.predict(x_train)
    pred_val = R.predict(x_val)

    res_dict["parcelwise"] = {}
    res_dict["parcelwise"]["train_R2"] = (
        pearson_corr(y_train.T, pred_train.T) ** 2
    ).tolist()
    res_dict["parcelwise"]["val_R2"] = (pearson_corr(y_val.T, pred_val.T) ** 2).tolist()

    # export RR results
    Path(f"{data_config.output_dir}").mkdir(parents=True, exist_ok=True)
    with open(
        f"{data_config.output_dir}/{data_config.subject_id}_ridgeReg_{data_config.atlas}_{data_config.parcel}_result.json",
        "w",
    ) as fp:
        json.dump(res_dict, fp)

    # export parcelwise scores as .nii images for visualization
    if data_config.bold_dir is not None:
        export_images(
            data_config,
            res_dict,
        )
