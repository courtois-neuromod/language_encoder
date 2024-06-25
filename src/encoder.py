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
    """Trains ridge regression folding over the given groups.

    Args:
        X: Features
        y: Bold data
        groups: The data assigned to consequtive groups

    Return:
        Model that is fit with the training data.
    """
    alphas = np.logspace(0.1, 3, 10)

    # alphas = np.logspace(0.1, 6, 10)
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

    Args:
        target: Original data
        predicted: Output of the model predicton
        use_distane: True if to use cosine similarity

    Returns:
        Pairwise correlation score
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
    layer_indx: int,
    train_season: str,
    stage: str,
    episode: None,
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

    nii_file = atlas_masker.inverse_transform(
        np.array(results["parcelwise"][f"{stage}_R2"]),
    )
    if episode == None:

        nib.save(
            nii_file,
            f"{data_config.output_dir}/{data_config.subject_id}/{data_config.experiment}//{train_season}/{data_config.subject_id}_{data_config.atlas}_{data_config.parcel}_RidgeReg_R2_{s}_{data_config.base_model_name}_layer_{layer_indx}.nii.gz",
        )

    else:
         nib.save(
            nii_file,
            f"{data_config.output_dir}/{data_config.subject_id}/{data_config.experiment}//{train_season}/{data_config.subject_id}_{episode}_{data_config.atlas}_{data_config.parcel}_RidgeReg_R2_{s}_{data_config.base_model_name}_layer_{layer_indx}.nii.gz",
        )

    return


def test_ridgeReg(
    data_config,
    R,
    x_data,
    y_data,
    layer_indx,
    train_season,
    episode: None,
    stage: str,
) -> None:
    """.

    Exports RR results in .json file.
    """
    res_dict = {}
    res_dict["parcelwise"] = {}
    # Global R2 score
    res_dict[f"{stage}_R2"] = R.score(x_data, y_data)
    pred = R.predict(x_data)
    res_dict["parcelwise"][f"{stage}_R2"] = (
    pearson_corr(y_data.T, pred.T) ** 2
    ).tolist()

    Path(f"{data_config.output_dir}").mkdir(parents=True, exist_ok=True)


    # export parcelwise scores as .nii images for visualization
    if data_config.bold_dir is not None:
        export_images(
            data_config,
            res_dict,
            layer_indx,
            train_season,
            episode,
            stage
        )
