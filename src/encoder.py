from dataclasses import dataclass
from pathlib import Path
import pickle

import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from scipy.stats import zscore
from numpy.linalg import inv, svd
from scipy.stats import zscore
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from nilearn.image import load_img


def train_ridgeReg_noCV(
    X: np.array,
    y: np.array,
    alpha: float,
) -> Ridge:
    """Trains ridge regression on the given data.

    Args:
        X: Features
        y: Bold data
        alpha: Regularization strength

    Return:
        Model that is fit with the training data.
    """
    # Initialize the Ridge regression model with the specified alpha
    model = Ridge(alpha=alpha, fit_intercept=True)

    # Fit the model to the data
    return model.fit(X, y)

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


def export_images_within(
    data_config,
    results: dict,
    layer_indx: int,
    train_season: str,
    train_window: int,
    episode,
    
) -> None:
    """.

    Exports RR parcelwise scores as nifti files with
    subject-specific atlas used to extract timeseries.
    """

    if data_config.encoding_level == "parcelwise":
        atlas_path = Path(
            f"{data_config.parcelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_atlas-{data_config.atlas}_"
            f"desc-{data_config.parcel}_dseg.nii.gz",
        )
        masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        standardize=False,
        )
        masker.fit()
    elif data_config.encoding_level == "voxelwise":

        mask_path = Path(
            f"{data_config.voxelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-T1w_atlas-Freesurfer_label-GM_res-func_mask.nii.gz") 
        
        # atlas_img = load_img(mask_path)

        masker = NiftiMasker(mask_img=mask_path, standardize=False)
  
        masker.fit()

    # map Pearson correlations onto brain parcels

    nii_file = masker.inverse_transform(
        np.array(results["R2"]),
    )
    if episode == None:

        nib.save(
            nii_file,
            f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}//{train_season}/{data_config.subject_id}_RidgeReg_R2_train_{data_config.base_model_name}_layer_{layer_indx}_window_{train_window}.nii.gz",
        )

    else:
        nib.save(
            nii_file,
            f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}//{train_season}/{data_config.subject_id}_{episode}_RidgeReg_R2_val_{data_config.base_model_name}_layer_{layer_indx}_window_{train_window}.nii.gz",
        )

    return


def test_within_ridgeReg_parcelwise(
    data_config,
    R,
    x_data,
    y_data,
    layer_indx,
    train_seasons,
    train_window,
    episode=None,
) -> None:
    """.

    Exports RR results in .json file.
    """
    res_dict = {}
    res_dict["correlation"] = {}
    res_dict["R2"] = {}

    # Global R2 score
    res_dict["correlation"] = R.score(x_data, y_data)

    # Parcel-wise predictions
    pred = R.predict(x_data)
    res_dict["R2"] = (pearson_corr(y_data.T, pred.T) ** 2).tolist()

    # export parcelwise scores as .nii images for visualization

    export_images_within(
        data_config,
        res_dict,
        layer_indx,
        train_seasons,
        train_window,
        episode,
    )



def export_images(
    data_config,
    results: dict,
    layer_indx: int,
    train_season: str,
    dataset_name:str,
    segment,
    set,
    
) -> None:
    """.

    Exports RR parcelwise scores as nifti files with
    subject-specific atlas used to extract timeseries.
    """

    if data_config.encoding_level == "parcelwise":
        if dataset_name=="friends":
            atlas_path = Path(
            f"{data_config.parcelwise_bold_dir}/{dataset_name}/parcelwise/{dataset_name}.timeseries/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-{dataset_name}_space-MNI152NLin2009cAsym_atlas-{data_config.atlas}_"
            f"desc-{data_config.parcel}_dseg.nii.gz",
        )
        else: 
            atlas_path = Path(
            f"{data_config.parcelwise_bold_dir}/{dataset_name}/parcelwise/{dataset_name}.timeseries/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-{dataset_name}_space-MNI152NLin2009cAsym_atlas-{data_config.atlas}_"
            f"desc-{data_config.parcel}_dseg.nii.gz")
       
       
        masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        standardize=False,
        )
        masker.fit()
    elif data_config.encoding_level == "voxelwise":

        mask_path = Path(
            f"{data_config.voxelwise_bold_dir}/{data_config.subject_id}/func/"
            f"{data_config.subject_id}_task-friends_space-T1w_atlas-Freesurfer_label-GM_res-func_mask.nii.gz") 
        
        # atlas_img = load_img(mask_path)

        masker = NiftiMasker(mask_img=mask_path, standardize=False)
  
        masker.fit()

    # map Pearson correlations onto brain parcels

    nii_file = masker.inverse_transform(
        np.array(results["R2"]),
    )
    if segment == None:

        nib.save(
            nii_file,
            f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}//{train_season}/{data_config.subject_id}_RidgeReg_R2_train_{data_config.base_model_name}_layer_{layer_indx}.nii.gz",
        )

    else:
        nib.save(
            nii_file,
            f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}//{set}/{train_season}/{data_config.subject_id}_{segment}_RidgeReg_R2_val_{data_config.base_model_name}_layer_{layer_indx}.nii.gz",
        )

    return


def test_ridgeReg_parcelwise(
    data_config,
    R,
    x_data,
    y_data,
    set,
    layer_indx,
    train_seasons,
    dataset_name,
    segment=None,
) -> None:
    """.

    Exports RR results in .json file.
    """
    res_dict = {}
    res_dict["correlation"] = {}
    res_dict["R2"] = {}

    # Global R2 score
    res_dict["correlation"] = R.score(x_data, y_data)

    # Parcel-wise predictions
    pred = R.predict(x_data)
    res_dict["R2"] = (pearson_corr(y_data.T, pred.T) ** 2).tolist()

    # export parcelwise scores as .nii images for visualization

    export_images(
        data_config,
        res_dict,
        layer_indx,
        train_seasons,
        dataset_name,
        segment,
        set,
    )



def R2(Pred, Real):
    """Compute coefficient of determination (R^2)."""
    SSres = np.mean((Real - Pred) ** 2, 0)
    SStot = np.var(Real, 0)
    return np.nan_to_num(1 - SSres / SStot)


def corr(X, Y, axis=0):
    """Compute correlation coefficient."""
    return np.mean(zscore(X) * zscore(Y), axis)


def test_ridgeReg_voxelwise(
    data_config,
    weights,
    x_data,
    y_data,
    layer_indx,
    train_season,
    dataset_name,
    episode: None,
) -> None:
    """.

    Exports RR results in .json file.
    """
    res_dict = {}
    res_dict["correlation"] = {}
    res_dict["R2"] = {}
    # Global R2 score
    print(f"x_data.shape: {x_data.shape}")
    print(f"y_data.shape: {y_data.shape}")

    preds = np.dot(x_data, weights)
    res_dict["correlation"] = corr(preds, y_data)
    res_dict["R2"] = R2(preds, y_data)
    # res_dict["R2"] = (
    #     pearson_corr(y_data.T, preds.T) ** 2
    # ).tolist()

    Path(f"{data_config.output_dir}").mkdir(parents=True, exist_ok=True)

    # export parcelwise scores as .nii images for visualization
    export_images(
        data_config,
        res_dict,
        layer_indx,
        train_season,
        dataset_name,
        episode,
    )
