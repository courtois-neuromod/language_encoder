"""."""

import numpy as np
from pathlib import Path
from nilearn.image import load_img
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker

# from utils import get_season_average_images
from encoder import train_ridgeReg, test_ridgeReg_voxelwise, test_ridgeReg_parcelwise
from encoder_dataclass import DataBaseConfig
from ridge_tools import cross_val_ridge
from utils import (build_target, build_val_target, extract_feature_regressor,
                   process_embeddings, split_data_per_training_season)
data_config = DataBaseConfig()
print(f"Running {data_config.encoding_level} analysis!")

seasons = ["s01", "s02", "s03", "s04", "s05", "s06"]
seasons.remove(data_config.test_season)
print(f"Running encoding for: {data_config.subject_id}")

# atlas_path = Path(
#     f"{data_config.voxelwise_bold_dir}/{data_config.subject_id}/func/"
#     f"{data_config.subject_id}_task-friends_space-T1w_atlas-Freesurfer_label-GM_res-func_mask.nii.gz") 
# atlas_img = load_img(atlas_path)

# atlas_masker = NiftiMasker(
# labels_img=atlas_img,
# standardize=False,
# )

# atlas_masker.fit()


for training_season in seasons[:1]:

    print(f"Running training on season: {training_season}")
    train_segment_groups, train_runs, val_runs, test_runs, val_season = split_data_per_training_season(data_config, training_season)
    
    print(f"train_runs: {train_runs}")
    print(f"train_segment_groups:{train_segment_groups}")
    print(val_runs)

    
    print("Building targets!")
    y_train, length_train, train_data_groups = build_target(
        data_config,
        train_runs,
        train_segment_groups,
    )

    print(f"train_data_groups: {train_data_groups}")
    print(f"train_data_groups: {train_data_groups.shape}")

    print(f"y_train: {y_train}")

    for layer_indx in range(1, data_config.target_layer):
        print(f"The layer of embedding is: {layer_indx}")
        print("Processing embeddings!")


        train_embeddings, scaler = process_embeddings(
            data_config, train_runs, layer_indx
        )

        print("Extract feature regressors!")
        x_train = extract_feature_regressor(
            data_config, train_embeddings, train_runs, length_train
        )

        print(f"x_train.shape: {x_train.shape}")

       

        if data_config.encoding_level== "parcelwise":
            print("Running parcelwise!")

            model = train_ridgeReg(
                x_train,
                y_train,
                train_data_groups,
                data_config,
            )
            print("Estimating the training prediction!")
            test_ridgeReg_parcelwise(
                data_config,
                model,
                x_train,
                y_train,
                layer_indx,
                training_season,
                episode = None
            )

            print("Saved the training prediction map!")
        elif data_config.encoding_level== "voxelwise":
            print("Running voxelwise!")
 
            weights, picked_lambdas = cross_val_ridge(
                x_train, y_train, train_data_groups, data_config
            )
            picked_lambdas_np = np.array(picked_lambdas)
            alphas = f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}/{training_season}/ridge_alpha_values_layer_{layer_indx}.npy"
            np.save(alphas, picked_lambdas_np)
            print("Saved the alphas!")
            print("Estimating the training prediction!")

            # fit and save for the training data
            test_ridgeReg_voxelwise(
                data_config,
                weights,
                x_train,
                y_train,
                layer_indx,
                training_season,
                episode = None
            )
            print("Saved the training prediction map!")

        for val_run in val_runs:

            parts = val_run.split("_")
            task = parts[1].split("_")
            episode_name = task[0].split("_")
            episode = episode_name[0].split("_")[-1].split(".")[0][5:15]
            season = episode_name[0].split("_")[-1].split(".")[0][5:8]
            print(
                f"Processed validation run for training on {training_season}: {episode} "
            )

            val_run_list = [val_run]
            y_val, length_val = build_val_target(data_config, val_run_list)

            # print(f"Processing the layer {layer_indx}")

            val_embeddings, _ = process_embeddings(
                data_config, val_run_list, layer_indx, scaler
            )

            x_val = extract_feature_regressor(
                data_config, val_embeddings, val_run_list, length_val
            )

            print("Estimating the validation prediction!")

            if data_config.encoding_level== "parcelwise":
                test_ridgeReg_parcelwise(
                data_config,
                model,
                x_val,
                y_val,
                layer_indx,
                training_season,
                episode = episode)
            elif data_config.encoding_level== "voxelwise":
                # fit and save for the validation data

                test_ridgeReg_voxelwise(
                    data_config,
                    weights,
                    x_val,
                    y_val,
                    layer_indx,
                    training_season,
                    episode = episode,
                    )

            print("Saved the validation prediction map!")
