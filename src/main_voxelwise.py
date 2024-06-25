"""."""

from encoder import test_ridgeReg, train_ridgeReg
from encoder_dataclass import DataBaseConfig
from ridge_tools import cross_val_ridge
from utils import (
    build_target,
    build_val_target,
    extract_feature_regressor,
    process_embeddings,
    split_data_per_training_season,
)
from sklearn.model_selection import GroupKFold, KFold
import numpy as np

data_config = DataBaseConfig()

seasons = ["s01", "s02", "s03", "s04", "s05", "s06"]
seasons.remove(data_config.test_season)
print(f"Running encoding for: {data_config.subject_id}")
for training_season in seasons:


    print(f"Running training on season: {training_season}")
    train_groups, train_runs, val_runs, test_runs, val_season = (
        split_data_per_training_season(data_config, training_season)
    )

    print("Building targets!")

    y_train, length_train, train_groups = build_target(
        data_config,
        train_runs,
        train_groups,
    )


    print("Running validation!")

    for layer_indx in range(1, data_config.target_layer):
        print(f"The layer of embedding is: {layer_indx}")

        train_embeddings, scaler = process_embeddings(
            data_config, train_runs, layer_indx
        )
        x_train = extract_feature_regressor(
            data_config, train_embeddings, train_runs, length_train
        )

        # model = train_ridgeReg(
        #     x_train,
        #     y_train,
        #     train_groups,
        #     data_config,
        # )

        weights, picked_lambdas, model = cross_val_ridge(
            x_train, y_train, train_groups, data_config
        )
        picked_lambdas_np = np.array(picked_lambdas)
        alphas = f"{data_config.output_dir}/{data_config.subject_id}/{data_config.experiment}/{training_season}/ridge_alpha_values_layer_{layer_indx}.npy"
        np.save(alphas, picked_lambdas_np)

        print("Estimating the training prediction!")
        test_ridgeReg(
            data_config,
            model,
            x_train,
            y_train,
            layer_indx,
            training_season,
            stage = "train",
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

            test_ridgeReg(
                data_config,
                model,
                x_val,
                y_val,
                layer_indx,
                training_season,
                episode,
                stage = "val",
                )

            print("Saved the validation prediction map!")

