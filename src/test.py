"""."""

from dataclasses import dataclass

from encoder import test_ridgeReg, train_ridgeReg
from encoder_dataclass import DataBaseConfig
from utils import (build_target, build_val_target, extract_feature_regressor,
                   moderate_split_data, process_embeddings)

data_config = DataBaseConfig()
train_seasons= ["s01"]
train_groups, train_runs, val_runs, test_runs, val_season = moderate_split_data(data_config, train_seasons)


y_train, length_train, train_groups = build_target(
    data_config,
    train_runs,
    train_groups,
)

for layer_indx in range(2,13): #data_config.target_layer):
    print(f"The layer of embedding is: {layer_indx}")

    train_embeddings, scaler = process_embeddings(data_config, train_runs, layer_indx)
        # print(f"layer {layer_indx} embeddings: {train_embeddings[:3]}")
    x_train = extract_feature_regressor(data_config, train_embeddings, train_runs, length_train)
    # print(f"layer {layer_indx} embeddings: {x_train[:2]}")

    model = train_ridgeReg(
            x_train,
            y_train,
            train_groups,
            data_config,
        )

    for val_run in val_runs:

        parts = val_run.split("_")
        task = parts[1].split("_")
        episode_name = task[0].split("_")
        episode = episode_name[0].split("_")[-1].split(".")[0][5:15]
        season = episode_name[0].split("_")[-1].split(".")[0][5:8]
        print(f"Processed validation run for trainng on {train_seasons[0]}: {episode} ")

        val_run_list = [val_run]
        y_val, length_val = build_val_target(data_config, val_run_list)

            # print(f"Processing the layer {layer_indx}")

        val_embeddings, _= process_embeddings(data_config, val_run_list, layer_indx, scaler)


        x_val = extract_feature_regressor(data_config, val_embeddings, val_run_list, length_val)



        test_ridgeReg(
            data_config,
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            layer_indx,
            train_seasons,
            episode,
        )


# get_season_average_images(data_config, train_seasons)
# get_season_average_images(data_config, train_seasons)