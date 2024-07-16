"""."""

# from utils import get_season_average_images
from encoder import (
    test_ridgeReg_parcelwise,
    train_ridgeReg_noCV,
)
from encoder_dataclass import DataBaseConfig
from utils import (
    build_val_target,
    extract_feature_regressor,
    process_embeddings,
    split_within_dataset_friends,
    build_within_target,
    discrete_windows,
    find_best_alpha,
)

import time

# Record the start time
start_time = time.time()



data_config = DataBaseConfig()
print(f"Running {data_config.encoding_level} analysis!")

print(f"Running encoding for: {data_config.subject_id}")
layer_indx = 9
movies = ["figures", "wolf", "figures", "bourne"]

    
if data_config.subject_id == "sub-04":
    friends = ["s01", "s02", "s04",]
else:
    friends = ["s01", "s02", "s04", "s05", "s06"]




datasets = friends + movies

for dataset in datasets:

    print(f"Running training and testing for {data_config.subject_id} on data: {season}")
    train_runs, val_runs, test_runs, test_data = split_within_dataset_friends(
        data_config, season
    )    
   
    # Example usage
    best_alpha = find_best_alpha(data_config, windowed_data, val_runs)
    print(f"The best alpha is: {best_alpha}")


    print(len(windowed_data))
    for window in range(len(windowed_data)):
        print("Runing the ridge training on the window {window}!")
        train_runs = windowed_data[window]

        print(f"train_runsL{train_runs}")

        print("Building targets!")
        y_train, length_train = build_within_target(
            data_config,
            train_runs,
        )

        print(f"y_train: {y_train}")

        # for layer_indx in range(data_config.embedding_layer_start,data_config.embedding_layer_start+1): #data_config.target_layer):
        print(f"The layer of embedding is: {data_config.embedding_layer}")
        print("Processing embeddings!")

        train_embeddings, scaler = process_embeddings(
            data_config, train_runs, data_config.embedding_layer
        )

        print("Extract feature regressors!")
        x_train = extract_feature_regressor(
            data_config, train_embeddings, train_runs, length_train
        )

        print(f"x_train.shape: {x_train.shape}")

        if data_config.encoding_level == "parcelwise":
            print("Running parcelwise!")

            model = train_ridgeReg_noCV(
                x_train,
                y_train,
                alpha=best_alpha,
            )

            print("Estimating the training prediction!")
            test_ridgeReg_parcelwise(
                data_config,
                model,
                x_train,
                y_train,
                data_config.embedding_layer,
                season,
                window,
                episode=None,
            )

            print("Saved the training prediction map!")

        for val_run in val_runs:

            parts = val_run.split("_")
            task = parts[1].split("_")
            episode_name = task[0].split("_")
            episode = episode_name[0].split("_")[-1].split(".")[0][5:15]
            season = episode_name[0].split("_")[-1].split(".")[0][5:8]
            print(f"Processed validation run for training on {season}: {episode} ")

            val_run_list = [val_run]
            y_val, length_val = build_val_target(data_config, val_run_list)

            # print(f"Processing the layer {layer_indx}")

            val_embeddings, _ = process_embeddings(
                data_config, val_run_list, data_config.embedding_layer, scaler
            )

            x_val = extract_feature_regressor(
                data_config, val_embeddings, val_run_list, length_val
            )

            print("Estimating the validation prediction!")

            if data_config.encoding_level == "parcelwise":
                test_ridgeReg_parcelwise(
                    data_config,
                    model,
                    x_val,
                    y_val,
                    data_config.embedding_layer,
                    season,
                    window,
                    episode=episode,
                )

            print(f"Saved the validation prediction map for episode {episode}!")

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.2f} seconds")
