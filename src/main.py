from dataclasses import dataclass

from encoder import test_ridgeReg, train_ridgeReg
from encoder_dataclass import DataBaseConfig
from utils import (build_target, extract_feature_regressor, process_embeddings,
                   split_episodes)

data_config = DataBaseConfig()


train_groups, train_runs, val_runs, test_runs, val_season = split_episodes(
    data_config,
)


y_train, length_train, train_groups = build_target(
    data_config,
    train_runs,
    train_groups,
)

y_val, length_val, _ = build_target(data_config, val_runs)


for layer_indx in range(1,13): #data_config.target_layer):

    # print(f"Processing the layer {layer_indx}")
    train_embeddings, scaler = process_embeddings(data_config, train_runs, layer_indx)
    # print(f"layer {layer_indx} embeddings: {train_embeddings[:3]}")
    x_train = extract_feature_regressor(data_config, train_embeddings, train_runs, length_train)
    # print(f"layer {layer_indx} embeddings: {x_train[:2]}")

    val_embeddings, _= process_embeddings(data_config, val_runs, layer_indx, scaler)

    x_val = extract_feature_regressor(data_config, val_embeddings, val_runs, length_val)

    model = train_ridgeReg(
        x_train,
        y_train,
        train_groups,
        data_config,
    )

    test_ridgeReg(
        data_config,
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        layer_indx,
    )
        x_val,
        y_val,
        layer_indx,
    )
