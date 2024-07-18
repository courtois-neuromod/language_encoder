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
    split_across_dataset,
    build_within_target,
    discrete_windows,
    # find_best_alpha,
    find_best_alpha_across_data,
    build_across_target,
    get_hrf_convolved_embeddings_accross_dataset,
)

import time
import numpy as np
# Record the start time
start_time = time.time()



data_config = DataBaseConfig()
print(f"Running {data_config.encoding_level} analysis!")

print(f"Running encoding for: {data_config.subject_id}")
layer_indx = 9
movie10= ["figures", "wolf", "life", "bourne"]

    
if data_config.subject_id == "sub-04":
    friends = ["s01", "s02", "s04",]
else:
    friends = ["s01", "s02", "s04", "s05", "s06"]


for training_season in friends[:1]:

    print(f"Running training and testing for {data_config.subject_id} on data: {training_season}")
    train_friends_set, val_friends_set, train_movie10_set, val_movie10_set, train_movie10_data_list, val_movie10_data_list= split_across_dataset(
        data_config, training_season, friends, movie10) 

    # print(f"val_friends_set now:{val_friends_set}")   
    # print(f"val_movie10_data_list now:{val_movie10_data_list}")   

    # print(f"train_movie10_set:{train_movie10_set}")   
    # print(f"val_movie10_set:{val_movie10_set}")   


    # print(f"train_friends_set:{len(train_friends_set)}")   

    # Example usage
    best_alpha = find_best_alpha_across_data(data_config, training_season, train_friends_set, val_friends_set, train_movie10_set, val_movie10_set, train_movie10_data_list, val_movie10_data_list)
    print(f"The best alpha is: {best_alpha}")



    # print("Building targets!")
    # #concat the training data from different datasets
    # y_train, length_train = build_across_target(
    #     data_config,
    #     train_friends_set,
    #     train_movie10_set,
    # )
    # x_train = get_hrf_convolved_embeddings_accross_dataset(data_config, train_friends_set, train_movie10_set)
   

    # model = train_ridgeReg_noCV(
    #             x_train,
    #             y_train,
    #             alpha=best_alpha,
    #         )

    # print(f"the model is estimated!")   


    # print(f"friends train dataset number of file:{len(train_friends_set)}")   
    # print(f"movie10 train dataset number of file: {len(train_movie10_set)}")
    # print(f"y_train:{len(length_train)}")   


    # # for layer_indx in range(data_config.embedding_layer_start,data_config.embedding_layer_start+1): #data_config.target_layer):
    # print(f"The layer of embedding is: {data_config.embedding_layer}")
    # print("Processing embeddings!")





####################
#     if data_config.encoding_level == "parcelwise":
#         print("Running parcelwise!")

#        
#         for val_run in val_friends_set:

#             parts = val_run.split("_")
#             task = parts[1].split("_")
#             segment_name = task[0].split("_")
#             segment =segment_name[0].split("_")[-1].split(".")[0][5:15]
#             print(f"Processed validation run for {segment} ")

#             val_run_list = [val_run]

#             y_val, length_val = build_val_target(data_config, val_run_list, "friends")


#             # print(f"Processing the layer {layer_indx}")

#             val_embeddings, _ = process_embeddings(
#                 data_config, val_run_list, data_config.embedding_layer, scaler
#             )

#             x_val = extract_feature_regressor(
#                 data_config, val_embeddings, val_run_list, length_val
#             )

#             print("Estimating the validation prediction!")

#             if data_config.encoding_level == "parcelwise":
#                 test_ridgeReg_parcelwise(
#                     data_config,
#                     model,
#                     x_val,
#                     y_val,
#                     data_config.embedding_layer,
#                     training_season,
#                     dataset_name = "friends",
#                     segment=segment,
#                 )

#             print(f"Saved the validation prediction map for {segment}!")

#         for val_run in val_movie10_set:

#             parts = val_run.split("_")
#             task = parts[1].split("_")
#             segment_name = task[0].split("_")
#             segment = segment_name[0].split("_")[-1].split(".")[0][5:15]
#             dataset_name = segment_name[0].split("_")[-1].split(".")[0][5:8]
#             print(f"Processed validation run for {segment} ")

#             val_run_list = [val_run]

#             y_val, length_val = build_val_target(data_config, val_run_list, "friends")


#             # print(f"Processing the layer {layer_indx}")

#             val_embeddings, _ = process_embeddings(
#                 data_config, val_run_list, data_config.embedding_layer, scaler
#             )

#             x_val = extract_feature_regressor(
#                 data_config, val_embeddings, val_run_list, length_val
#             )

#             print("Estimating the validation prediction!")

#             if data_config.encoding_level == "parcelwise":
#                 test_ridgeReg_parcelwise(
#                     data_config,
#                     model,
#                     x_val,
#                     y_val,
#                     data_config.embedding_layer,
#                     training_season,
#                     dataset_name,
#                     segment=segment,
#                 )

#             print(f"Saved the validation prediction map for segment {segment}!")


# # Record the end time
# end_time = time.time()

# # Calculate the elapsed time
# elapsed_time = end_time - start_time

# print(f"Elapsed time: {elapsed_time:.2f} seconds")
