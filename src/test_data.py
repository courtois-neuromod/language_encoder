import tqdm
from encoder_dataclass import DataBaseConfig
from utils import build_target, get_embedding, split_episodes

data_config = DataBaseConfig()


data_config = DataBaseConfig()


train_groups, train_runs, val_runs, test_runs, val_season = split_episodes(
    data_config,
)


y_train, length_train, train_groups = build_target(
    data_config,
    train_runs,
    train_groups,
)
print(train_runs[1])
run = "ses-001_task-s01e02b_timeseries"
for layer_indx in range(1,6):
    parts = run.split("_")
    task = parts[1].split("_")
    episode_name = task[0].split("_")
    episode = episode_name[0].split("_")[-1].split(".")[0][5:15]
    season = episode_name[0].split("_")[-1].split(".")[0][5:8]

    emb = get_embedding(data_config, season, episode,layer_indx)
    print(f"embeding from layer {layer_indx} is {emb[:3]}")