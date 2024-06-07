

from encoder_dataclass import DataBaseConfig
from utils import get_season_average_images, moderate_split_data

data_config = DataBaseConfig()
train_seasons= ["s01"]
train_groups, train_runs, val_runs, test_runs, val_season = moderate_split_data(data_config, train_seasons)


get_season_average_images(data_config, train_seasons)