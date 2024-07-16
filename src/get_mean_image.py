from encoder_dataclass import DataBaseConfig
from utils import get_season_average_images

data_config = DataBaseConfig()
train_seasons= ["s01"]
# data_config.encoding_level = "parcelwise"
get_season_average_images(data_config, train_seasons)