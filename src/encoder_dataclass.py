from dataclasses import dataclass


@dataclass
class DataBaseConfig:
    """."""

    target_layer: int = 13
    atlas: str = "MIST"
    parcel: str = "444"
    subject_id: str = "sub-03"
    n_splits: int = 10
    random_state: int = 42
    test_season = "s03"  # season allocated for test
    TR: float = 1.49
    experiment = "within_dataset"
    bold_dir: str = "/scratch/ibilgin/friends.timeseries/"
    voxelwise_bold_dir: str = (
        "/scratch/ibilgin/datasets/friends_voxel_level/friends/Freesurfer_GM_res-func/"
    )
    stimuli_dir: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/"
    )
    output_dir: str = "/scratch/ibilgin/datasets/ridge_regression/voxel_wise"
    tr_tsv_path: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/word_alignment/"
    )
    seasons = ["s01", "s02", "s03", "s04", "s05", "s06"]

    hrf_model = "spm"
    finetuned = False
    sweep: str = "overrides#lr~0.0001161#wd~0.001303"
    base_model_name: str = "gpt2"
    task_type = "CAUSAL_LM"
    inference_mode = False
    r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    fan_in_fan_out = True
    context_size = 50
    bsz = 32
    feature_count = 768
    num_hidden_layers = 12
    friends_seasons = ["s01", "s02", "s03", "s04", "s05", "s06"]
    experiment = "experiment_within/"
