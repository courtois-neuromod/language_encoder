from dataclasses import dataclass


@dataclass
class DataBaseConfig:
    """."""

    target_layer: int = 13
    atlas: str = "MIST"
    parcel: str = "444"
    subject_id: str = "sub-03"
    n_splits: int = 8
    random_state: int = 42
    test_season = "s03"  # season allocated for test
    TR: float = 1.49

    bold_dir: str = "/scratch/ibilgin/friends.timeseries/"
    stimuli_dir: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/gpt2"
    )
    output_dir: str = "/scratch/ibilgin/Dropbox/friends_encoder/data/ridge_regression"
    tr_tsv_path: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/word_alignment/"
    )
    hrf_model = "spm"
    model = "BertModel"
    tokenizer = "BertTokenizer"
