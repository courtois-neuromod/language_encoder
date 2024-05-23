import string
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from encoder_dataclass import DataBaseConfig
from tqdm import tqdm

data_config = DataBaseConfig.DataConfig()


def extract_features(data_config) -> np.array:
    """."""

    max_tk = 8

    df = pd.read_csv(data_config.tsv_path, sep="\t")

    # raw_text = ""
    tokens = []
    np_token = []
    token_features = []
    pooled_features = []
    max_seq_length = data_config.model.config.max_position_embeddings

    for i in range(df.shape[0]):
        # print(i)
        num_tokens = 0
        if not df.iloc[i]["is_na"]:
            tr_text = df.iloc[i]["text_per_tr"]
            # raw_text += tr_text
            # print(tr_text)

            # tokenize raw punctuated text
            tokens.extend(data_config.tokenizer.tokenize(tr_text))

            tr_np_tokens = data_config.tokenizer.tokenize(
                tr_text.translate(str.maketrans("", "", string.punctuation)),
            )
            num_tokens = len(tr_np_tokens)
            np_token.extend(tr_np_tokens)

        if len(tokens) > 0:
            # for each TR, extract features from window <= 512 of the latest tokens
            input_ids = (
                [101]
                + data_config.tokenizer.convert_tokens_to_ids(
                    tokens[-max_seq_length - 2 :]
                )
                + [102]
            )
            tensor_tokens = torch.tensor(input_ids).unsqueeze(0)

            with torch.no_grad():
                outputs = data_config.model(tensor_tokens)

            pooled_features.append(
                np.array(
                    outputs["pooler_output"][0].detach().numpy(),
                    dtype="float32",
                )
            )

            last_feat = np.repeat(np.nan, 768 * max_tk).reshape((max_tk, 768))
            if num_tokens > 0:

                tk_idx = min(max_tk, num_tokens)
                # truncate raw text to last 510 tokens (BERT maximum)
                # np_tr_text = tokenizer.convert_tokens_to_string(np_token[-(STUDY_PARAMS["max_tokens"]-2):])
                # data = pipe(np_tr_text)
                # last_embeddings = np.array(data[0][-(tk_idx+1):-1], dtype='float32')
                input_ids_np = (
                    [101]
                    + data_config.tokenizer.convert_tokens_to_ids(
                        np_token[-(max_seq_length - 2) :]
                    )
                    + [102]
                )
                np_tensor_tokens = torch.tensor(input_ids_np).unsqueeze(0)

                with torch.no_grad():
                    np_outputs = np.array(
                        data_config.model(np_tensor_tokens)["last_hidden_state"][0][
                            1:-1
                        ]
                        .detach()
                        .numpy(),
                        dtype="float32",
                    )

                last_feat[-tk_idx:, :] = np_outputs[-tk_idx:]

            token_features.append(last_feat)

        else:
            token_features.append(
                np.repeat(np.nan, 768 * max_tk).reshape((max_tk, 768))
            )
            pooled_features.append(np.repeat(np.nan, 768))

    return np.array(pooled_features, dtype="float32"), np.array(
        token_features, dtype="float32"
    )


def save_features(
    episode: str,
    pool_features: np.array,
    tk_features: np.array,
    outfile_name: str,
    comp_args: dict,
) -> None:
    """.

    Save episode's text features into .h5 file.
    """
    flag = "a" if Path(outfile_name).exists() else "w"

    with h5py.File(outfile_name, flag) as f:
        group = f.create_group(episode)

        group.create_dataset(
            "text_pooled",
            data=pool_features,
            **comp_args,
        )
        group.create_dataset(
            "text_token",
            data=tk_features,
            **comp_args,
        )
