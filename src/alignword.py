
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any
import argparse
import glob
import string
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# from peft.tuners.lora.config import LoraConfig
from torch import nn
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerBase,
)

# from cneuromax.fitting.deeplearning.litmodule import BaseLitModuleConfig
# from cneuromax.projects.friends_language_encoder.litmodule_peft import (
#     FriendsFinetuningModel,
# )
# from cneuromax.projects.friends_language_encoder.utils import (
#     list_episodes,
#     list_seasons,
#     preprocess_words,
#     set_output,
# )
STUDY_PARAMS = {
    "tr": 1.49,
    "max_tokens": 512,
}



def extract_features(
    tsv_path: str,
    model: BertModel,
    tokenizer:BertTokenizer,
) -> np.array:
    """.
    """
    max_tk = 8

    df = pd.read_csv(tsv_path, sep = '\t')
    df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())

    #raw_text = ""
    tokens = []
    np_token = []
    token_features = []
    pooled_features = []

    for i in range(40): #df.shape[0]):
        #print(i)
        num_tokens = 0
        if not df.iloc[i]["is_na"]:
            tr_text = df.iloc[i]["text_per_tr"]
            #raw_text += tr_text
            # print(tr_text)

            # tokenize raw punctuated text
            tokens.extend(tokenizer.tokenize(tr_text))


            tr_np_tokens = tokenizer.tokenize(
                tr_text.translate(str.maketrans('', '', string.punctuation)),
            )
            num_tokens = len(tr_np_tokens)
            np_token.extend(tr_np_tokens)

        if len(tokens) > 0:
            # for each TR, extract features from window <= 512 of the latest tokens
            input_ids = [101] + tokenizer.convert_tokens_to_ids(
                tokens[-(STUDY_PARAMS["max_tokens"]-2):]
            ) + [102]
            print(input_ids)
            tensor_tokens = torch.tensor(input_ids).unsqueeze(0)
            print(tensor_tokens)

            with torch.no_grad():
                outputs = model(tensor_tokens)

            pooled_features.append(
                np.array(
                    outputs["pooler_output"][0].detach().numpy(),
                    dtype='float32',
                )
            )

            last_feat = np.repeat(np.nan, 768*max_tk).reshape((max_tk, 768))
            if num_tokens > 0:

                tk_idx = min(max_tk, num_tokens)
                # truncate raw text to last 510 tokens (BERT maximum)
                #np_tr_text = tokenizer.convert_tokens_to_string(np_token[-(STUDY_PARAMS["max_tokens"]-2):])
                #data = pipe(np_tr_text)
                #last_embeddings = np.array(data[0][-(tk_idx+1):-1], dtype='float32')
                input_ids_np = [101] + tokenizer.convert_tokens_to_ids(
                    np_token[-(STUDY_PARAMS["max_tokens"]-2):]
                ) + [102]
                np_tensor_tokens = torch.tensor(input_ids_np).unsqueeze(0)

                with torch.no_grad():
                    np_outputs = np.array(
                        model(
                            np_tensor_tokens
                        )['last_hidden_state'][0][1:-1].detach().numpy(),
                        dtype='float32',
                    )

                last_feat[-tk_idx:, :] = np_outputs[-tk_idx:]

            token_features.append(last_feat)

        else:
            token_features.append(
                np.repeat(np.nan, 768*max_tk).reshape((max_tk, 768))
            )
            pooled_features.append(
                np.repeat(np.nan, 768)
            )

    """
    # old pipeline for aligned-scripts.tsv files, one row per timestamped word...

    tokens = []
    features = []

    start_times = [x for x in np.arange(0, df['onset'].tolist()[-1] + 1.0, STUDY_PARAMS["tr"])]

    for start in start_times:
        words_tr = df[np.logical_and(df['onset'] > start, df['onset'] <= start+STUDY_PARAMS["tr"])]["word"].tolist()
        if len(words_tr) > 0:
            for word in words_tr:
                tokens.extend(tokenizer.tokenize(word))

        if len(tokens) > 0:
            input_ids = [101] + tokenizer.convert_tokens_to_ids(tokens[-STUDY_PARAMS["max_tokens"]:]) + [102]
            tr_tokens = torch.tensor(input_ids).unsqueeze(0)

            with torch.no_grad():
                outputs = model(tr_tokens)
            features.append(
                np.array(outputs["pooler_output"][0].detach().numpy(), dtype='float32')
            )

        else:
            features.append(
                np.repeat(np.nan, 768)
            )

    return np.concatenate(features, axis=0)
    """
    return np.array(pooled_features, dtype='float32'), np.array(token_features, dtype='float32')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


pooled_features, token_features = extract_features(
    tsv_path="/home/ibilgin/brain_encoding/tr_alignment/s01/friends_s01e01a_aligned-text.tsv",
    model=model,
    tokenizer=tokenizer
)