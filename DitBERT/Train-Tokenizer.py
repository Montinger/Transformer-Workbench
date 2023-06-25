# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import re
import time
import pathlib

import tokenizers

import unicodedata

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import trange, tqdm

import params
import model_def

# %%
# Reconfigure file list to cleans directory

input_files_path = pathlib.Path("/data_ssd/tmp_text_preparation/cleansed-parsed")

output_path = pathlib.Path("Output")
output_path.mkdir(exist_ok=True, parents=False)

all_train_files = list( (input_files_path.glob("*")) )

# filter files
all_train_files = [f for f in all_train_files if ('removed' not in f.name) and ('_de' not in f.name)
                  # and ('enwikisource-20230101-cleansed' not in f.name) -> removed wikisource because then the memory exploded
                  ]

all_train_files

# %%
# for testing only use cc news
TEST = False
if TEST:
    all_train_files = [f for f in all_train_files if 'cc_news_cleansed' in f.name]
all_train_files

# %% [markdown]
# ## Define Tokenizer and Train file list

# %%
tokenizer = model_def.tokenizer

# %%
# Train files list has to be string only
# Not needed if we use iterator
# tok_train_files = [str(f) for f in all_train_files]
# tok_train_files

# %% [markdown]
# ## Train tokenizer

# %%
trainer = tokenizers.trainers.BpeTrainer(
    special_tokens = params.other_params['special_tokens'],
    vocab_size=params.other_params['bpe_vocab_size'],
    min_frequency=3,
    show_progress=True
)


# %%
def file_iterator():
    for path in all_train_files:
        print(f"Iterating over file {path}")
        with open(path, "r", encoding='utf-8') as ff:
            for line in tqdm(ff):
                yield line



# %%
# %%time
tokenizer.train_from_iterator(file_iterator(), trainer=trainer)

# %%
# Prefer the above due to lower memory requirements
# # %%time
# tokenizer.train(tok_train_files, trainer)

# %%
# save tokenizer
tokenizer.save(str(output_path / 'tokenizer.json'))

# %% [markdown]
# ## Test tokenizer

# %%
tokenizer.encode("Hello! This is an example sentence.").ids

# %%
tokenizer.encode("Hello! This is an example sentence.").tokens

# %%
tokenizer.decode(tokenizer.encode("Hello! This is an example sentence.").ids)

# %%
tokenizer.decode(tokenizer.encode("1234 ℌ𝔢𝔩𝔩𝔬    𝔱𝔥𝔢𝔯𝔢 𝓂𝓎 𝒹ℯ𝒶𝓇 𝕕𝕖𝕒𝕣    𝕗𝕣𝕚𝕖𝕟𝕕!").ids)

# %%
# Test special tokens
for token in params.other_params["special_tokens"]:
    print(f"{token} -- {tokenizer.token_to_id(token)}")

# %%
tokenizer.token_to_id('&')

# %%
# Try reloading tokenizer
tokenizer = tokenizer.from_file(str(output_path / 'tokenizer.json'))

# %%
tokenizer.decode(tokenizer.encode("Hello! This is an example sentence.").ids)

# %% [markdown]
# ## Assess Data and Write Meta Dict

# %% [markdown]
# Prepares a meta dictionary with the number of tokens per file. This will later be used to provide importance sampling during the data loading process.

# %%
# %%time

token_count_dict = {}

# Add special tokens
for token in params.other_params["special_tokens"]:
    token_count_dict[token] = 0

for line in file_iterator():
    token_ids = tokenizer.encode(line).ids
    for token in token_ids:
        token_count_dict[token] = token_count_dict.get(token, 0) + 1

# %%
# Convert to pandas and save file
dict_for_df = {'token_id': [], 'token': [], 'count': []}
for k, v in token_count_dict.items():
    dict_for_df['token_id'].append(k)
    dict_for_df['token'].append(repr(tokenizer.decode([k])))
    dict_for_df['count'].append(v)
    
df = pd.DataFrame(dict_for_df)
df = df.sort_values(by='count', ascending=False).reset_index(drop=True)

# Save to csv
df.to_csv(output_path / 'token_count.csv')

df.head(10)
