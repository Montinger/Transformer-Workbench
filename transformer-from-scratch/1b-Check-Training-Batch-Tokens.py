# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1b. Check Training Batch Tokens

# %% [markdown]
# Check the distribution of tokens in the training batch. At the end of the script a csv 'df_token_counts.csv' is created which contains a count on how many and which tokens occur in the batched data.

# %%
import gc
import re
import time
import pickle
import random
import pathlib

import tokenizers

import numpy as np
import pandas as pd

import common

import torch

import params

# %%
input_data_dir_path = pathlib.Path("Output/0_data")
batch_data_dir_path = pathlib.Path("Output/1_batch_data")

# %%
batch_file_name = f"train_batches-{params.batch_nr_tokens}.pkl"
batch_file_name_debug = f"train_batches-{params.batch_nr_tokens}-debug.pkl"

print(f"batch_file_name : {batch_file_name}")
print(f"batch_file_name_debug : {batch_file_name_debug}")

# %%
# load data
print(f"batch_file_name: {batch_file_name}")
with open((batch_data_dir_path / batch_file_name), 'rb') as handle:
    batch_collector = pickle.load(handle)

# %%
nr_batches = len(batch_collector)

print(f"Nr batches: {nr_batches}")
print("")
print(f"Mean mini-batch size: {np.mean([s[0].size(0) for s in batch_collector])}")
print(f"Min mini-batch size:: {np.min([s[0].size(0) for s in batch_collector])}")
print(f"Max mini-batch size:: {np.max([s[0].size(0) for s in batch_collector])}")
print("")
print(f"Mean training sentence length en (tokens): {np.mean([s[0][0].size(0) for s in batch_collector])}")
print(f"Min training sentence length en (tokens): {np.min([s[0][0].size(0) for s in batch_collector])}")
print(f"Max training sentence length en (tokens): {np.max([s[0][0].size(0) for s in batch_collector])}")
print("")
print(f"Mean training sentence length en (tokens): {np.mean([s[1][0].size(0) for s in batch_collector])}")
print(f"Min training sentence length en (tokens): {np.min([s[1][0].size(0) for s in batch_collector])}")
print(f"Max training sentence length en (tokens): {np.max([s[1][0].size(0) for s in batch_collector])}")

# %%
# load tokenizer
tokenizer = tokenizers.Tokenizer.from_file(str(input_data_dir_path / 'tokenizer.json'))

# %%
batch_collector[1000][0][0]

# %%
print(tokenizer.decode( batch_collector[100][0][1].tolist() ))
print('-'*20)
print(tokenizer.decode( batch_collector[100][1][1].tolist() ))

# %%
print_nr = 30
print_cnt = 0
with torch.no_grad():
    random.shuffle(batch_collector)
    for b_idx, (src, tgt) in enumerate(batch_collector):
        print(tokenizer.decode( src[0].tolist() ))
        print('-'*20)
        print(tokenizer.decode( tgt[0].tolist() ))
        print('='*20)
        time.sleep(2)
        print_cnt += 1
        if print_cnt > print_nr:
            break

# %%
# decoder can decode like this (you have to provide a sequence
tokenizer.decode([34])

# %%
tokenizer.decode([3], skip_special_tokens=False)

# %%
test_enc = tokenizer.encode("But now John put his hand into his pocket, brought out a whistle, and blew upon it several modulated blasts that rang far across the heated air.")


# %%
for a,b in zip(test_enc.ids, test_enc.tokens):
    print(f"{a} -- {b}")

# %%
test_enc.ids

# %%
tokenizer.decode(test_enc.ids)

# %%
# Check which tockens are present in the data

# dict which maps from token_id to the three sub-keys: 'src_count', 'tgt_count', 'token_decoded'
token_counter = {} 

with torch.no_grad():
    for b_idx, (src, tgt) in enumerate(batch_collector):
        src_counts = src.unique(return_counts=True)
        tgt_counts = tgt.unique(return_counts=True)

        for cnt_idx in range(len(src_counts[0])):
            curr_token = int(src_counts[0][cnt_idx])
            curr_count = int(src_counts[1][cnt_idx])
            # Initialize if not exists yet
            if curr_token not in token_counter:
                token_counter[curr_token] = {'src_count': 0, 'tgt_count': 0, 'token_decoded': tokenizer.decode([curr_token], skip_special_tokens=False)}
            token_counter[curr_token]['src_count'] += curr_count

        for cnt_idx in range(len(tgt_counts[0])):
            curr_token = int(tgt_counts[0][cnt_idx])
            curr_count = int(tgt_counts[1][cnt_idx])
            # Initialize if not exists yet
            if curr_token not in token_counter:
                token_counter[curr_token] = {'src_count': 0, 'tgt_count': 0, 'token_decoded': tokenizer.decode([curr_token], skip_special_tokens=False)}
            token_counter[curr_token]['tgt_count'] += curr_count

        if b_idx % 1_000 == 0:
            print(f"Processing b_idx {b_idx}")

# %%
print(f"Nr keys in token_counter: {len(token_counter.keys())}")

# %%
# delete batch_collector to free memory
del batch_collector
gc.collect()

# %%
# Create pandas dataframe
sorted_keys = sorted(token_counter.keys())
df = pd.DataFrame({
    "Token-ID": sorted_keys,
    "Token-Decoded": [token_counter[s]['token_decoded'] for s in sorted_keys],
    "src_count": [token_counter[s]['src_count'] for s in sorted_keys],
    "tgt_count": [token_counter[s]['tgt_count'] for s in sorted_keys],
})
df['total_count'] = df['src_count'] + df['tgt_count']
df = df.sort_values(by='total_count', ascending=False).reset_index(drop=True)

# %%
df.head(25)

# %%
# save
df.to_csv(batch_data_dir_path / f"df_token_counts.csv")
