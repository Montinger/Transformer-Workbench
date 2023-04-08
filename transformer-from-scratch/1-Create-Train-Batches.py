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
# # 1. Create Train Batches

# %%
import gc
import re
# import joblib
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
# Pads in a way that source and target have the same length. Is requried because my transformer implementation cannot handle different lengths
same_length = True

# %%
input_dir_path = pathlib.Path("Output/0_data")
output_dir_path = pathlib.Path("Output/1_batch_data")
output_dir_path.mkdir(exist_ok=True, parents=False)

# %%
if same_length:
    out_file_name = f"train_batches-{params.batch_nr_tokens}-same.pkl"
    out_file_name_debug = f"train_batches-{params.batch_nr_tokens}-same-debug.pkl"
else:
    out_file_name = f"train_batches-{params.batch_nr_tokens}.pkl"
    out_file_name_debug = f"train_batches-{params.batch_nr_tokens}-debug.pkl"
print(f"out_file_name : {out_file_name}")
print(f"out_file_name_debug : {out_file_name_debug}")

# %%
df_train = pd.read_pickle(input_dir_path / "df_train.pkl")

# %%
df_train = df_train.reset_index(drop=True)

# %%
df_train['max_length'].max()

# %%
df_train['max_length'].min()

# %%
df_train['de_length'].min()

# %%
df_train['en_length'].min()

# %%
len_from_df_train = len(df_train)
len_from_df_train


# %%
def prep_batch_to_add_old(list1, list2, max_token_length):
    if len(list1)==0 or len(list2)==0:
        raise ValueError("List of length 0")
    # Padding
    # loop over sequences (list1 and list2 have the same length dimension here)
    for idx in range(len(list1)):
        while len(list1[idx]) < max_token_length:
            list1[idx].append(params.padding_token_id)
        while len(list2[idx]) < max_token_length:
            list2[idx].append(params.padding_token_id)

    # English to German, thus list 1 should be the english one
    # int16 is sufficent for vocab_size < 32_767
    src = torch.tensor(list1, dtype=torch.int16)
    tgt = torch.tensor(list2, dtype=torch.int16)
    
    return [src, tgt]

def prep_batch_to_add(list1, max_token_length):
    if len(list1)==0:
        raise ValueError("List of length 0")
    # Padding
    # loop over sequences (list1 and list2 have the same length dimension here)
    for idx in range(len(list1)):
        while len(list1[idx]) < max_token_length:
            list1[idx].append(params.padding_token_id)

    # English to German, thus list 1 should be the english one
    # int16 is sufficent for vocab_size < 32_767
    out_tensor = torch.tensor(list1, dtype=torch.int16)
    
    return out_tensor


# %%
curr_batch_idx = 0
batch_idx_list = []
nr_obs_in_batch = 0
max_length_de = 0
max_length_en = 0

for df_idx, row in df_train.iterrows():
    if df_idx % 250_000 == 0:
        print(f"Processing row {df_idx}")
    
    nr_obs_in_batch += 1
    max_length_de = max(max_length_de, row['de_length'])
    max_length_en = max(max_length_en, row['en_length'])
    nr_tokens_de = nr_obs_in_batch * max_length_de
    nr_tokens_en = nr_obs_in_batch * max_length_en
    if ( (nr_tokens_de > params.batch_nr_tokens) or (nr_tokens_en > params.batch_nr_tokens) ):
        # start a new batch for this row and reset params
        curr_batch_idx += 1
        nr_obs_in_batch = 1
        max_length_de = max(0, row['de_length'])
        max_length_en = max(0, row['en_length'])

    batch_idx_list.append(curr_batch_idx)
        
df_train['batch_idx'] = batch_idx_list 

# %%
batch_collector = []

max_batch_idx = df_train['batch_idx'].max()
print(f"max_batch_idx: {max_batch_idx}")

for idx in range(max_batch_idx+1):
    if idx % 1_000 == 0:
        print(f"Processing batch {idx}")
        
    tmp_df = df_train[ df_train['batch_idx']==idx ]
    
    tmp_de_collector = []
    tmp_en_collector = []
    max_token_length_de = 0
    max_token_length_en = 0
    for df_idx, row in tmp_df.iterrows():
        max_token_length_de = max(max_token_length_de, len(row['de']) )
        max_token_length_en = max(max_token_length_en, len(row['en']) )
        tmp_de_collector.append(row['de'])
        tmp_en_collector.append(row['en'])
    if same_length:
        max_token_length_both = max(max_token_length_en, max_token_length_de-1) # -1 and then +1 because target is one shorter
        batch_collector.append( 
            [
                prep_batch_to_add(tmp_en_collector, max_token_length_both),
                prep_batch_to_add(tmp_de_collector, max_token_length_both+1),
            ]
        )
    else:
        batch_collector.append( 
            [
                prep_batch_to_add(tmp_en_collector, max_token_length_en),
                prep_batch_to_add(tmp_de_collector, max_token_length_de),
            ]
        )

# %%
len(batch_collector)

# %%
df_train = None
del df_train
gc.collect()

# %%
# Check that no sentence was lost
len_from_batches = sum([s[0].shape[0] for s in batch_collector])
len_from_batches

# %%
assert len_from_df_train==len_from_batches, "A sentence was lost. Please recheckt the code."

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
with open((output_dir_path / out_file_name), 'wb') as handle:
    pickle.dump(batch_collector, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # joblib.dump(batch_collector, handle)

# %%
batch_collector_debug = random.sample(batch_collector, 12).copy()
with open((output_dir_path / out_file_name_debug), 'wb') as handle:
    pickle.dump(batch_collector_debug, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
# Test dataset

# %%
random.shuffle(batch_collector)

# %%
len(batch_collector)

# %% [markdown]
# With this length this means that the original paper trained for approx 20 epochs

# %%
for b_idx, (src, tgt) in enumerate(batch_collector):
    if b_idx % 1_000 == 0:
        print(f"Processing b_idx {b_idx}")

# %%
src.shape

# %%
tgt.shape
