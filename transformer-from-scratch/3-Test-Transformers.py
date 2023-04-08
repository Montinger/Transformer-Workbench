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
# # 3. Test Transformers

# %% [markdown]
# Trains a transformer from the pytorch class, so that we have a baseline to compare ourselves to

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

import nltk

#from ignite.metrics.nlp import Bleu

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import trange, tqdm

import model_defs
import params

# %%
sns.set_style("whitegrid")

# %%
print(f"torch version: {torch.__version__}")
print(f"nltk version: {nltk.__version__}")

# %%
# torch.autograd.set_detect_anomaly(True)

# %%
output_dir_path = pathlib.Path("Output")
output_dir_path.mkdir(exist_ok=True, parents=False)
output_dir_path = output_dir_path / "3_test"
output_dir_path.mkdir(exist_ok=True, parents=False)

input_data_dir_path = pathlib.Path("Output/0_data")
input_batch_dir_path = pathlib.Path("Output/1_batch_data")
input_model_dir_path = pathlib.Path("/data/Transformers/transformers-from-scratch")
# input_model_dir_path_2 = pathlib.Path("Output/2_train")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu' # manual overwrite
print(f"device: {device}")

# %% [markdown]
# ## Load Models

# %%
# load tokenizer and models
tokenizer = tokenizers.Tokenizer.from_file(str(input_data_dir_path / 'tokenizer.json'))
model_ref = torch.load(input_model_dir_path / 'model-mask-25.pth', map_location=torch.device(device)) #  f'model-ref-25.pth'
model_own = torch.load(input_model_dir_path / 'model-own-30.pth', map_location=torch.device(device))

# %% [markdown]
# ## Some Simple Manual Tests

# %%
translated_text = common.translate_text_greedy("The car is red.", model=model_own, tokenizer=tokenizer)
translated_text

# %%
translated_text = common.translate_text_beam_search("The car is blue. The other one is red!", model=model_own, tokenizer=tokenizer)
translated_text

# %%
translated_text = common.translate_text_greedy("The car is red.", model=model_ref, tokenizer=tokenizer)
translated_text

# %%
translated_text = common.translate_text_beam_search("The car is blue. The other one is red!", model=model_ref, tokenizer=tokenizer)
translated_text

# %% [markdown]
# ## Plot Losses over iterations

# %%
# load loss lists
with open((input_model_dir_path / "model-mask-loss_list.pkl"), 'rb') as handle:
    loss_list_ref = pickle.load(handle)

with open((input_model_dir_path / "model-own-loss_list.pkl"), 'rb') as handle:
    loss_list_own = pickle.load(handle)

# %%
df_loss_ref = pd.DataFrame({
    'iteration': [i for i in range(len(loss_list_ref))], 
    'loss': loss_list_ref,
})
df_loss_own = pd.DataFrame({
    'iteration': [i for i in range(len(loss_list_own))],
    'loss': loss_list_own,
})


df_loss_ref = df_loss_ref.rolling(100, min_periods=3).mean()
df_loss_own = df_loss_own.rolling(100, min_periods=3).mean()

df_loss_ref['network'] = "reference"
df_loss_own['network'] = "own from sratch"

df_loss = pd.concat([df_loss_ref, df_loss_own])# df_loss_own.merge(df_loss_ref, on='iteration', how='left')

df_loss['perplexity'] = np.exp(df_loss['loss'])

df_loss = df_loss.sort_values(by=['iteration', 'network']).reset_index(drop=True)

df_loss.head()

# %%
    
    sns.histplot(data=df_val_ref, x=score_type, kde=True, hue='decoding_type') # , label='En')
    plt.title(f"{score_type} for Validation Dataset NewsTest 2013 with Reference Model")
    plt.xlim((0.0, 1.0))

    plt.savefig(output_dir_path / f"val_newstest_2013_{score_type}_ref.png")
    plt.show()
    plt.clf()
    plt.close('all')

# %%
fig = plt.figure(figsize=(10, 6), dpi=300)
sns.lineplot(data=df_loss, x='iteration', y='loss', hue='network')
plt.savefig(output_dir_path / f"loss_by_iteration.png")
plt.show()
plt.clf()
plt.close('all')

fig = plt.figure(figsize=(10, 6), dpi=300)
sns.lineplot(data=df_loss[df_loss['iteration'] > 50_000], x='iteration', y='loss', hue='network')
plt.savefig(output_dir_path / f"loss_by_iteration_after_50k.png")
plt.show()
plt.clf()
plt.close('all')

# %%
fig = plt.figure(figsize=(10, 6), dpi=300)
sns.lineplot(data=df_loss, x='iteration', y='perplexity', hue='network')
plt.savefig(output_dir_path / f"perplexity_by_iteration.png")
plt.show()
plt.clf()
plt.close('all')


fig = plt.figure(figsize=(10, 6), dpi=300)
sns.lineplot(data=df_loss[df_loss['iteration'] > 50_000], x='iteration', y='perplexity', hue='network')
plt.savefig(output_dir_path / f"perplexity_by_iteration_after_50k.png")
plt.show()
plt.clf()
plt.close('all')

# %% [markdown]
# ## Some Further Tests of Models and Decoder Functions

# %%
tokenizer.encode("Hello! This is an example sentence.")

# %%
# Test calc_bleu_score function
common.calc_bleu_score(
    target_actual =    "I am an elephant! What would this mean: I am unsure. But let's see what happens. I am superman",
    target_predicted = "I'm an elephant! This means: Not sure. But let's see what happens.",
    debug = True
)

common.calc_bleu_score(
    target_actual =    "I am an elephant! What would this mean: I am unsure. But let's see what happens.",
    target_predicted = "I'm an elephant! This means: Not sure. But let's see what happens. I am superman",
    debug = True
)

common.calc_bleu_score(
    target_actual =    "I am an elephant! What would this mean: I am unsure. But let's see what happens.",
    target_predicted = ".",
    debug = True
)

# %%
translated_text = common.translate_text_greedy("Hello. How are you doing? I'm doing well. This is an elephant. I am a tiger.", 
                                               model=model_ref, tokenizer=tokenizer, debug=True)
translated_text

# %%
translated_text = common.translate_text_greedy("The car is red.", model=model_ref, tokenizer=tokenizer)
translated_text

# %%
translated_text = common.translate_text_beam_search("The car is red.", model=model_ref, tokenizer=tokenizer, device="cuda")
translated_text

# %%
translated_text = common.translate_text_greedy("We have decided.", model=model_ref, tokenizer=tokenizer)
translated_text

# %%
translated_text = common.translate_text_greedy("The car is blue. The other one is red!", model=model_ref, tokenizer=tokenizer)
translated_text

# %%
translated_text = common.translate_text_beam_search("The car is blue. The other one is red!", model=model_ref, tokenizer=tokenizer)
translated_text

# %%
translated_text = common.translate_text_greedy("The first proposal was not accepted, and another proposal is now before the Council of Ministers.", 
                                               model=model_ref, tokenizer=tokenizer, device="cuda")
translated_text

# %%
translated_text = common.translate_text_beam_search("The first proposal was not accepted, and another proposal is now before the Council of Ministers.", 
                                               model=model_own, tokenizer=tokenizer, device="cuda")
translated_text

# %% [markdown]
# ## Test BLEU on Val and Test

# %%
score_type_list = ['bleu_score_ntlk', 'bleu_score_torchmetric', 'sacrebleu_score_torchmetric']

def calc_bleu_two_files(source_file_path, target_file_path, model, tokenizer):
    
    bleu_results = {
        'source': [],
        'target_actual': [],
        'target_predicted': [],
        'bleu_score_ntlk': [],
        'bleu_score_torchmetric': [],
        'sacrebleu_score_torchmetric': [],
        'decoding_type': []
    }
    
    num_lines = sum(1 for line in open(source_file_path, 'r'))
    
    with open(source_file_path, 'r', encoding='utf-8') as src_ff, open(target_file_path, 'r', encoding='utf-8') as tgt_ff:
        for src_line, tgt_line in tqdm(zip(src_ff, tgt_ff), total=num_lines):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            
            # some of the files contain empty lines at the top or bottom
            if len(src_line)==0 and len(tgt_line)==0:
                continue

            tgt_line = common.trafos_to_text_before_bleu(tgt_line)    
                
            # Greedy decoding
            tgt_predicted = common.translate_text_greedy(src_line, model=model, tokenizer=tokenizer, device="cuda")
            tgt_predicted = common.trafos_to_text_before_bleu(tgt_predicted)
            
            bleu_score_ntlk = common.calc_bleu_score(target_actual = tgt_line, target_predicted = tgt_predicted)
            bleu_score_torchmetric, sacrebleu_score_torchmetric = common.calc_bleu_score_torchmetrics(target_actual = tgt_line, target_predicted = tgt_predicted)
            
            bleu_results['source'].append(src_line)
            bleu_results['target_actual'].append(tgt_line)
            bleu_results['target_predicted'].append(tgt_predicted)
            bleu_results['bleu_score_ntlk'].append(bleu_score_ntlk)
            bleu_results['bleu_score_torchmetric'].append(bleu_score_torchmetric)
            bleu_results['sacrebleu_score_torchmetric'].append(sacrebleu_score_torchmetric)
            bleu_results['decoding_type'].append('greedy')
            
            
            # Beam search
            tgt_predicted = common.translate_text_beam_search(src_line, model=model, tokenizer=tokenizer, device="cuda")
            tgt_predicted = common.trafos_to_text_before_bleu(tgt_predicted) 
            
            bleu_score_ntlk = common.calc_bleu_score(target_actual = tgt_line, target_predicted = tgt_predicted)
            bleu_score_torchmetric, sacrebleu_score_torchmetric = common.calc_bleu_score_torchmetrics(target_actual = tgt_line, target_predicted = tgt_predicted)
            
            bleu_results['source'].append(src_line)
            bleu_results['target_actual'].append(tgt_line)
            bleu_results['target_predicted'].append(tgt_predicted)
            bleu_results['bleu_score_ntlk'].append(bleu_score_ntlk)
            bleu_results['bleu_score_torchmetric'].append(bleu_score_torchmetric)
            bleu_results['sacrebleu_score_torchmetric'].append(sacrebleu_score_torchmetric)
            bleu_results['decoding_type'].append('beam')

    return bleu_results

# %%
val_input_path = pathlib.Path("Output/0_data/cleans/val")
source_file_path = val_input_path / "newstest2013.en"
target_file_path = val_input_path / "newstest2013.de"

bleu_val_results = calc_bleu_two_files(source_file_path=source_file_path, target_file_path=target_file_path, model=model_ref, tokenizer=tokenizer)

# %%
df_val_ref = pd.DataFrame(bleu_val_results)
df_val_ref.to_csv(output_dir_path / f"df_val_ref.csv")

# %%
df_val_ref.dtypes

# %%
val_input_path = pathlib.Path("Output/0_data/cleans/val")
source_file_path = val_input_path / "newstest2013.en"
target_file_path = val_input_path / "newstest2013.de"

bleu_val_results = calc_bleu_two_files(source_file_path=source_file_path, target_file_path=target_file_path, model=model_own, tokenizer=tokenizer)

df_val_own = pd.DataFrame(bleu_val_results)
df_val_own.to_csv(output_dir_path / f"df_val_own.csv")

# %%
df_val_ref.groupby('decoding_type').mean(numeric_only=True)

# %%
df_val_own.groupby('decoding_type').mean(numeric_only=True)

# %% [markdown]
# The ref dataset seems to be the right one. The other has sentences without matching targets.

# %%
test_input_path = pathlib.Path("Output/0_data/cleans/test")
source_file_path = test_input_path / "newstest2014-deen-ref.en.sgm"
target_file_path = test_input_path / "newstest2014-deen-ref.de.sgm"

bleu_test_results = calc_bleu_two_files(source_file_path=source_file_path, target_file_path=target_file_path, model=model_ref, tokenizer=tokenizer)

# %%
df_test_ref = pd.DataFrame(bleu_test_results)
df_test_ref.to_csv(output_dir_path / f"df_test_ref.csv")

# %%
test_input_path = pathlib.Path("Output/0_data/cleans/test")
source_file_path = test_input_path / "newstest2014-deen-ref.en.sgm"
target_file_path = test_input_path / "newstest2014-deen-ref.de.sgm"

bleu_test_results = calc_bleu_two_files(source_file_path=source_file_path, target_file_path=target_file_path, model=model_own, tokenizer=tokenizer)

# %%
df_test_own = pd.DataFrame(bleu_test_results)
df_test_own.to_csv(output_dir_path / f"df_test_own.csv")

# %%
df_test_ref.groupby('decoding_type').mean(numeric_only=True)

# %%
df_test_own.groupby('decoding_type').mean(numeric_only=True)

# %% [markdown]
# ### Create plots of results

# %%
for score_type in score_type_list:
    fig = plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(data=df_val_ref, x=score_type, kde=True, hue='decoding_type') # , label='En')
    plt.title(f"{score_type} for Validation Dataset NewsTest 2013 with Reference Model")
    plt.xlim((0.0, 1.0))

    plt.savefig(output_dir_path / f"val_newstest_2013_{score_type}_ref.png")
    plt.show()
    plt.clf()
    plt.close('all')
    
    fig = plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(data=df_val_own, x=score_type, kde=True, hue='decoding_type') # , label='En')
    plt.title(f"{score_type} for Validation Dataset NewsTest 2013 with Own From-Scratch Model")
    plt.xlim((0.0, 1.0))

    plt.savefig(output_dir_path / f"val_newstest_2013_{score_type}_own.png")
    plt.show()
    plt.clf()
    plt.close('all')

# %%
for score_type in score_type_list:
    fig = plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(data=df_test_ref, x=score_type, kde=True, hue='decoding_type') # , label='En')
    plt.title(f"{score_type} for Test Dataset NewsTest 2014 with Reference Model")
    plt.xlim((0.0, 1.0))

    plt.savefig(output_dir_path / f"test_newstest_2014_{score_type}_ref.png")
    plt.show()
    plt.clf()
    plt.close('all')
    
    fig = plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(data=df_test_own, x=score_type, kde=True, hue='decoding_type') # , label='En')
    plt.title(f"{score_type} for Test Dataset NewsTest 2014 with Own From-Scratch Model")
    plt.xlim((0.0, 1.0))

    plt.savefig(output_dir_path / f"test_newstest_2014_{score_type}_own.png")
    plt.show()
    plt.clf()
    plt.close('all')
