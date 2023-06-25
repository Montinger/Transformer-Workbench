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
import gc
import re
import math
import time
import torch
import pickle
import random
import pathlib
import tabulate

import tokenizers

import numpy as np
import pandas as pd

from torch import nn

from tqdm.auto import trange, tqdm

import params
import model_def
import glue_utils

# %%
# Set input and output paths
input_path = pathlib.Path("GLUE-Outputs")

output_path = pathlib.Path("")

# %%
csv_files_list = input_path.glob("**/*.csv")
csv_files_list = list(csv_files_list)

# Remove ipynb checkpoints (sometimes they creep in)
csv_files_list = [f for f in csv_files_list if "/.ipynb_checkpoints" not in str(f)]

ditbert_csv_files_list = [f for f in csv_files_list if "Bert" not in f.parent.name and "RoBERTa" not in f.parent.name]
bert_csv_files_list = [f for f in csv_files_list if "Bert" in f.parent.name]
roberta_csv_files_list = [f for f in csv_files_list if "RoBERTa" in f.parent.name]
print(f"DitBERT GLUE files: {ditbert_csv_files_list}\n")
print(f"Bert GLUE files: {bert_csv_files_list}\n")
print(f"RoBERTa GLUE files: {roberta_csv_files_list}")


# %%
def parse_glue_csv_list(csv_files_list):
    """Aggregates the results for a certain list of files"""
    res_dict = {
        'run_id': [],
        'glue_task': [],
        'best_epoch': [],
        'best_val': [],
    }

    for csv_file in csv_files_list:
        run_id = csv_file.relative_to(input_path).parent
        glue_task = csv_file.stem.replace('results_', '')
        df_csv = pd.read_csv(csv_file)
        mean_list = []
        for idx, row in df_csv.iterrows():
            mean_list.append(np.mean([row[c] for c in row.keys() if "validation_" in c]))
        df_csv['mean_validation_metric'] = pd.Series(mean_list)

        max_val = df_csv['mean_validation_metric'].max()
        idxmax = df_csv['mean_validation_metric'].idxmax()
        best_epoch = df_csv['epoch'][idxmax]

        res_dict['run_id'].append(run_id)
        res_dict['glue_task'].append(glue_task)
        res_dict['best_epoch'].append(best_epoch)
        res_dict['best_val'].append(max_val)

    df_res = pd.DataFrame(res_dict)
    return df_res


# %%
df_res = parse_glue_csv_list(ditbert_csv_files_list)
df_res_bert = parse_glue_csv_list(bert_csv_files_list)
df_res_roberta = parse_glue_csv_list(roberta_csv_files_list)


# %%
def aggregate_best_glue_results(df_res, remove_wnli=True):
    """remove wnli because the task is ill designed. Many researchers remove it too."""

    # Group by 'Category' and get the index of the maximum 'Value' in each group
    max_indices = df_res.groupby('glue_task')['best_val'].idxmax()

    # Select the rows with the maximum 'Value' in each group using the indices
    df_best_glue = df_res.loc[max_indices]
    
    # filter out wnli if true
    if remove_wnli:
        df_best_glue = df_best_glue[~df_best_glue['glue_task'].isin(['wnli'])]

    df_total = pd.DataFrame({
        'run_id': ["Total"],
        'glue_task': ["GLUE"],
        'best_epoch': [-1],
        'best_val': [np.mean(df_best_glue['best_val'])]
    })

    df_best_glue = pd.concat([df_best_glue, df_total])
    return df_best_glue


# %%
df_res_agg = aggregate_best_glue_results(df_res)
df_res_agg

# %%
df_res_bert_agg = aggregate_best_glue_results(df_res_bert)
df_res_bert_agg

# %%
df_res_roberta_agg = aggregate_best_glue_results(df_res_roberta)
df_res_roberta_agg

# %%
df_tot_res = df_res_agg.merge(df_res_bert_agg, on='glue_task', suffixes=('_DitBERT', '_BERT'))\
    .merge(df_res_roberta_agg, on='glue_task', suffixes=('', '_RoBERTa'))
df_tot_res

# %%
df_tot_res_T = df_tot_res.rename(
    columns={
        'best_val': "RoBERTa", 
        'best_val_BERT': "BERT",
        'best_val_DitBERT': "DitBERT",
    }).set_index('glue_task')[['DitBERT', 'BERT', 'RoBERTa']].T
df_tot_res_T

# %%
print(df_tot_res_T.to_markdown(index=True, floatfmt=".2%"))

# %%
df_tot_res_T.to_csv(output_path / f"GLUE-aggregated-results.csv")
