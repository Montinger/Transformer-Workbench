# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Load LoRA Weights & Test them

# %% [markdown]
# This file loads all the lora-parameters from the output folder (as determine by the corresponding Train script) loads them and then applies itself to the benchmarks. Note that this file is actually a notebook which was converted into a py file with jupytext.

# %% [markdown]
# ## Imports

# %%
# Basic Imports
import time
import pathlib
import pickle

import numpy as np

import torch
import torch.nn as nn

# Basic Model Imports
import peft
import bitsandbytes as bnb
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoTokenizer, BitsAndBytesConfig

from tqdm.auto import tqdm

# Import Own Modules
import lora_utils
from glue_squad_utils import *
import params

# %% [markdown]
# ## Settings

# %%
io_path = pathlib.Path("Output_PEFT")

model_id = "roberta-base" # "roberta-base" or "roberta-large"

io_path = io_path / model_id


# %% [markdown]
# ## Test on Tasks

# %%
# Helper functions for the training to initialize model

def initialize_model(task, nr_classes):
    if task in glue_task_list:
        model = peft.AutoPeftModel.from_pretrained(io_path / f'model_{task}.pth', num_labels=nr_classes)
        model = model.merge_and_unload()
        return model
    else:
        model = peft.AutoPeftModel.from_pretrained(io_path / f'model_{task}.pth')
        model = model.merge_and_unload()
        return model


# %%
# Set device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu' # manual overwrite
print(f"device: {device}")

# %%
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
df_res_collector = pd.DataFrame() # collector for results

# Training loop
for task in train_task_list:
    print(f"\n\nRunning for Task: {task}")
    print("==============================\n")

    # 0. Setup Task Loader and Testing objects
    task_loader = GlueSquadTaskLoader(tokenizer=tokenizer, task=task, batch_size=params.finetune_params['batch_size'])
    nr_batches = task_loader.get_nr_batches()
    print(f"Nr batches: {nr_batches}")
    nr_classes = task_loader.get_nr_classes()
    print(f"Nr classes: {nr_classes}")
    
    # 1. Initialize model
    weights_input_path = io_path / f"model_{task}.pth"
    if not weights_input_path.exists():
        print(f"\nDid not found weights for task in location {weights_input_path}. Skipping.")
        continue
        
    model = initialize_model(task, nr_classes=nr_classes)
    
    model = model.to(device)
    
    val_test_keys = [s for s in task_loader.dataset.keys() if 'val' in s or 'test' in s] # -> test is always empty for the GLUE tasks
    
    res_dict = {'task': [task]}

    ####################################################
    ## Validation & Test Loop                         ##
    ####################################################
    with torch.no_grad():
        for k in val_test_keys:
            model.eval()
            predictions = [] # also calc the in-training metric
            if task in ["squad", "squad_v1", "squad_v2"]:
                predictions = {'id': [], 'prediction_text': []}
                if task == "squad_v2":
                    predictions['no_answer_probability'] = []

            tqdm_looper = tqdm(task_loader.epoch_iterator(split_type=k), total=task_loader.get_nr_batches(split_type=k))

            for raw_batch in tqdm_looper:

                # Eval load input data
                input_tensor = raw_batch['input_ids'].clone().detach().to(device).long()
                attention_mask = raw_batch['attention_mask'].clone().detach().to(device).long()
        
                if task in ["squad", "squad_v1", "squad_v2"]:
                    start_positions = raw_batch['start_positions'].long().clone().detach().to(device)
                    end_positions = raw_batch['end_positions'].long().clone().detach().to(device)
                    if task == "squad_v2":
                        non_answerable = raw_batch['non_answerable'].float().to(device)   
                elif task == "stsb":
                    target_labels = raw_batch['label'].clone().detach().to(device).float()
                else: # glue, non stsb
                    target_labels = raw_batch['label'].clone().detach().to(device).long()
                        

                # Eval model predictions
                if task in ["squad", "squad_v1", "squad_v2"]:
                    output_tensor = model(input_tensor, attention_mask=attention_mask)
                    # Extract the outputs
                    start_logits = output_tensor['start_logits']
                    end_logits = output_tensor['end_logits']
                    if task == "squad_v2":
                        na_prob_logits = output_tensor['start_logits'][:, 0] + output_tensor['end_logits'][:, 0]
                else:
                    output_tensor = model(input_tensor, attention_mask=attention_mask)['logits']
            
                # Eval save model predictions
                if task == "stsb":
                    predictions += output_tensor.clone().detach().squeeze().tolist()
                elif task in ["squad", "squad_v1", "squad_v2"]:
                    start_positions = start_logits.clone().detach().argmax(-1).tolist()
                    end_positions = end_logits.clone().detach().argmax(-1).tolist()
                    # If end position is before the start position set it to the start position to extract an empty string
                    end_positions = [e if e >= s else s for s, e in zip(start_positions, end_positions)]
                    
                    
                    # Reconstruct answer and check
                    recon_answer_tokens = [ ts[start:end] for ts, start, end in zip(raw_batch['input_ids'], start_positions, end_positions)]
                    recon_answer = task_loader.tokenizer.batch_decode(recon_answer_tokens, skip_special_tokens = True, clean_up_tokenization_spaces = True)
                    predictions['id'] += raw_batch['id']
                    predictions['prediction_text'] += recon_answer
                    if task == "squad_v2":
                        na_pred_probabilities = torch.sigmoid(na_prob_logits.clone().detach()).tolist()
                        predictions['no_answer_probability'] += na_pred_probabilities
                else:
                    predictions += output_tensor.clone().detach().argmax(-1).tolist()
                
            # Eval apply metric
            if task in ["squad", "squad_v1", "squad_v2"]:
                references_dict = [
                    {'id': id, 'answers': answer}  # , 'no_answer_threshold': 0.5
                    for id, answer in zip(task_loader.shuffled_dataset['id'], task_loader.shuffled_dataset['answers'])
                ]
                if task == "squad_v2":
                    predictions_dict = [
                        {'id': id, 'prediction_text': prediction_text, 'no_answer_probability': no_answer_probability} 
                        for id, prediction_text, no_answer_probability in zip(predictions['id'], predictions['prediction_text'], predictions['no_answer_probability'])
                    ]
                else:
                    predictions_dict = [
                        {'id': id, 'prediction_text': prediction_text} 
                        for id, prediction_text in zip(predictions['id'], predictions['prediction_text'])
                    ]
                res_score = task_loader.metric.compute(predictions=predictions_dict, references=references_dict)
            elif "test" not in k:
                # Don't run this for GLUE problems with test, as it is not defined
                res_score = task_loader.metric.compute(predictions=predictions, references=task_loader.shuffled_dataset['label'])
            else:
                res_score = {"not_calculated": np.nan}
            print(f"score for {k}: {res_score}")
            for score_key, score_value in res_score.items():
                k_add = f"{k}_{score_key}"
                if k_add not in res_dict:
                    res_dict[k_add] = []
                res_dict[k_add].append(score_value)

    df_res = pd.DataFrame(res_dict)
    df_res_collector = pd.concat([df_res_collector, df_res])

print("\nResults:")
print(df_res_collector)

df_res_collector.to_csv(io_path / "load_lora_peft_test_results.csv")

