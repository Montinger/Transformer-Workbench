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
# # Train LoRA on Benchmarks Notebook

# %% [markdown]
# This script train the LoRA model on the benchmarks and saves the result to the Output folder. Note that this file is actually a notebook which was converted into a py file with jupytext.

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
from transformers import RobertaTokenizer, RobertaModel

from LoraWrapperRoberta import LoraWrapperRoberta

from tqdm.auto import tqdm

# Import Own Modules
import lora_utils

# %% [markdown]
# ## Settings

# %%
output_path = pathlib.Path("Output")
output_path.mkdir(exist_ok=True, parents=False)

model_id = "roberta-base" # "roberta-base" or "roberta-large"

output_path = output_path / model_id
output_path.mkdir(exist_ok=True, parents=False)

TRAIN_BIASES = True  # Whether to also train all biases in the model together with the LoRA matricies
LORA_RANK = 8 # The rank used for each pair of LoRA matrices

squad_v2_na_boost_factor = 2.1 # reduce weight of NA loss
 
DEBUG = False

# %% [markdown]
# ## Initialize Model

# %%
# Code to check the parameters of the original roberta-base model
## model_id = 'roberta-base'
## model = RobertaModel.from_pretrained(model_id)
## lora_utils.parameter_count_table(model)

# %%
model = LoraWrapperRoberta(model_id=model_id, task_type="squad_v1", lora_rank=LORA_RANK, train_biases=TRAIN_BIASES)

# %%
# model.replace_multihead_attention()
# model.freeze_parameters_except_lora_and_bias()

# %%
lora_parameter_count = lora_utils.count_parameters(model)
print(f"Parameter count, after LoRA adjustments: {lora_utils.format_with_underscore(lora_parameter_count)}")

lora_parameter_ratio = lora_parameter_count / model.base_model_param_count
print(f"Ratio: {lora_parameter_ratio:.6%}")

# %%
lora_utils.parameter_count_table(model, output_file_path = (output_path/"parameter_count_after_lora.txt"))

# %%
# Test model
# device = "cpu"
# test_input = tokenizer(["Hello World", "This is a test"], padding="max_length")
# test_input_tensor = torch.tensor(test_input.input_ids, dtype=torch.long).long().to(device)
# test_input_tensor
# model(test_input_tensor)

# %% [markdown]
# ## GLUE and SQUAD Training

# %%
from glue_squad_utils import *
import params

# %%
train_task_list_loop = train_task_list

# %%
# Keep tokenizer around, such that we can initialize the task loader first and get the number of classes, before initializing the model
tokenizer = model.tokenizer

# Set device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu' # manual overwrite
print(f"device: {device}")


# %%
# Helper functions for the training
def initialize_model(task, model_id, nr_classes):
    if task in glue_task_list:
        return LoraWrapperRoberta(model_id=model_id, task_type="glue", num_classes=nr_classes, lora_rank=LORA_RANK, train_biases=TRAIN_BIASES)
    else:
        return LoraWrapperRoberta(model_id=model_id, task_type=task, lora_rank=LORA_RANK, train_biases=TRAIN_BIASES)


# %%
# Training loop
for task in train_task_list_loop:
    print(f"\n\nRunning for Task: {task}")
    print("==============================\n")

    # 0. Setup Task Loader and Training objects
    task_loader = GlueSquadTaskLoader(tokenizer=tokenizer, task=task, batch_size=params.finetune_params['batch_size'], debug=DEBUG)
    nr_batches = task_loader.get_nr_batches()
    print(f"Nr batches: {nr_batches}")
    nr_classes = task_loader.get_nr_classes()
    print(f"Nr classes: {nr_classes}")
    
    # 1. Initialize model
    del model
    model = initialize_model(task, model_id, nr_classes)
    model = model.to(device)
    
    nr_warmup_epochs = params.finetune_params['nr_warmup_epochs']
    nr_epochs = params.finetune_params['nr_epochs']
    nr_total_epochs = (nr_warmup_epochs + nr_epochs)
    
    val_test_keys = [s for s in task_loader.dataset.keys() if 'val' in s or 'test' in s] # -> test is always empty for the GLUE tasks
    
    # Define Optimizer and Schedules
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], **params.adamW_parameters_finetune)
    loss_list = []

    warm_up_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.000000001, end_factor=1.0, 
                                                         total_iters=(nr_batches * nr_warmup_epochs),
                                                         last_epoch=-1, verbose=False)
    main_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                                      total_iters=(nr_batches * nr_epochs),
                                                      last_epoch=-1, verbose=False)

    ####################################################
    ## Training                                       ##
    ####################################################

    # dict to save results
    res_dict = {
        'epoch': []
    }

    model.to(device)

    if task == "stsb":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    scaler = torch.cuda.amp.GradScaler()

    print("Starting model training")

    for epoch in range(nr_total_epochs):
        print(f"Running for epoch {epoch}")
        res_dict['epoch'].append(epoch)

        # Train loop
        model.train()
        
        predictions = [] # also calc the in-training metric
        if task in ["squad", "squad_v1", "squad_v2"]:
            predictions = {'id': [], 'prediction_text': []}
            if task == "squad_v2":
                predictions['no_answer_probability'] = []
        tqdm_looper = tqdm(task_loader.epoch_iterator(), total=nr_batches)

        for raw_batch in tqdm_looper:
            
            optimizer.zero_grad()
    
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
            
            with torch.cuda.amp.autocast():
                if task in ["squad", "squad_v1", "squad_v2"]:
                    if task == "squad_v2":
                        start_logits, end_logits, na_prob_logits = model(input_tensor, attention_mask=attention_mask)
                    else:
                        start_logits, end_logits = model(input_tensor, attention_mask=attention_mask)
                        
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index = -1)
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                    total_loss = (start_loss + end_loss) / 2
                    if task == "squad_v2":
                        loss_fct_cls = torch.nn.BCEWithLogitsLoss()
                        cls_loss = loss_fct_cls(na_prob_logits, non_answerable)
                        total_loss += cls_loss * squad_v2_na_boost_factor
                    loss = total_loss
                else:
                    output_tensor = model(input_tensor, attention_mask=attention_mask)
                    loss = criterion(output_tensor, target_labels)
 
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            # Steps for schedule
            if epoch < nr_warmup_epochs:
                warm_up_schedule.step()
            else:
                main_schedule.step()

            with torch.no_grad():
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

            loss_list.append(loss.item())
            tqdm_looper.set_description(f"loss: {np.mean(loss_list[-100:]):.6f}; lr: {lora_utils.get_lr(optimizer):.8f}")

        # Evaluate with Metrics for Train
        k = "train"
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
        else:
            res_score = task_loader.metric.compute(predictions=predictions, references=task_loader.shuffled_dataset['label'])
        print(f"score for {k}: {res_score}")
        for score_key, score_value in res_score.items():
            k_add = f"{k}_{score_key}"
            if k_add not in res_dict:
                res_dict[k_add] = []
            res_dict[k_add].append(score_value)


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
                        if task == "squad_v2":
                            start_logits, end_logits, na_prob_logits = model(input_tensor, attention_mask=attention_mask)
                        else:
                            start_logits, end_logits = model(input_tensor, attention_mask=attention_mask)
                    else:
                        output_tensor = model(input_tensor, attention_mask=attention_mask)
                    
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
    df_res.to_csv(output_path / f"results_{task}.csv")

    # save model
    model.save_lora_state_dict((output_path / f"model_{task}.pth"))

    # save loss list
    with open((output_path / f"model-{task}-loss_list.pkl"), 'wb') as handle:
        pickle.dump(loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("\nBest Results:")
    print(df_res.max())
    print('-'*42)

