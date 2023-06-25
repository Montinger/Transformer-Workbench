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
import datasets

# %%
run_id = "run-01"

# %%
# Set input and output paths
input_path = pathlib.Path("Output")

model_path = (input_path / 'DitBERT_model_100k.pth')

output_path = pathlib.Path("GLUE-Outputs")
output_path.mkdir(parents=False, exist_ok=True)
output_path = output_path / run_id
output_path.mkdir(parents=False, exist_ok=True)

# %%
with open( (output_path / 'finetune_params.txt'), 'w', encoding='utf-8') as ff:
    ff.write("finetune_params:\n")
    for k,v in params.finetune_params.items():
        ff.write(f"{k}: {v}\n")
    ff.write("\nadamW_parameters_finetune:\n")
    for k,v in params.adamW_parameters_finetune.items():
        ff.write(f"{k}: {v}\n")

# %%
run_glue_tasks = glue_utils.glue_train_task_list
run_glue_tasks = ["stsb"]

# %%
for glue_task in run_glue_tasks:
    print(f"\n\nRunning for Task: {glue_task}")
    print("==============================\n")

    # Set device for model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu' # manual overwrite
    print(f"device: {device}")

    # Load Tokenizer
    tokenizer = model_def.tokenizer
    tokenizer = tokenizer.from_file(str(input_path / 'tokenizer.json'))

    # Setup Training objects
    glue_task_loader = glue_utils.GLUE_TaskLoader(tokenizer=tokenizer, glue_task=glue_task, max_sequence_length=130, batch_size=params.finetune_params['batch_size'])

    nr_batches = glue_task_loader.get_nr_batches()
    print(f"Nr batches: {nr_batches}")
    nr_classes = glue_task_loader.get_nr_classes()
    print(f"Nr classes: {nr_classes}")

    nr_warmup_epochs = params.finetune_params['nr_warmup_epochs']
    nr_epochs = params.finetune_params['nr_epochs']
    nr_total_epochs = (nr_warmup_epochs + nr_epochs)

    # randomly shuffle the dataset
    glue_task_loader.dataset.shuffle()
    
    val_test_keys = [s for s in glue_task_loader.dataset.keys() if 'val' in s] # or 'test' in s] -> test is always empty
    print(val_test_keys)


    # Define Model with new Head
    class DitBERT_Classifier(nn.Module):
        def __init__(self, base_model, num_classes, dropout_rate=0.1):
            """If you want to use the Head for Regression just pass num_classes=1"""
            super(DitBERT_Classifier, self).__init__()

            # replace the token output map layer by identity, such that we get a d_model output
            base_model.to_token_map = nn.Identity()

            # Register the base_model as a submodule
            self.add_module("base_model", base_model)

            # Get the output size of the base model (assuming it's a linear layer)
            d_model = self.base_model.d_model

            # Define the additional linear layer and dropout
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(d_model, num_classes)

            # Norm
            self.norm = nn.LayerNorm(d_model)

            # save other parameters
            self.d_model = d_model
            self.num_classes = num_classes


        def forward(self, x):

            x = self.base_model(x)
            
            x = x[:, 0, :] # Take output from zeroth position token
            
            # or: Perform mean-pooling
            # x = x.mean(dim=1)
            
            x = self.norm(x)
            x = self.dropout(x)
            x = self.classifier(x)
            
            if self.num_classes == 1:  # If it's a regression task
                x = torch.sigmoid(x) * 5  # Scale the output to the range [0, 5]

            return x


    # Load the base model from the .pth file
    base_model = torch.load(model_path, map_location=torch.device(device))

    # Instantiate your new model with your base model and the number of classes
    model = DitBERT_Classifier(base_model=base_model, num_classes=nr_classes)

    # Adjust the dropout to a higher value again for training
    new_dropout = params.finetune_params['new_dropout']

    model_def.adjust_model_dropout(model, new_dropout, verbose=0)

    # Define Optimizer and Schedules
    optimizer = torch.optim.AdamW(model.parameters(), **params.adamW_parameters_finetune)
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
    # Manually move the base model's embedding layer to the specified device (wasn't registered properly) -> false alarm, other bug
    # model.base_model.full_embedding.to(device)
    # model.base_model.full_embedding.embedding.to(device)

    if glue_task == "stsb":
        criterion = torch.nn.MSELoss() # torch.nn.L1Loss() # torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss() # label_smoothing=0.1, ignore_index=params.padding_token_id)
    scaler = torch.cuda.amp.GradScaler()

    print("Starting model training")

    for epoch in range(nr_total_epochs):
        print(f"Running for epoch {epoch}")
        res_dict['epoch'].append(epoch)

        # Train loop
        model.train()
        predictions = [] # also calc the in-training metric
        tqdm_looper = tqdm(glue_task_loader.epoch_iterator(), total=nr_batches)

        for raw_batch, batch_target_labels in tqdm_looper:
            optimizer.zero_grad()

            input_tensor = torch.tensor(raw_batch, dtype=torch.long).long().to(device)

            if glue_task == "stsb":
                target_labels = torch.tensor(batch_target_labels, dtype=torch.float).float().to(device)
            else:
                target_labels = torch.tensor(batch_target_labels, dtype=torch.long).long().to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output_tensor = model(input_tensor)
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
                if glue_task == "stsb":
                    predictions += output_tensor.detach().squeeze().tolist() 
                else:
                    predictions += output_tensor.detach().argmax(-1).tolist()

            loss_list.append(loss.item())
            tqdm_looper.set_description(f"loss: {np.mean(loss_list[-100:]):.6f}; lr: {model_def.get_lr(optimizer):.8f}")

        k = "train"
        res_score = glue_task_loader.metric.compute(predictions=predictions, references=glue_task_loader.dataset[k]['label'])
        print(f"score for {k}: {res_score}")
        for score_key, score_value in res_score.items():
            k_add = f"{k}_{score_key}"
            if k_add not in res_dict:
                res_dict[k_add] = []
            res_dict[k_add].append(score_value)

        # Validation & Test Loop
        with torch.no_grad():
            for k in val_test_keys:
                model.eval()
                predictions = []

                tqdm_looper = tqdm(glue_task_loader.epoch_iterator(data_type=k), total=glue_task_loader.get_nr_batches(data_type=k))

                for raw_batch, batch_target_labels in tqdm_looper:

                    input_tensor = torch.tensor(raw_batch, dtype=torch.long).long().to(device)
                    if glue_task == "stsb":
                        target_labels = torch.tensor(batch_target_labels, dtype=torch.float).float().to(device)
                    else:
                        target_labels = torch.tensor(batch_target_labels, dtype=torch.long).long().to(device)

                    output_tensor = model(input_tensor)

                    if glue_task == "stsb":
                        predictions += output_tensor.detach().squeeze().tolist() 
                    else:
                        predictions += output_tensor.argmax(-1).tolist()

                res_score = glue_task_loader.metric.compute(predictions=predictions, references=glue_task_loader.dataset[k]['label'])
                print(f"score for {k}: {res_score}")
                for score_key, score_value in res_score.items():
                    k_add = f"{k}_{score_key}"
                    if k_add not in res_dict:
                        res_dict[k_add] = []
                    res_dict[k_add].append(score_value)
        print('-'*42)

    # save model
    torch.save(model, (output_path / f"model_{glue_task}.pth"))

    df_res = pd.DataFrame(res_dict)
    df_res.to_csv(output_path / f"results_{glue_task}.csv")

    print(df_res.max())
