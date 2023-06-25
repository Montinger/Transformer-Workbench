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

from transformers import RobertaModel, RobertaTokenizer
from transformers import BertModel, BertTokenizer

import params
import model_def
import glue_utils

# %%
import datasets

# %%
# run_id = "run-Bert-01"
run_id = "run-RoBERTa-01"

# %%
# Set input and output paths
input_path = pathlib.Path("Output")

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

# %%
run_glue_tasks


# %%
class GLUE_TaskLoader:
    """We have to adjust the GLUE TaskLoader for the RoBERTa or BERT model from huggingface"""
    def __init__(self, tokenizer, glue_task, max_sequence_length, batch_size):
        self.tokenizer = tokenizer
        self.start_token_id = tokenizer.convert_tokens_to_ids('<s>')
        self.end_token_id = tokenizer.convert_tokens_to_ids('</s>')
        self.pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')

        self.glue_task = glue_task
        self.dataset = datasets.load_dataset("glue", glue_task)
        self.metric = datasets.load_metric("glue", glue_task)

        self.max_sequence_length = max_sequence_length
        self.max_sublength_shorten = ((max_sequence_length - 4) // 2)
        self.batch_size = batch_size

        self.glue_single_sentence_tasks = ["cola", "sst2"]

        self.nr_shortened_sequences = 0


    def convert_row_to_tokens(self, curr_row):
        # For some of the sentences the key names are different. We align them here
        if self.glue_task=='mnli':
            curr_row['sentence1'] = curr_row['premise']
            curr_row['sentence2'] = curr_row['hypothesis']
        elif self.glue_task=='qnli':
            curr_row['sentence1'] = curr_row['question']
            curr_row['sentence2'] = curr_row['sentence']
        elif self.glue_task=='qqp':
            curr_row['sentence1'] = curr_row['question1']
            curr_row['sentence2'] = curr_row['question2']

        if self.glue_task in self.glue_single_sentence_tasks:
            tokenized_sentence = [self.start_token_id]\
                + self.tokenizer.encode(curr_row['sentence'], add_special_tokens=False)\
                + [self.end_token_id]
        else:
            tokenized_sentence = [self.start_token_id]\
                + self.tokenizer.encode(curr_row['sentence1'], add_special_tokens=False)\
                + [self.end_token_id]\
                + self.tokenizer.encode(curr_row['sentence2'], add_special_tokens=False)\
                + [self.end_token_id]
            
            if len(tokenized_sentence) > self.max_sequence_length:
                self.nr_shortened_sequences += 1
                tokenized_sentence = [self.start_token_id]\
                    + self.tokenizer.encode(curr_row['sentence1'])[:self.max_sublength_shorten]\
                    + [self.end_token_id]\
                    + self.tokenizer.encode(curr_row['sentence2'])[:self.max_sublength_shorten]\
                    + [self.end_token_id]

        return tokenized_sentence

    def epoch_iterator(self, data_type='train'):
        batch = []
        batch_target_labels = []
        self.nr_shortened_sequences = 0
        for curr_row in self.dataset[data_type]:
            batch.append(self.convert_row_to_tokens(curr_row))
            batch_target_labels.append(curr_row['label'])
            if len(batch) >= self.batch_size:
                yield self.prepare_batch(batch), batch_target_labels
                batch = []
                batch_target_labels = []
        # return the last batch
        if len(batch) > 0:
            yield self.prepare_batch(batch), batch_target_labels
            batch = []
            batch_target_labels = []
        print(f"Number sequences shortened due to large length: {self.nr_shortened_sequences}")

    def get_nr_batches(self, data_type='train'):
        nr_obs = len(self.dataset[data_type])
        print(nr_obs)
        return math.ceil(nr_obs / self.batch_size)

    def get_nr_classes(self):
        """Returns the number of classes for the current GLUE task
        """
        if self.glue_task == "stsb":
            return 1 # return 1 because it is actually a regression task, then the classification layer output with "1 class" can be used for regression
        else:
            return pd.Series(self.dataset['train']['label']).nunique()

    def prepare_batch(self, batch):
        max_length = max([len(s) for s in batch])
        # This is a backup check, as it should no longer trigger due to check during tokenization
        if max_length > self.max_sequence_length:
            print(f"WARNING: Your sequence is too long for model {max_length}")
        for idx in range(len(batch)):
            while len(batch[idx]) < max_length:
                batch[idx].append(self.pad_token_id)
        return batch

# %%
for glue_task in run_glue_tasks:
    print(f"\n\nRunning for Task: {glue_task}")
    print("==============================\n")

    # Set device for model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu' # manual overwrite
    print(f"device: {device}")

    # Load Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Setup Training objects
    glue_task_loader = GLUE_TaskLoader(tokenizer=tokenizer, glue_task=glue_task, max_sequence_length=130, batch_size=params.finetune_params['batch_size'])

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


    class BERT_Classifier(nn.Module):
        def __init__(self, num_classes, dropout_rate=0.1):
            """If you want to use the Head for Regression just pass num_classes=1"""
            super(BERT_Classifier, self).__init__()

            # Load the base model from Huggingface's RoBERTa
            self.base_model = RobertaModel.from_pretrained("roberta-base")
            # self.base_model = BertModel.from_pretrained("bert-base-cased")

            # Get the output size of the base model
            d_model = self.base_model.config.hidden_size

            # Define the additional linear layer and dropout
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(d_model, num_classes)

            # Norm
            self.norm = nn.LayerNorm(d_model)

            # save other parameters
            self.d_model = d_model
            self.num_classes = num_classes


        def forward(self, x):

            outputs = self.base_model(x, None)

            # Take the hidden states output from the base model
            x = outputs.last_hidden_state
            x = x[:, 0, :]  # Take output from [CLS] token

            x = self.norm(x)
            x = self.dropout(x)
            x = self.classifier(x)

            if self.num_classes == 1:  # If it's a regression task
                x = torch.sigmoid(x) * 5  # Scale the output to the range [0, 5]

            return x

    # Instantiate your new model with the number of classes
    model = BERT_Classifier(num_classes=nr_classes)

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
