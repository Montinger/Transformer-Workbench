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

output_path = pathlib.Path("Output")

final_model_path = (output_path / 'DitBERT_model_100k.pth')

# implements a random data-loader to check the memory usage per step size if False here
real_data_train = True

re_initialize_schedules = False # If set to True restart the training with the current model from the current checkpoint, but reinitialize all the schedules and files

change_dropout = False # overwrite the dropout value
new_dropout = 0.025

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu' # manual overwrite
print(f"device: {device}")


####################################################
## Load and Check Model                           ##
####################################################

print("")
print("Initalizing Model\n")

# LOAD Tokenizer
tokenizer = model_def.tokenizer
tokenizer = tokenizer.from_file(str(output_path / 'tokenizer.json'))

# Check for Checkpoint
checkpoint_path = (output_path / 'checkpoint.pth')
found_checkpoint = checkpoint_path.is_file()

if found_checkpoint:
    print("Checkpoint found! Loading model from checkpoint ...")
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    start_step_nr = checkpoint['step_nr']
    optimizer = checkpoint['optimizer']
    warm_up_schedule = checkpoint['warm_up_schedule']
    main_schedule = checkpoint['main_schedule']
    loss_list = checkpoint['loss_list']
    perplexity_list = checkpoint['perplexity_list']
    lm_file_loader_positions = checkpoint['lm_file_loader_positions']

    print(f"Loaded checkpoint from step: {start_step_nr}")

    if re_initialize_schedules:
        print("Will initialize new learning rate schedule")
        start_step_nr = 0
        warm_up_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0000001, end_factor=1.0,
            total_iters=params.other_params['nr_warmup_steps'], last_epoch=-1, verbose=False)
        main_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
            total_iters=params.other_params['nr_total_steps'], last_epoch=-1, verbose=False)

        loss_list = []
        perplexity_list = []
        lm_file_loader_positions = None

    if change_dropout:
        print(f"\nWill adjust all dropout values to {new_dropout}")
        model_def.adjust_model_dropout(model, new_dropout)
        print("all dropouts adjusted\n")
        time.sleep(5)


else:
    print("No Checkpoint found. Initializing new model ...")
    model = model_def.DitBERTModel(
        single_encoder_layer_params = params.single_encoder_layer_params,
        encoder_stack_params = params.encoder_stack_params,
        vocab_size = params.other_params['bpe_vocab_size'],
        max_seq_length = params.other_params['max_sequence_length']+2, # +2 to account for the START and END token
        padding_idx = params.other_params['padding_idx'],
    )

    optimizer = torch.optim.AdamW(model.parameters(), **params.adamW_parameters)

    start_step_nr = 0
    warm_up_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0000001, end_factor=1.0,
        total_iters=params.other_params['nr_warmup_steps'], last_epoch=-1, verbose=False)
    main_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
        total_iters=params.other_params['nr_total_steps'], last_epoch=-1, verbose=False)

    loss_list = []
    perplexity_list = []

    lm_file_loader_positions = None

# short sleep before printing number of parameters
time.sleep(5)

print(f"Number trainable parameters of model: {model_def.count_parameters(model):_}")
print("")

model_def.parameter_count_table(model)


####################################################
## Initialize LM File Loader                      ##
####################################################


print(f"\nInitializing LM File Loader\n")

lm_file_loader = model_def.LM_FileLoader(
    file_list = params.training_files_list,
    tokenizer = tokenizer,
    max_sequence_length = params.other_params['max_sequence_length'],
    sub_batch_size = params.other_params['sub_batch_size'],
    vocab_size = params.other_params['bpe_vocab_size'],
    open_starting_position = lm_file_loader_positions
)


####################################################
## Training                                       ##
####################################################

model.to(device)

criterion = torch.nn.CrossEntropyLoss() # label_smoothing=0.1, ignore_index=params.padding_token_id)
scaler = torch.cuda.amp.GradScaler()

model.train()

print("\nStarting model training\n")

nr_total_steps_with_warmup = params.other_params['nr_warmup_steps'] + params.other_params['nr_total_steps']
batch_size = params.other_params['batch_size']
sub_batch_size = params.other_params['sub_batch_size']
max_sequence_length = params.other_params['max_sequence_length']
vocab_size = params.other_params['bpe_vocab_size']
nr_sub_batches = batch_size // sub_batch_size
nr_special_tokens = len(params.other_params['special_tokens'])
mask_token_id = tokenizer.token_to_id('[MASK]')


tqdm_looper = trange(start_step_nr, nr_total_steps_with_warmup)

for step_nr in tqdm_looper:
    optimizer.zero_grad()

    for sub in range(nr_sub_batches):

        if real_data_train:
            input_tensor = lm_file_loader.get_batch().to(device)
        else:
            input_tensor = lm_file_loader.get_random_batch().to(device)

        masked_tensor, masked_indices, original_values = \
            model_def.mask_tokens(input_tensor, mask_token_id=mask_token_id, nr_special_tokens=nr_special_tokens, vocab_size=vocab_size)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output_tensor = model(input_tensor)
            predicted_values = output_tensor[masked_indices].clone()
            loss = criterion(predicted_values.view(-1, predicted_values.size(-1)), original_values.view(-1))

        # loss.backward()
        scaler.scale(loss).backward()

    # scaler.unscale_(optimizer)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    # scaler.step() first unscales the gradients of the optimizer's assigned params.
    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
    # otherwise, optimizer.step() is skipped.
    scaler.step(optimizer)

    # Updates the scale for next iteration.
    scaler.update()
    # optimizer.step()


    if step_nr <= params.other_params['nr_warmup_steps']:
        warm_up_schedule.step()
    if step_nr > params.other_params['nr_warmup_steps']:
        main_schedule.step()


    loss_list.append(loss.item())
    perplexity_list.append(np.exp(loss.item())) # # The perplexity is just the exp of the loss
    tqdm_looper.set_description(f"ppl: {np.mean(perplexity_list[-100:]):.4f}; loss: {np.mean(loss_list[-100:]):.6f}; lr: {model_def.get_lr(optimizer):.8f}")

    optimizer.zero_grad()

    # save checkpoint every x steps
    if (step_nr % params.other_params['save_after_steps'] == 0) and step_nr > 0:
        # we avoid saving at step 0 (to not accidentally overwrite our proper checkpoint)
        checkpoint = {
            'step_nr': step_nr,
            'optimizer': optimizer,
            'warm_up_schedule': warm_up_schedule,
            'main_schedule': main_schedule,
            'model': model,
            'loss_list': loss_list,
            'perplexity_list': perplexity_list,
            'lm_file_loader_positions': lm_file_loader.get_line_positions(),

        }
        torch.save(checkpoint, checkpoint_path)

torch.save(model, final_model_path)
