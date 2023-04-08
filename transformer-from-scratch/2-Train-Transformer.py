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
# # 2. Train Pytorch Transformer

# %% [markdown]
# Trains a transformer from the pytorch class, so that we have a baseline to compare ourselves to

# %%
import gc
import re
import torch
import pickle
import random
import pathlib

# import tokenizers

import numpy as np
import pandas as pd

import common


from tqdm.auto import trange, tqdm

import model_defs
import params

# %%
print(f"torch version: {torch.__version__}")

# %%
# activate this for debuging:
# torch.autograd.set_detect_anomaly(True)

# %%
# For info on the padding masks see here:
# https://pytorch.org/tutorials/beginner/translation_transformer.html
# https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask

# %%
output_dir_path = pathlib.Path("Output")
output_dir_path.mkdir(exist_ok=True, parents=False)
output_dir_path = output_dir_path / "2_train"
output_dir_path.mkdir(exist_ok=True, parents=False)

input_data_dir_path = pathlib.Path("Output/0_data")
input_batch_dir_path = pathlib.Path("Output/1_batch_data")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu' # manual overwrite
print(f"device: {device}")

# %% [markdown]
# ## Load Batch Data

# %%
# load data
print(f"params.train_batch_file: {params.train_batch_file}")
with open((input_batch_dir_path / params.train_batch_file), 'rb') as handle: # "train_batches.pkl" or "train_batches-debug.pkl" 
    batch_collector = pickle.load(handle)

# %% [markdown]
# there is a downsampled version for the train-batches for quick debugging, which was created like this:
#     
# ~~~~~~~~python
# batch_collector_debug = random.sample(batch_collector, 12).copy()
# with open((input_dir_path / "train_batches-debug.pkl"), 'wb') as handle:
#     pickle.dump(batch_collector_debug, handle, protocol=pickle.HIGHEST_PROTOCOL)
# ~~~~~~~~

# %%
nr_batches = len(batch_collector)

print(f"Nr batches: {nr_batches}")

print(f"Mean mini-batch size: {np.mean([s[0].size(0) for s in batch_collector])}")
print(f"Min mini-batch size:: {np.min([s[0].size(0) for s in batch_collector])}")
print(f"Max mini-batch size:: {np.max([s[0].size(0) for s in batch_collector])}")

print(f"Mean training sentence length (tokens): {np.mean([s[0][0].size(0) for s in batch_collector])}")
print(f"Min training sentence length (tokens): {np.min([s[0][0].size(0) for s in batch_collector])}")
print(f"Max training sentence length (tokens): {np.max([s[0][0].size(0) for s in batch_collector])}")

# %%
# nr tokens per batch
np.max([s[1].size(0) * s[1].size(1) for s in batch_collector])

# %% [markdown]
# ## Define Model

# %%
# define models
print(f"params.train_trafo_type: {params.train_trafo_type}")
print(f"params.transformer_params: {params.transformer_params}")
print(f"params.bpe_vocab_size: {params.bpe_vocab_size}")
print(f"params.max_nr_tokens: {params.max_nr_tokens}")
print(f"params.padding_token_id: {params.padding_token_id}")
print("")

model = model_defs.TransformerModel(
    use_transformer = params.train_trafo_type, # own or ref
    transformer_params = params.transformer_params,
    vocab_size = params.bpe_vocab_size,
    max_seq_length = params.max_nr_tokens,
    padding_idx = params.padding_token_id,
    initilize_positional_with_sinusoids=True

) 

print(f"Number trainable parameters of model: {common.count_parameters(model):_}")
model = model.to(device)

# %% [markdown]
# ## Define loss, optimizer, and schedules

# %%
# Define objects

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=params.padding_token_id) # .1 label smoothing from original attention paper; ignore padding token important!

# roberta used lr 6e-4 for base model; whisper used 1e-3; both used linear decay with approx 1 epoch or less of warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, betas=(0.9, 0.98), eps=1e-06, weight_decay=0.001) # try 0.001 instead of 0.01

warm_up_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0000001, end_factor=1.0, total_iters=nr_batches, last_epoch=-1, verbose=False)
main_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=params.nr_epochs, last_epoch=-1, verbose=True)

# SGD is way more memory friendly, but often transformer models trained with it become unstable
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001, nesterov=True)

# %% [markdown]
# ## Temporary Meta-Test Functions

# %%
# Test parameter counts of submodules
# Test how the reference and own implementation match in terms of parameter count
if False:
    test_nr_heads = 8 # params.transformer_params["nhead"]
    test_d_model = params.transformer_params["d_model"]

    ref_multihead_attention = torch.nn.MultiheadAttention(embed_dim=test_d_model, num_heads=test_nr_heads, batch_first=True)

    own_multihead_attention = model_defs.MultiAttentionHeads(embedding_dim=test_d_model, nr_attention_heads=test_nr_heads)

    print(f"Number trainable parameters of model ref_multihead_attention: {common.count_parameters(ref_multihead_attention):_}")
    print(f"Number trainable parameters of model own_multihead_attention: {common.count_parameters(own_multihead_attention):_}")

# %%
# Test learning rate schedules
# Tests learning rate schedules in a dry run
if False:
    nr_batches = len(batch_collector)
    nr_epochs = 25
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00075, betas=(0.9, 0.98), eps=1e-06, weight_decay=0.01)
    warm_up_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0000001, end_factor=1.0, total_iters=nr_batches, last_epoch=-1, verbose=True)
    main_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=nr_epochs, last_epoch=-1, verbose=True)
    
    for epoch in range(nr_epochs+1):
        print(f"Running for epoch {str(epoch).zfill(2)}/{nr_epochs}")
        for batch_idx in range(nr_batches):
            if epoch == 0:
                warm_up_schedule.step()
        if epoch > 0:
            main_schedule.step()

# %%
# Test memory requirements:
# Perform some approximate memory requirements checks with randomly generated batches
if False:
    # 15 * 2000 as a worst case test
    mini_batch_size = 50 # -> 10*500 = 5k takes approx 10 GB on cpu.
    sim_length = 125 # 16 GB should be sufficient if you run single sentence mode
    sing_src = torch.randint(low=0, high=32000, size=(mini_batch_size, sim_length)).to(device)
    sing_tgt = torch.randint(low=0, high=32000, size=(mini_batch_size, sim_length)).to(device)

    model.train()
    optimizer.zero_grad()

    output = model(sing_src, sing_tgt)
    loss = criterion(output.view(-1, params.bpe_vocab_size), sing_tgt.view(-1))
    loss.backward(retain_graph=False)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

# %% [markdown]
# ## Define Training Helper Functions

# %%
loss_list = []
perplexity_list = []

# Train test helper functions
def train_subbatch(epoch, nr_sub_batch = 32):
    """This function can be used to train with sub-batches on a smaller GPU (with less than 40 GB RAM)"""
    model.train()
    
    tqdm_looper = tqdm(batch_collector)
    for b_idx, (src, tgt) in enumerate(tqdm_looper):
        #if b_idx % 1 == 0:
        #    print(f"\tProcessing b_idx {str(b_idx).zfill(5)}/{nr_batches}")
        
        optimizer.zero_grad()
                  
        src = src.int()
        tgt = tgt.long()
        
        # TMP: Test predict the position to see if embeddings work
        # Test with predicting constant token
        # tgt = torch.zeros(tgt.size()) + 12 # should always predict 12
        
        # Test predicting the position
        # tgt = torch.arange(tgt.size(1), dtype=torch.int, device=tgt.device).unsqueeze(0).expand_as(tgt) + 5 # add a constant because 1 is padding
        # tgt = tgt.long()
        # END OF TMP
        
        nr_sub_batch_items = src.size(0)
        
        for sub_batch_idx in range(0, nr_sub_batch_items, nr_sub_batch):
            
            # sing_src = src[sub_batch_idx].unsqueeze(0).detach().clone().to(device)
            sing_src = src[sub_batch_idx:(sub_batch_idx+nr_sub_batch)].to(device)
            sing_tgt = tgt[sub_batch_idx:(sub_batch_idx+nr_sub_batch)].to(device)

            sing_tgt_in = sing_tgt[:, :-1].detach().clone()
            sing_tgt_out = sing_tgt[:, 1:].detach().clone()
            
            ## print(sing_src.shape)
            ## print(sing_tgt_in.shape)
            ## print(sing_tgt_out.shape)
            
            output = model(sing_src, sing_tgt_in)
            # print(f"output shape :{output.shape}")
            # print(f"sing_tgt_out shape :{sing_tgt_out.shape}")
            # print(f"output shape loss in :{output.view(-1, output.size(-1)).shape}")
            # print(f"sing_tgt_out shape loss in :{sing_tgt_out.view(-1).shape}")
            loss = criterion(output.view(-1, output.size(-1)), sing_tgt_out.view(-1))
            
            # retain graph is not needed
            loss.backward()

            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if epoch == 0:
            warm_up_schedule.step()
        
        loss_list.append(loss.item())
        perplexity_list.append(np.exp(loss.item())) # # The perplexity is just the exp of the loss
        tqdm_looper.set_description(f"ppl: {np.mean(perplexity_list[-100:]):.4f}; loss: {np.mean(loss_list[-100:]):.6f}; lr: {common.get_lr(optimizer):.8f}")

        
# Train test helper function with full batch
def train(epoch):
    """This function uses the batch in full. Requires approx 40 GB of GPU RAM (i.e. A100) to fit the whole batch for the base model""" 
    model.train()
    
    tqdm_looper = tqdm(batch_collector)
    for b_idx, (src, tgt) in enumerate(tqdm_looper):
        optimizer.zero_grad()
                  
        src = src.int().to(device)
        tgt = tgt.long().to(device)
        
        tgt_in = tgt[:, :-1].detach().clone()
        tgt_out = tgt[:, 1:].detach().clone()
        
        output = model(src, tgt_in)
        loss = criterion(output.view(-1, params.bpe_vocab_size), tgt_out.view(-1))
        loss.backward()
                  
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if epoch == 0:
            warm_up_schedule.step()
        
        loss_list.append(loss.item())
        perplexity_list.append(np.exp(loss.item())) # # The perplexity is just the exp of the loss
        tqdm_looper.set_description(f"ppl: {np.mean(perplexity_list[-100:]):.4f}; loss: {np.mean(loss_list[-100:]):.6f}; lr: {common.get_lr(optimizer):.8f}")
        
        del src
        del tgt
        
# to test the schedules in the proper loop
def dummy_train(epoch):
    tqdm_looper = tqdm(batch_collector)
    for b_idx, (src, tgt) in enumerate(tqdm_looper):

        if epoch == 0:
            warm_up_schedule.step()


# %% [markdown]
# ## Training Loop

# %%
# we run for +1 epochs, as epoch 0 is for warm-up

for epoch in range(params.nr_epochs+1):
    print(f"Running for epoch {str(epoch).zfill(2)}/{params.nr_epochs}")
    
    random.shuffle(batch_collector)
    train(epoch)
    # train_subbatch(epoch)
    if epoch > 0:
        main_schedule.step()
    # save model and checkpoint optimizers etc
    model_save_filename = output_dir_path / f'model-{model.transformer_type}-{str(epoch).zfill(2)}.pth'
    torch.save(model, model_save_filename)
    checkpoint = {
        'epoch': epoch,
        'optimizer': optimizer,
        'warm_up_schedule': warm_up_schedule,
        'main_schedule': main_schedule,
    }
    torch.save(checkpoint, (output_dir_path / 'checkpoint.pth') )

    # save loss list (overwrite each time, to plot in Test script)
    with open((output_dir_path / f"model-{model.transformer_type}-loss_list.pkl"), 'wb') as handle:
        pickle.dump(loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # free cuda cache
    torch.cuda.empty_cache()
