# Transformer from Scratch

This folder contains code for the reproduction of the *Attention is all you need* paper, and the corresponding *from scratch* implementation of the transformer architecture.

For more details see the corresponding blogpost

- here <https://martin-dittgen.de/blogpost/transformer-from-scratch.html>
- or here <https://medium.com/@martin.p.dittgen/reproducing-the-attention-is-all-you-need-paper-from-scratch-d2fb40bb25d4>

The data, with the trained transformer models, the tokenizer, and the model results on the validation and test set can be downloaded here: TODO


## Overview of Files

### Notebooks

Note that the notebooks are stored in git in py-percent format. See further below on how to convert them back to notebooks with jupytext.

- **0-Cleans-Data-and-Tokenize.ipynb:** Cleanses and tokenizes all texts into a pandas dataframe
- **1-Create-Train-Batches.ipynb:** Creates train batches with certain tokens per batch from the dataframe and saves them as list of pytorch tensors
- **1b-Check-Training-Batch-Tokens.ipynb:** Perform some checks on the created batches, checking that sizes are ok and that the translation pairs are still aligned
- **2-Train-Transformer.ipynb:** The actual script training the transformer
- **3-Test-Transformers.ipynb:** Tests the transformer with some simple sentences and evaluates the BLEU scores on the validation and test sets.


### Python Files

- **microservice.py:** Contains code to run the model as a small flask web app. See further below for how to run
- **params.py:** This file specifies some configuration parameters of the batch creation, model structure and training process
- **common.py:** Some common utility functions for cleansing text, beam search etc.
- **model_defs.py:** The models and *from-scratch* components are defined here

## Run the Code

### 1. Setup Environment

~~~~~~~~~bash
# 0. Create environment (here with conda, alternatively with venv)
conda create -n torch-gpu python=3.9

# 1. activate environment
conda activate torch-gpu

# 2. install pytorch
# see here: https://pytorch.org/get-started/locally/

# 3. install other packages with pip
pip install numpy pandas tabulate tokenizers flask waitress jupyter jupytext seaborn matplotlib tqdm nltk torchmetrics
~~~~~~~~~

### 2. Convert Notebooks back from py:percent

All numbered python scripts are actually notebooks, stored via jupytext in py-percent format.

~~~~~~~~~bash
# convert back to *.ipynb
jupytext --to notebook [0-9]*.py
~~~~~~~~~

### 3. Setup Microservice

The microservice can be stared with 

~~~~~~bash
python microservice.py
~~~~~~

The *waitress* package is used to host the microservice. This package should work on any plattform, from windows to linux.
