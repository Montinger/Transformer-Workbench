# DitBERT

This folder contains code for my own BERT-like model *DitBERT*.


For more details see the corresponding blogpost

- here <https://martin-dittgen.de/blogpost/training-ditbert.html>
- or here <https://medium.com/@martin.p.dittgen/training-a-bert-model-from-scratch-on-a-single-nvidia-rtx-3060-1a7a2b1039a5>


The pre-trained *DitBERT* model and tokenizer can be downloaded from:

- <https://martin-dittgen.de/downloads/DitBERT-with-tokenizer.zip>
- Mirror: <https://storage.googleapis.com/transformer-from-scratch-results/DitBERT-with-tokenizer.zip>

All raw GLUE-Output csv-files are in the git repository as well. You can find them in the subfolder *GLUE-Outputs*.


## Overview of Files

### Notebooks

Note that the notebooks are stored in git in py-percent format. See further below on how to convert them back to notebooks with jupytext.

- **Train-Tokenizer.ipynb:** Trains and tests a tokenizer on the data.
- **GLUE-Finetune.ipynb:** Benchmarks the DitBERT model on the GLUE-Benchmarks.
- **GLUE-Finetune-from-mnli.ipynb:** Benchmarks the DitBERT model on the GLUE-Benchmarks, starting from the pre-trained mnli model.
- **GLUE-Finetune-huggingface.ipynb:** Finetunes the BERT-base and RoBERTa-base model from huggingface on the GLUE-Benchmarks.
- **GLUE-Finetune-huggingface-from-mnli.ipynb:** Finetunes the BERT-base and RoBERTa-base model from huggingface on the GLUE-Benchmarks, starting from the pre-trained mnli model.
- **Aggregate-GLUE-results.ipynb:** Aggregates the results from the GLUE-Benchmarks from the *GLUE-Outputs* folder.


### Python Files

- **params.py:** This file specifies some configuration parameters of the model structure and training process
- **glue_utils.py:** Contains some utils for the fine-tuning on the GLUE-Benchmarks.
- **model_defs.py:** Contains the definition of the *DitBERT* model, the file loader, and the masked language model helper class.


### CSV & YAML Files

- **GLUE-aggregated-results.csv:** Contains the aggregated GLUE results of *DitBERT*, *BERT-base*, and *RoBERTa-base* models.
- **conda_env_linux.yaml:** File to reproduce the environment via conda with `conda env create --file=conda_env_linux.yaml`


## Run the Code

### 1. Setup Environment

You can create the environment on a Linux computer from *conda_env_file_linux.txt*.

Alternatively you should be able to recreate it from scratch like this

~~~~~~~~~bash
# 0. Create environment (here with conda, alternatively with venv)
conda create -n torch-gpu python=3.9

# 1. activate environment
conda activate torch-gpu

# 2. install pytorch
# see here: https://pytorch.org/get-started/locally/

# 3. install other packages with pip
pip install numpy pandas tabulate tokenizers datasets transformers flask waitress jupyter jupytext seaborn matplotlib tqdm nltk torchmetrics
~~~~~~~~~

### 2. Convert Notebooks back from py:percent

The python files starting with a capital letter are actually notebooks, stored via jupytext in py-percent format for easy tracking in git. You can convert them back via:

~~~~~~~~~bash
# convert back to *.ipynb
jupytext --to notebook GLUE-*.py Train-Tokenizer.py Aggregate-GLUE-results.py
~~~~~~~~~
