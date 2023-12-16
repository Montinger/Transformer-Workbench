# LoRA-from-scratch

This folder contains code for a from scratch implementation of *Low-Rank Adaptation* (LoRA) for a RoBERTa model. It also a cleaner and optimized implementation of QLoRA (Quantized LoRA) with the huggingface *PEFT* library.

For more details see the corresponding blogpost

- here <https://medium.com/towards-data-science/implementing-lora-from-scratch-20f838b046f1>
- or here <https://martin-dittgen.de/blogpost/lora-from-scatch.html>

As the trained LoRA parameters are quite small you can find them in the *Output* folder.


## Overview of Files

### Notebooks

Note that the notebooks are stored in git in py-percent format. See further below on how to convert them back to notebooks with jupytext.

- **Train-LoRA-Benchmarks.ipynb:** Trains the from-scratch LoRA model on the GLUE and SQuAD benchmarks.
- **Train-QLoRA-with-PEFT.ipynb:** Trains a QLoRA implentation with bitsandbytes and the PEFT library and trains it on the GLUE and SQuAD benchmarks.
- **Load-LoRA-Weights.ipynb:** Loads the from-scratch LoRA weights to finally benchmark the performance on the GLUE and SQuAD benchmarks.
- **Load-LoRA-Weights-PEFT.ipynb:** Loads the PEFT LoRA weights to finally benchmark the performance on the GLUE and SQuAD benchmarks.


### Python Files

- **LoraWrapperRoberta.py:** Contains the code which actually implements the from-scratch LoRA for the RoBERTa model.
- **params.py:** This file specifies some configuration parameters of the training process
- **glue_squad_utils.py:** Contains some utils for the fine-tuning on the GLUE and SQuAD benchmarks.
- **lora_utils.py:** Contains some general functions to help with LoRA model training, listing parameters etc.



## Run the Code

### 1. Setup Environment

You should be able to create the environment for training the models like this:

~~~~~~~~~bash
# 0. Create environment (here with conda, alternatively with venv)
conda create -n torch-gpu python=3.11

# 1. activate environment
conda activate torch-gpu

# 2. install pytorch
# see here: https://pytorch.org/get-started/locally/

# 3. install other packages with pip
pip install numpy pandas tabulate tokenizers datasets transformers peft accelerate xformers bitsandbytes jupyter jupytext seaborn matplotlib tqdm
~~~~~~~~~

### 2. Convert Notebooks back from py:percent

The python files starting with a capital letter are actually notebooks, stored via jupytext in py-percent format for easy tracking in git. You can convert them back via:

~~~~~~~~~bash
# convert back to *.ipynb
jupytext --to notebook GLUE-*.py Train-Tokenizer.py Aggregate-GLUE-results.py
~~~~~~~~~
