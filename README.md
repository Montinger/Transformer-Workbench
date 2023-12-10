# Transformer-Workbench

Playground for some experiments with Transformer models.

## transformer-from-scratch

This folder contains code, which implements a transformer from scratch and trains it on the original *Attention is all you need* translation task. See the README in this folder for further details.

## DitBERT

This folder contains the code to define and train my own BERT-like model *DitBERT*, which is also benchmarked with GLUE against *BERT-base* and *RoBERTa-base*. See the README in this folder for further details.

## parse-text-corpora

Contains the code used to parse the text corpora, used to train e.g. DitBERT.

## LoRA-from-scratch

A from scratch implementation of *Low-Rank Adaptation* (LoRA) for a RoBERTa model. This blogpost also shows an optimized version of QLoRA (Quantized LoRA) with the huggingface *PEFT* library and bitsandbytes for quantization.

