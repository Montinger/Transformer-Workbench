"""Contains Utils and wrappers for the GLUE tasks

The GLUE benchmark consists of nine tasks, each with a different number of classes.

**CoLA** (Corpus of Linguistic Acceptability): 2 classes
- 0: Ungrammatical sentence
- 1: Grammatical sentence

**SST-2** (Stanford Sentiment Treebank): 2 classes
- 0: Negative sentiment
- 1: Positive sentiment

**MRPC** (Microsoft Research Paraphrase Corpus): 2 classes
- 0: Not paraphrases
- 1: Paraphrases

**STS-B** (Semantic Textual Similarity Benchmark):
    Not a classification task, it's a regression task where the model predicts the similarity score between two sentences,
    ranging from 0 (no relation) to 5 (semantic equivalence).

**QQP** (Quora Question Pairs): 2 classes
- 0: Not duplicate questions
- 1: Duplicate questions

**MNLI** (Multi-Genre Natural Language Inference): 3 classes
- entailment
- contradiction
- neutral

**QNLI** (Question Natural Language Inference): 2 classes
- entailment
- not_entailment

**RTE** (Recognizing Textual Entailment): 2 classes
- entailment
- not_entailment

**WNLI** (Winograd Natural Language Inference): 2 classes
- 0: Pronoun coreference incorrect
- 1: Pronoun coreference correct


Keep in mind that for the STS-B task, the objective is to predict the similarity score between two sentences,
which is a regression problem rather than a classification problem.

Also the MNLI model should be used to predict on the special diagnostic ax-dataset.

WNLI is often left out of benchmarks because it has some problems.
"""

# Imports
import math
import pathlib

import numpy as np
import pandas as pd

from typing import List, Tuple

# Huggingface imports
import tokenizers
import datasets

from torch import nn

from tqdm.auto import trange, tqdm

import params
import model_def



# List of all GLUE tasks
glue_task_list = [
    "ax", # -> special task, only for additional test on mnli model
    "cola",
    "mnli",
    # "mnli_matched", -> should be already part of mnli
    # "mnli_mismatched", -> should be already part of mnli
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb", # -> regression task
    "wnli"
]

glue_train_task_list = glue_task_list.copy()
glue_train_task_list.remove('ax')



class GLUE_TaskLoader:
    def __init__(self, tokenizer: tokenizers.Tokenizer, glue_task: str, max_sequence_length: int, batch_size: int):
        """
        Initialize a GLUE Task Loader. This is a helper to provide a the data when fine-tuning and evaluating
        a BERT model on the GLUE Benchmark.

        Parameters
        ----------
        tokenizer : Tokenizer
            The tokenizer to be used, from the huggingface tokenizers library.
            Is expected to have [START], [END], [DOC], and [PAD] special tokens.
        glue_task : str
            The task from the GLUE Benchmark to be loaded.
        max_sequence_length : int
            The maximum sequence length for tokenization (i.e. the max context length of the BERT model)
        batch_size : int
            The number of samples per batch.
        """
        self.tokenizer = tokenizer
        self.start_token_id = tokenizer.token_to_id('[START]')
        self.end_token_id = tokenizer.token_to_id('[END]')
        self.doc_token_id = tokenizer.token_to_id('[DOC]')
        self.pad_token_id = tokenizer.token_to_id('[PAD]')

        self.glue_task = glue_task
        self.dataset = datasets.load_dataset("glue", glue_task)
        self.metric = datasets.load_metric("glue", glue_task)

        self.max_sequence_length = max_sequence_length
        self.max_sublength_shorten = ((max_sequence_length - 4) // 2) # we have to subtract 4 to allow for the special tokens
        self.batch_size = batch_size

        self.glue_single_sentence_tasks = ["cola", "sst2"]

        self.nr_shortened_sequences = 0


    def convert_row_to_tokens(self, curr_row: dict) -> List[int]:
        """
        Convert a given row to tokens.

        Parameters
        ----------
        curr_row : dict
            The current row to be converted.

        Returns
        -------
        list of int
            The list of tokens representing the current row.
        """
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
                + self.tokenizer.encode(curr_row['sentence']).ids\
                + [self.end_token_id]
        else:
            tokenized_sentence = [self.start_token_id]\
                + self.tokenizer.encode(curr_row['sentence1']).ids\
                + [self.doc_token_id]\
                + self.tokenizer.encode(curr_row['sentence2']).ids\
                + [self.end_token_id]
            if len(tokenized_sentence) > self.max_sequence_length:
                self.nr_shortened_sequences += 1
                tokenized_sentence = [self.start_token_id]\
                    + self.tokenizer.encode(curr_row['sentence1']).ids[:self.max_sublength_shorten]\
                    + [self.doc_token_id]\
                    + self.tokenizer.encode(curr_row['sentence2']).ids[:self.max_sublength_shorten]\
                    + [self.end_token_id]

        return tokenized_sentence

    def epoch_iterator(self, data_type: str='train') -> Tuple[List[List[int]], List[int]]:
        """
        Iterate through the epoch.

        Parameters
        ----------
        data_type : str, optional
            The type of the data to be iterated over, by default 'train'.

        Yields
        -------
        Tuple[List[List[int]], List[int]]
            The prepared batch and corresponding labels.
        """
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

    def get_nr_batches(self, data_type: str='train') -> int:
        """
        Get the number of batches.

        Parameters
        ----------
        data_type : str, optional
            The type of the data to get number of batches for, by default 'train'.

        Returns
        -------
        int
            The number of batches.
        """
        nr_obs = len(self.dataset[data_type])
        print(nr_obs)
        return math.ceil(nr_obs / self.batch_size)

    def get_nr_classes(self) -> int:
        """
        Get the number of classes for the current GLUE task.

        Returns
        -------
        int
            The number of classes.
        """
        if self.glue_task == "stsb":
            return 1 # return 1 because it is actually a regression task, then the classification layer output with "1 class" can be used for regression
        else:
            return pd.Series(self.dataset['train']['label']).nunique()

    def prepare_batch(self, batch: List[List[int]]) -> List[List[int]]:
        """
        Prepare a batch.

        Parameters
        ----------
        batch : List[List[int]]
            The batch to be prepared.

        Returns
        -------
        List[List[int]]
            The prepared batch.
        """
        max_length = max([len(s) for s in batch])
        # This is a backup check, as it should no longer trigger due to check during tokenization
        if max_length > self.max_sequence_length:
            print(f"WARNING: Your sequence is too long for model {max_length}")
        for idx in range(len(batch)):
            while len(batch[idx]) < max_length:
                batch[idx].append(self.pad_token_id)
        return batch




# If the script is called directly we download all GLUE datasets and test everything
if __name__ == "__main__":

    # Download all GLUE tasks
    for glue_task in glue_task_list:
        print(f"Downloading for task {glue_task}")
        dataset = datasets.load_dataset("glue", glue_task)
        if glue_task != "ax":
            metric = datasets.load_metric("glue", glue_task)
    print("\n") # empty line

    # Load Tokenizer
    import model_def
    tokenizer = model_def.tokenizer
    input_path = pathlib.Path("Output")
    tokenizer = tokenizer.from_file(str(input_path / 'tokenizer.json'))

    for glue_task in glue_train_task_list:
        print(f"Testing class GLUE_TaskLoader for task {glue_task}")
        glue_task_loader = GLUE_TaskLoader(tokenizer=tokenizer, glue_task=glue_task, max_sequence_length=130, batch_size=params.finetune_params['batch_size'])

        nr_batches = glue_task_loader.get_nr_batches()
        print(f"  Nr train batches: {nr_batches}")
        nr_classes = glue_task_loader.get_nr_classes()
        print(f"  Nr classes: {nr_classes}")

        print("Testing loop over train dataset")
        tqdm_looper = tqdm(glue_task_loader.epoch_iterator(), total=nr_batches)

        for raw_batch, batch_target_labels in tqdm_looper:
            pass
        print('-'*42)
