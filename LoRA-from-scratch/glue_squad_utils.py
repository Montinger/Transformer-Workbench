"""Contains Utils and wrappers for the GLUE tasks, training with RoBERTa or BERT from huggingface

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

Also the MNLI model should be used to predict on the special diagnostic ax-dataset, if you decide to do so.

WNLI is often left out of benchmarks because it has some problems.
"""

# Imports
import math
import time
import pathlib

import numpy as np
import pandas as pd

from typing import List, Tuple, Dict

# Huggingface imports
import transformers
import datasets

from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import trange, tqdm



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

# Create a full list with also the race and squad tasks
train_task_list = ['squad_v1', 'squad_v2', *glue_train_task_list]



def filter_first_100_rows(example, idx):
    """Helper function to filter huggingface datasets for debugging and testing"""
    return idx < 100


class GlueSquadTaskLoader:
    def __init__(self, tokenizer, task: str, batch_size: int, debug=False):
        """
        Initialize a GLUE and SQuAD Task Loader. This is a helper to provide the data when fine-tuning and evaluating
        a BERT model on these Benchmarks.

        Parameters
        ----------
        tokenizer : Tokenizer
            The tokenizer to be used, from the huggingface transformers library.
        task : str
            The task from the GLUE Benchmark to be loaded, or one of ['squad_v1', 'squad_v2'].
        batch_size : int
            The number of samples per batch.
        debug : bool
            If debug then only a few observations are processed to quickly test the models
        """
        self.tokenizer = tokenizer

        self.task = task
        if task in glue_task_list:
            self.dataset = datasets.load_dataset("glue", task)
            self.metric = datasets.load_metric("glue", task)
        elif task in ["squad", "squad_v2"]:
            self.dataset = datasets.load_dataset(task)
            self.metric = datasets.load_metric(task)
        elif task == "squad_v1":
            # squad_v1 will also map to "squad"
            self.dataset = datasets.load_dataset("squad")
            self.metric = datasets.load_metric("squad")
        else:
            raise TypeError(f"Task {task} is not recognized with GlueSquadTaskLoader")

        self.batch_size = batch_size
        self.glue_single_sentence_tasks = ["cola", "sst2"]
        self.debug = debug


    @staticmethod
    def cleanse_squad_answers(text):
        """
        Cleanse the given answers for a SQuAD task according to the specifications (see comments for details).
        """
        # Strip the text
        text = text.strip()

        # Remove punctuation at the end, except period (as this is often in the answer)
        if text.endswith(',') or text.endswith(':'):
            text = text[:-1]

        # If the text contains an unmatched " or ' at the beginning or end, remove it
        if text.startswith('"') and text.count('"') == 1:
            text = text[1:]
        if text.endswith('"') and text.count('"') == 1:
            text = text[:-1]
        if text.startswith("'") and text.count("'") == 1:
            text = text[1:]
        if text.endswith("'") and text.count("'") == 1:
            text = text[:-1]

        # If it starts '(', remove the symbol
        if text.startswith('('):
            text = text[1:]

        if text.startswith('�'):
            text = text[1:]

        if text.endswith("%.") or text.endswith("%;") or text.endswith("%:") or text.endswith("%)"):
            text = text[:-1]

        if text.endswith("%).") or text.endswith("%);") or text.endswith("%):"):
            text = text[:-2]

        # The SQuAD text is rather inconsisteny with the dollar sign for monetary amounts -> probably best to leave is as is for now
        ## if text.startswith('$'):
        ##    text = text[1:]

        # If text has no more '(' in it after the above and it ends with ')' or ').' or '),' remove the closing parentheses expression
        if '(' not in text and (text.endswith(')') or text.endswith(').') or text.endswith('),')):
            text = text.rsplit(')', 1)[0]

        # If texts starts with '-' or '—' remove it
        if text.startswith('-') or text.startswith('—') or text.startswith('–') or text.startswith('–'):
            text = text[1:]

        return text


    def encode_single_sentence_task(self, batch):
        """Helper function to use the datasets.map function for single task GLUE sentences"""
        return self.tokenizer(text=batch[self.key_text_1], padding='longest', truncation=True)


    def encode_two_sentence_task(self, batch):
        """Helper function to use the datasets.map function for most two-text pieces tasks, like most GLUE and SQuAD."""
        return self.tokenizer(text=batch[self.key_text_1], text_pair=batch[self.key_text_2], padding='longest', truncation=True)


    def encode_squad_task(self, batch, DEBUG=False):
        """Convert SQuAD-like targets into tokenized input compatible targets.
        """
        # time.sleep(10)
        start_positions = []
        end_positions = []

        tokenized_main = self.tokenizer(text=batch[self.key_text_1], text_pair=batch[self.key_text_2], padding='longest', truncation=True)

        # List comprehension with conditional to handle unanswerable questions
        answer_texts = [item['text'][0] if item['text'] else "" for item in batch['answers']]

        # Create the non_answerable flag
        non_answerable = [1 if not item['text'] else 0 for item in batch['answers']]

        answer_starts = [item['answer_start'][0] if item['text'] else -1 for item in batch['answers']]
        answer_text_length = [len(t) for t in answer_texts]

        text_till_answer_start = [item[:start] for item, start in zip(batch['context'], answer_starts)]
        text_till_answer_end = [item[:(start+len_answer)] for item, start, len_answer in zip(batch['context'], answer_starts, answer_text_length)]
        recon_answer_text_raw = [item[start:(start+len_answer)] for item, start, len_answer in zip(batch['context'], answer_starts, answer_text_length)]

        tokenized_main = self.tokenizer(text=batch[self.key_text_1], text_pair=batch[self.key_text_2], padding='longest', truncation=True)

        for answer_orig, answer_reconstructed in zip(answer_texts, recon_answer_text_raw):
            if answer_orig != answer_reconstructed:
                print(f"No match for orig: {answer_orig} with extraction: {answer_reconstructed}")

        tok_texts_till_answer_start = self.tokenizer(text=batch[self.key_text_1], text_pair=text_till_answer_start, padding='do_not_pad', truncation=True)
        tok_texts_till_answer_end = self.tokenizer(text=batch[self.key_text_1], text_pair=text_till_answer_end, padding='do_not_pad', truncation=True)

        # -1 to account for last end token, which should not be counted
        # for start -2 to account for the added space with the first token
        start_positions = [len(p)-2 for p in tok_texts_till_answer_start['input_ids']]
        end_positions = [len(p)-1 for p in tok_texts_till_answer_end['input_ids']]

        # Overwrite if non-answerable with -1. We will ignore this index later in the loss function via ignore_index=-1
        start_positions = [-1 if na else sp for na, sp in zip(non_answerable, start_positions)]
        end_positions = [-1 if na else sp for na, sp in zip(non_answerable, end_positions)]

        # Reconstruct answer and check
        recon_answer_tokens = [ ts[start:end] for ts, start, end in zip(tokenized_main['input_ids'], start_positions, end_positions)]
        recon_answer = self.tokenizer.batch_decode(recon_answer_tokens, skip_special_tokens = True, clean_up_tokenization_spaces = True)

        recon_answer_clensed = [self.cleanse_squad_answers(t) for t in recon_answer]

        wrong_decoding_count = 0
        for answer_orig, answer_reconstructed, question, context, na in zip(answer_texts, recon_answer_clensed, batch['question'], batch['context'], non_answerable):
            # Sometimes there is a $ sign in the answers and sometimes not. Only count these mistakes if they are not
            if na==0:
                if answer_orig.replace("$", '').replace('.', '') != answer_reconstructed.replace("$", '').replace('.', ''):
                    if DEBUG:
                        print(f"No match for orig: '{answer_orig}' with decoded: '{answer_reconstructed}' for question {question}")
                        print(context)
                        print('-'*42)
                    wrong_decoding_count += 1
        if DEBUG: print(f"Number wrong SQuAD decodings: {wrong_decoding_count}")

        return {**tokenized_main,
            'start_positions': start_positions,
            'end_positions': end_positions,
            'non_answerable': non_answerable,
            'id': batch['id'],
        }



    def epoch_iterator(self, split_type: str='train') -> Dict:
        """
        Iterate through the epoch.

        Parameters
        ----------
        split_type : str, optional
            The type of the data to be iterated over, by default 'train'.

        Yields
        -------
        Dict
            Dictionary with keys for each element
        """

        self.key_text_1 = "sentence1"
        self.key_text_2 = "sentence2"
        if self.task in self.glue_single_sentence_tasks:
            self.key_text_1 = "sentence"
            self.key_text_2 = None
        elif self.task=='mnli':
            self.key_text_1 = "premise"
            self.key_text_2 = "hypothesis"
        elif self.task=='qnli':
            self.key_text_1 = "question"
            self.key_text_2 = "sentence"
        elif self.task=='qqp':
            self.key_text_1 = "question1"
            self.key_text_2 = "question2"
        elif "squad" in self.task:
            self.key_text_1 = "question"
            self.key_text_2 = "context"

        # Shuffle the dataset
        dataset = self.dataset[split_type].shuffle()

        if self.debug:
            dataset = dataset.filter(filter_first_100_rows, with_indices=True)
            print("WARNING: The dataset was filtered to 100 obs for debugging and testing")

        self.shuffled_dataset = dataset

        # Save the current logging level
        old_level = transformers.logging.get_verbosity()

        # Set the new logging level to prevent the weird, non-sense tokenizer warning, which is repeated for every batch
        transformers.logging.set_verbosity_error()

        # Apply the encoding function to all batches.
        if self.task in self.glue_single_sentence_tasks:
            dataset = dataset.map(self.encode_single_sentence_task, batched=True, batch_size=self.batch_size)
            add_batch_keys = ['label', 'idx']
        elif self.task in ['squad', 'squad_v1', 'squad_v2']:
            dataset = dataset.map(self.encode_squad_task, batched=True, batch_size=self.batch_size)
            add_batch_keys = ['start_positions', 'end_positions', 'non_answerable', 'id']
        else:
            dataset = dataset.map(self.encode_two_sentence_task, batched=True, batch_size=self.batch_size)
            add_batch_keys = ['label', 'idx']

        # Revert back to the old logging level
        transformers.logging.set_verbosity(old_level)

        # Set pytorch data loader for the batches
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', *add_batch_keys])
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        for batch in dataloader:
            # batch is a dictionary with keys corresponding to the features of the dataset
            # and values are the batched values. The main ones you wanne use are 'input_ids' and 'attention_mask'
            yield batch


    def get_nr_batches(self, split_type: str='train') -> int:
        """
        Get the number of batches.

        Parameters
        ----------
        split_type : str, optional
            The type of the data to get number of batches for, by default 'train'.

        Returns
        -------
        int
            The number of batches.
        """
        nr_obs = len(self.dataset[split_type])
        return math.ceil(nr_obs / self.batch_size)


    def get_nr_classes(self) -> int:
        """
        Get the number of classes for the current GLUE task.

        Returns
        -------
        int
            The number of classes.
        """
        if self.task == "stsb":
            return 1 # return 1 because it is actually a regression task, then the classification layer output with "1 class" can be used for regression
        elif self.task in ["squad", "squad_v1"]:
            # because it is a context mapper task it is actually None, but we will return the dimension of the linear output we require
            return 2
        elif self.task in ["squad_v2"]:
            # like squad_v1 but with one dimension more to predict whether the question is actually answerable
            return 3
        else:
            return pd.Series(self.dataset['train']['label']).nunique()




# If the script is called directly we download all datasets and test everything for roberta-large
if __name__ == "__main__":

    PRINT_EVERY_BATCH = False

    # Load Tokenizer
    from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel
    model_id = "roberta-large"
    tokenizer = RobertaTokenizer.from_pretrained(model_id)

    for task in train_task_list:
        print(f"Testing class GlueSquadTaskLoader for task {task}")
        task_loader = GlueSquadTaskLoader(tokenizer=tokenizer, task=task, batch_size=32)

        nr_batches = task_loader.get_nr_batches()
        print(f"  Nr train batches: {nr_batches}")
        nr_classes = task_loader.get_nr_classes()
        print(f"  Nr classes: {nr_classes}")

        print("Testing loop over train dataset")

        tqdm_looper = tqdm(task_loader.epoch_iterator())


        for raw_batch in tqdm_looper: # , batch_target_labels
            if PRINT_EVERY_BATCH:
                print(raw_batch.keys())
                print(len(raw_batch['input_ids']))
                print(raw_batch['input_ids'].shape)
        print('-'*42)
