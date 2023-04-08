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
# # 0. Cleans Data and Tokenize

# %% [markdown]
# Training data was downloaded from https://www.statmt.org/wmt14/translation-task.html
# taking all files which have a EN-DE part

# %%
import re
import time
import pathlib

import tokenizers

import unicodedata

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import params
import common

# %%
sns.set_style("whitegrid")

# %%
# Test with what to cleanse umlaute
print("ö Ö ä Ä ü Ü ß".encode())
print(unicodedata.normalize('NFC', "ö Ö ä Ä ü Ü ß").encode())
print(unicodedata.normalize('NFD', "ö Ö ä Ä ü Ü ß").encode())
print(unicodedata.normalize('NFKC', "ö Ö ä Ä ü Ü ß").encode())
print(unicodedata.normalize('NFKD', "ö Ö ä Ä ü Ü ß").encode())
# NFKC seems best

# %% [markdown]
# ## 0.1 Cleans input texts from artifcats and special characters

# %%
output_dir_path = pathlib.Path("Output")
output_dir_path.mkdir(exist_ok=True, parents=False)
output_dir_path = output_dir_path / "0_data"
output_dir_path.mkdir(exist_ok=True, parents=False)

cleans_out_path = output_dir_path / "cleans"
cleans_out_path.mkdir(exist_ok=True, parents=False)

input_dir_path = pathlib.Path("/home/martin/data/WMT-2014-DE-EN")
all_train_files = list( (input_dir_path / "train").glob("*"))
all_val_files = list( (input_dir_path / "val").glob("*"))
all_files_dict = {'train': all_train_files, 'val': all_val_files}

for file_type, files_list in all_files_dict.items():
    print(f"Files for type {file_type}")
    for file in files_list:
        print(f"\t{file}")

# %%
# common.cleanse_text("Test Text abc")

# %%
# %%time
# Cleans texts and determine unique characters

unique_chars = set()

for file_type, files_list in all_files_dict.items():
    tmp_out_path = cleans_out_path / file_type
    tmp_out_path.mkdir(exist_ok=True, parents=False)
    
    for file in files_list:
        print(f"Processing file {file}")
        
        with open(file, 'r') as ff:
            full_text = ff.read()
            full_text = common.cleanse_text(full_text)
            with open(tmp_out_path / file.name, 'w', encoding='utf-8') as ff_out:
                ff_out.write(full_text)
                
            unique_chars.update(set(full_text))

nr_unique_chars = len(unique_chars)
print(f"nr_unique_chars: {nr_unique_chars}")

# %%
# Reconfigure file list to cleans directory
all_train_files = list( (cleans_out_path / "train").glob("*"))
all_val_files = list( (cleans_out_path / "val").glob("*"))
all_files_dict = {'train': all_train_files, 'val': all_val_files}

for file_type, files_list in all_files_dict.items():
    print(f"Cleansed files for type {file_type}")
    for file in files_list:
        print(f"\t{file}")

# %%
# %%time
# Get counts per unique character
char_count_dict = {key: 0 for key in unique_chars}

for file in all_train_files:
    print(f"Processing file {file}")
    with open(file, 'r') as ff:
        full_text = ff.read()
        for char in unique_chars:
            char_count_dict[char] += full_text.count(char)

# %%
df_char_counts = pd.Series(char_count_dict).sort_values(ascending=False)
df_char_counts.to_csv(output_dir_path/"df_char_counts.csv")

# %%
# Cleans the test texts
all_test_files = list( (input_dir_path / "test").glob("*"))
print(all_test_files)

tmp_out_path = cleans_out_path / "test"
tmp_out_path.mkdir(exist_ok=True, parents=False)

for file in all_test_files:
    print(f"Processing file {file}")
        
    with open(file, 'r') as ff:
        full_text = ff.read()
        # don't remove unicode from testset
        full_text = common.cleanse_text(full_text, remove_unicode=False)
        full_text = common.cleans_test_text(full_text)
        with open(tmp_out_path / file.name, 'w', encoding='utf-8') as ff_out:
            ff_out.write(full_text)

# %%
del full_text

# %% [markdown]
# ## 0.2 Train Tokenizer

# %%
# Reconfigure file list to cleans directory
all_train_files = list( (cleans_out_path / "train").glob("*"))
all_val_files = list( (cleans_out_path / "val").glob("*"))
all_test_files = list( (cleans_out_path / "test").glob("*"))

tok_train_files = [str(f) for f in [*all_train_files, *all_val_files] ]
tok_test_files  = [str(f) for f in all_test_files]

# %%
tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))

trainer = tokenizers.trainers.BpeTrainer(
    special_tokens=["[UNK]", "[PAD]", "[START]", "[END]"], # START and END are for the target sentence only
    vocab_size=params.bpe_vocab_size,
    min_frequency=3,
    show_progress=True
)

tokenizer.normalizer = tokenizers.normalizers.Sequence([tokenizers.normalizers.NFKC()]) # , tokenizers.normalizers.StripAccents()]) -> destorys German Umlaute
tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace()
tokenizer.decoder = tokenizers.decoders.Metaspace()

# %%
tokenizer.train(tok_train_files, trainer)

# %%
# save tokenizer
tokenizer.save(str(output_dir_path / 'tokenizer.json'))

# %%
# tokenizer.get_vocab_size()
# tokenizer.get_vocab()

# %%
tokenizer.encode("[UNK][PAD]").ids

# %%
# Test encoding
tokenizer.encode("Hello! This is an example sentence.")

# %%
tokenizer.encode("Hello! This is an example sentence.").ids

# %%
tokenizer.encode("Hello! This is an example sentence.").tokens

# %%
tokenizer.encode("[START]Hello! This is an example sentence.[END]").tokens

# %%
test_text = """I almost wish I hadn't gone down that rabbit-hole -- and yet -- and yet -- it's rather curious, you know, this sort of life!"""

tokenizer.decode(tokenizer.encode(test_text).ids)

# %%
test_text = """[START]I almost wish I hadn't gone down that rabbit-hole -- and yet -- and yet -- it's rather curious, you know, this sort of life![END]"""

tokenizer.decode(tokenizer.encode(test_text).ids)

# %% [markdown]
# ## 0.3 Tokenize Inputs

# %%
all_files_list = [*all_train_files, *all_val_files, *all_test_files]

de_files_list = [f for f in all_files_list if f.suffix=='.de' or 'de.sgm' in f.name]
en_files_list = [f for f in all_files_list if f.suffix=='.en' or 'en.sgm' in f.name]
de_files_list.sort()
en_files_list.sort()

# %% [markdown]
# The direction from the original trafo paper was English to German.

# %%
import gc
gc.collect()

# %%
# %%time

max_token_length = params.max_nr_tokens # everything larger is going to be clipped
print(f"max_token_length: {max_token_length}")

tokenized_dict = {
    'train':     {'de': [], 'en': [], 'de_length': [], 'en_length': [] },
    'val':       {'de': [], 'en': [], 'de_length': [], 'en_length': [] },
    'test-src':  {'de': [], 'en': [], 'de_length': [], 'en_length': [] },
    'test-ref':  {'de': [], 'en': [], 'de_length': [], 'en_length': [] },
}


for idx, (de_file, en_file) in enumerate(zip(de_files_list, en_files_list)):
    print(f"Processing step {idx}")
    print(f"\tde_file: {de_file}")
    print(f"\ten_file: {en_file}")
    
    file_stem = de_file.stem.split('.')[0]
    print(f"\tfile_stem: {file_stem}")
    
    if "newstest2013" in file_stem:
        save_type = "val"
    elif "newstest2014" in file_stem and '-ref' in file_stem:
        save_type = "test-ref"
    elif "newstest2014" in file_stem and '-src' in file_stem:
        save_type = "test-src"
    else:
        save_type = "train"
    
    print(f"\tsave_type: {save_type}")
    
    with open(de_file, 'r', encoding='utf-8') as de_ff, open(en_file, 'r', encoding='utf-8') as en_ff:
        for de_line, en_line in zip(de_ff, en_ff):
            de_line = de_line.strip()
            en_line = en_line.strip()
            if len(de_line)==0 or len(en_line)==0:
                continue
            # add START and END token to de_string (as it is target)
            de_line = ('[START]' +  de_line + '[END]')
            en_line = ('[START]' +  en_line + '[END]')
                
            de_token_obj = tokenizer.encode(de_line)
            en_token_obj = tokenizer.encode(en_line)
            de_token_ids = de_token_obj.ids[:max_token_length]
            en_token_ids = en_token_obj.ids[:max_token_length]
            tokenized_dict[save_type]['de'].append( de_token_ids )
            tokenized_dict[save_type]['en'].append( en_token_ids )
            tokenized_dict[save_type]['de_length'].append( len(de_token_ids) )
            tokenized_dict[save_type]['en_length'].append( len(en_token_ids) )

# %%
# max length is already covered above via clipping very large inputs
# max_length_en = np.quantile(tokenized_dict['train']['en_length'], q=.99999)
# max_length_de = np.quantile(tokenized_dict['train']['de_length'], q=.99999)
# print(f"max_length_en: {max_length_en}")
# print(f"max_length_de: {max_length_de}")

for key, sub_dict in tokenized_dict.items():
    df = pd.DataFrame(sub_dict)
    
    df['max_length'] = df[['en_length', 'de_length']].max(axis=1)
    # if key=='train':
    #     df = df[ df['en_length']<= max_length_en ]
    #     df = df[ df['de_length']<= max_length_de ]
    df = df.sort_values(by='max_length', ascending=False)
    df.to_pickle(output_dir_path / f"df_{key}.pkl")    
        
    fig = plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(data=df, x='en_length', kde=True, color='steelblue', label='En')
    sns.histplot(data=df, x='de_length', kde=True, color='orange', label='De')
    plt.title(f"Token Length of Texts for {key}")
    plt.savefig(output_dir_path / f"token_length_{key}.png")
    plt.legend(loc='upper right')
    plt.show()
    plt.clf()
    plt.close('all')
    
del df

# %%
# enable_padding
# ( direction = 'right'pad_id = 0pad_type_id = 0pad_token = '[PAD]'length = Nonepad_to_multiple_of = None ) 

# %%
tokenizer.token_to_id("[UNK]")

# %%
tokenizer.token_to_id("[PAD]")

# %%
tokenizer.token_to_id("[START]")

# %%
tokenizer.token_to_id("[END]")

# %%
tokenizer.token_to_id("ü")

# %%
tokenizer.token_to_id("ä")

# %%
tokenizer.token_to_id("ö")

# %%
tokenizer.encode("[START]Hello! This is an example [UNK] sentence.[END]").ids
