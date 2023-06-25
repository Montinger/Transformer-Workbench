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
import re
import bz2
import json
import time
import random
import pathlib

from tqdm.auto import trange, tqdm

import common

# %%
main_input_path = pathlib.Path("/data_ssd/tmp_text_preparation/Edgar-Corpus/2020")

output_path = pathlib.Path('/data_ssd/tmp_text_preparation/cleansed-parsed')
output_path.mkdir(parents=False, exist_ok=True)

# %%
json_file_list = main_input_path.glob(r'**/*.json')

json_file_list = list(json_file_list)
# json_file_list.sort()
random.shuffle(json_file_list) # random shuffle
print(f"Found {len(json_file_list)} json files")

# %%
with open( (output_path / "edgar_cleansed.txt"), 'w', encoding='utf-8') as ff_out:
    for file in tqdm(json_file_list):
        ff_out.write("[DOC]\n")

        with open(file, 'r', encoding='utf-8') as ff:
            # print(ff.read())
            json_in = json.load(ff)
            section_keys = [s for s in json_in.keys() if 'section' in s]
            for k in section_keys:
                text = json_in[k].strip()
                text = common.cleanse_text_chars(text)
                ff_out.write(text)
                ff_out.write('\n\n')
