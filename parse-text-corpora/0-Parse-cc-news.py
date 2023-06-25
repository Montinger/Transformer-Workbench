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
# Test sub criteria
text = """
Voll's sensual female figures were nevertheless met with dire reactions from more conservative segments of society; reactions that at times were expressed through vandalism. This reactionary view of the art was nevertheless temporary, and between 1960 and 1964 a federally organized memorial exhibition of Voll toured German museums (Baden-Baden, Bremen, Kaiserslautern, Karlsruhe, Mannheim, Munich, Pforzheim and Saarbrücken). This led to a number of museum purchases from the collection, but not much more.date=June 2019

"""

text = re.sub(r"^date\=.*?\n", '', text, flags=re.MULTILINE)
text

# %%
main_input_path = pathlib.Path("/data_ssd/tmp_text_preparation/cc_news/cc_news/cc_download_articles")

output_path = pathlib.Path('/data_ssd/tmp_text_preparation/cleansed-parsed')
output_path.mkdir(parents=False, exist_ok=True)

# %%
json_file_list = main_input_path.glob(r'**/*.json')

json_file_list = list(json_file_list)
# json_file_list.sort()
random.shuffle(json_file_list) # random shuffle
print(f"Found {len(json_file_list)} json files")

# %%
nr_parsed = 0
nr_cleansed = 0
nr_removed = 0

with open( (output_path / "cc_news_cleansed.txt"), 'w', encoding='utf-8') as ff_out, \
     open( (output_path / "cc_news_cleansed_removed.txt"), 'w', encoding='utf-8') as ff_out_removed:
    tqdm_looper = tqdm(json_file_list)
    for file in tqdm_looper:
        nr_parsed += 1
        with open(file, 'r', encoding='utf-8') as ff:
            json_in = json.load(ff)
            language = json_in['language']
            if language != 'en':
                print(f"Found no english text in {file}")
                
            title_text = json_in['title']
            if title_text is not None:
                title_text = title_text.strip()
                title_text = common.cleanse_text_chars(title_text)
            
            description_text = json_in['description']
            if description_text is not None:
                description_text = description_text.strip()
                description_text = common.cleanse_text_chars(description_text)
            
            main_text = json_in['maintext']
            if main_text is not None:
                main_text = main_text.strip()
                main_text = common.cleanse_text_chars(main_text)

            if language != 'en':
                nr_removed += 1
                ff_out_removed.write('\n\n\n')
                ff_out_removed.write(f'Removed text because lanuage not English en but {language}')
                ff_out_removed.write('\n' + '\n'.join([str(s) for s in [title_text, description_text, main_text]]))
            elif title_text is None:
                nr_removed += 1
                ff_out_removed.write('\n\n\n')
                ff_out_removed.write(f'Removed text because title is missing')
                ff_out_removed.write('\n' + '\n'.join([str(s) for s in [title_text, description_text, main_text]]))
            elif main_text is None:
                nr_removed += 1
                ff_out_removed.write('\n\n\n')
                ff_out_removed.write(f'Removed text because maintext is missing')
                ff_out_removed.write('\n' + '\n'.join([str(s) for s in [title_text, description_text, main_text]]))
            elif len(main_text) < 20:
                nr_removed += 1
                ff_out_removed.write('\n\n\n')
                ff_out_removed.write(f'Removed text because maintext has less than 20 characters')
                ff_out_removed.write('\n' + '\n'.join([str(s) for s in [title_text, description_text, main_text]]))
            else:
                nr_cleansed += 1
                # Output text to proper file
                ff_out.write("[DOC]\n")
                output_list = [o for o in [title_text, description_text, main_text] if o is not None]
                ff_out.write('\n\n'.join([str(s) for s in output_list]))
                ff_out.write('\n\n')
        # time.sleep(1)
        tqdm_looper.set_description(f"removed: {nr_removed:_}")
