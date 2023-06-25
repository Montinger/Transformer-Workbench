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
import io
import re
import bz2
import json
import time
import random
import pathlib
import zipfile

import pandas as pd

from tqdm.auto import trange, tqdm

import common

# %% [markdown]
# ~~~~~~~~~
# *** START OF THIS PROJECT GUTENBERG
#
#
# text here
#
# End of Project Gutenberg's The Insidious Dr. Fu-Manchu, by Sax Rohmer
#
# *** END OF THIS PROJECT GUTENBERG EBOOK THE INSIDIOUS DR. FU-MANCHU ***
#
#
#
# import io
# import zipfile
#
# with zipfile.ZipFile("files.zip") as zf:
#     with io.TextIOWrapper(zf.open("text1.txt"), encoding="utf-8") as f:
#         
# ~~~~~~~~~

# %%
main_input_path = pathlib.Path("/data_ssd/tmp_text_preparation/Gutenberg/en-txt/aleph.gutenberg.org")

output_path = pathlib.Path('/data_ssd/tmp_text_preparation/cleansed-parsed')
output_path.mkdir(parents=False, exist_ok=True)

# %%
zip_file_list = main_input_path.glob(r'**/*.zip')

zip_file_list = list(zip_file_list)
zip_file_list.sort()
# random.shuffle(json_file_list) # random shuffle
print(f"Found {len(zip_file_list)} zip_file_list files")

folder_list = [d.parent for d in zip_file_list]
# make unique
folder_list = list(set(folder_list))
print(f"in {len(folder_list)} folders")

# %% [markdown]
# Sometimes there are multiple zips in one folder.
#
# The one ending in `-0` should be the correct one in this case, as it is UTF-8.
#
# If this doesn't exist take `-8` (iso encoding with extended ASCII).

# %%
# The following list of files has a faulty encoding. We remove then, thus the ascii variant will be taken
files_with_wrong_encoding = ['1316-0.zip', '134-0.zip', '1344-0.zip', '1460-0.zip', '15700-0.zip', '16350-0.zip', '1723-0.zip', 
                             '1856-0.zip', '1939-0.zip', '22203-0.zip', '22206-0.zip', '22535-0.zip', '23060-0.zip', '26535-0.zip', 
                             '2770-0.zip', '2774-0.zip', '2994-0.zip', '3184-0.zip', '3185-0.zip', '3445-0.zip', '3446-0.zip', '3448-0.zip', 
                             '3449-0.zip', '3450-0.zip', '35589-0.zip', '37960-0.zip', '3803-0.zip', '38507-0.zip', '3926-0.zip', '39526-0.zip', 
                             '4002-0.zip', '40161-0.zip', '40815-0.zip', '4107-0.zip', '4405-0.zip', '44433-0.zip', '45385-0.zip', '462-0.zip', 
                             '4699-0.zip', '5124-0.zip', '51924-0.zip', '51936-0.zip', '51939-0.zip', '51940-0.zip', '51948-0.zip', '51951-0.zip', 
                             '51970-0.zip', '5373-0.zip', '56913-0.zip', '57334-0.zip', '57589-0.zip', '58079-0.zip', '58171-0.zip', '58172-0.zip', 
                             '58250-0.zip', '58329-0.zip', '58440-0.zip', '58572-0.zip', '58676-0.zip', '58825-0.zip', '58845-0.zip', '58993-0.zip', 
                             '59023-0.zip', '59024-0.zip', '59049-0.zip', '59194-0.zip', '59228-0.zip', '59279-0.zip', '59325-0.zip', '59391-0.zip', 
                             '59508-0.zip', '59766-0.zip', '59767-0.zip', '6191-0.zip', '6302-0.zip', '63448-0.zip', '664-0.zip', '7684-0.zip', '9255-0.zip', 
                             '9830-0.zip']

# %%
# remove the files with wrong encoding
print(f"files before wrong encoding filter: {len(zip_file_list)}")
zip_file_list = [z for z in zip_file_list if z.name not in files_with_wrong_encoding]
print(f"files after wrong encoding filter:  {len(zip_file_list)}")

# %%
nr_removed = 0
tqdm_looper = tqdm(folder_list)
for folder in tqdm_looper:
    sub_list = [z for z in zip_file_list if z.parent==folder]
    if len(sub_list) > 1:
        sub_list.sort() # after sorting the -0 always comes first, followed by -8, then the ASCII one without extension. 
        # Thus we can always take element 0 from this list and remove the other elements from the full list
        # print(f"Found for folder {folder} the files: {[s.name for s in sub_list]}")
        for e in sub_list[1:]:
            zip_file_list.remove(e)
            nr_removed += 1
    tqdm_looper.set_description(f"removed: {nr_removed:_}")

# %%
# Check that we now have the same number of documents as we have folders
print(f"After cleansing {len(zip_file_list)} zip_file_list files")
print(f"in {len(folder_list)} folders")


# %%
def cleanse_gutenberg(text, file_name=''):
    """cleanses a text like da Gutenberg would have"""
    text_lower = text.lower()
    
    start_idx_1 = text_lower.find('*** START OF THIS PROJECT GUTENBERG'.lower())
    start_idx_2 = text_lower.find('*** START OF THE PROJECT GUTENBERG'.lower())
    start_idx_3 = text_lower.find('***START OF THIS PROJECT GUTENBERG'.lower())
    start_idx_4 = text_lower.find('***START OF THE PROJECT GUTENBERG'.lower())
    start_idx_5 = text.find('*END*THE SMALL PRINT')
    start_idx_6 = text.find('*END* THE SMALL PRINT')
    start_idx_7 = text.find('*END THE SMALL PRINT')
    start_idx_8 = text_lower.find('OR FOR MEMBERSHIP.>>')
    start_list = [e for e in [
        start_idx_1, start_idx_2, start_idx_3, start_idx_4, start_idx_5, start_idx_6, start_idx_7, start_idx_8
    ] if e != -1 and e <= 25_000] # should be within first 25_000 characters, otherwise is probably a false positive
    if len(start_list) > 0:
        start_idx = min(start_list)
    else:
        start_idx = -1
    
    # if not found throw error
    end_idx_1 = text_lower.find('End of Project Gutenberg'.lower())
    end_idx_2 = text_lower.find('End of the Project Gutenberg'.lower())
    end_idx_3 = text_lower.find('End of this Project Gutenberg'.lower())
    end_idx_4 = text_lower.find('*** END OF THE PROJECT GUTENBERG'.lower())
    end_idx_5 = text_lower.find('*** END OF THIS PROJECT GUTENBERG'.lower())
    end_idx_6 = text_lower.find('***END OF THE PROJECT GUTENBERG'.lower())
    end_idx_7 = text_lower.find('***END OF THIS PROJECT GUTENBERG'.lower())
    end_idx_8 = text.find('*** END')
    end_idx_9 = text.find('***END') 
    
    end_list = [e for e in [
        end_idx_1, end_idx_2, end_idx_3, end_idx_4, end_idx_5, end_idx_6, end_idx_7, end_idx_8, end_idx_9
    ] if e != -1]
    if len(end_list) > 0:
        end_idx = min(end_list)
    else:
        end_idx = -1
    
    # If any of these is -1 you should set it to 0 for start
    #  for end -1 is correct
    if start_idx == -1:
        start_idx = 0
        print(f"Did not find start_idx for file {file_name}")
    
    if end_idx == -1:
        print(f"Did not find end_idx for file {file_name}")
    
    text = text[start_idx:end_idx]

    # The start still contains the *** *** part. Remove this via regular expression
    text = re.sub(r"\*\*\* START OF THIS PROJECT GUTENBERG.*?\*\*\*", '', text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*\* START OF THE PROJECT GUTENBERG.*?\*\*\*", '', text, flags=re.IGNORECASE)
    text = re.sub(r"\*END\*THE SMALL PRINT.*?\*END\*", '', text, flags=re.IGNORECASE)
    text = re.sub(r"\*END THE SMALL PRINT.*?\*END\*", '', text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*\* START .*?\*\*\*", '', text)
    text = text.replace('OR FOR MEMBERSHIP.>>', '')

    text = text.replace('\n\n', '***REMEMBER_DOUBLE_LINE_BREAK***').replace('\n', ' ').replace('***REMEMBER_DOUBLE_LINE_BREAK***', '\n')
    # maybe remove one \n and then double the existing ones. Ore just leave it like this then
    # replace \n by a space (otherwise you might connect words. The double spaces will be removed by the common_function
    # also apply the main function with duplication
    text = common.cleanse_text_chars(text)
    
    text = text.strip()
    
    if len(text) < 100:
        print(f"Warning at file {file_name}. Text has below 100 characters")
    
    return text


# %%
random.shuffle(zip_file_list)

# %%
# Test loading of a zip-wrapped txt file
error_reading_list = [] # keep a list of files which failed to read (due to faulty utf-8 encoding), so we can add them to the removal list above

with open( output_path / "gutenberg_cleansed.txt", 'w', encoding='utf-8' ) as ff_out:
    for file_name in tqdm(zip_file_list):
        # time.sleep(2)
        with zipfile.ZipFile(file_name) as zf:
            # print (zf.namelist())
            for inner_file in zf.namelist():
                encoding = 'ISO-8859-1' # some files without are not really ascii, so set to the iso-latin one
                if '-0.txt' in inner_file:
                    encoding = 'utf-8'
                elif '-8.txt' in inner_file:
                    encoding = 'ISO-8859-1'
                with io.TextIOWrapper(zf.open(inner_file), encoding=encoding) as f:
                    try:
                        text = f.read()
                    except:
                        print(f"Error reading file {file_name}")
                        error_reading_list.append(file_name)
                        continue
                    text = cleanse_gutenberg(text, file_name=file_name)
                    ff_out.write("[DOC]\n")
                    ff_out.write(text)
                    ff_out.write("\n")
