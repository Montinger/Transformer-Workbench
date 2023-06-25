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
import time
import html
# import mwxml
import pathlib

import pandas as pd

import xml.etree.ElementTree as ET

from lxml import etree

import common

# %% [markdown]
# some useful stuff taken from here:
# https://www.heatonresearch.com/2017/03/03/python-basic-wikipedia-parsing.html

# %% [markdown]
# Determine the number of different section counts. Note that these command take a long time to run
#
# ~~~~~~bash
# bzgrep -P "^\=+ .*? \=+" dewiki-20230101-pages-articles-multistream.xml.bz2 > dewiki_section_headings.txt
# bzgrep -P "^\=+ .*? \=+" enwiki-20230101-pages-articles-multistream.xml.bz2 > enwiki_section_headings.txt
# ~~~~~~

# %%
main_input_path = pathlib.Path("/data_ssd/tmp_text_preparation/Wiki-January-2023")

output_path = main_input_path / 'cleansed-parsed'
output_path.mkdir(parents=False, exist_ok=True)


# %%
class PageCollector:
    
    def __init__(self, max_print_length=200, filestream_cleansed_out=None, filestream_rm_sects_out=None, filestream_rm_pages_out=None):
        """A little helper class to process each page
        """
        self.clear()
        self.max_print_length = max_print_length
        self.filestream_cleansed_out = filestream_cleansed_out
        self.filestream_rm_sects_out = filestream_rm_sects_out
        self.filestream_rm_pages_out = filestream_rm_pages_out
        
    def clear(self):
        """clear all to receive a new page"""
        self.exclude_page = False
        self.exclusion_info = None
        self.removed_text = None
        self.title = ""
        self.text = ""
        self.id = None
        
    def write_page(self):
        """Writes the page to the filestreams. Regular text is outputed to regular text. Removed to bz2"""
        if self.exclude_page:
            # output to removed pages stream
            if self.filestream_rm_pages_out is not None:
                self.filestream_rm_pages_out.write( ('\n[DOC]\n'+self.title+'\n\n'+self.text +'\n').encode() )
        else:
            if (self.removed_text is not None) and (len(self.removed_text) > 0):
                # output to removed sections stream
                if self.filestream_rm_sects_out is not None:
                    self.filestream_rm_sects_out.write( ('\n[DOC]\n'+self.title+'\n\n'+self.removed_text +'\n').encode() )
                
            # output regular text:
            if self.filestream_cleansed_out is not None and len(self.text) > 100:
                self.filestream_cleansed_out.write( ('\n[DOC]\n'+self.title+'\n\n'+self.text +'\n') )
    
        
    def cleanse(self):
        """apply the cleansing to the page and title"""
        self.exclude_page, self.exclusion_info, self.title, self.text, self.removed_text = common.cleanse_wiki_page(self.title, self.text, wiki_source=True)
        
    def print_page(self, title_only=True):
        # print(f"page {self.id} title: {self.title}")
        if not title_only:
            print(self.text[:self.max_print_length])


# %%
def strip_tag_name(t):
    t = elem.tag
    idx = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t


# %%
all_other_wiki_lists = [
    "enwikisource-20230101-pages-articles-multistream.xml.bz2",
    "enwiktionary-20230101-pages-articles-multistream.xml.bz2",
    "enwikibooks-20230101-pages-articles-multistream.xml.bz2",
    "enwikiversity-20230101-pages-articles-multistream.xml.bz2",
]

for file_name in all_other_wiki_lists:


    test_file = (main_input_path / file_name)

    test_file_out          = output_path / (test_file.stem.split('-')[0] + '-' + test_file.stem.split('-')[1] + '.txt')
    test_file_rm_sects_out = output_path / (test_file.stem.split('-')[0] + '-' + test_file.stem.split('-')[1] + '-removed-sections.txt.bz2')
    test_file_rm_pages_out = output_path / (test_file.stem.split('-')[0] + '-' + test_file.stem.split('-')[1] + '-removed-pages.txt.bz2')

    print(f"test_file:              {test_file}")
    print(f"test_file_out:          {test_file_out}")
    print(f"test_file_rm_sects_out: {test_file_rm_sects_out}")
    print(f"test_file_rm_pages_out: {test_file_rm_pages_out}")

    with bz2.open(test_file, mode='r') as ff_in, open(test_file_out, mode='w', encoding='utf-8') as ff_out, bz2.open(test_file_rm_pages_out, mode='w') as ff_rm_pages_out, bz2.open(test_file_rm_sects_out, mode='w') as ff_rm_sects_out:
        tree = etree.iterparse(ff_in, events=['start', 'end'])

        is_in_page = False
        is_in_revision = False

        page = PageCollector(filestream_cleansed_out=ff_out, filestream_rm_sects_out=ff_rm_sects_out, filestream_rm_pages_out=ff_rm_pages_out)

        for event, elem in tree:
            simple_tag = strip_tag_name(elem.tag)

            if event=='start':
                if simple_tag=='page':
                    is_in_page = True
                elif simple_tag=='revision':
                    is_in_revision = True

            if event=='end':
                #print(f"for event: {event}")
                #print(f"\t tag: {elem.tag}")
                #print(f"\t simple_tag: {simple_tag}")
                #print(f"\t is_in_page: {is_in_page}")
                #print(f"\t is_in_revision: {is_in_revision}")

                if simple_tag=='page':
                    is_in_page = False
                    page.cleanse()
                    page.print_page()
                    page.write_page()
                    page.clear()
                    elem.clear()
                else: # elif is_in_page:
                    if simple_tag=='title':
                        page.title = elem.text
                    elif simple_tag=='id' and not is_in_revision:
                        page.id = int(elem.text)
                    elif simple_tag=='text':
                        page.text = elem.text
                    elif simple_tag=='revision':
                        is_in_revision = False

                    # sub_elem = etree.SubElement(elem, "id") -> for some reason cannot access text there
                    #print(str(sub_elem))
                    #print(etree.tostring(sub_elem, method = "text"))
                    # print(elem.findtext('title', default = 'None'))
                    # for sub_elem in elem:
                    #    simple_sub_tag = strip_tag_name(sub_elem.tag)
                    #    print(simple_sub_tag)
                    # print(elem.xpath('//title/text()'))
                    #print(elem[0].text)
                    #page.title = etree.SubElement(elem, "title").text
                    #page.id = etree.SubElement(elem, "id").text
                    #page.text = etree.SubElement(elem, "text").text
                    #page.print_page()


# %% [markdown]
# ## Cleanse texts further. Especially wiki-source

# %%
test_file = (output_path / "enwikisource-20230101.txt")
out_file = (output_path / "enwikisource-20230101-cleansed.txt")

# %%
remove_sub_lists = [
    r"^:",
    r"^verse=",
    r"\<section.*/\>",
    r"^Category:.*$",
    r"^\[\[Category:.*\]$",
    r"^\[\[.*?\]$",
    r"^\{\{.*?\}$",
    r"\<br\>",
    r'\<br /\>',
    r"\<tr\>",
    r"\</tr\>",
    r"\<caption.*\</caption\>",
    r"\<th\.*\</th\>",
    r"\<TABLE.*?\>",
    r"\<div.*?\</div\>",
]
    

with open(test_file, 'r') as ff_in, open(out_file, 'w', encoding='utf-8') as ff_out:
    for text in ff_in:
        text_in = text
        text = text.replace('<br>', '\n').replace('<br />', '\n')
        for replace_part in remove_sub_lists:
            text = re.sub(replace_part, '', text, flags=re.IGNORECASE)
        text = text.strip()
        if text_in == text:                                                 
            ff_out.write(text+'\n')
        elif text is not None:
            ff_out.write(text+'\n')
            
