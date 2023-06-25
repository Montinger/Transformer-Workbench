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
test_file = (main_input_path / "enwiki-20230101-pages-articles-multistream.xml.bz2")

test_file_out          = output_path / (test_file.stem.split('-')[0] + '-' + test_file.stem.split('-')[1] + '.txt')
test_file_rm_sects_out = output_path / (test_file.stem.split('-')[0] + '-' + test_file.stem.split('-')[1] + '-removed-sections.txt.bz2')
test_file_rm_pages_out = output_path / (test_file.stem.split('-')[0] + '-' + test_file.stem.split('-')[1] + '-removed-pages.txt.bz2')

# %%
print(f"test_file:              {test_file}")
print(f"test_file_out:          {test_file_out}")
print(f"test_file_rm_sects_out: {test_file_rm_sects_out}")
print(f"test_file_rm_pages_out: {test_file_rm_pages_out}")

# %%
# Count the number of sections
if False:
    min_count = 5

    for lang in ["de", "en"]:
        count_dict = {}

        with open( (main_input_path / f"{lang}wiki_section_headings.txt"), 'r', encoding='utf-8') as ff:
            for line in ff:
                line = line.replace('=', '').strip().lower()
                count_dict[line] = count_dict.get(line, 0) + 1

        print(f"Counts for {lang} before min-count filter: {len(count_dict)}") 

        # remove all sections which only occur a certain number of times


        key_list = list(count_dict.keys()).copy()
        for key in key_list:
            if count_dict[key] <= min_count:
                del count_dict[key]

        print(f"Counts for {lang} after min-count filter: {len(count_dict)}")

        df_section_counts = pd.DataFrame({'section_name': count_dict.keys(), 'count': count_dict.values()})
        df_section_counts = df_section_counts.sort_values(by='count', ascending=False).reset_index(drop=True)
        df_section_counts.to_csv(f"df_section_counts_{lang}.csv")

    del count_dict
    del df_section_counts

# %%
# {{§|265|hgb|juris}}  -> how to process that
# or this:
# {{cite web|url=http://news.nationalgeographic.com/news/2007/03/070316-robot-ethics.html |title=Robot Code of Ethics to Prevent Android Abuse, Protect Humans |publisher=News.nationalgeographic.com |date=28 October 2010 |access-date=22 November 2011}}



# %%
totalCount = 0
articleCount = 0
redirectCount = 0
templateCount = 0
title = None
start_time = time.time()

# %% [markdown]
# don't write the page down if it starts with:
# - Liste von
# - "#REDIRECT" (first part of text)
# - "#WEITERLEITUNG" (first part of text)
#
# - clean the headers
# - clean the [[]] tags. beware that stuff like this can happen:
#
# [[Funtsch#Johann Baptist Funtsch|Johann Baptist Funtsch]]
#
# - have a look at https://pypi.org/project/pinyin/ module
# - (or just use the binary BPE tokenizer) and don't worry about it

# %%
test_text = '''Seit 1. Juli 2010 ist die bisherige [[Lufthansa Cityline|CityLine]] Canadair Simulator und Training GmbH Teil der Lufthansa Aviation Training GmbH. Dabei richtet Lufthansa Aviation [[Funtsch#Johann Baptist Funtsch|Johann Baptist Funtsch]] Training am Standort Berlin einen Schwerpunkt auf den Trainingsbedarf von Regionalflugges'''


# %%
re.sub(r"\[\[.*?\]\]", lambda m: m[0].split('|')[-1].strip(']]'), test_text)

# %%
test_text = '''
== Geschichte ==
=== Vorkoloniale Geschichte ===
=== Koloniale Herrschaft ===
=== Einseitig erklärte Unabhängigkeit ===
=== Von der international anerkannten Unabhängigkeit 1980 bis etwa 2007 ===
=== Von der Wahl 2008 bis zum Jahr 2017 ===
=== Absetzung Robert Mugabes im Jahr 2017 ===
== Politik ==
=== Politisches System ===
=== Politische Indizes ===
=== Menschenrechte ===
=== Außenpolitik ===
== Wirtschaft ==
=== Reformen ===
=== Sektoren ===
bla
=== Kennzahlen ===
yada yada
=== Währung ===
abc
=== Inflation ===
=== Staatshaushalt ===
== Infrastruktur ==
=== Verkehr ===
=== Kommunikation ===
=== Rundfunk ===
== Kultur ==
=== Denkmäler ===
==== Höhlenzeichnungen ====
==== Steinbauten ====
=== Zeitgenössische Kultur ===
==== Bildhauerei ====
==== Malerei ====
==== Literatur ====
==== Musik ====

'''

# %%
re.findall(r"^\=+ .*? \=+", test_text, flags=re.MULTILINE)

# %%
tmp_text = test_text[test_text.find("=== Sektoren ==="): test_text.find("=== Währung ===")]
test_text.replace(tmp_text, '')

# %%
re.sub(r"^\=+ .*? \=+", lambda m: ('#' * (m[0].count('=')//2)) + ' ' + m[0].replace('=', '').strip(), 
       test_text,
       flags=re.MULTILINE
)
# m[0].replace('=', '').strip(), # for testing

# %%
test_text = """was to construct a length ''x'' so that the cube of side ''x'' contained the same volume as the rectangular box ''a''&lt;sup&gt;2&lt;/sup&gt;''b'' for given sides ''a'' and ''b''. ["""


# %%
import html
html.unescape(test_text)

# %%
html.unescape(html.unescape('&lt;&amp;ndash;'))
# ndash; -> &ndash; # does this appear more often?
# recursively unescape a few times

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
        self.exclude_page, self.exclusion_info, self.title, self.text, self.removed_text = common.cleanse_wiki_page(self.title, self.text)
        
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
test_text = "hello <!-- Important! Strive to explain how anarchists perceive authority and oppression and why they reject them. Jun (2019), p. 41. -->"
re.sub(r"<!--.*?-->", '', test_text)

# %%
# Files
# [[File:Portrait of Pierre Joseph Proudhon 1865.jpg|thumb|upright|Pierre-Joseph Proudhon is the primary proponent of mutualism and influenced many future individualist anarchist and social anarchist thinkers.{{sfn|Wilbur|2019|pp=216–218}}]]


# %%
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
                
