import re
import html
import warnings

# python mapping dict
remap_dict_1 = {
    '„' : '"', # fix non-aligned beginnings -> no, don't because it screwed up more than it helped
    '“' : '"', # fix non-aligned beginnings
    '\u00a0' : ' ', # non-breaking white space
    '\u202f' : ' ', # narrow non-breaking white space
    'Ã¶' : 'ö', # german oe
    'Ã¼' : 'ü', # german ue
    'Ã¤' : 'ä', # german ae
}

remap_dict_2 = {
    '„'  : '"',
    '“'  : '"',
    '‟'  : '"',
    '”'  : '"',
    '″'  : '"',
    '‶'  : '"',
    '”'  : '"',
    '‹'  : '"',
    '›'  : '"',
    '’'  : "'",
    '′'  : "'",
    '′'  : "'",
    '‛'  : "'",
    '‘'  : "'",
    '`'  : "'",
    '–'  : '--',
    '‐'  : '-',
    '»'  : '"',
    '«'  : '"',
    '≪'  : '"',
    '≫'  : '"',
    '》' : '"',
    '《' : '"',
    '？' : '?',
    '！' : '!',
    '…'  : ' ... ',
    '\t' : ' ',
    '。' : '.', # chinese period
    '︰' : ':',
    '〜' : '~',
    '；' : ';',
    '）' : ')',
    '（' : '(',
    'ﬂ'  : 'fl', # small ligature characters
    'ﬁ'  : 'fi',
    '¶'  : ' ',
    chr(8211) : chr(45),
    '—'  : '-', #hypehen to normal minus
}


def cleanse_text_chars(text):
    """Cleans a text, removing special characters etc."""
    for old, new in remap_dict_1.items():
        text = text.replace(old, new)
    for old, new in remap_dict_2.items():
        text = text.replace(old, new)

    # text = unicodedata.normalize('NFKD', text) -> does not work well

    # remove double spaces
    while text.find('  ') >= 0:
        text = text.replace('  ', ' ').replace('  ', ' ')

    return text

def check_remove_p_ref(text):
    """Checks if a p= occurs in a string and returns empty string if it does.
    Update: also remove alt="""
    if "p=" in text or "alt=" in text or "pages=" in text or "page=" in text or "bgcolor=" in text or "link=" in text or "group=" in text or "loc=" in text or "label=" in text or "language=" in text or "lang=" in text or "adj=" in text or '=' in text:
        # last rule beats them all. Test it then you could remove rest
        return ''
    else:
        return text

def remove_cite(text):
    """Removes some citation stuff"""
    if "{{Cite " in text:
        return ''
    elif '[[plural' in text:
        return text.split('|')[0] # e.g. '''Analysis''' ([[plural|{{sc|pl}}]]: '''analyses''')
    else:
        return text

def cleanse_wiki_page_text(text, wiki_source=False):
    # some html comments are included in the text, e.g.
    # <!-- Important! Strive to explain how anarchists perceive authority and oppression and why they reject them. Jun (2019), p. 41. -->
    text = re.sub(r"<!--.*?-->", '', text)

    # cleanse the [[]] things, e.g.  "die bisherige [[Lufthansa Cityline|CityLine]] Canadair Simulator und"
    # some are file refs, there we select the alt text and should remove the {{}}, e.g.
    # [[File:Comedic Wet Cat Food sign in an ASDA supermarket.jpg|thumb|Which is wet: the food, or the cat?]]{{sometimes|this|stuff}}
    text = re.sub(r"\"\{\{.*?\}\}", '"', text) # e.g.  anarchism.&quot;{{sfn|Morris|2015|p=64}}|group=nb}}
    text = re.sub(r"\[\[.*?\]\]\]\]", lambda m: check_remove_p_ref(remove_cite(m[0]).split('|')[-1].replace(']]', '').replace(']]', '').replace('[[', '')), text)
    text = re.sub(r"\[\[.*?\]\]", lambda m: check_remove_p_ref(remove_cite(m[0]).split('|')[-1].replace(']]', '').replace(']]', '').replace('[[', '')), text)
    # this can also occur with {, e.g. "{{chem|39|Ar}}"
    text = re.sub(r"\{\{.*?\}\}\}\}", lambda m: check_remove_p_ref(remove_cite(m[0]).split('|')[-1].replace('}}', '').replace('}}', '').replace('{{', '')), text)
    text = re.sub(r"\{\{.*?\}\}", lambda m: check_remove_p_ref(remove_cite(m[0]).split('|')[-1].replace('}}', '').replace('}}', '').replace('{{', '')), text)
    # some look like this # {{cite web|url=http://news.nationalgeographic.com/news/2007/03/070316-robot-ethics.html |title=Robot Code of Ethics to Prevent Android Abuse, Protect Humans |publisher=News.nationalgeographic.com |date=28 October 2010 |access-date=22 November 2011}}
    # maybe we should add a special rule for them. most of these should occur in the web references though

    # put section headings to markdown format (for language model)
    # text = re.sub(r"^\=+.*?\=+", lambda m: ('#' * (m[0].count('=')//2)) + ' ' + m[0].replace('=', '').strip(), text, flags=re.MULTILINE)
    # or remove them for masked language model
    text = re.sub(r"^\=+.*?\=+", lambda m: (m[0].replace('=', '').strip()), text, flags=re.MULTILINE)
    # in wikibooks etc there are some headings without spaces, i.e. ==My Heading==

    # replace multiple (most are double) '' with underscore -> most of them are emphasis, and we want everything in markdown style
    for i in range(10, 1, -1):
        # text = text.replace( ("'"*i), '_') # for markdown language models
        text = text.replace( ("'"*i), '') # for masked lm

    text = cleanse_text_chars(text)

    # remove the long {{ }} parts over multiple lines (usualy these are tables)
    text = re.sub(r"<!--.*?-->", '', text, flags=re.DOTALL)
    text = re.sub(r"\<ref.*?\</ref\>", '', text, flags=re.DOTALL)
    text = re.sub(r"\<ref .*?\>", '', text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", '', text, flags=re.DOTALL)
    text = re.sub(r"\[\[.*?\]\]", '', text, flags=re.DOTALL)
    text = re.sub(r"\[\[Category:.*?\]", '', text, flags=re.DOTALL)
    text = re.sub(r"\{\|.*?\|\}", '', text, flags=re.DOTALL)
    text = re.sub(r"\<Placemark.*?\</Placemark\>", '', text, flags=re.DOTALL)
    text = re.sub(r"\<Schema.*?\</Schema\>", '', text, flags=re.DOTALL)
    text = re.sub(r"\<placemark\.*?\</placemark\>", '', text, flags=re.DOTALL)
    text = re.sub(r"\<schema.*?\</schema\>", '', text, flags=re.DOTALL)
    text = re.sub(r"\<timeline.*?\</timeline\>", '', text, flags=re.DOTALL)
    text = re.sub(r"\<noinclude\>.*?\</noinclude\>", '', text, flags=re.DOTALL)
    text = re.sub(r"\<gal{1,2}ery.*?\</gal{1,2}ery\>", '', text, flags=re.DOTALL)
    text = re.sub(r"\<Gal{1,2}ery.*?\</Gal{1,2}ery\>", '', text, flags=re.DOTALL)
    text = re.sub(r"^\|", '', text, flags=re.MULTILINE)
    if wiki_source:
        text = re.sub(r"^:", '', text, flags=re.MULTILINE)
    else:
        text = re.sub(r"^:.*?$", '', text, flags=re.MULTILINE) # -> you probably don't want this for wiki source
    text = re.sub(r"^\{\{.*?\}$", '', text, flags=re.MULTILINE)
    text = re.sub(r"\.date\=.*?$", '', text, flags=re.MULTILINE)
    text = re.sub(r'style=".*?"', '', text, flags=re.DOTALL)
    text = re.sub(r'^colwidth=.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[0-9]{1,3}em$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\*\s*?$', '', text, flags=re.MULTILINE) #  removes empty lists with *
    text = re.sub(r'^-\s*?$', '', text, flags=re.MULTILINE)
    text = text.replace('</ref>}}', '')
    text = text.replace('</ref>', '')
    text = text.replace(']]', '').replace('[[', '')
    text = re.sub(r'^\}\}\s*?$', '', text, flags=re.MULTILINE) # removy empty lines which only start with }}
    text = re.sub(r'^colend\s*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^source=\s*?$', '', text, flags=re.MULTILINE)
    text = text.replace('<blockquote>', '').replace('</blockquote>', '').replace('<poem>', '').replace('</poem>', '')
    text = text.replace('<blockquote >', '').replace('</blockquote >', '').replace('<poem >', '').replace('</poem >', '')
    text = re.sub(r'\<span .*?\</span\>', '', text, flags=re.MULTILINE) # e.g. <span class"anchor" id="Summer annuals"></span>
    text = re.sub(r'={2,5}$', '', text, flags=re.MULTILINE)
    text = text.replace('<br>', '').replace('<br />', '')

    text = re.sub(r"^Category:.*?$", '', text, flags=re.MULTILINE)
    text = re.sub(r"^File:.*?$", '', text, flags=re.MULTILINE)
    text = re.sub(r'^\}\}\s*?$', '', text, flags=re.MULTILINE) # removy empty lines which only start with }}
    # text = text.replace('}}', '').replace('{{', '')

    while text.find('\n\n\n\n') > -1:
        text = text.replace('\n\n\n\n', '\n\n\n')

    return text




remove_titles_list = ["list of ", "liste "]
# below identified with: bzgrep -i -P "<text .*?>#" dewiki-20230101-pages-articles-multistream.xml.bz2
remove_redirect_pages = ['#redirect', '#weiterleitung']
remove_sections_list = [
    'weblinks',
    'einzelnachweise',
    'literatur',
    'auszeichnungen',
    'quellen',
    'filmografie (auswahl)',
    'diskografie',
    'ehrungen',
    'schriften (auswahl)',
    'werke (auswahl)',
    'filmografie',
    'belege',
    'references',
    'external links',
    'licensing',
    'licensing:',
    'sources',
    'bibliography',
    'awards',
    'discography',
    'filmography',
    'honours',
    'awards and nominations',
    'notes and references',
    'awards and honors',
    'accolades',
    'redirect request',
    'lizenzvorlagen für bilder',
    'singles',
    'veröffentlichungen (auswahl)',
    'titelliste',
    'erfolge und auszeichnungen',
    'quelle',
    'nachweise',
    'auszeichnungen (auswahl)',
    'einzelbelege',
    'cited sources',
    'see also',
    'selected bibliography',
    'further reading',
    'citations',
    'works cited',
    'notes',
]

def cleanse_wiki_page(title, text, wiki_source=False):
    exclude_page = False
    exclusion_info = ""
    removed_text = ""

    if text is None or title is None:
        warnings.warn(f"Warning at title {title}. Text is None")
        text = ""
        if title is None:
            title = ""
        exclude_page = True

    else:
        text = text.strip()

        # 0. html decode special characters
        for _ in range(5):
            title = html.unescape(title)
            text = html.unescape(text)

    title = title.strip()
    title_lower = title.lower()


    # 1. Exclude if title matches removal list
    if any([s in title_lower for s in remove_titles_list]):
        exclude_page = True
        exclusion_info = "Page excluded due to title match with remove_titles_list"

    # 2. check if it is only a redirect page
    tmp_redirect_text = text[:100].lower().strip()
    if any([tmp_redirect_text.startswith(s) for s in remove_redirect_pages]):
        exclude_page = True
        exclusion_info = "Page excluded due to redirect page"

    # 3. remove all sections which match the section removal list
    if not exclude_page:
        list_of_all_sections_raw = re.findall(fr"^\=+.*?\=+", text, flags=re.MULTILINE)
        list_of_all_sections_lower = [sec.replace('=', '').strip().lower() for sec in list_of_all_sections_raw]
        list_of_section_locations = [text.find(sec) for sec in list_of_all_sections_raw]

        # Add an element for the start of the article
        list_of_all_sections_raw = ["ARTICLE_START", *list_of_all_sections_raw]
        list_of_all_sections_lower = ["ARTICLE_START", *list_of_all_sections_lower]
        list_of_section_locations = [0, *list_of_section_locations, -1] # Add start and end of article

        list_of_section_texts = [ text[ list_of_section_locations[idx]:list_of_section_locations[idx+1] ] for idx in range(len(list_of_all_sections_raw)) ]

        text_to_keep = ""
        removed_text = ""

        for idx in range(len(list_of_section_texts)):
            sec_heading = list_of_all_sections_raw[idx]
            if list_of_all_sections_lower[idx] in remove_sections_list:
                removed_text += '\n\n'+list_of_section_texts[idx]
                exclusion_info = "Some sections were removed"
            else:
                text_to_keep += '\n\n'+list_of_section_texts[idx].replace(sec_heading, sec_heading.replace('=','').strip())

        text = cleanse_wiki_page_text(text_to_keep, wiki_source=wiki_source)
        text = text.strip()

    return exclude_page, exclusion_info, title, text, removed_text
