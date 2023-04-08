import re
import nltk
import torch
import pathlib
import unicodedata
import torchmetrics


def get_lr(optimizer):
    """
    Get the learning rate of an optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The PyTorch optimizer instance whose learning rate needs to be retrieved.

    Returns
    -------
    float
        The learning rate of the optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


# python mapping dicts
# The following mapping dicts are used to cleanse texts later
# They are used by helper functions further below
remap_dict_1 = {
    '„ ' : '"', # fix non-aligned beginnings
    ' “' : '"', # fix non-aligned beginnings
    '\u0093' : '"',
    '\u0094' : '"',
    '\u0097' : ' ',
    ' “' : '"', # fix non-aligned beginnings
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
}

filter_unicode_ranges_dict = {
    'chinese'    : "\u4e00-\u9fff",
    'japanese'   : "\u3040-\u309f", # Hiragana
    'japanese2'  : "\u30a0-\u30ff", # Hiragana
    'cyrillic'   : "\u0400-\u04ff",
    'devanagari' : "\u0900-\u0954", # hindi is here
    'korean1'    : "\uac00-\ud7a3",
    'korean2'    : "\u1100-\u11ff",
    'korean3'    : "\u3130-\u318f",
    'korean4'    : "\ua960-\ua97f",
    'korean5'    : "\ud7b0-\ud7ff",
    'malayalam'  : "\u0d00-\u0d7f",
    'arabic1'    : "\u0600-\u06ff",
    'arabic2'    : "\u0750-\u077f",
    'arabic3'    : "\u0870-\u089f",
    'arabic4'    : "\u08a0-\u08ff",
    'arabic5'    : "\ufb50-\ufdff",
    'arabic6'    : "\ufe70-\ufeff",
    'hebrew'     : "\u0590-\u05ff",
    'ethiopic'   : "\u1200-\u137f",
    'chinese1'   : "\u4e00-\u4fff",
    'chinese2'   : "\u5000-\u57ff",
    'chinese3'   : "\u5800-\u5fff",
    'chinese4'   : "\u6000-\u67ff",
    'chinese5'   : "\u6800-\u6fff",
    'chinese6'   : "\u7000-\u77ff",
    'chinese7'   : "\u7800-\u7fff",
    'chinese8'   : "\u8000-\u87ff",
    'chinese9'   : "\u8800-\u8fff",
    'chinese10'  : "\u9000-\u97ff",
    'chinese11'  : "\u9800-\u9fff",
    'chinese12'  : "\u3100-\u312f",
    'chinese13'  : "\u31a0-\u31bf",
    'glagolitic1': "\u2c00-\u2c5f",
    'bengali'    : "\u0980-\u09ff",
    'telugu'     : "\u0c00-\u0c7f",
    'rumi'       : "\U00010e60-\U00010e7e",
    'arabic7'    : "\U00010ec0-\U00010eff",
    'indic'      : "\U0001ec70-\U0001ecbf",
    'ottoman'    : "\U0001ed00-\U0001ed4f",
    'arabic-m'   : "\U0001ee00-\U0001eeff",
    'glagolitic2': "\U0001e000-\U0001e02f",

}



def remove_unicode_range(text, unicode_range):
    """
    Remove a specified Unicode range from a given text and replace it with a placeholder.

    Parameters
    ----------
    text : str
        The input text from which the specified Unicode range needs to be removed.
    unicode_range : str
        The range of Unicode characters to remove from the input text.

    Returns
    -------
    str
        The text with the specified Unicode range removed and replaced with the placeholder '\[UNK\]'.

    Example
    -------
    >>> sample = 'I am from 美国。We should be friends. 朋友。'
    >>> remove_unicode_range(sample, '\u4e00-\u9fff')
    'I am from \[UNK\]We should be friends. \[UNK\]'
    """
    # print(unicode_range.encode('unicode-escape'), str) # for debugging
    return re.sub(rf'[{unicode_range}]+', '\[UNK\]', text, flags=re.U)



def cleanse_text(text, remove_unicode=True):
    """
    Cleans a text, removing special characters and optionally Unicode characters.

    Parameters
    ----------
    text : str
        The input text to be cleaned.
    remove_unicode : bool, optional
        Whether to remove Unicode characters from the text (default is True).

    Returns
    -------
    str
        The cleaned text.
    """
    for old, new in remap_dict_1.items():
        text = text.replace(old, new)
    for old, new in remap_dict_2.items():
        text = text.replace(old, new)
    if remove_unicode:
        for key, unicode_range in filter_unicode_ranges_dict.items():
            text = remove_unicode_range(text, unicode_range)

    # remove double spaces
    while text.find('  ') >= 0:
        text = text.replace('  ', ' ').replace('  ', ' ')

    return text


def cleans_test_text(text):
    """
    Cleans the weird format for the test set to be like the other datasets by removing
    additional HTML-like tags.

    Parameters
    ----------
    text : str
        The input text to be cleaned.

    Returns
    -------
    str
        The cleaned text with HTML-like tags removed.
    """
    text = text.replace('</seg>', '')
    text = text.replace('</doc>', '')
    text = text.replace('</refset>', '')
    text = re.sub(rf'<seg id="[0-9]+">', '', text, flags=re.U)
    text = re.sub(rf'<doc .+?>', '', text, flags=re.U)
    text = re.sub(rf'<refset .+?>', '', text, flags=re.U)
    while text.find('\n\n') >= 0:
        text = text.replace('\n\n', '\n')
    return text


def count_parameters(model):
    """
    Counts the number of trainable parameters of a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model for which the number of trainable parameters will be counted.

    Returns
    -------
    int
        The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def calc_bleu_score(target_actual, target_predicted, debug=False):
    """
    Calculates the sentence BLEU score between two strings.

    Parameters
    ----------
    target_actual : str
        The actual target text.
    target_predicted : str
        The predicted target text.
    debug : bool, optional
        Whether to print debugging information (default is False).

    Returns
    -------
    float
        The BLEU score between the actual and predicted target texts.
    """
    nltk_tgt_pred_sentences = nltk.tokenize.sent_tokenize(target_predicted)
    nltk_tgt_actl_sentences = nltk.tokenize.sent_tokenize(target_actual)
    nltk_tgt_pred_sent_words = [  nltk.tokenize.word_tokenize(sent)  for sent in nltk_tgt_pred_sentences ]
    nltk_tgt_actl_sent_words = [ [nltk.tokenize.word_tokenize(sent)] for sent in nltk_tgt_actl_sentences ] # reference has to be wrapped in list, as you can provide multiple per sentence

    max_sentences = max(len(nltk_tgt_pred_sent_words), len(nltk_tgt_actl_sent_words))
    while len(nltk_tgt_pred_sent_words) < max_sentences:
        nltk_tgt_pred_sent_words.append(['.'])
    while len(nltk_tgt_actl_sent_words) < max_sentences:
        nltk_tgt_actl_sent_words.append([['.']])
    corpus_blue_score = nltk.translate.bleu_score.corpus_bleu(
        list_of_references = nltk_tgt_actl_sent_words,
        hypotheses = nltk_tgt_pred_sent_words
    )

    if debug:
        print(f"nltk_tgt_pred_sentences : {nltk_tgt_pred_sentences}")
        print(f"nltk_tgt_actl_sentences : {nltk_tgt_actl_sentences}")
        print(f"nltk_tgt_pred_sent_words : {nltk_tgt_pred_sent_words}")
        print(f"nltk_tgt_actl_sent_words : {nltk_tgt_actl_sent_words}")
        print(f"corpus_blue_score : {corpus_blue_score}")

    return corpus_blue_score


def calc_bleu_score_torchmetrics(target_actual, target_predicted):
    """
    Calculates the BLEU score, once with regular smoothed BLEU, and one sacreBLEU metrics.

    Parameters
    ----------
    target_actual : str
        The actual target text.
    target_predicted : str
        The predicted target text.

    Returns
    -------
    tuple of float
        The BLEU score calculated with torchmetrics.BLEUScore and torchmetrics.SacreBLEUScore.
    """
    bleu = torchmetrics.BLEUScore(smooth=True)
    sacrebleu = torchmetrics.SacreBLEUScore(smooth=True, tokenize='intl', lowercase=True)
    preds = [target_predicted]
    target = [[ target_actual]]
    return float(bleu(preds, target).numpy()), float(sacrebleu(preds, target).numpy())



def translate_text_greedy(text, model, tokenizer, add_tokens=50, debug=False, device="cpu", same_shape=False):
    """
    Greedily translates a text.

    This function translates a given input text using a specified PyTorch model and tokenizer
    from the Hugging Face tokenizers library with greedy search. Greedy search always selects
    the highest probability token at each decoding step.

    Parameters
    ----------
    text : str
        The input text to be translated.
    model : torch.nn.Module
        The PyTorch model used for translation.
    tokenizer : tokenizers.Tokenizer
        The tokenizer from the Hugging Face tokenizers library used for encoding and decoding
        the input and output text.
    add_tokens : int, optional, default=50
        The number of additional tokens to the source tokens.
    debug : bool, optional, default=False
        If True, print debugging information during the greedy search.
    device : str, optional, default="cpu"
        The device on which to run the model, e.g., "cpu" or "cuda".
    same_shape : bool, optional, default=False
        If True, make the source and target the same size.

    Returns
    -------
    str
        The translated text.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        text = text.strip()

        end_token = tokenizer.token_to_id("[END]")

        src_tokens = tokenizer.encode("[START]"+text+"[END]")
        src_token_ids = src_tokens.ids
        if same_shape:
            for _ in range(add_tokens):
                src_token_ids.append(tokenizer.token_to_id("[PAD]"))
        src_token_ids = torch.tensor(src_token_ids, dtype=torch.int).unsqueeze(0).to(device)

        if same_shape:
            predicted_tgt_token_ids = torch.ones((1, (src_token_ids.size(1)) ), dtype=torch.long)
        else:
            predicted_tgt_token_ids = torch.ones((1, (src_token_ids.size(1)+add_tokens) ), dtype=torch.long)
        predicted_tgt_token_ids[0][0] = tokenizer.token_to_id("[START]")
        predicted_tgt_token_ids.to(device)

        if debug:
            print('src_token_ids')
            print(src_token_ids)
            print(f"decoded: {tokenizer.decode(src_token_ids[0].tolist())}")
            print('-'*20)
            print('predicted_tgt_token_ids')
            print(predicted_tgt_token_ids)
            print('-'*20)

        for idx in range(0, predicted_tgt_token_ids.size(1)-1):
            ## print(f"running loop for idx {idx}")
            tmp_tgt_tokens = predicted_tgt_token_ids.clone().to(device)
            tgt_predicted_dist = model(src_token_ids, tmp_tgt_tokens)

            tgt_predicted_dist = torch.nn.functional.softmax(tgt_predicted_dist, dim=-1) # softmax not strictly needed for greedy but maps to nice probability
            tgt_predicted = tgt_predicted_dist.argmax(-1)

            if debug:
                tgt_top_k_probs, tgt_top_k_idx = torch.topk(tgt_predicted_dist[0], k=10, dim=-1)
                for prob, token in zip(tgt_top_k_probs[idx], tgt_top_k_idx[idx]):
                    print(f"{token} -- {tokenizer.decode([int(token)])} -- {prob:.2%}")
                print(tokenizer.decode([int(tgt_predicted[0][idx])]))

            predicted_tgt_token_ids[0][idx+1] = int(tgt_predicted[0][idx])

            # break the prediction if end token is predicted
            if int(tgt_predicted[0][idx]) == end_token:
                break

        if debug:
            print(predicted_tgt_token_ids)
        translated_text = tokenizer.decode(predicted_tgt_token_ids[0].tolist())

        return translated_text



def translate_text_beam_search(text, model, tokenizer, beam_size=10, add_tokens=50, debug=False, device="cpu"):
    """
    Translates a given input text using a specified PyTorch model and tokenizer from the Hugging Face tokenizers library with beam search.

    Beam search is a search algorithm that explores the most likely translations by keeping a fixed number of alternative hypotheses, or "beams," at each step. At each decoding step, the algorithm considers the top `beam_size` candidates from the previous step, scores each of them by predicting the next token, and then selects the top `beam_size` hypotheses for the next step. The process is repeated until the desired output length or an end token is reached. Beam search helps to mitigate the risk of generating suboptimal translations by considering multiple hypotheses, which can lead to better translations compared to greedy search, which always chooses the highest probability token at each step.

    Parameters
    ----------
    text : str
        The input text to be translated.
    model : torch.nn.Module
        The PyTorch model used for translation.
    tokenizer : tokenizers.Tokenizer
        The tokenizer from the Hugging Face tokenizers library used for encoding and decoding the input and output text.
    beam_size : int, optional, default=10
        The number of alternative hypotheses to keep at each step of the beam search.
    add_tokens : int, optional, default=50
        The number of additional tokens to the source tokens.
    debug : bool, optional, default=False
        If True, print debugging information during the beam search.
    device : str, optional, default="cpu"
        The device on which to run the model, e.g., "cpu" or "cuda".

    Returns
    -------
    str
        The translated text.

    Example
    -------
    >>> import torch
    >>> from tokenizers import Tokenizer
    >>> model = torch.load("path/to/pytorch/model")
    >>> tokenizer = Tokenizer.from_file("path/to/tokenizer.json")
    >>> input_text = "Translate this text to French."
    >>> translated_text = translate_text_beam_search(input_text, model, tokenizer, beam_size=5, add_tokens=10, debug=False, device="cpu")
    >>> print(translated_text)
    'Traduisez ce texte en français.'
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        text = text.strip()

        end_token = tokenizer.token_to_id("[END]")

        src_tokens = tokenizer.encode("[START]"+text+"[END]")
        src_token_ids = src_tokens.ids
        src_token_ids = torch.tensor(src_token_ids, dtype=torch.int).unsqueeze(0).to(device)

        predicted_tgt_token_ids = torch.ones((1, (src_token_ids.size(1)+add_tokens) ), dtype=torch.long)
        predicted_tgt_token_ids[:][0] = tokenizer.token_to_id("[START]")
        predicted_tgt_token_ids.to(device)

        beam = [{'sequence': [tokenizer.token_to_id("[START]")], 'score': 0.0}]

        for idx in range(0, (src_token_ids.size(1)+add_tokens)):
            candidates = []

            for j, hypothesis in enumerate(beam):
                # If the hypothesis ends with the end token, it is complete
                if hypothesis['sequence'][-1] == end_token:
                    candidates.append(hypothesis)
                    continue

                # Decode the last token in the hypothesis sequence
                decoder_input = torch.LongTensor(hypothesis['sequence']).unsqueeze(0).to(device)
                if debug: print(f"decoder_input: {decoder_input}")

                tgt_predicted_dist = model(src_token_ids, decoder_input)
                if debug: print(f"tgt_predicted_dist.shape: {tgt_predicted_dist.shape}")
                log_probs = tgt_predicted_dist.log_softmax(dim=-1)
                if debug: print(f"log_probs: {log_probs}")
                if debug: print(f"log_probs-red: {log_probs[0][idx]}")

                # Compute the scores of the next tokens
                next_scores = hypothesis['score'] + log_probs[0][idx]
                if debug: print(f"next_scores: {next_scores}")

                # Collect the top beam_size candidates
                top_next_scores, top_next_tokens = next_scores.topk(beam_size)
                if debug: print(f"top_next_tokens: {top_next_tokens}")
                if debug: print(hypothesis['sequence'] + [top_next_tokens[1].item()])
                for k in range(beam_size):
                    candidate = {
                        'sequence': hypothesis['sequence'] + [top_next_tokens[k].item()],
                        'score': top_next_scores[k].item(),
                    }
                    candidates.append(candidate)

            # Select the top beam_size candidates
            beam = sorted(candidates, key=lambda x: x['score'], reverse=True)[:beam_size]
            if debug: print(f"beam: {beam}")
        # Return the top-scoring hypothesis
        top_hypothesis = max(beam, key=lambda x: x['score'])
        translated_text = tokenizer.decode(top_hypothesis['sequence'])

        return translated_text


def trafos_to_text_before_bleu(text):
    """Does some transformations to the text before BLEU, which appear to always be done in the literature.
    i.e. 'ABC-DEF' is split into 'ABC ##AT##-##AT## DEF' before the BLEU is assessed.
    see also the way seq2seq handles this: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/get_ende_bleu.sh
    """
    return re.sub("(\S-\S)", lambda x: x.group(1).replace('-', " ##AT##-##AT## "), text, flags=re.IGNORECASE)


def tokenize_sentences(text):
    """
    Tokenizes a text into sentences using regular expressions.

    This function splits the input text into sentences, considering special cases like
    abbreviations and initials. It is used in the microservice implementation to translate
    a text sentence by sentence for longer sequences.

    Additionally a split occurs if there are two line breaks

    Parameters
    ----------
    text : str
        The input text to be tokenized into sentences.

    Returns
    -------
    list of str
        The tokenized sentences.

    Example
    -------
    >>> text = "Dr. Smith went to the park. She saw a dog! The dog's name was Max. What a beautiful day."
    >>> sentences = tokenize_sentences(text)
    >>> print(sentences)
    ['Dr. Smith went to the park.', 'She saw a dog!', "The dog's name was Max.", 'What a beautiful day.']
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s|(\n{2,})', text.replace('\r', ''))
    sentences = [s for s in sentences if s is not None and s.strip() != ''] # remove empty by double line break rule
    sentences = [s.strip() for s in sentences]
    return sentences
