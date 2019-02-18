#!/usr/bin/env python
# coding: utf-8



import sys
import numpy as np
import pandas
import tensorflow as tf
import collections
import six
import copy
import unicodedata
import re
from operator import itemgetter
from nltk.stem.wordnet import WordNetLemmatizer
from bert_serving.client import BertClient
from nltk.corpus import wordnet

def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    """Constructs a BasicTokenizer.

    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = self._clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text = self._tokenize_chinese_chars(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """

    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat.startswith("C"):
    return True
  return False


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False



def line2bert_service(line, target2candidates, tokenizer):
    
    instances = [] # returned input lines for bert
    
    full_target_key = line.split('\t')[0]
    target = '.'.join(full_target_key.split('.')[:2])
    index = line.split('\t')[1]
    
    candidate_words = list(target2candidates[target])
    #print(candidate_words)
    
    text_a_raw = line.split('\t')[3].split(' ') # text_a untokenized
    
    masked_id = int(line.split('\t')[2])
    
    masked_word = text_a_raw[masked_id] # get the target word
    
    #text_a_raw.insert(masked_id + 1,'and')
    #text_a_raw.insert(masked_id + 2, 'MASK')
    
    text_a_raw = ' '.join(text_a_raw)
    
    #pat_letter = re.compile(r'[^a-zA-Z \']+')
    #text_a_raw = pat_letter.sub(' ', text_a_raw).strip().lower()
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    # to find the 's following the letters
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    # to find the ' following the words ending by s
    pat_s2 = re.compile("(?<=s)\'s?")
    # to find the abbreviation of not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    # to find the abbreviation of would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    # to find the abbreviation of will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    # to find the abbreviation of am
    pat_am = re.compile("(?<=[I|i])" "\'m")
    # to find the abbreviation of are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    # to find the abbreviation of have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    new_text = pat_is.sub(r"\1 is", text_a_raw)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    text_a_raw = new_text.replace('\'', ' ')
    text_a_raw = text_a_raw.split(' ')
    while '' in text_a_raw:
        text_a_raw.remove('')
        
    masked_id = text_a_raw.index(masked_word)
    masked_word_tokenized = tokenizer.tokenize(masked_word)

    text_a_tokenized = tokenizer.tokenize(' '.join(text_a_raw))

    indexs = [i for (i,wordpiece) in enumerate(text_a_tokenized) if wordpiece == masked_word_tokenized[0]]
    for index_now in indexs:
        flag = True
        if index_now < masked_id:
            continue
        for i in range(len(masked_word_tokenized)):
            if masked_word_tokenized[i] != text_a_tokenized[index_now + i]:
                flag = False
        if flag == True:
            target_start_id = index_now
            target_end_id = index_now + len(masked_word_tokenized) - 1
            break
    
    instances.append('\t'.join([full_target_key, index, ' '.join(text_a_tokenized), str(target_start_id), str(target_end_id)]))

    text_target_masked = copy.deepcopy(text_a_tokenized)
    text_target_masked[target_start_id] = "[MASK]"
    instances.append('\t'.join([full_target_key, index, ' '.join(text_target_masked), str(offset)]))

    candidates = []

    for candidate_word in candidate_words:

        if len(candidate_word.split(' ')) != 1 or len(candidate_word.split('-')) != 1:
        #print("candidate word %s filtered" %(candidate_word))
            continue
        line_replaced = copy.deepcopy(text_a_raw)
        line_replaced[masked_id] = candidate_word
        line_replaced_tokenized = tokenizer.tokenize(' '.join(line_replaced))

        candidates.append(candidate_word)

        candidate_word_tokenized = tokenizer.tokenize(candidate_word)

        indexs = [i for i,word in enumerate(line_replaced_tokenized) if word == candidate_word_tokenized[0]]
        for index_now in indexs:
            flag = True
            if index_now < target_start_id:
                continue
            for i in range(len(candidate_word_tokenized)):
                if candidate_word_tokenized[i] != line_replaced_tokenized[index_now + i]:
                    flag = False
            if flag == True:
                candidate_start_id = index_now
                break
        candidate_end_id = candidate_start_id + len(candidate_word_tokenized) - 1       

        instances.append('\t'.join([full_target_key, index, ' '.join(line_replaced_tokenized), str(candidate_start_id), str(candidate_end_id)]))

    return instances, candidates



def line2bert_service_mask_context(line, target2candidates, tokenizer, bow_size=6):
    
    instances = [] # returned input lines for bert
    
    full_target_key = line.split('\t')[0]
    target = '.'.join(full_target_key.split('.')[:2])
    index = line.split('\t')[1]
    
    candidate_words = list(target2candidates[target])
    #print(candidate_words)
    
    text_a_raw = line.split('\t')[3].split(' ') # text_a untokenized
    
    masked_id = int(line.split('\t')[2])
    
    masked_word = text_a_raw[masked_id] # get the target word
    
    #text_a_raw.insert(masked_id + 1,'and')
    #text_a_raw.insert(masked_id + 2, 'MASK')
    
    text_a_raw = ' '.join(text_a_raw)
    
    #pat_letter = re.compile(r'[^a-zA-Z \']+')
    #text_a_raw = pat_letter.sub(' ', text_a_raw).strip().lower()
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    # to find the 's following the letters
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    # to find the ' following the words ending by s
    pat_s2 = re.compile("(?<=s)\'s?")
    # to find the abbreviation of not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    # to find the abbreviation of would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    # to find the abbreviation of will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    # to find the abbreviation of am
    pat_am = re.compile("(?<=[I|i])" "\'m")
    # to find the abbreviation of are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    # to find the abbreviation of have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    new_text = pat_is.sub(r"\1 is", text_a_raw)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    text_a_raw = new_text.replace('\'', ' ')
    text_a_raw = text_a_raw.split(' ')
    while '' in text_a_raw:
        text_a_raw.remove('')

    masked_id = text_a_raw.index(masked_word)
    masked_word_tokenized = tokenizer.tokenize(masked_word)

    text_a_tokenized = tokenizer.tokenize(' '.join(text_a_raw))
    target_sentence_length = len(text_a_tokenized)

    indexs = [i for (i,wordpiece) in enumerate(text_a_tokenized) if wordpiece == masked_word_tokenized[0]]
    for index_now in indexs:
        flag = True
        if index_now < masked_id:
            continue
        for i in range(len(masked_word_tokenized)):
            if masked_word_tokenized[i] != text_a_tokenized[index_now + i]:
                flag = False
        if flag == True:
            target_start_id = index_now
            target_end_id = index_now + len(masked_word_tokenized) - 1
            break
    
    instances.append('\t'.join([full_target_key, index, ' '.join(text_a_tokenized), str(target_start_id), str(target_end_id)]))

    offset_before_allowed = min(target_start_id, bow_size)
    
    offset_after_allowed = min(target_sentence_length - 1, target_end_id + bow_size) - target_end_id
    
    for offset in range(offset_before_allowed):
        text_offset = copy.deepcopy(text_a_tokenized)
        text_offset[target_start_id - offset - 1] = "[MASK]"
        instances.append('\t'.join([full_target_key, index, ' '.join(text_offset), str(-offset)]))
    for offset in range(offset_after_allowed):
        text_offset = copy.deepcopy(text_a_tokenized)
        text_offset[target_start_id + offset + 1] = "[MASK]"
        instances.append('\t'.join([full_target_key, index, ' '.join(text_offset), str(offset)]))

    num_lines_before = offset_before_allowed
    num_lines_after = offset_after_allowed


    text_target_masked = copy.deepcopy(text_a_tokenized)
    text_target_masked[target_start_id] = "[MASK]"
    instances.append('\t'.join([full_target_key, index, ' '.join(text_target_masked), str(offset)]))

    candidates = []

    for candidate_word in candidate_words:

        if len(candidate_word.split(' ')) != 1 or len(candidate_word.split('-')) != 1:
        #print("candidate word %s filtered" %(candidate_word))
            continue
        line_replaced = copy.deepcopy(text_a_raw)
        line_replaced[masked_id] = candidate_word
        line_replaced_tokenized = tokenizer.tokenize(' '.join(line_replaced))
        candidate_sentence_length = len(line_replaced_tokenized)

        candidates.append(candidate_word)

        candidate_word_tokenized = tokenizer.tokenize(candidate_word)

        indexs = [i for i,word in enumerate(line_replaced_tokenized) if word == candidate_word_tokenized[0]]
        for index_now in indexs:
            flag = True
            if index_now < target_start_id:
                continue
            for i in range(len(candidate_word_tokenized)):
                if candidate_word_tokenized[i] != line_replaced_tokenized[index_now + i]:
                    flag = False
            if flag == True:
                candidate_start_id = index_now
                break
        candidate_end_id = candidate_start_id + len(candidate_word_tokenized) - 1       

        instances.append('\t'.join([full_target_key, index, ' '.join(line_replaced_tokenized), str(candidate_start_id), str(candidate_end_id)]))

        offset_before_allowed = min(candidate_start_id, bow_size)
    
        offset_after_allowed = min(candidate_sentence_length - 1, candidate_end_id + bow_size) - candidate_end_id
    
        for offset in range(offset_before_allowed):
            text_offset = copy.deepcopy(line_replaced_tokenized)
            text_offset[candidate_start_id - offset - 1] = "[MASK]"
            instances.append('\t'.join([full_target_key, index, ' '.join(text_offset), str(-offset)]))
        for offset in range(offset_after_allowed):
            text_offset = copy.deepcopy(line_replaced_tokenized)
            text_offset[candidate_start_id + offset + 1] = "[MASK]"
            instances.append('\t'.join([full_target_key, index, ' '.join(text_offset), str(offset)]))

    return instances, candidates


def proposal2bert_service_mask(line, candidate_words, tokenizer, context_indice, context_weight):
    
    instances = [] # returned input lines for bert
    
    full_target_key = line.split('\t')[0]
    target = '.'.join(full_target_key.split('.')[:2])
    index = line.split('\t')[1]
    
    #print(candidate_words)
    
    text_a_raw = line.split('\t')[3].split(' ') # text_a untokenized
    
    masked_id = int(line.split('\t')[2])
    
    masked_word = text_a_raw[masked_id] # get the target word
    
    #text_a_raw.insert(masked_id + 1,'and')
    #text_a_raw.insert(masked_id + 2, 'MASK')
    
    text_a_raw = ' '.join(text_a_raw)
    
    #pat_letter = re.compile(r'[^a-zA-Z \']+')
    #text_a_raw = pat_letter.sub(' ', text_a_raw).strip().lower()
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    # to find the 's following the letters
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    # to find the ' following the words ending by s
    pat_s2 = re.compile("(?<=s)\'s?")
    # to find the abbreviation of not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    # to find the abbreviation of would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    # to find the abbreviation of will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    # to find the abbreviation of am
    pat_am = re.compile("(?<=[I|i])" "\'m")
    # to find the abbreviation of are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    # to find the abbreviation of have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    new_text = pat_is.sub(r"\1 is", text_a_raw)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    text_a_raw = new_text.replace('\'', ' ')
    text_a_raw = text_a_raw.split(' ')
    while '' in text_a_raw:
        text_a_raw.remove('')
        
    masked_id = text_a_raw.index(masked_word)
    masked_word_tokenized = tokenizer.tokenize(masked_word)

    text_a_tokenized = tokenizer.tokenize(' '.join(text_a_raw))
    target_sentence_length = len(text_a_tokenized)

    indexs = [i for (i,wordpiece) in enumerate(text_a_tokenized) if wordpiece == masked_word_tokenized[0]]
    for index_now in indexs:
        flag = True
        if index_now < masked_id:
            continue
        for i in range(len(masked_word_tokenized)):
            if masked_word_tokenized[i] != text_a_tokenized[index_now + i]:
                flag = False
        if flag == True:
            target_start_id = index_now
            target_end_id = index_now + len(masked_word_tokenized) - 1
            break
    
    instances.append('\t'.join([full_target_key, index, ' '.join(text_a_tokenized), str(target_start_id), str(target_end_id)]))


    selected_indice = []
    selected_weight = []
    for i,indice in enumerate(context_indice):
        #print(indice)
        if int(indice) - 1 >= target_start_id and int(indice) -1 <= target_end_id:
            continue
        if int(indice) - 1 >= len(text_a_tokenized):
            continue
        line = copy.deepcopy(text_a_tokenized)

        if int(indice) != 0:    
            #print(line)   
            line[int(indice) - 1] = "[MASK]"

        instances.append('\t'.join([full_target_key, index, ' '.join(line), str(1)]))
        
        selected_indice.append(indice)
        selected_weight.append(float(context_weight[i]))

    selected_weight_sum = np.sum(selected_weight)
    weights = [weight / selected_weight_sum for weight in selected_weight]



    text_target_masked = copy.deepcopy(text_a_tokenized)
    text_target_masked[target_start_id] = "[MASK]"
    instances.append('\t'.join([full_target_key, index, ' '.join(text_target_masked), str(1)]))

    candidates = []

    for candidate_word in candidate_words[:-1]:

        if len(candidate_word.split(' ')) != 1 or len(candidate_word.split('-')) != 1:
        #print("candidate word %s filtered" %(candidate_word))
            continue
        line_replaced = copy.deepcopy(text_a_raw)
        line_replaced[masked_id] = candidate_word
        line_replaced_tokenized = tokenizer.tokenize(' '.join(line_replaced))
        candidate_sentence_length = len(line_replaced_tokenized)
        offset = candidate_sentence_length - target_sentence_length

        candidates.append(candidate_word)
        #print(candidate_word)
        candidate_word_tokenized = tokenizer.tokenize(candidate_word)
        #print(candidate_word_tokenized)
        indexs = [i for i,word in enumerate(line_replaced_tokenized) if word == candidate_word_tokenized[0]]
        for index_now in indexs:
            flag = True
            if index_now < target_start_id:
                continue
            for i in range(len(candidate_word_tokenized)):
                if candidate_word_tokenized[i] != line_replaced_tokenized[index_now + i]:
                    flag = False
            if flag == True:
                candidate_start_id = index_now
                break
        candidate_end_id = candidate_start_id + len(candidate_word_tokenized) - 1       

        instances.append('\t'.join([full_target_key, index, ' '.join(line_replaced_tokenized), str(candidate_start_id), str(candidate_end_id)]))


        for indice in context_indice:
            if int(indice) - 1 >= target_start_id and int(indice) -1 <= target_end_id:
                continue
            if int(indice) - 1 >= len(text_a_tokenized):
                continue
            assert(indice in selected_indice)
            line = copy.deepcopy(line_replaced_tokenized)
            if int(indice) -1 > target_start_id:
                indice_p = int(indice) - 1 - target_end_id + candidate_end_id
            else:
                indice_p = int(indice) - 1
            if indice_p >= 0:
                line[indice_p] = '[MASK]'
            instances.append('\t'.join([full_target_key, index, ' '.join(line), str(1)]))

    return instances, candidates, selected_indice, weights



def proposal2bert_service(line, candidate_words, tokenizer):
    
    instances = [] # returned input lines for bert
    
    full_target_key = line.split('\t')[0]
    target = '.'.join(full_target_key.split('.')[:2])
    index = line.split('\t')[1]
    
    #print(candidate_words)
    
    text_a_raw = line.split('\t')[3].split(' ') # text_a untokenized
    
    masked_id = int(line.split('\t')[2])
    
    masked_word = text_a_raw[masked_id] # get the target word
    
    #text_a_raw.insert(masked_id + 1,'and')
    #text_a_raw.insert(masked_id + 2, 'MASK')
    
    text_a_raw = ' '.join(text_a_raw)
    
    #pat_letter = re.compile(r'[^a-zA-Z \']+')
    #text_a_raw = pat_letter.sub(' ', text_a_raw).strip().lower()
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    # to find the 's following the letters
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    # to find the ' following the words ending by s
    pat_s2 = re.compile("(?<=s)\'s?")
    # to find the abbreviation of not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    # to find the abbreviation of would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    # to find the abbreviation of will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    # to find the abbreviation of am
    pat_am = re.compile("(?<=[I|i])" "\'m")
    # to find the abbreviation of are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    # to find the abbreviation of have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    new_text = pat_is.sub(r"\1 is", text_a_raw)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    text_a_raw = new_text.replace('\'', ' ')
    text_a_raw = text_a_raw.split(' ')
    while '' in text_a_raw:
        text_a_raw.remove('')
        
    masked_id = text_a_raw.index(masked_word)
    masked_word_tokenized = tokenizer.tokenize(masked_word)

    text_a_tokenized = tokenizer.tokenize(' '.join(text_a_raw))

    indexs = [i for (i,wordpiece) in enumerate(text_a_tokenized) if wordpiece == masked_word_tokenized[0]]
    for index_now in indexs:
        flag = True
        if index_now < masked_id:
            continue
        for i in range(len(masked_word_tokenized)):
            if masked_word_tokenized[i] != text_a_tokenized[index_now + i]:
                flag = False
        if flag == True:
            target_start_id = index_now
            target_end_id = index_now + len(masked_word_tokenized) - 1
            break
    
    instances.append('\t'.join([full_target_key, index, ' '.join(text_a_tokenized), str(target_start_id), str(target_end_id)]))

    text_target_masked = copy.deepcopy(text_a_tokenized)
    text_target_masked[target_start_id] = "[MASK]"
    instances.append('\t'.join([full_target_key, index, ' '.join(text_target_masked), str(1)]))

    candidates = []

    for candidate_word in candidate_words[:-1]:

        if len(candidate_word.split(' ')) != 1 or len(candidate_word.split('-')) != 1:
        #print("candidate word %s filtered" %(candidate_word))
            continue
        line_replaced = copy.deepcopy(text_a_raw)
        line_replaced[masked_id] = candidate_word
        line_replaced_tokenized = tokenizer.tokenize(' '.join(line_replaced))

        candidates.append(candidate_word)

        candidate_word_tokenized = tokenizer.tokenize(candidate_word)
        #print(candidate_word)

        indexs = [i for i,word in enumerate(line_replaced_tokenized) if word == candidate_word_tokenized[0]]
        for index_now in indexs:
            flag = True
            if index_now < target_start_id:
                continue
            for i in range(len(candidate_word_tokenized)):
                if candidate_word_tokenized[i] != line_replaced_tokenized[index_now + i]:
                    flag = False
            if flag == True:
                candidate_start_id = index_now
                break
        candidate_end_id = candidate_start_id + len(candidate_word_tokenized) - 1       

        instances.append('\t'.join([full_target_key, index, ' '.join(line_replaced_tokenized), str(candidate_start_id), str(candidate_end_id)]))

    return instances, candidates



def cosine_similarity(vector_a, vector_b):

    nominator = np.dot(vector_a, vector_b)
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    return np.divide(nominator,denominator)

def target_sim(target_embedding, candidate_embedding, sim_methode='cosine_similarity'):

    if sim_methode == 'cosine_similarity':
        return cosine_similarity(target_embedding, candidate_embedding)
    elif sim_methode == 'dot_product':
        return np.dot(target_embedding, candidate_embedding)
    elif sim_methode == 'euclidien distance':
        return np.linalg.norm(target_embedding - candidate_embedding)
    else:
        print("sim methode not valid")
        return

def context_sim(target_context, candidate_context, weights, context_methode='sim_first', sim_methode='cosine_similarity', rank_methode='balmult'):

    #print(target_context.shape)
    #print(candidate_context.shape)
    assert(target_context.shape == candidate_context.shape)
    context_length = target_context.shape[0]

    if context_methode == 'average_first':
        #print(candidate_context.shape)
        weights = np.array(weights)
        target_context = np.average(target_context, axis = 0, weights = weights)
        candidate_context = np.average(candidate_context, axis = 0, weights = weights)
        target_context = np.reshape(target_context,-1)
        candidate_context = np.reshape(candidate_context,-1)
        #print(target_context.shape)
        #print(candidate_context.shape)
        if rank_methode == 'balmult' or rank_methode == 'mult':
            return target_sim(target_context, candidate_context, sim_methode) ** context_length
        elif rank_methode == 'add' or rank_methode == 'baladd':
            return target_sim(target_context, candidate_context, sim_methode)
        else:
            print("rank methode not valid")
            return
    elif context_methode == 'sim_first':
        context_similarity = []
        for i in range(context_length):
            context_similarity.append(target_sim(target_context[i,:], candidate_context[i,:], sim_methode))
        if rank_methode == 'balmult' or rank_methode == 'mult':
            return np.prod(context_similarity)
        elif rank_methode == 'add' or rank_methode == 'baladd':
            weights = np.array(weights)
            context_similarity = np.array(context_similarity)
            assert(weights.shape == context_similarity.shape)
            return np.dot(context_similarity,weights)
        else:
            print("rank methode not valid")
            return
    else:
        print("context methode not valid")
        return


def score(target_candidate_similarity, context_context_similarity, context_length, rank_methode='balmult'):
    
    if rank_methode == 'balmult':
        return pow((target_candidate_similarity ** context_length) * context_context_similarity, 1/(context_length * 2))
    elif rank_methode == 'mult':
        return pow(target_candidate_similarity * context_context_similarity, 1/(context_length + 1))
    elif rank_methode == 'baladd':
        return (target_candidate_similarity * context_length + context_context_similarity) / (2 * context_length)
    elif rank_methode == 'add':
        return (target_candidate_similarity + 2 * context_context_similarity) / 3
    elif rank_methode == 'target_only':
        return target_candidate_similarity
    else:
        print("rank methode not valid")
        return




def rank_one_instance(instances, candidates, lemmatizer, bow_size=5, sim_methode='cosine_similarity', context_methode='average_first', rank_methode='balmult'):
    
    #print(instances)
    to_embedding_list = [instance.split('\t')[2].split(' ') for instance in instances]
    #print(to_embedding_list)
    embedded = bc.encode(to_embedding_list, is_tokenized=True)
    
    target_line = instances[0]
    target = target_line.split('\t')[0]
    target_word = target.split('.')[0]
    target_pos = target.split('.')[-1]

    to_wordnet_pos = {'N':wordnet.NOUN,'J':wordnet.ADJ,'V':wordnet.VERB,'R':wordnet.ADV}
    from_lst_pos = {'j':'J','a':'J', 'v':'V', 'n':'N', 'r':'R'}
    pos = to_wordnet_pos[from_lst_pos[target_pos]] 
    #pos = to_wordnet_pos[target_pos] 
    index = target_line.split('\t')[1]
    #print(target_line)
    
    target_start_position = int(target_line.split('\t')[3])
    target_end_position = int(target_line.split('\t')[4])

    target_sentence_length = len(target_line.split('\t')[2].split(' '))

    target_embedding = np.mean(embedded[0, 1 + target_start_position : 1 + target_end_position + 1, :],axis=0)
    #print(target_embedding.shape)

    offset_before_allowed = min(target_start_position, bow_size)
    target_context_before = embedded[0, 1 + target_start_position - offset_before_allowed : 1 + target_start_position]
    
    offset_after_allowed = min(target_sentence_length - 1, target_end_position + bow_size) - target_end_position
    target_context_after = embedded[0, 2 + target_end_position : 2 + target_end_position + offset_after_allowed]

    #print(target_context_before.shape)
    #print(target_context_after.shape)
    target_context = np.concatenate((target_context_before, target_context_after), axis=0)

    target_masked_embedding = embedded[1, 1 + target_start_position : 1 + target_start_position + 1, :]

    num_candidates = len(instances) - 2

    candidate_scores = []

    for candidate_id in range(num_candidates):

        candidate_word = candidates[candidate_id] 

        if candidate_word == target_word or lemmatizer.lemmatize(candidate_word,pos) == target_word or target_word in candidate_word:
            continue
        #print(candidate_word)

        candidate_word = lemmatizer.lemmatize(candidate_word,pos)
        candidate_line = instances[2 + candidate_id]
        target = candidate_line.split('\t')[0]
        index = candidate_line.split('\t')[1]
        
        candidate_start_postition = int(candidate_line.split('\t')[3])
        candidate_end_position = int(candidate_line.split('\t')[4])

        #print(candidate_start_postition)
        #print(candidate_end_position)

        candidate_sentence_length = len(candidate_line.split('\t')[2].split(' '))

        #print(to_embedding_list[candidate_id + 1][candidate_start_postition])

        candidate_embedding = np.mean(embedded[2 + candidate_id, 1 + candidate_start_postition : 1 + candidate_end_position + 1, :],axis=0)

        offset_before_allowed = min(candidate_start_postition, bow_size)
        candidate_context_before = embedded[2 + candidate_id, 1 + candidate_start_postition - offset_before_allowed : 1 + candidate_start_postition]
        #print(candidate_context_before.shape)

        offset_after_allowed = min(candidate_sentence_length - 1, candidate_end_position + bow_size) - candidate_end_position
        #print(offset_after_allowed)
        #print(embedded.shape)
        candidate_context_after = embedded[2 + candidate_id, 1 + candidate_end_position + 1 : 2 + candidate_end_position + offset_after_allowed, :]
        #print(candidate_context_after.shape)

        candidate_context = np.concatenate((candidate_context_before, candidate_context_after), axis=0)

        target_candidate_similarity = target_sim(target_embedding, candidate_embedding, sim_methode=sim_methode)
        #print(target_candidate_similarity)

        target_masked_candidate_similarity = target_sim(target_masked_embedding, candidate_embedding, sim_methode = sim_methode)

        if rank_methode != "target_only":
            context_context_similarity = context_sim(target_context, candidate_context, sim_methode=sim_methode, context_methode=context_methode, rank_methode=rank_methode)
            #print(context_context_similarity)
        else:
            context_context_similarity = 0.1

        context_length = candidate_context.shape[0]
        #print(context_length)
        #print(candidate_word)
        #print(target_candidate_similarity)
        #print(context_context_similarity)
        substituability_score = score(target_candidate_similarity, context_context_similarity, context_length, rank_methode=rank_methode)
        #substituability_score = target_candidate_similarity + target_masked_candidate_similarity
        #print(substituability_score)
        if sim_methode == 'euclidien distance':
            substituability_score = -substituability_score

        candidate_scores.append([candidate_word, substituability_score])


    candidate_scores.sort(key=itemgetter(1), reverse=True)
    sub_strs = [' '.join([candidate_word, str(substituability_score)]) for candidate_word, substituability_score in candidate_scores]

    output_line = "RANKED\t" + ' '.join([target, index]) + '\t' + '\t'.join(sub_strs) + '\n'

    return output_line




def rank_one_instance_mask_context(instances, candidates, lemmatizer, selected_indice, weights, sim_methode='cosine_similarity', context_methode='sim_first', rank_methode='add'):
    
    #print(instances)
    to_embedding_list = [instance.split('\t')[2].split(' ') for instance in instances]
    #print(to_embedding_list)
    embedded = bc.encode(to_embedding_list, is_tokenized=True)
    
    target_line = instances[0]
    target = target_line.split('\t')[0]
    index = target_line.split('\t')[1]

    target_word = target.split('.')[0]
    target_pos = target.split('.')[-1]

    to_wordnet_pos = {'N':wordnet.NOUN,'J':wordnet.ADJ,'V':wordnet.VERB,'R':wordnet.ADV}
    from_lst_pos = {'j':'J','a':'J', 'v':'V', 'n':'N', 'r':'R'}
    pos = to_wordnet_pos[from_lst_pos[target_pos]] 
    #pos = to_wordnet_pos[target_pos]

    target_start_postition = int(target_line.split('\t')[3])
    target_end_position = int(target_line.split('\t')[4])

    target_sentence_length = len(target_line.split('\t')[2].split(' '))

    target_embedding = np.mean(embedded[0, 1 + target_start_postition : 1 + target_end_position + 1, :],axis=0)
    #print(target_embedding.shape)

    embedding_index = 1

    target_context = embedded[embedding_index, int(selected_indice[0]) : int(selected_indice[0]) + 1, :]
    embedding_index += 1

    for indice in selected_indice[1:]:
        target_context = np.concatenate((target_context, embedded[embedding_index, int(indice) : int(indice) + 1, : ]), axis=0)
        embedding_index += 1
   

    #print(target_context_before.shape)
    #print(target_context_after.shape)

    target_masked_embedding = embedded[embedding_index, target_start_postition : target_start_postition + 1, :]
    embedding_index += 1

    #print(target_context.shape)

    num_candidates = len(candidates)

    candidate_scores = []
    scores = []

    for candidate_id in range(num_candidates):

        candidate_word = candidates[candidate_id]
        #print(candidate_word)
        candidate_line = instances[embedding_index]


        #print(candidate_word)

        candidate_word = lemmatizer.lemmatize(candidate_word,pos)

        #print(candidate_line)
        target = candidate_line.split('\t')[0]
        index = candidate_line.split('\t')[1]
        
        candidate_start_postition = int(candidate_line.split('\t')[3])
        candidate_end_position = int(candidate_line.split('\t')[4])

        #print(candidate_start_postition)
        #print(candidate_end_position)

        candidate_sentence_length = len(candidate_line.split('\t')[2].split(' '))

        #print(to_embedding_list[candidate_id + 1][candidate_start_postition])

        candidate_embedding = np.mean(embedded[embedding_index, 1 + candidate_start_postition : 1 + candidate_end_position + 1, :],axis=0)
        embedding_index += 1

        candidate_context_indice = []
        for indice in selected_indice:
            if int(indice) - 1 > candidate_start_postition:
                indice_p = int(indice) - target_end_position + candidate_end_position
            else:
                indice_p = int(indice)
            candidate_context_indice.append(indice_p)

        candidate_context = embedded[embedding_index, candidate_context_indice[0] : candidate_context_indice[0] + 1, :]
        embedding_index += 1

        for indice in candidate_context_indice[1:]:
            candidate_context = np.concatenate((candidate_context, embedded[embedding_index, indice : indice + 1, : ]), axis=0)
            embedding_index += 1


        target_candidate_similarity = target_sim(target_embedding, candidate_embedding, sim_methode=sim_methode)
        #print(target_candidate_similarity)
        target_masked_candidate_similarity = target_sim(target_masked_embedding, candidate_embedding, sim_methode=sim_methode)

        if rank_methode != "target_only":
            context_context_similarity = context_sim(target_context, candidate_context, weights, sim_methode=sim_methode, context_methode=context_methode, rank_methode=rank_methode)
            #print(context_context_similarity)
        else:
            context_context_similarity = 0.1

        context_length = candidate_context.shape[0]
        #print(context_length)
        #print(candidate_word)
        #print(target_candidate_similarity)
        #print(context_context_similarity)
        #print(context_length)
        substituability_score = score(target_candidate_similarity, context_context_similarity, context_length, rank_methode=rank_methode)
        #substituability_score = target_candidate_similarity + target_masked_candidate_similarity
        #print(substituability_score)
        
        if sim_methode == 'euclidien distance':
            substituability_score = -substituability_score
        if candidate_word == target_word or lemmatizer.lemmatize(candidate_word,pos) == target_word or target_word in candidate_word:
            continue
        candidate_scores.append([candidate_word, substituability_score, candidate_id])
        scores.append(substituability_score)

    stddev = np.std(scores)
    final_candidate_scores = []
    for candidate_word, substituability_score,candidate_id in candidate_scores:
        lm_score = (50 - candidate_id)/50 * stddev
        substituability_score += lm_score
        final_candidate_scores.append([candidate_word, substituability_score])
    final_candidate_scores.sort(key=itemgetter(1), reverse=True)
    sub_strs = [' '.join([candidate_word, str(substituability_score)]) for candidate_word, substituability_score in final_candidate_scores]

    output_line = "RANKED\t" + ' '.join([target, index]) + '\t' + '\t'.join(sub_strs) + '\n'

    return output_line





def rank_all_proposal_new(input_file, candidate_file, context_file, tokenizer, lemmatizer,  sim_methode='cosine_similarity', context_methode='sim_first', rank_methode='add'):

    inputfile = open(input_file,'r',encoding="Windows-1252")
    input_lines = inputfile.readlines()
    candidate_lines = open(candidate_file,'r').readlines()
    context_lines = open(context_file,'r').readlines()
    with tf.gfile.GFile("result.generated_example",'w') as writer:
        for index,line in enumerate(input_lines):
            candidate_words = candidate_lines[index].split(' ')
            context_indice = [pair.split(' ')[0] for pair in context_lines[index].split('\t')]
            context_score = [pair.split(' ')[1] for pair in context_lines[index].split('\t')]
            instances, candidates, selected_indice, weights = proposal2bert_service_mask(line, candidate_words, tokenizer, context_indice, context_score)
            #print(instances)
            #print(candidates)
            output_line = rank_one_instance_mask_context(instances, candidates, lemmatizer, selected_indice, weights, sim_methode=sim_methode, context_methode=context_methode, rank_methode=rank_methode)
            writer.write(output_line)

            if index % 10 == 0:
                print("%d lines written" %(index))
    inputfile.close()



if __name__ == '__main__':

    vocab = load_vocab('../uncased_L-24_H-1024_A-16/vocab.txt')

    tokenizer = FullTokenizer("../uncased_L-24_H-1024_A-16/vocab.txt")

    #target2candidates = read_candidates('./lexsub/datasets/coinco.no_problematic.candidates')
    #target2candidates = read_candidates('./lexsub/datasets/lst.gold.candidates')

    lemmatizer = WordNetLemmatizer()

    bc = BertClient()

    #rank_all('./lexsub/datasets/lst_all.preprocessed', target2candidates, tokenizer, bow_size=6,sim_methode='dot_product',context_methode='average_first',rank_methode='baladd')
    #rank_all_new('./lexsub/datasets/lst_all.preprocessed', target2candidates, tokenizer, bow_size=6,sim_methode='dot_product',context_methode='average_first',rank_methode='baladd')
    #rank_all_proposal('./lexsub/datasets/lst_all.preprocessed', './candidates', tokenizer, lemmatizer, bow_size=6,sim_methode='cosine_similarity',context_methode='average_first',rank_methode='baladd')
    rank_all_proposal_new('./lst_all.preprocessed', './candidates', './context', tokenizer, lemmatizer, sim_methode='cosine_similarity',context_methode='sim_first',rank_methode='add')