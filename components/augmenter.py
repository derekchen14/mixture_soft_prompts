import os, pdb, sys
import numpy as np
import random
import math

from torch import nn, no_grad
from torch.nn import functional as F
from assets.static_vars import device, MAX_ATTEMPTS

from tqdm import tqdm as progress_bar
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


class BaseAugmenter(object):
  """ Performs traditional non-controlled data augmentation.  The key difference between 
    DataAug and CTG is that the augmenters do not condition on a set of attributes, which is
    why they are not label preserving. This lack of control is the distinguishing factor. """
  def __init__(self, args):
    self.name = 'base-augmenter'

  def augment(self, seed_examples):
    # notice that the augmenters do not accept a set of contraints to condition on
    raise NotImplementedError


class EasyDataAugmenter(BaseAugmenter):

  def __init__(self, args, pieces):
    super().__init__(args)
    self.name = 'easy-data-augmenter'
    self.prob = args.threshold
    self.max_attempts = MAX_ATTEMPTS

    self.wordnet = pieces['wordnet']
    self.stopwords = pieces['stopwords'].words('english')

  def _swap_word(self, new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
      random_idx_2 = random.randint(0, len(new_words) - 1)
      counter += 1
      if counter >= self.max_attempts:
        return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

  def _get_synonyms(self, word):
    synonyms = set()
    for syn in self.wordnet.synsets(word):
      for lemma in syn.lemmas():
        synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
        synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
        synonyms.add(synonym)
    if word in synonyms:
      synonyms.remove(word)
    return list(synonyms)

  def _add_word(self, new_words):
    synonyms = []
    random_word_list = list([word for word in new_words if word not in self.stopwords])
    if len(random_word_list) < 1:
      return new_words

    counter = 0
    while len(synonyms) < 1:
      rand_index = random.randrange(len(random_word_list))
      random_word = random_word_list[rand_index]
      synonyms = self._get_synonyms(random_word)
      counter += 1
      if counter > self.max_attempts:
        synonyms.append(random_word)
    random_synonym = random.choice(synonyms)
    random_idx = random.randrange(len(new_words))
    new_words.insert(random_idx, random_synonym)
    return new_words

  def swap_tokens(self, words):
    new_words = words.copy()
    num_swaps = math.ceil(len(words) * self.prob)
    for _ in range(num_swaps):
      new_words = self._swap_word(new_words)
    return TreebankWordDetokenizer().detokenize(new_words)

  def delete_tokens(self, words):
    new_words = list()
    for word in words:
      if random.random() > self.prob:
        new_words.append(word)
    # if all words are deleted, just return a random word
    if len(new_words) == 0:
        return random.choice(words)
    return TreebankWordDetokenizer().detokenize(new_words)

  def insert_tokens(self, words):
    num_insertions = math.ceil(len(words) * self.prob)
    new_words = words.copy()
    for _ in range(num_insertions):
      new_words = self._add_word(new_words)
    return TreebankWordDetokenizer().detokenize(new_words)

  def replace_synonym(self, tokens):
    found = False

    for _ in range(self.max_attempts):
      selected_idx = random.randrange(len(tokens))  # select a word to replace
      matches = self._get_synonyms(tokens[selected_idx])

      if len(matches) >= 1:
        tokens[selected_idx] = random.choice(matches)
        found = True
        break

    text = TreebankWordDetokenizer().detokenize(tokens)
    return text, found

  def augment(self, utterance):
    augmentations = []
    words = word_tokenize(utterance)

    insertion = self.insert_tokens(words)
    augmentations.append(insertion)
    if len(words) > 2:
      swapped = self.swap_tokens(words)
      augmentations.append(swapped)
    if len(words) > 4:
      deletion = self.delete_tokens(words)
      augmentations.append(deletion)
    if len(words) > 6:
      replacement, found = self.replace_synonym(words)
      if found:
        augmentations.append(replacement)

    return augmentations


class Paraphraser(BaseAugmenter):
  def __init__(self, args, pieces):
    super().__init__(args)
    self.name = 'paraphrase-augmenter'
    self.temp = args.temperature # set reasonably high to avoid duplicates
    self.num_generations = args.num_generations
    self.max_tokens = args.target_max_len

    self.encoder = pieces['encoder']
    self.embedder = pieces['embedder']

  def augment(self, utterance):
    embedded = self.embedder(utterance, return_tensors='pt')
    inputs = embedded.input_ids.to(device)
    with no_grad():
      latent = self.encoder.generate(inputs, do_sample=False, num_beams=self.num_generations,
                                     max_new_tokens=self.max_tokens, num_return_sequences=self.num_generations,
                                     temperature=self.temp)
    augmentations = self.embedder.batch_decode(latent.detach(), skip_special_tokens=True)
    return list(set(augmentations))


class TextInfiller(BaseAugmenter):
  def __init__(self, args, pieces):
    super().__init__(args)
    self.name = 'mask-in-filling'
    self.temp = args.temperature
    self.mask_filler = pieces['infiller']
    self.stopwords = pieces['stopwords'].words('english')

  def add_mask(self, utterance):
    tokens = utterance.split()
    allowed = []
    for idx, token in enumerate(tokens):
      if token not in self.stopwords and len(token) > 1:
        allowed.append(idx)
    
    if len(allowed) > 0:
      chosen_idx = random.choice(allowed)
      tokens[chosen_idx] = '[MASK]'
      masked_utt = ' '.join(tokens)
      return masked_utt, True
    else:
      return utterance, False

  def augment(self, utterance):
    masked_utt, successful = self.add_mask(utterance)
    
    if successful:
      suggestions = self.mask_filler(masked_utt)  # gives 5 suggestions
      return [masked_utt.replace('[MASK]', suggest['token_str']) for suggest in suggestions]
    else:
      return [utterance]

class RoundTripTranslator(BaseAugmenter):

  def __init__(self, args, pieces):
    super().__init__(args)
    self.name = 'round-trip-translation'
    self.encoder = pieces['encoder']
    self.embedder = pieces['embedder']
    self.decoder = pieces['decoder']
    self.debedder = pieces['debedder']

  def augment(self, utterance):
    augmentations = set()
    augmentations.add(utterance)

    for language in ['fra', 'spa', 'por', 'ru', 'de']: 
      input_utt = utterance if language in ['ru', 'de'] else f">>{language}<< {utterance}"
      lang = language if language in ['ru', 'de'] else 'roa'
      
      with no_grad():
        forward_vector = self.embedder[lang](input_utt, return_tensors='pt').to(device)
        latent_vector = self.encoder[lang].generate(**forward_vector)
        translated_utt = self.embedder[lang].decode(latent_vector[0], skip_special_tokens=True)

        backward_vector = self.debedder[lang](translated_utt, return_tensors='pt').to(device)
        output_vector = self.decoder[lang].generate(**backward_vector)
        new_utt = self.debedder[lang].decode(output_vector[0], skip_special_tokens=True)

      augmentations.add(new_utt)
    # remove the original to prevent duplicate
    augmentations.remove(utterance)
    return list(augmentations)
