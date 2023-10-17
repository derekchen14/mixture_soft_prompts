import os, pdb, sys
import numpy as np
import random
import mmap
import re
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from assets.static_vars import ATTRIBUTE_TOKEN_LEN, MAX_MIXTURE_SIZE, device, DATASETS

class BaseDataset(Dataset):
  def __init__(self, args, examples, tokenizer, split):
    self.args = args
    self.data = examples
    self.tokenizer = tokenizer
    self.name = 'base-dataset'
    self.split = split
    self.shuffle = (split == 'train')

    self.dataset = args.dataset
    self.model_type = args.model
    self.engineer = None

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

  def _pad_null(self, targets, max_tokens, direction='right'):
    """ null means value of -100 which tells the model to ignore when computing the loss.
    The additional benefit is being able to specify the direction of padding. """
    padded = []
    for vector in targets.input_ids:
      tensor = torch.tensor(vector[:max_tokens])   # truncate to max sequence length
      diff = max_tokens - tensor.shape[0]
      if diff > 0:
        if direction == 'left':
          tensor = nn.functional.pad(tensor, (diff, 0), value=-100)
        elif direction == 'right':
          tensor = nn.functional.pad(tensor, (0, diff), value=-100)
      padded.append(tensor)

    target = torch.stack(padded).to(device)
    return target

  @staticmethod
  def target_to_tensor(targets, max_len):
    """ Transform a list of strings into a list of tensors of the same length """
    tensors = []
    for label_string in targets:
      numbered = [ord(char) for char in label_string]

      if len(numbered) > max_len:
        tensors.append(numbered[:max_len])
      elif len(numbered) < max_len:
        gap = max_len - len(numbered)
        filler = [ord('$')] * gap
        tensors.append(numbered + filler)
      else:
        tensors.append(numbered)

    transformed = torch.tensor(tensors, dtype=torch.long).to(device)
    return transformed

  @staticmethod
  def tensor_to_target(tensors):
    """ Transform the list of label tensors back into the original strings """
    transformed = []
    for label_tensor in tensors.tolist():
      lettered = [chr(char) for char in label_tensor]

      ptr = len(lettered) - 1
      while lettered[ptr] == '$':
        ptr -= 1
      string = ''.join(lettered[:ptr+1])

      transformed.append(string)
    return transformed

  def collate_lm(self, args, examples):
    raise NotImplementedError

  def collate_seq2seq(self, args, examples):
    raise NotImplementedError

  def collate(self, args, examples):
    if self.model_type in ['gpt', 'opt']:
      return self.collate_lm(args, examples)
    elif self.model_type in ['godel', 't5']:
      return self.collate_seq2seq(args, examples)

  def collate_func(self, examples):
    return examples


class InContextDataset(BaseDataset):

  def remove_special(self, text):
    text = text.replace('<agent>', ' agent:')
    text = text.replace('<customer>', ' customer:')
    text = text.replace('<none>', 'none')
    text = text.replace('<label>', 'answer:')
    text = text.replace('<sep>', ';')
    text = text.replace('<remove>', 'none')
    # text = text.replace('<pad>', '[PAD]')
    return text

  def collate(self, args, examples):
    """ train and dev splits should not occur since you do not need gradient based training """
    # assert(self.split not in ['train', 'dev'])
    texts, labels = [], []

    for example in examples:
      prompt = self.engineer.icl_with_exemplars(example)

      if args.task == 'basic':
        prompt = 'customer:' if examples[-1]['text'].startswith('<agent>') else 'agent:'
        if self.model_type == 't5':
          prompt = self.remove_special(f"{prompt} <extra_id_0>")
        else:
          prompt = self.remove_special(f"{prompt}")

      texts.append(self.remove_special(prompt))

      label = self.remove_special(example['target'])
      labels.append(label)

    inputs = self.tokenizer(texts, padding=True, max_length=args.source_max_len,
                                truncation=True, return_tensors='pt').to(device)
    return inputs.to(device), labels


class SoftPromptMixDataset(BaseDataset):
  """ Used by MSP for augmenting the seed data.  The
  seed data is the few shot data available from NLU++, CrossNER and TopV2 """
  def __init__(self, args, examples, tokenizer, split):
    super().__init__(args, examples, tokenizer, split)
    self.name = 'soft-prompt-mix-dataset'

  def collate_seq2seq(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into a GPT-like model"""
    input_texts, labels = [], []
    metadata = { 'constraints': [], 'pad_lengths': [] }

    for example in examples:
      input_text = self.engineer.prompt_with_exemplars(example, add_tail=True)
      input_texts.append(input_text)
      # labels add 'text' since data augmentation performs p(x|y) rather than p(y|x)
      labels.append(example['text'])

      if self.dataset == 'topv2':
        unique_attrs, unique_vals =  set(example['attributes']), set(example['values'])
        ua_size, uv_size = len(unique_attrs), len(unique_vals)
        if ua_size + uv_size > MAX_MIXTURE_SIZE:
          cutoff = MAX_MIXTURE_SIZE - ua_size 
          unique_vals = list(unique_vals)[:cutoff]
        metadata['constraints'].append([unique_attrs, unique_vals])
      else:
        metadata['constraints'].append(set(example['attributes']))

    max_length = args.source_max_len + (ATTRIBUTE_TOKEN_LEN * MAX_MIXTURE_SIZE)
    inputs = self.tokenizer(input_texts, padding=True, max_length=max_length,
                              truncation=True, return_tensors='pt').to(device)
    tokens = self.tokenizer(input_texts, padding=False, max_length=max_length, truncation=True)
    batch_size, max_seq = inputs['input_ids'].shape
    datatype = inputs['input_ids'].dtype
    metadata['pad_lengths'] = [max_seq - len(text) for text in tokens['input_ids']]

    # Structure is <instruct> <padding> <attribute> <content>
    instruct_tokens = torch.full((batch_size, args.n_tokens), -1, dtype=datatype).to(device)
    inputs['input_ids'] = torch.cat([instruct_tokens, inputs['input_ids']], dim=1)
    instruct_mask = torch.ones((batch_size, args.n_tokens), dtype=datatype).to(device)
    inputs['attention_mask'] = torch.cat([instruct_mask, inputs['attention_mask']], dim=1)

    if self.split == 'train':
      targets = self.tokenizer(labels)  # we do not want to return tensors
      max_vector_len = min(max([len(v) for v in targets.input_ids]), args.target_max_len)
      target_tensor = self._pad_null(targets, max_vector_len, direction='right')
      return inputs, target_tensor, metadata
    else:
      return inputs, labels, metadata

  def collate_lm(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into a GPT-like model"""
    input_texts, labels = [], []
    metadata = { 'constraints': [], 'pad_lengths': [] }

    for example in examples:

      if self.split == 'train':
        add_tail, include_seed = False, True
        input_text = self.engineer.prompt_with_exemplars(example, add_tail, include_seed)
        input_text += self.tokenizer.eos_token
        max_length = args.source_max_len + args.target_max_len + (ATTRIBUTE_TOKEN_LEN * MAX_MIXTURE_SIZE)

      elif self.split in ['dev', 'test']:
        add_tail, include_seed = True, False
        input_text = self.engineer.prompt_with_exemplars(example, add_tail, include_seed)
        max_length = args.source_max_len + (ATTRIBUTE_TOKEN_LEN * MAX_MIXTURE_SIZE)

      input_texts.append(input_text)
      # labels add 'text' since data augmentation performs p(x|y) rather than p(y|x)
      labels.append(example['text'])

      if self.dataset == 'topv2':
        unique_attrs, unique_vals =  set(example['attributes']), set(example['values'])
        ua_size, uv_size = len(unique_attrs), len(unique_vals)
        if ua_size + uv_size > MAX_MIXTURE_SIZE:
          cutoff = MAX_MIXTURE_SIZE - ua_size 
          unique_vals = list(unique_vals)[:cutoff]
        metadata['constraints'].append([unique_attrs, unique_vals])
      else:
        metadata['constraints'].append(set(example['attributes']))

    inputs = self.tokenizer(input_texts, padding=True, max_length=max_length,
                              truncation=True, return_tensors='pt').to(device)
    tokens = self.tokenizer(input_texts, padding=False, max_length=max_length, truncation=True)
    batch_size, max_seq = inputs['input_ids'].shape
    datatype = inputs['input_ids'].dtype
    metadata['pad_lengths'] = [max_seq - len(text) for text in tokens['input_ids']]

    # Structure is <instruct> <padding> <attribute> <content>
    instruct_tokens = torch.ones((batch_size, args.n_tokens), dtype=datatype).to(device)
    inputs['input_ids'] = torch.cat([instruct_tokens, inputs['input_ids']], dim=1)
    instruct_mask = torch.ones((batch_size, args.n_tokens), dtype=datatype).to(device)
    inputs['attention_mask'] = torch.cat([instruct_mask, inputs['attention_mask']], dim=1)

    if self.split == 'train':
      return inputs, inputs['input_ids'], metadata
    else:
      return inputs, labels, metadata

class ConditionalLMDataset(BaseDataset):
  """ Used by CLM and Prefix CTG """
  def __init__(self, args, examples, tokenizer, split):
    super().__init__(args, examples, tokenizer, split)
    self.name = 'conditional-lm-dataset'

  def collate(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into a GPT-like model"""

    dialogues, labels = [], []
    eos = self.tokenizer.eos_token
    bos = self.tokenizer.bos_token

    format_for_train = self.split == 'train' and self.args.do_train
    # In GPT, for training pad to right and for generation pad to left.
    if format_for_train:
      self.tokenizer.padding_side = 'right'
    for example in examples:
      dialog = f"{bos}{example['constraint_str']}<label>"

      if format_for_train:
        dialog += f"{example['text']}{eos}"
        max_len = args.source_max_len + args.target_max_len
      else:
        max_len = args.source_max_len
      dialogues.append(dialog)
      labels.append(example['text'])

    inputs = self.tokenizer(dialogues, padding=True, max_length=max_len, truncation=True, return_tensors='pt').to(device)
    self.tokenizer.padding_side = 'left'

    if self.split == 'train':
      inputs['labels'] = torch.where(inputs['input_ids']!=self.tokenizer.pad_token_id, inputs['input_ids'], -100)
      return inputs, inputs['input_ids']
    else:
      return inputs, labels


class DExpertDataset(BaseDataset):
  """ Used by DExperts CTG. Each Dataset is for a particular attribute """
  def __init__(self, args, examples, tokenizer, split, attribute=None):
    super().__init__(args, examples, tokenizer, split)
    self.attribute = attribute
    self.filter_relevant_examples(examples)
    self.name = 'dexpert-dataset'

  def filter_relevant_examples(self, examples):
    relevant_exs = []

    for example in examples:
      if (self.attribute is None or
          (self.attribute in example['attributes']) or
          (self.dataset in {'topv2', 'nlu++'} and (self.attribute in example['values']))):
        relevant_exs.append(example)
    self.data = relevant_exs
    print(f"Made a dataset with {len(self.data)} examples for attribute {self.attribute}")

  def cluster(self):
    assert self.attribute == None, f"We only need to cluster when synthesizing data, and we should synthesize from all attrs"
    most_common = {}
    for ex in self.data:
      for attr in ex['attributes']:
        if attr not in most_common: most_common[attr] = 0
        most_common[attr] += 1
      if self.dataset in {'topv2', 'nlu++'}:
        for attr in ex['values']:
          if attr not in most_common: most_common[attr] = 0
          most_common[attr] += 1
    most_common = [attr for attr, count in sorted(most_common.items(), key=lambda item: item[1], reverse=True)]

    if self.dataset in {'topv2', 'nlu++'}:
      sorted_data = sorted(self.data, key=lambda ex: [most_common.index(attr) for attr in ex['attributes']+ex['values']])
    else:
      sorted_data = sorted(self.data, key=lambda ex: [most_common.index(attr) for attr in ex['attributes']])
    self.data = sorted_data

  def collate(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into a GPT-like model"""

    dialogues, labels = [], []
    eos, bos = self.tokenizer.eos_token, self.tokenizer.bos_token

    # In GPT, for training pad to right and for generation pad to left.
    if self.args.do_train:
      self.tokenizer.padding_side = 'right'
    for example in examples:

      dialog = f"{bos}"
      if self.args.do_train and self.split == 'train':
        dialog += f"{example['text']}{eos}"
        max_len = args.source_max_len
      else:
        max_len = args.source_max_len + args.target_max_len
      dialogues.append(dialog)
      labels.append(example['text'])

    inputs = self.tokenizer(dialogues, padding=True, max_length=max_len, truncation=True, return_tensors='pt').to(device)
    self.tokenizer.padding_side = 'left'

    if self.split == 'train':
      inputs['labels'] = inputs['input_ids']
      return inputs, inputs['input_ids']
    else:
      return inputs, labels


class CVAEDataset(BaseDataset):
  """ Used by CVAE CTG."""
  def __init__(self, args, examples, tokenizer, split): #, config):
    super().__init__(args, examples, tokenizer, split)
    self.name = 'cvae-dataset'

  def collate(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into the CVAE BERT model"""
    inputs, labels = [], []

    for example in examples:

      inputs.append(example['constraint_str'])
      labels.append(example['text'])

    if self.split == 'train':
      tokenized = self.tokenizer(inputs, labels, padding=True, max_length=args.source_max_len+args.target_max_len, truncation=True, return_tensors='pt').to(device)
    else:
      tokenized = self.tokenizer(inputs, padding=True, max_length=args.source_max_len+args.target_max_len, truncation=True, return_tensors='pt').to(device)
    tokenized['labels'] = self.tokenizer(labels, padding=True, max_length=args.target_max_len, truncation=True, return_tensors='pt').input_ids.to(device)
    return tokenized, labels


class FineTuneDataset(BaseDataset):
  def __init__(self, args, examples, tokenizer, split):
    super().__init__(args, examples, tokenizer, split)
    self.name = 'fine-tune-dataset'

  def collate_seq2seq(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into a T5 or BART model"""
    dialogues, labels = [], []
    for example in examples:
      dialogues.append(example['text'])
      labels.append(example['target'])

    inputs = self.tokenizer(dialogues, padding='longest', max_length=args.source_max_len,
                          truncation=True, pad_to_multiple_of=8, return_tensors='pt').to(device)

    if args.task == 'soft_prompt':
      batch_size = len(examples)      # set to negative to differentiate from decoder inputs
      prompt_tokens = torch.full((batch_size,args.n_tokens), -1, device=device)
      inputs['input_ids'] = torch.cat([prompt_tokens, inputs['input_ids']], 1)
      prompt_mask = torch.ones((batch_size, args.n_tokens)).to(device)
      inputs['attention_mask'] = torch.cat([prompt_mask, inputs['attention_mask']], 1)

    if self.split == 'train':
      targets = self.tokenizer(labels)  # we do not want to return tensors
      max_vector_len = min(max([len(v) for v in targets.input_ids]), args.target_max_len)
      target_tensor = self._pad_null(targets, max_vector_len, direction='right')
      return inputs, target_tensor

    else:
      return inputs, labels

  def collate_lm(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into a GPT-like model"""
    dialogues, labels = [], []
    eos = self.tokenizer.eos_token

    for example in examples:
      context_prompt, answer_prompt = self.engineer.get_prompt()
      target = example['target']

      if self.split == 'train':
        # dialog = f"{history} {prompt} {target['value']}{eos}"
        dialog = f"{context_prompt}{example['text']}{answer_prompt}{target}{eos}"
        max_length = args.source_max_len + args.target_max_len
      elif self.split in ['dev', 'test']:
        dialog = f"{context_prompt}{example['text']}{answer_prompt}"
        max_length = args.source_max_len

      dialogues.append(dialog)
      labels.append(target)

    inputs = self.tokenizer(dialogues, padding=True, max_length=max_length,
                              truncation=True, return_tensors='pt').to(device)

    if args.task == 'soft_prompt':
      batch_size = len(examples)
      prompt_tokens = torch.full((batch_size, args.n_tokens), 1, device=device)
      inputs['input_ids'] = torch.cat([prompt_tokens, inputs['input_ids']], 1)
      prompt_attn = torch.full((batch_size, args.n_tokens), 1, device=device)
      inputs['attention_mask'] = torch.cat([prompt_attn, inputs['attention_mask']], 1)

    if args.accelerate:
      labels = super().target_to_tensor(labels, args.target_max_len)

    if self.split == 'train':
      return inputs, inputs['input_ids']
    else:
      return inputs, labels

class ClassifyDataset(BaseDataset):
    def __init__(self, args, examples, tokenizer, split):
      super().__init__(args, examples, tokenizer, split)
      self.name = 'classify-dataset'
      self.attr2id = None
      self.slot2id = None
      self.domain = args.domain

    def _labels_to_multihot(self, attributes):
      num_classes = len(self.attr2id)
      attribute_ids = [self.attr2id[attr] for attr in attributes]
      labels = torch.tensor(attribute_ids)
      return torch.zeros(num_classes).scatter_(0, labels, 1.)

    def _slots_to_multihot(self, slots):
      num_slots = len(self.slot2id)
      slot_ids = [self.slot2id[slot] for slot in slots]
      labels = torch.tensor(slot_ids)
      return torch.zeros(num_slots).scatter_(0, labels, 1.)

    def collate(self, args, examples, tokenizer=None):
      if args.do_eval:
        if self.dataset == 'topv2':
          return self.dual_inference(args, examples, tokenizer)
        else:
          return self.inference(args, examples, tokenizer)
      else:
        if self.dataset == 'topv2':
          return self.dual_classify(examples, args.source_max_len)
        else:
          return self.single_classify(examples, args.source_max_len)

    def dual_inference(self, args, examples, tokenizer=None):
      if tokenizer is None: tokenizer = self.tokenizer
      texts, intent_labels, slot_labels = [], [], []
      for example in examples:
        try:
          intent_label = self._labels_to_multihot(example['attributes'])
          if 'values' not in example:
            example['values'] = []
          slot_label = self._slots_to_multihot(example['values'])
          intent_labels.append(intent_label)
          slot_labels.append(slot_label)
          texts.append(example['text'])
        except KeyError as err:
          print("The topV2 miss attribute is {}".format(err))
          continue

      if len(texts) == 0: return None, None
      inputs = tokenizer(texts, padding='longest', max_length=args.source_max_len,
                          truncation=True, pad_to_multiple_of=8, return_tensors='pt')
      targets = {
        'intent': torch.stack(intent_labels).to(device),
        'slot': torch.stack(slot_labels).to(device),
      }
      return inputs.to(device), targets

    def inference(self, args, examples, tokenizer=None):
      if tokenizer is None: tokenizer = self.tokenizer
      texts, labels, lexical_match = [], [], []
      for example in examples:
        try:
          labels.append(self._labels_to_multihot(example['attributes']))
        except(KeyError): continue
        if self.dataset == 'crossner':
          matches = [1 for val in example['values'] if val in example['text']]
          if len(example['values']) == 0:
              lex_score = 1.0
          else: lex_score = round(sum(matches) / len(example['values']), 3)
          lexical_match.append(lex_score)
        texts.append(example['text'])
      if len(texts) == 0: return None, None, None
      inputs = tokenizer(texts, padding='longest', max_length=args.source_max_len,
                          truncation=True, pad_to_multiple_of=8, return_tensors='pt')
      labels = torch.stack(labels)
      return inputs.to(device), labels.to(device), lexical_match

    def dual_classify(self, examples, source_max_len):
      texts, intent_labels, slot_labels = [], [], []
      for example in examples:
        try:
          intent_label = self._labels_to_multihot(example['attributes'])
          slot_label = self._slots_to_multihot(example['values'])
          intent_labels.append(intent_label)
          slot_labels.append(slot_label)
          texts.append(example['text'])
        except KeyError as err:
          print("The topV2 miss attribute is {}".format(err))
          continue

      inputs = self.tokenizer(texts, padding='longest', max_length=source_max_len,
                          truncation=True, pad_to_multiple_of=8, return_tensors='pt')
      targets = {
        'intent': torch.stack(intent_labels).to(device),
        'slot': torch.stack(slot_labels).to(device),
      }
      return inputs.to(device), targets


    def single_classify(self, examples, source_max_len):
      texts, labels = [], []
      for example in examples:
        try:
          labels.append(self._labels_to_multihot(example['attributes']))
          texts.append(example['text'])
        except(KeyError): continue

      inputs = self.tokenizer(texts, padding='longest', max_length=source_max_len,
                          truncation=True, pad_to_multiple_of=8, return_tensors='pt')
      labels = torch.stack(labels)
      return inputs.to(device), labels.to(device)
