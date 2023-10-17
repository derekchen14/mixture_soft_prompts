import os, pdb, sys
import numpy as np
import random
import pickle as pkl
from numpy.linalg import norm
import collections
import copy

import torch
from torch import nn
from tqdm import tqdm as progress_bar
from collections import defaultdict
from assets.static_vars import CROSS, ATTRIBUTE_TOKEN_LEN, device

from sentence_transformers import SentenceTransformer, util

class PromptEngineer(object):
  """ Finds exemplars for in-context learning, and makes prompts """

  def __init__(self, args, ontology):
    self.task = args.task
    self.method = args.method
    self.mixture = args.mixture
    self.dataset_name = args.dataset
    self.do_guide = args.do_guide
    self.model = args.model

    self.pool_size = args.pool_size
    self.num_exemplars = args.num_shot

    self.domain = CROSS[args.domain] if args.setting == 'cross' and args.model != "openaigpt" else args.domain
    self.domain_data = []

    if self.method == 'msp':
      self._set_attribute_by_domain(ontology)
    else:
      self._set_attributes(ontology)
    self._set_placeholders()

    if args.task == 'in_context':
      self.sim_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
      self.embedding_map = {}

  def embed_samples(self, samples):
    print(f"Embedding {len(samples)} samples")
    for sample in progress_bar(samples):
      emb = self.sim_model.encode(sample['text'], show_progress_bar=False)
      sample.update({
        'embedding': emb
      })
      self.embedding_map[sample['uuid']] = sample

  def _overlap(self, source_constraints, target_constraints):
    matches = 0
    # there are no attributes to condition on
    if len(source_constraints) == 0:
      # find candidates that also have no attributes
      if len(target_constraints) == 0:
        matches += 1
    else:
      for constraint in source_constraints:
        if constraint in target_constraints:
          matches += 1

    is_match = matches > 0
    return is_match, matches

  def _set_attribute_by_domain(self, ontology):
    self.attr_by_domain = defaultdict(list)

    if self.dataset_name == 'crossner':
      for domain, entities in ontology.items():
        for entity, description in entities.items():
          self.attr_by_domain[domain].append(entity)
    elif self.dataset_name == 'nlu++':
      general_attributes = list(ontology['general']['intents'].keys())
      for domain in ['hotels', 'banking']:
        domain_specific = list(ontology[domain]['intents'].keys())
        self.attr_by_domain[domain] = general_attributes + domain_specific
    elif self.dataset_name == 'topv2':
      self.attr_by_domain = ontology['intents']

  def _set_all_attributes(self, ontology):
    if self.dataset_name == 'crossner':
      desired_attributes = []
      for domain_attrs in ontology.values():
        desired_attributes.extend(list(domain_attrs.keys()))
    elif self.dataset_name == 'nlu++':
      desired_attributes = []
      for domain_attrs in ontology.values():
        desired_attributes.extend(list(domain_attrs['intents'].keys()))
    elif self.dataset_name == 'topv2':
      desired_attributes = []
      for domain_attrs in ontology['intents'].values():
        desired_attributes.extend(list(domain_attrs))
    self.desired_attrs = desired_attributes

  def _set_attributes(self, ontology):
    if self.domain is None and self.dataset_name != 'actdial':
      self._set_all_attributes(ontology)
    else:
      desired_attributes = []
      if self.dataset_name == 'crossner':
        desired_attributes = list(ontology[self.domain].keys())
      elif self.dataset_name == 'nlu++':
        general_attributes = list(ontology['general']['intents'].keys())
        domain_specific = list(ontology[self.domain]['intents'].keys())
        desired_attributes = general_attributes + domain_specific
      elif self.dataset_name == 'topv2':
        desired_attributes = ontology['intents'][self.domain]
      self.desired_attrs = desired_attributes

  def _set_placeholders(self):
    """ Create placeholder text that is exactly the length of the attribute_token_len """
    num_repeat = int(ATTRIBUTE_TOKEN_LEN / 4)    # defaults to 5
    if self.model == 'gpt':
      self.attr_text = "attribute_placeholder" * num_repeat
    else:
      self.attr_text = "attribute_token " * num_repeat     # the space is necessary

  def find_exemplars_by_distance(self, sample: dict):
    candidate_exemplars = []
    for candidate in list(self.embedding_map.values()):
      if candidate['uuid'] != sample['uuid']:
        candidate_exemplars.append(candidate)

    corpus_vectors = [cand['embedding'] for cand in candidate_exemplars]
    corpus_embeddings = torch.tensor(corpus_vectors)

    sample_embedding = self.embedding_map[sample['uuid']]['embedding']
    cos_scores = util.cos_sim(sample_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=self.num_exemplars)

    exemplar_pool = []
    for score, result_idx in zip(top_results[0], top_results[1]):
      exemplar = candidate_exemplars[result_idx]
      exemplar_pool.append((exemplar, score))
    return self.sample_from_candidates(exemplar_pool)

  def find_exemplars_by_label(self, sample: dict):
    """ Finds exemplars for a given sample and its constraints"""
    exemplar_pool = []
    sample_attributes = [attr for attr in sample['attributes'] if attr in self.desired_attrs]

    for candidate in self.domain_data:
      if candidate['uuid'] == sample['uuid']: continue

      # filter out generic attributes and values
      cand_attributes = [attr for attr in candidate['attributes'] if attr in self.desired_attrs]
      # decide if the candidate matches the sample
      cand_match, cms = self._overlap(sample_attributes, cand_attributes)
      # if both constraints are met, then add the candidate to the pool
      if cand_match:
        exemplar_pool.append((candidate, cms))

    return self.sample_from_candidates(exemplar_pool)

  def sample_from_candidates(self, args, exemplar_pool):
    if len(exemplar_pool) == 0:
      raise ValueError("exemplar_pool is empty!")

    pool_size = self.pool_size
    if self.num_exemplars > pool_size and args.model != 'api':
      pool_size = self.num_exemplars

    random.shuffle(exemplar_pool)
    exemplar_pool.sort(reverse=True, key=lambda x: x[1])
    top_exemplars = [exemplar for exemplar, score in exemplar_pool][:pool_size]
    size = min(len(top_exemplars), self.num_exemplars)
    chosen = np.random.choice(top_exemplars, size=size, replace=False)
    return list(chosen)

  def get_prompt(self):
    """ returns the instruction prefix and the response separator """
    if self.task == 'synthesize':
      if self.method == 'msp':
        instruction = "" #  <instruct_prompt>
        answer_sep = f" <label> in the {self.domain} domain:"
      else:  # interaction
        if self.do_guide:
          instruction = "Here are four utterances "
          if self.dataset_name == 'crossner':
            answer_sep = f" entity types in the {self.domain} domain:"
          elif self.dataset_name == 'actdial':
            answer_sep = f" dialogue acts in the {self.domain} domain:"
          elif self.dataset_name == 'nlu++':
            answer_sep = f" intents in the {self.domain} domain:"
          elif self.dataset_name == 'topv2':
            answer_sep = f" intents and slots in the {self.domain} domain:"
        else:
          return "", ""

    elif self.task == 'in_context':
      instruction = "Predict the answer for each of these utterances "
      answer_sep = f" in the {self.domain} domain:"

    else: #  for [fine_tune, soft_prompt, end_to_end]
      instruction = ""   # empty string
      if self.dataset_name == 'actdial':
        answer_sep = "\nDialogue acts are "
      else:
        answer_sep = "<label>"

    return instruction, answer_sep

  def attach_dataset(self, domain, dataset):
    if self.method == 'msp':
      self.data_by_domain = defaultdict(list)
      for exp in dataset:
        dom = exp['domain']
        self.data_by_domain[dom].append(exp)
    else:
      filtered = [exp for exp in dataset if exp['domain'] == domain]
      self.domain_data.extend(filtered)

  def sample_input(self, do_guide, verbose):
    sample = random.choice(self.domain_data)
    if do_guide:
      if verbose:
        print(f"Seed utterance: {sample['text']}")
      input_text = self.prompt_with_exemplars(sample, add_tail=True, include_seed=True)
    else:
      context, answer_sep = self.get_prompt()
      input_text = f"{context}{sample['text']}{answer_sep}"
    return input_text

  def generate_standard_exemplars(self, args, ontology):
    """ Generates exemplars for the standard domains """
    candidate_exemplars = []
    if args.dataset == 'crossner':
      attributes = list(ontology[args.domain].keys())
    elif args.dataset == 'nlu++':
      attributes = list(ontology["general"]["intents"].keys())
    elif args.dataset == 'topv2':
      attributes = list(ontology["intents"][args.domain])
      slots = list(ontology["slots"]["general"])
    attribute_score = collections.defaultdict(int)
    all_attributes = copy.deepcopy(attributes)
    if args.dataset == "topv2":
      all_attributes.extend(slots)
    for data in self.domain_data:
      if not all_attributes:
        for attr in data['attributes']:
          attribute_score[attr] += 1
        if args.dataset == 'topv2':
          for slot in data["values"]:
            attribute_score[slot] += 1
        break

      if args.dataset == 'topv2':
        all_attr = data["attributes"] + data["values"]
      else:
        all_attr = data["attributes"]
      for attr in all_attr:
        if all_attributes and attr in all_attributes:
          candidate_exemplars.append(data)
          for attr in all_attr:
            if attr in all_attributes:
              all_attributes.remove(attr)
            attribute_score[attr] += 1
          break
        else:
          attribute_score[attr] += 1

    exemplar_pool = []
    for exemplar in candidate_exemplars:
      score = 0
      for attr in exemplar['attributes']:
        score += attribute_score[attr]
      exemplar_pool.append((exemplar, score))

    exemplars = self.sample_from_candidates(args, exemplar_pool)
    instruction, answer_sep = self.get_prompt()
    attribute_text = "from " + ", ".join([f'"{attr}"' for attr in attributes]) + " attributes"
    if args.dataset == 'topv2':
      attribute_text += " and from " + ", ".join([f'"{value}"' for value in slots]) + " slot types"
    prompt = f"{instruction}{attribute_text}{answer_sep}"
    with_exemplars = self._join_numbers(exemplars, add_tail=False, show_label=True, args=args)
    return prompt + with_exemplars + f"\n{len(exemplars) + 1}) "

  def icl_with_exemplars(self, example):
    """
    Compose prompt with exemplars including labels:
    - find exemplars based on similarity to example, assume we know the domain
    - retrieve exemplars from datasets using TF-IDF or SBERT by embedding distance
    - compose the discrete text prompt with the exemplars

    Predict the key attributes for each of these utterances in the banking domain:
    1) Q: do you offer accounts for small businesses?
       A: request_info, business, account
    2) Q: I am really not sure which address is connected to that credit card.
       A: card, credit, dont_know
    3) Q: Why is it still pending after 3 days?
       A: more_higher_after, wrong_notworking_notshowing, why
    4) Q: Which way can I increase the overdraft for my current account?
       A: request_info, account, more_higher_after, limits, current
                                   ...
    N) Q: yes, my current account movements are not showing
       A:       ( the target output is "affirm, wrong_notworking_notshowing, current, account" )
    """
    # random exemplars
    # exemplars = self.sample_from_candidates(
    #   list(zip(self.domain_data, [1] * len(self.domain_data)))
    # )

    exemplars = self.find_exemplars_by_distance(example)
    instruction, answer_sep = self.get_prompt()

    cleaned_attrs = [f'"{attr.replace("_", " ")}"' for attr in example['attributes']]
    attribute_text = self._join_constraints(cleaned_attrs)

    prompt = f"{instruction}{attribute_text}{answer_sep}"
    with_exemplars = self._join_numbers(exemplars, add_tail=False, show_label=True)
    query = f"\n{self.num_exemplars + 1}) Q: {example['text']}\n     A: "
    return prompt + with_exemplars + query


  def prompt_with_exemplars(self, example, add_tail, include_seed=False):
    """
    Compose prompt with exemplars without labels:
    - find exemplars given a sample with constraints OR just a list of constraints
    - get a prompt from datasets using the labels from seed example
    - compose the instruction prompt and attribute prompts with discrete exemplars

    Prompt is of the form:
    Here are five sentences containing <algorithm_attr> and <metric_attr> in the science domain:
    1) These Intelligent Chatbots make use of all kinds of artificial intelligence like image moderation and
       natural language understanding (NLU), natural language generation (NLG), machine learning and deep learning.
    2) It includes an ontology, created by the IEEE working group P1600.1 (originally by Ian Niles and Adam Pease).
    3) In VLDB ' 8 : Proceedings of the 34th International Conference on Very Large Data Bases , pages 422--433.
       showed that the given values for mathC / math and mathK / math generally imply relatively low accuracy
       of iteratively computed SimRank scores.
    4)
    """
    if self.method == 'msp':
      self.domain = example['domain']
      self.desired_attrs = self.attr_by_domain[self.domain]
      self.domain_data = self.data_by_domain[self.domain]

    exemplars = self.find_exemplars_by_label(example)
    if include_seed:
      if add_tail:
        exemplars.pop()             # drop the last one
      exemplars.append(example)   # replace with seed example
    instruction, answer_separator = self.get_prompt()

    if self.method == 'msp':
      attribute_text = self._encode_attributes(example)
    else:  # interactive
      cleaned_attrs = [f'"{attr.replace("_", " ")}"' for attr in example['attributes']]
      attribute_text = self._join_constraints(cleaned_attrs)

    prompt = f"{instruction}{attribute_text}{answer_separator}"
    with_exemplars = self._join_numbers(exemplars, add_tail)
    return prompt + with_exemplars

  def _encode_attributes(self, example):
    if self.dataset_name == 'nlu++':
      encoding = self._encode_nluplusplus(example)
    elif self.dataset_name == 'crossner':
      encoding = self._encode_crossner(example)
    elif self.dataset_name == 'topv2':
      encoding = self._encode_topv2(example)
    return encoding

  def _encode_nluplusplus(self, example):
    num_attr = len(set(example['attributes']))
    intents = self.attr_text * num_attr if self.mixture == 'concat' else self.attr_text
    slots = []
    for value in example['values']:
      slot = value.split('[')[0]
      slots.append(slot.strip())
    encoding = intents + '<sep>' + ', '.join(slots)
    return encoding

  def _encode_crossner(self, example):
    parts = example['target'].split(' <sep> ')
    encoding = ""
    types = set()
    values = []
    # take advantage of knowing which entities are supposed to be present
    for part in parts:
      if self.mixture == 'concat':
        for entity_type in example['attributes']:
          if entity_type in part and entity_type not in types:
            encoding += self.attr_text
            types.add(entity_type)
      else:
        encoding += self.attr_text
      for entity_val in example['values']:
        if entity_val in part:
          values.append(entity_val)
    
    encoding += '<sep>'
    encoding += ', '.join(values)
    return encoding

  def _encode_topv2(self, example):
    if self.mixture == 'concat':
      intents = self.attr_text * len(set(example['attributes']))
      slots = self.attr_text * len(set(example['values']))
      intent_and_slot = intents + slots
    else:
      intent_and_slot = self.attr_text * 2
    values = self._slotval_text(example['target'])
    encoding = intent_and_slot + '<sep>' + values
    return encoding

  def _join_numbers(self, exemplars, add_tail, show_label=False, args=None):
    exemplar_text = ""
    for i, exemplar in enumerate(exemplars):
      if show_label:
        exemplar_text += f"\n{i+1}) Q: {exemplar['text']}"
        if args.icl_type == 'base':
          exemplar_text += f"\n     A: {exemplar['target']}"
        elif args.icl_type == 'cot':  # For the Chain-of-thought
          if args.dataset == 'nlu++':
            attribute_size = len(exemplar['attributes'])
            answer_string = \
              f"Given the utterance above, please tell me what domain this is? {args.domain} \n" \
              f"        For this utterance in the {args.domain} domain, how many attributes are present? {attribute_size}\n" \
              f"        Within this utterance, there are {attribute_size} key attributes, which ones are they?\n" \
              f"        Answer: "
            answer_string += exemplar['target']
            exemplar_text += f"\n     A: {answer_string}"
          elif args.dataset == 'crossner':
            unique_attribute = ", ".join((exemplar['attributes']))
            exemplar_list = exemplar['target'].split("<sep>")
            ner_number = len(exemplar_list)
            unique_attribute_size = len(exemplar['attributes'])
            answer_string = \
              f"Given the utterance above, please tell me what domain this is? {args.domain} \n" \
              f"        For this utterance in the {args.domain} domain, which attribute types are present? {unique_attribute}\n" \
              f"        Within this utterance, there are {ner_number} entities within {unique_attribute_size} attributes. What are these entities?\n" \
              f"        Answer: "
            answer_string += exemplar['target']
            exemplar_text += f"\n     A: {answer_string}"
          elif args.dataset == 'topv2':
            attribute_string = ", ".join([i.replace("_", " ")for i in exemplar['attributes']])
            attribute_size = len(exemplar['attributes'])
            slot_size = len(exemplar['values'])
            answer_string = \
              f"Given the utterance above, please tell me what domain this is? {args.domain} \n" \
              f"        For this utterance in the {args.domain} domain, which attributes are present? {attribute_string}\n" \
              f"        Within this utterance, there are {attribute_size} attributes of {attribute_string} which lead to {slot_size} slot-values.  What are these slot-values?\n" \
              f"        Answer: "
            answer_string += exemplar['target'].split("<sep>")[-1].strip()
            exemplar_text += f"\n     A: {answer_string}"
      else:
        exemplar_text += f"\n{i+1}) {exemplar['text']}"

    if add_tail:
      exemplar_text +=  f"\n{len(exemplars) + 1}) "
    return exemplar_text

  def _join_constraints(self, strings: list):
    prompt = ""
    if len(strings) == 0:
      prompt += "without any special"
      return prompt

    prompt += "containing "
    if len(strings) == 1:
      prompt += strings[0]
    elif len(strings) == 2:
      prompt += f'{strings[0]} and {strings[1]}'
    else:
      prompt += f'{", ".join(strings[:-1])} and {strings[-1]}'
    return prompt

  def _slotval_text(self, target):
    slotvals = []
    _, value_string = target.split(' <sep> ')

    value = ""
    inside = False
    for char in value_string:
      if inside:
        if char == ']':
          inside = False
          slotvals.append(value)
          value = ""
        else:
          value += char

      else:  # outside
        if char == '[':
          inside = True

    slot_val_string = ', '.join(slotvals)
    return slot_val_string
