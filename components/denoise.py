import os, pdb, sys
import numpy as np
import random
from numpy.linalg import norm
import collections

from torch import nn
from sentence_transformers import SentenceTransformer, util
from assets.static_vars import dtype, device

def attribute_frequency(seed_data):
  attr_freq = Counter()
  for example in seed_data:
    for attribute in example['attributes']:
      attr_freq[attribute] += 1

  # convert frequency counts into normalized fraction
  frequency_ratio = {}
  total_size = sum(attr_freq.values())
  for attribute, count in attr_freq.items():
    frequency_ratio[attribute] = float(count) / total_size
  return frequency_ratio

def attribute_overlap_score(example, frequency_ratio):
  """ Motivated by the observation that certain attribute classes are over-represented in the seed data,
  we score each example according to amount of overlap between its attributes and the original data.
  As a result, higher scores indicate too much overlap, which is 'bad'. If we were to sample, this would
  have the effect of re-balancing the data so all attributes have an equal chance of appearing. """
  attribute_ratios = []
  for attribute in example['attributes']:
    attr_ratio = frequency_ratio[attribute]
    attribute_ratios.append(attr_ratio)
  # take the average so that examples with more attributes are not unfairly penalized
  overlap_rate = np.mean(attribute_ratios)
  return overlap_rate

def embedding_distance_score(new_exp, old_exp, embedder_model):
  """ We want to keep a generated example if it is semantically closer to the seed example
  Intuitively, this increases the chance that the new example is correctly labeled. This is achieved by
  calculating the cosine distance between the synthesized and its original. Once again, a larger score
  is 'bad' because larger distance means the new example is semantically farther from the original."""
  sentences = [new_exp['text'], old_exp['text']]
  new_embed, old_embed = embedder_model.encode(sentences)
  similar = nn.functional.cosine_similarity(new_embed, old_embed, dim=1)
  embed_dist = 1 - similar
  return embed_dist

def assign_noise_scores(args, synthetic_data, seed_data):
  """ Assign a noise score to each synthetic example, and also flatten the grouping. """
  frequency_ratio = attribute_frequency(seed_data)
  embedder_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

  noise_scores, flat_data = [], []
  for synth_group, old_example in zip(synthetic_data, seed_data):
    for new_example in synth_group:
      attr_over = attribute_overlap_score(new_example, frequency_ratio)
      embed_dist = embedding_distance_score(new_example, old_example, embedder_model)
      lambda_ao, lambda_ed = 0.5, 0.5  # tuning helps negligible amount, just set to half to keep it simple
      final_score = (lambda_ao * attr_over) + (lambda_ed * embed_dist)

      noise_scores.append(final_score)
      flat_data.append(new_example)
  return noise_scores, flat_data

