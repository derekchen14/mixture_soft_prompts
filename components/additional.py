import json
import random
import numpy as np

prompt_one = """Show me {size} distinct utterances that all express the following attributes of {attribute} in the {domain} domain. {explanations}"""
prompt_two = """Give me examples of {size} utterances that all include the {attribute} attributes in the {domain} domain. {explanations}"""
prompt_three = """You are a helpful assistant in the {domain} domain.  You will generate {size} example utterances that each include the attributes of {attribute}. {explanations}"""
prompt_four = """We are going to perform data augmentation today. Given two example utterance, you will generate {size} new examples that all include the required attributes of {attributes} within the {domain} domain. {explanations}"""


def explain_description(args, attributes, meanings, num_attributes, values):
  if args.dataset == "topv2":
    return ""

  parts = []
  for attr, meaning in zip(attributes, meanings):
    part = attr + " refers to " + meaning
    parts.append(part)
  explanation = join_together(parts, num_attributes) + ". "
  if args.dataset == "crossner":
    explanation += f"The resulting utterance must include the following keywords {', '.join(values)}."
  
  return explanation

def make_prompts(args, example, engineer, ontology):
  attributes = []
  meanings = []
  num_attributes = 0
  for attr_name in example['attributes']:
    if args.dataset == "nlu++":
        attr_dict = ontology["general"]["intents"]
        attr_dict_domain = ontology[args.domain]["intents"]
        attr_dict.update(attr_dict_domain)
        meaning = attr_dict[attr_name]
        meanings.append(meaning)
    elif args.dataset == "crossner":
        try:
          meaning = ontology[args.domain][attr_name]
        except KeyError:
          meaning = ontology['general'][attr_name]
        meanings.append(meaning)
    attributes.append(attr_name)
    num_attributes += 1

  for value_name in example["values"]:
    if args.dataset == "nlu++":
      value_dict = ontology["general"]["slots"]
      value_dict_domain = ontology[args.domain]["slots"]
      value_dict.update(value_dict_domain)
      meaning = value_dict[value_name]
      meanings.append(meaning)
      attributes.append(value_name)
      num_attributes += 1
    elif args.dataset == "crossner":
      continue
    elif args.dataset == "topv2":
      attributes.append(value_name)
      num_attributes += 1

  attribute_string = join_together(attributes, num_attributes)
  explain_string = explain_description(args, attributes, meanings, num_attributes, example['values'])

  instruction = prompt_one.format(size=args.num_generations, attribute=attribute_string, 
                                  domain=example['domain'], explanations=explain_string)
  exemplars = find_exemplars_by_label(args, example, engineer)
  with_exemplars = join_numbers(exemplars)
  return instruction + with_exemplars

def join_together(parts, size):
  if size == 1:
    return parts[0]
  elif size == 2:
    return parts[0] + " and " + parts[1]
  elif size > 2:
    suffix = f", and {parts[-1]}"
    prefix = ', '.join(parts[:-1])
    return prefix + suffix

def find_exemplars_by_label(args, sample: dict, engineer):
  """ Finds exemplars for a given sample and its constraints"""
  exemplar_pool = []
  sample_attributes = [attr for attr in sample['attributes'] if attr in engineer.desired_attrs]

  for candidate in engineer.domain_data:
    if candidate['uuid'] == sample['uuid']: continue

    # filter out generic attributes and values
    cand_attributes = [attr for attr in candidate['attributes'] if attr in engineer.desired_attrs]
    # decide if the candidate matches the sample
    cand_match, cms = engineer._overlap(sample_attributes, cand_attributes)
    # if both constraints are met, then add the candidate to the pool
    if cand_match:
      exemplar_pool.append((candidate, cms))

  return sample_from_candidates(args, exemplar_pool, engineer)

def sample_from_candidates(args, exemplar_pool, engineer):
  if len(exemplar_pool) == 0:
    raise ValueError("exemplar_pool is empty!")

  pool_size = args.pool_size
  if args.num_shot > pool_size and args.model != 'api':
    pool_size = args.num_shot

  random.shuffle(exemplar_pool)
  exemplar_pool.sort(reverse=True, key=lambda x: x[1])
  top_exemplars = [exemplar for exemplar, score in exemplar_pool][:pool_size]
  size = min(len(top_exemplars), args.num_shot)
  chosen = np.random.choice(top_exemplars, size=size, replace=False)
  return list(chosen)

def join_numbers(exemplars):
  exemplar_text = ""
  for i, exemplar in enumerate(exemplars):
    exemplar_text += f"\n{i + 1}) {exemplar['text']}"
  exemplar_text += f"\n{len(exemplars) + 1}) "
  return exemplar_text