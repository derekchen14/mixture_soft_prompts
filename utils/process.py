import os, pdb, sys
import random
import json
import math
import pickle as pkl
import numpy as np
import re

from assets.static_vars import *
from components.datasets import (
  InContextDataset, FineTuneDataset, ClassifyDataset,
  ConditionalLMDataset, SoftPromptMixDataset, DExpertDataset, CVAEDataset
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm as progress_bar
from collections import defaultdict, Counter
from utils.load import load_generated_data

def check_cache(args):
  if args.method == 'dexpert':
    cache_file = f'{args.task}_{args.model}_{args.method}_direct.pkl'
  else:
    cache_file = f'{args.task}_{args.model}_direct.pkl'
  if args.domain:
    cache_path = os.path.join(args.input_dir, 'cache', args.dataset, args.domain, cache_file)
  else:
    cache_path = os.path.join(args.input_dir, 'cache', args.dataset, cache_file)
  use_cache = not args.ignore_cache

  if os.path.exists(cache_path) and use_cache:
    res = pkl.load( open( cache_path, 'rb' ) )
    if args.do_train:
      print(f"Loaded {len(res['train'])} train and {len(res['dev'])} dev examples from {cache_path}")
    elif args.do_eval:
      print(f"Loaded {len(res['test'])} test examples from {cache_path} for evaluation")
    return res, True
  elif args.task == 'classify' and args.do_eval:
    return cache_path, True
  else:
    print(f'Creating new dataset for {args.dataset.upper()} from scratch ...')
    return cache_path, False

def build_nluplusplus(args, data, ontology, split):
  """ Example return:
  {'uuid': 'dev-banking-1983',
   'text': 'When will you finally enable contactless payments with Marks and Spencer?',
   'target': 'when;make_open_apply_setup_get_activate;contactless;transfer_payment_deposit',
   'domain': 'banking',
   'attributes': ['when', 'make_open_apply_setup_get_activate', 'contactless', 'transfer_payment_deposit'],   
   'values': ['company_name'],
   'constraint_str': 'when , make_open_apply_setup_get_activate , contactless , transfer_payment_deposit <sep> company_name' } 
  """
  examples = []
  for idx, utterance in progress_bar(enumerate(data), total=len(data)):
    if 'intents' not in utterance:
      continue
    intents = list(set(utterance['intents']))
    slot_info = utterance.get('slots', {})
    slots = []
    for slot_category in slot_info.keys():
      # slot_dict: {'text': the text, 'span': [start, end], 'value' or 'values': dict or value, depending}
      slots.append(f"{slot_category}")

    text = utterance['text'].rstrip()
    if text.endswith(" .") or text.endswith(" ?") or text.endswith(" !"):
      punctuation = text[-1]
      text = text[:-2] + punctuation + " "

    domain = utterance['domain']
    utterance_uid = f"{split}-{domain}-{idx+1}"

    constraint_str = ' , '.join(intents) + ' <sep> ' +  ' , '.join(slot_info.keys())

    examples.append({
      'uuid': utterance_uid,
      'text': utterance['text'],
      'target': ";".join(sorted(intents)),
      'domain': domain,
      'attributes': intents,      # list of unique intents
      'values': slots,            # list of slots
      'constraint_str': constraint_str 
    })
  return examples

def build_actdial(args, data, ontology, split):
  examples = []
  ont_mapping = {label: idx for idx, label in enumerate(ontology)}

  for conversation in progress_bar(data, total=len(data)):
    for utterances in conversation:

      target_texts, target_ids = set(), set()
      utterance = ""
      for segment in utterances:
        label = segment['label']
        if label == 'No Need to Label':
          continue

        target_ids.add(ont_mapping[label])
        if "_" in label:
          _, label = label.split("_")
        target_texts.add(label)

        text = segment['segment'].rstrip()
        if text.endswith(" .") or text.endswith(" ?") or text.endswith(" !"):
          punctuation = text[-1]
          text = text[:-2] + punctuation + " "
        utterance += text

      target = ";".join(sorted(list(target_texts)))
      exp = {'uuid': segment['uuid'], 'text': utterance, 'target': target, 'label_id': list(target_ids)}
      examples.append(exp)
  return examples

def build_banking(args, data, ontology, split):
  examples = []
  ontology = {label.replace('_', ' '): i for i, label in enumerate(ontology)}

  for idx, exp in progress_bar(enumerate(data), total=len(data)):
    convo_id = split + "-" + str(idx + 1)
    example = {'uuid': convo_id, 'text': exp[0], 'target': exp[1], 'label_id': ontology[exp[1]]}
    examples.append(example)
  return examples

def build_crossner(args, data, ontology, split):
  """ Example output
  {'uuid': 'train-science-402',
   'text': 'August Kopff, a colleague of Wolf at Heidelberg, then discovered 617 Patroclus 
                eight months after Achilles trojan',
   'target': 'August Kopff [scientist] <sep> Wolf [scientist] <sep> Heidelberg [location] <sep>
                617 Patroclus [astronomy] <sep> Achilles trojan [astronomy]',
   'domain': 'science',
   'attributes': ['scientist', 'location', 'astronomy'],
   'values': ['August Kopff', 'Wolf', 'Heidelberg', '617 Patroclus', 'Achilles trojan'],
   'constraint_str': 'August Kopff [scientist] , Wolf [scientist] , Heidelberg [location] ,
                617 Patroclus [astronomy] , Achilles trojan [astronomy]'}
  """
  examples = []
  for utterance in progress_bar(data, total=len(data)):
    domain = utterance["domain"]
    targets = []
    entity_types, entity_values = set(), set()
    for ent_type, ent_val in utterance["labels"]:
      entity_types.add(ent_type)
      entity_values.add(ent_val)
      # 'Jim [person] <sep> Acme Corp. [organization] <sep> 2006 [time]'
      targets.append(f"{ent_val} [{ent_type}]")
    if len(entity_types) == 0 and args.task == 'synthesize':  continue

    examples.append({
      'uuid': utterance['uid'],
      'text': utterance['text'],
      'target': ' <sep> '.join(targets),
      'domain': domain,
      'attributes': list(entity_types),    # unique entity types
      'values': list(entity_values),       # entity values
      'constraint_str': ' , '.join(targets)
    })
  return examples

def build_topv2(args, data, ontology, split):
  """ Example return:
  {'uuid': 'train-alarm-32',
   'text': 'Cancel my alarm for 10',
   'target': 'delete alarm , get time <sep> alarm name [get time] , date time [for 10]',
   'domain': 'alarm',
   'attributes': ['delete_alarm' , 'get_time'],
   'values': ['alarm_name', 'date_time'],
   'constraint_str': 'delete_alarm , get_time <sep> alarm_name [get time] , date_time [for 10]'   }
  """
  target_domains = [dom for dom, dom_type in ontology["domains"] if dom_type == "target"]
  if (args.task != 'synthesize') and (args.domain not in target_domains):
    raise ValueError('The domain for TopV2 must be set either [weather] or [reminder]')

  examples = []
  for utterance in progress_bar(data, total=len(data)):
    target = utterance["label"].strip()
    # regenerate utterance["label"] to target string.
    # 1. clean data - remove the fillers;
    simplified_target = simplify_topv2(target)
    # 2. generate intents list and slots_types list and corresponding slots_values list
    intents, slots_types, slots_values = decouple_topv2(simplified_target)

    intents_string = " , ".join(intents)
    slots = []
    for type, value in zip(slots_types, slots_values):
      slots.append(type + " " + "[" + value + "]")
    slots_string = " , ".join(slots)

    final_target = intents_string + " <sep> " + slots_string
    orig_intents = [intent.replace(' ', '_') for intent in intents]
    orig_slots_types = [slot.replace(' ', '_') for slot in slots_types]

    examples.append({
      'uuid': utterance["uid"],
      'text': utterance["text"],
      'target': final_target.replace('_', ' '),
      'domain': utterance["domain"],
      'attributes': orig_intents,     # list of intents
      'values': orig_slots_types,     # list of slot types
      'constraint_str': final_target
    })
  return examples

def divide_crossner(args, data, ontology, split):
  """ Given the data from a split, decide which data to keep.
  During training consider adding data from the 'general' domain
  For dev and test, only use data from the specific target domain
  """
  target_data, source_data = [], []  # limited target domain data and source data from other domains
  for example in data:

    if args.task == 'synthesize':
      # when training for synthesis, use all data
      if example['domain'] == 'general' and args.do_train and split=='train':
        if random.random() < 0.5:
          source_data.append(example)  # add a bit more data during training
      elif (args.domain is None or example['domain'] == args.domain):
        target_data.append(example) # always use in-domain data
      
    else:
      # always keep all data from the target domain
      if example['domain'] == args.domain:
        target_data.append(example)
      # add some additional data for training from the general domain 
      elif example['domain'] == 'general' and split == 'train':
        if random.random() < 0.5:  # uncomment to speed up debugging
          source_data.append(example)  

  if args.setting == 'few_shot' and split == 'train':
    assert (args.num_shot == 50)  # comment out if you are certain you want a different number of shots
    target_data = few_shot_crossner(args, target_data, ontology)
  final_data = source_data + target_data

  if args.verbose:
    print(f"Started with {len(data)} examples in {split} with {len(target_data)} from {args.domain}")
    print(f"Ended with {len(target_data)} {args.domain} + {len(source_data)} general = {len(final_data)} examples")
  return final_data

def divide_topv2(args, data, ontology, split):
  source_domains = [dom for dom, dom_type in ontology["domains"] if dom_type == "source"]
  target_domains = [dom for dom, dom_type in ontology["domains"] if dom_type == "target"]

  target_data, source_data = [], []  # limited target domain data and source data from other domains
  for utterance in data:
    # resolve target data
    if utterance["domain"] in target_domains:
      if not utterance['uid'].endswith(args.setting): continue   # either few_shot or full
      if args.task == 'synthesize' and args.do_train:     # always include during training
        target_data.append(utterance)
      elif utterance["domain"] == args.domain and utterance["uid"].endswith(args.setting):
        target_data.append(utterance)   # should match the given domain and setting as well

    # resolve source data, surprisingly same way for all task setups
    if utterance["domain"] in source_domains and split == "train" and args.do_train:
      if utterance["domain"] in ["alarm", "navigation"]:
        keep_rate = 0.15
      elif utterance["domain"] in ["music", "timer"]:
        keep_rate = 0.25
      elif utterance["domain"] in ["messaging"]:  
        keep_rate = 0.3
      else:  # event
        keep_rate = 0.35

      if random.random() < keep_rate:  # filter out source data to speed up training
        source_data.append(utterance)
  final_data = source_data + target_data

  if args.verbose:
    print(f"Started with {len(data)} examples in {split} with {len(target_data)} from {args.domain}")
    print(f"Ended with {len(target_data)} + {len(source_data)} = {len(final_data)} examples")
  return final_data

  if args.verbose:
    print(f"Started with {len(data)} examples, ended with  will train on {len(final_data)}")

  return final_data

def divide_nluplusplus(args, data, ontology, split):
  final_data = []
  if args.setting == "cross":
    allowed_domains = [domain for domain in ontology.keys() if domain != "general"]
    if args.domain not in allowed_domains:
      raise ValueError('Please select an appropriate domain from 2 allowed options')

    general_candidates = ontology["general"]["intents"].keys()
    for utterance in progress_bar(data, total=len(data)):
      if "intents" not in utterance:
        continue
      if split == "train":
        if not args.do_train and args.task == 'synthesize':
          # when synthesizing, which we do from the train data, use data in-domain to get constraints
          target_domains = [args.domain]
        else:
          # when training the models for args.domain, go cross and use training data only from other domains. then save it at args.domain
          target_domains = [dom for dom in allowed_domains if dom != args.domain]
        if utterance["domain"] in target_domains:
          final_data.append(utterance)
      else:
        # for eval sets, only evaluate on general domain
        target_general_tokens = [tt for tt in utterance["intents"] if tt.strip() in general_candidates]
        if target_general_tokens and utterance["domain"] == args.domain:
          final_data.append(utterance)
  elif args.setting == "kfold":
    if args.num_shot == 20:
      train_fold = int(args.domain)
      for utterance in progress_bar(data, total=len(data)):
        if split == "train":
          if utterance["fold"] == train_fold:
            final_data.append(utterance)
        else:
          if utterance["fold"] != train_fold:
            final_data.append(utterance)
    elif args.num_shot == 10:
      train_folds = [int(args.domain) * 2, int(args.domain) * 2 + 1]
      for utterance in progress_bar(data, total=len(data)):
        if split == "train":
          if utterance["fold"] in train_folds:
            final_data.append(utterance)
        else:
          if utterance["fold"] not in train_folds:
            final_data.append(utterance)
  elif args.setting == "full":
    test_folds = [int(args.domain) * 2, int(args.domain) * 2 + 1]
    for utterance in progress_bar(data, total=len(data)):
      if split == "train":
        if utterance["fold"] not in test_folds:
          final_data.append(utterance)
      else:
        if utterance["fold"] in test_folds:
          final_data.append(utterance)
  else:
    for utterance in progress_bar(data, total=len(data)):
      if utterance["domain"] == args.domain:
        final_data.append(utterance)
  return final_data

def simplify_topv2(target):
  target_list = []
  target_list_tmp = target.split(" ")
  index = 0
  while index < len(target_list_tmp):
    if target_list_tmp[index].startswith("[") or target_list_tmp[index].startswith("]"):
      target_list.append(target_list_tmp[index])
      value = []
      while index < len(target_list_tmp) - 1 and \
              not (target_list_tmp[index + 1].startswith("[") or target_list_tmp[index + 1].startswith("]")):
        value.append(target_list_tmp[index + 1])
        index += 1
      target_list.append(" ".join(value))
      index += 1
      continue
    else:
      target_list.append(target_list_tmp[index])
      index += 1

  simplified_target = target_list[0] + " "
  for index in range(1, len(target_list) - 1):
    prev_token = target_list[index - 1]
    current_token = target_list[index]
    next_token = target_list[index + 1]
    if (prev_token.startswith("[") or prev_token.startswith("]")) and next_token.startswith("["):
      continue
    if prev_token.startswith("]") and next_token.startswith("]"):
      continue
    simplified_target += current_token + " "
  simplified_target += target_list[-1]

  return simplified_target

def decouple_topv2(simplified_target):
  intents = []
  slots_types = []
  slots_values = []
  token_list = simplified_target.split()
  idx = 0
  while idx < len(token_list):
    token = token_list[idx]
    if token.startswith("[IN:"):
      intent = token[4:].lower()
      # intent = intent.replace("_", " ")
      intents.append(intent)
      idx += 1
    elif token.startswith("[SL:"):
      slot = token[4:].lower()
      # slot = slot.replace("_", " ")
      slots_types.append(slot)
      slots_value = []
      idx += 1
      next_token = token_list[idx]
      if next_token.startswith("[IN:") or next_token.startswith("[SL:"):
        slots_values.append(next_token[4:].lower()) #.replace("_", " "))
        continue
      else:
        slots_value.append(next_token.lower())
      idx += 1
      while token_list[idx] != "]" and not token_list[idx].startswith("["):
        slots_value.append(token_list[idx].lower())
        idx += 1
      slots_values.append(" ".join(slots_value))
    else:
      idx += 1

  return intents, slots_types, slots_values

def few_shot_crossner(args, target_data, ontology):
  # get at least args.num_shot number of examples per entity
  # if the total number of entity is smaller than args.num_shot, get all of them.
  entities = ontology[args.domain]
  entities_count = {key: 0 for key in entities}
  few_shot_data = []
  index = 0
  while index < len(target_data) and not all(value >= args.num_shot for value in entities_count.values()):
    example = target_data[index]
    skip = []
    for name, _ in example["labels"]:
      if name in entities:
        skip.append(entities_count[name] >= args.num_shot)
        entities_count[name] += 1
    if not all(skip):
      few_shot_data.append(example)
    index += 1

  if args.verbose:
    print(f"From {len(target_data)} down to {len(few_shot_data)} for {args.domain}")
  return few_shot_data

def remix_crossner(raw_data, domain, split):
  if split == 'train':
    data = []
    for utterance in raw_data['train']:
      if utterance['domain'] == domain:
        data.append(utterance)      # keep all of in-domain train data
      elif utterance['domain'] == 'general':
        if random.random() < 0.5:
          data.append(utterance)    # mix in some generic NER data
    for utterance in raw_data['dev']:
      if utterance['domain'] == domain:
        data.append(utterance)      # add all of in-domain dev data

  elif split == 'dev':              
    raw_test = [x for x in raw_data['test'] if x['domain'] == domain]
    size = len(raw_test) // 2
    data = raw_test[:size]  # use first half of test

  elif split == 'test':             
    data = [x for x in raw_data[split] if x['domain'] == domain]          # just leave the same
  return data

def remix_nluplusplus(raw_data, domain, split):
  data = [x for x in raw_data[split] if x['domain'] != domain]
  return data

def remix_topv2(raw_data, domain, split):
  if split == 'train':
    in_domain_train = [x for x in raw_data[split] if domain is None or x['domain'] == domain]
    in_domain_dev = [x for x in raw_data['dev'] if domain is None or x['domain'] == domain]
    data = in_domain_train + in_domain_dev
  else:
    in_domain_test = [x for x in raw_data[split] if domain is None or x['domain'] == domain]
    if split == 'dev':  
      size = len(in_domain_test) // 2
      data = in_domain_test[:size]  # use first half of test
    elif split == 'test':             
      data = in_domain_test
  return data

def get_dataloader(args, dataset, split='train'):
  if args.model == 'api':
    dataset.collate_func
  sampler = RandomSampler(dataset) if dataset.shuffle else SequentialSampler(dataset)
  collate = dataset.collate_func
  dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)
  print(f"Loaded {split} data with {len(dataloader)} batches")
  return dataloader

def finalize_splits(args, raw_data, ontology, split):
  # for classify we mix in the dev and test set since we want to train an oracle model
  if args.task == 'classify':
    if args.dataset == 'crossner':
      remixed_data = remix_crossner(raw_data, args.domain, split)
    elif args.dataset == 'nlu++':
      assert(args.setting == 'cross')
      remixed_data = remix_nluplusplus(raw_data, args.domain, split)
    elif args.dataset == 'topv2':
      assert(args.setting == 'full') or (args.setting == 'few_shot' and args.task == 'synthesize')
      remixed_data = remix_topv2(raw_data, args.domain, split)
    return remixed_data

  # for other tasks, we consider how much data from other domains to include
  right_task = args.task in ['in_context', 'fine_tune', 'synthesize', 'soft_prompt', 'end_to_end']
  if args.dataset == 'crossner' and right_task:
    data = divide_crossner(args, raw_data[split], ontology, split)
  elif args.dataset == 'nlu++' and right_task:
    data = divide_nluplusplus(args, raw_data[split], ontology, split)
  elif args.dataset == 'topv2' and right_task:
    data = divide_topv2(args, raw_data[split], ontology, split)
  else:
    data = raw_data[split]
  return data

def prepare_examples(args, data, ontology, split):
  """ Each example is a dict which should have:
    uuid - universal utterance identifier
    text - input context for the encoder
    target - desired output text from the decoder
    domain - string to represent the domain (eg. music, politics, banking, alarm)
    attributes - list of the main aspect for that dataset, such as intent or entity type
    values - list of the secondary aspect for that dataset, such as slot type or entity value
  """
  if args.dataset == 'actdial':  # ActDial from Daily Dialog
    examples = build_actdial(args, data, ontology, split)
  elif args.dataset == 'banking':   # Banking 77 intents
    examples = build_banking(args, data, ontology, split)
  elif args.dataset == 'crossner':   # Banking 77 intents
    examples = build_crossner(args, data, ontology, split)
  elif args.dataset == 'nlu++':   # NL understanding plus plus
    examples = build_nluplusplus(args, data, ontology, split)
  elif args.dataset == 'topv2':   # Banking 77 intents
    examples = build_topv2(args, data, ontology, split)
  return examples

def process_data(args, cache_path, raw_data, tokenizer=None):
  ontology = raw_data['ontology']
  datasets = {}
  for split in ['train', 'dev', 'test']:
    data = finalize_splits(args, raw_data, ontology, split)
    examples = prepare_examples(args, data, ontology, split)
    if args.model == "api":
      datasets[split] = examples
      continue

    if args.task == 'synthesize':
      if args.method == 'clm':
        datasets[split] = ConditionalLMDataset(args, examples, tokenizer, split)
      elif args.method == 'msp':
        datasets[split] = SoftPromptMixDataset(args, examples, tokenizer, split)
      elif args.method == 'dexpert':
        datasets[split] = DExpertDataset(args, examples, tokenizer, split)
        if args.dataset == 'nlu++':
          attributes = set()
          for domain_dict in ontology.values():
            attributes |= set(domain_dict['intents'])
            attributes |= set(domain_dict['slots'])
        elif args.dataset == 'crossner':
          attributes = set()
          for entities in ontology.values():
            attributes |= set(entities)
        elif args.dataset == 'topv2':
          attributes = set()
          for domain_list in ontology['intents'].values():
            attributes |= set(domain_list)
          for slot_list in ontology['slots'].values():
            attributes |= set(slot_list)
        for attribute in attributes:
          if attribute not in datasets:
            datasets[attribute] = {}
          datasets[attribute][split] = DExpertDataset(args, examples, tokenizer, split, attribute)
      elif args.method in ['eda', 'para', 'rtt', 'fill']:
        datasets[split] = FineTuneDataset(args, examples, tokenizer, split)
      elif args.method == 'cvae':
        datasets[split] = CVAEDataset(args, examples, tokenizer, split)

    elif args.task == 'end_to_end':
      if split == 'train':
        seed_data = examples
        generated_data = load_generated_data(args)
        examples = generated_data + seed_data   # combine two lists
      datasets[split] = FineTuneDataset(args, examples, tokenizer, split)

    elif args.task == 'in_context':
      datasets[split] = InContextDataset(args, examples, tokenizer, split)
    elif args.task == 'classify':
      datasets[split] = ClassifyDataset(args, examples, tokenizer, split)
    else:
      datasets[split] = FineTuneDataset(args, examples, tokenizer, split)

    if split in datasets:
      print(f"Running with {len(datasets[split])} {split} examples")
  pkl.dump(datasets, open(cache_path, 'wb'))

  return datasets
