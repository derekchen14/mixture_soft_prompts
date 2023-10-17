import os, pdb, sys
import json
import re
import random
import glob
import csv
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import errno

from tqdm import tqdm as progress_bar
from components.soft_embedder import AttributeEmbedding, CausalEmbedding, Seq2SeqEmbedding
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, \
                         T5ForConditionalGeneration, T5Config, T5Tokenizer, logging
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, \
                        GPTNeoXForCausalLM, GPTNeoXConfig, GPTNeoXTokenizerFast, \
                        GPTJForCausalLM, GPTJConfig, AutoModelForCausalLM

from assets.static_vars import device, dtype, DATASETS, CHECKPOINTS, STOP_TOKENS, CROSS
from components.models import SentenceBERT, CVAEModel, SingleClassifier
from utils.help import model_match, find_best_model_path

logging.set_verbosity_error()

def load_data(args):
  data = {}
  for split in ['train', 'dev', 'test', 'ontology']:
    split_path = os.path.join(args.input_dir, args.dataset, f"{split}.json")
    split_data = json.load(open(split_path, 'r'))
    if split == 'ontology':
      data[split] = split_data
      example_type = 'domains'
    else:
      data[split] = split_data
      example_type = 'conversations'
    if args.verbose:
      print(f"Loaded {split} data with {len(data[split])} {example_type}")
  return data

def load_generated_data(args):
  if args.method == 'msp':
    cache_file = f'msp_{args.mixture}_{args.domain}_gen{args.num_generations}.json'
  else:
    cache_file = f'{args.method}_{args.domain}_{args.setting}.json'
  cache_path = os.path.join(args.input_dir, 'cache', args.dataset, cache_file)

  if os.path.exists(cache_path):
    res = json.load( open(cache_path, 'r') )
    print(f"Loaded {len(res)} previously generated examples from {cache_path}")
  else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cache_path)
  return res

def load_tokenizer(args):
  if args.model == 'aug': return load_pretrained_tokenizer(args.method)
  special = { 'additional_special_tokens': ['<customer>', '<agent>', '<spt>',
      '<label>', '<none>'], 'sep_token': '<sep>', 'pad_token': '<pad>'}
  token_ckpt = CHECKPOINTS[args.model][args.size]
  if args.model == 'api':
    return {}
  if args.model == 't5':
    tokenizer = T5Tokenizer.from_pretrained(token_ckpt, truncation_side='left',
            model_max_length=args.source_max_len, pad_to_multiple_of=8, truncation="longest_first")
  elif args.model == 'gpt':
    tokenizer = AutoTokenizer.from_pretrained(token_ckpt, truncation_side='left')
  elif args.model in ['godel', 'bert']:
    # token_ckpt_path = os.path.join(args.input_dir, args.model, token_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(token_ckpt)

  if args.task == 'in_context':
    tokenizer.pad_token = tokenizer.eos_token
  else:
    print(f"Adding special tokens {special}")
    tokenizer.add_special_tokens(special)

  if args.model != 'bert':
    tokenizer.padding_side = 'left'
  return tokenizer

def load_pretrained_tokenizer(method):
  if method == 'para':
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
  elif method in ['eda', 'fill']:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # not actually used
  elif method == 'rtt':
    from transformers import MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-roa')

  tokenizer.pad_token = '<pad>'
  return tokenizer

def load_pretrained_model(args, ckpt_name):
  """ Load pretrained HF models, since they will not change, we hardcode the strings """
  if args.method == 'para':
    from transformers import BartForConditionalGeneration
    model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
  elif args.method == 'fill':
    from transformers import pipeline
    model = pipeline('fill-mask', model='bert-large-cased', device='cuda:0', top_k=args.num_generations)  # 336 mil params
    return model
  elif args.method == 'eda':
    return {}
  elif args.method == 'rtt':
    from transformers import MarianMTModel  # based on BART
    try:
      model = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{ckpt_name}')
    except(OSError):
      model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-roa')

  model.to(device)
  return model

def load_model(args, tokenizer, load_dir='', ckpt_path=''):
  if args.model == 'aug': return load_pretrained_model(args, ckpt_path)
  print(f"Setting up {args.size} {args.model} model on a {dtype} machine")
  ckpt_name = CHECKPOINTS[args.model][args.size] if len(ckpt_path) == 0 else ckpt_path

  if args.model == 'gpt':
    if args.size == 'giant':
      # https://huggingface.co/docs/transformers/model_doc/gptj
      model = GPTJForCausalLM.from_pretrained(ckpt_name, low_cpu_mem_usage=True)
    else:
      config = GPT2Config.from_pretrained(ckpt_name)
      model = GPT2LMHeadModel.from_pretrained(ckpt_name, config=config, low_cpu_mem_usage=True)
  elif args.model == 'api':
    return {}
  elif args.model == 't5':
    model = T5ForConditionalGeneration.from_pretrained(ckpt_name)
  elif args.model == 'godel':
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_name)
  elif args.model == 'bert' and args.method == 'cvae':
    model = CVAEModel.from_pretrained(args, ckpt_name, tokenizer)

  model.config.pad_token = tokenizer.pad_token
  model.config.pad_token_id = tokenizer.pad_token_id
  model.resize_token_embeddings(len(tokenizer))  # transformer_check

  if args.n_gpu > 1:
    model.parallelize()
  else:
    model.to(device)
  return model

def load_classifier(args, tokenizer, ontology):
  if args.do_train:
    checkpoint = CHECKPOINTS[args.model][args.size]
  else:
    setting = "cross" if args.dataset == "nlu++" else "full"
    load_dir = os.path.join(args.output_dir, args.dataset, args.task, 'none', setting, args.domain)
    checkpoint = find_best_model_path(load_dir)
  print(f'Attempting to load {checkpoint} as best classifier')

  if args.dataset == 'crossner':
    specific_attribute = ontology[args.domain].keys()
    generic_attribute = ontology['general'].keys()
  elif args.dataset == 'topv2':
    specific_attribute = ontology['intents'][args.domain]
    generic_attribute = ontology['intents'].get('general', [])
  elif args.dataset == 'nlu++':
    specific_attribute = ontology[args.domain]['intents'].keys()
    generic_attribute = ontology['general']['intents'].keys()
  attributes = list(specific_attribute) + list(generic_attribute)
  id2label = {idx: attr for idx, attr in enumerate(attributes)}
  label2id = {attr: idx for idx, attr in enumerate(attributes)}

  if args.dataset == "topv2" and args.do_eval:
    classifier = torch.load(checkpoint, map_location=torch.device(dtype))
  else:
    classifier = SingleClassifier.from_pretrained(
      checkpoint, num_labels=len(label2id.keys()),
      problem_type="multi_label_classification",
      id2label=id2label, label2id=label2id
    )
    classifier.resize_token_embeddings(len(tokenizer))
  return classifier.to(device), label2id

def load_glove(size=300):
  if size > 0:
    root_path = "/persist"
    path_name = ".embeddings/glove/"
    file_name = f"glove.6B.{size}d.txt"
    full_path = os.path.join(root_path, path_name, file_name)
    print(f'Loading embeddings from {full_path} ...')
    return Embedder(full_path, size)
  else:
    return None  # embedder is not needed for this task

def load_best_model(args, exp_logger, tokenizer):
  load_dir = exp_logger.save_path
  print(f'Loading best finetuned model from {load_dir} ...')

  if len(args.checkpoint) > 0 and not args.accelerate:
    top_filename = args.checkpoint
    top_folder = os.path.join(load_dir, top_filename)
  else:
    # folders = glob.glob(load_dir)
    top_folder = find_best_model_path(load_dir, metric=args.metric)
  if top_folder is None:
    # raise RuntimeError(f'No models were found in {load_dir}')
    print(f'No checkpoints were found in {load_dir}, loading the default parameters')
    ckpt_path = ''
  else:
    ckpt_path = top_folder
    print(f'Attempting to load {ckpt_path} as best model')
  # checkpoint = torch.load(ckpt_path, map_location='cpu')
  # model.load_state_dict(checkpoint)
  model = load_model(args, tokenizer, load_dir, ckpt_path)
  return model

def load_best_attribute_embedder(args, model, exp_logger):
  load_dir = exp_logger.save_path
  print(f'Loading best prompt from {load_dir} ...')

  if len(args.checkpoint) > 0 and not args.accelerate:
    top_filename = args.checkpoint
    top_folder = os.path.join(load_dir, top_filename)
  else:
    top_folder = find_best_model_path(load_dir, metric=args.metric)

  if top_folder is None:
    raise RuntimeError(f'No checkpoints were found in {load_dir}')

  ckpt_path = top_folder.replace('attr_map_', '').replace('attention_', '')
  print(f'Attempting to load {ckpt_path} as best model')
  original_emb = model.get_input_embeddings()
  attr_embedding = AttributeEmbedding.from_saved_embedding(args, original_emb, ckpt_path)
  if args.model == 'gpt':
    attr_embedding.instruction_prompt = CausalEmbedding.from_saved_embedding(
                                                           args, original_emb, ckpt_path)
  else:
    attr_embedding.instruction_prompt = Seq2SeqEmbedding.from_saved_embedding(
                                                           args, original_emb, ckpt_path)
  attr_embedding.to(device)
  attr_embedding.instruction_prompt.to(device)
  model.set_input_embeddings(attr_embedding)
  # ensure embeddings are all on the same device now
  # if the model isn't parallelized
  if args.n_gpu <= 1:
    model.to(device)
  return model, attr_embedding


def load_best_soft_prompt(args, model, exp_logger):
  load_dir = exp_logger.save_path

  if len(args.checkpoint) > 0 and not args.accelerate:
    top_filename = args.checkpoint
    top_folder = os.path.join(load_dir, top_filename)
  else:
    top_folder = find_best_model_path(load_dir, args.metric)

  if top_folder is None:
    raise RuntimeError(f'No checkpoints were found in {load_dir}')

  ckpt_file = top_folder
  print(f'Attempting to load {ckpt_file} as best model')

  if args.model == 'gpt':
    soft_prompt_embed = CausalEmbedding.from_saved_embedding(args, model.get_input_embeddings(), ckpt_file)
  else:
    soft_prompt_embed = Seq2SeqEmbedding.from_saved_embedding(args, model.get_input_embeddings(), ckpt_file)
  soft_prompt_embed.to(device)
  model.set_input_embeddings(soft_prompt_embed)
  # ensure embeddings are all on the same device now
  # if the model isn't parallelized
  if args.n_gpu <= 1:
    model.to(device)
  return model

