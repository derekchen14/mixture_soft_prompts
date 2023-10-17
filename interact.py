import os, sys, pdb
import numpy as np
import random
import pickle as pkl

from torch import nn, no_grad
from torch.cuda.amp import autocast
from components.engineer import PromptEngineer
from components.datasets import FineTuneDataset

from utils.help import *
from utils.load import load_tokenizer, load_model, load_data
from utils.process import check_cache, prepare_examples, divide_crossner, divide_nluplusplus, divide_topv2
from utils.arguments import solicit_params
from assets.static_vars import STOP_TOKENS, CROSS, dtype, debug_break, accelerator

def parse_command(args, command, engineer, dataset):
  command = command.strip().lower()
  if command in ['sample', 'sa']:
    prompt_text = engineer.sample_input(args.do_guide, args.verbose)
    if args.do_guide:
      prompt_text = prompt_text[:-4]
    print(f"In: {prompt_text}")
    return prompt_text
  elif command in ['threshold', 'thresh']:
    result = input("New threshold? ")
    args.threshold = float(result)
    return ""
  elif command in ['temperature', 'temp']:
    result = input("New temperature? ")
    args.temperature = float(result)
    return ""
  else:
    return command  # just use this as the input

def run_interaction(args, model, dataset, tokenizer, engineer):
  model.eval()
  domain = CROSS[args.domain] if args.setting == 'cross' else args.domain
  engineer.attach_dataset(domain, dataset)
  prompt_text = engineer.sample_input(args.do_guide, args.verbose)
  print(f"In: {prompt_text}")

  if args.accelerate:
    model = accelerator.prepare(model)

  # for _ in range(5):
  while prompt_text not in STOP_TOKENS:
    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
    ds = args.temperature != 1.0

    if args.model == 'gpt' and args.size == 'giant':
      with autocast(dtype=torch.float16):
        with no_grad():
          output = model.generate(**inputs, max_new_tokens=args.target_max_len, early_stopping=True,
                        repetition_penalty=args.threshold, temperature=args.temperature, do_sample=ds)
    else:
      with no_grad():
        output = model.generate(**inputs, max_new_tokens=args.target_max_len, early_stopping=True,
                      repetition_penalty=args.threshold, temperature=args.temperature, do_sample=ds)

    out_string = tokenizer.decode(output[0].detach(), skip_special_tokens=True)
    if args.model == 'gpt':
      history_size = len(prompt_text)
      out_string = out_string[history_size:]
    if args.do_guide:
      print(f"4){out_string}\n")
    else:
      print(f"Out: {out_string}\n")

    result = ""
    while len(result) == 0:
      command = input("In: ")
      # command = 'sample'
      result = parse_command(args, command, engineer, dataset)
    prompt_text = result

def build_data(args, tokenizer):
  raw_data = load_data(args)
  ontology = raw_data['ontology']
    
  if args.dataset == 'crossner':
    data = divide_crossner(args, raw_data['train'], ontology, 'train')
  elif args.dataset == 'nlu++':
    data = divide_nluplusplus(args, raw_data['train'], ontology, 'train')
  elif args.dataset == 'topv2':
    data = divide_topv2(args, raw_data['train'], ontology, 'train')

  examples = prepare_examples(args, data, ontology, split='train')
  print(f"Built {len(examples)} examples")
  datasets = {'train': FineTuneDataset(args, examples, tokenizer, split='train') }  
  pkl.dump(datasets, open(cache_path, 'wb'))
  return datasets

if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  args, save_path = check_directories(args)
  set_seed(args)
  print(args)
  tokenizer = load_tokenizer(args)

  cache_file = f'fine_tune_{args.model}_direct.pkl'
  cache_path = os.path.join(args.input_dir, 'cache', args.dataset, args.domain, cache_file)
  if os.path.exists(cache_path):
    datasets = pkl.load( open( cache_path, 'rb' ) )
  else:
    datasets = build_data(args, tokenizer)

  ontology = json.load(open(f'assets/{args.dataset}/ontology.json', 'r'))
  engineer = PromptEngineer(args, ontology)
  datasets = recruit_engineer(args, datasets, engineer)
  model = load_model(args, tokenizer, save_path)
  run_interaction(args, model, datasets['train'], tokenizer, engineer)
