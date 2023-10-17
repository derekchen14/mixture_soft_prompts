import os, pdb, sys, glob
import numpy as np
import pickle as pkl
import torch
import random
import json
import re
import shutil

from collections import defaultdict, Counter
from tqdm import tqdm as progress_bar
from assets.static_vars import device, dtype
from copy import deepcopy
from transformers import get_scheduler
from torch.optim import AdamW
import torch_optimizer as ada_optim
from retry import retry
import openai

from assets.static_vars import CHECKPOINTS

def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

def setup_gpus(args):
  n_gpu = 0  # set the default to 0
  if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
  args.n_gpu = n_gpu
  if n_gpu > 0:   # this is not an 'else' statement and cannot be combined
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
  return args

def check_directories(args):
  dataset_path = os.path.join(args.output_dir, args.dataset)
  save_path = os.path.join(dataset_path, args.task, args.method, args.setting)

  if args.method == 'msp':
    save_path = os.path.join(save_path, args.mixture)
  elif args.task == 'synthesize' and args.dataset != 'nlu++':
    pass # For CrossNER and TOPV2, we train synthesis on all data.
  elif args.domain:
    save_path = os.path.join(save_path, args.domain)

  if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    print(f"Created {dataset_path} for {args.dataset} results")
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Created {save_path} directory")

  cache_path = os.path.join(args.input_dir, 'cache', args.dataset)
  if args.domain:
    cache_path = os.path.join(cache_path, args.domain)

  if args.method == 'msp':
    cache_path = os.path.join(cache_path, args.mixture)

  if args.filter:
    assert(args.num_generations == 5)  # increase by 25%, which will later get filtered out
  else:
    assert(args.num_generations == 4)

  if not os.path.exists(cache_path):
    os.makedirs(cache_path)
    print(f"Created {cache_path} directory")
  if args.task == 'end_to_end':
    assert(args.method != 'none')  # specify a DataAug or CtrlTextGen method

  if args.dataset == 'topv2' and args.do_train and args.task not in ['synthesize', 'classify']:
    assert(args.metric == 'accuracy')
  if args.do_train or args.do_eval:
    assert(args.qualify or args.quantify)
  if (args.task == 'synthesize' and args.do_train) or args.dataset == 'actdial':
    pass  # unless you are training any model for data synthesis
  else:
    assert(args.domain is not None)  # you must specify a domain
  if args.accelerate:
    print("  --->  Remember to uncomment the accelerate.device !  <---")
  if args.debug:
    args.log_interval /= 10

  return args, save_path

def setup_optimization(args, model, total_steps):
  no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
  if args.task == 'soft_prompt':
    optimizer_grouped_parameters = model.parameters() # model is actually soft prompt embeds
  else:
    optimizer_grouped_parameters = [
      {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
      },
      {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0},
    ]

  warmup = int(total_steps * args.warmup_steps)
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

  schedule_type = 'cosine' if dtype == 'cuda' else 'constant'
  scheduler = get_scheduler(schedule_type, optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)
  return optimizer, scheduler

def review_inputs(args, inputs, targets, tokenizer):
  if args.debug and args.verbose:
    if args.model == 'gpt':
      tbd = tokenizer.batch_decode(inputs['input_ids'])
      print(f"Batch with {len(tbd)} items")
      for batch_item in tbd:
        print(batch_item.replace(tokenizer.pad_token, '|'))
    else:
      triple_i = inputs['input_ids'].detach().clone()
      sp_token_id = tokenizer('=')['input_ids'][0]
      triple_i[triple_i==-1] = sp_token_id  # transform soft prompt token placeholder into "=" token
      tbdi = tokenizer.batch_decode(triple_i)
      targi = targets.detach().clone()
      targi[targets==-100] = 0      # transform skip label into <pad> token
      tbdt = tokenizer.batch_decode(targi)
      print(f"Original batch contains {len(tbdi)} items")
      display_size = 7
      for batch_input, batch_target in zip(tbdi, tbdt):
        print(batch_input.replace('<pad>', '|'))
        print(batch_target.replace('<pad>', '|'))
        display_size -= 1
        if display_size == 0: break
    pdb.set_trace()

def batchify(args, turn, global_id, prior_pred_state):
  """ returns a list of batches where the ground_truth prev_state has been
  replaced with the predicted prior_state from the previous turn """
  batches = []

  convo_id, turn_str = global_id.split('_')
  turn_count = int(turn_str)
  if turn_count == 1:
    prev_state = {}
  else:
    prev_gid = f"{convo_id}_{turn_count - 1}"
    prev_state = prior_pred_state[prev_gid]

  batch = []
  for example in turn:
    example['prev_state'] = prev_state
    batch.append(example)

    if len(batch) == args.batch_size:
      batches.append(batch)
      batch = []

  if len(batch) > 0:
    batches.append(batch)
  return batches

def show_samples(debug_mode, full_data, size=20):
  if debug_mode:
    if size > len(full_data):
      print(f"Only {len(full_data)} examples were found, but wanted {size}")
    samples = np.random.choice(full_data, size, replace=False)
    for sample in samples:
      print(sample)
    pdb.set_trace()

def get_all_checkpoints(args, load_dir):
  print('Loading all finetuned models ...')
  filenames = [f for f in os.listdir(load_dir) if f.endswith('.pt')]
  if len(filenames) == 0:
    raise RuntimeError(f'No models were found in {load_dir}')

  checkpoints = []
  for fname in filenames:
    ckpt_path = os.path.join(load_dir, fname)
    print(f'Found {ckpt_path} in directory')
    checkpoints.append(ckpt_path)
  return checkpoints

def memstat(message):
  malloc = torch.cuda.memory_allocated()
  human_malloc = str(round( (malloc / 1000000), 2)) + "MB"
  maxmem = torch.cuda.max_memory_allocated()
  human_maxmem = str(round( (maxmem / 1000000), 2)) + "MB"
  print(f"{message} -- Current memory: {human_malloc}, Max: {human_maxmem}")

def display_domains(args, datasets):
  dataset = datasets['train']
  print("Total examples: ", len(dataset))
  domains = Counter()
  for example in dataset:
    dom = example['domain']
    domains[dom] += 1
  for dom, count in domains.items():
    print(dom, count)

def model_match(fname, args):
  """
  check if the ckpt with path 'fname' fits the current args

  follow the format:
  f'results/{dataset}/{task}/{model}_{size}/{prompt_style}_lr{}_{saliency}_epoch{}_acc{}.pt'
  """
  model_type, model_size = fname.split('/')[-2].split("_")[0], fname.split('/')[-2].split("_")[1]
  if len(fname.split('/')[-1].split("_")) != 5:
    return False
  prompt_style, lr, saliency, epoch, _ = fname.split('/')[-1].split("_")

  type_match = model_type == args.model
  size_match = model_size == args.size
  prompt_match = prompt_style == args.prompt_style
  lr_match = lr == f'lr{args.learning_rate}'

  if type_match and size_match and prompt_match and lr_match:
    return True
  return False

def recruit_engineer(args, datasets, engineer):
  for split, dataset in datasets.items():
    if args.task == 'in_context' or args.do_guide or args.method == 'msp':
      if split == 'train':
        engineer.attach_dataset(args.domain, dataset)
    dataset.engineer = engineer
  return datasets


def _flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def find_best_model_path(folder_path, metric='accuracy'):
  folders = glob.glob(os.path.join(folder_path, "*pt"))
  top_acc, top_folder = 0, ''

  for fname in folders:
    re_str = r'.*/acc([0-9]{3}).*\.pt$'
    current_score = re.findall(re_str, fname)
    score = int(current_score[0]) if len(current_score) > 0 else 0

    if score > top_acc:
      top_acc = score
      top_folder = fname

  if len(top_folder) == 0:
    print(f'No checkpoints were found in {folder_path}, skipping this attribute model')
    return None
  return top_folder


@retry(Exception, tries=5, delay=2)
def gpt_chat_response(args, final_prompt):
  openai.api_key = args.openai_key
  gpt_model_name = CHECKPOINTS[args.model][args.size]
  completion = openai.ChatCompletion.create(
              model=gpt_model_name,
              messages=[{"role": "system", "content": final_prompt}],
              temperature=args.temperature
            )
  response = completion['choices'][0]['message']['content']

  return response

@retry(Exception, tries=5, delay=2)
def gpt_response(args, final_prompt):
  openai.api_key = args.openai_key
  gpt_model_name = CHECKPOINTS[args.model][args.size]
  completion = openai.Completion.create(
          engine=gpt_model_name,
          prompt=final_prompt,
          temperature=args.temperature,
          max_tokens=args.target_max_len
        )
  response = completion['choices'][0]['text']
  return response