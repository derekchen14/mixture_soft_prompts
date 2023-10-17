import math
import os, pdb, sys
import re
import numpy as np
import random

import torch
from tqdm import tqdm as progress_bar
from assets.static_vars import ATTRIBUTE_TOKEN_LEN, MAX_MIXTURE_SIZE, device
from collections import defaultdict
from utils.load import load_model
from utils.help import _flatten_list

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag

from transformers import top_k_top_p_filtering

class ControlledTextGenerator(object):
  """ Performs controlled multi-attribute text generation given a set of labels
      The key distinction between CTG and DataAug is that the generators additionally
      take in a set of attributes to condition on.  This is what allows for control. """
  def __init__(self, args):
    self.name = 'base-controlled-text-generator'
    self.dataset = args.dataset
    self.new_tokens = args.target_max_len
    self.rep_penalty = args.threshold
    self.temp = args.temperature
    self.num_generations = args.num_generations

  def synthesize(self, seed_example, constraints) -> list:
    # not called "generate" since that name space is already used by huggingface models
    raise NotImplementedError

  @staticmethod
  def extract_constraints(example, dataset):
    if dataset == 'topv2':
      unique_attrs, unique_vals =  set(example['attributes']), set(example['values'])
      ua_size, uv_size = len(unique_attrs), len(unique_vals)
      if ua_size + uv_size > MAX_MIXTURE_SIZE:
        cutoff = MAX_MIXTURE_SIZE - ua_size 
        unique_vals = list(unique_vals)[:cutoff]
      constraints = [unique_attrs, unique_vals]
    else:
      constraints = set(example['attributes']) 
    return constraints

  def collect_results(self, output_strings, remove_from_outstr):
    results = []      
    for out_str in output_strings:
      text = out_str
      for to_remove in remove_from_outstr:
        text = text.replace(to_remove, '')
      results.append(text)
    return results

class SoftPromptMixer(ControlledTextGenerator):
  def __init__(self, args, pieces):
    super().__init__(args)
    self.name = 'mixture-of-soft-prompts-generator'
    self.generator = pieces['model']
    self.attribute_embedding = pieces['attribute_embedding']
    self.tokenizer = pieces['tokenizer']
    self.engineer = pieces['prompt_engineer']

    self.mixture_type = args.mixture
    self.prefix_len = args.n_tokens
    self.model_type = args.model

  def synthesize(self, seed_example, constraints):
    self.generator.eval()
    prompt_text = self.engineer.prompt_with_exemplars(seed_example, add_tail=True, include_seed=True)
    tokens = self.tokenizer(prompt_text, return_tensors='pt')
    
    content = tokens['input_ids'][0]  # 2 dim --> 1
    attn_content = tokens['attention_mask'][0]
    
    instruct_tokens = torch.full((self.prefix_len,), -1, dtype=content.dtype)
    input_ids = torch.cat([instruct_tokens, content]).unsqueeze(0).to(device)
    attn_tokens = torch.ones(self.prefix_len, dtype=content.dtype)
    attention_mask = torch.cat([attn_tokens, attn_content]).unsqueeze(0).to(device)

    beams = self.num_generations   # math.ceil(self.num_generations / 2) for two num completions
    metadata = {'constraints': [constraints] * beams, 'pad_lengths': [0] * beams}
    self.attribute_embedding.set_constraints(metadata)

    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    with torch.no_grad():
      outputs = self.generator.generate(**inputs, max_new_tokens=self.new_tokens, do_sample=True,
                                num_return_sequences=beams, repetition_penalty=self.rep_penalty, 
                                temperature=self.temp, num_beams=beams, early_stopping=True)
      output_strings = self.tokenizer.batch_decode(outputs.detach())

    context_len = self.prefix_len + len(prompt_text) - 1 
    generations = []
    for output in output_strings:
      if self.model_type == 'gpt':
        generated_str = output[context_len:]
      else:
        generated_str = output.replace('<pad>', '')

      for end in ['\n', '...', self.tokenizer.eos_token]:
        if end in generated_str:
          end_index = generated_str.index(end)
          generated_str = generated_str[:end_index].strip()
      generations.append(generated_str.strip())
 
    return generations

class ConditionalLM(ControlledTextGenerator):
  def __init__(self, args, pieces):
    super().__init__(args)
    self.name = 'class-conditional-language-model'
    self.generator = pieces['clm']
    self.tokenizer = pieces['tokenizer']

  def synthesize(self, seed_example, constraints):
    bos, eos = self.tokenizer.bos_token, self.tokenizer.eos_token

    raw_inputs = [f"{bos}{seed_example['constraint_str']}<label>"]
    # For generation with GPT, pad to left.
    self.tokenizer.padding_side = 'left'
    inputs = self.tokenizer(raw_inputs, return_tensors='pt').to(device)
    outputs = self.generator.generate(
      **inputs, max_new_tokens=self.new_tokens,
      repetition_penalty=self.rep_penalty,
      temperature=self.temp,
      num_beams=self.num_generations,
      num_return_sequences=self.num_generations,
      eos_token_id=self.tokenizer.eos_token_id
    )
    output_strings = self.tokenizer.batch_decode(outputs.detach())
    input_strings = self.tokenizer.batch_decode(inputs['input_ids'])
    results = self.collect_results(output_strings, [input_strings[0], eos, self.tokenizer.pad_token])
    return results

class DExpertGenerator(ControlledTextGenerator):
  def __init__(self, args, pieces):
    super().__init__(args)
    self.name = 'decoding-time-expert-generator'
    self.alpha = pieces['alpha'] # This should probably be a parameter passed in from args? they chose 2.0 in the paper
    self.base_lm = pieces['model']
    self.tokenizer = pieces['tokenizer']
    self.attr_to_lm_paths = {attr: lm for attr, lm in pieces.items() if attr not in {'model', 'tokenizer', 'alpha'}}

    self.args = args # Need this for model loading
    self.max_constraints = 6
    self.loaded_models = {}

  def synthesize(self, seed_example, constraints):
    # https://github.com/alisawuffles/DExperts/blob/main/generation/dexperts_generation.py#L76
    bos = self.tokenizer.bos_token
    eos = self.tokenizer.eos_token
    
    if len(constraints) > self.max_constraints: constraints = random.sample(list(constraints), self.max_constraints)
    
    if len(set(self.loaded_models.keys()) | set(constraints)) > self.max_constraints:
      # unload irrelevant models
      self.loaded_models = {attr: model for attr, model in self.loaded_models.items() if attr in constraints}

    # load relevant models, using already-loaded ones when we can
    used_constraints = []
    for attr in constraints:
      if attr not in self.attr_to_lm_paths: continue
      expert_path = self.attr_to_lm_paths[attr]
      used_constraints.append(attr)
      if attr not in self.loaded_models: 
        expert_model = load_model(self.args, self.tokenizer, ckpt_path=expert_path)
        expert_model.eval()
        expert_model.to(device)
        self.loaded_models[attr] = expert_model
    if len(used_constraints) == 0: return []
    constraints = used_constraints

    # For generation with GPT, pad to left.
    self.tokenizer.padding_side = 'left'

    inputs = self.tokenizer([f'{bos}']*self.num_generations, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    batch_size, input_seq_len = input_ids.shape
    attention_mask = inputs['attention_mask']
    position_ids = attention_mask.cumsum(dim=1) - 1
    # Keep track of complete candidats
    unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=device)

    with torch.no_grad():
        for step in range(self.new_tokens):
          # base prediction
          base_output= self.base_lm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
          )
          base_logits = base_output['logits']
          base_past = base_output['past_key_values']
          ensemble_logits = base_logits

          # expert prediction
          expert_logits = None
          for attr in used_constraints: 
            expert_lm = self.loaded_models[attr]
            attr_expert_output = expert_lm(
              input_ids,
              attention_mask=attention_mask,
              position_ids=position_ids,
            )
            attr_expert_logits = attr_expert_output['logits']
            ensemble_logits += self.alpha * attr_expert_logits
          # in the first decoding step, we want to use the 'real' last position for each sentence
          if step == 0:
            last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
            next_token_logits = ensemble_logits[range(batch_size), last_non_masked_idx, :]
          else:
            next_token_logits = ensemble_logits[:, -1, :]
          next_token_logits = next_token_logits / self.temp
          # Sampling to get some diversity in outputs
          probs = next_token_logits.softmax(dim=-1)
          next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
          # either append a padding token here if <EOS> has been seen or append next token
          tokens_to_add = next_tokens * unfinished_sents + self.tokenizer.pad_token_id * (1 - unfinished_sents)

          # Updates which sentences are now done.
          eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
          unfinished_sents.mul_((~eos_in_sents).long())

          # Stop when all sentences are done.
          if unfinished_sents.max() == 0:
            break

          # Update input_ids, attention_mask and position_ids
          input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
          attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
          position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

        output_strings = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for output in input_ids[:, input_seq_len:]]

    results = self.collect_results(output_strings, [eos]) 
    return results

class CVAEGenerator(ControlledTextGenerator):
  def __init__(self, args, pieces):
    super().__init__(args)
    self.name = 'conditional-variational-autoencoder'
    self.tokenizer = pieces['tokenizer']
    self.cvae = pieces['autoencoder']

  def synthesize(self, seed_example, constraints):
    cls, eos = self.tokenizer.cls_token, self.tokenizer.eos_token

    raw_inputs = [f"{seed_example['constraint_str']}"]
    
    inputs = self.tokenizer(raw_inputs, return_tensors='pt')
    with torch.no_grad():
        output_strings = []
        for _ in range(self.num_generations):
          outputs = self.cvae.generate(**inputs, 
            max_new_tokens=self.new_tokens,
            repetition_penalty=self.rep_penalty,
            temperature=self.temp,
            num_beams=self.num_generations,
            num_return_sequences=self.num_generations,
            eos_token_id=self.tokenizer.eos_token_id
          )
          output_strings.extend(self.tokenizer.batch_decode(outputs.detach()))
    results = self.collect_results(output_strings, [cls, eos, self.tokenizer.pad_token])
    return results
