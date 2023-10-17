import os, sys, pdb, glob, re
from components.soft_embedder import AttributeEmbedding, CausalEmbedding, Seq2SeqEmbedding
import numpy as np
import random
import copy

from torch import nn, no_grad
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm as progress_bar
from nltk.corpus import wordnet, stopwords

from utils.help import *
from utils.load import load_best_model, load_best_attribute_embedder, \
                      load_best_soft_prompt, load_model, load_tokenizer
from utils.evaluate import eval_quantify, run_eval
from utils.help import _flatten_list, find_best_model_path

from components.augmenter import TextInfiller, Paraphraser, EasyDataAugmenter, RoundTripTranslator
from components.ct_generator import ControlledTextGenerator, SoftPromptMixer
from components.ct_generator import ConditionalLM, DExpertGenerator, CVAEGenerator
from components.additional import make_prompts
from components.denoise import assign_noise_scores
from utils.process import get_dataloader
from assets.static_vars import ATTRIBUTE_TOKEN_LEN, device, dtype, debug_break, accelerator

def build_data_generator(args, model, datasets, exp_logger, ontology):
  """ Trains a data augmenter or controllable text generator """
  assert(args.method in ['cvae', 'clm', 'msp', 'dexpert'])

  if args.method == 'msp':
    if args.accelerate:
      accelerated_prompt_mixing(args, model, datasets, exp_logger, ontology)
    else:
      soft_prompt_mixing(args, model, datasets, exp_logger, ontology)
  elif args.method == 'dexpert':
    # for dexperts, we need one model per attribute. regardless of setting, make a set of all attributes
    attributes = set()
    if args.dataset == 'nlu++':
      for domain_dict in ontology.values():
        attributes |= set(domain_dict['intents'].keys())
        attributes |= set(domain_dict['slots'].keys()) 
    elif args.dataset == 'crossner':
      for _, entities in ontology.items():
        attributes |= set(entities)
    elif args.dataset == 'topv2':
      for intent_list in ontology['intents'].values():
        attributes |= set(intent_list)
      for slot_list in ontology['slots'].values():
        attributes |= set(slot_list)
      
    orig_save_path = exp_logger.save_path

    for attribute in attributes:
      exp_logger.reset()
      exp_logger.save_path = os.path.join(orig_save_path, attribute)
      attr_datasets = datasets[attribute]
      attr_model = copy.deepcopy(model)
      if len(attr_datasets['train']) == 0:
        print(f"Attribute {attribute} has no data. Not training one for it.")
        continue
      generator_training(args, attr_model, attr_datasets, exp_logger)
    # Set parent save path back
    exp_logger.save_path = orig_save_path
  else:
    generator_training(args, model, datasets, exp_logger)

def soft_prompt_mixing(args, model, datasets, exp_logger, ontology):
  """ Similar to run_prompt_train except:
    1) This contains attributes that require mixing
    2) Therefore we have a dataset that returns three items, including constraints
    3) Set default text for the intruction prompt and attribute embeddings
  """
  dataset, dev_dataset = datasets['train'], datasets['dev']
  tokenizer = dataset.tokenizer
  train_dataloader = get_dataloader(args, dataset)
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs
  scaler = GradScaler()

  # freeze the large LM
  for param in model.parameters():
    param.requires_grad = False

  # create instruction_prompt and attribute embeddings
  original_emb = model.get_input_embeddings()
  if args.model == 'gpt':
    instruction_prompt = CausalEmbedding(original_emb, args.n_tokens, args.num_shot, tokenizer=tokenizer)
  else:
    instruction_prompt = Seq2SeqEmbedding(original_emb, args.n_tokens, args.num_shot, tokenizer=tokenizer)

  attribute_embeddings = setup_attribute_embeddings(args, original_emb, ontology, tokenizer)
  attribute_embeddings.instruction_prompt = instruction_prompt
  model.set_input_embeddings(attribute_embeddings)
  optimizer, scheduler = setup_optimization(args, attribute_embeddings, total_steps)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    model.train()

    for step, batch in enumerate(train_dataloader):
      inputs, targets, metadata = dataset.collate(args, batch)
      attribute_embeddings.set_constraints(metadata)
      review_inputs(args, inputs, targets, tokenizer)
      with autocast(dtype=torch.bfloat16):
        outputs = model(**inputs, labels=targets)
      exp_logger.tr_loss += outputs.loss.item()
      loss = outputs.loss / args.grad_accum_steps
      loss = scaler.scale(loss)
      loss.backward()

      if (step + 1) % args.grad_accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()

      exp_logger.log_train(step, scheduler)
      if exp_logger.train_stop(args, step, debug_break): break

    eval_res = run_eval(args, model, dev_dataset, exp_logger, 'dev', attribute_embeddings)
    if eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
      exp_logger.best_score = eval_res
      exp_logger.save_best_soft_prompt(args, attribute_embeddings)
      # exp_logger.save_best_soft_prompt(args, instruction_prompt)

    early_stop = exp_logger.end_epoch()
    if early_stop: break

def setup_attribute_embeddings(args, original_emb, ontology, tokenizer):
  if args.dataset == 'topv2':

    all_intents, all_slots = [], []
    for domain, intents in ontology['intents'].items():
      all_intents.extend(intents)
    for domain, slots in ontology['slots'].items():
      all_slots.extend(slots) 
    attributes = [list(set(all_intents)), list(set(all_slots))]

    primary_descriptions = [intent.replace('_', ' ') for intent in set(all_intents)]
    secondary_descriptions = [slot.replace('_', ' ') for slot in set(all_slots)]
    descriptions = [primary_descriptions, secondary_descriptions]
    attr_embeds = AttributeEmbedding(args, attributes, original_emb, num_sets=len(attributes),
                                      tokenizer=tokenizer, attribute_init_texts=descriptions)
  else:
    attributes, descriptions = [], []

    if args.dataset == 'crossner':
      for domain, entities in ontology.items():
        for entity, desc in entities.items():
          attributes.append(entity)
          descriptions.append(desc)

    elif args.dataset == 'actdial':
      for dialog_act in ontology:
        attributes.append(dialog_act)
        descriptions.append(dialog_act.replace('_', ' '))

    elif args.dataset == 'nlu++':
      for domain, entities in ontology.items():
        for intent, desc in entities['intents'].items():
          attributes.append(intent)
          desc = desc.replace('is the intent to ', '').replace('is the intent asking ', '').replace('?', '')
          descriptions.append(desc) 

    descriptions = AttributeEmbedding.repeat_to_fill(descriptions, tokenizer)
    attr_embeds = AttributeEmbedding(args, attributes, original_emb,
                                      tokenizer=tokenizer, attribute_init_texts=descriptions)
  return attr_embeds

def generator_training(args, model, datasets, exp_logger):
  dataset, dev_dataset = datasets['train'], datasets['dev']
  train_dataloader = get_dataloader(args, dataset)
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs
  optimizer, scheduler = setup_optimization(args, model, total_steps)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    model.train()

    for step, batch in enumerate(train_dataloader):
      inputs, targets = dataset.collate(args, batch)
      outputs = model(**inputs)
      exp_logger.tr_loss += outputs.loss.item()
      loss = outputs.loss / args.grad_accum_steps
      loss.backward()

      if (step + 1) % args.grad_accum_steps == 0:
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()

      exp_logger.log_train(step, scheduler)
      if exp_logger.train_stop(args, step, debug_break): break

    eval_res = run_eval(args, model, dev_dataset, exp_logger)
    if (eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]):
      exp_logger.best_score = eval_res
      exp_logger.save_best_model(model, dataset.tokenizer, args.prune_keep)

    early_stop = exp_logger.end_epoch()
    if early_stop: break

def generate_data(args, generator, data, exp_logger, engineer=None, ontology=None):
  new_data, data_group, pairings = [], [], []
  relevant = 0  # number of examples from the relevant domain
  if args.method == 'dexpert': data.cluster()
  for example in progress_bar(data, total=len(data)):
    relevant += 1
    original_text = example['text']

    if args.method in ['eda', 'para', 'rtt', 'fill']:
      generations = generator.augment(original_text)  # each utterance will generate up to 4 more
    elif args.model == "api":
      prompt = make_prompts(args, example, engineer, ontology)
      if args.verbose:
        print(prompt)
      if args.size in ["large", "giant"]:  # gpt4 / gpt3.5
        output_strings = gpt_chat_response(args, prompt)
      else:  # text-curie, text-da-vinci
        output_strings = gpt_response(args, prompt)
      generations = []
      temp = output_strings.split("\n")
      for item in temp:
        updated_string = re.sub(r'^\d+\)', '', item)
        generations.append(updated_string)
      if args.verbose:
        print(generations)

    else:
      constraints = ControlledTextGenerator.extract_constraints(example, args.dataset)
      generations = generator.synthesize(example, constraints)

    group = []
    for gen_text in generations:
      exp = example.copy()
      if gen_text == exp['text'] or len(gen_text) <= 1:  # skip duplicates and blanks
        continue
      exp['text'] = gen_text
      group.append(exp)

    if args.filter:
      data_group.append(group)
    else:
      new_data.extend(group)

    if args.debug:
      pairings.append({'old': original_text, 'new': gen_text})
      if len(pairings) > 50: break

  exp_logger.log_info(f"Generated {len(new_data)} new examples from {relevant} with {args.method}")
  show_samples(args.debug, pairings)
  new_data = filter_generated_data(args, new_data, data_group, data)
  save_synthesized_data(args, new_data)

def filter_generated_data(args, new_data, synthetic_data, seed_data):
  """ calculate scores and sort to truncate to the desired number of examples """
  if args.filter:
    noise_scores, flat_data = assign_noise_scores(args, synthetic_data, seed_data)
    # sorts from smallest to largest, which means least noise to most noise
    data_with_scores = sorted(zip(noise_scores, flat_data))
    new_data = [example for _, example in data_with_scores]
    max_allowed = len(seed_data) * args.num_generations
    new_data = new_data[:max_allowed]
    # new_data = random.choice(flat_data, size=max_allowed, replace=False, p=noise_scores)
  return new_data

def save_synthesized_data(args, new_data):
  if args.method == 'msp':
    cache_file = f'msp_{args.mixture}_{args.domain}_shot{args.num_generations}.json'
  else:
    cache_file =  f'{args.method}_{args.domain}_{args.setting}.json'

  if args.do_save:
    data_save_path = os.path.join(args.input_dir, 'cache', args.dataset, cache_file)
    json.dump(new_data, open(data_save_path, "w"), indent=4)
    print(f"Saving results to {data_save_path}")

def prepare_generator(args, default_model, default_tokenizer, exp_logger, engineer, ontology):
  """ Loads a previously trained generator for use, along with other pieces """

  # Data Augmentation Methods
  if args.method == 'eda':
    pieces = {'wordnet': wordnet, 'stopwords': stopwords}
    generator = EasyDataAugmenter(args, pieces)
  elif args.method == 'para':
    pieces = {'encoder': default_model, 'embedder': default_tokenizer}
    generator = Paraphraser(args, pieces)
  elif args.method == 'fill':
    pieces = {'infiller': default_model, 'stopwords': stopwords}
    generator = TextInfiller(args, pieces)
  elif args.method == 'rtt':
    from transformers import MarianTokenizer
    pieces = {piece: {} for piece in ['encoder', 'embedder', 'decoder', 'debedder']}
    for lang in ["roa", "de", "ru"]:
      forward_tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-en-{lang}')
      backward_tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{lang}-en')

      pieces['encoder'][lang] = load_model(args, forward_tokenizer, 'none', f'en-{lang}')
      pieces['embedder'][lang] = forward_tokenizer
      pieces['decoder'][lang] = load_model(args, backward_tokenizer, 'none', f'{lang}-en')
      pieces['debedder'][lang] = backward_tokenizer
    generator = RoundTripTranslator(args, pieces)

  # Controlled Text Generation Methods
  elif args.method == 'cvae':
    pretrained_vae = load_best_model(args, exp_logger, default_tokenizer)
    pieces = {'autoencoder': pretrained_vae, 'tokenizer': default_tokenizer}
    generator = CVAEGenerator(args, pieces)
  elif args.method == 'dexpert':
    pieces = {'alpha': 2.0, 'model': default_model, 'tokenizer': default_tokenizer}

    attributes = set()
    if args.dataset == 'nlu++':
      # when generating data for args.domain, we use the models trained on CROSS[args.domain] which are saved at args.domain. But we only care about the attributes in-domain anyway
      attributes = set(ontology[args.domain]['intents'].keys()) | set(ontology['general']['intents'].keys())
      attributes |= set(ontology[args.domain]['slots'].keys()) | set(ontology['general']['slots'].keys())
    elif args.dataset == 'crossner':
      attributes = list(ontology[args.domain].keys()) + list(ontology['general'].keys())
    elif args.dataset == 'topv2':
      for domain, intent_list in ontology['intents'].items():
        attributes |= set(intent_list)
      for slot_list in ontology['slots'].values():
        attributes |= set(slot_list)
    for intent in attributes:
      load_dir = os.path.join(exp_logger.save_path, intent)
      top_folder = find_best_model_path(load_dir)
      if top_folder is None:
        print(f'No checkpoints were found in {load_dir}, skipping this attribute model')
        continue
      pieces[intent] = top_folder
    generator = DExpertGenerator(args, pieces)
  elif args.method == 'clm':
    pretrained_lm = load_best_model(args, exp_logger, default_tokenizer)
    pieces = {'clm': pretrained_lm, 'tokenizer': default_tokenizer}
    generator = ConditionalLM(args, pieces)
  elif args.method == 'msp':
    model, attribute_embedding = load_best_attribute_embedder(args, default_model, exp_logger)
    pieces = {'model': model, 'attribute_embedding': attribute_embedding, 
              'tokenizer': default_tokenizer, 'prompt_engineer': engineer }
    generator = SoftPromptMixer(args, pieces)

  print(f"Loaded a {generator.name} for data synthesis")
  return generator
