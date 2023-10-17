import os, sys, pdb
import numpy as np
import random
from tqdm import tqdm as progress_bar
import openai

from torch import nn, no_grad
from torch.cuda.amp import autocast, GradScaler
from components.logger import ExperienceLogger
from components.engineer import PromptEngineer
from components.ct_generator import SoftPromptMixer
from components.soft_embedder import CausalEmbedding, Seq2SeqEmbedding

from utils.help import *
from utils.synthesize import build_data_generator, generate_data, prepare_generator
from utils.process import process_data, get_dataloader, check_cache
from utils.arguments import solicit_params
from utils.evaluate import eval_quantify, eval_qualify, run_eval, accelerated_eval, run_openai_eval
from utils.load import *
from assets.static_vars import dtype, debug_break, accelerator, CHECKPOINTS
from utils.help import gpt_chat_response, gpt_response

def run_in_context(args, model, dataset, exp_logger, engineer, ontology):
  if args.model == "api": # openai api
    assert (args.openai_key is not None)
    if args.verbose:
      print(f'the length of the dataset is {len(dataset)}')

    all_inputs, all_outputs, all_targets = [], [], []
    engineer.attach_dataset(args.domain, dataset)
    prompt = engineer.generate_standard_exemplars(args, ontology)
    if args.verbose:
      print("\n")
      print(f"{prompt}")
    count = 0
    except_count = 0
    for example in progress_bar(dataset, total=len(dataset)):
      all_targets.append(example['target'])
      all_inputs.append(example['text'])
      query = f"Q: {example['text']}\n     A: "
      final_prompt = prompt + query
      if args.size in ["large", "giant"]:  # gpt4 / gpt3.5
        response = gpt_chat_response(args, final_prompt)
      else:  # text-curie, text-da-vinci
        response = gpt_response(args, final_prompt)

      if args.icl_type == "base":
        all_outputs.append(response)
      elif args.icl_type == "cot":
        try:
          if args.dataset == "topv2":
            if "? " not in response.split("\n")[1]:
              attribute = ""
            else:
              attribute_start = int(response.split("\n")[1].index("?")) + 2
              attribute = response.split("\n")[1][attribute_start:]
            if "Answer: " not in response:
              slots = ""
            else:
              slots_start = int(response.index("Answer: ")) + 8
              slots = response[slots_start:]
            final_response = attribute.strip() + " <sep> " + slots.strip()
          else:
            answer_start = response.index("Answer: ")
            final_response = response[answer_start + 8:]
        except Exception:
          final_response = response
          print("No answer found!")
          print(final_response)
          except_count += 1
        all_outputs.append(final_response)
      count += 1
      if args.debug and count % 20 == 0:
        run_openai_eval(args, all_inputs, all_outputs, all_targets, exp_logger)
        print(except_count)
        break
    run_openai_eval(args, all_inputs, all_outputs, all_targets, exp_logger)

  else:
    dataloader = get_dataloader(args, dataset, 'ICL')
    num_batches = debug_break if args.debug else len(dataloader)
    exp_logger.start_eval(num_batches)
    run_eval(args, model, dataset, exp_logger)

def run_local_train(args, model, datasets, exp_logger):
  dataset, dev_dataset = datasets['train'], datasets['dev']
  train_dataloader = get_dataloader(args, dataset)
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs
  optimizer, scheduler = setup_optimization(args, model, total_steps)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    model.train()

    for step, batch in enumerate(train_dataloader):
      inputs, targets = dataset.collate(args, batch)
      review_inputs(args, inputs, targets, datasets['train'].tokenizer)

      outputs = model(**inputs, labels=targets)
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
    if eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
      exp_logger.best_score = eval_res
      exp_logger.save_best_model(model, tokenizer, args.prune_keep)

    early_stop = exp_logger.end_epoch()
    if early_stop: break

  test_res = run_eval(args, model, datasets['test'], exp_logger)
  return model

def run_train_loop(args, model, datasets, exp_logger, soft_embeds=None):
  if dtype == 'cpu':
    return run_local_train(args, model, datasets, exp_logger)

  dataset, dev_dataset = datasets['train'], datasets['dev']
  train_dataloader = get_dataloader(args, dataset)
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs

  scaler = GradScaler()
  if soft_embeds:
    optimizer, scheduler = setup_optimization(args, soft_embeds, total_steps)
  else:
    optimizer, scheduler = setup_optimization(args, model, total_steps)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    model.train()

    for step, batch in enumerate(train_dataloader):
      inputs, targets = dataset.collate(args, batch)
      review_inputs(args, inputs, targets, dataset.tokenizer)
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

    eval_res = run_eval(args, model, dev_dataset, exp_logger)
    if eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
      exp_logger.best_score = eval_res
      if soft_embeds:
        exp_logger.save_best_soft_prompt(args, soft_embeds)
      else:
        exp_logger.save_best_model(model, tokenizer, args.prune_keep)

    early_stop = exp_logger.end_epoch()
    if early_stop: break

  return model

def accelerated_train_loop(args, model, datasets, exp_logger, soft_embeds):
  dataset, dev_dataset = datasets['train'], datasets['dev']
  assert(dataset.name == 'fine-tune-dataset')
  train_dataloader = get_dataloader(args, dataset)
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs

  optimizer, scheduler = setup_optimization(args, soft_embeds, total_steps)
  accelerator.gradient_accumulation_steps = args.grad_accum_steps
  accelerated_parts = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
  model, optimizer, train_dataloader, scheduler = accelerated_parts

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    model.train()

    for step, batch in enumerate(train_dataloader):
      with accelerator.accumulate(model):
        inputs, targets = dataset.collate(args, batch)
        review_inputs(args, inputs, targets, dataset.tokenizer)

        with autocast(dtype=torch.float16):
          outputs = model(**inputs, labels=targets)

        exp_logger.tr_loss += outputs.loss.item()
        accelerator.backward(outputs.loss)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()

      exp_logger.log_train(step, scheduler)
      if exp_logger.train_stop(args, step, debug_break): break

    if exp_logger.current_loss > 400:
      curr_loss = round(exp_logger.current_loss, 2)
      accelerator.print(f"Skipping evaluation since {curr_loss} loss is too high")
    else:
      eval_res = accelerated_eval(args, model, dev_dataset, exp_logger)
      # we check for eval_res since 3 out of 4 processes will return None result
      if eval_res and eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
        exp_logger.best_score = eval_res
        if args.do_save:
          state = accelerator.get_state_dict(model)
          exp_logger.save_best_soft_prompt(args, state['gpt_neox.embed_in.soft_prompt'])

    exp_logger.end_epoch()  # remove option to early stop since non-main process will hang

  accelerator.wait_for_everyone()
  return model

def run_prompt_train(args, model, datasets, exp_logger, ontology):
  # freeze the large LM
  parameters = list(model.parameters())
  # can also tune the vocab embeddings by freezing first params
  # for param in parameters:
  for param in parameters:
    param.requires_grad = False

  # create and then set the soft prompt embeddings
  if args.model == 'gpt':
    soft_prompt_embed = CausalEmbedding(model.get_input_embeddings(), args.n_tokens)
  else:
    soft_prompt_embed = Seq2SeqEmbedding(model.get_input_embeddings(), args.n_tokens)
  model.set_input_embeddings(soft_prompt_embed)

  if args.accelerate:
    model = accelerated_train_loop(args, model, datasets, exp_logger, soft_prompt_embed)
  else:
    model = run_train_loop(args, model, datasets, exp_logger, soft_prompt_embed)
  return model


if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  args, save_path = check_directories(args)
  set_seed(args)

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)
  if already_exist:
    datasets = cache_results
    ont_path = os.path.join(args.input_dir, args.dataset, "ontology.json")
    ontology = json.load(open(ont_path, 'r'))
  else:
    raw_data = load_data(args)
    ontology = raw_data['ontology']
    datasets = process_data(args, cache_results, raw_data, tokenizer)

  args.ont_size = len(ontology)
  exp_logger = ExperienceLogger(args, save_path)
  engineer = PromptEngineer(args, ontology)
  if args.method != 'dexpert' and args.model != "api":
    datasets = recruit_engineer(args, datasets, engineer)
  if args.verbose: display_domains(args, datasets)

  model = load_model(args, tokenizer, save_path)
  if args.do_train:
    if args.task == 'soft_prompt':
      run_prompt_train(args, model, datasets, exp_logger, ontology)
    elif args.task in ['fine_tune', 'end_to_end']:
      run_train_loop(args, model, datasets, exp_logger)
    elif args.task == 'synthesize':
      build_data_generator(args, model, datasets, exp_logger, ontology)

  elif args.do_eval and args.task != 'synthesize':
    if args.task == 'soft_prompt':
      model = load_best_soft_prompt(args, model, exp_logger)
    else:
      model = load_best_model(args, exp_logger, tokenizer)
    run_eval(args, model, datasets['test'], exp_logger, 'test')

  elif args.task == 'in_context':
    engineer.embed_samples(datasets['test'])
    run_in_context(args, model, datasets['test'], exp_logger, engineer, ontology)

  elif args.task == 'synthesize':
    if args.model == 'aug':
      model = load_pretrained_model(args, args.checkpoint)
      tokenizer = load_pretrained_tokenizer(args.method)
      generator = prepare_generator(args, model, tokenizer, exp_logger, engineer, ontology)
      generated_data = generate_data(args, generator, datasets['train'], exp_logger)
    elif args.model == 'api':
      for split, dataset in datasets.items():
        engineer.attach_dataset(args.domain, dataset)
      generator = {}
      generated_data = generate_data(args, generator, datasets['train'], exp_logger, engineer, ontology)
    else:
      generator = prepare_generator(args, model, tokenizer, exp_logger, engineer, ontology)
      generated_data = generate_data(args, generator, datasets['train'], exp_logger)
