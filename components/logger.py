import os, pdb, sys
import torch
import random
import logging
import re
import glob
import shutil
import math

import pandas as pd
import numpy as np
import time as tm
from datetime import datetime
from collections import defaultdict
from utils.help import model_match
from assets.static_vars import accelerator

class TextEvaluationLogger:
  def __init__(self, args, save_dir):
    self.args = args
    self.save_path = save_dir
    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(__name__)
    log_name = f'automatic_text_eval.log'
    log_path = os.path.join(save_dir, log_name)
    self.logger.addHandler(logging.FileHandler(log_path))
    self.logger.debug(args)
    self.log_info(args)

  def log_info(self, text):
    self.logger.info(text)



class ExperienceLogger:
  def __init__(self, args, save_dir):
    self.args = args
    self.learning_rate = args.learning_rate
    self.model_type = args.model

    self.save_path = save_dir
    self.optimizer = None
    self.scheduler = None

    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(__name__)
    log_name = f'trial_lr{args.learning_rate}.log'
    log_path = os.path.join(save_dir, log_name)
    self.logger.addHandler(logging.FileHandler(log_path))
    self.logger.debug(args)
    self.log_info(args)

    self.global_step = 0
    self.eval_step = 0
    self.log_interval = args.log_interval
    self.epoch = 1   # epoch count
    self.num_epochs = args.n_epochs

    self.best_score = { 'epoch': 1 }
    self.metric = args.metric
    self.best_score[self.metric] = 0

    self.do_save = args.do_save
    self.differences = []
    self.past_metrics = []

    self.logging_loss = 0.0
    self.tr_loss = 0.0
    self.eval_loss = 0.0
    self.current_loss = float('inf')
    self.patience = args.patience
    self.accelerate = args.accelerate

  def log_info(self, text):
    self.logger.info(text)

  def reset(self):
    self.global_step = 0
    self.eval_step = 0
    self.epoch = 1
    self.best_score = { 'epoch': 1 }
    self.best_score[self.metric] = 0
    self.differences = []
    self.past_metrics = []

    self.logging_loss = 0.0
    self.tr_loss = 0.0
    self.eval_loss = 0.0

  def start_train(self, total_step):
    self.logger.info("***** Running training *****")
    self.logger.info("  Num Epochs = %d" % self.args.n_epochs)
    self.logger.info("  Total train batch size  = %d" % self.args.batch_size)
    self.logger.info("  Total optimization steps = %d" % total_step)
    self.logger.info("  Running experiment for {}".format(self.style))

  def start_epoch(self, dataloader, percent=0):
    self.logger.info(f"Starting epoch {self.epoch} of {self.num_epochs}")
    self.start_time = tm.time()
    self.num_steps = len(dataloader)
    self.breakpoint = int(self.num_steps * percent)
    self.break_early = percent > 0

  def end_epoch(self):
    self.epoch += 1
    self.end_time = tm.time()

    raw_diff = self.end_time - self.start_time
    minute_diff = round(raw_diff / 60.0, 3)
    self.differences.append(minute_diff)
    avg_diff = round(np.average(self.differences), 3)

    met = round(self.best_score[self.metric] * 100, 2)
    self.logger.info(f"Best epoch is {self.best_score['epoch']} with {met} {self.metric}")
    self.logger.info(f"Current epoch took {minute_diff} min, average is {avg_diff} min")

    return self.early_stop(met)

  def early_stop(self, metric):
    below_threshold = False

    if self.epoch > 3 and self.args.debug:
      below_threshold = True

    self.past_metrics.append(metric)
    if len(self.past_metrics) >= self.patience:
      trail = self.past_metrics[-1*self.patience:]
      if all(x == trail[0] for x in trail):
        below_threshold = True

    if below_threshold:
      self.logger.info(f"Ran out of patience, early stopped at epoch {self.epoch}")
    return below_threshold

  def start_eval(self, num_batches):
    self.eval_step = 0
    self.final_step = num_batches

  def train_stop(self, args, step, debug_break):
    if args.debug and step >= debug_break*args.log_interval:
      return True
    if self.break_early and step > self.breakpoint:
      print(f"Training stopped early at step {step} to save time")
      return True
    return False

  def log_train(self, step, scheduler):
    self.global_step += 1
    now = datetime.now().strftime("%d-%H:%M:%S")

    step_report = f'{step}/{self.num_steps}'
    adjusted_lr = round(scheduler.get_last_lr()[0], 6)
    lr_report = f"Learning rate: {adjusted_lr}"
    self.current_loss = 10000 * ((self.tr_loss - self.logging_loss) / self.log_interval)
    loss_report = 'Mean loss: %.5f' % self.current_loss
    self.logging_loss = self.tr_loss

    if not self.accelerate and self.global_step < 100 and self.global_step % 10 == 0:
      print(self.global_step)
    if step % self.log_interval == 0:
      print(f"[{now}] Steps: {step_report}, {lr_report}, {loss_report}")

  def save_best_model(self, model, tokenizer, prune_keep):
    if self.do_save and self.best_score[self.metric] > 0.1:
      learning_rate = str(self.args.learning_rate)
      accuracy = '{0:.3f}'.format(self.best_score[self.metric])[2:]
      ckpt_name = f'acc{accuracy}_lr{learning_rate}_epoch{self.epoch}.pt'
      ckpt_path = os.path.join(self.save_path,ckpt_name)
      # model_to_save = model.module if hasattr(model, 'module') else model
      # torch.save(model_to_save.state_dict(), ckpt_path)   # Standard Pytorch method
      model.save_pretrained(ckpt_path)
      # tokenizer.save_pretrained(ckpt_path)  # Huggingface method, creates a new folder
      print(f"Saved a model at {ckpt_path}")
      if prune_keep > 0:
        is_directory = not (self.args.method == 'cvae' and self.model_type == 'bert')
        self.prune_saves(num_keep=prune_keep, is_directory=is_directory)

  def save_best_soft_prompt(self, args, embedder):
    score = self.best_score[self.metric]
    if self.do_save and not np.isnan(score):
      learning_rate = str(self.args.learning_rate)
      score = '{0:.3f}'.format(score)[2:]
      ckpt_name = f'acc{score}_lr{learning_rate}_epoch{self.epoch}.pt'
      ckpt_path = os.path.join(self.save_path, ckpt_name)

      if args.accelerate:
        accelerator.save(embedder, ckpt_path)
      else:
        embedder.save_prompt_embedding(self.save_path, ckpt_name)

      if args.prune_keep > 0:
        self.prune_saves(is_directory=False, num_keep=args.prune_keep)

  def prune_saves(self, is_directory=True, num_keep=5):
    if is_directory:
      folders = glob.glob(os.path.join(self.save_path, "*pt"))
      if len(folders) > num_keep:
        acc_and_folders = []
        for fname in folders:
          ckpt_name = fname.split('/')[-1]
          re_str = r'.*acc([0-9]{3}).*\.pt$'
          regex_found = re.findall(re_str, ckpt_name)
          if regex_found:
            accuracy = int(regex_found[0])
            acc_and_folders.append((accuracy, fname))
        acc_and_folders.sort(key=lambda tup: tup[0], reverse=True)
        for _, folder in acc_and_folders[num_keep:]:
          shutil.rmtree(folder) # for recursive removal
          print(f'removed {folder} due to pruning')

    else:
      files = [f for f in os.listdir(self.save_path) if f.endswith('.pt')]

      if len(files) > num_keep:
        scores_and_files = []
        for fname in files:
          ckpt_name = fname.split('/')[-1]
          re_str = r'.*acc([0-9]{3}).*\.pt$'
          regex_found = re.findall(re_str, ckpt_name)
          if regex_found:
            accuracy = int(regex_found[0])
            scores_and_files.append((accuracy, fname))

        scores_and_files.sort(key=lambda tup: tup[0], reverse=True)
        for _, file_to_remove in scores_and_files[num_keep:]:
          os.remove(os.path.join(self.save_path, file_to_remove))
          print(f'removed {file_to_remove} due to pruning')

  def update_optimization(self, optimizer, scheduler):
    self.optimizer = optimizer
    self.scheduler = scheduler

  def save_eval_result(self, predictions, targets, precisions, recalls):
    with open(os.path.join(self.save_path, 'predictions.txt'), 'a') as fp:
      for prediction in predictions:
        fp.write("%s\n" % prediction)
    with open(os.path.join(self.save_path, 'targets.txt'), 'a') as fp:
      for target in targets:
        fp.write("%s\n" % target)
    with open(os.path.join(self.save_path, 'prec.txt'), 'a') as fp:
      for prec in precisions:
        fp.write("%s\n" % prec)
    with open(os.path.join(self.save_path, 'rec.txt'), 'a') as fp:
      for rec in recalls:
        fp.write("%s\n" % rec)


class DualLogger(ExperienceLogger):
  def __int__(self, args, save_dir):
    super().__init__(args, save_dir)
  def save_best_model(self, model, tokenizer, prune_keep):
    if self.do_save and self.best_score[self.metric] > 0.1:
      learning_rate = str(self.args.learning_rate)
      accuracy = '{0:.3f}'.format(self.best_score[self.metric])[2:]
      ckpt_name = f'acc{accuracy}_lr{learning_rate}_epoch{self.epoch}.pt'
      ckpt_path = os.path.join(self.save_path,ckpt_name)
      torch.save(model, ckpt_path)
      print(f"Saved a model at {ckpt_path}")
      if prune_keep > 0:
        self.prune_saves(is_directory=False, num_keep=prune_keep)
