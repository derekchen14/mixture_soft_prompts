import os, pdb, sys
import numpy as np
import random
import json

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from transformers import MaxLengthCriteria, StoppingCriteriaList, BeamSearchScorer, LogitsProcessorList
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import BertModel, GPT2LMHeadModel

from tqdm import tqdm as progress_bar
from collections import defaultdict
from assets.static_vars import device
from sentence_transformers import SentenceTransformer
from sentence_transformers.model_card_templates import ModelCardTemplate
from typing import List, Dict, Tuple, Type, Callable
    
class BaseModel(nn.Module):
  def __init__(self, args, core, tokenizer):
    super(BaseModel, self).__init__()
    self.name = 'base-model'
    self.encoder = core
    self.model_type = args.model
    self.tokenizer = tokenizer

    self.verbose = args.verbose
    self.debug = args.debug
    self.weight_decay = args.weight_decay
    self.dropout = nn.Dropout(args.drop_rate)

    self.dense = nn.Linear(core.config.hidden_size, args.hidden_size)
    self.gelu = nn.GELU()
    self.classify = nn.Linear(args.hidden_size, args.ont_size) 
    self.softmax = nn.LogSoftmax(dim=1)
    self.criterion = nn.CrossEntropyLoss()  # performs LogSoftmax and NegLogLike Loss

  def forward(self, inputs, targets, outcome='logit'):
    if self.model_type == 'roberta':
      """ By default, the encoder returns result of (batch_size, seq_len, vocab_size) under 'logits'
      When the output_hs flag is turned on, the output will also include a tuple under 'hidden_states'
      The tuple has two parts, the first is embedding output and the second is hidden_state of last layer
      """
      enc_out = self.encoder(**inputs, output_hidden_states=True) 
      cls_token = enc_out['hidden_states'][1][:, 0, :]
    else:
      enc_out = self.encoder(**inputs)
      cls_token = enc_out['last_hidden_state'][:, 0, :]   # batch_size, embed_dim
    
    hidden1 = self.dropout(cls_token)
    hidden2 = self.dense(hidden1)
    hidden3 = self.gelu(hidden2)
    hidden4 = self.dropout(hidden3)
    logits = self.classify(hidden4)  # batch_size, num_classes
    logits = logits.squeeze()
    
    loss = self.criterion(logits, targets)
    output = logits if outcome == 'logit' else self.softmax(logits)
    return output, loss


def prepare_inputs_for_generation(input_ids, past=None, encoder_hidden_states=None, **kwargs):
  token_type_ids = kwargs.get("token_type_ids", None)
  # only last token for inputs_ids if past is defined in kwargs
  if past:
    input_ids = input_ids[:, -1].unsqueeze(-1)
    if token_type_ids is not None:
      token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

  attention_mask = kwargs.get("attention_mask", None)
  position_ids = kwargs.get("position_ids", None)
  
  if attention_mask is not None and position_ids is None:
      # create position_ids on the fly for batch generation
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if past:
      position_ids = position_ids[:, -1].unsqueeze(-1)
  else:
    position_ids = None
  # !!!!!!!!!!!!!!!!!!! start: modified vs original, to pass inputs_embeds when they are available
  model_inputs = {"input_ids": input_ids}
  if encoder_hidden_states is not None:
    model_inputs["encoder_hidden_states"] = encoder_hidden_states
  model_inputs.update({
    "past_key_values": past,
    "use_cache": kwargs.get("use_cache"),
    "position_ids": position_ids,
    "attention_mask": attention_mask,
    "token_type_ids": token_type_ids,
  })
  return model_inputs

class CVAEModel(BaseModel):
  def __init__(self, args, encoder, decoder_config, decoder, tokenizer): 
    super().__init__(args, encoder, tokenizer)
    self.name = 'cvae-model'
    self.embedder = encoder.embeddings
    self.config = decoder_config
    self.decoder = decoder
    self.decoder.prepare_inputs_for_generation = prepare_inputs_for_generation
    self.mu_linear = torch.nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size)
    self.logvar_linear = torch.nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size)

    self.stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=args.target_max_len)])
    self.num_beams = args.num_generations
    self.beam_scorer_batch_size = args.num_generations
    self.beam_scorer = BeamSearchScorer(self.beam_scorer_batch_size, num_beams=self.num_beams, device=device)
    self.logits_warpers = self.decoder._get_logits_warper(temperature=args.temperature, top_k=0, top_p=1.0)
    self.logits_processors = self.decoder._get_logits_processor(
            repetition_penalty=args.threshold,
            no_repeat_ngram_size=decoder.config.no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=None,
            input_ids_seq_length=None,
            encoder_input_ids=None,
            bad_words_ids=None,
            min_length=None,
            max_length=None,
            eos_token_id=None,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            prefix_allowed_tokens_fn=None,
            num_beams=None,
            num_beam_groups=None,
            diversity_penalty=None,
            remove_invalid_values=None,
            exponential_decay_length_penalty=None,
            logits_processor=LogitsProcessorList(),
            renormalize_logits=None, 
            suppress_tokens=None, 
            begin_suppress_tokens=None, 
            forced_decoder_ids=None, 
        )

  
  def get_input_embeddings(self, input_ids, attention_mask):
    # roberta does not use token type ids
    H0 = self.embedder(input_ids=input_ids.detach().clone()) # (B, seq_len, hidden_size)
    # encode, get h_0_last
    H0 = H0.to(device)
    h_0_last = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0] # (B, hidden_size)
    # h_0_last = self.encoder(input_embeds=H0, attention_mask=attention_mask).last_hidden_state[:,0] # (B, hidden_size) # we can just pass in input ids and token type ids, itll do input embeds internally
    return H0, h_0_last

  def reparameterize(self, h_0_last):
    # reparameterize to get z
    mu = self.mu_linear(h_0_last)
    logvar = self.logvar_linear(h_0_last)
    std = (0.5*logvar).exp()
    eps = torch.randn_like(mu)
    z = mu + eps*std # (B, hidden_size)
    z = z[:, None, :] # (B, 1, hidden_size)
    return z, logvar, mu

  def decode(self, input_ids, attention_mask, labels, z):
    # decode
    # For RoBERTa, sep token id is eos token id
    indices_of_sep = 1 + (input_ids == self.tokenizer.eos_token_id).max(dim=1).indices
    y_attn_mask = torch.zeros(attention_mask.shape)
    y_attn_mask[(torch.arange(attention_mask.shape[0]), indices_of_sep)] = 1 
    y_attn_mask = y_attn_mask.cumsum(dim=1).to(device)
    y_attn_mask = 1 - y_attn_mask
    y_ids = torch.where(y_attn_mask==1, input_ids, self.tokenizer.pad_token_id).to(device)
    H0_y = self.embedder(input_ids=y_ids)

    # get H12'
    decoder_inputs = torch.cat((z, H0_y[:,1:]), dim=1) # (B, y_seq_len, hidden_size)
    _, seq_len, _ = decoder_inputs.shape
    decoder_inputs = torch.cat((decoder_inputs, z.repeat(1, seq_len, 1)), dim=-1) # (B, y_seq_len, hidden_size*2)
    d_labels = torch.where(labels==self.tokenizer.pad_token_id, -100, labels)
    labels_attn_mask = torch.where(labels == self.tokenizer.pad_token_id, 0, 1)
    outputs = self.decoder(input_ids=labels, attention_mask=labels_attn_mask, encoder_hidden_states=decoder_inputs, labels=d_labels, encoder_attention_mask=y_attn_mask)

    return outputs

  def forward(self, input_ids, attention_mask, labels):
    _, h_0_last = self.get_input_embeddings(input_ids, attention_mask)
    z, logvar, mu = self.reparameterize(h_0_last)
    outputs = self.decode(input_ids, attention_mask, labels, z)
    
    # https://arxiv.org/pdf/1312.6114.pdf
    kld = -0.5 * (1+logvar-mu**2-logvar.exp()).sum() # or avg before sum? https://github.com/shj1987/ControlVAE-ICML2020/blob/master/Language_modeling/Text_gen_PTB/model.py
    outputs.loss += kld
    
    return outputs

  def generate(self, input_ids, attention_mask, **kwargs):
    input_ids = input_ids.to(device)
    H0 = self.embedder(input_ids=input_ids) #, token_type_ids=token_type_ids) # (B, seq_len, hidden_size)\
    B, seq_len, hidden_size = H0.shape # should be just the z and y
    z = torch.randn((B, hidden_size)).to(device) #  (B, hidden_size)
    z = z[:, None, :] # (B, 1, hidden_size)

    decoder_inputs = torch.cat((z, H0[:,1:]), dim=1) # (B, seq_len, hidden_size)
    decoder_inputs = torch.cat((decoder_inputs, z.repeat(1, seq_len, 1)), dim=-1).to(device) # (B, seq_len, hidden_size*2)
    decoder_inputs = decoder_inputs.repeat_interleave(self.num_beams, dim=0)
    encoder_attention_mask = attention_mask.repeat_interleave(self.num_beams, dim=0)
    # https://github.com/huggingface/transformers/issues/6535#issuecomment-1353658984
    starter_inputs = torch.tensor([[self.tokenizer.cls_token_id]]*B*self.num_beams).to(device)

    if B != self.beam_scorer_batch_size: # Could also drop last batch
        beam_scorer = BeamSearchScorer(batch_size=B, num_beams=self.num_beams, device=device)
    else:
        beam_scorer = self.beam_scorer
    outputs = self.decoder.beam_sample(starter_inputs, beam_scorer=beam_scorer, encoder_hidden_states=decoder_inputs, \
            encoder_attention_mask=encoder_attention_mask, stopping_criteria=self.stopping_criteria, \
            pad_token_id=self.config.pad_token_id, logits_processor=self.logits_processors, logits_warper=self.logits_warpers, **kwargs)
    return outputs

  def resize_token_embeddings(self, new_size):
    self.decoder.resize_token_embeddings(new_size)
    self.encoder.resize_token_embeddings(new_size)

  def to(self, device):
    self.encoder.to(device)
    self.decoder.to(device)
    self.mu_linear.to(device)
    self.logvar_linear.to(device)

  def save_pretrained(self, path):
    torch.save({'model_state_dict': self.state_dict(),
        'ckpt_name': self.config.name_or_path,
        'encoder_model_name': self.encoder.name_or_path,
        'decoder_model_name': self.decoder.name_or_path}, path)
    
  @staticmethod
  def from_pretrained(args, path, tokenizer):  
    if 'bert' not in path:
      prev_weights = torch.load(path)
      ckpt_name = prev_weights['ckpt_name']
      encoder_model_name = prev_weights['encoder_model_name']
      decoder_model_name = prev_weights['decoder_model_name']
    else:  
      encoder_model_name = path
      decoder_model_name = 'gpt2'
    encoder_config = AutoConfig.from_pretrained(encoder_model_name)
    encoder_config.vocab_size = len(tokenizer)
    encoder = BertModel(encoder_config)
    decoder_config = AutoConfig.from_pretrained(decoder_model_name)
    decoder_config.hidden_size *= 2
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder_config.vocab_size = len(tokenizer)
    decoder = GPT2LMHeadModel(decoder_config)
    model = CVAEModel(args, encoder, decoder_config, decoder, tokenizer)
    if 'bert' not in path:
      model.load_state_dict(prev_weights['model_state_dict'])
    model.to(device)
    return model

class DualClassifier(BaseModel):
  # Model for predicting both intents and slots at once
  def __init__(self, args, core, tokenizer, primary_size, secondary_size):
    super().__init__(args, core, tokenizer)
    self.name = 'dual-classify'

    self.model_type = 'roberta' if args.size in ['small', 'medium'] else 'deberta'
    self.primary_classify = nn.Linear(args.hidden_size, primary_size) 
    self.secondary_classify = nn.Linear(args.hidden_size, secondary_size) 

  def forward(self, inputs, targets, outcome='logit'):
    enc_out = self.encoder(**inputs, output_hidden_states=True) 
    cls_token = enc_out['hidden_states'][-1][:, 0, :]
    
    hidden1 = self.dropout(cls_token)
    hidden2 = self.dense(hidden1)
    hidden3 = self.gelu(hidden2)
    hidden4 = self.dropout(hidden3)

    intent_logits = self.primary_classify(hidden4)
    slot_logits = self.secondary_classify(hidden4)         # batch_size, num_classes
    intent_loss = self.criterion(intent_logits, targets['intent'])
    slot_loss = self.criterion(slot_logits, targets['slot'])
    loss = intent_loss + slot_loss

    if outcome == 'logit':
      output = {'intent': intent_logits, 'slot': slot_logits}
    else:
      output = {'intent': self.softmax(intent_logits), 'slot': self.softmax(slot_logits)}
    return output, loss

class GenerateModel(BaseModel):
  # Main model for general classification prediction
  def __init__(self, args, core, tokenizer):
    super().__init__(args, core, tokenizer)
    self.name = 'generate'

  def forward(self, inputs, targets, outcome='logit'):
    enc_out = self.encoder(**inputs)
    cls_token = enc_out['last_hidden_state'][:, 0, :]   # batch_size, embed_dim
    
    logits = cls_token
    loss = self.criterion(logits, targets)
    output = logits if outcome == 'logit' else self.softmax(logits)
    return output, loss

class SentenceBERT(SentenceTransformer):

  def qualify(self, features, utterances):
    chosen_id = random.randint(0, len(utterances) - 1)
    chosen_utt = utterances[chosen_id]
    chosen_embed = features['sentence_embedding'][chosen_id].unsqueeze(0)

    comparables = []
    for sent_embed, utterance in zip(features['sentence_embedding'], utterances):
      with torch.no_grad():
        score = torch.cosine_similarity(chosen_embed, sent_embed.unsqueeze(0))
      comp = (utterance, round(score.item(), 3))
      comparables.append(comp)
    comparables.sort(key=lambda x: x[1], reverse=True)

    print("Target utterance:", chosen_utt)
    print(f"Out of {len(utterances)} utterances, the 3 closest are:")
    count = 1
    for close, score in comparables[1:4]:
      print(f"   {count})", close, score)
      count += 1
    print(f"And the three furthest are:")
    count = 1
    for far, score in comparables[-3:]:
      print(f"   {count})", far, score)
      count += 1

  def fit(self, train_objective: Tuple[object, nn.Module],
      evaluator, epochs: int = 1,
      steps_per_epoch = None,
      scheduler_name: str = 'WarmupLinear',
      warmup_steps: int = 10000,
      optimizer_class = optim.AdamW,
      optimizer_params : Dict[str, object]= {'lr': 3e-5},
      weight_decay: float = 0.01,
      logging_steps: int = 0,
      evaluation_steps: int = 0,
      output_path: str = None,
      save_best_model: bool = True,
      max_grad_norm: float = 3,
      do_qual: bool=False,
      callback: Callable[[float, int, int], None] = None,
      checkpoint_path: str = None,
      checkpoint_save_steps: int = 2000,
      checkpoint_save_total_limit: int = 0,
      args=None):
    """
    Train the model with the given training objective
    Each training objective is sampled in turn for one batch.
    We sample only as many batches from each objective as there are in the smallest one
    to make sure of equal training with each dataset.

    :param train_objectives: Tuples of (DataLoader, LossFunction). Only accepts on tuple now.
    :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model 
            performance during training on held-out dev data. Used to determine the best model that is saved to disc.
    :param epochs: Number of epochs for training
    :param steps_per_epoch: Number of training steps per epoch. If set to None (default), 
            one epoch is equal the DataLoader size from train_objectives.
    :param scheduler_name: Learning rate scheduler. Available schedulers: 
            constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
    :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), 
            the learning rate is increased from o up to the maximal learning rate. 
            After these many training steps, the learning rate is decreased linearly back to zero.
    :param optimizer_class: Optimizer
    :param optimizer_params: Optimizer parameters
    :param weight_decay: Weight decay for model parameters
    :param logging_steps: If > 0, evaluate the model using evaluator after each number of training steps
    :param evaluation_steps: If > 0 and do qualify print out the closest relations per batch
    :param output_path: Storage path for the model and evaluation files
    :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
    :param max_grad_norm: Used for gradient normalization.
    :param callback: Callback function that is invoked after each evaluation.
        It must accept the following three parameters in this order:
        `score`, `epoch`, `steps`
    :param checkpoint_path: Folder to save checkpoints during training
    :param checkpoint_save_steps: Will save a checkpoint after so many steps
    :param checkpoint_save_total_limit: Total number of checkpoints to store
    """

    ##Add info to model card
    dataloader, loss_model = train_objective
    info_loss_functions =  ModelCardTemplate.get_train_objective_info(dataloader, loss_model)
    info_loss_functions = "\n\n".join([text for text in info_loss_functions])
    eval_name = evaluator.__class__.__module__
    
    info_fit_parameters = {"evaluator": eval_name, "epochs": epochs, "steps_per_epoch": steps_per_epoch,
        "scheduler": scheduler_name, "warmup_steps": warmup_steps, "weight_decay": weight_decay,
        "optimizer_class": str(optimizer_class), "optimizer_params": optimizer_params, 
        "evaluation_steps": evaluation_steps, "logging_steps": logging_steps, "max_grad_norm": max_grad_norm}
    print(info_fit_parameters)
    ifp = json.dumps(info_fit_parameters, indent=4, sort_keys=True)

    self._model_card_text = None
    self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", ifp)
    self.best_score = -9999999
    self.to(self._target_device)
    loss_model.to(self._target_device)

    # Use smart batching
    dataloader.collate_fn = self.smart_batching_collate
    if steps_per_epoch is None or steps_per_epoch == 0:
      steps_per_epoch = len(dataloader)
    num_train_steps = int(steps_per_epoch * epochs)

    # Prepare optimizers
    param_optimizer = list(loss_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
    scheduler = self._get_scheduler(optimizer, scheduler=scheduler_name, 
              warmup_steps=warmup_steps, t_total=num_train_steps)

    global_step = 0
    data_iterators = []
    tok = self._first_module().tokenizer
    
    for epoch in progress_bar(range(epochs), desc="Epoch", total=epochs):
      training_steps = 0
      loss_model.zero_grad()
      loss_model.train()
      chosen_batch = random.randint(0, 100-1) # len(dataloader)

      losses = []
      for features, labels in dataloader:
        if labels.dtype == torch.int64:
          labels = labels.type(torch.float32)

        loss_value = loss_model(features, labels)
        losses.append(loss_value.item())

        if args.loss_function == 'default':
          torch.set_grad_enabled(False)
        else:
          loss_value.backward()

          torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
          optimizer.step()
          optimizer.zero_grad()
          scheduler.step()

        training_steps += 1
        global_step += 1

        if logging_steps > 0 and training_steps % logging_steps == 0:
          avg_loss = round(np.mean(losses), 3) 
          print(f"Step {training_steps}/{steps_per_epoch}, Loss: {avg_loss}")
        if checkpoint_path is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
          print("Saving checkpoint")
          self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
        if do_qual and training_steps == chosen_batch:
          fzero = features[0]
          utterances = tok.batch_decode(fzero['input_ids'], skip_special_tokens=True)

      if do_qual:
        self.qualify(fzero, utterances)
      avg_loss = round(np.mean(losses), 3)
      def caller(raw_score, epoch, steps):
        score = round(raw_score, 3)
        print(f"Step {steps}/{steps_per_epoch}, Loss: {avg_loss}, Score: {score}")
      self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, caller)

    if checkpoint_path is not None:
      self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

class SingleClassifier(AutoModelForSequenceClassification):
  def __init__(self, *args, **kwargs):
    super(SingleClassifier, self).__init__(*args, **kwargs)
    self.name = 'single-classify'
