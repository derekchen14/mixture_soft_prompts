import os, pdb, sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from assets.static_vars import ATTRIBUTE_TOKEN_LEN, MAX_MIXTURE_SIZE, device
num_to_string = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]

class SoftEmbedding(nn.Module):
  """ Performs traditional non-controlled data augmentation """

  def __init__(self, original_emb: nn.Embedding, n_tokens: int=10, num_exemplars: int=5,
              init_from_vocab: bool=True, tokenizer=None):
    """appends learned embedding to original embedding
    Args:
      original_emb (nn.Embedding): original transformer word embedding
      n_tokens (int, optional): number of tokens for task. Defaults to 10.
      num_exemplars (int, optional): number of exemplars which determines the initialization text
      init_from_vocab (bool, optional): initalizes from default vocab.
      tokenizer: a tokenier for init_text
    """
    super().__init__()
    self.name = 'base-embedding'
    self.original_emb = original_emb
    self.n_tokens = n_tokens

    init_text = f"Show me {num_to_string[num_exemplars + 1]} distinct utterances that all express the "
    init_prompt_value = self.init_embedding(
      original_emb, n_tokens, init_from_vocab, tokenizer, init_text
    )
    self.soft_prompt = nn.Parameter(init_prompt_value, requires_grad=True).to(device)
    print(f"Initialized soft prompts with dimension {self.soft_prompt.shape}")

  def init_embedding(self, original_emb, n_tokens, init_from_vocab, tokenizer, init_text):
    """initializes learned embedding
      either from vocab, random initialization or a custom set of init_text

    Returns:
      torch.float: initialized using original schemes
    """
    if init_from_vocab:
      init_embd = self.original_emb.weight[:n_tokens].clone().detach()
      if tokenizer is not None:
        # replace the embedding with init_text from static vars
        tokens = tokenizer(init_text)
        for i, token in enumerate(tokens['input_ids']):
          init_embd[i] = self.original_emb.weight[token]
          if i + 1 >= init_embd.shape[0]:
            break
        print(f"Initialized embedding with '{init_text}'")
      else:
        print(f"Initialized embedding with tokens from the vocabulary")
    else:
      rr = 0.5 # random_range
      dimension = original_emb.weight.size(1)
      init_embd = torch.FloatTensor(n_tokens, dimension).uniform_(-rr, rr)
      print(f"Initialized embedding with random vectors")
    return init_embd

  def forward(self, tokens):
    raise NotImplementedError

  @classmethod
  def from_saved_embedding(cls, args, original_emb, prompt_path):
    if args.accelerate:
      weights = torch.nn.Parameter(torch.load(prompt_path).half())
    else:
      weights = torch.load(prompt_path)

    num_prompt_tokens = weights.shape[0]
    previous_embed = cls(original_emb, num_prompt_tokens)
    previous_embed.soft_prompt = weights
    print(f"Loaded prompt weights from {prompt_path}")
    return previous_embed

  def save_prompt_embedding(self, save_path, prompt_file):
    prompt_path = os.path.join(save_path, prompt_file)
    torch.save(self.soft_prompt, prompt_path)
    print(f"Saved a soft prompt at {prompt_path}")

class CausalEmbedding(SoftEmbedding):

  def __init__(self, original_emb: nn.Embedding, n_tokens: int=10, num_exemplars: int=5,
              init_from_vocab: bool=True, tokenizer=None):
    super().__init__(original_emb, n_tokens, num_exemplars, init_from_vocab, tokenizer=tokenizer)
    self.name = 'causal-embedding'

  def forward(self, tokens):
    """run forward pass
    Args:
      tokens (torch.long): input tokens before encoding
    Returns:
      torch.float: encoding of text concatenated with learned task specifc embedding
    
    Reasoning: During the first pass, we are operating in the encoding phase, so we
      modify the input sequence to use the soft prompt.  In subsequent passes, we are
      now operating in the generation phase, so we just process the tokens normally.
      Since generation operates one token at a time, we check whether the sequence
      length is <= 1 token to recognize when we are in the generation phase.
    """
    batch_size, seq_len = tokens.shape
    # use soft prompt unless we are using the autoregressive `.generate()`
    if seq_len > 1:
      input_embed = self.original_emb(tokens[:, self.n_tokens:])
      learned_embed = self.soft_prompt.repeat(batch_size, 1, 1)
      final_embed = torch.cat([learned_embed, input_embed], 1)
    else:
      final_embed = self.original_emb(tokens)
    return final_embed

class Seq2SeqEmbedding(SoftEmbedding):

  def __init__(self, original_emb: nn.Embedding, n_tokens: int=10, num_exemplars: int=5,
              init_from_vocab: bool=True, tokenizer=None):
    super().__init__(original_emb, n_tokens, num_exemplars, init_from_vocab, tokenizer=tokenizer)
    self.name = 'seq2seq-embedding'

  def forward(self, tokens):
    """run forward pass
    Args:
      tokens (torch.long): input tokens before encoding
    Returns:
      torch.float: encoding of text concatenated with learned task specifc embedding

    Reasoning: During the first pass, we are operating in the encoding phase, which we
      recognize by checking that the first token in the first example contains a negative
      value.  This token_id == -1 since we manually set it as the placeholder earlier.
      When this is not the case, then we are in the generation phase, so we may simply
      proceed as normal with the original embedding.
    """
    if tokens[0][0] < 0:  # if first token is a soft prompt placeholder
      input_embed = self.original_emb(tokens[:, self.n_tokens:])
      learned_embed = self.soft_prompt.repeat(tokens.shape[0], 1, 1)
      final_embed = torch.cat([learned_embed, input_embed], 1)
    else:
      final_embed = self.original_emb(tokens)
    return final_embed


class AttributeAttention(nn.Module):
  def __init__(self, in_dim, temperature):
    super().__init__()
    self.attn = nn.Linear(in_dim, in_dim, bias=False)
    self.attn_non_linear = nn.SiLU()
    self.layer_norm = nn.LayerNorm(in_dim)
    self.temperature = temperature

  def forward(self, input_embed, mixture):
      # First we project the input_embeding into a new space that fits with mixtures
      # We are learning to project the embedding such that multiplication with the mixture
      # produces attention scores
      max_pool_inputs_embeds, _ = torch.max(input_embed, 0)
      x = self.attn(max_pool_inputs_embeds)
      x = self.layer_norm(x)
      # now we get attention scores by mutipling mixture and the projection
      # softmax produces a weighting scheme
      attn_scores = (mixture * x) / self.temperature
      normalized_attn_scores = F.softmax(attn_scores, -1)
      mixture = torch.einsum('bpl, bpd -> pd', normalized_attn_scores, mixture)
      return mixture

class AttributeBottleneck(nn.Module):
  def __init__(self, in_dim, hidden_dim, temperature):
    super().__init__()
    self.attn_W_down = nn.Linear(in_dim, hidden_dim, bias=False)
    self.attn_W_up = nn.Linear(hidden_dim, in_dim, bias=False)
    self.attn_non_linear = nn.SiLU()
    self.layer_norm = nn.LayerNorm(in_dim)
    self.temperature = temperature

  def forward(self, input_embed, mixture):
      # First we project the input_embeding into a new space that fits with mixtures
      # We are learning to project the embedding such that multiplication with the mixture
      # produces attention scores
      max_pool_inputs_embeds, _ = torch.max(input_embed, 0)
      x = self.attn_W_down(max_pool_inputs_embeds)
      x = self.attn_non_linear(x)
      x = self.attn_W_up(x)
      x = self.layer_norm(x)
      # now we get attention scores by mutipling mixture and the projection
      # softmax produces a weighting scheme
      attn_scores = (mixture * x) / self.temperature
      normalized_attn_scores = F.softmax(attn_scores, -1)
      mixture = torch.einsum('bpl, bpd -> pd', normalized_attn_scores, mixture)
      return mixture


class AttributeConvolution(nn.Module):
  """
  Mixes prompts through convolution
  """
  def __init__(self, emb_dim, stack_height=MAX_MIXTURE_SIZE):
    super().__init__()
    self.emb_dim = emb_dim
    self.max_stack_height = stack_height
    self.attr_len = ATTRIBUTE_TOKEN_LEN
    self.cnn = nn.Sequential(
        nn.Conv2d(self.max_stack_height, self.max_stack_height // 2, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(self.max_stack_height // 2, 1, kernel_size = 3, padding = 1),
        nn.ReLU()
    )

  def forward(self, x):
    padded_tensor = torch.ones((1, self.max_stack_height, ATTRIBUTE_TOKEN_LEN, self.emb_dim)).to(device)
    padded_tensor[0, :x.shape[0], :, :] = x
    return self.cnn(padded_tensor).squeeze()


class AttributeEmbedding(nn.Module):

  def __init__(
      self, args, attributes: list, original_emb: nn.Embedding, num_sets: int=1, 
          frozen: bool=False, tokenizer = None, attribute_init_texts = None):
    """ Introduces new custom parameters to represent the attributes
    Args:
      args (dict): group of argument flags
      attributes (list): decides how many unique embeddings to create
      original_emb (nn.Embedding): to be used when initializing from attribute name
      num_sets (int): number of attribute sets, if greater than 1 then we are
        encoding different types of attributes, such as intents and domains
      frozen (bool): if True, freeze the parameters to their initial encoding
      tokenizer: a tokenizer for init_text
      attribute_init_texts (list[str]): list of texts to initialize the attributes from
        must match the length of attributes
    """
    super().__init__()
    self.name = 'attribute-embedding'
    self.original_emb = original_emb
    self.multi_attribute = num_sets > 1

    self.instruction_prompt = None
    self.constraints = None
    self.attribute_map = None
    self.attribute_embedding = None
    
    self.mixture_type = args.mixture
    self.model_type = args.model

    if self.mixture_type == 'attention':
      self.attention = AttributeAttention(original_emb.weight.size(1), args.temperature)
      self.attention.to(device)

    if self.mixture_type == 'bottleneck':
      self.bottleneck = AttributeBottleneck(original_emb.weight.size(1), args.hidden_size, args.temperature)
      self.bottleneck.to(device)

    if self.mixture_type == 'cnn':
      stack_height = MAX_MIXTURE_SIZE
      if args.dataset == 'nlu++':
        stack_height = 6
      elif args.dataset == 'crossner':
        stack_height = 8
      elif args.dataset == 'topv2':
        stack_height = 10
      self.cnn_mixture = AttributeConvolution(
        original_emb.weight.size(1), stack_height=stack_height
      )
      self.cnn_mixture.to(device)

    if len(attributes) > 0:
      if self.multi_attribute:
        self.attribute_map = []
        self.attribute_embedding = []
        categories = ['intent', 'slot']  # Remove to generalize
        assert(len(categories) == len(attributes))

        for idx, attrs in enumerate(attributes):
          attr_map = {attr:j for j, attr in enumerate(attrs)}
          self.attribute_map.append(attr_map)
          category, attr_init_text = categories[idx], attribute_init_texts[idx]

          init_attr_values = self.initialize_tokens(original_emb, len(attrs), tokenizer, attr_init_text)
          attr_embed = nn.Parameter(init_attr_values, requires_grad=not frozen).to(device)
          self.attribute_embedding.append(attr_embed)
          print(f"Initialized {category} tokens with dimension {attr_embed.shape}")

      else:
        self.num_attributes = len(attributes)
        self.attribute_map = {attr:idx for idx, attr in enumerate(attributes)}

        init_attr_values = self.initialize_tokens(original_emb, len(attributes), tokenizer, attribute_init_texts)
        self.attribute_embedding = nn.Parameter(init_attr_values, requires_grad=not frozen).to(device)
        print(f"Initialized attribute tokens with dimension {self.attribute_embedding.shape}")

  def initialize_tokens(self, original_emb, n_attributes, tokenizer=None, attribute_init_texts=None):
    """initializes learned embedding
    random_range (float, optional): range to init embedding, only applies
      when not initializing from vocab. Defaults to 0.5.
    Returns:
      torch.float: initialized using original schemes
    """
    start, stop = 0, ATTRIBUTE_TOKEN_LEN
    init_embeds = []
    for _ in range(n_attributes):
      embed = self.original_emb.weight[start:stop].clone().detach()
      init_embeds.append(embed)

      start += ATTRIBUTE_TOKEN_LEN
      stop += ATTRIBUTE_TOKEN_LEN

    if attribute_init_texts:
      if not tokenizer:
        raise ValueError("tokenizer must be provided if attribute_init_texts is provided")
      if n_attributes != len(attribute_init_texts):
        raise ValueError(f"Number of attributes {n_attributes} does not match number of attribute_init_texts")

      for n in range (n_attributes):
        tokens = tokenizer(attribute_init_texts[n])
        for i, token in enumerate(tokens['input_ids']):
          init_embeds[n][i] = self.original_emb.weight[token]
          if i + 1 >= init_embeds[n].shape[0]:
            break

      print(f"Initialized embedding with texts for attribute embeds")
    return torch.stack(init_embeds)

  def forward(self, token_batch):
    """run forward pass
    Args:
      token_batch (torch.long): input tokens before encoding (batch_size x seq_len)
    Returns:
      final_embed (torch.float): encoding of text prepended with learned task specifc
      embeddings of shape (batch_size x seq_len x embed_dim)
    """
    if token_batch[0][0] < 0 or (self.model_type=='gpt' and token_batch.shape[1] > 1):
      final_embeddings = []
      instruct_embed = self.instruction_prompt.soft_prompt
      instruct_len = self.instruction_prompt.n_tokens

      for tokens, attributes, pad_len in zip(token_batch, self.constraints, self.pad_lengths):
        attr_len = self.calc_attribute_length(attributes)
        prefix_len = instruct_len + pad_len

        pad_embed = self.original_emb(tokens[instruct_len:prefix_len])
        input_embed = self.original_emb(tokens[prefix_len+attr_len:])
        attr_embed = self.embed_constraints(input_embed, attributes)
        final_embed = torch.cat([instruct_embed, pad_embed, attr_embed, input_embed])
        final_embeddings.append(final_embed)

      return torch.stack(final_embeddings).to(device)

    else: # do not use soft prompt if we are in the generation phase
      return self.original_emb(token_batch)

  @staticmethod
  def repeat_to_fill(descriptions, tokenizer):
    desc_embedding = tokenizer(descriptions)['input_ids']

    filled = []
    for tokens, description in zip(desc_embedding, descriptions):
      num_repeats = (ATTRIBUTE_TOKEN_LEN // len(tokens)) + 1  # add one so we go over
      filled.append( f"{description} " * num_repeats )
    return filled

  def _get_tokens(self, constraint_queries, level):
    """ given a list of attribute strings written in canonical form, will return a list of the
    attribute token embeddings for feeding into a model. """
    attribute_embeds = []
    for query in constraint_queries:
      if self.multi_attribute:
        attr_index = self.attribute_map[level][query]
        attr_embed = self.attribute_embedding[level][attr_index]
      else:
        attr_index = self.attribute_map[query]
        attr_embed = self.attribute_embedding[attr_index]
      attribute_embeds.append(attr_embed)
    return attribute_embeds

  def set_constraints(self, metadata):
    """
    set_constraints that will be used when running a forward pass
    If no constraints are set, only the instruction prompt is used with the domain
    sanity check that constraints are found within self.attribute_map keys.
    """
    for constraints in metadata['constraints']:

      if self.multi_attribute:
        num_sets = len(constraints)  # should be 2
        for index in range(num_sets):
          for category_constraint in constraints[index]:
            if category_constraint not in self.attribute_map[index].keys():
              raise ValueError(f'Constraint: {category_constraint} not in the mapping')

      else:
        # random.shuffle(constraints)  # if you want to shuffle the order
        for constraint in constraints:
          if constraint not in self.attribute_map.keys():
            raise ValueError(f'Constraint: {constraint} not in the ontology')

    self.constraints = metadata['constraints']
    self.pad_lengths = metadata['pad_lengths']

  def calc_attribute_length(self, constraint_set, level=-1):
    if len(constraint_set) == 0:
      return 0

    if self.multi_attribute and level < 0:
      con_sets, levels = constraint_set, [0,1]
      lengths = [self.calc_attribute_length(cs, lvl) for cs, lvl in zip(con_sets, levels)]
      attr_len = sum(lengths)
    else:
      if self.mixture_type == 'concat':
        attr_len = len(constraint_set) * ATTRIBUTE_TOKEN_LEN
      else:
        attr_len = ATTRIBUTE_TOKEN_LEN
    return attr_len

  def embed_constraints(self, input_embed, constraint_set, level=-1):
    if len(constraint_set) == 0:
      _, embed_dim = input_embed.shape
      active_device = input_embed.device
      return torch.empty((0, embed_dim), device=active_device)
    
    if self.multi_attribute and level < 0:
      con_sets, levels = constraint_set, [0,1]
      mixed = [self.embed_constraints(input_embed, cs, lvl) for cs, lvl in zip(con_sets, levels)]
      attr_embed = torch.concat(mixed)
    else:
      constraint_tokens = self._get_tokens(constraint_set, level)
      attr_embed = self.mix_operation(constraint_tokens, input_embed)
    return attr_embed

  def mix_operation(self, constraint_tokens, input_embed):
    attr_embed = torch.cat(constraint_tokens)  # (attr_token_len * n, embed_dim)

    if self.mixture_type == 'attention':
      joined_embed = torch.stack(constraint_tokens, dim=0)
      attr_embed = self.attention(input_embed, joined_embed)
    elif self.mixture_type == 'bottleneck':
      joined_embed = torch.stack(constraint_tokens, dim=0)
      attr_embed = self.bottleneck(input_embed, joined_embed)
    elif self.mixture_type == 'cnn':
      joined_embed = torch.stack(constraint_tokens, dim=0)
      attr_embed = self.cnn_mixture(joined_embed)
    elif self.mixture_type == 'pooling':
      stacked_embed = torch.stack(constraint_tokens, dim=0)
      attr_embed = torch.mean(stacked_embed, dim=0)

    return attr_embed

  @classmethod
  def from_saved_embedding(cls, args, original_emb, ckpt_path):
    if not ckpt_path:
      return cls(args, [], original_emb, frozen=True)

    prompt_file = ckpt_path.split('/')[-1]
    embedding_path = ckpt_path.replace(prompt_file, f"attr_vals_{prompt_file}")
    mapping_path = ckpt_path.replace(prompt_file, f"attr_map_{prompt_file}")

    num_sets = 2 if args.dataset == 'topv2' else 1
    previous_embed = cls(args, [], original_emb, num_sets, frozen=True)
    previous_embed.attribute_embedding = torch.load(embedding_path)
    previous_embed.attribute_map = torch.load(mapping_path)

    if args.mixture == 'attention':
      attention_path = ckpt_path.replace(prompt_file, f"attention_{prompt_file}")
      previous_embed.attention = torch.load(attention_path)
    elif args.mixture == 'bottleneck':
      bottleneck_path = ckpt_path.replace(prompt_file, f"bottleneck_{prompt_file}")
      previous_embed.bottleneck = torch.load(bottleneck_path)
    elif args.mixture == 'cnn':
      cnn_path = ckpt_path.replace(prompt_file, f"cnn_{prompt_file}")
      previous_embed.cnn_mixture = torch.load(cnn_path)
    
    print(f"Loaded prompt weights from {embedding_path} and {mapping_path}")
    return previous_embed

  def save_prompt_embedding(self, save_path, filename):
    attr_path = os.path.join(save_path, f"attr_vals_{filename}")
    torch.save(self.attribute_embedding, attr_path)

    attr_map = os.path.join(save_path, f"attr_map_{filename}")
    torch.save(self.attribute_map, attr_map)

    if self.mixture_type == 'attention':
      attention_path = os.path.join(save_path, f"attention_{filename}")
      torch.save(self.attention, attention_path)
    elif self.mixture_type == 'bottleneck':
      bottleneck_path = os.path.join(save_path, f"bottleneck_{filename}")
      torch.save(self.bottleneck, bottleneck_path)
    elif self.mixture_type == 'cnn':
      cnn_path = os.path.join(save_path, f"cnn_{filename}")
      torch.save(self.cnn_mixture, cnn_path)
    print(f"Saved attribute prompts at {attr_path} and {attr_map}")
