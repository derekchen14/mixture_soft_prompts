import torch
from accelerate import Accelerator
accelerator = Accelerator()

if torch.cuda.is_available():
    dtype = 'cuda'
# elif torch.backends.mps.is_available():
#     dtype = 'mps'
else:
    dtype = 'cpu'

# device = torch.device(dtype)
device = accelerator.device
debug_break = 5

mg = "microsoft/GODEL-v1_1-"
eai = "EleutherAI/"
md = "microsoft/deberta-"
gf = "google/flan-"

CHECKPOINTS = {
    'gpt': {
        'small': 'gpt2',                    #   335 mil
        'medium': 'gpt2-large',             #   774 mil
        'large': 'gpt2-xl',                 #  1.5 bil
        'giant': eai+'gpt-j-6B'},           #  6.0 bil
    't5': {
        'small': 't5-base',                 #   220 mil
        'medium': 't5-large',               #   770 mil
        'large': 't5-3b',                   #  3.0 bil
        'giant': 't5-11b'},                 # 11.0 bil
    'opt': {
        'small': 'facebook/opt-350m',       #   350 mil
        'medium': 'facebook/opt-1.3b',      #  1.3 bil
        'large': 'facebook/opt-6.7b',       #  6.7 bil
        'giant': 'facebook/opt-13.0b'},     # 13.0 bil
    'godel': {
        'small': mg+'base-seq2seq',         #   220 mil
        'medium': mg+'large-seq2seq',       #   770 mil
        'large': gf+'t5-xl',                #  3.0 bil
        'giant': gf+'t5-xxl'},              # 11.0 bil
    'bert': {
        'small': "roberta-base",            #   125 mil
        'medium': "roberta-large",          #   355 mil
        'large': md+"large",                #   400 mil
        'giant': md+"xxlarge-v2",           #  1.5 bil
    },
    'api': {
        'small': "text-curie-001",
        'medium': "text-da-vinci-003",
        'large': "gpt-3.5-turbo",
        'giant': "gpt-4",
    }
}

DATASETS = {
    'nlu++': 'Multi-aspect Intent Detection',
    'topv2': 'Compositional Semantic Parsing',
    'banking': 'Single-aspect Intent Detection',
    'crossner': 'Cross-domain Named Entity Recognition',
}
STOP_TOKENS = ['done', 'exit', 'finish', 'stop', 'end', 'q', 'quit']
CROSS = {'banking': 'hotels', 'hotels': 'banking'}

MAX_ATTEMPTS = 7
ATTRIBUTE_TOKEN_LEN = 20  # must be divisible by 4
MAX_MIXTURE_SIZE = 10
