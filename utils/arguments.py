import argparse
import os

def solicit_params():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument("--input-dir", default='assets', type=str,
        help="The input training data file (a text file).")
  parser.add_argument("--output-dir", default='results', type=str,
        help="Output directory where the model predictions and checkpoints are written.")
  parser.add_argument("--dataset", default='actdial', type=str,
        choices=['actdial', 'banking', 'crossner', 'nlu++', 'topv2'],
        help="which dataset to choose from out of all possible options")
  parser.add_argument("--domain", default=None, type=str,
        choices=['ai', 'literature', 'music', 'politics', 'science', 'banking', \
        'hotels', 'reminder', 'weather'], help="Target domain to be tested on.")
  parser.add_argument("--task", default='fine_tune', type=str,
        choices=['synthesize', 'end_to_end', 'fine_tune', 'soft_prompt', 'in_context', 'classify'],
        help="synthesize is for DataAug and CTG, e2e trains with generated data, \
        fine_tune updates all gradients, soft_prompt trains prompts only, \
        in_context performs no backprop at all, just inference, \
        classify trains an attribute classifier for correctness automatic evaluation")
  parser.add_argument("--model", default='gpt', type=str, 
        choices=['t5', 'gpt', 'godel', 'aug', 'bert', 'api'],
        help="The model architecture to be trained or fine-tuned.")
  parser.add_argument("--size", default='small', type=str, choices=['small', 'medium', 'large', 'giant'],
        help="Size of the model, use small for debugging, but report results on giant")
  parser.add_argument("--openai-key", default='', type=str,
        help="The API key for OpenAI's GPT-3 API")
  parser.add_argument("--checkpoint", default='', type=str,
        help="Enter the filename of a checkpoint for manual override")
  parser.add_argument("--icl-type", default='base', type=str, 
        choices=['base', 'cot'],
        help="Prompt engineering for ICL prediction: base, chain of thought.")
  parser.add_argument("--seed", default=42, type=int)

  # Custom paper parameters
  parser.add_argument("--method", default='none', type=str,
        choices=['eda', 'para', 'fill', 'rtt', 'cvae', 'dexpert', 'clm', 'msp', 'none'],
        help="Method of dataset generation, includes both DataAug and CTG")
  parser.add_argument("--mixture", default='concat', type=str,
        choices=['concat', 'attention', 'bottleneck', 'cnn', 'pooling', 'injection'],
        help="How to mix the soft prompts together")
  parser.add_argument("--num-shot", default=2, type=int,
        help="How many exemplars or K-shot to include when performing few shot synthesis")
  parser.add_argument("--num-generations", default=4, type=int,
        help="The multiplier on the number of generations compared to size of the seed set")
  parser.add_argument("--threshold", default=1.4, type=float,
        help="Used as the repetition penalty during inference of generation")
  parser.add_argument("--temperature", default=1.0, type=float,
        help="Temperature for increasing diversity when decoding, mainly for paraphrase")
  parser.add_argument("--source-max-len", default=256, type=int,
        help="Default input length for a model")
  parser.add_argument("--target-max-len", default=128, type=int,
        help="Default output length for a model")
  parser.add_argument("--n-tokens", default=100, type=int,
        help="Number of soft prompt tokens to tune")
  parser.add_argument("--do-guide", action="store_true",
        help="Use additional guidance such as domain during generation")
  parser.add_argument("--filter", action="store_true",
        help="Run additional denoising to clean up the dataset")
  parser.add_argument("--metric", default="f1_score", type=str,
        choices=["accuracy", "f1_score", "intents_acc", "slots_acc", "bleu"],
        help="type of metric to optimize")
  parser.add_argument("--setting", default='few_shot', type=str,
        choices=['few_shot', 'full', 'additional', 'kfold', 'cross'],
        help="Method of dataset preparation, details still unclear")

  # Key settings
  parser.add_argument("--accelerate", action="store_true",
        help="Whether to use accelerate during training for multiple machines.")
  parser.add_argument("--ignore-cache", action="store_true",
        help="Whether to ignore cache and create a new input data")
  parser.add_argument("--debug", action="store_true",
        help="Whether to run in debug mode which is exponentially faster")
  parser.add_argument("--verbose", action="store_true",
        help="Whether to run with extra prints to help debug")
  parser.add_argument("--do-train", action="store_true",
        help="Whether to run training.")
  parser.add_argument("--do-eval", action="store_true",
        help="Whether to run eval on the test set.")
  parser.add_argument("--do-save", action="store_true",
        help="Whether to save models, which override previous checkpoints")
  parser.add_argument("--log-interval", type=int, default=500,
        help="Log every X updates steps.")
  parser.add_argument("--qualify", action='store_true',
        help="Whether to include joint accuracy scores during evaluation")
  parser.add_argument("--quantify", action='store_true',
        help="Whether to include inform/success/BLEU scores during evaluation")
  parser.add_argument("--prune-keep", default=-1, type=int,
        help="Number of models to keep around after pruning, by default does not prune")
  parser.add_argument("--parallel", action="store_true",
        help="Whether to run in parallel")
  parser.add_argument("--patience", default=4, type=int,
        help="patience for early stop, applies to both chunks and epochs")
  # temporary flag for experiments in 0084
  parser.add_argument("--pool-size", default=10, type=int,
        help="Number of exemplars to randomly sample from to put in the prompt")

  # Hyper-parameters for tuning
  parser.add_argument("--batch-size", default=12, type=int,
        help="Batch size per GPU/CPU for training and evaluation.")
  parser.add_argument('--grad-accum-steps', default=1, type=int,
        help='Number of steps for gradient accumulation')
  parser.add_argument("--learning-rate", default=3e-4, type=float,
        help="Model learning rate starting point.")
  parser.add_argument("--drop-rate", default=0.1, type=float,
        help="Dropout rate with default of 10%")
  parser.add_argument("--hidden-size", default=300, type=int,
        help="Hidden dimension for intermediate projection of ranking model")
  parser.add_argument("--embed-dim", default=768, type=int,
        help="Embed dimension for intermediate projection of ranking model")
  parser.add_argument("--weight-decay", default=0.0, type=float,
        help="Weight decay if we apply some.")
  parser.add_argument("--n-epochs", default=3, type=int,
        help="Total number of training epochs to perform.")
  parser.add_argument("--warmup-steps", default=0.1, type=float,
        help="Linear warmup over warmup-steps ratio of total steps")
  parser.add_argument("--sig-threshold", default=0.5, type=float,
        help="sigmoid threshold for multilabel classifier evaluation")

  args = parser.parse_args()
  return args
