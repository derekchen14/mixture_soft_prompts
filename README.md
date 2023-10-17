# Mixture of Soft Prompts for Controllable Data Generation (MSP)"

This respository contains the code for "Mixture of Soft Prompts for Controllable Data Generation" (Chen et al., 2021), accepted at EMNLP 2023

## Introduction
LLMs effectively generate fluent text when the target output follows natural language patterns found on the Internet. However, they struggle with structured prediction tasks when a certain format is required. The difficulty of using LLMs for direct prediction is exacerbated in few-shot learning scenarios, which commonly arise due to domain shift and resource limitations.  We flip the problem on its head by leveraging the LLM as a tool for data augmentation rather than direct prediction. Our proposed Mixture of Soft Prompts (MSP) serves as a parameter-efficient procedure for generating multi-attribute data in a controlled manner. Denoising mechanisms are further applied to improve the quality of synthesized data.

Given an arbitrary NLU task, MSP can produce high quality training data for that task. To the extent that the downstream task continues to exhibit low accuracy after training, we can continue to run MSP again to target specific attributes of the task that are not solved. MSP is currently the only feasible method that exists to enable such a continually improving ML system because our method is both time-efficient and highly reliable. In contrast, while prompt engineering is efficient, it can also be brittle and unreliable. Training a model from scratch requires millions of dollars in compute and months of time.  If you want control *and* consistency, MSP serves as a strong baseline to consider. For more details, please see the full paper linked below.

Paper link: [https://arxiv.org/abs/2109.03079](https://arxiv.org/abs/2303.01580)

## Data Generation Pipeline

### Data Prep
Before running any of the models, the data will need to be in the right place. To use, first download the data for [NLU++](https://aclanthology.org/2022.findings-naacl.154/) (https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/nlupp), [CrossNER](https://arxiv.org/abs/2012.04373) (https://github.com/zliucr/CrossNER) and [TopV2](https://aclanthology.org/2020.emnlp-main.413/) (https://fb.me/TOPv2Dataset). This should be loaded into an assets folder with three new directories and a handful of other files, such as at the static-vars. When running for the first time, additional pre-processing will occur, with the results saved to cache.

### General Usage
All experiment code is run by executing the corresponding command within the shell script `run.sh`, which will kick off the data preparation and training within `main.py`. Please comment or uncomment the appropriate lines in the shell script to get desired behavior. For example, the code to mix the soft prompts, and then generate new data follows:
```
# Parameter efficient fine-tuning with soft prompts
python3 main.py --dataset nlu++ --task soft_prompt --do-train --n-tokens 100 --domain hotels \
      --model godel --size large --source-max-len 128 --quantify --qualify --log-interval 100 \
      --n-epochs 14 --learning-rate 0.1 --batch-size 8 --setting cross --verbose
# Generate the synthetic data
python3 main.py --dataset crossner --task synthesize --n-tokens 100 --source-max-len 64 --setting full \
   --model godel --size large --quantify --method msp --mixture concat --num-shot 2 --domain music \
   --filter --do-save --temperature 2.0 --threshold 2.0 --checkpoint {name_of_saved_ckpt_file}
# Use generated data for training on downstream task
python3 main.py --dataset crossner --task end_to_end --log-interval 500 --do-train --do-save \
     --n-epochs 7 --model godel --size medium --learning-rate 3e-5 --quantify --domain literature \
     --source-max-len 256 --threshold 1.2 --method msp --mixture pooling --setting full --verbose
```

Also important is to install all major dependencies within `requirements.txt` in a new environment. Next, create an folder to store the model outputs, where the default is folder name is results. As with all other settings, this can be changed by updating the params within `run.sh`.  The params are found within `utils/arguments.py`.

### Training and Evaluation
The entire system is governed by argument flags. Kick off training with the `--do-train` option. Activate evaluation using the `--do-eval` flag. To specify the task for training, simply use the `--task` option with either 'synthesize', 'end_to_end', 'fine_tune', 'soft_prompt', or 'in_context'. Options for different source datasets include 'actdial', 'banking', 'crossner', 'nlu++' and 'topv2' which are specified with the `--dataset` flag. Baseline methods we compare against depend on the  `--method` flag. Check for other argument details in the utils/arguments.py file or use the `--help` option of argparse.

Contact
Please email [dc3761@columbia.edu](dc3761@columbia.edu) or reach out on [Twitter](https://twitter.com/derekchen14) for questions or feedback.
