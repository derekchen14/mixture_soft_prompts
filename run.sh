#!/bin/bash
set -xue
# ___________________________________________________
# ________ Fine-tuned Student Model Training ________
# >> Training models with standard train/dev/test splits
# python3 main.py --dataset topv2 --task fine_tune --log-interval 600 --do-train --do-save \
#       --model godel --size medium --learning-rate 3e-5 --source-max-len 256 --batch-size 14 \
#       --domain reminder --threshold 1.2 --n-epochs 10 --quantify --metric accuracy --prune-keep 2
# python3 main.py --dataset crossner --task fine_tune --log-interval 200 --do-train --do-save  \
#       --n-epochs 10 --model godel --size small --learning-rate 1e-4 --quantify --batch-size 10 \
#       --threshold 1.2 --source-max-len 256 --domain literature --verbose

# >> Different dataset settings (full data, few shot, cross-domain, etc.)
# python3 main.py --dataset topv2 --task fine_tune --log-interval 500 --do-train --do-save \
#       --n-epochs 14 --model godel --size small --learning-rate 1e-4 --quantify --domain reminder \
#       --source-max-len 64 --threshold 1.2 --verbose --setting full --metric accuracy
# python3 main.py --dataset nlu++ --task fine_tune --log-interval 500 --do-train \
#       --model godel --size small --learning-rate 1e-4 --setting cross --quantify \
#       --source-max-len 256 --threshold 1.1 --ignore-cache --domain banking

# >> Training and evaluating end-to-end with synthesized data
# python3 main.py --dataset topv2 --task end_to_end --log-interval 600 --do-train --verbose \
#      --n-epochs 7 --model godel --size medium --learning-rate 3e-5 --quantify --domain reminder \
#      --source-max-len 128 --threshold 1.2 --method msp --mixture bottleneck --setting few_shot \
#      --ignore-cache --metric accuracy
# python3 main.py --dataset crossner --task end_to_end --log-interval 500 --do-train --do-save \
#      --n-epochs 7 --model godel --size medium --learning-rate 3e-5 --quantify --domain literature \
#      --source-max-len 256 --threshold 1.2 --method msp --mixture pooling --setting full --verbose
# python3 main.py --dataset nlu++ --task end_to_end --do-eval --model godel --size medium \
#     --domain hotels --quantify --batch-size 24 --threshold 1.2 --checkpoint acc650_lr0.0001_epoch7.pt

# _______________________________________________________
# _______  Mixing Soft Prompts w/ Frozen Large LM  ______
# >> Soft Prompt Tuning, roughly 30% faster
# python3 main.py --dataset nlu++ --task soft_prompt --do-train --n-tokens 100 --domain hotels \
#       --model godel --size large --source-max-len 128 --quantify --qualify --log-interval 100 \
#       --n-epochs 14 --learning-rate 0.1 --batch-size 8 --setting cross --verbose
# accelerate launch main.py --dataset crossner --task soft_prompt --do-train --n-tokens 100 \
#     --model gpt --size giant --source-max-len 256 --quantify --setting full --log-interval 140 \
#     --n-epochs 10 --learning-rate 1e-4 --domain literature --batch-size 4 --accelerate \
#     --grad-accum-steps 8 --target-max-len 100 --verbose
# python3 main.py --dataset actdial --task soft_prompt --do-train --n-tokens 100 \
#   --model gpt --size medium --source-max-len 256 --target-max-len 64 --quantify \
#   --n-epochs 10 --learning-rate 3e-3 --batch-size 7 --warmup-steps 0.05 --do-save

# >> In Context Learning, Inference Only
# python main.py --dataset nlu++ --task in_context --n-tokens 100 --batch-size 16 \
#    --model t5 --size medium --source-max-len 64 --quantify --log-interval 300 \
# python main.py --dataset nlu++ --task in_context --do-train --n-tokens 100 --setting cross \
#   --model gpt --size giant --source-max-len 64 --quantify --batch-size 4 --domain hotels

# >> In context learning, OpenAI GPT
# base
# python main.py --dataset nlu++ --domain banking --task in_context \
#  --model api --size large --openai-key <api_key> --source-max-len 64 \
#  --quantify --log-interval 300 --verbose --pool-size 5 --num-shot 5 --ignore-cache
# chain of thought
#  python main.py --dataset nlu++ --domain hotels --task in_context \
#  --model api --size giant --openai-key <api_key> --source-max-len 64 \
#  --quantify --log-interval 300 --verbose --pool-size 5 --num-shot 5 --icl-type cot --ignore-cache
# python main.py --dataset crossner --domain politics --task in_context \
#  --model api --size giant --openai-key <api_key> --source-max-len 64 \
#  --quantify --log-interval 300 --verbose --pool-size 5 --num-shot 5 --icl-type cot --ignore-cache
# python main.py --dataset topv2 --domain weather --task in_context \
#  --model api --size giant --openai-key <api_key> \
#  --quantify --log-interval 300 --verbose --metric accuracy --pool-size 5 --num-shot 5 --icl-type cot --ignore-cache


# ______________________________________________________
# ________ Synthesize task for DataAug and CTG and OpenAI  ________
# >> Pre-train a Generator for Synthesizing Data
# python main.py --dataset nlu++ --task synthesize --do-train --quantify --qualify --do-save \
#    --size medium --batch-size 4 --n-epochs 12 --learning-rate 4e-5 --patience 3 --prune-keep 1\
#    --source-max-len 64 --target-max-len 64 --ignore-cache --verbose --metric bleu --method dexpert
# python main.py --dataset topv2 --setting few_shot --task synthesize --do-train --quantify --do-save \
#    --size medium --batch-size 4 --n-epochs 12 --learning-rate 4e-5 --patience 5 --prune-keep 1\
#    --source-max-len 64 --target-max-len 128 --ignore-cache --metric bleu --method dexpert
# python main.py --dataset crossner --setting full --task synthesize --do-train --quantify --do-save \
#    --size medium --batch-size 4 --n-epochs 12 --learning-rate 4e-5 --patience 5 --prune-keep 2\
#    --source-max-len 256 --target-max-len 256 --ignore-cache  --metric bleu --method dexpert

# >> Data Augmentation (no training)
# python main.py --dataset nlu++ --task synthesize --do-save --model aug --method para \
#      --temperature 1.4 --debug
# python main.py --dataset crossner --task synthesize --model aug --method eda --threshold 0.2
# python main.py --dataset topv2 --task synthesize --model aug --method rtt --do-save
# python main.py --dataset nlu++ --task synthesize --model aug --method fill --do-save --verbose

# >> Controlled Text Generation (no training)
# python main.py --dataset nlu++ --domain hotels --setting cross --task synthesize --target-max-len 48 \
#       --size medium --model bert --method cvae --verbose --n-tokens --do-save 
# python main.py --dataset topv2 --domain weather --task synthesize --target-max-len 48 \
#       --size medium --method prefix --verbose --threshold 1.4 --temperature 0.7 --do-save
# python main.py --num-shot 50 --dataset crossner --domain music --task synthesize --target-max-len 48 \
#       --size medium --method clm --verbose --threshold 1.4 --temperature 0.7 --do-save
# python main.py --dataset nlu++ --task synthesize --n-tokens 100 --batch-size 8 --do-save \
#    --model gpt --size large --source-max-len 64 --quantify --log-interval 100 --verbose \
#    --method msp --mixture concat --num-shot 1 --metric bleu --domain banking --threshold 1.2

# >> OpenAI GPT
# python main.py --dataset crossner --domain music --task synthesize --verbose \
#  --model api --size giant --openai-key <api_key> \
#  --num-shot 2 --method none --setting full --do-save --ignore-cache --num-generations 6
# python main.py --dataset nlu++ --domain hotels --task synthesize --verbose \
#  --model api --size giant --openai-key <api_key> \
#  --num-shot 2 --method none --setting cross --do-save --ignore-cache --num-generations 6
# python main.py --dataset topv2 --domain weather --task synthesize --verbose \
#  --model api --size giant --openai-key <api_key> \
#  --num-shot 2 --method none --setting few_shot --do-save --ignore-cache --num-generations 6

# __________________________________________________________
# ________ Experiments with Mixture of Soft Prompts ________
# >> Changing the composition attributes
# python3 main.py --dataset nlu++ --task synthesize --do-train --do-save --quantify --method msp \
#    --n-epochs 14 --model godel --size giant --source-max-len 64 --target-max-len 64 --metric bleu \
#    --learning-rate 0.1 --mixture concat --batch-size 6 --grad-accum-steps 4 --log-interval 100
# python3 main.py --dataset crossner --task synthesize --do-train --quantify --verbose --do-save \
#    --n-epochs 10 --model godel --size large --source-max-len 128 --target-max-len 64 --metric bleu \
#    --learning-rate 0.3 --method msp --mixture concat --batch-size 6 --grad-accum-steps 4 \
#    --filter --log-interval 400 --setting full --qualify --prune-keep 6
# >> Generate data using a trained MSP model
# python3 main.py --dataset crossner --task synthesize --n-tokens 100 --source-max-len 64 --setting full \
#    --model godel --size large --quantify --method msp --mixture concat --num-shot 2 --domain music \
#    --filter --do-save --temperature 2.0 --threshold 2.0 --checkpoint acc118_lr0.3_epoch2.pt
# python3 main.py --dataset topv2 --task synthesize --n-tokens 100 --source-max-len 64 --ignore-cache \
#    --model godel --size giant --quantify --method msp --mixture bottleneck  --num-generations 10 \
#    --filter --do-save --temperature 1.6 --threshold 2.0 --domain reminder --checkpoint acc380_lr0.3_epoch3.pt

# >> Ablations
# python3 main.py --dataset crossner --task synthesize --do-train --debug --learning-rate 0.03 \
#    --model gpt --size giant --quantify --qualify --metric bleu --setting full \
#    --n-epochs 10 --method msp --mixture concat --batch-size 4 --grad-accum-steps 6
# python3 main.py --dataset topv2 --task synthesize --do-train --do-save --n-tokens 100 \
#    --model gpt --size giant --source-max-len 64 --quantify --qualify --metric bleu \
#    --n-epochs 14 --learning-rate 0.01 --batch-size 6 --prune-keep 6 --patience 5 \
#    --method msp --mixture concat --verbose --grad-accum-steps 4 --num-shot 1 --threshold 1.2
# python main.py --dataset nlu++ --task synthesize --do-train --debug --n-tokens 50 \
#    --model gpt --size small --source-max-len 128 --quantify --log-interval 200 \
#    --n-epochs 7 --learning-rate 1e-5 --method msp --mixture concat --filter

# _____________________________________________
# ______________ Special Modes ________________
# >> Qualitative Testing
# python interact.py --task synthesize --model gpt --size small --threshold 1.2 --temperature 1.0 \
#       --dataset nlu++ --domain hotels --setting cross --num-shot 3 --target-max-len 32 --do-guide
# python interact.py --task synthesize --model gpt --size small --threshold 1.2 --temperature 1.0 \
#       --dataset crossner --domain music --setting full --num-shot 3 --target-max-len 32
# python interact.py --task synthesize --model gpt --size small --threshold 1.2 --temperature 1.0 \
#       --dataset topv2 --domain reminder --num-shot 3 --target-max-len 32 --do-guide --verbose
# >> Train a classifier for automated text evaluation
# python automatic.py --task classify --model bert --size small --do-train --debug  --quantify \
#         --learning-rate 3e-5 --dataset topv2 --domain reminder --setting full

# >> Automated Text Evaluation of a given generated data file
# python3 automatic.py --dataset nlu++ --task classify --do-eval --quantify \
#       --generated-data-file ./results/msp_example.json
# >> Train oracle classifier for synthesis
# python automatic.py --dataset nlu++ --task classify --do-train --do-save \
# python automatic.py --dataset topv2 --task classify --do-train --do-save \
#       --model bert --size large --qualify --ignore-cache --n-epochs 15 \
#       --verbose  --learning-rate 5e-5 --batch-size 8 --setting full --prune-keep 1
# >> Automated Text Evaluation of a given generated data file with a trained discriminator
# python3 automatic.py --dataset topv2 --domain weather --task classify --do-eval --quantify \
#        --generated-data-file assets/cache/topv2/msp_meanpool.json --size medium \
#        --checkpoint assets/topv2_weather_intent_classifier_roberta_large \
# python automatic.py --dataset nlu++ --domain hotels --method clm --qualify --do-guide \
#   --setting cross --task synthesize
# python automatic.py --dataset topv2 --task classify --do-eval --verbose \
#     --do-save --model bert --size large --ignore-cache --qualify \
#     --batch-size 8 --setting full --generated-data-path assets/cache/topv2/clm_weather_few_shot.json

# >> Evaluation from saved checkpoint
# python main.py --dataset nlu++ --task soft_prompt --do-eval --n-tokens 100 \
#   --model gpt --size small --source-max-len 64 --quantify \
#   --batch-size 8 --checkpoint acc070_lr0.003_epoch5.pt
# python3 main.py --dataset crossner --task fine_tune --do-eval --model godel --size medium \
#     --domain literature --quantify --batch-size 10 --checkpoint acc650_lr0.0001_epoch7.pt

# >> Smoke test to make sure the code is operational, should reach ~80% F1-score
python main.py --dataset actdial --task fine_tune --do-train --debug --source-max-len 64 \
    --model t5 --size small --quantify --log-interval 140 --learning-rate 3e-4
