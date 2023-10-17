import json
import pdb, sys, os
from tqdm import tqdm as progress_bar
from collections import Counter

task = "daily"  # "convai2"
data = json.load(open(f"{task}.json", "r"))

acts = Counter()
splits = ["train", "dev", "test"]
all_data = {split: [] for split in splits}

true_split = "dev"
for idx, example in enumerate(data):
    dataset, split, convo_id, utt_id, segment_id = example["index"].split("_")

    if int(segment_id) == 0:
        if idx > 0 and len(utterance) > 0:
            convo.append(utterance)
        utterance = []   # new utterance

        if int(utt_id) == 0:
            if idx > 0 and len(convo) > 0:
                all_data[true_split].append(convo)
                true_split = split
            convo = []      # new conversation

    seg = {
        "uuid": example["index"],
        "convo_id": convo_id,
        "turn_count": utt_id,
        "segment": example["fs_text"],
        "label": example["annotation"]
    }
    ll = example["annotation"]
    acts[ll] += 1

    utterance.append(seg)
all_data[true_split].append(convo)

# Build out the ontology
total = sum(acts.values())
for act, val in acts.items():
    answer = val / total 
    ratio = round(answer * 100, 1)
    print(f"{act}: {ratio}%")
ontology = list(acts.keys())
ontology = [x for x in ontology if " " not in x] # drops the "No Need to Label"
json.dump(ontology, open("ontology.json" , "w"), indent=4)


print("---")
for split in splits:
    split_data = all_data[split]
    print(f"{len(split_data)} {split} examples")
    json.dump(split_data, open(f"{split}.json" , "w"), indent=4)
