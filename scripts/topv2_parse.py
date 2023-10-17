import json
import os, pdb, sys
import pandas as pd
from collections import Counter, defaultdict

splits = ["train", "dev", "test"]
domains = ["alarm", "event", "messaging", "music", "navigation", "reminder", "timer", "weather"]
low_resource = ["reminder", "weather"]
num_spis = 25 # 25, 50, 100, 250, 500, 1000

intent_set = set()
slot_set = set()
intents_by_domain = defaultdict(list)
slots_by_domain = defaultdict(list)
all_data = {split: [] for split in splits}

def extract_labels(parse):
  intents, slots = [], []

  idx = 1
  while idx < (len(parse) - 3):
    part = parse[idx:idx + 3]

    if part == "IN:":
      idx += 3
      phrase = ""
      char = parse[idx]
      while char != " ":
        phrase += char
        idx += 1
        char = parse[idx]
      intents.append(phrase.lower())

    elif part == "SL:":
      idx += 3
      phrase = ""
      char = parse[idx]
      while char != " ":
        phrase += char
        idx += 1
        char = parse[idx]
      slots.append(phrase.lower())

    else:
      idx += 1
  return intents, slots


for split in splits:
  for domain in domains:
    extra = "full"

    if split == "dev":
      filepath = os.path.join("original", f"{domain}_eval.tsv")
    else:
      filepath = os.path.join("original", f"{domain}_{split}.tsv")

    df = pd.read_csv(filepath, sep="\t")
    for idx, row in df.iterrows():
      parse_tree = row["semantic_parse"]
      intents, slots = extract_labels(parse_tree)
      for intent in intents:
        intent_set.add(intent)
        intents_by_domain[domain].append(intent)
      for slot in slots:
        slot_set.add(slot)
        slots_by_domain[domain].append(slot)

      example = {
        'uid': f"{split}-{domain}-{idx + 1}-{extra}",
        'text': row["utterance"],
        'label': parse_tree,
        'intents': intents,
        'slots': slots,
        'domain': domain
      }
      all_data[split].append(example)
    print(f"processed {domain} in {split}")

  for domain in low_resource:
    extra = "few_shot"
    if split == "train":
      target_name = "seqlogical"
      filepath = os.path.join("original", "low_resource_splits", f"{domain}_{split}_{num_spis}spis.tsv")
    elif split == "dev":
      target_name = "seqlogical"
      filepath = os.path.join("original", "low_resource_splits", f"{domain}_valid_{num_spis}spis.tsv")
    else:  # test split
      target_name = "semantic_parse"
      filepath = os.path.join("original", f"{domain}_{split}.tsv")

    df = pd.read_csv(filepath, sep="\t")
    for idx, row in df.iterrows():
      parse_tree = row[target_name]
      intents, slots = extract_labels(parse_tree)
      for intent in intents:
        intent_set.add(intent)
        intents_by_domain[domain].append(intent)
      for slot in slots:
        slot_set.add(slot)
        slots_by_domain[domain].append(slot)

      example = {
        'uid': f"{split}-{domain}-{idx + 1}-{extra}",
        'text': row["utterance"],
        'label': parse_tree,
        'intents': intents,
        'slots': slots,
        'domain': domain
      }
      all_data[split].append(example)
    print(f"processed {domain} in low-resource")

  json.dump(all_data[split], open(f"{split}.json", "w"), indent=4)
  size = len(all_data[split])
  print(f"Saved {size} {split} examples")

# Generate naive ontology, comment out after finalized the ontology.json
domains_ontology = []
for domain in domains:
  if domain in low_resource:
    domains_ontology.append([domain, "target"])
  else:
    domains_ontology.append([domain, "source"])

# move general slots into their own section
slot_counts = Counter()
for domain, slots in slots_by_domain.items():
  for slot in set(slots):
    slot_counts[slot] += 1

generic_slots = []
for slot, count in slot_counts.items():
  if count > 5:
    generic_slots.append(slot)
for domain, slots in slots_by_domain.items():
  trimmed = [i for i in slots if i not in generic_slots]
  slots_by_domain[domain] = []
  for item in trimmed:
    slots_by_domain[domain].append(item)
slots_by_domain['general'] = []
for item in generic_slots:
  slots_by_domain['general'].append(item)

for domain in intents_by_domain:
  intents_by_domain[domain] = list(set(intents_by_domain[domain]))

for domain in slots_by_domain:
  slots_by_domain[domain] = list(set(slots_by_domain[domain]))

ontology = {"domains": domains_ontology,
            "intents": intents_by_domain,
            "slots": slots_by_domain}
json.dump(ontology, open("ontology.json", "w"), indent=4)
