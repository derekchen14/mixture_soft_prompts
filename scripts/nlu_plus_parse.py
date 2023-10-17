import copy
import json
import os, pdb, sys

domains = ["hotels", "banking"]
# Build the cross domain splits
fold_splits = {"train": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
               "dev": [0, 1]}
fold_map = {index: split for split, ids in fold_splits.items() for index in ids}

data = {split: [] for split in fold_splits.keys()}
count = 1
for domain in domains:
  for fold_id in range(20):
    fold_path = os.path.join("original_data", domain, f"fold{fold_id}.json")
    fold_data = json.load(open(fold_path, "r"))

    split = fold_map[fold_id]

    for example in fold_data:
      example["domain"] = domain
      example["fold"] = fold_id
      example["uid"] = f"{split}-{domain}-{count}"
      data[split].append(example)
      count += 1
  print(f"Completed {domain}")

data["test"] = copy.deepcopy(data["dev"])
for split, results in data.items():
    json.dump(results, open(f"{split}.json", "w"), indent=4)

# Build the K-fold splits
data = {"train":[], "dev": [], "test": []}
count = 1
for domain in domains:
  for fold_id in range(20):
    fold_path = os.path.join("original_data", domain, f"fold{fold_id}.json")
    fold_data = json.load(open(fold_path, "r"))


    for example in fold_data:
      example["domain"] = domain
      example["fold"] = fold_id
      for split in data.keys():
        example["uid"] = f"{split}-{domain}-{count}"
        data[split].append(example)
      count += 1
  print(f"Completed {domain}")

if not os.path.exists("kfold_splits"):
  os.makedirs("kfold_splits")
  print(f"Created kfold_splits folder")
for split, results in data.items():
  json.dump(results, open(f"kfold_splits/{split}.json", "w"), indent=4)

# Build the ontology
raw_ont = json.load(open("original_data/ontology.json", "r"))
ontology = {"general": {"intents": {}, "slots": {}},
             "hotels": {"intents": {}, "slots": {}},
            "banking": {"intents": {}, "slots": {}}
           }

for category in ["intents", "slots"]:
  attribute_data = raw_ont[category]

  for attr, details in attribute_data.items():
    desc = details["description"]
    for domain in details["domain"]:
      ontology[domain][category][attr] = desc

json.dump(ontology, open("ontology.json","w"), indent=4)
json.dump(ontology, open("kfold_splits/ontology.json","w"), indent=4)
