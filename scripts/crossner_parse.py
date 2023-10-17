import json
import os, pdb, sys

# conll2003 is general news
domains = ["literature", "music", "politics", "science", "conll2003", "ai"]
splits = ["train", "dev", "test"]
tag_map = {
	"organisation": "organization",
	"misc": "miscellaneous",
	"politicalparty": "party",
	"literarygenre": "genre",
	"musicalinstrument": "instrument",
	"musicalartist": "artist",
	"musicgenre": "genre",
	"chemicalcompound": "compound",
	"chemicalelement": "element",
 	"astronomicalobject": "astronomy",
 	"academicjournal": "journal",
 	"programlang": "language"
}
domain_map = {
	"literature": "literary",
	"music": "musical",
	"politics": "political",
	"science": "scientific or academic",
	"ai": "artificial intelligence"
}
entity_map = {
	"location": "location place or area",
	"person": "person or individual",
	"miscellaneous": "misc miscellaneous or other",
	"organization": "organization organisation building or group"
}


all_data = {split: [] for split in splits}
ontology = {}
for split in splits:

	count = 1
	for domain in domains:
		if split == "train":
			name_entity = set()

		filepath = os.path.join("original", domain, f"{split}.txt")

		labels = []
		tokens = []
		inside = False
		named = ""
		with open(filepath, "r") as file:
			for line in file:
				row = line.rstrip('\n')
				try:
					token, tag = row.split('\t')
					tokens.append(token)

					if inside:
						if tag.startswith("B"):
							entity = " ".join(parts)
							labels.append((named, entity))
							name_entity.add(named)

							state, named = tag.split('-')
							if named in tag_map:
								named = tag_map[named]
							parts = [token]
							inside = True

						if tag.startswith("O"):
							entity = " ".join(parts)
							labels.append((named, entity))
							name_entity.add(named)
							inside = False

						if tag.startswith("I"):
							parts.append(token)

					else:
						if tag.startswith("B"):
							state, named = tag.split('-')
							if named in tag_map:
								named = tag_map[named]
							parts = [token]
							inside = True

				except(ValueError):
					if len(labels) == 0 and len(tokens) < 5:
						continue

					if len(tokens) > 0:
						example = {
							'uid': f"{split}-{domain}-{count}",
							'text': ' '.join(tokens),
							'labels': labels,
							'domain': 'general' if domain == 'conll2003' else domain
						}
						all_data[split].append(example)
					tokens = []
					labels = []
					count += 1

			if len(tokens) > 0:
				example = {
					'uid': f"{split}-{domain}-{count}",
					'text': ' '.join(tokens),
					'labels': labels,
					'domain': 'general' if domain == 'conll2003' else domain
				}
				all_data[split].append(example)

		if split == "train":
			if domain == "conll2003":
				ontology["general"] = {}
				for entity in name_entity:
					description = entity_map[entity]
					ontology["general"][entity] = description
			else:
				ontology[domain] = {}
				for entity in name_entity:
					description = f"{domain_map[domain]} {entity}"
					ontology[domain][entity] = description

	print(f"Saving {split}.")
	json.dump(all_data[split], open(f"{split}.json", "w"), indent=4)

# Generate naive ontology, comment out after finalized the ontology.json
general_named = ontology.get("general")
for domain, name_entities in ontology.items():
	if domain != "general":
		for named in general_named:
			name_entities.pop(named, None)
json.dump(ontology, open(f"ontology.json", "w"), indent=4)
