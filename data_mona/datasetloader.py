import json
from datasets import Dataset, DatasetDict

class ConversationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['conversation'][0]['prompt']
        answer = item['conversation'][0]['answer']

        return prompt, answer

data = json.load(open('self_data.json')) 

# Processing of data to keep previous context in responses
expanded_data = []
id = 0
for entry in data:
    conv = entry['conversation']
    ctx = ''
    for pair in conv:
        expanded_data.append({
            "id": id,
            "conversation": {
                'prompt': ctx + pair['prompt'],
                'answer': pair['answer']
            }
        })
        id += 1
        ctx += (pair['prompt'] + '\n' + pair['answer'] + '\n') 

with open('train.json', 'w') as f:
    json.dump(expanded_data, f, indent=4)

