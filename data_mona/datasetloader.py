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
dataset = Dataset.from_dict(data)
# dataset = ConversationDataset(data)

# Example: Accessing a single data point
prompt, answer = dataset[593]
print("Prompt:", prompt)
print("Answer:", answer)