from torch.utils.data import Dataset, DataLoader
import json

class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.add_tokens(['<H>', '<R>', '<T>'])    # try with special tokens later
        self.data_path = data_path
        self.data = []
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            for entry in data['entries']:
                for value in entry.values():
                    triple_set = []
                    for triple in value['modifiedtripleset']:
                        triple_set.append('<H>')
                        triple_set.append(triple['subject'])
                        triple_set.append('<R>')
                        triple_set.append(triple['property'])
                        triple_set.append('<T>')
                        triple_set.append(triple['object'])
                    for lex in value['lexicalisations']:
                        self.data.append((' '.join(triple_set), lex['lex']))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        triple_set = self.data[idx][0]
        lex = self.data[idx][1]
        source = self.tokenizer(triple_set, return_tensors='pt', padding='max_length', truncation=True)
        target = self.tokenizer(lex, return_tensors='pt', padding='max_length', truncation=True)

        return {'source_input_ids': source['input_ids'].squeeze(), 'source_attention_mask': source['attention_mask'].squeeze(), 'target_input_ids': target['input_ids'].squeeze(), 'target_attention_mask': target['attention_mask'].squeeze()}

def get_dataloader(data_path, tokenizer, batch_size, shuffle=True):
    dataset = MyDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader