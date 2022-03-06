from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pandas as pd

class DS(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length):
        self.df = pd.read_csv(data_path)
        self.df = self.df[:len(self.df)]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_text = str(self.df.iloc[idx]['input_text'])
        target_text = self.df.iloc[idx]['output_text']

        input_encoding = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_source_length ,padding='max_length', truncation=True)
        target_encoding = self.tokenizer(target_text, return_tensors='pt', max_length=self.max_target_length ,padding='max_length', truncation=True)

        input_ids, attention_mask = input_encoding['input_ids'], input_encoding['attention_mask']
        labels = target_encoding['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100    # for ignoring the cross-entropy loss at padding locations

        return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'labels': labels.squeeze()}    


class DataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)

    def setup(self, stage=None):
        self.train = DS(self.hparams.train_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length)
        self.val = DS(self.hparams.val_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length)
        self.test = DS(self.hparams.test_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.train_batch_size, num_workers=8,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size, num_workers=8,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=8,shuffle=False)

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Datamodule args')
        parser.add_argument('--train_path', default='data/train.csv', type=str)
        parser.add_argument('--val_path', default='data/val.csv', type=str)
        parser.add_argument('--test_path', default='data/test.csv', type=str)
        parser.add_argument('--tokenizer_name_or_path', type=str)
        parser.add_argument('--max_source_length', type=int, default=128)
        parser.add_argument('--max_target_length', type=int, default=128)
        parser.add_argument('--train_batch_size', type=int, default=4)
        parser.add_argument('--val_batch_size', type=int, default=4)
        parser.add_argument('--test_batch_size', type=int, default=4)
        return parent_parser