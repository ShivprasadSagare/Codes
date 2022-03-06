import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM
import torch
from sacrebleu import BLEU


class FineTuner(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.model_name_or_path
        )
        self.cal_bleu = BLEU()

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def _step(self, batch):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs[0]
        return loss

    def _generative_step(self, batch):
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            use_cache=True,
            num_beams=self.hparams.eval_beams,
            max_length=self.hparams.tgt_max_seq_len
            # understand above 3 arguments
            )

        input_text = self.hparams.tokenizer.batch_decode(
            batch['input_ids'],
            skip_special_tokens=True)
        pred_text = self.hparams.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        batch['labels'][batch['labels'] == -100] = self.hparams.tokenizer.pad_token_id
        ref_text = self.hparams.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        return input_text, pred_text, ref_text

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

        input_text, pred_text, ref_text = self._generative_step(batch)
        return {'val_loss': loss, 'input_text': input_text, 'pred_text': pred_text, 'ref_text': ref_text}

    def validation_epoch_end(self, outputs):
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # self.log("epoch_val_loss", avg_loss)

        input_text = []
        pred_text = []
        ref_text = []
        for x in outputs:
            input_text.extend(x['input_text'])
            pred_text.extend(x['pred_text'])
            ref_text.extend(x['ref_text'])
        bleu = self.cal_bleu.corpus_score(pred_text, [ref_text])    
        self.log("val_bleu", bleu.score)

        random_indices = set([len(input_text)//i for i in range(2, 7)])
        epoch_list = [self.trainer.current_epoch for i in range(len(random_indices))]

        input_text = [input_text[i] for i in random_indices]
        pred_text = [pred_text[i] for i in random_indices]
        ref_text = [ref_text[i] for i in random_indices]

        data = [i for i in zip(epoch_list, input_text, ref_text, pred_text)]
        self.trainer.logger.log_text(key='validation_predictions', data=data, columns=['epoch', 'input_text', 'ref_text', 'pred_text'])

    def test_step(self, batch, batch_idx):
        input_text, pred_text, ref_text = self._generative_step(batch)
        return {'input_text': input_text, 'pred_text': pred_text, 'ref_text': ref_text}

    def test_epoch_end(self, outputs):
        pred_text = []
        ref_text = []
        for x in outputs:
            pred_text.extend(x['pred_text'])
            ref_text.extend(x['ref_text'])
        bleu = self.cal_bleu.corpus_score(pred_text, [ref_text])
        self.log("test_bleu", bleu.score)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Model args')
        parser.add_argument('--learning_rate', default=2e-5, type=float)
        parser.add_argument('--model_name_or_path', default='t5-base', type=str)
        parser.add_argument('--eval_beams', default=4, type=int)
        parser.add_argument('--tgt_max_seq_len', default=128, type=int)
        return parent_parser