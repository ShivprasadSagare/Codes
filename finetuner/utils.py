import pytorch_lightning as pl
import wandb
from typing import Optional, Any

class EarlyStopping(pl.Callback):
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        if trainer.current_epoch > 1:
            print("ending the batch ", trainer.current_epoch)
            return -1
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx, unused)

class LogPredictionSamples(pl.Callback):
    def __init__(self): 
        pass

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.input_text = []
        self.pred_text = []
        self.ref_text = []
        return super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.input_text.extend(outputs['input_text'])
        self.pred_text.extend(outputs['pred_text'])
        self.ref_text.extend(outputs['ref_text'])
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.random_indices = set([len(self.input_text)//i for i in range(2, 7)])
        self.epoch_list = [trainer.current_epoch for i in range(len(self.random_indices))]

        self.input_text = [self.input_text[i] for i in self.random_indices]
        self.pred_text = [self.pred_text[i] for i in self.random_indices]
        self.ref_text = [self.ref_text[i] for i in self.random_indices]

        data = [i for i in zip(self.epoch_list, self.input_text, self.ref_text, self.pred_text)]
        trainer.logger.log_text(key='validation_predictions', data=data, columns=['epoch', 'input_text', 'ref_text', 'pred_text'])
        return super().on_validation_epoch_end(trainer, pl_module)

