from model.model import FineTuner 
from model.dataloader import DataModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import os
from datetime import datetime

def init_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser = DataModule.add_datamodule_specific_args(parser)
    parser = FineTuner.add_model_specific_args(parser)
    # add miscellaneous arguments below
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--strategy', type=str, default='ddp')
    parser.add_argument('--log_dir', type=str, default='/scratch/shivprasad.sagare/experiments')
    parser.add_argument('--project_name', type=str, default='finetuner')
    parser.add_argument('--run_name', type=str, default='run_1')
    return parser

def main():
    parser = init_args()
    args = parser.parse_args()
    args = vars(args)

    dm = DataModule(**args)

    args.update({'tokenizer': dm.tokenizer})
    model = FineTuner(**args)

    os.makedirs(args['log_dir'], exist_ok=True)

    now = datetime.now()

    logger = WandbLogger(
        name=args['run_name'] + '_' + now.strftime("%m/%d-%H%M"),
        project=args['project_name'],
        save_dir=args['log_dir'], 
        log_model=True
    )
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')
    trainer = pl.Trainer(
        gpus=args['gpus'], 
        max_epochs=args['max_epochs'], 
        strategy=args['strategy'], 
        logger=logger, 
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, dm)
    trainer.test(datamodule=dm)

if __name__ == '__main__':
    main()

