from logging import getLogger
import wandb
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from model.dataloader import get_dataloader
import argparse
import os
import json
from evaluate import evaluate
from utils import get_logger, save_checkpoint

def train_one_epoch(model, train_dataloader, tokenizer, optimizer, config, device, logger, epoch):
    model.train()
    total_loss = 0
    for id, batch in enumerate(train_dataloader):
        source_input_ids = batch['source_input_ids'].to(device)
        source_attention_mask = batch['source_attention_mask'].to(device)
        labels = batch['target_input_ids'].to(device)
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(source_input_ids, attention_mask=source_attention_mask, labels=labels)
        loss = outputs[0]
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if id % 100 == 0:
            wandb.log({'train_loss': loss.item()})


def train_and_validate(model, train_dataloader, val_dataloader, tokenizer, optimizer, config, device, logger):
    config.best_val_loss = float('inf')
    logger.info('Start training')
    for epoch in range(config.epochs):
        train_one_epoch(model, train_dataloader, tokenizer, optimizer, config, device, logger, epoch)
        state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        save_checkpoint(state, os.path.join(root_dir, config.model_dir), 'last.pth.tar')

        val_loss = evaluate(model, val_dataloader, tokenizer, device)
        logger.info(f'epoch: {epoch}, val_loss: {val_loss}')
        wandb.log({'val_loss': val_loss})
        if val_loss < config.best_val_loss:
            save_checkpoint(state, os.path.join(root_dir, config.model_dir), 'best.pth.tar')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='model directory', required=True)
    parser.add_argument('--data_dir', type=str, help='data directory', required=True)
    args = parser.parse_args()

    wandb.init()
    config = wandb.config
    config.update(args)

    root_dir = os.path.dirname(os.path.abspath(__file__))

    logger = get_logger(os.path.join(root_dir, config.model_dir, 'train.log'))
    
    json_path = os.path.join(root_dir, config.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    with open(json_path, 'r') as f:
        config.update(json.load(f))
    
    logger.info('Done with reading the hyperparameters and command line arguments')
    for k, v in config.items():
        logger.info(f'{k} = {v}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device: {device}')

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    logger.info(f'Tokenizer: {tokenizer.__class__.__name__} is loaded')
    
    train_data_path = os.path.join(root_dir, config.data_dir, 'train', 'train.json')
    train_dataloader = get_dataloader(train_data_path, tokenizer, batch_size=config.train_batch_size, shuffle=True)
    logger.info(f'Train dataloader is loaded')

    val_data_path = os.path.join(root_dir, config.data_dir, 'dev', 'dev.json')
    val_dataloader = get_dataloader(val_data_path, tokenizer, batch_size=config.val_batch_size, shuffle=True)
    logger.info(f'Val dataloader is loaded')

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.to(device)
    logger.info(f'Model: {model.__class__.__name__} is loaded')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    wandb.watch(model, log='all')

    train_and_validate(model, train_dataloader, val_dataloader, tokenizer, optimizer, config, device, logger)
    

