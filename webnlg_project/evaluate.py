import torch

def evaluate(model, val_dataloader, tokenizer, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for id, batch in enumerate(val_dataloader):
            source_input_ids = batch['source_input_ids'].to(device)
            source_attention_mask = batch['source_attention_mask'].to(device)
            labels = batch['target_input_ids'].to(device)
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(source_input_ids, attention_mask=source_attention_mask, labels=labels)
            loss = outputs[0]
            total_loss += loss
    return total_loss / len(val_dataloader)

        