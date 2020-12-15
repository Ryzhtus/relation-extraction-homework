import torch
import torch.nn as nn
import tqdm

from relation_extraction.metrics import calculate_score

def train_epoch(model, criterion, optimizer, data, indexer, device):
    epoch_loss = 0
    epoch_score = 0

    model.train()

    for batch in data:
        tokens = batch[0].to(device)
        tags = batch[1].to(device)

        predictions = model(tokens)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags_mask = tags != indexer['O']
        tags_mask = tags_mask.view(-1)
        labels = torch.where(tags_mask, tags.view(-1), torch.tensor(criterion.ignore_index).type_as(tags))

        loss = criterion(predictions, labels)

        predictions = predictions.argmax(dim=1, keepdim=True)

        f_score = calculate_score(predictions, labels)

        epoch_loss += loss.item()
        epoch_score += f_score

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        torch.cuda.empty_cache()

    print('Train Average Loss = {:.5f}, F1-score = {:.3%}'.format(epoch_loss / len(data), epoch_score / len(data)))


def eval_epoch(model, criterion, data, indexer, device):
    epoch_loss = 0
    epoch_score = 0

    model.eval()

    with torch.no_grad():
        for batch in data:
            tokens = batch[1].to(device)
            tags = batch[3].to(device)

            predictions = model(tokens)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags_mask = tags != indexer['O']
            tags_mask = tags_mask.view(-1)
            labels = torch.where(tags_mask, tags.view(-1), torch.tensor(criterion.ignore_index).type_as(tags))

            loss = criterion(predictions, labels)

            predictions = predictions.argmax(dim=1, keepdim=True)

            f_score = calculate_score(predictions, labels)

            epoch_loss += loss.item()
            epoch_score += f_score

    print('Eval Average Loss = {:.5f}, F1-score = {:.3%}'.format(epoch_loss / len(data), epoch_score / len(data)))


def train_model(model, criterion, optimizer, train_data, test_data, indexer, device, epochs=1):
    for epoch in range(epochs):
        print('Epoch {} / {}'.format(epoch + 1, epochs))
        train_epoch(model, criterion, optimizer, train_data, indexer, device)
        eval_epoch(model, criterion, test_data, indexer, device)