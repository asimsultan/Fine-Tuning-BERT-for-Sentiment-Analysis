import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from utils import get_device, tokenize_data, create_data_loader

# Parameters
model_name = 'bert-base-uncased'
max_len = 256
batch_size = 16
epochs = 3
learning_rate = 2e-5

# Load Dataset
dataset = load_dataset('imdb')
train_texts, train_labels = dataset['train']['text'], dataset['train']['label']
test_texts, test_labels = dataset['test']['text'], dataset['test']['label']

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Tokenize Data
train_inputs, train_masks, train_labels = tokenize_data(tokenizer, train_texts, train_labels, max_len)
test_inputs, test_masks, test_labels = tokenize_data(tokenizer, test_texts, test_labels, max_len)

# DataLoader
train_loader = create_data_loader(train_inputs, train_masks, train_labels, batch_size, RandomSampler)
test_loader = create_data_loader(test_inputs, test_masks, test_labels, batch_size, SequentialSampler)

# Model
device = get_device()
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training Function
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    total_correct = 0

    for batch in data_loader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        optimizer.zero_grad()

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1)
        total_correct += torch.sum(preds == b_labels)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct.double() / len(data_loader.dataset)

    return avg_loss, accuracy

# Training Loop
for epoch in range(epochs):
    train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device, scheduler)
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Train Loss: {train_loss}, Train Accuracy: {train_accuracy}')

# Save Model
model_dir = './models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
