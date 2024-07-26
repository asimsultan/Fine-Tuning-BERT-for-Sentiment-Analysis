import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from utils import get_device, tokenize_data, create_data_loader

# Parameters
model_dir = './models'
batch_size = 16

# Load Model and Tokenizer
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

# Device
device = get_device()
model.to(device)

# Load Dataset
dataset = load_dataset('imdb')
test_texts, test_labels = dataset['test']['text'], dataset['test']['label']

# Tokenize Data
test_inputs, test_masks, test_labels = tokenize_data(tokenizer, test_texts, test_labels, max_len)

# DataLoader
test_loader = create_data_loader(test_inputs, test_masks, test_labels, batch_size, SequentialSampler)

# Evaluation Function
def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_preds = []

    with torch.no_grad():
        for batch in data_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            total_correct += torch.sum(preds == b_labels)
            total_preds.extend(preds.cpu().numpy())

    accuracy = total_correct.double() / len(data_loader.dataset)
    f1 = f1_score(test_labels, total_preds, average='weighted')

    return accuracy, f1

# Evaluate
test_accuracy, test_f1 = evaluate(model, test_loader, device)
print(f'Test Accuracy: {test_accuracy}')
print(f'Test F1 Score: {test_f1}')
