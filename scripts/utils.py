import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_data(tokenizer, texts, labels, max_len):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

def create_data_loader(inputs, masks, labels, batch_size, sampler):
    data = torch.utils.data.TensorDataset(inputs, masks, labels)
    data_sampler = sampler(data)
    data_loader = DataLoader(data, sampler=data_sampler, batch_size=batch_size)
    return data_loader
