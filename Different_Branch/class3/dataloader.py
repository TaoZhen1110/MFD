from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os

class Data_Loader(Dataset):
    def __init__(self, jsondata, tokenizer):
        super(Data_Loader, self).__init__()
        self.tokenizer = tokenizer
        self.sentences = []  # 存储所有的句子
        self.labels = []  # 存储句子对应的标签

        for data in jsondata:
            origin_text_sentences = data["text"].split('</s>')
            labels = data["label"]

            for i, sentence in enumerate(origin_text_sentences):
                sentence = sentence.strip()
                if sentence:
                    self.sentences.append(sentence)
                    self.labels.append(torch.tensor(labels[i], dtype=torch.float32))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoded_input_1 = self.tokenizer(
            sentence,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=32
        )
        input_ids_1 = encoded_input_1.input_ids.squeeze()
        attention_mask_1 = encoded_input_1.attention_mask.squeeze()

        label = self.labels[idx]

        return {
            "input_ids_1": input_ids_1,
            "attention_mask_1": attention_mask_1,
            "Label": label
        }


def dataset_loader(dataset, batch_size, shuffle=False):
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader