from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os


class Data_Loader(Dataset):
    def __init__(self, jsondata, tokenizer):
        super(Data_Loader, self).__init__()
        self.tokenizer = tokenizer
        self.sentences = []  # 存储所有的句子
        self.third_level_features = []  # 存储句子对应的特征
        self.labels = []  # 存储句子对应的标签

        for data in jsondata:
            origin_text_sentences = data["text"].split('</s>')
            third_level_features = data["Third_Level_Features"]
            labels = data["label"]

            for i, sentence in enumerate(origin_text_sentences):
                sentence = sentence.strip()
                if sentence:
                    self.sentences.append(sentence)
                    self.third_level_features.append(torch.tensor(third_level_features[i], dtype=torch.float32))
                    self.labels.append(torch.tensor(labels[i], dtype=torch.float32))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):

        # 获取对应的特征和标签
        third_level_feature = self.third_level_features[idx]

        label = self.labels[idx]

        return {
            "Third_level_feature": third_level_feature,
            "Label": label,
        }


def dataset_loader(dataset, batch_size, shuffle=False):
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader