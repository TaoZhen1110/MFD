from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os

class data_encoder(Dataset):
    def __init__(self, jsondata, tokenizer):
        super(data_encoder, self).__init__()
        self.tokenizer = tokenizer
        self.jsondata = [item for item in jsondata]

    def __len__(self):
        return len(self.jsondata) // 4

    def __getitem__(self, idx):
        start_idx = idx * 4

        # 处理每组的四个数据项
        input_ids_list = []
        attention_mask_list = []

        for i in range(4):
            enconder_input = self.tokenizer(
                self.jsondata[start_idx + i]["text"],
                max_length=512,  # 假设模型的最大输入长度为512
                padding="max_length",
                truncation=True,
                return_tensors="pt"  # 返回PyTorch格式的张量
            )

            input_ids_list.append(enconder_input.input_ids.squeeze())
            attention_mask_list.append(enconder_input.attention_mask.squeeze())

        return (
            torch.stack(input_ids_list),
            torch.stack(attention_mask_list)
        )



############  封装预处理后的数据集
def dataset_loader(dataset, batch_size, shuffle=False):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=nw)

    return data_loader

