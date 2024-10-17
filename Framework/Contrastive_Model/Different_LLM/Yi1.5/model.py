import torch
import torch.nn as nn
from transformers import RobertaModel

class Yi_Roberta_Encoder(nn.Module):
    def __init__(self, Roberta_Path):
        super(Yi_Roberta_Encoder, self).__init__()
        self.model = RobertaModel.from_pretrained(Roberta_Path)

    def forward(self, input_ids, attention_mask):
        # 获取 RoBERTa 模型的输出 (batch_size, seq_length, hidden_size)
        outputs = self.model(input_ids, attention_mask=attention_mask)[0]

        # 将 attention_mask 扩展成与输出相同的维度 (batch_size, seq_length, hidden_size)
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()

        # 将输出和 attention_mask 相乘，然后对序列维度求和 (batch_size, hidden_size)
        sum_embeddings = torch.sum(outputs * attention_mask_expanded, 1)

        # 对 attention_mask 进行求和，确保不会除以0，保持数值稳定
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)

        # 平均池化 (batch_size, hidden_size)
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings

