import torch
import torch.nn as nn
from transformers import T5EncoderModel

class T5_Encoder(nn.Module):
    def __init__(self, T5_Path):
        super(T5_Encoder, self).__init__()
        # 加载预训练的 T5 编码器模型
        self.model = T5EncoderModel.from_pretrained(T5_Path)

    def forward(self, input_ids, attention_mask):
        # 获取 T5 编码器的输出 (batch_size, seq_length, hidden_size)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 将 attention_mask 扩展成与输出相同的维度 (batch_size, seq_length, hidden_size)
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()

        # 通过将 outputs 和 attention_mask 相乘，只保留有效 token 的表示
        sum_embeddings = torch.sum(outputs * attention_mask_expanded, dim=1)

        # 对 attention_mask 进行求和，避免除以0，保持数值稳定
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)

        # 平均池化，得到句子的表示 (batch_size, hidden_size)
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings
