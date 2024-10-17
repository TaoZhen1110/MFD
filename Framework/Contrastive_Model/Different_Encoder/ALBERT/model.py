import torch
import torch.nn as nn
from transformers import AlbertModel

class ALBERT_Encoder(nn.Module):
    def __init__(self, ALBERT_Path):
        super(ALBERT_Encoder, self).__init__()
        # 加载预训练的 ALBERT 模型
        self.model = AlbertModel.from_pretrained(ALBERT_Path)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 获取 ALBERT 模型的输出 (batch_size, seq_length, hidden_size)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state  # 提取 hidden states (batch_size, seq_length, hidden_size)

        # 将 attention_mask 扩展成与输出相同的维度 (batch_size, seq_length, hidden_size)
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        # 通过将 last_hidden_state 和 attention_mask 相乘，只保留有效 token 的表示
        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)

        # 对 attention_mask 进行求和，避免除以0，保持数值稳定
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)

        # 平均池化，得到句子的表示 (batch_size, hidden_size)
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings
