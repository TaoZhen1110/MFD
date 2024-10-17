import torch
import torch.nn as nn
from transformers import RobertaModel



class Thesis_Roberta_Encoder(nn.Module):
    def __init__(self, Roberta_Path):
        super(Thesis_Roberta_Encoder, self).__init__()
        self.model = RobertaModel.from_pretrained(Roberta_Path)


    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)[0][:, 0, :]

        return outputs








