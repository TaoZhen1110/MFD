import torch
import torch.nn as nn
from transformers import T5EncoderModel


################  加载第一个分支模型  ##################
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



def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict



#######################  总模型参数  ###############################
class Multi_Level_Framework(nn.Module):
    def __init__(self, T5_Path, Thesis_T5_Encoder_Path):
        super(Multi_Level_Framework, self).__init__()

        ##########  第一个分支
        self.First_Level = T5_Encoder(T5_Path=T5_Path)

        state_dict = torch.load(Thesis_T5_Encoder_Path)
        state_dict = remove_module_prefix(state_dict)
        self.First_Level.load_state_dict(state_dict)

        # 冻结所有参数
        for param in self.First_Level.parameters():
            param.requires_grad = False

        for i in range(-5, 0):
            for param in self.First_Level.model.encoder.block[i].parameters():  # 使用 block 而不是 layer
                param.requires_grad = True

        # 如果你还想更新池化层，可以添加以下代码
        # for param in self.First_Level.model.pooler.parameters():
        #     param.requires_grad = True


        ##########  特征对齐
        self.align_x1 = nn.Sequential(
            nn.Linear(in_features=768, out_features=128),
            nn.BatchNorm1d(128)
        )

        self.align_x3 = nn.Sequential(
            nn.Linear(in_features=283, out_features=128),
            nn.BatchNorm1d(128)
        )


        ##########  融合特征及分类
        self.weight1 = nn.Parameter(torch.randn(1, 128))
        self.weight3 = nn.Parameter(torch.randn(1, 128))

        self.batch_norm_fusion = nn.BatchNorm1d(256)


        ##########  融合特征及分类
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # 添加 Dropout
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, input_ids_1, attention_mask_1, third_level_feature):

        x1 = self.First_Level(input_ids_1, attention_mask_1)
        x1 = self.align_x1(x1)


        x3 = self.align_x3(third_level_feature)


        # 加权融合
        batch_size1 = x1.size(0)
        weight1_expanded = self.weight1.expand(batch_size1, -1)
        x1_weight = weight1_expanded * x1


        batch_size3 = x3.size(0)
        weight3_expanded = self.weight3.expand(batch_size3, -1)
        x3_weight = weight3_expanded * x3

        x_fusion = torch.cat((x1_weight, x3_weight), dim=-1)
        x_fusion = self.batch_norm_fusion(x_fusion)

        # 预测
        output = self.classifier(x_fusion)

        return output