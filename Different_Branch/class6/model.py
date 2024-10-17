import torch
import torch.nn as nn
import torch.nn.functional as F
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


################  自注意力模块  ##################
class SelfAttention_Module(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.2):
        super(SelfAttention_Module, self).__init__()
        self.multihead_attenion = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.multihead_attenion(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x


###################  交叉注意力  ######################
class Cross_Attention_Module(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.2):
        super(Cross_Attention_Module, self).__init__()
        self.multihead_attenion = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # self.norm1 = nn.LayerNorm(embed_dim)
        # self.ffn = nn.Sequential(
        #     nn.Linear(embed_dim, ff_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(ff_hidden_dim, embed_dim)
        # )
        # self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, thesis_feature, LLM_analysis_feature):
        attn_output, _ = self.multihead_attenion(thesis_feature, LLM_analysis_feature, LLM_analysis_feature)
        out = self.dropout(attn_output)
        # out = self.norm1(out)
        return out





# def Dissimilarity_Attention_Module(thesis_feature, cross_analysis_feature):
#     Q = thesis_feature
#     K, V = cross_analysis_feature, cross_analysis_feature
#
#     # 获取Key的维度
#     Q_norm = F.normalize(Q, p=2, dim=1)
#     K_norm = F.normalize(K, p=2, dim=1)
#
#     # 计算缩放后的点积
#     scores = Q_norm * K_norm
#
#     # 计算不相似度权重
#     # dissimilarity_weights = F.softmax(1 - scores, dim=-1)
#     dissimilarity_weights = torch.exp(-1.0 * scores)
#     # print(f"dissimilarity weights: {dissimilarity_weights}")
#
#     # 打印 dissimilarity_weights 的维度
#     # print(f"dissimilarity_weights shape: {dissimilarity_weights.shape}")
#
#     # 使用不相似度权重对值向量 V 进行加权求和
#     output = torch.mul(dissimilarity_weights, V)
#
#     # 打印 output 的维度
#     # print(f"output shape: {output.shape}")
#
#     return output



#####################  LLM_Semantics_Analysis Branch  ###############################
class LLM_Semantics_Analysis(nn.Module):
    def __init__(self, T5_Path):
        super(LLM_Semantics_Analysis, self).__init__()

        ########### 利用SCIBert对论文编码
        self.T5_thesis = T5EncoderModel.from_pretrained(T5_Path)
        for name, param in self.T5_thesis.named_parameters():
            if name.startswith("encoder.block.10"):  # 修改为 block.10 来访问第 11 层
                param.requires_grad = True  # 确保只有第 11 层的参数可以在训练过程中更新。
            else:
                param.requires_grad = False  # 冻结其他层的参数

        # 加载 T5 预训练模型，用于 LLM 分析
        self.T5_LLM_analysis = T5EncoderModel.from_pretrained(T5_Path)
        for name, param in self.T5_LLM_analysis.named_parameters():
            if name.startswith("encoder.block.10"):  # 修改为 block.10 来访问第 11 层
                param.requires_grad = True  # 确保只有第 11 层的参数可以在训练过程中更新。
            else:
                param.requires_grad = False  # 冻结其他层的参数


        ##########  可学习的矩阵参数
        self.learnable_matrix = nn.Parameter(torch.rand(1, 768))

        ##########  计算LLM分析的有效性
        self.LLM_analysis_importance = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),  # 对 256 维度的输出进行批归一化
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 3),
            nn.Sigmoid()
            )

        ##########  大模型分析文本的自注意力模块
        self.LLM_analysis_selfattention = SelfAttention_Module(768, 8, 256)

        ##########  提取大模型分析中与论文中不重复的特征
        # self.Cross_Attention = Cross_Attention_Module(768, 8, 256)
        self.Dissimilarity_Attention = Cross_Attention_Module(768, 8, 256)
        ##########  去除交叉特征中的冗余特征
        # self.Dissimilarity_Attention = Dissimilarity_Attention_Module


    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):

        thesis_feature = self.T5_thesis(input_ids_1, attention_mask_1)[0][:, 0, :]

        LLM_analysis_feature = self.T5_LLM_analysis(input_ids_2, attention_mask_2)[0][:, 0, :]


        batch_size = LLM_analysis_feature.size(0)
        learnable_matrix_expanded = self.learnable_matrix.expand(batch_size, -1)
        LLM_analysis_feature = learnable_matrix_expanded * LLM_analysis_feature


        LLM_analysis_prediction_result = self.LLM_analysis_importance(LLM_analysis_feature)


        LLM_analysis_feature = self.LLM_analysis_selfattention(LLM_analysis_feature)

        output = self.Dissimilarity_Attention(thesis_feature, LLM_analysis_feature)

        # output = self.Dissimilarity_Attention(thesis_feature, LLM_analysis_feature)

        # cross_analysis_feature = self.Cross_Attention(thesis_feature, LLM_analysis_feature)
        #
        # output = self.Dissimilarity_Attention(thesis_feature, cross_analysis_feature)

        return output, LLM_analysis_prediction_result


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


        ##########  第二个分支
        self.Second_Level = LLM_Semantics_Analysis(T5_Path)


        ##########  特征对齐
        self.align_x1 = nn.Sequential(
            nn.Linear(in_features=768, out_features=128),
            nn.BatchNorm1d(128)
        )
        self.align_x2 = nn.Sequential(
            nn.Linear(in_features=768, out_features=128),
            nn.BatchNorm1d(128)
        )


        ##########  融合特征及分类
        self.weight1 = nn.Parameter(torch.randn(1, 128))
        self.weight2 = nn.Parameter(torch.randn(1, 128))

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

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):

        x1 = self.First_Level(input_ids_1, attention_mask_1)
        x1 = self.align_x1(x1)

        x2, second_prediction_result = self.Second_Level(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        x2 = self.align_x2(x2)


        # 加权融合
        batch_size1 = x1.size(0)
        weight1_expanded = self.weight1.expand(batch_size1, -1)
        x1_weight = weight1_expanded * x1

        batch_size2 = x2.size(0)
        weight2_expanded = self.weight2.expand(batch_size2, -1)
        x2_weight = weight2_expanded * x2


        x_fusion = torch.cat((x1_weight, x2_weight), dim=-1)
        x_fusion = self.batch_norm_fusion(x_fusion)

        # 预测
        output = self.classifier(x_fusion)

        return output, second_prediction_result
