import torch
import torch.nn as nn
from transformers import T5EncoderModel


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
        self.dropout = nn.Dropout(dropout)

    def forward(self, thesis_feature, LLM_analysis_feature):
        attn_output, _ = self.multihead_attenion(thesis_feature, LLM_analysis_feature, LLM_analysis_feature)
        out = self.dropout(attn_output)

        return out


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

        ########### 利用SCIBert对大模型分析进行编码
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

        ##########  第二个分支
        self.Second_Level = LLM_Semantics_Analysis(T5_Path)

        ##########  特征对齐

        self.align_x2 = nn.Sequential(
            nn.Linear(in_features=768, out_features=128),
            nn.BatchNorm1d(128)
        )

        ##########  融合特征及分类
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # 添加 Dropout
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, input_ids_1, attention_mask_1,
                input_ids_2, attention_mask_2):

        x2, second_prediction_result = self.Second_Level(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        x2 = self.align_x2(x2)

        # 预测
        output = self.classifier(x2)

        return output, second_prediction_result