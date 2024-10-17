import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


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

###################  不相似度注意力  ######################
def Dissimilarity_Attention_Module(thesis_feature, LLM_analysis_feature):
    Q = thesis_feature
    K, V = LLM_analysis_feature, LLM_analysis_feature

    # # 打印输入维度
    # print(f"Q (thesis_feature) shape: {Q.shape}")
    # print(f"K (LLM_analysis_feature[0]) shape: {K.shape}")
    # print(f"V (LLM_analysis_feature[1]) shape: {V.shape}")

    # 获取Key的维度
    d_k = K.size(-1)
    Q_norm = F.normalize(Q, p=2, dim=1)
    K_norm = F.normalize(K, p=2, dim=1)

    # 计算缩放后的点积
    scores = Q_norm * K_norm

    # 打印 scores 的维度
    # print(f"scores shape: {scores.shape}")

    # 计算不相似度权重
    # dissimilarity_weights = F.softmax(1 - scores, dim=-1)
    dissimilarity_weights = torch.exp(-1.0 * scores)
    # print(f"dissimilarity weights: {dissimilarity_weights}")

    # 打印 dissimilarity_weights 的维度
    # print(f"dissimilarity_weights shape: {dissimilarity_weights.shape}")

    # 使用不相似度权重对值向量 V 进行加权求和
    output = torch.mul(dissimilarity_weights, V)

    # 打印 output 的维度
    # print(f"output shape: {output.shape}")

    return output


#####################  LLM_Semantics_Analysis Branch  ###############################
class LLM_Semantics_Analysis(nn.Module):
    def __init__(self, SCIBERT_Path):
        super(LLM_Semantics_Analysis, self).__init__()

        ########### 利用SCIBert对论文编码
        self.SCIBERT_thesis = AutoModel.from_pretrained(SCIBERT_Path).requires_grad_(False)
        for name, param in self.SCIBERT_thesis.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True  # 确保只有第 11 层的参数可以在训练过程中更新。
            else:
                param.requires_grad = False

        ########### 利用SCIBert对大模型分析进行编码
        self.SCIBERT_LLM_analysis = AutoModel.from_pretrained(SCIBERT_Path).requires_grad_(False)
        for name, param in self.SCIBERT_LLM_analysis.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True  # 确保只有第 11 层的参数可以在训练过程中更新。
            else:
                param.requires_grad = False


        ##########  可学习的矩阵参数
        self.learnable_matrix = nn.Parameter(torch.rand(1, 768))

        ##########  计算LLM分析的有效性
        self.LLM_analysis_importance = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),  # 对 256 维度的输出进行批归一化
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid()
            )

        ##########  大模型分析文本的自注意力模块
        self.LLM_analysis_selfattention = SelfAttention_Module(768, 8, 256)

        ##########  提取大模型分析中与论文中不重复的特征
        self.Dissimilarity_Attention = Dissimilarity_Attention_Module



    def forward(self, thesis_input_ids, thesis_token_type_ids, thesis_attention_mask,
        LLM_analysis_input_ids, LLM_analysis_token_type_ids, LLM_analysis_attention_mask):

        thesis_feature = self.SCIBERT_thesis(input_ids = thesis_input_ids, token_type_ids = thesis_token_type_ids,
                                             attention_mask = thesis_attention_mask)[0][:, 0, :]

        LLM_analysis_feature = self.SCIBERT_LLM_analysis(input_ids = LLM_analysis_input_ids,
                                                         token_type_ids = LLM_analysis_token_type_ids,
                                             attention_mask = LLM_analysis_attention_mask)[0][:, 0, :]


        LLM_analysis_feature = LLM_analysis_feature * self.learnable_matrix

        LLM_analysis_prediction_result = self.LLM_analysis_importance(LLM_analysis_feature)


        LLM_analysis_feature = self.LLM_analysis_selfattention(LLM_analysis_feature)

        output = self.Dissimilarity_Attention(thesis_feature, LLM_analysis_feature)

        return output, LLM_analysis_prediction_result











