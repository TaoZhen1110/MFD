import torch
import torch.nn as nn


#######################  总模型参数  ###############################
class Multi_Level_Framework(nn.Module):
    def __init__(self, Roberta_Path, Thesis_Roberta_Encoder_Path):
        super(Multi_Level_Framework, self).__init__()

        ##########  特征对齐

        self.align_x3 = nn.Sequential(
            nn.Linear(in_features=283, out_features=128),
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

    def forward(self, third_level_feature):

        x3 = self.align_x3(third_level_feature)

        # 预测
        output = self.classifier(x3)

        return output