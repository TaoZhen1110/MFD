import torch
import torch.nn as nn
import argparse
import json
import os
from transformers import T5Tokenizer
import numpy as np
from Dataloader import Data_Loader, dataset_loader
from Model1 import Multi_Level_Framework
from sklearn.metrics import accuracy_score


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='3,7')
    parser.add_argument('-test_batch_size', type=int, default=64)
    parser.add_argument('-T5_Path', type=str,
                        default="/mnt/data132/taozhen/AI_Thesis_Detection/Framework/Contrastive_Model/Different_Encoder/T5/t5_pretrained/")
    parser.add_argument('-Thesis_T5_Encoder_Path', type=str,
                        default="/mnt/data132/taozhen/AI_Thesis_Detection/Framework/Contrastive_Model/Different_Encoder/T5/t5_save/run_0/contra_t5_model.pth")
    parser.add_argument("-model_path", type=str,
                        default='/mnt/data132/taozhen/AI_Thesis_Detection/Different_Encoder/T5/T5_save/run_0/t5_model.pth')
    parser.add_argument('-test_dataset_path', type=str,
                        default="/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/MOE/test3.json")

    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ###################  Dataset prepare #################
    tokenizer = T5Tokenizer.from_pretrained(args.T5_Path)

    test_data = []
    with open(args.test_dataset_path) as f:
        for line in f:
            test_data.append(json.loads(line))
    test_loader = dataset_loader(dataset=Data_Loader(jsondata=test_data, tokenizer=tokenizer),
                                batch_size=args.test_batch_size, shuffle=False)

    ################### 加载模型 #################
    model = Multi_Level_Framework(args.T5_Path, args.Thesis_T5_Encoder_Path)
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.model_path))

    ################### 计算测试指标 #################
    count = 0
    all_test_outputs = []
    all_test_labels = []
    model.eval()

    for step, sample_batched in enumerate(test_loader):
        sample_batched = {key: value.cuda() for key, value in sample_batched.items()}

        input_ids_1 = sample_batched['input_ids_1']
        attention_mask_1 = sample_batched['attention_mask_1']

        input_ids_2 = sample_batched['input_ids_2']
        attention_mask_2 = sample_batched['attention_mask_2']

        Third_level_feature = sample_batched["Third_level_feature"]

        Label = sample_batched['Label']

        with torch.no_grad():
            output, second_prediction_result = model(input_ids_1, attention_mask_1,
                                                     input_ids_2, attention_mask_2,
                                                     Third_level_feature)

        all_test_outputs.append(output.cpu().numpy())
        all_test_labels.append(Label.cpu().numpy())
        count += 1

    numpy_all_test_outputs = np.concatenate(all_test_outputs, axis=0)
    numpy_all_test_labels = np.concatenate(all_test_labels, axis=0)

    Lexical_outputs = numpy_all_test_outputs[:, 0]
    Lexical_labels = numpy_all_test_labels[:, 0]

    Syntax_outputs = numpy_all_test_outputs[:, 1]
    Syntax_labels = numpy_all_test_labels[:, 1]

    Semantic_outputs = numpy_all_test_outputs[:, 2]
    Semantic_labels = numpy_all_test_labels[:, 2]


    ########## 计算accuracy ##########
    # 定义不同的阈值
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # thresholds = [0.2]
    results = []

    for threshold in thresholds:
        # 计算每个阈值下的预测
        Lexical_predictions = (Lexical_outputs >= threshold).astype(int)
        Syntax_predictions = (Syntax_outputs >= threshold).astype(int)
        Semantic_predictions = (Semantic_outputs >= threshold).astype(int)

        Lexical_labels_binary = Lexical_labels  # 直接使用标签
        Syntax_labels_binary = Syntax_labels  # 直接使用标签
        Semantic_labels_binary = Semantic_labels  # 直接使用标签

        Lexical_accuracy = accuracy_score(Lexical_labels_binary, Lexical_predictions)
        Syntax_accuracy = accuracy_score(Syntax_labels_binary, Syntax_predictions)
        Semantic_accuracy = accuracy_score(Semantic_labels_binary, Semantic_predictions)

        mean_accuracy = (Lexical_accuracy + Syntax_accuracy + Semantic_accuracy) / 3

        result = {
            "Threshold": threshold,
            "Lexical_accuracy": Lexical_accuracy,
            "Syntax_accuracy": Syntax_accuracy,
            "Semantic_accuracy": Semantic_accuracy,
            "Mean_accuracy": mean_accuracy
        }

        results.append(result)

    # 输出所有结果
    for res in results:
        print(f"Threshold: {res['Threshold']}")
        print(f"Lexical Accuracy: {res['Lexical_accuracy']}")
        print(f"Syntax Accuracy: {res['Syntax_accuracy']}")
        print(f"Semantic Accuracy: {res['Semantic_accuracy']}")
        print(f"Mean Accuracy: {res['Mean_accuracy']}")
        print("\n")



if __name__ == '__main__':
    args = get_arguments()
    main(args)



