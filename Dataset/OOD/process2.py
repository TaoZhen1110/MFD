import json
import numpy as np


# 读取每行JSON文件的函数
def load_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# 提取 "Third_Level_Features" 数据的函数
def extract_all_features(data):
    all_features = []
    for item in data:
        if "Third_Level_Features" in item:
            all_features.extend(item["Third_Level_Features"])  # 提取所有子列表
    return all_features


# 进行归一化的函数 (Min-Max 归一化)
def normalize_features(features):
    features_array = np.array(features)  # 转换为numpy数组
    min_vals = features_array.min(axis=0)
    max_vals = features_array.max(axis=0)

    # 防止除以零的情况
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # 如果范围为0，设为1，避免除以0

    normalized = (features_array - min_vals) / range_vals
    return normalized.tolist()  # 转回列表格式


# 将归一化后的结果分配回原始数据的函数
def assign_normalized_features(data, normalized_features):
    index = 0
    for item in data:
        if "Third_Level_Features" in item:
            num_features = len(item["Third_Level_Features"])
            item["Third_Level_Features"] = normalized_features[index:index + num_features]
            index += num_features


# 主程序：处理并保存多个文件的数据
def process_and_save_global_normalized_data(input_files, output_files):
    # 1. 读取所有输入文件的数据
    all_data = []
    for input_file in input_files:
        data = load_json_file(input_file)
        all_data.append(data)

    # 2. 提取所有JSON对象中的 "Third_Level_Features" 子列表
    all_features = []
    for data in all_data:
        all_features.extend(extract_all_features(data))

    # 3. 对所有子列表进行全局的归一化
    normalized_features = normalize_features(all_features)

    # 4. 将归一化后的数据分配回原始数据
    index = 0
    for data in all_data:
        num_features = len(extract_all_features(data))
        assign_normalized_features(data, normalized_features[index:index + num_features])
        index += num_features

    # 5. 分别保存处理后的数据到输出文件
    for i, data in enumerate(all_data):
        output_file = output_files[i]
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


# 使用时调用的代码
input_files = [
    "/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/OOD/test1.json"
]


output_files = [
    "/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/OOD/test2.json"
]

process_and_save_global_normalized_data(input_files, output_files)

