import json

# 输入文件路径
input_file = '/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/HPPT/Origin_data/val.json'
output_file = '/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/HPPT/Origin_data/val_final.json'

# 存储处理后的数据
merged_data = []

# 读取原始文件并处理
with open(input_file, 'r', encoding='utf-8') as f:
    data_list = json.load(f)

id = 0
# 假设每两个为一组进行合并
for i in range(0, len(data_list), 2):
    if i + 1 < len(data_list):  # 确保成对存在
        data_1 = data_list[i]
        data_2 = data_list[i + 1]

        # 合并成新的结构
        merged_item = {
            "ID": id,
            "Human_text": data_1["text"],
            "AI_text": data_2["text"]
        }

        merged_data.append(merged_item)
        id += 1

# 将合并后的数据写入新的文件
with open(output_file, 'w', encoding='utf-8') as f:
    for item in merged_data:
        json_line = json.dumps(item, ensure_ascii=False)  # 将字典转换为JSON字符串
        f.write(json_line + '\n')  # 写入文件，并添加换行符

print(f"成功合并 {len(merged_data)} 组数据，并保存到 {output_file}")
