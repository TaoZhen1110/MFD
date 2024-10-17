import json

# 输入文件路径
train_file = '/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/HPPT/Origin_data/train_final.json'
test_file = '/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/HPPT/Origin_data/test_final.json'
output_file = '/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/HPPT/train.json'

# 存储所有数据的列表
combined_data = []

# 初始化ID计数器
current_id = 0

# 读取train.json文件并将数据追加到列表中，同时重新生成ID
with open(train_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        data["ID"] = current_id  # 重新设置ID
        combined_data.append(data)
        current_id += 1  # ID递增

# 读取test.json文件并将数据追加到列表中，同时重新生成ID
with open(test_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        data["ID"] = current_id  # 重新设置ID
        combined_data.append(data)
        current_id += 1  # ID递增

# 将所有数据写入到合并后的文件中
with open(output_file, 'w', encoding='utf-8') as f:
    for item in combined_data:
        json_line = json.dumps(item, ensure_ascii=False)  # 将字典转换为JSON字符串
        f.write(json_line + '\n')  # 写入文件，每行一个JSON对象

print(f"成功合并 {len(combined_data)} 条数据，并保存到 {output_file}")
