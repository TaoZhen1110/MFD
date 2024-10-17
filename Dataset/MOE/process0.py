import json


ID = 0
new_data = []
with open("/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/MOE/test.json", 'r') as f:
    for line in f:
        data = json.loads(line)
        data["ID"] = ID
        new_data.append(data)
        ID += 1


with open("/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/MOE/test0.json", 'w', encoding='utf-8') as f:
    for item in new_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")


