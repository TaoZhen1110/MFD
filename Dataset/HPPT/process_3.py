import json
import requests

with open("/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/HPPT/Origin_data/train_1.json") as f:
    data = json.load(f)

dict_data = {item['id']: item['text'] for item in data}


def LLM_Generated_text(user_message, timeout=None):
    api_url = "http://172.27.33.101:8024/v1/chat/completions"
    model_path = "/mnt/data102_d2/huggingface/models/glm-4-9b-chat-1m/"
    temperature = 0.2

    payload = {
        "model": model_path,
        "messages": [
            {
                "role": "user",
                "content": user_message
            },
        ],
        "temperature": temperature,
    }

    try:
        response = requests.post(api_url, data=json.dumps(payload), timeout=timeout)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        response_data = response.json()

        if 'choices' in response_data and len(response_data['choices']) > 0:
            return response_data['choices'][0]['message']['content']
        else:
            return "No story generated. Check the API response for more details."
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
    except json.JSONDecodeError:
        return "Failed to decode the JSON response from the API."
    except KeyError as e:
        return f"Missing key in the API response: {e}"


def insert_new_entries(data, num_groups):
    new_entries = []
    for group in range(1, num_groups+1):
        expand_group = []
        for index in range(1, 5):
            entry_id = f"{group}_{index}"

            if index == 3:
                thesis = dict_data[f"{group}_{index - 1}"]
                prompt1 = f"""
                Please rewrite the essay and imitate its word using habits: {thesis}. Try to be different from the original text.

                Finally, you only need to output the generated text in English."""
                LLM_rewrite_text = LLM_Generated_text(prompt1).replace('\n', '')
                new_entry1 = {
                    "id": entry_id,
                    "text": LLM_rewrite_text,
                    "label": 1
                }
                expand_group.append(new_entry1)

            if index == 4:
                thesis = dict_data[f"{group}_{index - 2}"]
                prompt2 = f"""
                Please rewrite the following academic text to make its style more natural and closely resemble human academic writing, 
                avoiding detectable AI-generated patterns. Pay attention to using diverse sentence structures, 
                common academic vocabulary, and natural expressions.
                
                {thesis}
                
                Finally, you only need to output the generated text in English.
                """
                LLM_humanlike_text = LLM_Generated_text(prompt2).replace('\n', '')
                new_entry2 = {
                    "id": entry_id,
                    "text": LLM_humanlike_text,
                    "label": 1
                }
                expand_group.append(new_entry2)

        new_entries.append(expand_group)


    # 在每组末尾插入新条目
    for new_entry1 in new_entries:
        group_number = int(new_entry1[0]['id'].split('_')[0])
        insert_position = group_number * 3 - 1  # 每组三个条目，新数据在每组末尾
        data.insert(insert_position, new_entry1[0])

    for new_entry1 in new_entries:
        group_number = int(new_entry1[1]['id'].split('_')[0])
        insert_position = group_number * 4 - 1  # 每组三个条目，新数据在每组末尾
        data.insert(insert_position, new_entry1[1])


num_groups = len(data) // 2
insert_new_entries(data, num_groups)


with open('/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/HPPT/GLM4/train.json', 'a', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)





