import json
import requests
from tqdm import tqdm
from multiprocessing import Pool
import os


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


def process_data(data):
    AI_text = data["AI_text"]

    prompt1 = f"""
    Please rewrite the essay and imitate its word using habits: {AI_text}. Try to be different from the original text.
    
    Finally, you only need to output the generated text in English.
    """

    Rewrite_text = LLM_Generated_text(prompt1).replace('\n', ' ').replace('  ', ' ')
    data["Rewrite_text"] = Rewrite_text


    prompt2 = f"""
    Please rewrite the following academic text to make its style more natural and closely resemble human academic writing,\ 
    avoiding detectable AI-generated patterns. Pay attention to using diverse sentence structures,\ 
    common academic vocabulary, and natural expressions.

    {AI_text}

    Finally, you only need to output the generated text in English.
    """

    Humanlike_text = LLM_Generated_text(prompt2).replace('\n', ' ').replace('  ', ' ')
    data["Humanlike_text"] = Humanlike_text

    return data


def process_line_with_retry(line, max_attempts=3):

    for attempt in range(1, max_attempts + 1):
        try:
            return process_data(line)
        except Exception as e:
            print(f"处理失败，尝试次数 {attempt}/{max_attempts}: {e}")
            if attempt == max_attempts:
                # 达到最大尝试次数，可以选择返回一个特定的错误标记，或者抛出异常
                return None  # 或者 raise


def save_data(data, file_path):
    """
    将处理后的数据立即保存到文件中。
    """
    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")



if __name__ == "__main__":

    processes = 28
    p = Pool(processes=processes)

    file_list = ["train.json", "val.json"]

    for file_path in file_list:

        input_file_path = f"/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/HPPT/{file_path}"
        output_file_path = f"/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/HPPT/GLM4/{file_path}"

        # 如果输出文件不存在，创建并处理数据
        if not os.path.exists(output_file_path):
            print(f"Creating {output_file_path}")

            # 读取输入数据
            with open(input_file_path, 'r', encoding='utf-8') as file:
                data_list = [json.loads(line) for line in file]

            # 使用 imap_unordered 获取迭代器，允许在任务完成时立即处理结果
            with tqdm(total=len(data_list), desc="Processing Data") as progress_bar:
                for result in p.imap_unordered(process_line_with_retry, data_list):
                    if result is not None:
                        save_data(result, output_file_path)
                        progress_bar.update(1)  # 更新进度条

        else:
            print(f"Loading {output_file_path} to check missing items")
            existing_ids = set()
            missing_items = []

            # 读取已有的文件，并记录已有的 ID
            with open(output_file_path, 'r', encoding='utf-8') as existing_file:
                for line in existing_file:
                    data = json.loads(line)
                    existing_ids.add(data['ID'])

            # 读取源数据文件，查找缺失的条目
            with open(input_file_path, 'r', encoding='utf-8') as original_file:
                data_list = [json.loads(line) for line in original_file]


            # 识别缺失项
            for item in data_list:
                if item['ID'] not in existing_ids:
                    missing_items.append(item)


            # 处理缺失的条目
            with tqdm(total=len(missing_items), desc="Processing Missing Items") as progress_bar:
                for result in p.imap_unordered(process_line_with_retry, missing_items):
                    if result is not None:
                        save_data(result, output_file_path)
                        progress_bar.update(1)


    p.close()
    p.join()