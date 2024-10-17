import json
import requests
from tqdm import tqdm
from multiprocessing import Pool
import os


def base_prompt_template() -> str:
    template = """<reserved_195>{query}<reserved_196>"""
    return template


def LLM_Generated_text(prompt):

    url = "http://172.27.33.102:8025/generate"      # /mnt/huggingface/models/Llama-3-8B-Instruct/

    template = base_prompt_template()
    query = template.format(query=prompt)

    payload = json.dumps({
        "prompt": query,
        "temperature": 0.2,
        "max_tokens": 3000,
        "n": 1,
        "stop": ["<|eot_id|>", "<|end_of_text|>", "<|end_header_id|>"],
    })

    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)

    try:
        content = response.json()['text'][0].replace(query, '')
    except (KeyError, IndexError, TypeError):
        content = "1234567"  # or raise an error if you prefer
    return content






def process_data(data):
    """
    处理单个数据，生成LLM分析结果并返回。
    """
    origin_text = data["text"]
    # origin_sentences = origin_text.split('</s>')
    all_text = origin_text.replace("</s>", " ")

    # analysis_results = []
    # for sentence in origin_sentences:

    prompt = f"""
    Please read the following text and analyze the extent of large language models involvement in its creation. 
    Your analysis should be based on three aspects: Lexicon, Grammar, and Syntax.

    {all_text}

    Finally, you only need to write your analysis text in English without any introductory statements.
    """
    LLM_analysis = LLM_Generated_text(prompt).replace('\n', ' ').replace('  ', ' ')
    # analysis_results.append(LLM_analysis_sentences)

    # analysis_text = '</s>'.join(analysis_results)
    data["LLM_analysis_alltext"] = LLM_analysis

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

    processes = 20
    p = Pool(processes=processes)

    file_list = ["train_large.json", "val_large.json", "test_large.json"]

    for file_path in file_list:

        input_file_path = f"/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/PASTED/{file_path}"
        output_file_path = f"/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/PASTED/Baichuan2/{file_path}"

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

