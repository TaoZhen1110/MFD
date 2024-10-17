import textstat
import nltk
import collections
import authorstyle.features as features
from multiprocessing import Pool
import os
import json
from tqdm import tqdm



###################### 计算文本可读性 ##########################
def calculate_readability_metrics(text):

    readability_metrics = {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Flesch-Kincaid Grade Level": textstat.flesch_kincaid_grade(text),
        "SMOG Index": textstat.smog_index(text),
        "Coleman-Liau Index": textstat.coleman_liau_index(text),
        "Automated Readability Index": textstat.automated_readability_index(text),
        "Dale-Chall Readability Score": textstat.dale_chall_readability_score(text),
        "Difficult Words Count": textstat.difficult_words(text),
        "Linsear Write Formula": textstat.linsear_write_formula(text),
        "Gunning Fog Index": textstat.gunning_fog(text),
        # "Text Standard": textstat.text_standard(text),
        "Fernandez-Huerta Index": textstat.fernandez_huerta(text),
        "Szigriszt-Pazos Index": textstat.szigriszt_pazos(text),
        "Gutierrez-Polini Index": textstat.gutierrez_polini(text),
        "Crawford Index": textstat.crawford(text),
        "Gulpease Index": textstat.gulpease_index(text),
        "Osman Index": textstat.osman(text)
    }

    return readability_metrics


###################### 计算文本风格 ##########################
class Text:
    def __init__(self, text):
        self.text = text
        self.tokens = nltk.word_tokenize(text)
        self.tokens_alphabetic = [t for t in self.tokens if t.isalpha()]
        self.tokens_without_stopwords = [t for t in self.tokens_alphabetic if t.lower() not in nltk.corpus.stopwords.words('english')]
        self.sentences = nltk.sent_tokenize(text)
        self.pos_tags = nltk.pos_tag(self.tokens)
        self.document = self
        self.fragments = [self]

        # For simplicity, using the entire text as a single fragment
        self.sentence_length = [len(nltk.word_tokenize(sentence)) for sentence in self.sentences]
        self.pos_tag_trigram_freq = collections.Counter(nltk.ngrams([tag for word, tag in self.pos_tags], 3))
        self.words_without_stopwords_frequency = collections.Counter(self.tokens_without_stopwords)
        self.word_bigram_frequency = collections.Counter(nltk.bigrams(self.tokens))
        self.bigram_frequency = collections.Counter(nltk.ngrams(self.tokens, 2))
        self.frequency_distribution = nltk.FreqDist(self.tokens_alphabetic)  # Adding frequency_distribution


def pos_tag_trigram_frequency(text, top=10):

    doc_trigrams = text.document.pos_tag_trigram_freq.most_common(top)
    feature_vector = []
    tag_fd = text.pos_tag_trigram_freq
    total_count = sum(tag_fd.values())

    for n_gram, freq in doc_trigrams:
        feature_vector.append(0.0 if total_count == 0 else (freq / total_count))

    # 如果特征向量不足 20 个，则填充 0.0
    while len(feature_vector) < top:
        feature_vector.append(0.0)


    return feature_vector


def most_common_words_without_stopwords(text, top=10):
    """
    Returns the frequency of the documents top n words in the text
    :type text: Text
    :param text: The text to be analysed
    :param top: int
    :returns Returns list of frequencies
    """
    top_max = min([sum(text.document.words_without_stopwords_frequency.values()), top])
    doc_n_grams = text.document.words_without_stopwords_frequency.most_common(top_max)
    n_freq = text.words_without_stopwords_frequency
    feature_vector = []
    total_count = sum(n_freq.values())

    for n_gram, freq in doc_n_grams:
        feature_vector.append(0.0 if total_count == 0 else (freq / total_count))

    if len(feature_vector) != top:
        feature_vector += [0.0] * (top - len(feature_vector))

    return feature_vector


def top_word_bigram_frequency(text, top=10):
    """
    Returns the frequency of the documents top n word bigrams in the text
    :type text: Text
    :param text: The text to be analysed
    :param top: int
    :returns Returns list of frequencies
    """
    top_max = min([sum(text.document.word_bigram_frequency.values()), top])
    doc_n_grams = text.document.word_bigram_frequency.most_common(top_max)
    n_freq = text.word_bigram_frequency
    feature_vector = []
    total_count = sum(n_freq.values())

    for n_gram, freq in doc_n_grams:
        feature_vector.append(0.0 if total_count == 0 else (freq / total_count))

    if len(feature_vector) != top:
        feature_vector += [0.0] * (top - len(feature_vector))

    return feature_vector


def top_bigram_frequency(text, top=10):
    """
    Returns the frequency of the documents top n character bigrams in the text
    :type text: Text
    :param text: The text to be analysed
    :param top: int
    :returns Returns list of frequencies
    """
    top_max = min([sum(text.document.bigram_frequency.values()), top])
    doc_n_grams = text.document.bigram_frequency.most_common(top_max)
    n_freq = text.bigram_frequency
    feature_vector = []
    total_count = sum(n_freq.values())

    for n_gram, freq in doc_n_grams:
        feature_vector.append(0.0 if total_count == 0 else (freq / total_count))

    if len(feature_vector) != top:
        feature_vector += [0.0] * (top - len(feature_vector))

    return feature_vector


def top_3_gram_frequency(text, top=10):
    """
    Returns the frequency of the documents top n character 3-grams in the text
    :type text: Text
    :param text: The text to be analysed
    :param top: int
    :returns Returns list of frequencies
    """
    doc_n_grams = collections.Counter(nltk.ngrams(text.document.text, 3))
    top_max = min([sum(doc_n_grams.values()), top])
    doc_n_grams = doc_n_grams.most_common(top_max)
    n_freq = collections.Counter(nltk.ngrams(text.text, 3))
    feature_vector = []
    total_count = sum(n_freq.values())

    for n_gram, freq in doc_n_grams:
        feature_vector.append(0.0 if total_count == 0 else (freq / total_count))

    if len(feature_vector) != top:
        feature_vector += [0.0] * (top - len(feature_vector))

    return feature_vector



def calculate_authorstyle_metrics(text):
    text_obj = Text(text)

    authorstyle_features = {
        "average_word_length": features.lexical_features.average_word_length(text_obj),
        "pos_tag_frequency": features.lexical_features.pos_tag_frequency(text_obj),
        "pos_tag_trigram_frequency": pos_tag_trigram_frequency(text_obj),
        "word_length_distribution": features.lexical_features.word_length_distribution(text_obj),
        "average_sentence_length_words": features.lexical_features.average_sentence_length_words(text_obj),
        "average_syllables_per_word": features.lexical_features.average_syllables_per_word(text_obj),
        "average_sentence_length_chars": features.lexical_features.average_sentence_length_chars(text_obj),
        "sentence_length_distribution": features.lexical_features.sentence_length_distribution(text_obj),
        "yule_k_metric": features.lexical_features.yule_k_metric(text_obj),
        "sichel_s_metric": features.lexical_features.sichel_s_metric(text_obj),
        "average_word_frequency_class": features.lexical_features.average_word_frequency_class(text_obj),
        "punctuation_frequency": features.stylometric_features.punctuation_frequency(text_obj),
        "special_character_frequency": features.stylometric_features.special_character_frequency(text_obj),
        "uppercase_frequency": features.stylometric_features.uppercase_frequency(text_obj),
        "number_frequency": features.stylometric_features.number_frequency(text_obj),
        "functionword_frequency": features.stylometric_features.functionword_frequency(text_obj),
        "most_common_words_without_stopwords": most_common_words_without_stopwords(text_obj),
        "stopword_ratio": features.stylometric_features.stopword_ratio(text_obj),
        "top_word_bigram_frequency": top_word_bigram_frequency(text_obj),
        "top_bigram_frequency": top_bigram_frequency(text_obj),
        "top_3_gram_frequency": top_3_gram_frequency(text_obj)
    }

    return authorstyle_features


def all_feature(text):
    readability_metrics = calculate_readability_metrics(text)
    authorstyle_metrics = calculate_authorstyle_metrics(text)

    # 可读性特征值
    readability_vector = list(readability_metrics.values())

    # 文本风格特征值
    authorstyle_vector = []
    for value in authorstyle_metrics.values():
        # 检查值是否为字典，如果是字典，提取其值
        if isinstance(value, dict):
            for sub_value in value.values():
                if isinstance(sub_value, list):
                    authorstyle_vector.extend(sub_value)
                else:
                    authorstyle_vector.append(sub_value)
        elif isinstance(value, list):
            authorstyle_vector.extend(value)
        else:
            authorstyle_vector.append(value)


    final_vector = readability_vector + authorstyle_vector

    # 将向量转换为numpy数组 (如果需要)
    # final_vector = np.array(final_vector)

    return final_vector





def Third_Level_Feature(data):

    origin_text = data["text"]
    origin_sentences = origin_text.split('</s>')

    sentence_features = []
    for sentence in origin_sentences:
        third_level_feature = all_feature(sentence)
        sentence_features.append(third_level_feature)

    data["Third_Level_Features"] = sentence_features

    return data



def process_line_with_retry(line, max_attempts=3):

    for attempt in range(1, max_attempts + 1):
        try:
            return Third_Level_Feature(line)
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
    processes = 6
    p = Pool(processes=processes)

    input_file_path = "/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/OOD/test.json"
    output_file_path = "/mnt/data132/taozhen/AI_Thesis_Detection/Dataset/OOD/test1.json"

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








