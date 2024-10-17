import textstat
import nltk
import collections
import authorstyle.features as features
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


###################### 计算可读性 ##########################
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
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype list
    :returns List of pos tag trigram frequencies with consistent dimensions
    """
    # 提取最常见的三元组，如果不足 20 个则返回所有现有的
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

    print(feature_vector)

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

    print(len(feature_vector))

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

    print(len(feature_vector))

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

    print(feature_vector)

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

###################### 计算文本困惑度 ##########################

# model_name = "/mnt/data132/taozhen/AI_Thesis_Detection/Framework/gpt2/"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)
#
#
# # 计算困惑度的函数
# def compute_perplexity(text):
#     input_ids = tokenizer.encode(text, return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(input_ids, labels=input_ids)
#         loss = outputs.loss
#         perplexity = torch.exp(loss)
#     return perplexity.item()
#
#
# # 处理长文本
# def compute_perplexity_for_long_text(text, max_length=256):
#     sentences = text.split('.')  # 根据句号分割句子
#     total_perplexity = 0
#     count = 0
#
#     for sentence in sentences:
#         sentence = sentence.strip()
#         if sentence:
#             if len(sentence) > max_length:
#                 chunks = [sentence[i:i + max_length] for i in range(0, len(sentence), max_length)]
#                 for chunk in chunks:
#                     total_perplexity += compute_perplexity(chunk)
#                     count += 1
#             else:
#                 total_perplexity += compute_perplexity(sentence)
#                 count += 1
#
#     average_perplexity = total_perplexity / count if count > 0 else float('inf')
#     return average_perplexity


def all_feature(text):
    readability_metrics = calculate_readability_metrics(text)
    authorstyle_metrics = calculate_authorstyle_metrics(text)
    # perplexity = compute_perplexity_for_long_text(text)

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

    # # 困惑度
    # perplexity_vector = [perplexity]
    # print("perplexity_vector:", perplexity_vect
    # or)

    # final_vector = readability_vector + authorstyle_vector + perplexity_vector
    final_vector = readability_vector + authorstyle_vector
    # 将向量转换为numpy数组 (如果需要)
    # final_vector = np.array(final_vector)

    # scaler = StandardScaler()
    # final_vector = scaler.fit_transform(final_vector.reshape(-1, 1)).flatten()

    return final_vector


text = """
The beauty of language lies in its ability to connect people from different cultures and backgrounds. Through communication, we share ideas, express emotions, and build understanding. No matter what language we speak, the essence of human connection remains universal. It reminds us that we are all part of the same global community, united by our desire to learn and grow together.
"""


final = all_feature(text)
print(len(final))





