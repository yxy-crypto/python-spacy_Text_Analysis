import spacy
import nltk
import pandas as pd
import openpyxl
from collections import Counter
from textblob import TextBlob
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords

# 载入 NLP 模型
nlp = spacy.load("en_core_web_sm")

# 读取文本文件
with open("ArticleAnalysis.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 进度条设置
tqdm.pandas()

# 1️⃣ 句法结构分析（带原句对照）
def analyze_syntax(text):
    doc = nlp(text)
    sentence_structure_data = []

    for sent in tqdm(doc.sents, desc="分析句法结构"):
        original_sentence = sent.text
        syntax_pattern = " ".join([token.dep_ for token in sent])
        sentence_structure_data.append((original_sentence, syntax_pattern))

    return sentence_structure_data

# 2️⃣ 连接词依赖分析
def analyze_conjunctions(text):
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)

    # 使用NLTK的词性标注来识别连接词
    # CC: 并列连词 (and, but, or)
    # IN: 从属连词/介词 (because, although, if, when, since)
    conjunction_tags = {'CC', 'IN'}

    # 一些特殊的副词也可以作为连接词 (however, therefore, consequently, etc.)
    adverb_conjunctions = {
        'however', 'therefore', 'consequently', 'moreover', 'furthermore',
        'nevertheless', 'meanwhile', 'thus', 'hence', 'otherwise', 'instead'
    }

    conjunctions = []
    for word, tag in tqdm(pos_tags, desc="分析连接词"):
        if tag in conjunction_tags:
            conjunctions.append(word.lower())
        elif tag == 'RB' and word.lower() in adverb_conjunctions:
            conjunctions.append(word.lower())

    return Counter(word for word in tqdm(words, desc="分析连接词") if word in conjunctions)

# 3️⃣ 句型多样性分析
# 加载预训练的模型
nlp = spacy.load('en_core_web_sm')

def analyze_sentence_types(text):
    doc = nlp(text)
    sentence_types = {"simple": 0, "complex": 0, "compound": 0}

    for sentence in tqdm(doc.sents, desc="分析句型多样性"):
        coordinators = []  # 存储并列连词
        subordinators = []  # 存储从属连词
        clause_depth = 0  # 用于记录从句嵌套深度

        # 分析句子中的词和依赖关系
        for token in sentence:
            # 并列连词
            if token.dep_ == 'cc':
                coordinators.append(token.text)
            # 从属连词
            if token.dep_ == 'mark':
                subordinators.append(token.text)
            # 从句的标志
            if token.dep_ in ['ccomp', 'acl', 'relcl']:
                clause_depth += 1

        # 判断句型
        if len(coordinators) > 0:  # 如果有并列连词，认为是复合句
            sentence_types["compound"] += 1
        elif len(subordinators) > 0 and clause_depth > 0:  # 如果有从属连词且从句嵌套深度大于0，认为是复杂句
            sentence_types["complex"] += 1
        else:  # 否则认为是简单句
            sentence_types["simple"] += 1

    return sentence_types


# 4️⃣ 词性多样性分析
def analyze_pos_diversity(text):
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    return Counter(tag for _, tag in tqdm(pos_tags, desc="分析词性多样性")).most_common(10)

# 5️⃣ 词汇丰富度分析
def analyze_vocabulary_richness(text):
    words = word_tokenize(text.lower())
    unique_words = set(words)
    ttr = len(unique_words) / len(words)  # 计算 TTR

    word_synonym_usage = {}
    for word in tqdm(unique_words, desc="分析词汇丰富度"):
        synonyms = set(lemma.name() for syn in wn.synsets(word) for lemma in syn.lemmas())
        if len(synonyms) > 1:
            word_synonym_usage[word] = synonyms

    return {
        "TTR": round(ttr, 3),
        "Common Words Without Synonym Variants": {word: syns for word, syns in word_synonym_usage.items() if
                                                  len(syns) < 3}
    }

# 获取单词的同义词、词性、释义、语域和频率
def get_synonyms_and_details(word, word_freq):
    # 获取该单词的所有同义词集
    synsets = wn.synsets(word)
    result = []

    # 获取该单词的频率
    word_frequency = word_freq[word] if word in word_freq else 0

    for synset in synsets:
        pos = synset.pos()  # 词性
        definition = synset.definition()  # 释义
        lexname = synset.lexname()  # 语域

        # 获取同义词
        synonyms = [lemma.name() for lemma in synset.lemmas()]

        # 结果存储
        result.append({
            'Word': word,
            'POS': pos,
            'Definition': definition,
            'Synonyms': ', '.join(synonyms),
            'Lexname (Domain)': lexname,
            'Frequency': word_frequency  # 添加频率信息
        })

    return result

# 获取停用词表（仅限英文）
stop_words = set(stopwords.words("english"))

# 预处理文本（去除停用词 + 词形归一化）
def preprocess_text(text):
    words = word_tokenize(text.lower())  # 转小写并分词
    lemmatized_words = [token.lemma_ for token in nlp(" ".join(words))]  # 词形还原
    filtered_words = [word for word in lemmatized_words if word.isalpha() and word not in stop_words]  # 去除停用词和非字母字符
    return filtered_words

# 计算词频（去除停用词后）
filtered_words = preprocess_text(text)
word_freq = Counter(filtered_words)

# 获取前 10 高频词
top_10_words = [word for word, _ in word_freq.most_common(10)]

# 获取这些高频词的详细信息
vocab_details = []
for word in top_10_words:
    vocab_details.extend(get_synonyms_and_details(word, word_freq))  # 传递频率

# 运行所有分析
syntax_results = analyze_syntax(text)
conj_results = analyze_conjunctions(text)
sentence_types_results = analyze_sentence_types(text)
pos_diversity_results = analyze_pos_diversity(text)
vocab_richness_results = analyze_vocabulary_richness(text)

# 创建 Excel 文件并写入分析结果
with pd.ExcelWriter("ArticleResults.xlsx", engine="openpyxl") as writer:
    # 句法结构分析（原句与句法结构）
    syntax_df = pd.DataFrame(syntax_results, columns=["Original Sentence", "Sentence Structure"])
    syntax_df.to_excel(writer, sheet_name="Syntax Analysis", index=False)

    # 连接词依赖分析
    conj_df = pd.DataFrame(conj_results.items(), columns=["Conjunction", "Frequency"])
    conj_df.to_excel(writer, sheet_name="Conjunction Analysis", index=False)

    # 句型多样性分析
    sentence_types_df = pd.DataFrame(list(sentence_types_results.items()), columns=["Sentence Type", "Count"])
    sentence_types_df.to_excel(writer, sheet_name="Sentence Type Diversity", index=False)

    # 词性多样性分析
    pos_diversity_df = pd.DataFrame(pos_diversity_results, columns=["POS Tag", "Frequency"])
    pos_diversity_df.to_excel(writer, sheet_name="POS Diversity", index=False)

    # 词汇丰富度分析
    vocab_richness_df = pd.DataFrame(
        [{"TTR": vocab_richness_results["TTR"], "Common Words Without Synonym Variants": word}
         for word in vocab_richness_results["Common Words Without Synonym Variants"].keys()],
        columns=["TTR", "Common Words Without Synonym Variants"]
    )
    vocab_richness_df.to_excel(writer, sheet_name="Vocabulary Richness", index=False)

    # 单词详细信息写入新工作表
    vocab_details_df = pd.DataFrame(vocab_details)
    vocab_details_df.to_excel(writer, sheet_name="Word Details", index=False)

print("分析结果已成功写入 'ArticleResults.xlsx' 文件。")

# 打印结果到控制台
print("1️⃣ 句法结构分析:")
print(syntax_results)
print("\n2️⃣ 连接词依赖分析:")
print(conj_results)
print("\n3️⃣ 句型多样性:")
print(sentence_types_results)
print("\n4️⃣ 词性多样性:")
print(pos_diversity_results)
print("\n5️⃣ 词汇丰富度:")
print(vocab_richness_results)

# 结果写入文件
with open("ArticleResults.txt", "w", encoding="utf-8") as f:
    f.write(f"1️⃣ 句法结构分析:\n")
    for sentence, structure in syntax_results:
        f.write(f"原句: {sentence}\n句法结构: {structure}\n\n")
    f.write(f"2️⃣ 连接词依赖分析: {conj_results}\n")
    f.write(f"3️⃣ 句型多样性: {sentence_types_results}\n")
    f.write(f"4️⃣ 词性多样性: {pos_diversity_results}\n")
    f.write(f"5️⃣ 词汇丰富度: {vocab_richness_results}\n")
    f.write(f"\n词汇详细信息: {vocab_details}\n")
    print("分析结果已成功写入 'ArticleResults.txt' 文件。")
