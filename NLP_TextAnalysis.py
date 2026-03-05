import pandas as pd
from collections import Counter
import nltk
import string
from tqdm import tqdm
from spacy import displacy
import spacy

# 加载 spaCy 英文小模型
nlp = spacy.load('en_core_web_sm')

# 1. 读取文本文件
with open('ArticleAnalysis.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 2. 预处理文本
# 转换为小写并去掉标点符号
text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))

# 3. 分词
words = nltk.word_tokenize(text)

# 4. 计算每个词的频率
word_freq = Counter()

# 使用 tqdm 添加进度条
for word in tqdm(words, desc="Processing words", unit="word"):
    word_freq[word] += 1

# 5. 获取最高频的 30 个词
top_30_words = word_freq.most_common(30)

# 6. 将结果保存到 Excel 文件
# 创建 DataFrame
df = pd.DataFrame(top_30_words, columns=['word', 'frequency'])

# 将 DataFrame 写入 Excel 文件
df.to_excel('wordlist.xlsx', index=False)

print("The top 20 words have been saved to 'wordlist.xlsx'.")

# 读取文章
with open("ArticleAnalysis.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 解析文本
doc = nlp(text)

# 可视化句法结构
displacy.render(doc.sents, style="dep", jupyter=True)

# 渲染句子的依赖关系并保存为 HTML 文件
displacy.render(doc.sents, style="dep", page=True, jupyter=False)

# 保存为 HTML 文件
with open("dependency_visualization.html", "w", encoding="utf-8") as f:
    f.write(displacy.render(doc.sents, style="dep", page=True, jupyter=False))

print("已经保存为HTML文件dependency_visualization.html'.")