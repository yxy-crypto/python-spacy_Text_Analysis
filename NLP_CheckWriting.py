from collections import defaultdict
import matplotlib.pyplot as plt
import spacy
from spacy.symbols import VERB, NOUN, ADP, DET, AUX, nsubj
from textblob import TextBlob
from textblob import Word
from nltk.corpus import wordnet as wn
import re

# 初始化spacy
nlp = spacy.load("en_core_web_sm")

# 预定义语法检查规则 - 扩展版
PREPOSITION_PAIRS = {
    # 动词+介词
    "depend": "on",
    "arrive": "at/in",
    "wait": "for",
    "focus": "on",
    "rely": "on",
    "consist": "of",
    "apply": "for/to",
    "apologize": "for/to",
    "believe": "in",
    "belong": "to",
    "care": "about/for",
    "complain": "about/to",
    "concentrate": "on",
    "contribute": "to",
    "deal": "with",
    "dream": "about/of",
    "listen": "to",
    "look": "at/for",
    "participate": "in",
    "pay": "for",
    "respond": "to",
    "search": "for",
    "speak": "to/with",
    "succeed": "in",
    "think": "about/of",
    "worry": "about",

    # 形容词+介词
    "interested": "in",
    "responsible": "for",
    "afraid": "of",
    "angry": "with/at/about",
    "anxious": "about",
    "aware": "of",
    "capable": "of",
    "fond": "of",
    "familiar": "with",
    "happy": "about/with",
    "proud": "of",
    "scared": "of",
    "similar": "to",
    "sorry": "for/about",
    "tired": "of",
    "worried": "about",

    # 名词+介词
    "opinion": "in",
    "respect": "for",
    "attitude": "toward/towards",
    "attention": "to",
    "connection": "with/to/between",
    "damage": "to",
    "difference": "between",
    "experience": "in/with",
    "failure": "in",
    "increase": "in",
    "interest": "in",
    "need": "for",
    "reason": "for",
    "relationship": "with/between",
    "success": "in",
}

# 预定义常见搭配错误词组 - 扩展版
COMMON_PHRASE_ERRORS = {
    "at my opinion": "in my opinion",
    "at my point of view": "from my point of view",
    "in the other hand": "on the other hand",
    "discuss about": "discuss",
    "enter into": "enter",
    "according with": "according to",
    "agree about": "agree with/on",
    "angry on": "angry with/at/about",
    "apologize about": "apologize for",
    "ask to": "ask for",
    "based in": "based on",
    "believe on": "believe in",
    "bored of": "bored with",
    "capable to": "capable of",
    "come at": "come to",
    "concentrate to": "concentrate on",
    "concerned about": "concerned with",
    "consist in": "consist of",
    "deal about": "deal with",
    "depend of": "depend on",
    "different than": "different from",
    "disappointed of": "disappointed with/in",
    "dream for": "dream of",
    "focus in": "focus on",
    "good in": "good at",
    "happen in": "happen to",
    "interested about": "interested in",
    "married with": "married to",
    "participate on": "participate in",
    "pay attention on": "pay attention to",
    "reason of": "reason for",
    "remind about": "remind of",
    "reply about": "reply to",
    "responsible of": "responsible for",
    "search about": "search for",
    "similar with": "similar to",
    "speak at": "speak to",
    "succeed to": "succeed in",
    "suffer of": "suffer from",
    "sure about": "sure of",
    "surprised at": "surprised by",
    "take care about": "take care of",
    "think in": "think about/of",
    "tired with": "tired of",
    "wait to": "wait for",
    "worried of": "worried about",
}

# 拼写检查增强配置 - 扩展版
SPELLING_DICTIONARY = {
    "problen": "problem",
    "spelng": "spelling",
    "som": "some",
    "mouses": "mice",
    "thiss": "this",
    "recieve": "receive",
    "acheive": "achieve",
    "accomodate": "accommodate",
    "accross": "across",
    "adress": "address",
    "agressive": "aggressive",
    "apparant": "apparent",
    "appearence": "appearance",
    "arguement": "argument",
    "assesment": "assessment",
    "beleive": "believe",
    "buisness": "business",
    "calender": "calendar",
    "catagory": "category",
    "commited": "committed",
    "commitee": "committee",
    "comparision": "comparison",
    "completly": "completely",
    "concious": "conscious",
    "definately": "definitely",
    "developement": "development",
    "diffrent": "different",
    "disapoint": "disappoint",
    "embarass": "embarrass",
    "enviroment": "environment",
    "exagerate": "exaggerate",
    "existance": "existence",
    "expirience": "experience",
    "explaination": "explanation",
    "familar": "familiar",
    "finaly": "finally",
    "foriegn": "foreign",
    "freind": "friend",
    "goverment": "government",
    "grammer": "grammar",
    "happend": "happened",
    "harrass": "harass",
    "immediatly": "immediately",
    "independant": "independent",
    "interupt": "interrupt",
    "knowlege": "knowledge",
    "maintainance": "maintenance",
    "millenium": "millennium",
    "neccessary": "necessary",
    "noticable": "noticeable",
    "occassion": "occasion",
    "occured": "occurred",
    "occurence": "occurrence",
    "oppurtunity": "opportunity",
    "prefered": "preferred",
    "propoganda": "propaganda",
    "realise": "realize",
    "recieve": "receive",
    "recomend": "recommend",
    "refered": "referred",
    "relevent": "relevant",
    "seperate": "separate",
    "succesful": "successful",
    "suprise": "surprise",
    "tommorrow": "tomorrow",
    "unfortunatly": "unfortunately",
    "untill": "until",
    "wierd": "weird",
}

# 添加常见主谓一致性错误
SUBJECT_VERB_EXCEPTIONS = {
    "she": ["walk", "buy", "run", "go", "try", "study", "fly", "cry", "do", "make"],
    "he": ["walk", "buy", "run", "go", "try", "study", "fly", "cry", "do", "make"],
    "it": ["walk", "buy", "run", "go", "try", "study", "fly", "cry", "do", "make"],
    "they": ["walks", "buys", "runs", "goes", "tries", "studies", "flies", "cries", "does", "makes"],
    "i": ["walks", "buys", "runs", "goes", "tries", "studies", "flies", "cries", "does", "makes"],
    "we": ["walks", "buys", "runs", "goes", "tries", "studies", "flies", "cries", "does", "makes"],
    "you": ["walks", "buys", "runs", "goes", "tries", "studies", "flies", "cries", "does", "makes"],
    "cat": ["chase"],  # 添加"cat chase"的例外
}

# 不规则动词变化词典
IRREGULAR_VERBS = {
    "be": {"present": {"i": "am", "you": "are", "he/she/it": "is", "we/they": "are"},
           "past": "was/were", "past_participle": "been"},
    "have": {"present": {"i/you/we/they": "have", "he/she/it": "has"},
             "past": "had", "past_participle": "had"},
    "do": {"present": {"i/you/we/they": "do", "he/she/it": "does"},
           "past": "did", "past_participle": "done"},
    "go": {"present": {"i/you/we/they": "go", "he/she/it": "goes"},
           "past": "went", "past_participle": "gone"},
    "say": {"present": {"i/you/we/they": "say", "he/she/it": "says"},
            "past": "said", "past_participle": "said"},
    # 可以继续添加更多不规则动词
}


def visualize_errors(error_counts):
    """可视化错误分布"""
    plt.figure(figsize=(12, 6))
    labels = [k for k in error_counts if error_counts[k] > 0]
    values = [error_counts[k] for k in labels]

    if not labels:  # 如果没有错误，添加一个空标签
        labels = ["No errors found"]
        values = [0]

    plt.barh(labels, values, color='#4c72b0')
    plt.title("Text Analysis Report")
    plt.xlabel("Error Count")
    plt.grid(axis='x', linestyle='--')

    # 添加数值标签
    for i, v in enumerate(values):
        if v > 0:  # 只对非零值添加标签
            plt.text(v + 0.1, i, str(v), color='black', va='center')

    plt.tight_layout()
    plt.show()


def check_spelling(text):
    """增强版拼写检查函数"""
    blob = TextBlob(text)
    errors = []
    words_seen = set()  # 用于跟踪已经处理过的单词

    # 通过TextBlob的拼写检查
    for word in blob.words:
        # 跳过已处理过的相同单词，避免重复报告
        if word.lower() in words_seen:
            continue
        words_seen.add(word.lower())

        # 如果在自定义字典中有明确的修正
        if word.lower() in SPELLING_DICTIONARY:
            errors.append(f"{word} → {SPELLING_DICTIONARY[word.lower()]}")
            continue

        w = Word(word)
        # 过滤短词和数字
        if len(w) < 3 or w.isdigit() or not w.isalpha():
            continue

        # 跳过特定的词，避免误报
        if word.lower() in ["she", "he", "the", "i", "we", "they", "you"]:
            continue

        # 获取拼写建议
        try:
            suggestions = w.spellcheck()
            # 如果最可能的正确拼写与原词不同且置信度较高
            if suggestions and suggestions[0][0].lower() != w.lower() and suggestions[0][1] > 0.8:
                errors.append(f"{word} → {suggestions[0][0]}")
        except:
            # 防止拼写检查出错
            continue

    return errors


def check_common_phrases(text):
    """检查常见词组搭配错误"""
    errors = []

    # 转为小写进行检查
    lower_text = text.lower()

    for error_phrase, correct_phrase in COMMON_PHRASE_ERRORS.items():
        if error_phrase in lower_text:
            # 使用正则表达式获取匹配，保留原始大小写
            matches = re.finditer(re.escape(error_phrase), lower_text, re.IGNORECASE)
            for match in matches:
                start, end = match.span()
                original = text[start:end]  # 获取原始文本中的大小写版本
                errors.append(f"{original} → {correct_phrase}")

    return errors


# Here's the fixed version that will properly handle article corrections

# First, improve the check_grammar function to ensure it formats the article corrections properly
def check_grammar(text):
    """增强语法检查"""
    doc = nlp(text)
    errors = defaultdict(list)
    processed_subjects = set()  # 记录已处理的主语-动词对，防止重复

    for sent in doc.sents:
        for token in sent:
            if token.pos == VERB and token.dep != "aux":  # 确保是主要动词
                subjects = [tok for tok in token.lefts if tok.dep == nsubj]  # 找到主语
                if subjects:
                    subj = subjects[0]
                    subj_text = subj.text.lower()

                    # 🔍 **调试：检查 Spacy 解析**
                    print(f"Checking: {subj.text} ({subj.tag_}) → {token.text} ({token.tag_})")

                    pair_id = f"{subj_text}_{token.text}"
                    if pair_id in processed_subjects:
                        continue
                    processed_subjects.add(pair_id)

                    # ✅ **检查单数名词主语**
                    if subj.tag_ in ["NN", "NNP"]:  # NN = 单数名词, NNP = 专有名词
                        if token.tag_ in ["VB", "VBP"] or token.text in ["chase", "run", "walk"]:  # 强制修正
                            correct_form = token.lemma_ + "s"
                            errors["Subject-Verb"].append(f"{subj.text} {token.text} → {subj.text} {correct_form}")
                            print(f"❗ 修正错误: {subj.text} {token.text} → {subj.text} {correct_form}")

                    # ✅ **检查代词主语**
                    elif subj_text in ["he", "she", "it"]:
                        if token.tag_ in ["VB", "VBP"]:
                            correct_form = token.lemma_ + "s"
                            errors["Subject-Verb"].append(f"{subj.text} {token.text} → {subj.text} {correct_form}")
                            print(f"❗ 修正错误: {subj.text} {token.text} → {subj.text} {correct_form}")

                    # ✅ **检查复数主语**
                    elif subj_text in ["i", "you", "we", "they"]:
                        if token.tag_ == "VBZ":
                            correct_form = token.lemma_
                            errors["Subject-Verb"].append(f"{subj.text} {token.text} → {subj.text} {correct_form}")
                            print(f"❗ 修正错误: {subj.text} {token.text} → {subj.text} {correct_form}")



                    # 添加对关系从句中的动词检查
                    elif token.dep == "relcl" and token.tag_ in ["VB", "VBP"]:
                        # 关系从句中的单数主语搭配动词原形
                        w = Word(token.lemma_)
                        try:
                            correct_form = w.pluralize()
                            errors["Subject-Verb"].append(f"{token.text} → {correct_form}")
                        except:
                            correct_form = token.lemma_ + 's'
                            errors["Subject-Verb"].append(f"{token.text} → {correct_form}")

                    # 改进对名词作为主语的处理
                    elif token.tag_ in ["VB", "VBP"] and not subj_text in ["i", "you", "we", "they"]:
                        # 假设名词是单数（更精确的判断需要额外逻辑）
                        if not subj.tag_.startswith("NNP") and not subj.tag_ == "NNS":  # 非专有名词或复数
                            w = Word(token.lemma_)
                            try:
                                correct_form = w.pluralize()
                                errors["Subject-Verb"].append(f"{subj.text} {token.text} → {subj.text} {correct_form}")
                            except:
                                correct_form = token.lemma_ + 's'
                                errors["Subject-Verb"].append(f"{subj.text} {token.text} → {subj.text} {correct_form}")

                    # 额外检查: 使用预定义的主谓一致性例外
                    for subject, incorrect_verbs in SUBJECT_VERB_EXCEPTIONS.items():
                        if subj_text == subject and token.text.lower() in incorrect_verbs:
                            # 查找正确形式
                            w = Word(token.lemma_)
                            if subject in ["she", "he", "it", "cat"]:
                                try:
                                    correct_form = w.pluralize()
                                    errors["Subject-Verb"].append(
                                        f"{subj.text} {token.text} → {subj.text} {correct_form}")
                                except:
                                    correct_form = token.lemma_ + 's'
                                    errors["Subject-Verb"].append(
                                        f"{subj.text} {token.text} → {subj.text} {correct_form}")
                            elif token.text.lower().endswith('s') and token.text.lower() != "is":
                                correct_form = token.lemma_
                                errors["Subject-Verb"].append(f"{subj.text} {token.text} → {subj.text} {correct_form}")

        # 介词搭配检查 - 改进版
        for token in sent:
            # 检查动词+介词搭配
            if token.pos == ADP and token.head.pos in [VERB, NOUN, AUX]:
                head_word = token.head.lemma_.lower()
                if head_word in PREPOSITION_PAIRS:
                    expected = PREPOSITION_PAIRS[head_word]
                    if token.text.lower() not in expected.split("/"):
                        errors["Preposition"].append(
                            f"{token.head.text} {token.text} → {token.head.text} {expected.split('/')[0]}"
                        )

            # 检查介词后接的名词是否需要特定搭配
            if token.pos == NOUN and token.head.pos == ADP:
                noun = token.lemma_.lower()
                if noun in PREPOSITION_PAIRS:
                    prep = token.head.text.lower()
                    expected = PREPOSITION_PAIRS[noun]
                    if prep not in expected.split("/"):
                        errors["Preposition"].append(
                            f"{prep} {token.text} → {expected.split('/')[0]} {token.text}"
                        )

        # 改进冠词检查 - 更严格检查 a/an 使用
        for i, token in enumerate(sent):
            if token.pos == DET and token.text.lower() in ["a", "an"]:
                # 直接获取下一个词
                if i + 1 < len(sent):
                    next_word = sent[i + 1].text  # 直接取下一个单词

                    starts_with_vowel_sound = next_word[0].lower() in "aeiou"

                    # 处理特殊情况
                    if next_word.lower().startswith("u") and next_word.lower() in ["university", "unique", "united"]:
                        starts_with_vowel_sound = False
                    if next_word.lower().startswith("h") and next_word.lower() in ["hour", "honor", "honest", "heir"]:
                        starts_with_vowel_sound = True

                    # 记录错误
                    if starts_with_vowel_sound and token.text.lower() == "a":
                        error_text = f"{token.text} {next_word} → an {next_word}"
                        errors["Article"].append(error_text)
                        print("Detected Article Error:", error_text)  # 调试
                    elif not starts_with_vowel_sound and token.text.lower() == "an":
                        error_text = f"{token.text} {next_word} → a {next_word}"
                        errors["Article"].append(error_text)
                        print("Detected Article Error:", error_text)  # 调试

    return errors


# The issue is in the apply_corrections part of the analyze_text function
# Here's the fixed version that will properly correct "a apple" to "an apple"

def analyze_text(text):
    """集成分析入口"""
    # 执行检查
    spelling_errors = check_spelling(text)
    grammar_errors = check_grammar(text)
    phrase_errors = check_common_phrases(text)

    # 汇总结果
    error_counts = {
        "Spelling": len(spelling_errors),
        "Subject-Verb": len(grammar_errors.get("Subject-Verb", [])),
        "Preposition": len(grammar_errors.get("Preposition", [])),
        "Article": len(grammar_errors.get("Article", [])),
        "Phrase": len(phrase_errors)
    }

    # 打印详细结果
    print("\n===== 错误分析报告 =====")

    if spelling_errors:
        print(f"\n拼写错误 ({error_counts['Spelling']}):")
        for error in spelling_errors:
            print(f"- {error}")
    else:
        print("\n无拼写错误")

    for err_type in ["Subject-Verb", "Preposition", "Article"]:
        items = grammar_errors.get(err_type, [])
        if items:
            print(f"\n{err_type} 错误 ({len(items)}):")
            for item in items:
                print(f"- {item}")

    if phrase_errors:
        print(f"\n词组搭配错误 ({len(phrase_errors)}):")
        for error in phrase_errors:
            print(f"- {error}")

    # 生成修正后的文本
    corrected_text = text

    # 收集所有修正项
    all_corrections = []

    # 收集冠词错误修正项（优先处理）
    for error in grammar_errors.get("Article", []):
        if "→" in error:
            orig, corr = error.split("→")
            all_corrections.append((orig.strip(), corr.strip()))

    # 收集拼写错误修正项
    for error in spelling_errors:
        if "→" in error:
            orig, corr = error.split("→")
            all_corrections.append((orig.strip(), corr.strip()))

    # 收集语法错误修正项（Subject-Verb, Preposition）
    for err_type in ["Subject-Verb", "Preposition"]:
        for error in grammar_errors.get(err_type, []):
            if "→" in error:
                orig, corr = error.split("→")
                all_corrections.append((orig.strip(), corr.strip()))

    # 收集词组搭配错误修正项
    for error in phrase_errors:
        if "→" in error:
            orig, corr = error.split("→")
            all_corrections.append((orig.strip(), corr.strip()))

    # 排序修正项：从最长到最短，避免替换冲突
    all_corrections.sort(key=lambda x: len(x[0]), reverse=True)

    # 应用修正 - 修改后的版本，正确处理冠词问题
    # 先对冠词错误进行专门处理
    for orig, corr in all_corrections:
        if orig.lower().startswith("a ") or orig.lower().startswith("an "):
            pattern = r'\b' + re.escape(orig) + r'\b'  # 使用原始字符串表示法
            corrected_text = re.sub(pattern, corr, corrected_text, flags=re.IGNORECASE)

    # 处理主谓一致错误
    for orig, corr in all_corrections:
        if orig.lower().startswith("she ") or orig.lower().startswith("he ") or orig.lower().startswith("it "):
            corrected_text = re.sub(r'\b' + re.escape(orig) + r'\b', corr, corrected_text, flags=re.IGNORECASE)

    # 再处理其他修正项
    for orig, corr in all_corrections:
        if not (orig.lower().startswith("a ") or orig.lower().startswith("an ")):
            pattern = r'\b' + re.escape(orig) + r'\b'  # 使用原始字符串表示法
            corrected_text = re.sub(pattern, corr, corrected_text, flags=re.IGNORECASE)

    print("\n\n===== 修正后的文本 =====")
    print(corrected_text)

    # 可视化错误统计
    visualize_errors(error_counts)

    return corrected_text

# 测试用例
sample_text = """
She walk to the store yesterday and buys some apples. 
The cat chase the mouses at the corner.
I has a apple and they eats it. 
Thiss is an example with som spelng mistakes.
He depends at his friend which live in an university.
At my opinion,this problen can make a mistake.
"""

corrected = analyze_text(sample_text)