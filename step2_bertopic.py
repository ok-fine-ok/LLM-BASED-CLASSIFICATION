# -*- coding: utf-8 -*-
"""
Step 2: BERTopic Exploratory Semantic Clustering
==============================================
Purpose: Help understand corpus structure, assist taxonomy development.

Random sampling: default 5000 articles (configurable via SAMPLE_SIZE).

As exploratory semantic clustering:
- Does NOT pursue optimal topic coherence
- Does NOT pursue optimal topic count
- Does NOT feed into the final mixed model

Final classification is decided by step3 LLM taxonomy.

Key design choices:
  1. Uses title + first 500 chars (not full text) to avoid noise from body text
  2. Embedding model: BAAI/bge-base-zh-v1.5
  3. min_topic_size: 50 (preserve fine-grained topics)
  4. Includes Chinese stopwords (prevent meaningless words from dominating)
  5. KeyBERTInspired representation (improve keyword interpretability)
  6. Output: Topic Summary Table with representative docs
  7. Does NOT write back to Excel as final topic column
"""
import pandas as pd
import numpy as np
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# ==================== Configuration ====================
EXCEL_PATH = r'.xlsx'
RESULTS_DIR = r''

SAMPLE_SIZE = 5000        # Random sample size (0 = use full corpus)
RANDOM_SEED = 42           # For reproducibility
MAX_TEXT_LEN = 500        # First 500 characters per article
MIN_TOPIC_SIZE = 50        # Minimum docs per topic
TOP_N_WORDS = 12          # Keywords per topic
EMBEDDING_MODEL = 'BAAI/bge-base-zh-v1.5'

print("=" * 70)
print("Step 1: BERTopic Exploratory Semantic Clustering")
print("=" * 70)

# ==================== 1. Load Data ====================
print("\nLoading data...")
df = pd.read_excel(EXCEL_PATH)
print(f"Total articles: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# ==================== 1a. Random Sampling ====================
if SAMPLE_SIZE > 0 and len(df) > SAMPLE_SIZE:
    print(f"\nSampling: {len(df)} -> {SAMPLE_SIZE} (seed={RANDOM_SEED})...")
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Sampled: {len(df)} articles")
elif SAMPLE_SIZE > 0:
    print(f"\nData ({len(df)}) < target ({SAMPLE_SIZE}), using full corpus")
else:
    print("\nUsing full corpus")

# Construct analysis text: title + first 500 chars of body (avoid noise)
text_col = df['text'].astype(str)

if 'title' in df.columns:
    title_col = df['title'].fillna('')
    df['_analysis_text'] = (
        title_col.str.strip() + ' ' + text_col.str[:MAX_TEXT_LEN]
    ).str.strip()
    print(f"Using fields: title + text[:{MAX_TEXT_LEN}]")
else:
    df['_analysis_text'] = text_col.str[:MAX_TEXT_LEN]
    print(f"Using fields: text[:{MAX_TEXT_LEN}] (no title column)")

texts = df['_analysis_text'].tolist()
print(f"Sample text:\n  [{texts[0][:100]}...]")

# ==================== 2. Check Dependencies ====================
print("\nChecking dependencies...")
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from bertopic.representation import KeyBERTInspired
    print("Dependencies OK")
except ImportError:
    print("Installing bertopic and dependencies...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "bertopic", "sentence-transformers",
        "umap-learn", "hdbscan", "sklearn", "-q"
    ])
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from bertopic.representation import KeyBERTInspired
    print("Installation complete")

# ==================== 3. Chinese Stopwords ====================
print("\nLoading Chinese stopwords...")
CN_STOPWORDS = [
    # General
    "的", "了", "是", "在", "和", "与", "或", "及", "等", "对", "于", "这", "那",
    "有", "没有", "可以", "能够", "会", "能", "将", "要", "就", "都",
    "也", "很", "还", "而", "而且", "并", "并且", "但", "但是",
    "如果", "因为", "所以", "因此", "虽然", "然而", "不过", "只是", "或者",
    "这个", "那个", "一个", "一些", "每个", "各个", "自己", "我们", "他们",
    # WeChat public account noise
    "关注", "公众号", "转发", "分享", "点赞", "在看", "收藏", "阅读", "原文",
    "关注我们", "点击上方", "识别二维码", "扫描二维码", "长按", "识别",
    "预约", "电话", "地址", "门诊", "专家", "科室", "医院", "医生",
    "健康", "养生", "科普", "知识", "内容", "文章", "推送", "读者",
    "建议", "仅供参考", "如有", "如有需要", "请联系", "授权", "声明",
    "免责声明", "平台", "转载", "来源", "来源：", "出品", "编辑",
    # Common verbs/adjectives
    "出现", "发生", "引起", "导致", "造成", "进行", "使用", "采用",
    "通过", "相关", "研究", "表明", "显示", "发现", "认为",
    "情况", "问题", "方面", "因素", "作用", "效果", "方法", "方式",
    "时间", "过程", "结果", "目的", "意义", "价值", "内容", "条件",
    "注意", "需要", "应该", "必须", "可能", "主要", "重要", "一般",
    "不同", "各种", "各类", "其中", "其他", "以上", "以下",
    "以及", "包括", "除了", "关于", "对于", "经过",
    # Non-informative high-frequency words
    "什么", "怎么", "如何", "为什么", "多少", "哪些", "哪个", "是否",
]
print(f"  Stopwords count: {len(CN_STOPWORDS)}")

# ==================== 4. Build Model ====================
print(f"\nLoading embedding model: {EMBEDDING_MODEL} ...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("  Loaded")

print("\nBuilding BERTopic (KeyBERTInspired + Chinese stopwords)...")
representation_model = KeyBERTInspired()
vectorizer_model = CountVectorizer(stop_words=CN_STOPWORDS)

topic_model = BERTopic(
    embedding_model=embedding_model,
    representation_model=representation_model,
    language="chinese",
    calculate_probabilities=True,
    verbose=True,
    nr_topics="auto",
    min_topic_size=MIN_TOPIC_SIZE,
    top_n_words=TOP_N_WORDS,
    vectorizer_model=vectorizer_model,
)

# ==================== 5. Train ====================
print(f"\n{'='*70}")
print(f"Running BERTopic (min_topic_size={MIN_TOPIC_SIZE})...")
print(f"{'='*70}")

topics, probs = topic_model.fit_transform(texts)
print(f"\nDone!")

# ==================== 6. Topic Overview ====================
topic_info = topic_model.get_topic_info()
n_real_topics = len(topic_info) - 1
print(f"\nFound {n_real_topics} topics (+ {topic_info[topic_info['Topic']==-1]['Count'].values[0]} outliers)")

# ==================== 7. Topic Summary Table ====================
print(f"\n{'='*70}")
print("Topic Summary Table")
print("(Keywords + Representative docs + Interpretation hints)")
print(f"{'='*70}")

summary_rows = []

for _, row in topic_info.iterrows():
    topic_id = row['Topic']
    count = row['Count']

    if topic_id == -1:
        continue

    keywords_raw = topic_model.get_topic(topic_id)
    if keywords_raw:
        top_kw = [w for w, _ in keywords_raw[:10]]
        keywords_str = " / ".join(top_kw)
    else:
        keywords_str = "(no keywords)"

    try:
        rep_docs = topic_model.get_representative_docs(topic_id)
        if rep_docs:
            rep_doc = rep_docs[0][:200].strip()
        else:
            rep_doc = "(no representative doc)"
    except Exception:
        rep_doc = "(unavailable)"

    print(f"\n┌──────────────────────────────────────────────────────────")
    print(f"│ Topic {topic_id:3d}  |  Docs: {count:5d}  ({count/len(texts)*100:.1f}%)")
    print(f"├──────────────────────────────────────────────────────────")
    print(f"│ Keywords: {keywords_str}")
    print(f"├──────────────────────────────────────────────────────────")
    print(f"│ Representative: {rep_doc}...")
    print(f"└──────────────────────────────────────────────────────────")

    summary_rows.append({
        "topic_id": topic_id,
        "doc_count": count,
        "pct": f"{count/len(texts)*100:.1f}%",
        "keywords": keywords_str,
        "representative_doc": rep_doc,
        "interpretation": "",
    })

# ==================== 8. Save Topic Summary Table ====================
summary_df = pd.DataFrame(summary_rows)
summary_path = f"{RESULTS_DIR}\\bertopic_topic_summary.xlsx"
summary_df.to_excel(summary_path, index=False)
print(f"\nTopic Summary Table saved: {summary_path}")

# ==================== 9. Save Doc-Topic Assignments ====================
assign_df = df[['_analysis_text']].copy()
assign_df['bertopic_topic'] = topics
assign_df['bertopic_prob'] = [
    float(p.max()) if (isinstance(p, np.ndarray) and p.size > 0) else 0.0
    for p in probs
]
assign_path = f"{RESULTS_DIR}\\bertopic_assignments.xlsx"
assign_df.to_excel(assign_path, index=False)
print(f"Doc-topic assignments saved: {assign_path}")

bertopic_res_path = f"{RESULTS_DIR}\\bertopic_results.xlsx"
topic_info.to_excel(bertopic_res_path, index=False)
print(f"BERTopic results saved: {bertopic_res_path}")

# ==================== 10. Outlier Analysis ====================
outlier_mask = np.array(topics) == -1
outlier_count = outlier_mask.sum()
print(f"\nOutliers: {outlier_count} ({outlier_count/len(texts)*100:.1f}%)")
if outlier_count > 0 and outlier_count <= 20:
    print("Outlier examples:")
    for i, (orig_text, is_outlier) in enumerate(zip(texts, outlier_mask)):
        if is_outlier:
            print(f"  [{i}] {orig_text[:100]}...")
            if i - list(outlier_mask).index(True) >= 5:
                break

# ==================== 11. Tiered Topic Overview ====================
print(f"\n{'='*70}")
print("Topic Size Tiers (for taxonomy mapping)")
print(f"{'='*70}")

for label, lower, upper in [
    ("Large topics (>=500)", 500, 999999),
    ("Medium topics (100-499)", 100, 499),
    ("Small topics (50-99)", MIN_TOPIC_SIZE, 99),
]:
    subset = summary_df[
        (summary_df['doc_count'] >= lower) &
        (summary_df['doc_count'] <= upper)
    ]
    if subset.empty:
        continue
    print(f"\n{label}: {len(subset)} topics")
    for _, r in subset.iterrows():
        print(f"  T{r['topic_id']:3d} | {r['doc_count']:5d} ({r['pct']:5s}) | {r['keywords'][:80]}")

print(f"\n{'='*70}")
print("Step 1 Complete!")
print("Fill in the 'interpretation' column in the Topic Summary Table,")
print("then use step3 for final LLM-based classification.")
print(f"{'='*70}")
