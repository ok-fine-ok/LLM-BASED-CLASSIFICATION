# -*- coding: utf-8 -*-
"""
Step 4: Framing Type Classification
====================================
Uses DeepSeek LLM to classify the framing type of health science articles.

Framing types:
1. Informational framing - objective knowledge explanation, mechanism description
2. Behavioral guidance framing - telling readers what to do, action recommendations
3. Warning/risk framing - emphasizing consequences, risks, danger signals
4. Narrative framing - patient stories, clinical cases, situational expression
5. Emotional reassurance framing - reducing anxiety, empathy, psychological support

Configuration:
  - DEEPSEEK_API_KEY (env var)
  - EXCEL_PATH (data file path)
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import sys
import io
import os
import time
import threading
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("Step 4: Framing Type Classification")
print("=" * 70)

# ==================== Configuration ====================
EXCEL_PATH = r'.xlsx'

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4-Flash")

if DEEPSEEK_API_KEY:
    print(f"DEEPSEEK_API_KEY detected: {DEEPSEEK_API_KEY[:10]}...")
else:
    print("WARNING: DEEPSEEK_API_KEY not set.")

from openai import OpenAI
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.siliconflow.cn/v1"
)
print(f"Using model: {DEEPSEEK_MODEL}")

# ==================== Framing Types ====================
FRAMING_TYPES = {
    "Informational framing": {
        "description": "知识解释型：客观介绍医疗健康知识，解释疾病原理机制，不强调行动或风险",
        "examples": "什么是脂肪肝、高血压的危险因素、糖尿病的发病机制、疫苗的工作原理"
    },
    "Behavioral guidance framing": {
        "description": "行动指导型：明确告诉读者该怎么做，强调具体行为建议和操作步骤",
        "examples": "3个方法控制血糖、夏天这样喝水更健康、每天走多少步最健康"
    },
    "Warning/risk framing": {
        "description": "风险警示型：强调疾病后果、健康风险、危险信号、警示性内容",
        "examples": "这种症状可能是癌症、长期熬夜会增加猝死风险、身体出现这些信号要警惕"
    },
    "Narrative framing": {
        "description": "叙事/案例型：以患者故事、临床案例、具体情境来切入，情境化表达",
        "examples": "42岁男子体检发现肝癌、一位糖尿病患者的经历、门诊故事"
    },
    "Emotional reassurance framing": {
        "description": "情绪安抚型：旨在减轻焦虑、共情表达、提供心理支持",
        "examples": "感染后不要过度紧张、焦虑时可以这样做、别怕这只是小问题"
    }
}

FRAMING_PROMPT_TEMPLATE = """你是一个医疗科普内容框架分析专家。请分析以下文本，判断它属于哪种框架类型。

文本内容：
{text}

可选框架类型：
1. Informational framing (知识解释型)：客观介绍医疗健康知识，解释疾病原理机制，不强调行动或风险
   例子：什么是脂肪肝、高血压的危险因素、糖尿病的发病机制
2. Behavioral guidance framing (行动指导型)：明确告诉读者该怎么做，强调具体行为建议和操作步骤
   例子：3个方法控制血糖、夏天这样喝水更健康、每天走多少步最健康
3. Warning/risk framing (风险警示型)：强调疾病后果、健康风险、危险信号、警示性内容
   例子：这种症状可能是癌症、长期熬夜会增加猝死风险、身体出现这些信号要警惕
4. Narrative framing (叙事/案例型)：以患者故事、临床案例、具体情境来切入，情境化表达
   例子：42岁男子体检发现肝癌、一位糖尿病患者的经历、门诊故事
5. Emotional reassurance framing (情绪安抚型)：旨在减轻焦虑、共情表达、提供心理支持
   例子：感染后不要过度紧张、焦虑时可以这样做、别怕这只是小问题

请仔细分析文本的写作风格和目的，选择最匹配的框架类型。

输出格式：只需输出一个框架类型名称，精确匹配以下五个之一：
- Informational framing
- Behavioral guidance framing
- Warning/risk framing
- Narrative framing
- Emotional reassurance framing"""


def classify_framing_by_api(text, client, max_retries=5):
    """Classify framing type using DeepSeek API with exponential backoff."""
    prompt = FRAMING_PROMPT_TEMPLATE.format(text=text[:1500])

    valid_types = list(FRAMING_TYPES.keys())

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个医疗科普内容框架分析专家，只输出框架类型名称。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            result = response.choices[0].message.content.strip()

            for ft in valid_types:
                if ft in result or result in ft:
                    return ft

            result_lower = result.lower()
            if 'informational' in result_lower or '知识' in result:
                return "Informational framing"
            if 'behavioral' in result_lower or 'guidance' in result_lower or '行动' in result or '指导' in result:
                return "Behavioral guidance framing"
            if 'warning' in result_lower or 'risk' in result_lower or '警示' in result or '风险' in result:
                return "Warning/risk framing"
            if 'narrative' in result_lower or '叙事' in result or '案例' in result or '故事' in result:
                return "Narrative framing"
            if 'emotional' in result_lower or 'reassurance' in result_lower or '情绪' in result or '安抚' in result:
                return "Emotional reassurance framing"

            return "Informational framing"
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) * 0.5
                time.sleep(wait)
            else:
                print(f"  API error (max retries): {e}")
                return "Informational framing"


# ==================== Main ====================
print("\nLoading data...")
df = pd.read_excel(EXCEL_PATH)
print(f"Data: {len(df)} articles")
print(f"Columns: {df.columns.tolist()}")

if 'Framing' not in df.columns:
    df['Framing'] = None
    print("Created 'Framing' column")

has_framing = df['Framing'].notna() & (df['Framing'] != '')
done_framing = has_framing.sum()
print(f"Already classified: {done_framing}")

need_framing = df['Framing'].isna() | (df['Framing'] == '')
total_framing = need_framing.sum()
print(f"Need to process: {total_framing}")

if total_framing == 0:
    print("\nAll articles classified. Saving...")
    df.to_excel(EXCEL_PATH, index=False)
    print(f"Saved to: {EXCEL_PATH}")
    sys.exit(0)

if total_framing > 0:
    print(f"\n{'='*70}")
    print(f"Starting Framing classification: {total_framing} articles")
    print(f"{'='*70}")
    print(f"Mode: DeepSeek API")
    print()

    tasks = [(idx, str(row.get('text', ''))) for idx, row in df[need_framing].iterrows()]
    total_tasks = len(tasks)

    framing_results = []
    results_lock = threading.Lock()
    counter_lock = threading.Lock()
    completed_count = [0]

    def process_one(task):
        idx, text = task
        try:
            framing = classify_framing_by_api(text, client)
            return idx, framing, None
        except Exception as e:
            return idx, "Informational framing", str(e)

    def process_batch(batch):
        with results_lock:
            for idx, framing, err in batch:
                df.at[idx, 'Framing'] = framing
                with counter_lock:
                    completed_count[0] += 1
                if err:
                    print(f"  [Error] idx={idx}: {err}")
            if completed_count[0] % save_interval == 0:
                try:
                    df.to_excel(EXCEL_PATH, index=False)
                    print(f"  [Auto-saved] {completed_count[0]} articles")
                except Exception as save_err:
                    print(f"  [Save failed] {save_err}")

    save_interval = 1000
    batch_size = 100
    pending_write = []
    max_workers = 50

    print(f"Launching {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_one, t): t for t in tasks}

        for future in as_completed(future_to_task):
            idx, framing, err = future.result()
            pending_write.append((idx, framing, err))

            if len(pending_write) >= batch_size:
                process_batch(pending_write)
                pending_write = []

            with counter_lock:
                done = completed_count[0]
            if done % 50 == 0 or done == total_tasks:
                pct = done / total_tasks * 100
                print(f"  Progress: {done}/{total_tasks} ({pct:.1f}%)")

    if pending_write:
        process_batch(pending_write)

    for idx, row in df[need_framing].iterrows():
        framing_results.append(df.at[idx, 'Framing'])

    print(f"\n{'='*70}")
    print("Framing Results")
    print(f"{'='*70}")
    framing_counts = pd.Series(framing_results).value_counts()
    for ft, cnt in framing_counts.items():
        pct = cnt / len(framing_results) * 100
        print(f"  {ft}: {cnt} ({pct:.1f}%)")

# ==================== Save ====================
print(f"\n{'='*70}")
print("Saving")
print(f"{'='*70}")

df.to_excel(EXCEL_PATH, index=False)
print(f"Saved to: {EXCEL_PATH}")
print(f"  Framing column: {'updated' if total_framing > 0 else 'no change'}")

print(f"\n{'='*70}")
print("Step 4 Complete!")
print(f"{'='*70}")
