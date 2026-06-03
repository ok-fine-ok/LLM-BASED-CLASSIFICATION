#!/usr/bin/env python
# coding: utf-8
"""
Step 1: Identify whether an article is health-related science content (based on title).
Adds a column "是否科普" (is/isn't health science) to the Excel file.

Configuration:
  - Set environment variable DEEPSEEK_API_KEY (do NOT hardcode keys or commit to repo).
  - Configure input/output file paths via command line arguments.
"""
from __future__ import annotations
import argparse
import threading
import pandas as pd
import requests
import time
import tempfile
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== Configuration ==========
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

# Title column candidates (in order of priority)
TITLE_COLUMN_CANDIDATES = ("文章标题", "标题", "图文标题", "text", "笔记标题")

SYSTEM_PROMPT = (
    "你是一个严格的医学类健康类科普分类器，只能输出“是”或“否”。\n\n"
    "【输入说明】\n"
    "用户输入的是文章或视频的「标题」（可能很短），请仅根据标题判断。\n\n"
    "【医学科普定义】\n"
    "医学科普 = 用通俗语言向大众传播医学、健康、疾病、治疗、预防等相关知识；"
    "中医药、节气、养生类健康解释也算科普。\n"
    "标题明显指向健康知识传播（如“甲状腺饮食指南”“甲沟炎如何预防”），算科普 → 是。\n\n"
    "【不算医学科普】\n"
    "新闻报道/政策宣传/节日祝福/门诊通知/直播预告/纯广告营销；"
    "纯娱乐、八卦、与健康知识传播无关 → 否。\n\n"
    "【输出】\n"
    "只输出一个字：是 或 否。\n"
)

MAX_WORKERS = 6
MAX_RETRY = 4
TIMEOUT_S = 25
RESULT_COL = "是否科普"

# Thread-local session storage
_tls = threading.local()


def _thread_session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(
            {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "Connection": "keep-alive",
            }
        )
        _tls.session = s
    return s


def pick_title_column(df: pd.DataFrame) -> str:
    for name in TITLE_COLUMN_CANDIDATES:
        if name in df.columns:
            return name
    if len(df.columns) >= 2:
        c = df.columns[1]
        print(f"提示：未找到标题列名 {TITLE_COLUMN_CANDIDATES}，使用第 2 列「{c}」作为标题。")
        return str(c)
    raise ValueError("表格至少需要 2 列（账号 + 标题），请检查文件。")


def ensure_dir_exists(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def safe_save_excel(df: pd.DataFrame, path: str) -> None:
    ensure_dir_exists(path)
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    with tempfile.NamedTemporaryFile(
        prefix="tmp_", suffix=".xlsx", delete=False, dir=out_dir
    ) as tf:
        tmp_path = tf.name
    try:
        df.to_excel(tmp_path, index=False, engine="openpyxl")
        if os.path.exists(path):
            os.remove(path)
        os.replace(tmp_path, path)
        print("结果已保存到：", path)
    except Exception as e:
        print("保存 Excel 失败：", e)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def classify_one(i: int, txt: str) -> tuple[int, str]:
    session = _thread_session()
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": txt},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    backoff = 0.8
    for _ in range(MAX_RETRY):
        try:
            r = session.post(API_URL, json=payload, timeout=TIMEOUT_S)
            code = r.status_code
            if code == 200:
                try:
                    out = r.json()["choices"][0]["message"]["content"].strip()
                except Exception:
                    out = "否"
                if out not in ("是", "否"):
                    if "否" in out and "是" not in out:
                        out = "否"
                    elif "是" in out and "否" not in out:
                        out = "是"
                    else:
                        out = "否"
                return i, out
            if code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 1.6
                continue
            return i, "否"
        except Exception:
            time.sleep(backoff)
            backoff *= 1.6
    return i, "否"


def run(
    inp: str,
    outp: str,
    *,
    max_workers: int | None = None,
) -> None:
    if not API_KEY:
        raise SystemExit("请设置环境变量 DEEPSEEK_API_KEY。")

    df = pd.read_excel(inp)
    title_col = pick_title_column(df)

    if RESULT_COL in df.columns:
        df = df.drop(columns=[RESULT_COL])

    total_rows = len(df)
    print(f"读取 {total_rows} 行；标题列：「{title_col}」；输出：{outp}")

    titles = df[title_col].astype(str).str.strip()
    results: list[str | None] = [None] * total_rows

    workers = max_workers or MAX_WORKERS
    print("开始调用 API（首批结果取决于网络，通常几秒内会有进度）…")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(classify_one, i, titles.iat[i] if i < len(titles) else "")
            for i in range(total_rows)
        ]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="标题判定",
            ncols=100,
            mininterval=0.5,
        ):
            i, r = fut.result()
            results[i] = r

    df[RESULT_COL] = results
    safe_save_excel(df, outp)


def main() -> None:
    p = argparse.ArgumentParser(description="按标题判断是否科普，结果列加在表末")
    p.add_argument("-i", "--input", default=".xlsx", help="输入 xlsx")
    p.add_argument("-o", "--output", default=".xlsx", help="输出 xlsx")
    p.add_argument("-w", "--workers", type=int, default=MAX_WORKERS, help="并发线程数")
    args = p.parse_args()
    run(args.input, args.output, max_workers=args.workers)


if __name__ == "__main__":
    main()
