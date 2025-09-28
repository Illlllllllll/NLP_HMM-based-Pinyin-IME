"""构建 unigram / bigram / (char,pinyin) 发射统计

使用 PFR 人民日报标注语料：每行若为分词+词性（假设格式： 词/词性 ），将词拆成单字用于字级 HMM。
（不同 PFR 分发版本格式可能不同；此脚本做稳健解析，允许无词性。）

发射计数策略（当前增强版）：
    - 基于提供的 base pinyin -> chars 映射，将每个字出现次数累加到其所有可能 base pinyin 上；
    - 若提供聚合 lexicon JSON（由 load_lexicons.py 生成），优先使用其中的 base_pinyin_to_chars；
    - 使用聚合 JSON 中的 char_frequency（来自 `usrs/count.out` 词频）作为未覆盖汉字的先验补充。

输出：
    resources/freq_unigram.json
    resources/freq_bigram.json
    resources/freq_emit.json  (char -> {pinyin: count})  [可改进为真实对齐统计]

运行示例：
    python -m src.preprocess.build_stats --corpus peopledaily/PeopleDaily199801.txt \
                --lexicon resources/lexicon_aggregate.json
或（兼容旧参数）
    python -m src.preprocess.build_stats --corpus peopledaily/PeopleDaily199801.txt \
                --pinyin-map path/to/custom_pinyin_map.json
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import re

TOKEN_SPLIT_RE = re.compile(r"\s+")
WORD_POS_RE = re.compile(r"^(.+?)/(\w+)$")  # 词/词性
PINYIN_TONE_RE = re.compile(r'([a-z]+)[1-5]$')

def tone_to_base(p: str) -> str:
    m = PINYIN_TONE_RE.match(p)
    return m.group(1) if m else p

def load_base_pinyin_map(pinyin_map_path: Path | None, lexicon_path: Path | None):
    """兼容两种来源：
    1) 单一 pinyin_map.json （无声调）
    2) 聚合 lexicon JSON -> key 'base_pinyin_to_chars'
    优先使用 lexicon_path。
    """
    base_map = {}
    char_freq = {}
    if lexicon_path and lexicon_path.exists():
        with lexicon_path.open('r', encoding='utf-8') as f:
            obj = json.load(f)
        if 'base_pinyin_to_chars' in obj:
            base_map = obj['base_pinyin_to_chars']
        if 'char_frequency' in obj and isinstance(obj['char_frequency'], dict):
            char_freq = {k: int(v) for k, v in obj['char_frequency'].items()}
        return base_map, char_freq
    if pinyin_map_path and pinyin_map_path.exists():
        with pinyin_map_path.open('r', encoding='utf-8') as f:
            base_map = json.load(f)
        return base_map, char_freq
    raise FileNotFoundError("Neither lexicon aggregate nor pinyin_map file found")

def iter_chars_from_corpus(corpus_path: Path):
    with corpus_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 按空白分词
            parts = TOKEN_SPLIT_RE.split(line)
            for tok in parts:
                if not tok:
                    continue
                m = WORD_POS_RE.match(tok)
                word = m.group(1) if m else tok
                # 去除可能的标点
                for ch in word:
                    if '\u4e00' <= ch <= '\u9fff':
                        yield ch

def build_stats(corpus_path: Path):
    unigram = Counter()
    bigram = Counter()
    prev = None
    for ch in iter_chars_from_corpus(corpus_path):
        unigram[ch] += 1
        if prev is not None:
            bigram[(prev, ch)] += 1
        prev = ch
    return unigram, bigram

def attach_pinyin_emission(unigram: Counter, base_map: dict, char_prior: dict | None = None):
    """构造发射计数 (简化版)：
    - 先将语料中的字 unigram 次数复制到其所有候选 base pinyin。
    - 若语料未覆盖某汉字且提供了字频先验（来自 `count.out`），使用 log 平滑后的先验计数填充。
    """
    emit = defaultdict(Counter)
    char_to_pinyin = defaultdict(set)
    for py, chars in base_map.items():
        for ch in chars:
            char_to_pinyin[ch].add(py)
    for ch, cnt in unigram.items():
        for py in char_to_pinyin.get(ch, []):
            emit[ch][py] += cnt
    if char_prior:
        for ch, pys in char_to_pinyin.items():
            if ch in emit:
                continue
            prior = char_prior.get(ch)
            if not prior:
                continue
            smoothed = max(1, int(round(math.log1p(prior))))
            for py in pys:
                emit[ch][py] += smoothed
    
    # 手动提升特定字符的发射概率（如"你"在"ni"上）
    '''
    manual_boosts = {
        ('你', 'ni'): 5000,  # 额外增加计数
        ('的', 'de'): 3000,
        ('了', 'le'): 2000,
        ('是', 'shi'): 2000,
    }
    for (ch, py), boost in manual_boosts.items():
        if ch in emit and py in char_to_pinyin.get(ch, set()):
            emit[ch][py] += boost
    '''
    return emit

def save_counter_json(counter: Counter, path: Path):
    obj = {"__type__": "counter", "data": {"|".join(k) if isinstance(k, tuple) else k: v for k, v in counter.items()}}
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)

def save_emit_json(emit: dict, path: Path):
    out = {ch: dict(c.items()) for ch, c in emit.items()}
    with path.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--pinyin-map', help='旧版无声调拼音映射 JSON')
    group.add_argument('--lexicon', help='由 load_lexicons.py 生成的聚合 JSON')
    ap.add_argument('--out-dir', default='resources')
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    pinyin_map_path = Path(args.pinyin_map) if args.pinyin_map else None
    lexicon_path = Path(args.lexicon) if args.lexicon else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    base_map, char_prior = load_base_pinyin_map(pinyin_map_path, lexicon_path)
    unigram, bigram = build_stats(corpus_path)
    emit = attach_pinyin_emission(unigram, base_map, char_prior)

    save_counter_json(unigram, out_dir / 'freq_unigram.json')
    save_counter_json(bigram, out_dir / 'freq_bigram.json')
    save_emit_json(emit, out_dir / 'freq_emit.json')
    print('Saved stats to', out_dir)

if __name__ == '__main__':
    main()
