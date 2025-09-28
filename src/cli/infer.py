"""命令行推理入口

示例：
    python -m src.cli.infer "jin tian tian qi" --pinyin-map resources/lexicon_aggregate.json \
      --hmm resources/hmm_params.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

from src.models.hmm import HMMParams
from src.decoder.viterbi import viterbi_decode


def load_pinyin_map(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pinyin', help='空格分隔的拼音序列')
    ap.add_argument('--pinyin-map', required=True, help='可以是 base 映射 JSON 或聚合 lexicon JSON')
    ap.add_argument('--hmm', required=True, help='保存的 HMM 参数 JSON 文件')
    args = ap.parse_args()

    pinyin_seq: List[str] = args.pinyin.strip().split()
    raw_map = load_pinyin_map(Path(args.pinyin_map))
    # 若是聚合 lexicon，取其 base_pinyin_to_chars 字段
    word_bigram_bonus = None
    if 'base_pinyin_to_chars' in raw_map and isinstance(raw_map['base_pinyin_to_chars'], dict):
        pinyin_map = raw_map['base_pinyin_to_chars']
        word_bigram_bonus = raw_map.get('word_bigram_bonus')
    else:
        pinyin_map = raw_map
    hmm = HMMParams.load(Path(args.hmm))

    # 将 bigram 奖励转换为 {pair: bonus}
    bigram_bonus = None
    if isinstance(word_bigram_bonus, dict):
        bigram_bonus = {str(k): float(v) for k, v in word_bigram_bonus.items()}

    result = viterbi_decode(pinyin_seq, pinyin_map, hmm, bigram_bonus=bigram_bonus)
    print(result)

if __name__ == '__main__':
    main()
