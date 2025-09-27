"""HMM 参数与加载/保存逻辑。

约定：全部使用 log 概率；缺失时返回极小值 `NEG_INF`。
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Dict

NEG_INF = -1e9

@dataclass
class HMMParams:
    init_log_probs: Dict[str, float]          # P(c0)
    trans_log_probs: Dict[str, Dict[str, float]]  # P(c_i | c_{i-1})
    emit_log_probs: Dict[str, Dict[str, float]]   # P(pinyin | char)

    def get_init(self, ch: str) -> float:
        return self.init_log_probs.get(ch, NEG_INF)

    def get_trans(self, prev_ch: str, ch: str) -> float:
        return self.trans_log_probs.get(prev_ch, {}).get(ch, NEG_INF)

    def get_emit(self, ch: str, py: str) -> float:
        return self.emit_log_probs.get(ch, {}).get(py, NEG_INF)

    @staticmethod
    def from_frequency(unigram_path: Path, bigram_path: Path, emit_path: Path, add_k: float = 1e-6) -> 'HMMParams':
        with unigram_path.open('r', encoding='utf-8') as f:
            uni_obj = json.load(f)
        with bigram_path.open('r', encoding='utf-8') as f:
            bi_obj = json.load(f)
        with emit_path.open('r', encoding='utf-8') as f:
            emit_counts = json.load(f)

        # 解析 counter 格式
        uni_counts = uni_obj['data'] if '__type__' in uni_obj else uni_obj
        bi_counts = bi_obj['data'] if '__type__' in bi_obj else bi_obj

        # 初始化概率：按 unigram 频次归一化
        total_uni = sum(uni_counts.values())
        init_log = {ch: math.log(cnt / total_uni) for ch, cnt in uni_counts.items() if cnt > 0}

        # 转移概率：bigram((a,b))/unigram(a) + 平滑
        # bi_counts key 为 "a|b"
        trans_tmp = {}
        for k, cnt in bi_counts.items():
            a, b = k.split('|')
            trans_tmp.setdefault(a, {})[b] = cnt
        trans_log = {}
        for a, next_map in trans_tmp.items():
            base = uni_counts.get(a, 0)
            vocab_size = len(uni_counts)
            trans_log[a] = {}
            for b, cnt in next_map.items():
                prob = (cnt + add_k) / (base + add_k * vocab_size) if base > 0 else add_k / (add_k * vocab_size)
                trans_log[a][b] = math.log(prob)

        # 发射概率
        emit_log = {}
        for ch, py_counts in emit_counts.items():
            total = sum(py_counts.values())
            if total <= 0:
                continue
            emit_log[ch] = {}
            vocab_size = len(py_counts)
            for py, c in py_counts.items():
                prob = (c + add_k) / (total + add_k * vocab_size)
                emit_log[ch][py] = math.log(prob)

        return HMMParams(init_log, trans_log, emit_log)

    def save(self, path: Path):
        obj = {
            'init': self.init_log_probs,
            'trans': self.trans_log_probs,
            'emit': self.emit_log_probs,
        }
        with path.open('w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False)

    @staticmethod
    def load(path: Path) -> 'HMMParams':
        with path.open('r', encoding='utf-8') as f:
            obj = json.load(f)
        return HMMParams(obj['init'], obj['trans'], obj['emit'])
