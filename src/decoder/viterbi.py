"""Viterbi 解码 (Top-1) 与简易 Beam/Top-K 扩展。"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Mapping
import math
from heapq import nlargest

NEG_INF = -1e9

class HMMLike:
    def get_init(self, ch: str) -> float: ...
    def get_trans(self, prev_ch: str, ch: str) -> float: ...
    def get_emit(self, ch: str, py: str) -> float: ...


def viterbi_decode(
    pinyin_seq: List[str],
    candidates_map: Dict[str, List[str]],
    hmm: HMMLike,
    bigram_bonus: Optional[Mapping[str, float]] = None,
) -> str:
    if not pinyin_seq:
        return ''
    # dp[t][char] = (score, prev_char)
    dp: List[Dict[str, Tuple[float, str | None]]] = []

    first_py = pinyin_seq[0]
    first_cands = candidates_map.get(first_py, [])
    layer0 = {}
    for ch in first_cands:
        score = hmm.get_init(ch) + hmm.get_emit(ch, first_py)
        layer0[ch] = (score, None)
    dp.append(layer0)

    for t in range(1, len(pinyin_seq)):
        py = pinyin_seq[t]
        cands = candidates_map.get(py, [])
        layer = {}
        prev_layer = dp[-1]
        if not cands:
            # 若无候选，直接复制占位（也可抛错）
            dp.append({})
            continue
        for ch in cands:
            best_score = -math.inf
            best_prev = None
            emit = hmm.get_emit(ch, py)
            for prev_ch, (prev_score, _) in prev_layer.items():
                trans = hmm.get_trans(prev_ch, ch)
                bonus = 0.0
                if bigram_bonus:
                    pair = prev_ch + ch
                    bonus = bigram_bonus.get(pair, 0.0)
                s = prev_score + trans + emit
                s += bonus
                if s > best_score:
                    best_score = s
                    best_prev = prev_ch
            layer[ch] = (best_score, best_prev)
        dp.append(layer)

    # 回溯
    last_layer = dp[-1]
    if not last_layer:
        return ''
    best_last = max(last_layer.items(), key=lambda x: x[1][0])[0]
    chars = [best_last]
    prev = last_layer[best_last][1]
    for layer in reversed(dp[:-1]):
        if prev is None:
            break
        chars.append(prev)
        prev = layer[prev][1]
    return ''.join(reversed(chars))

def viterbi_topk(
    pinyin_seq: List[str],
    candidates_map: Dict[str, List[str]],
    hmm: HMMLike,
    k: int = 5,
    beam_size: Optional[int] = None,
    bigram_bonus: Optional[Mapping[str, float]] = None,
) -> List[Tuple[str, float]]:
    """返回 Top-K 序列 (string, log_prob)。

    使用 Beam Search 近似：
      - 每步保留 beam_size(默认与 k 相同) 条路径。
      - 路径表示 (score, sequence, last_char)。
    若需要精确 k-best，可改为对完整 DP 图进行 k-best 回溯；当前实现权衡简洁与实用性。
    """
    if not pinyin_seq:
        return []
    if beam_size is None:
        beam_size = k

    # 初始化
    first_py = pinyin_seq[0]
    first_cands = candidates_map.get(first_py, [])
    beam: List[Tuple[float, List[str], str]] = []  # (score, seq_chars, last_char)
    for ch in first_cands:
        score = hmm.get_init(ch) + hmm.get_emit(ch, first_py)
        beam.append((score, [ch], ch))
    beam.sort(key=lambda x: x[0], reverse=True)
    beam = beam[:beam_size]

    # 迭代
    for py in pinyin_seq[1:]:
        cands = candidates_map.get(py, [])
        if not cands:
            continue  # 跳过无候选拼音
        new_beam: List[Tuple[float, List[str], str]] = []
        for score, seq_chars, last_char in beam:
            for ch in cands:
                emit = hmm.get_emit(ch, py)
                trans = hmm.get_trans(last_char, ch)
                bonus = 0.0
                if bigram_bonus and last_char:
                    pair = last_char + ch
                    bonus = bigram_bonus.get(pair, 0.0)
                new_score = score + trans + emit + bonus
                new_beam.append((new_score, seq_chars + [ch], ch))
        if not new_beam:
            break
        # 选择前 beam_size
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_size]

    # 输出前 k
    results = [("".join(seq), sc) for sc, seq, _ in beam]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]