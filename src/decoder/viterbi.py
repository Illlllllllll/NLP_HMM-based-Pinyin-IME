"""Viterbi 解码 (Top-1) 与简易 Beam/Top-K 扩展，以及增量式解码器。"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Mapping
import math
from heapq import nlargest

NEG_INF = -1e9

class HMMLike:
    def get_init(self, ch: str) -> float: ...
    def get_trans(self, prev_ch: str, ch: str) -> float: ...
    def get_emit(self, ch: str, py: str) -> float: ...


class IncrementalViterbi:
    """增量式 Viterbi 解码器，支持逐个拼音输入并实时返回候选结果。
    
    核心特性：
    - 保持历史状态，避免重复计算
    - 支持增加拼音（add_pinyin）
    - 支持删除最后一个拼音（delete_last）
    - 支持前缀预测（add_pinyin_prefix）
    - 使用 Beam Search 控制状态空间大小
    """
    
    def __init__(
        self,
        candidates_map: Dict[str, List[str]],
        hmm: HMMLike,
        beam_size: int = 100,
        bigram_bonus: Optional[Mapping[str, float]] = None,
    ):
        self.candidates_map = candidates_map
        self.hmm = hmm
        self.beam_size = beam_size
        self.bigram_bonus = bigram_bonus
        
        # 历史状态
        self.pinyin_buffer: List[str] = []
        # dp_states[t] = {char: (score, prev_char)}
        self.dp_states: List[Dict[str, Tuple[float, Optional[str]]]] = []
    
    def reset(self):
        """重置解码器状态"""
        self.pinyin_buffer.clear()
        self.dp_states.clear()
    
    def add_pinyin(self, pinyin: str) -> List[Tuple[str, float]]:
        """添加一个完整拼音，返回当前 Top-K 候选序列。
        
        Args:
            pinyin: 完整的拼音字符串（如 "ni", "hao"）
            
        Returns:
            候选列表 [(sequence, score), ...]，按分数降序排列
        """
        if not pinyin:
            return self.get_topk_sequences(5)
        
        self.pinyin_buffer.append(pinyin)
        t = len(self.pinyin_buffer) - 1
        
        # 获取当前拼音的候选字
        candidates = self.candidates_map.get(pinyin, [])
        if not candidates:
            # 无候选时，保持前一状态
            self.dp_states.append({})
            return self.get_topk_sequences(5)
        
        if t == 0:
            # 初始状态
            dp_t = {}
            for char in candidates:
                init_prob = self.hmm.get_init(char)
                emit_prob = self.hmm.get_emit(char, pinyin)
                score = init_prob + emit_prob
                dp_t[char] = (score, None)
            self.dp_states.append(dp_t)
        else:
            # 动态规划扩展
            prev_layer = self.dp_states[t - 1]
            if not prev_layer:
                # 前一层为空，重新初始化
                dp_t = {}
                for char in candidates:
                    init_prob = self.hmm.get_init(char)
                    emit_prob = self.hmm.get_emit(char, pinyin)
                    score = init_prob + emit_prob
                    dp_t[char] = (score, None)
                self.dp_states.append(dp_t)
            else:
                dp_t = {}
                for curr_char in candidates:
                    best_score = -math.inf
                    best_prev = None
                    emit_prob = self.hmm.get_emit(curr_char, pinyin)
                    
                    for prev_char, (prev_score, _) in prev_layer.items():
                        trans_prob = self.hmm.get_trans(prev_char, curr_char)
                        bonus = 0.0
                        if self.bigram_bonus and prev_char:
                            pair = prev_char + curr_char
                            bonus = self.bigram_bonus.get(pair, 0.0)
                        score = prev_score + trans_prob + emit_prob + bonus
                        
                        if score > best_score:
                            best_score = score
                            best_prev = prev_char
                    
                    dp_t[curr_char] = (best_score, best_prev)
                
                # Beam pruning：只保留最好的 beam_size 个状态
                if len(dp_t) > self.beam_size:
                    sorted_states = sorted(dp_t.items(), key=lambda x: x[1][0], reverse=True)
                    dp_t = dict(sorted_states[:self.beam_size])
                
                self.dp_states.append(dp_t)
        
        return self.get_topk_sequences(5)
    
    def delete_last(self) -> List[Tuple[str, float]]:
        """删除最后一个拼音（退格功能）"""
        if self.pinyin_buffer:
            self.pinyin_buffer.pop()
            self.dp_states.pop()
        return self.get_topk_sequences(5) if self.dp_states else []
    
    def add_pinyin_prefix(self, prefix: str) -> List[Tuple[str, float]]:
        """添加拼音前缀（未完成的拼音），返回可能的候选。
        
        Args:
            prefix: 拼音前缀（如 "n", "ha"）
            
        Returns:
            基于前缀可能完成的候选序列
        """
        if not prefix:
            return self.get_topk_sequences(5)
        
        # 找到所有匹配前缀的完整拼音
        matching_pinyins = [py for py in self.candidates_map.keys() if py.startswith(prefix)]
        if not matching_pinyins:
            return self.get_topk_sequences(5)
        
        # 对每个匹配的拼音，计算其候选
        all_candidates = []
        for pinyin in matching_pinyins[:10]:  # 限制匹配数量
            # 临时添加这个拼音
            temp_result = self._predict_with_pinyin(pinyin)
            all_candidates.extend(temp_result)
        
        # 去重并排序
        unique_candidates = {}
        for seq, score in all_candidates:
            if seq not in unique_candidates or score > unique_candidates[seq]:
                unique_candidates[seq] = score
        
        sorted_candidates = sorted(unique_candidates.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:5]
    
    def _predict_with_pinyin(self, pinyin: str) -> List[Tuple[str, float]]:
        """临时预测添加某个拼音后的结果（不修改状态）"""
        candidates = self.candidates_map.get(pinyin, [])
        if not candidates:
            return []
        
        if not self.dp_states:
            # 初始状态
            results = []
            for char in candidates:
                init_prob = self.hmm.get_init(char)
                emit_prob = self.hmm.get_emit(char, pinyin)
                score = init_prob + emit_prob
                results.append((char, score))
            return sorted(results, key=lambda x: x[1], reverse=True)[:5]
        
        # 基于最后一层状态扩展
        prev_layer = self.dp_states[-1]
        if not prev_layer:
            return []
        
        next_scores = {}
        for curr_char in candidates:
            best_score = -math.inf
            emit_prob = self.hmm.get_emit(curr_char, pinyin)
            
            for prev_char, (prev_score, _) in prev_layer.items():
                trans_prob = self.hmm.get_trans(prev_char, curr_char)
                bonus = 0.0
                if self.bigram_bonus and prev_char:
                    pair = prev_char + curr_char
                    bonus = self.bigram_bonus.get(pair, 0.0)
                score = prev_score + trans_prob + emit_prob + bonus
                best_score = max(best_score, score)
            
            next_scores[curr_char] = best_score
        
        # 回溯构建完整序列
        results = []
        for last_char, last_score in next_scores.items():
            seq = self._backtrack_from(len(self.dp_states) - 1, None) + last_char
            results.append((seq, last_score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:5]
    
    def get_topk_sequences(self, k: int = 5) -> List[Tuple[str, float]]:
        """获取当前最佳的 k 个候选序列。
        
        Returns:
            [(sequence, score), ...] 按分数降序排列
        """
        if not self.dp_states:
            return []
        
        last_layer = self.dp_states[-1]
        if not last_layer:
            return []
        
        # 获取最后一层的 Top-K 字符及其分数
        sorted_last = sorted(last_layer.items(), key=lambda x: x[1][0], reverse=True)
        top_k_last = sorted_last[:k]
        
        # 对每个字符回溯构建完整序列
        results = []
        for last_char, (last_score, _) in top_k_last:
            seq = self._backtrack_from(len(self.dp_states) - 1, last_char)
            results.append((seq, last_score))
        
        return results
    
    def _backtrack_from(self, layer_idx: int, start_char: Optional[str]) -> str:
        """从指定层和字符回溯构建序列。"""
        if layer_idx < 0 or not self.dp_states:
            return ''
        
        chars = []
        curr_char = start_char
        
        for t in range(layer_idx, -1, -1):
            if curr_char is None:
                break
            chars.append(curr_char)
            if t > 0 and curr_char in self.dp_states[t]:
                _, prev_char = self.dp_states[t][curr_char]
                curr_char = prev_char
            else:
                break
        
        return ''.join(reversed(chars))


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