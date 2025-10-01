"""测试增量式 Viterbi 解码器"""
from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decoder.viterbi import IncrementalViterbi
from src.models.hmm import HMMParams


def build_toy_hmm() -> HMMParams:
    """构建玩具 HMM 用于测试"""
    init = {
        '你': math.log(0.6),
        '尼': math.log(0.4),
        '好': math.log(0.5),
        '号': math.log(0.5),
    }
    trans = {
        '你': {
            '好': math.log(0.7),
            '号': math.log(0.3),
        },
        '尼': {
            '好': math.log(0.4),
            '号': math.log(0.6),
        },
        '好': {
            '你': math.log(0.5),
            '尼': math.log(0.5),
        },
        '号': {
            '你': math.log(0.5),
            '尼': math.log(0.5),
        },
    }
    emit = {
        '你': {'ni': math.log(1.0)},
        '尼': {'ni': math.log(1.0)},
        '好': {'hao': math.log(1.0)},
        '号': {'hao': math.log(1.0)},
    }
    return HMMParams(init, trans, emit)


def test_incremental_basic():
    """测试基础增量解码"""
    hmm = build_toy_hmm()
    pinyin_map = {
        'ni': ['你', '尼'],
        'hao': ['好', '号'],
    }
    
    decoder = IncrementalViterbi(pinyin_map, hmm, beam_size=10)
    
    # 添加第一个拼音
    candidates = decoder.add_pinyin('ni')
    assert len(candidates) > 0
    # 第一个候选应该是"你"，因为它的初始概率更高
    assert candidates[0][0] == '你'
    
    # 添加第二个拼音
    candidates = decoder.add_pinyin('hao')
    assert len(candidates) > 0
    # 最佳候选应该是"你好"
    assert candidates[0][0] == '你好'


def test_incremental_delete():
    """测试删除功能"""
    hmm = build_toy_hmm()
    pinyin_map = {
        'ni': ['你', '尼'],
        'hao': ['好', '号'],
    }
    
    decoder = IncrementalViterbi(pinyin_map, hmm, beam_size=10)
    
    # 添加两个拼音
    decoder.add_pinyin('ni')
    decoder.add_pinyin('hao')
    candidates = decoder.get_topk_sequences(3)
    assert len(candidates) > 0
    assert candidates[0][0] == '你好'
    
    # 删除最后一个
    candidates = decoder.delete_last()
    assert len(candidates) > 0
    assert candidates[0][0] == '你'
    
    # 再删除
    candidates = decoder.delete_last()
    assert len(candidates) == 0


def test_incremental_prefix():
    """测试拼音前缀预测"""
    hmm = build_toy_hmm()
    pinyin_map = {
        'ni': ['你', '尼'],
        'hao': ['好', '号'],
        'ha': ['哈'],  # 添加一个前缀匹配
    }
    
    decoder = IncrementalViterbi(pinyin_map, hmm, beam_size=10)
    decoder.add_pinyin('ni')
    
    # 使用前缀 "ha" 预测
    candidates = decoder.add_pinyin_prefix('ha')
    assert len(candidates) > 0
    # 应该包含以 "ha" 开头的拼音的候选


def test_incremental_topk():
    """测试 Top-K 功能"""
    hmm = build_toy_hmm()
    pinyin_map = {
        'ni': ['你', '尼'],
        'hao': ['好', '号'],
    }
    
    decoder = IncrementalViterbi(pinyin_map, hmm, beam_size=10)
    decoder.add_pinyin('ni')
    decoder.add_pinyin('hao')
    
    # 获取 Top-3
    candidates = decoder.get_topk_sequences(3)
    assert len(candidates) >= 2  # 至少应该有 2 个候选
    # 候选应该按分数降序排列
    for i in range(len(candidates) - 1):
        assert candidates[i][1] >= candidates[i+1][1]


def test_incremental_state_preservation():
    """测试状态保持"""
    hmm = build_toy_hmm()
    pinyin_map = {
        'ni': ['你', '尼'],
        'hao': ['好', '号'],
    }
    
    decoder = IncrementalViterbi(pinyin_map, hmm, beam_size=10)
    
    # 第一次添加
    decoder.add_pinyin('ni')
    state1 = len(decoder.dp_states)
    
    # 第二次添加
    decoder.add_pinyin('hao')
    state2 = len(decoder.dp_states)
    
    # 状态应该累积
    assert state2 == state1 + 1
    assert state2 == 2


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
