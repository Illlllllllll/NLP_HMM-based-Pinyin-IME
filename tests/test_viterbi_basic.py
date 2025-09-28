from __future__ import annotations

import math

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decoder.viterbi import viterbi_decode, viterbi_topk
from src.models.hmm import HMMParams


def build_toy_hmm() -> HMMParams:
    init = {
        '你': math.log(0.6),
        '尼': math.log(0.4),
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
        '好': {},
        '号': {},
    }
    emit = {
        '你': {'ni': math.log(1.0)},
        '尼': {'ni': math.log(1.0)},
        '好': {'hao': math.log(1.0)},
        '号': {'hao': math.log(1.0)},
    }
    return HMMParams(init, trans, emit)


def test_viterbi_decode_prefers_high_probability_sequence():
    hmm = build_toy_hmm()
    pinyin_map = {
        'ni': ['你', '尼'],
        'hao': ['好', '号'],
    }
    result = viterbi_decode(['ni', 'hao'], pinyin_map, hmm)
    assert result == '你好'


def test_viterbi_topk_orders_candidates_by_score():
    hmm = build_toy_hmm()
    pinyin_map = {
        'ni': ['你', '尼'],
        'hao': ['好', '号'],
    }
    results = viterbi_topk(['ni', 'hao'], pinyin_map, hmm, k=3)
    sequences = [seq for seq, _ in results]
    assert sequences[0] == '你好'
    assert '你号' in sequences


def test_bigram_bonus_can_promote_less_likely_pair():
    hmm = build_toy_hmm()
    pinyin_map = {
        'ni': ['你', '尼'],
        'hao': ['好', '号'],
    }
    result_without_bonus = viterbi_decode(['ni', 'hao'], pinyin_map, hmm)
    result_with_bonus = viterbi_decode(
        ['ni', 'hao'],
        pinyin_map,
        hmm,
        bigram_bonus={'你号': 1.0},
    )
    assert result_without_bonus == '你好'
    assert result_with_bonus == '你号'
