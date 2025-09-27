"""统一加载 usrs 目录下多种字典资源：
- Chara.gb : 汉字 -> 多个带声调拼音 (空格分隔)
- TONEPY.txt : 带声调拼音 -> 一串汉字
- Pth.gb : （可能是词/短语 -> 下划线分隔的带声调拼音序列）
- HSK词性表.txt : 词 -> 多个词性 (用于未来词级语言模型或过滤)
- count.out : 常用词词频（UTF-8），用于构建词频及字频先验

输出提供：
  1. char_to_pinyins (含声调)
  2. pinyin_tone_to_chars (含声调)
  3. base_pinyin_to_chars (去声调)
  4. word_to_pinyin_seq (来自 Pth.gb，已去声调)
  5. hsk_word_pos (词 -> set(pos))
    6. word_frequency / char_frequency (来自 count.out)

后续可用 word_to_pinyin_seq 构建词级候选或语言模型增强。
"""
from __future__ import annotations
from pathlib import Path
import re
from collections import defaultdict
from typing import Dict

PINYIN_TONE_RE = re.compile(r'([a-z]+)[1-5]$')

def tone_to_base(p: str) -> str:
    m = PINYIN_TONE_RE.match(p)
    return m.group(1) if m else p

def load_chara_gb(path: Path):
    d = {}
    if not path.exists():
        return d
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        ch = parts[0]
        pys = parts[1:]
        d[ch] = pys
    return d

def load_tonepy(path: Path):
    d = {}
    if not path.exists():
        return d
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if not line or '\t' not in line:
            continue
        py, chars = line.split('\t', 1)
        chars = chars.strip()
        d[py] = list(chars) if chars else []
    return d

def load_pth_gb(path: Path):
    """解析 Pth.gb: 观察行格式：<词或字符串><空格><下划线分隔拼音带声调序列>。
    注意文件似乎包含编码异常（可能为 GBK 转 UTF-8 乱码），此处仍按分隔策略提取第二段。
    返回：word -> [base_pinyin...]
    """
    d = {}
    if not path.exists():
        return d
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        word, py_seq = parts
        p_tones = [p for p in py_seq.split('_') if p]
        base_seq = [tone_to_base(p) for p in p_tones]
        d[word] = base_seq
    return d

def load_hsk_pos(path: Path):
    """解析 HSK 词性表: 可能格式： 词<空格>词性1(pos) 词性2(pos)
    过滤注释行 (#开头 或 非汉字起始)。"""
    d = defaultdict(set)
    if not path.exists():
        return d
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        raw = line.strip()
        if not raw or raw.startswith('#'):
            continue
        parts = raw.split()
        if not parts:
            continue
        word = parts[0]
        # 后续包含 形如 词性(pos)
        for seg in parts[1:]:
            if '(' in seg and seg.endswith(')'):
                d[word].add(seg)
    return d

def load_word_freq(path: Path) -> Dict[str, int]:
    freq = {}
    if not path.exists():
        return freq
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 2:
            continue
        word, count_str = parts[0], parts[1]
        try:
            count = int(count_str)
        except ValueError:
            continue
        freq[word] = count
    return freq


def build_char_frequency(word_freq: Dict[str, int]) -> Dict[str, int]:
    char_freq = defaultdict(int)
    for word, count in word_freq.items():
        if not word:
            continue
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                char_freq[ch] += count
    
    # 手动调整高频字权重（如"你"比"尼"更常用）
    priority_chars = {
        '你': 2.0,  # 权重倍数
        '的': 1.5,
        '了': 1.5,
        '是': 1.5,
        '在': 1.3,
        '有': 1.3,
    }
    for ch, multiplier in priority_chars.items():
        if ch in char_freq:
            char_freq[ch] = int(char_freq[ch] * multiplier)
    
    return dict(char_freq)


def build_base_pinyin_map(char_to_pinyins, pinyin_tone_to_chars, char_priority: Dict[str, int] | None = None):
    base_candidates = defaultdict(set)
    # from tonepy
    for py_tone, chars in pinyin_tone_to_chars.items():
        base = tone_to_base(py_tone)
        for ch in chars:
            base_candidates[base].add(ch)
    # supplement from char_to_pinyins
    for ch, py_list in char_to_pinyins.items():
        for py_tone in py_list:
            base = tone_to_base(py_tone)
            base_candidates[base].add(ch)

    base_map = {}
    for py, chars in base_candidates.items():
        if char_priority:
            base_map[py] = sorted(chars, key=lambda c: (-char_priority.get(c, 0), c))
        else:
            base_map[py] = sorted(chars)
    return base_map

def load_all(usrs_dir: Path):
    chara = load_chara_gb(usrs_dir / 'Chara.gb')
    tonepy = load_tonepy(usrs_dir / 'TONEPY.txt')
    pth = load_pth_gb(usrs_dir / 'Pth.gb')
    hsk = load_hsk_pos(usrs_dir / 'HSK词性表.txt')
    word_freq = load_word_freq(usrs_dir / 'count.out')
    char_freq = build_char_frequency(word_freq)
    base_map = build_base_pinyin_map(chara, tonepy, char_freq)
    return {
        'char_to_pinyins': chara,
        'pinyin_tone_to_chars': tonepy,
        'base_pinyin_to_chars': base_map,
        'word_to_pinyin_seq': pth,
        'hsk_word_pos': {k: list(v) for k, v in hsk.items()},
        'word_frequency': word_freq,
        'char_frequency': char_freq,
    }

if __name__ == '__main__':
    import json, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--usrs', default='usrs')
    ap.add_argument('--out', default='resources/lexicon_aggregate.json')
    args = ap.parse_args()
    data = load_all(Path(args.usrs))
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Wrote', out_path)
