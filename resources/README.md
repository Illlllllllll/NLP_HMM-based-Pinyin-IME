# resources 目录说明

## 主要文件
- `lexicon_aggregate.json`：`load_lexicons.py` 聚合 `usrs/` 中多源字典的输出，包含：
	- `char_to_pinyins` （汉字 -> 拼音）
	- `pinyin_tone_to_chars` （拼音 -> 汉字）
	- `base_pinyin_to_chars`（核心：无声调拼音 -> 候选字）
	- `word_to_pinyin_seq`（词/短语的无声调拼音序列）
	- `hsk_word_pos`（HSK 词 -> 词性集合）
	- `word_frequency`（常用词 -> 频次，来自 `usrs/count.out`）
	- `char_frequency`（汉字 -> 频次先验，由词频累积）
	- `word_bigram_bonus`（常见双字词奖励，用于解码时提升多字词概率）
- `freq_unigram.json`: 字 unigram 计数（`build_stats.py`）。
- `freq_bigram.json`: 字 bigram 计数（`build_stats.py`）。
- `freq_emit.json`: (char -> {pinyin: count}) 发射统计（当前简单复制策略）。
- `hmm_params.json`: 由 `HMMParams.from_frequency` 构建的 HMM log 概率参数。

## 生成流程示例
1. 聚合字典：
```bash
python -m src.preprocess.load_lexicons --usrs usrs --out resources/lexicon_aggregate.json
```
2. 统计频次：
```bash
python -m src.preprocess.build_stats --corpus usrs/peopledaily/PeopleDaily199801.txt \
		--lexicon resources/lexicon_aggregate.json
```
3. 构建 HMM 参数：
```python
from pathlib import Path
from src.models.hmm import HMMParams
hmm = HMMParams.from_frequency(Path('resources/freq_unigram.json'),
															 Path('resources/freq_bigram.json'),
															 Path('resources/freq_emit.json'))
hmm.save(Path('resources/hmm_params.json'))
```
4. 推理测试：
```bash
python -m src.cli.infer "ni hao" --pinyin-map resources/lexicon_aggregate.json --hmm resources/hmm_params.json
```
（注意：`infer.py` 当前读取的是 base 映射，需要你在调用前取 `base_pinyin_to_chars` 子字段或单独导出。）

## 文件格式说明
- 计数文件使用 JSON；若未来体积过大（>10MB），可切换为压缩（`.jsonl.gz`）。
- 发射计数当前并非真实多音字概率，需后续用 (char, pinyin) 对齐或外部频次替换。

## 改进建议
- 拆分 `lexicon_aggregate.json` 为多个专用文件以减少加载内存。
- `freq_emit.json` 增加基于多音字选择频率的归一化策略。
- 引入词级 n-gram：利用 `word_to_pinyin_seq` 与 HSK 词性过滤。 
