# 拼音（全拼）→ 汉字 联想输入法 Coursework

本项目目标：实现一个基于 HMM（可后续扩展为 n-gram LM 融合）的拼音输入解码器：输入用空格分隔的全拼序列，输出概率最高的汉字序列（可扩展 Top-K）。

## 当前进度
- 字典/语料集中到 `usrs/`：
  - `Chara.gb`：汉字 -> 多个带声调拼音
  - `TONEPY.txt`：带声调拼音 -> 一串汉字
  - `Pth.gb`：词/短语 -> 下划线分隔带声调拼音序列
  - `HSK词性表.txt`：词 -> 词性集合（用于未来词级特征 / 过滤）
  - `count.out`：常用词词频（UTF-8）
  - `peopledaily/PeopleDaily199801.txt`：PFR 人民日报标注语料
- `src/preprocess/load_lexicons.py`：统一聚合以上资源，生成 `resources/lexicon_aggregate.json`（主流程唯一推荐映射文件，已移除 `pinyin_map.example.json`）。
- `notebooks/01_build_dictionary.ipynb`：初始演示（可逐步迁移到使用 `load_lexicons.py`）。
- `notebooks/02_viterbi_demo.ipynb`：Viterbi 玩具示例。
- `src/preprocess/build_stats.py`：支持新版 `--lexicon` 输入，通过聚合 JSON 生成频次与发射计数。
- `src/models/hmm.py` / `src/decoder/viterbi.py` / `src/cli/infer.py`：HMM 参数与解码骨架。

## 推荐目录结构（逐步补齐）
```
/resources                # 轻量级映射/配置（保留 lexicon_aggregate.json 及统计结果）
/data                     # 大型语料（不提交仓库）
/src
  preprocess/             # 语料统计脚本（生成 freq_unigram.json, freq_bigram.json 等）
  models/                 # hmm.py / ngram.py
  decoder/                # viterbi.py / beam.py
  cli/                    # infer.py  (命令行入口)
  evaluation/             # metrics.py（准确率/错误率）
/notebooks                # 数据探索、统计验证
/usrs                     # 原始语料与外部字典（见 `usrs/README.md` 获取来源）
```

## 核心模块
- `src/models/hmm.py`：定义 `HMMParams`，提供 log 概率加载/保存与访问接口。
- `src/decoder/viterbi.py`：实现 Top-1 与 Beam Top-K 解码。
- `src/preprocess/load_lexicons.py`：聚合 `usrs/` 目录下字典资源并生成 `lexicon_aggregate.json`。
- `src/preprocess/build_stats.py`：从语料统计 unigram/bigram/发射计数，支持字频先验回退。
- `src/cli/infer.py`：命令行推理入口。
- `src/evaluation/metrics.py`：评估模块，计算句子准确率、字符准确率、字符错误率等指标。
- `UserApp.py`：PyQt6 图形界面，支持批量解码与自动准备。

## 环境准备
1. Python 3.10 及以上。
2. 安装依赖：
  ```powershell
  pip install -r requirements.txt
  ```
3. 可选：为数据处理准备 `conda`/`venv` 环境，确保使用 UTF-8 作为默认编码。

## 数据准备
- `usrs/` 目录需放置 BCC 等来源的字典与语料，详见 `usrs/README.md`（包含引用格式与授权说明）。
- 大型语料（如人民日报全量）建议放在 `data/` 或 `usrs/peopledaily/`，仓库仅提交统计产物。
- `.gitignore` 默认忽略 `data/` 与 `usrs/peopledaily/`，上传前请确认不包含受限数据。

## 快速开始（命令行）
```powershell
# 1. 聚合字典（若 `resources/lexicon_aggregate.json` 不存在）
python -m src.preprocess.load_lexicons --usrs usrs --out resources/lexicon_aggregate.json

# 2. 统计语料并生成 HMM 参数（会写入 resources/freq_*.json & hmm_params.json）
python -m src.preprocess.build_stats --corpus usrs/peopledaily/PeopleDaily199801.txt --lexicon resources/lexicon_aggregate.json

# 3. 运行推理
python -m src.cli.infer "ni hao" --pinyin-map resources/lexicon_aggregate.json --hmm resources/hmm_params.json

# 4. 评估性能（可选）
python -m src.evaluation.metrics --pinyin testword.txt --ref testword_ev.txt --lexicon resources/lexicon_aggregate.json --hmm-params resources/hmm_params.json --verbose
# 或使用快速评估脚本
python -m src.evaluation.evaluate_quick
```

## 图形界面
```powershell
python UserApp.py
```
界面启动后：
1. 程序会在后台检查并生成 lexicon/hmm；状态栏显示“准备完成”后即可使用。
2. 点击“导入拼音 TXT”，选择每行空格分隔拼音的文本文件。
3. 调整 Top-K、Beam 参数后点击“解码”，结果显示在下方滚动框中。

## Notebook
Notebook 用于探索式分析，可在 VS Code 或 Jupyter 中打开 `notebooks/01_build_dictionary.ipynb`、`02_viterbi_demo.ipynb`。

## 使用 jieba 的场景
- 若引入词级语言模型（如统计词频 / 词 bigram），可先用 jieba 对语料分词。
- 评估阶段：将 HMM 输出的汉字序列再分词，与参考语料分词对比（可选）。

## 质量检查
- 语法检查：`python -m compileall src`。
- 静态分析（可选）：`ruff check src` 或 `flake8 src`。
- 回归测试：待补充 `tests/` 后运行 `pytest`。

## 下一步建议
1. 引入更高阶语言模型（如词级 n-gram 或 Kneser-Ney 平滑），融合至解码评分。
2. 完善测试集并编写 `pytest` 用例（CLI/GUI 解码结果、Top-K 顺序等）。
3. 支持增量更新：追加语料后快速刷新统计与 HMM 参数。
4. 优化 GUI：支持结果导出、进度条、错误高亮等高级功能。
5. 扩展评估集：收集更多测试样本并建立标准化的评估基准。

## 数据与隐私
大规模语料请放入 `data/`，默认 `.gitignore`，只提交衍生统计 JSON（<10MB）。

## 许可证 / 学术诚信
仅用于本课程作业学习与演示。请遵守课程学术诚信要求，不直接复制他人实现。 如需复用外部开源代码片段，需注明来源。 
