# AI 协作开发说明 (Copilot Instructions)

本仓库当前仅包含课程作业说明 (`Coursework 1-2025F.txt`) 与字形文件 `Chara.gb`（可能为后续模型或字典素材占位）。尚未建立代码结构。以下指导帮助 AI 代理快速、规范地为“拼音（全拼）→ 汉字”联想输入法（基于 HMM / 语言模型 + 解码算法）搭建项目骨架并迭代。

## 目标范围
- 输入：用空格分隔的全拼串，例如: `jin tian tian qi bu cuo`。
- 输出：概率最高的汉字序列（可拓展为 Top-K 备选）。
- 模型：初始建议使用隐马尔科夫模型 (HMM) 或 n-gram 语言模型（可从简到繁）。
- 语料：需后续引入（暂未提交）。接口与代码需为后续接入《人民日报》或自建语料留好扩展点。

## 初始目录规划 (若不存在则创建)
```
/data                # 放置原始语料与字典（不入库可 .gitignore）
/resources           # 公共可提交的轻量字典（拼音->候选汉字映射等）
/src
  /preprocess        # 语料清洗、频次统计脚本
  /models            # HMM、语言模型实现
  /decoder           # Viterbi / Beam Search / N-best 相关代码
  /cli               # 命令行交互入口
  /evaluation        # 准确率 / 字错误率等评估
/tests               # 单元与集成测试
/notebooks           # 探索式分析 (可选)
```

## 关键组件与建议文件
- `src/models/hmm.py`: 定义 HMM 参数结构：states=汉字，observations=拼音；需包含：
  - 状态初始概率 π
  - 状态转移概率 A (字→字)
  - 发射概率 B (字→拼音)
  - 支持从统计结果 `freq_*.json` 加载。
- `src/preprocess/build_stats.py`: 从语料统计：
  - 单字频次、双字频次 → 平滑后生成转移概率
  - 字→拼音映射（结合拼音字典）→ 发射概率
- `src/decoder/viterbi.py`: 标准 Viterbi，需：
  - log 概率以避免下溢
  - 支持 Top-K (使用优先队列 / k-best backpointer)
- `src/cli/infer.py`: 命令行：
  - `python -m src.cli.infer "jin tian ..." --topk 5`
  - 屏幕输出最佳句子与候选列表

## 数据与字典占位
若语料尚未提供，可放置：
- `resources/pinyin_map.json`: 结构 `{ "jin": ["今","金","进", ...], ... }`
- `resources/smoothing.yaml`: 配置平滑参数 (如 add-k, k=1e-6)。
AI 生成示例时需留注释：真实概率需由语料统计替换。

## 统计与平滑策略
- 转移概率：`P(c_i | c_{i-1}) = (bigram(c_{i-1}, c_i) + α) / (unigram(c_{i-1}) + α * V)`。
- 发射概率：`P(pinyin | char)` 或逆向需要统一方向；推荐存储为 `P(pinyin | char)`，解码时枚举 char。
- 日志域：使用 `math.log`；对缺失条目返回极小值 (如 `-1e9`)。

## 编码规范
- 统一 UTF-8；Python ≥ 3.10。
- 全部概率计算在 `log` 空间；不要混用线性概率。
- 模块内提供 `load_*` 与 `save_*` 函数，避免在全局执行 I/O。
- 适度使用 `dataclasses.dataclass` 封装 HMM 参数。

## 最小可运行里程碑 (MVP)
1. 准备一个极小的示例 `resources/pinyin_map.json`（10~20 条）。
2. 构造简化转移概率（例如统一或基于假设频次）。
3. 实现 Viterbi 返回结果，使用 1~2 个测试用例验证。

## 测试策略
- `tests/test_viterbi_basic.py`: 断言给定构造的玩具概率下输出与期望一致。
- `tests/test_pipeline_cli.py`: 使用 `subprocess` 调用 CLI，检查输出非空且含候选。
- 后续可添加 Top-K 顺序正确性测试（概率单调不增）。

## 常见风险与注意
- 拼音多音字：需要枚举所有候选字符；建议预先裁剪（频次 Top-N）。
- 数据稀疏：无 bigram 时使用回退或平滑；占位实现需显式 `# TODO(backoff)` 注释。
- 性能：长句 Beam Search 替换全量 Viterbi；首版可不做优化。
- 标点：首版可直接忽略或原样透传。

## 代理协作指令（对未来 AI 修改者）
- 不要硬编码真实概率；若生成示例文件，清晰标注“示例 / 需替换”。
- 修改核心算法时：新增/更新对应测试，确保 `log` 概率路径一致。
- 新增外部依赖前，确认是否必要，优先使用标准库与 `numpy`。
- 若引入大型语料，不提交原始文件，可提交统计产物（JSON 压缩后 <10MB）。

## 后续可拓展（仅列出，不自动实现）
- 加入语言模型 (n-gram with Kneser-Ney) 融合：`λ * log P_hmm + (1-λ) * log P_lm`。
- 支持增量学习：追加语料后局部更新频次表。
- 模糊输入纠错：编辑距离 + 拼音候选扩展。

---
若需要我继续：
1) 生成骨架代码
2) 提供示例字典与测试
请明确告知，我会在此基础上扩展。
