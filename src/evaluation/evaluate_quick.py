from __future__ import annotations

import json
from pathlib import Path

from .metrics import evaluate_pinyin_system, print_evaluation_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Evaluate the default demo files shipped with the repository."""
    base_dir = PROJECT_ROOT
    pinyin_file = base_dir / 'testword.txt'
    ref_file = base_dir / 'testword_ev.txt'
    lexicon_file = base_dir / 'resources' / 'lexicon_aggregate.json'
    hmm_file = base_dir / 'resources' / 'hmm_params.json'

    missing = [path for path in [pinyin_file, ref_file, lexicon_file, hmm_file] if not path.exists()]
    if missing:
        for path in missing:
            print(f"错误: 文件不存在 {path}")
        return 1

    print("正在评估拼音输入法性能...")
    print(f"输入文件: {pinyin_file.name}")
    print(f"参考文件: {ref_file.name}\n")

    try:
        metrics = evaluate_pinyin_system(
            pinyin_file,
            ref_file,
            lexicon_file,
            hmm_file,
            topk=5,
        )
    except Exception as exc:  # pragma: no cover - 手动运行脚本时的异常说明
        print(f"评估失败: {exc}")
        return 1

    print_evaluation_report(metrics)

    output_file = base_dir / 'evaluation_results.json'
    metrics_to_dump = {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, list) or key in {"predictions", "references"}
    }
    output_file.write_text(json.dumps(metrics_to_dump, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n评估结果已保存到: {output_file}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
