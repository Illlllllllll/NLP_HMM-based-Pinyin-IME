#!/usr/bin/env python3
"""兼容入口：委托至 ``src.evaluation.evaluate_quick``。"""

from __future__ import annotations

import warnings

from src.evaluation.evaluate_quick import main


def _run() -> int:
    warnings.warn(
        "evaluate_quick.py 已迁移到 src/evaluation/evaluate_quick.py；建议使用"
        " `python -m src.evaluation.evaluate_quick`",
        DeprecationWarning,
        stacklevel=2,
    )
    return main()


if __name__ == '__main__':
    raise SystemExit(_run())