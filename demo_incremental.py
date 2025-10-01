#!/usr/bin/env python3
"""演示增量式拼音输入法的命令行版本

展示增量解码器的核心功能：
- 逐个添加拼音
- 实时显示候选
- 删除操作
- 前缀预测

运行：
    python demo_incremental.py
"""
from __future__ import annotations

import json
from pathlib import Path
from src.decoder.viterbi import IncrementalViterbi
from src.models.hmm import HMMParams

BASE_DIR = Path(__file__).parent.resolve()
RES_DIR = BASE_DIR / 'resources'


def load_resources():
    """加载词典和 HMM 模型"""
    print("正在加载资源...")
    
    # 加载词典
    lex_path = RES_DIR / 'lexicon_aggregate.json'
    with lex_path.open('r', encoding='utf-8') as f:
        lexicon = json.load(f)
    
    # 加载 HMM
    hmm_path = RES_DIR / 'hmm_params.json'
    hmm = HMMParams.load(hmm_path)
    
    print(f"✓ 词典加载完成，拼音数量: {len(lexicon['base_pinyin_to_chars'])}")
    print(f"✓ HMM 加载完成")
    return lexicon, hmm


def demo_basic():
    """演示基础增量输入"""
    print("\n" + "="*60)
    print("演示 1: 基础增量输入")
    print("="*60)
    
    lexicon, hmm = load_resources()
    decoder = IncrementalViterbi(
        lexicon['base_pinyin_to_chars'],
        hmm,
        beam_size=100
    )
    
    # 模拟输入 "ni hao"
    pinyins = ['ni', 'hao']
    
    for py in pinyins:
        print(f"\n添加拼音: '{py}'")
        candidates = decoder.add_pinyin(py)
        print("Top-3 候选:")
        for i, (seq, score) in enumerate(candidates[:3], 1):
            print(f"  {i}. {seq} (分数: {score:.2f})")


def demo_prefix():
    """演示前缀预测"""
    print("\n" + "="*60)
    print("演示 2: 拼音前缀预测")
    print("="*60)
    
    lexicon, hmm = load_resources()
    decoder = IncrementalViterbi(
        lexicon['base_pinyin_to_chars'],
        hmm,
        beam_size=100
    )
    
    # 先添加一个完整拼音
    decoder.add_pinyin('ni')
    print("\n已输入: 'ni'")
    
    # 测试不同前缀
    prefixes = ['h', 'ha', 'hao']
    for prefix in prefixes:
        print(f"\n前缀预测: 'ni {prefix}'")
        candidates = decoder.add_pinyin_prefix(prefix)
        print("可能的候选:")
        for i, (seq, score) in enumerate(candidates[:3], 1):
            print(f"  {i}. {seq}")


def demo_delete():
    """演示删除操作"""
    print("\n" + "="*60)
    print("演示 3: 删除操作（退格）")
    print("="*60)
    
    lexicon, hmm = load_resources()
    decoder = IncrementalViterbi(
        lexicon['base_pinyin_to_chars'],
        hmm,
        beam_size=100
    )
    
    # 添加多个拼音
    pinyins = ['jin', 'tian', 'tian', 'qi']
    for py in pinyins:
        decoder.add_pinyin(py)
    
    candidates = decoder.get_topk_sequences(3)
    print("\n完整输入后的候选:")
    for i, (seq, score) in enumerate(candidates, 1):
        print(f"  {i}. {seq}")
    
    # 删除最后一个
    print("\n删除最后一个拼音 ('qi')...")
    candidates = decoder.delete_last()
    print("删除后的候选:")
    for i, (seq, score) in enumerate(candidates[:3], 1):
        print(f"  {i}. {seq}")
    
    # 再删除一个
    print("\n再删除一个拼音 ('tian')...")
    candidates = decoder.delete_last()
    print("删除后的候选:")
    for i, (seq, score) in enumerate(candidates[:3], 1):
        print(f"  {i}. {seq}")


def demo_interactive():
    """交互式演示"""
    print("\n" + "="*60)
    print("演示 4: 交互式输入")
    print("="*60)
    print("\n输入拼音（用空格分隔），输入 'q' 退出")
    print("命令: ")
    print("  - 输入拼音: 直接输入如 'ni hao'")
    print("  - 删除: 输入 'd'")
    print("  - 清空: 输入 'r'")
    print("  - 退出: 输入 'q'")
    
    lexicon, hmm = load_resources()
    decoder = IncrementalViterbi(
        lexicon['base_pinyin_to_chars'],
        hmm,
        beam_size=100
    )
    
    while True:
        print(f"\n当前已输入: {' '.join(decoder.pinyin_buffer) or '(无)'}")
        cmd = input(">>> ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'd':
            candidates = decoder.delete_last()
            print("已删除最后一个拼音")
        elif cmd == 'r':
            decoder.reset()
            print("已重置")
            continue
        elif cmd:
            # 假设是拼音输入
            pinyins = cmd.split()
            for py in pinyins:
                if py in lexicon['base_pinyin_to_chars']:
                    decoder.add_pinyin(py)
                else:
                    print(f"  警告: 拼音 '{py}' 不在词典中")
        
        # 显示候选
        candidates = decoder.get_topk_sequences(5)
        if candidates:
            print("\n候选:")
            for i, (seq, score) in enumerate(candidates, 1):
                print(f"  {i}. {seq} (分数: {score:.2f})")
        else:
            print("\n(暂无候选)")
    
    print("\n再见！")


def main():
    print("="*60)
    print("增量式拼音输入法演示")
    print("="*60)
    
    try:
        # 运行各个演示
        demo_basic()
        demo_prefix()
        demo_delete()
        
        # 最后提供交互式模式
        choice = input("\n是否进入交互模式？(y/n): ").strip().lower()
        if choice == 'y':
            demo_interactive()
        
        print("\n演示结束！")
        
    except FileNotFoundError as e:
        print(f"\n错误: 资源文件未找到 - {e}")
        print("请先运行 UserApp.py 构建资源")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
