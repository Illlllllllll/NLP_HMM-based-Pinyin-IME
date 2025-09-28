"""评估模块：计算拼音输入法的准确率、字错误率等指标。

支持的评估指标：
- 句子级准确率（完全匹配）
- 字符级准确率 
- 字符错误率（CER, Character Error Rate）
- Top-K 准确率（参考答案在前K个候选中）

使用示例：
    python -m src.evaluation.metrics --pred predictions.txt --ref references.txt
    python -m src.evaluation.metrics --pinyin testword.txt --ref testword_ev.txt --hmm-params resources/hmm_params.json --lexicon resources/lexicon_aggregate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import editdistance

from src.models.hmm import HMMParams
from src.decoder.viterbi import viterbi_decode, viterbi_topk


def sentence_accuracy(predictions: List[str], references: List[str]) -> float:
    """计算句子级准确率（完全匹配）"""
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: predictions={len(predictions)}, references={len(references)}")
    
    correct = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
    return correct / len(references) if references else 0.0


def character_accuracy(predictions: List[str], references: List[str]) -> float:
    """计算字符级准确率"""
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: predictions={len(predictions)}, references={len(references)}")
    
    total_chars = 0
    correct_chars = 0
    
    for pred, ref in zip(predictions, references):
        pred = pred.strip()
        ref = ref.strip()
        total_chars += len(ref)
        
        # 逐字符比较（取较短长度）
        min_len = min(len(pred), len(ref))
        correct_chars += sum(1 for i in range(min_len) if pred[i] == ref[i])
    
    return correct_chars / total_chars if total_chars > 0 else 0.0


def character_error_rate(predictions: List[str], references: List[str]) -> float:
    """计算字符错误率（CER）= edit_distance / reference_length"""
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: predictions={len(predictions)}, references={len(references)}")
    
    total_distance = 0
    total_ref_chars = 0
    
    for pred, ref in zip(predictions, references):
        pred = pred.strip()
        ref = ref.strip()
        total_distance += editdistance.eval(pred, ref)
        total_ref_chars += len(ref)
    
    return total_distance / total_ref_chars if total_ref_chars > 0 else 0.0


def topk_accuracy(predictions_topk: List[List[str]], references: List[str], k: int = 5) -> float:
    """计算 Top-K 准确率（参考答案在前K个候选中）"""
    if len(predictions_topk) != len(references):
        raise ValueError(f"Length mismatch: predictions={len(predictions_topk)}, references={len(references)}")
    
    correct = 0
    for pred_list, ref in zip(predictions_topk, references):
        ref = ref.strip()
        if any(pred.strip() == ref for pred in pred_list[:k]):
            correct += 1
    
    return correct / len(references) if references else 0.0


def evaluate_from_files(pred_file: Path, ref_file: Path) -> Dict[str, float]:
    """从预测文件和参考文件计算评估指标"""
    predictions = pred_file.read_text(encoding='utf-8').strip().splitlines()
    references = ref_file.read_text(encoding='utf-8').strip().splitlines()
    
    metrics = {
        'sentence_accuracy': sentence_accuracy(predictions, references),
        'character_accuracy': character_accuracy(predictions, references),
        'character_error_rate': character_error_rate(predictions, references),
    }
    
    return metrics


def evaluate_pinyin_system(pinyin_file: Path, ref_file: Path, lexicon_file: Path, hmm_file: Path, topk: int = 5) -> Dict[str, Any]:
    """端到端评估拼音输入系统"""
    # 读取输入
    pinyin_lines = pinyin_file.read_text(encoding='utf-8').strip().splitlines()
    references = ref_file.read_text(encoding='utf-8').strip().splitlines()
    
    if len(pinyin_lines) != len(references):
        raise ValueError(f"Length mismatch: pinyin={len(pinyin_lines)}, references={len(references)}")
    
    # 加载模型
    with lexicon_file.open('r', encoding='utf-8') as f:
        lexicon = json.load(f)
    base_map = lexicon['base_pinyin_to_chars']
    bigram_bonus_raw = lexicon.get('word_bigram_bonus', {})
    bigram_bonus = {str(k): float(v) for k, v in bigram_bonus_raw.items()}
    hmm = HMMParams.load(hmm_file)
    
    # 生成预测
    predictions_top1 = []
    predictions_topk = []
    
    for line in pinyin_lines:
        pinyin_seq = line.strip().split()
        if not pinyin_seq:
            predictions_top1.append('')
            predictions_topk.append([''])
            continue
            
        # Top-1 预测
        top1 = viterbi_decode(pinyin_seq, base_map, hmm, bigram_bonus=bigram_bonus)
        predictions_top1.append(top1)

        # Top-K 预测
        topk_results = viterbi_topk(pinyin_seq, base_map, hmm, k=topk, bigram_bonus=bigram_bonus)
        topk_candidates = [result for result, _ in topk_results]
        predictions_topk.append(topk_candidates)
    
    # 计算指标
    metrics = {
        'sentence_accuracy': sentence_accuracy(predictions_top1, references),
        'character_accuracy': character_accuracy(predictions_top1, references),
        'character_error_rate': character_error_rate(predictions_top1, references),
        f'top{topk}_accuracy': topk_accuracy(predictions_topk, references, k=topk),
        'total_sentences': len(references),
        'predictions': predictions_top1,
        'references': references,
    }
    
    return metrics


def print_evaluation_report(metrics: Dict[str, Any]):
    """打印评估报告"""
    print("=" * 50)
    print("拼音输入法评估报告")
    print("=" * 50)
    print(f"总句子数: {metrics.get('total_sentences', 'N/A')}")
    print(f"句子级准确率: {metrics.get('sentence_accuracy', 0.0):.4f} ({metrics.get('sentence_accuracy', 0.0)*100:.2f}%)")
    print(f"字符级准确率: {metrics.get('character_accuracy', 0.0):.4f} ({metrics.get('character_accuracy', 0.0)*100:.2f}%)")
    print(f"字符错误率 (CER): {metrics.get('character_error_rate', 0.0):.4f} ({metrics.get('character_error_rate', 0.0)*100:.2f}%)")
    
    # Top-K 准确率
    for key, value in metrics.items():
        if key.startswith('top') and key.endswith('_accuracy'):
            print(f"{key.replace('_', '-').title()}: {value:.4f} ({value*100:.2f}%)")
    
    print("=" * 50)
    
    # 详细结果（如果有）
    if 'predictions' in metrics and 'references' in metrics:
        print("详细结果:")
        for i, (pred, ref) in enumerate(zip(metrics['predictions'], metrics['references']), 1):
            status = "✓" if pred.strip() == ref.strip() else "✗"
            print(f"  {i:2d}. {status} 预测: {pred:<15} | 参考: {ref}")


def main():
    parser = argparse.ArgumentParser(description='评估拼音输入法性能')
    
    # 模式1：从预测和参考文件评估
    parser.add_argument('--pred', type=Path, help='预测结果文件')
    parser.add_argument('--ref', type=Path, help='参考答案文件')
    
    # 模式2：端到端评估
    parser.add_argument('--pinyin', type=Path, help='拼音输入文件')
    parser.add_argument('--lexicon', type=Path, help='聚合字典文件')
    parser.add_argument('--hmm-params', type=Path, help='HMM参数文件')
    parser.add_argument('--topk', type=int, default=5, help='Top-K评估的K值')
    
    # 输出选项
    parser.add_argument('--output', type=Path, help='将评估结果保存到JSON文件')
    parser.add_argument('--verbose', action='store_true', help='显示详细结果')
    
    args = parser.parse_args()
    
    if args.pred and args.ref:
        # 模式1：从文件评估
        metrics = evaluate_from_files(args.pred, args.ref)
    elif args.pinyin and args.ref and args.lexicon and args.hmm_params:
        # 模式2：端到端评估
        metrics = evaluate_pinyin_system(args.pinyin, args.ref, args.lexicon, args.hmm_params, args.topk)
    else:
        parser.error("请提供 --pred 和 --ref，或者 --pinyin、--ref、--lexicon 和 --hmm-params")
    
    # 打印报告
    if args.verbose or not args.output:
        print_evaluation_report(metrics)
    
    # 保存结果
    if args.output:
        # 移除不能序列化的字段
        json_metrics = {k: v for k, v in metrics.items() if not isinstance(v, list) or k in ['predictions', 'references']}
        args.output.write_text(json.dumps(json_metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"评估结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
