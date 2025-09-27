"""简单图形界面：加载 HMM + 拼音映射文件，对输入 TXT 文件的每一行拼音解码输出汉字。

使用 tkinter grid：
  1. 选择映射 / HMM 文件
  2. 选择待解码拼音文本（每行空格分隔拼音）
  3. 设置 Top-K / Beam 参数
  4. 输出结果（首列最佳，附带 K 候选）

运行：
  python -m src.cli.ui
"""
from __future__ import annotations
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from typing import Dict

from src.models.hmm import HMMParams
from src.decoder.viterbi import viterbi_decode, viterbi_topk


class DecoderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("拼音→汉字 HMM 解码")

        # 路径变量
        self.var_map = tk.StringVar()
        self.var_hmm = tk.StringVar()
        self.var_input = tk.StringVar()
        self.var_k = tk.StringVar(value='5')
        self.var_beam = tk.StringVar(value='5')

        self.hmm = None
        self.base_map: Dict[str, list] = {}

        row = 0
        tk.Label(root, text='拼音映射/聚合 JSON:').grid(row=row, column=0, sticky='e')
        tk.Entry(root, textvariable=self.var_map, width=55).grid(row=row, column=1, sticky='we')
        tk.Button(root, text='浏览', command=self.browse_map).grid(row=row, column=2)
        row += 1
        tk.Label(root, text='HMM 参数 JSON:').grid(row=row, column=0, sticky='e')
        tk.Entry(root, textvariable=self.var_hmm, width=55).grid(row=row, column=1, sticky='we')
        tk.Button(root, text='浏览', command=self.browse_hmm).grid(row=row, column=2)
        row += 1
        tk.Label(root, text='拼音输入 TXT:').grid(row=row, column=0, sticky='e')
        tk.Entry(root, textvariable=self.var_input, width=55).grid(row=row, column=1, sticky='we')
        tk.Button(root, text='浏览', command=self.browse_input).grid(row=row, column=2)
        row += 1
        tk.Label(root, text='Top-K:').grid(row=row, column=0, sticky='e')
        tk.Entry(root, textvariable=self.var_k, width=8).grid(row=row, column=1, sticky='w')
        tk.Label(root, text='Beam大小:').grid(row=row, column=1, padx=120, sticky='w')
        tk.Entry(root, textvariable=self.var_beam, width=8).grid(row=row, column=1, padx=190, sticky='w')
        tk.Button(root, text='开始解码', command=self.run_decode).grid(row=row, column=2)
        row += 1
        tk.Label(root, text='输出:').grid(row=row, column=0, sticky='ne')
        self.text_out = scrolledtext.ScrolledText(root, width=80, height=28)
        self.text_out.grid(row=row, column=1, columnspan=2, sticky='nsew')

        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(row, weight=1)

    def browse_map(self):
        path = filedialog.askopenfilename(filetypes=[('JSON','*.json'), ('All','*.*')])
        if path:
            self.var_map.set(path)
            self.load_map()

    def browse_hmm(self):
        path = filedialog.askopenfilename(filetypes=[('JSON','*.json'), ('All','*.*')])
        if path:
            self.var_hmm.set(path)
            self.load_hmm()

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[('Text','*.txt'), ('All','*.*')])
        if path:
            self.var_input.set(path)

    def load_map(self):
        try:
            obj = json.loads(Path(self.var_map.get()).read_text(encoding='utf-8'))
            if 'base_pinyin_to_chars' in obj:
                self.base_map = obj['base_pinyin_to_chars']
            else:
                self.base_map = obj
            messagebox.showinfo('OK', f'加载拼音映射成功，拼音条目: {len(self.base_map)}')
        except Exception as e:
            messagebox.showerror('错误', f'加载映射失败: {e}')

    def load_hmm(self):
        try:
            self.hmm = HMMParams.load(Path(self.var_hmm.get()))
            messagebox.showinfo('OK', f'HMM 加载成功，init大小: {len(self.hmm.init_log_probs)}')
        except Exception as e:
            messagebox.showerror('错误', f'加载 HMM 失败: {e}')

    def run_decode(self):
        if not self.base_map:
            self.load_map()
        if self.hmm is None:
            self.load_hmm()
        if self.hmm is None or not self.base_map:
            return
        input_path = Path(self.var_input.get())
        if not input_path.exists():
            messagebox.showerror('错误', '输入文件不存在')
            return
        try:
            k = int(self.var_k.get())
            beam = int(self.var_beam.get())
        except ValueError:
            messagebox.showerror('错误', 'Top-K 和 Beam 必须是整数')
            return
        lines = input_path.read_text(encoding='utf-8').splitlines()
        self.text_out.delete('1.0', tk.END)
        for idx, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            pinyin_seq = line.split()
            topk = viterbi_topk(pinyin_seq, self.base_map, self.hmm, k=k, beam_size=beam)
            if not topk:
                self.text_out.insert(tk.END, f'[{idx}] {line}\n  <无结果>\n')
                continue
            best = topk[0]
            self.text_out.insert(tk.END, f'[{idx}] {line}\n  BEST: {best[0]}  (logP={best[1]:.2f})\n')
            for cand_seq, score in topk[1:]:
                self.text_out.insert(tk.END, f'    ALT: {cand_seq} (logP={score:.2f})\n')
        self.text_out.insert(tk.END, '--- 完成 ---\n')


def main():
    root = tk.Tk()
    DecoderApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()