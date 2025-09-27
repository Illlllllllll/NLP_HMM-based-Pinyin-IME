"""用户版启动脚本（PyQt6 界面）。

特点：
- 启动时后台检查 lexicon/hmm，如需则自动构建。
- 顶部为操作区：导入拼音 TXT、设置 Top-K / Beam、解码按钮与状态提示。
- 底部为可滚动文本框展示结果。

可执行：
    python UserApp.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.decoder.viterbi import viterbi_topk
from src.models.hmm import HMMParams
from src.preprocess.load_lexicons import load_all
from src.preprocess.build_stats import (
    build_stats,
    attach_pinyin_emission,
    save_counter_json,
    save_emit_json,
)

BASE_DIR = Path(__file__).parent.resolve()
USRS_DIR = BASE_DIR / 'usrs'
RES_DIR = BASE_DIR / 'resources'
CORPUS_FILE = USRS_DIR / 'peopledaily' / 'PeopleDaily199801.txt'

RES_DIR.mkdir(parents=True, exist_ok=True)


class PrepareWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict, dict, object)
    error = pyqtSignal(str)

    def __init__(self, usrs_dir: Path, res_dir: Path, corpus_path: Path):
        super().__init__()
        self.usrs_dir = usrs_dir
        self.res_dir = res_dir
        self.corpus_path = corpus_path

    def run(self):
        try:
            lex_path = self.res_dir / 'lexicon_aggregate.json'
            self.progress.emit('初始化字典…')
            if lex_path.exists():
                lex_data = json.loads(lex_path.read_text(encoding='utf-8'))
            else:
                lex_data = load_all(self.usrs_dir)
                lex_path.write_text(json.dumps(lex_data, ensure_ascii=False, indent=2), encoding='utf-8')

            base_map = lex_data.get('base_pinyin_to_chars', {}) or {}
            base_map = {py: list(chars) for py, chars in base_map.items()}
            char_prior = {ch: int(cnt) for ch, cnt in (lex_data.get('char_frequency') or {}).items()}
            self.progress.emit(f'拼音映射: {len(base_map)} 项')

            hmm_path = self.res_dir / 'hmm_params.json'
            if hmm_path.exists():
                hmm = HMMParams.load(hmm_path)
                self.progress.emit('加载 HMM 成功')
            else:
                if not self.corpus_path.exists():
                    raise FileNotFoundError('缺少语料：无法自动统计，请先放置 PeopleDaily199801.txt')
                self.progress.emit('统计语料…')
                unigram, bigram = build_stats(self.corpus_path)
                emit = attach_pinyin_emission(unigram, base_map, char_prior)
                save_counter_json(unigram, self.res_dir / 'freq_unigram.json')
                save_counter_json(bigram, self.res_dir / 'freq_bigram.json')
                save_emit_json(emit, self.res_dir / 'freq_emit.json')
                self.progress.emit('构建 HMM…')
                hmm = HMMParams.from_frequency(
                    self.res_dir / 'freq_unigram.json',
                    self.res_dir / 'freq_bigram.json',
                    self.res_dir / 'freq_emit.json',
                )
                hmm.save(hmm_path)
            self.progress.emit('准备完成')
            self.finished.emit(base_map, char_prior, hmm)
        except Exception as exc:  # pylint: disable=broad-except
            self.error.emit(str(exc))


class DecodeWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, base_map: Dict[str, list], hmm: HMMParams, input_path: Path, k: int, beam: int):
        super().__init__()
        self.base_map = base_map
        self.hmm = hmm
        self.input_path = input_path
        self.k = k
        self.beam = beam

    def run(self):
        try:
            lines = self.input_path.read_text(encoding='utf-8').splitlines()
        except Exception as exc:  # pylint: disable=broad-except
            self.error.emit(f'读取失败: {exc}')
            return

        outputs: list[str] = []
        for idx, line in enumerate(lines, 1):
            raw_line = line.strip()
            if not raw_line:
                continue
            seq = raw_line.split()
            topk = viterbi_topk(seq, self.base_map, self.hmm, k=self.k, beam_size=self.beam)
            header = f'[{idx}] {raw_line}'
            if not topk:
                outputs.append(f'{header}\n  <无结果>')
                continue
            best_seq, best_score = topk[0]
            block = [f'{header}\n  BEST: {best_seq} (logP={best_score:.2f})']
            for cand, score in topk[1:]:
                block.append(f'    ALT: {cand} (logP={score:.2f})')
            outputs.append('\n'.join(block))
            self.progress.emit(f'已解码 {idx} 行')

        text = '\n'.join(outputs)
        if text:
            text += '\n\n--- 完成 ---'
        else:
            text = '输入文件为空或无有效拼音序列。'
        self.finished.emit(text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('拼音→汉字 解码工具（PyQt）')
        self.resize(960, 680)

        self.base_map: Dict[str, list] = {}
        self.char_prior: Dict[str, int] = {}
        self.hmm: Optional[HMMParams] = None
        self.input_path: Optional[Path] = None

        self.prepare_thread: Optional[QThread] = None
        self.decode_thread: Optional[QThread] = None

        self.prepare_worker: Optional[PrepareWorker] = None
        self.decode_worker: Optional[DecodeWorker] = None

        self._setup_ui()
        self._start_prepare()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)

        self.choose_btn = QPushButton('导入拼音 TXT')
        self.choose_btn.clicked.connect(self.choose_file)  # type: ignore[arg-type]
        self.choose_btn.setMinimumHeight(36)
        self.choose_btn.setStyleSheet('QPushButton { background:#3b82f6; color:white; padding:6px 16px; border-radius:6px; }'
                                      'QPushButton:disabled { background:#9ca3af; }'
                                      'QPushButton:hover { background:#1d4ed8; }')

        self.k_edit = QLineEdit('5')
        self.k_edit.setFixedWidth(60)
        self.k_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.beam_edit = QLineEdit('5')
        self.beam_edit.setFixedWidth(60)
        self.beam_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.decode_btn = QPushButton('解码')
        self.decode_btn.clicked.connect(self.start_decode)  # type: ignore[arg-type]
        self.decode_btn.setMinimumHeight(36)
        self.decode_btn.setStyleSheet('QPushButton { background:#10b981; color:white; padding:6px 16px; border-radius:6px; }'
                                      'QPushButton:disabled { background:#9ca3af; }'
                                      'QPushButton:hover { background:#059669; }')

        self.status_label = QLabel('准备')
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setStyleSheet('color:#374151; font-family:Consolas;')

        top_layout.addWidget(self.choose_btn)
        top_layout.addWidget(QLabel('Top-K:'))
        top_layout.addWidget(self.k_edit)
        top_layout.addSpacing(8)
        top_layout.addWidget(QLabel('Beam:'))
        top_layout.addWidget(self.beam_edit)
        top_layout.addSpacing(12)
        top_layout.addWidget(self.decode_btn)
        top_layout.addStretch(1)
        top_layout.addWidget(self.status_label)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet('font-family:Consolas; font-size:12pt;')

        root_layout.addLayout(top_layout)
        root_layout.addWidget(self.output, stretch=1)

    # ------------------------------------------------------------------
    def _start_prepare(self):
        self.decode_btn.setEnabled(False)
        self.prepare_thread = QThread(self)
        self.prepare_worker = PrepareWorker(USRS_DIR, RES_DIR, CORPUS_FILE)
        self.prepare_worker.moveToThread(self.prepare_thread)
        self.prepare_thread.started.connect(self.prepare_worker.run)
        self.prepare_worker.progress.connect(self.set_status)
        self.prepare_worker.finished.connect(self.on_prepare_finished)
        self.prepare_worker.error.connect(self.on_prepare_error)
        self.prepare_worker.error.connect(self.prepare_thread.quit)
        self.prepare_worker.finished.connect(lambda *_: self.prepare_thread.quit())
        self.prepare_thread.finished.connect(self.on_prepare_thread_finished)
        self.prepare_thread.start()

    def on_prepare_finished(self, base_map: dict, char_prior: dict, hmm_obj: object):
        self.base_map = base_map
        self.char_prior = char_prior
        self.hmm = hmm_obj  # 已是 HMMParams 实例
        self.set_status('准备完成，可以解码')
        self.decode_btn.setEnabled(True)

    def on_prepare_error(self, message: str):
        self.set_status('初始化失败')
        QMessageBox.critical(self, '错误', f'自动准备失败：{message}')
        self.decode_btn.setEnabled(False)

    def on_prepare_thread_finished(self):
        if self.prepare_worker:
            self.prepare_worker.deleteLater()
        if self.prepare_thread:
            self.prepare_thread.deleteLater()
        self.prepare_worker = None
        self.prepare_thread = None

    # ------------------------------------------------------------------
    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择拼音 TXT', str(BASE_DIR), 'Text Files (*.txt);;All Files (*.*)')
        if path:
            self.input_path = Path(path)
            self.set_status(f'已选择：{self.input_path.name}')

    def start_decode(self):
        if not self.base_map or self.hmm is None:
            QMessageBox.warning(self, '等待', '系统尚未准备好，请稍后再试。')
            return
        if not self.input_path or not self.input_path.exists():
            QMessageBox.warning(self, '提醒', '请先导入有效的拼音 TXT 文件。')
            return

        try:
            k = int(self.k_edit.text())
            beam = int(self.beam_edit.text())
            if k <= 0 or beam <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, '输入错误', 'Top-K 与 Beam 必须为正整数。')
            return

        self.decode_btn.setEnabled(False)
        self.set_status('解码中…')
        self.output.clear()

        self.decode_thread = QThread(self)
        self.decode_worker = DecodeWorker(self.base_map, self.hmm, self.input_path, k, beam)
        self.decode_worker.moveToThread(self.decode_thread)
        self.decode_thread.started.connect(self.decode_worker.run)
        self.decode_worker.progress.connect(self.set_status)
        self.decode_worker.finished.connect(self.on_decode_finished)
        self.decode_worker.error.connect(self.on_decode_error)
        self.decode_worker.error.connect(lambda *_: self.decode_thread.quit())
        self.decode_worker.finished.connect(lambda *_: self.decode_thread.quit())
        self.decode_thread.finished.connect(self.on_decode_thread_finished)
        self.decode_thread.start()

    def on_decode_finished(self, text: str):
        self.output.setPlainText(text)
        self.set_status('完成')

    def on_decode_error(self, message: str):
        QMessageBox.critical(self, '错误', message)
        self.set_status('解码失败')

    def on_decode_thread_finished(self):
        if self.decode_worker:
            self.decode_worker.deleteLater()
        if self.decode_thread:
            self.decode_thread.deleteLater()
        self.decode_worker = None
        self.decode_thread = None
        self.decode_btn.setEnabled(True)

    def set_status(self, text: str):
        self.status_label.setText(text)


def main():
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
