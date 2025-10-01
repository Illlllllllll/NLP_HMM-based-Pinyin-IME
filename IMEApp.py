"""增量式拼音输入法应用（IME App）

UI 特性：
- 默认显示静态图片（serena.jpg），文本框隐藏
- 键盘输入时自动显示候选框和输入框
- 实时显示拼音输入和候选汉字
- 点击图片显示设置弹窗
- 数字键 1-9 快速选择候选
- 空格确认当前拼音分词，回车上屏
- ESC 清空输入

运行：
    python IMEApp.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QEvent
from PyQt6.QtGui import QPixmap, QKeyEvent, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDialog,
    QLineEdit,
    QMessageBox,
)

from src.decoder.viterbi import IncrementalViterbi
from src.models.hmm import HMMParams

BASE_DIR = Path(__file__).parent.resolve()
RES_DIR = BASE_DIR / 'resources'


class SettingsDialog(QDialog):
    """设置弹窗"""
    
    def __init__(self, parent=None, beam_size=100, top_k=5):
        super().__init__(parent)
        self.setWindowTitle('输入法设置')
        self.setModal(True)
        self.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Beam Size 设置
        beam_layout = QHBoxLayout()
        beam_layout.addWidget(QLabel('Beam Size:'))
        self.beam_input = QLineEdit(str(beam_size))
        self.beam_input.setFixedWidth(80)
        beam_layout.addWidget(self.beam_input)
        beam_layout.addStretch()
        layout.addLayout(beam_layout)
        
        # Top-K 设置
        topk_layout = QHBoxLayout()
        topk_layout.addWidget(QLabel('候选数量:'))
        self.topk_input = QLineEdit(str(top_k))
        self.topk_input.setFixedWidth(80)
        topk_layout.addWidget(self.topk_input)
        topk_layout.addStretch()
        layout.addLayout(topk_layout)
        
        # 按钮
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton('确定')
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton('取消')
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def get_values(self) -> Tuple[int, int]:
        """返回 (beam_size, top_k)"""
        try:
            beam = int(self.beam_input.text())
            topk = int(self.topk_input.text())
            return max(1, beam), max(1, topk)
        except ValueError:
            return 100, 5


class IMEApp(QWidget):
    """增量式拼音输入法主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 参数
        self.beam_size = 100
        self.top_k = 5
        
        # 数据
        self.lexicon = None
        self.hmm = None
        self.decoder: Optional[IncrementalViterbi] = None
        
        # 输入状态
        self.pinyin_buffer = ""  # 当前未完成的拼音
        self.confirmed_text = ""  # 已确认上屏的文本
        self.is_inputting = False
        
        # UI 组件
        self.image_label: Optional[QLabel] = None
        self.input_widget: Optional[QWidget] = None
        self.pinyin_label: Optional[QLabel] = None
        self.candidate_buttons: List[QPushButton] = []
        self.output_label: Optional[QLabel] = None
        
        self._setup_ui()
        self._load_resources()
        
        # 安装事件过滤器以捕获全局键盘输入
        self.installEventFilter(self)
    
    def _setup_ui(self):
        """设置 UI"""
        self.setWindowTitle('拼音输入法')
        self.setFixedSize(400, 600)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 1. 静态图片（默认显示）
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(380, 500)
        self.image_label.setStyleSheet(
            'QLabel { background: #111827; border-radius: 8px; }'
        )
        self._load_image()
        self.image_label.mousePressEvent = self._on_image_clicked
        layout.addWidget(self.image_label)
        
        # 2. 输入/候选区域（默认隐藏）
        self.input_widget = QWidget()
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(5, 5, 5, 5)
        input_layout.setSpacing(5)
        
        # 拼音输入显示
        self.pinyin_label = QLabel('')
        self.pinyin_label.setStyleSheet(
            'QLabel { background: white; padding: 8px; '
            'border: 2px solid #3b82f6; border-radius: 4px; '
            'font-size: 16pt; font-family: Consolas; }'
        )
        self.pinyin_label.setMinimumHeight(40)
        input_layout.addWidget(self.pinyin_label)
        
        # 候选区域
        candidates_layout = QHBoxLayout()
        candidates_layout.setSpacing(5)
        for i in range(5):
            btn = QPushButton('')
            btn.setFixedSize(70, 40)
            btn.setStyleSheet(
                'QPushButton { background: #f3f4f6; border: 1px solid #d1d5db; '
                'border-radius: 4px; font-size: 14pt; }'
                'QPushButton:hover { background: #e5e7eb; }'
            )
            btn.clicked.connect(lambda checked, idx=i: self._select_candidate(idx))
            self.candidate_buttons.append(btn)
            candidates_layout.addWidget(btn)
        input_layout.addLayout(candidates_layout)
        
        self.input_widget.setLayout(input_layout)
        self.input_widget.setVisible(False)
        layout.addWidget(self.input_widget)
        
        # 3. 输出文本显示
        self.output_label = QLabel('已上屏文本：')
        self.output_label.setStyleSheet(
            'QLabel { background: #f9fafb; padding: 10px; '
            'border: 1px solid #e5e7eb; border-radius: 4px; '
            'font-size: 12pt; }'
        )
        self.output_label.setWordWrap(True)
        self.output_label.setMinimumHeight(60)
        layout.addWidget(self.output_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # 设置焦点策略
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def _load_image(self):
        """加载显示图片"""
        img_path = BASE_DIR / 'serena.jpg'
        if img_path.exists():
            pixmap = QPixmap(str(img_path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    380, 500,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)
                return
        self.image_label.setText('图片未找到\n\n点击打开设置')
        self.image_label.setStyleSheet(
            'QLabel { background: #111827; color: #9ca3af; '
            'border-radius: 8px; font-size: 14pt; }'
        )
    
    def _on_image_clicked(self, event):
        """点击图片显示设置"""
        dialog = SettingsDialog(self, self.beam_size, self.top_k)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            beam, topk = dialog.get_values()
            if beam != self.beam_size or topk != self.top_k:
                self.beam_size = beam
                self.top_k = topk
                # 重新创建解码器
                if self.lexicon and self.hmm:
                    self.decoder = IncrementalViterbi(
                        self.lexicon['base_pinyin_to_chars'],
                        self.hmm,
                        beam_size=self.beam_size
                    )
                QMessageBox.information(
                    self,
                    '设置已更新',
                    f'Beam Size: {self.beam_size}\n候选数量: {self.top_k}'
                )
    
    def _load_resources(self):
        """加载词典和 HMM 模型"""
        try:
            # 加载词典
            lex_path = RES_DIR / 'lexicon_aggregate.json'
            if not lex_path.exists():
                QMessageBox.critical(
                    self,
                    '错误',
                    f'词典文件不存在：{lex_path}\n请先运行 UserApp.py 构建资源'
                )
                return
            
            with lex_path.open('r', encoding='utf-8') as f:
                self.lexicon = json.load(f)
            
            # 加载 HMM 参数
            hmm_path = RES_DIR / 'hmm_params.json'
            if not hmm_path.exists():
                QMessageBox.critical(
                    self,
                    '错误',
                    f'HMM 参数文件不存在：{hmm_path}\n请先运行 UserApp.py 构建资源'
                )
                return
            
            self.hmm = HMMParams.load(hmm_path)
            
            # 创建增量解码器
            self.decoder = IncrementalViterbi(
                self.lexicon['base_pinyin_to_chars'],
                self.hmm,
                beam_size=self.beam_size
            )
            
            self.statusBar().showMessage('模型加载成功', 3000) if hasattr(self, 'statusBar') else None
            
        except Exception as e:
            QMessageBox.critical(
                self,
                '加载失败',
                f'加载资源时出错：{e}'
            )
    
    def eventFilter(self, obj, event):
        """事件过滤器，捕获键盘输入"""
        if event.type() == QEvent.Type.KeyPress:
            return self._handle_key_press(event)
        return super().eventFilter(obj, event)
    
    def keyPressEvent(self, event: QKeyEvent):
        """处理键盘输入"""
        self._handle_key_press(event)
    
    def _handle_key_press(self, event: QKeyEvent) -> bool:
        """处理键盘按键"""
        if not self.decoder:
            return False
        
        key = event.key()
        text = event.text()
        
        # ESC - 清空输入
        if key == Qt.Key.Key_Escape:
            self._clear_input()
            return True
        
        # 数字键 1-9 - 选择候选
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            idx = key - Qt.Key.Key_1
            self._select_candidate(idx)
            return True
        
        # 回车 - 上屏当前最佳候选
        if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            self._commit_best_candidate()
            return True
        
        # 空格 - 确认当前拼音分词
        if key == Qt.Key.Key_Space:
            self._confirm_pinyin()
            return True
        
        # 退格 - 删除字符
        if key == Qt.Key.Key_Backspace:
            self._handle_backspace()
            return True
        
        # 字母输入
        if text and text.isalpha() and text.islower():
            self._handle_letter_input(text)
            return True
        
        return False
    
    def _handle_letter_input(self, letter: str):
        """处理字母输入"""
        self.pinyin_buffer += letter
        self._show_input_ui()
        self._update_display()
    
    def _handle_backspace(self):
        """处理退格"""
        if self.pinyin_buffer:
            self.pinyin_buffer = self.pinyin_buffer[:-1]
            if not self.pinyin_buffer and not self.is_inputting:
                self._hide_input_ui()
            else:
                self._update_display()
        elif self.decoder.pinyin_buffer:
            # 删除最后一个已确认的拼音
            self.decoder.delete_last()
            self._update_display()
    
    def _confirm_pinyin(self):
        """确认当前拼音（空格键）"""
        if not self.pinyin_buffer:
            return
        
        # 检查拼音是否有效
        if self.pinyin_buffer in self.lexicon['base_pinyin_to_chars']:
            self.decoder.add_pinyin(self.pinyin_buffer)
            self.pinyin_buffer = ""
            self._update_display()
        else:
            # 尝试找到最近的匹配
            matching = [py for py in self.lexicon['base_pinyin_to_chars'].keys() 
                       if py.startswith(self.pinyin_buffer)]
            if matching:
                # 使用第一个匹配的拼音
                self.decoder.add_pinyin(matching[0])
                self.pinyin_buffer = ""
                self._update_display()
    
    def _select_candidate(self, index: int):
        """选择候选项"""
        if not self.candidate_buttons[index].text():
            return
        
        # 确认当前拼音（如果有）
        if self.pinyin_buffer:
            self._confirm_pinyin()
        
        # 获取当前候选
        candidates = self.decoder.get_topk_sequences(self.top_k)
        if index < len(candidates):
            selected_text = candidates[index][0]
            self._commit_text(selected_text)
    
    def _commit_best_candidate(self):
        """上屏当前最佳候选"""
        if self.pinyin_buffer:
            self._confirm_pinyin()
        
        candidates = self.decoder.get_topk_sequences(1)
        if candidates:
            self._commit_text(candidates[0][0])
    
    def _commit_text(self, text: str):
        """上屏文本"""
        self.confirmed_text += text
        self.output_label.setText(f'已上屏文本：{self.confirmed_text}')
        
        # 清空输入状态
        self.decoder.reset()
        self.pinyin_buffer = ""
        self._hide_input_ui()
    
    def _clear_input(self):
        """清空输入"""
        self.decoder.reset()
        self.pinyin_buffer = ""
        self._hide_input_ui()
    
    def _show_input_ui(self):
        """显示输入界面"""
        if not self.is_inputting:
            self.is_inputting = True
            self.input_widget.setVisible(True)
            self.image_label.setFixedSize(380, 300)
            self._load_image()
    
    def _hide_input_ui(self):
        """隐藏输入界面"""
        if self.is_inputting:
            self.is_inputting = False
            self.input_widget.setVisible(False)
            self.image_label.setFixedSize(380, 500)
            self._load_image()
    
    def _update_display(self):
        """更新显示"""
        # 更新拼音显示
        confirmed_pinyins = ' '.join(self.decoder.pinyin_buffer)
        if self.pinyin_buffer:
            display_text = f"{confirmed_pinyins} {self.pinyin_buffer}" if confirmed_pinyins else self.pinyin_buffer
        else:
            display_text = confirmed_pinyins
        self.pinyin_label.setText(display_text or '(输入拼音...)')
        
        # 获取候选
        if self.pinyin_buffer:
            # 使用前缀预测
            candidates = self.decoder.add_pinyin_prefix(self.pinyin_buffer)
        else:
            # 使用完整拼音
            candidates = self.decoder.get_topk_sequences(self.top_k)
        
        # 更新候选按钮
        for i, btn in enumerate(self.candidate_buttons):
            if i < len(candidates):
                text, score = candidates[i]
                btn.setText(f'{i+1}. {text}')
                btn.setVisible(True)
            else:
                btn.setText('')
                btn.setVisible(False)


def main():
    app = QApplication(sys.argv)
    
    # 设置应用字体
    font = QFont('Microsoft YaHei', 10)
    app.setFont(font)
    
    window = IMEApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
