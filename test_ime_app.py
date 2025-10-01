"""测试 IMEApp 基本功能（不显示 GUI）"""
import sys
from pathlib import Path

# 确保可以导入
sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication
from IMEApp import IMEApp

def test_ime_app_loads():
    """测试 IME 应用能否正常加载"""
    app = QApplication(sys.argv)
    window = IMEApp()
    
    # 检查基本组件
    assert window.image_label is not None
    assert window.input_widget is not None
    assert window.pinyin_label is not None
    assert len(window.candidate_buttons) == 5
    assert window.output_label is not None
    
    # 检查初始状态
    assert window.is_inputting == False
    assert window.input_widget.isVisible() == False
    
    print("✓ IMEApp 加载成功")
    print(f"✓ 图片标签已创建: {window.image_label}")
    print(f"✓ 输入组件已创建: {window.input_widget}")
    print(f"✓ 候选按钮数量: {len(window.candidate_buttons)}")
    print(f"✓ 初始状态正确: 输入框隐藏")
    
    if window.decoder:
        print("✓ 解码器加载成功")
    else:
        print("⚠ 解码器未加载（可能缺少资源文件）")
    
    return True

if __name__ == '__main__':
    try:
        result = test_ime_app_loads()
        print("\n测试通过！")
        sys.exit(0)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
