import json
import pprint
import os

def create_test_cases_from_log(log_file_path, output_py_file_path):
    """
    从游戏进程日志中读取、解析并提取 game_state 对象，
    然后将它们作为测试用例列表写入一个新的 Python 文件。

    :param log_file_path: 包含游戏日志的 .txt 文件路径。
    :param output_py_file_path: 要生成的 .py 测试用例文件路径。
    """
    game_states = []
    print(f"开始读取日志文件: {log_file_path}")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # JSON行通常以 '{' 开头并以 '}' 结尾
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    try:
                        data = json.loads(line)
                        # 检查是否存在 'game_state' 键，并且其值不为空
                        if 'game_state' in data and data['game_state']:
                            game_states.append(data['game_state'])
                    except json.JSONDecodeError:
                        # 忽略无法解析为JSON的行
                        continue
        
        print(f"成功提取 {len(game_states)} 个游戏状态。")

        with open(output_py_file_path, 'w', encoding='utf-8') as f:
            f.write("# -*- coding: utf-8 -*-\n")
            f.write("# 此文件由 create_test_cases.py 自动生成，请勿手动编辑。\n")
            f.write("# 包含了从 process.txt 中提取的游戏状态，用于测试。\n\n")
            f.write("test_cases = " + pprint.pformat(game_states, indent=4))
        
        print(f"测试用例已成功写入: {output_py_file_path}")

    except FileNotFoundError:
        print(f"错误: 未找到日志文件 {log_file_path}")

if __name__ == '__main__':
    # 获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建相对于脚本位置的完整文件路径
    log_file = os.path.join(script_dir, 'process.txt')
    output_file = os.path.join(script_dir, 'game_state_test_cases.py')

    create_test_cases_from_log(log_file,
        output_file
    )