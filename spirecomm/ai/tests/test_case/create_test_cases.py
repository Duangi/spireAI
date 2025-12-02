import json
import pprint
import os
from spirecomm.spire.game import Game

def _extract_json_blocks(text):
    """从一段文本中提取所有第一个平衡大括号对的 JSON 子串（返回字符串列表）。"""
    blocks = []
    i = 0
    n = len(text)
    while i < n:
        # 找到下一个 '{'
        start = text.find('{', i)
        if start == -1:
            break
        level = 0
        end = None
        for k, ch in enumerate(text[start:], start):
            if ch == '{':
                level += 1
            elif ch == '}':
                level -= 1
                if level == 0:
                    end = k + 1
                    break
        if end is not None:
            blocks.append(text[start:end])
            i = end
        else:
            # 没有找到配对的 '}'，停止
            break
    return blocks

def create_test_cases_from_log(log_file_path, output_py_file_path):
    """
    从 process.txt 读取，提取每个 JSON 块，直接用 Game.from_json 构造 Game，
    然后尝试获取 game_state，最终把所有 game_state 写入 output_py_file_path。
    """
    game_states = []
    print(f"开始读取日志文件: {log_file_path}")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"错误: 未找到日志文件 {log_file_path}")
        return

    json_blocks = _extract_json_blocks(text)
    print(f"发现 {len(json_blocks)} 个 JSON 块，准备逐个用 Game.from_json 解析。")

    for idx, jb in enumerate(json_blocks):
        game_state_obj = None
        try:
            # 直接尝试用 Game.from_json（用户要求）
            try:
                game = Game.from_json(jb)
            except Exception as e:
                # 某些日志中的 JSON 可能不是 Game 完整格式，尝试解析并看是否包含外层对象
                # 再包一层解析尝试
                parsed = json.loads(jb)
                # 如果 parsed 本身是个包含更大结构的对象（例如 {"available_commands":..., "game_state": {...}}）
                # 尝试传入其 JSON 重序列化结果给 Game.from_json
                try:
                    game = Game.from_json(json.dumps(parsed))
                except Exception:
                    # 无法用 Game.from_json 构造，回退使用解析后的 dict 直接提取 game_state
                    game = None

            if game is not None:
                # 优先从 Game 对象取 game_state 属性
                game_state_obj = getattr(game, "game_state", None)
                # 某些实现可能把状态放在属性 state 或 raw_state，尝试常见备选名
                if game_state_obj is None:
                    for attr in ("state", "raw_state", "_state", "data"):
                        game_state_obj = getattr(game, attr, None)
                        if game_state_obj is not None:
                            break

            # 回退：如果上面没有得到 game_state，则尝试直接 json.loads 并取 "game_state" 字段
            if game_state_obj is None:
                try:
                    parsed = json.loads(jb)
                    if isinstance(parsed, dict) and "game_state" in parsed and parsed["game_state"]:
                        game_state_obj = parsed["game_state"]
                    else:
                        # 若解析出来的就是一个 game_state（直接为 game_state），则保存该对象
                        # 或者保存整个 parsed 对象以供测试
                        game_state_obj = parsed
                except Exception as e:
                    print(f"[块 {idx}] JSON 解析失败: {e}")
                    continue

            game_states.append(game_state_obj)
        except Exception as e:
            print(f"[块 {idx}] 处理失败，回退并保存原始解析对象: {e}")
            try:
                parsed = json.loads(jb)
                game_states.append(parsed)
            except Exception:
                # 最后不得已跳过
                print(f"[块 {idx}] 无法解析或回退，已跳过。")
                continue

    print(f"成功提取 {len(game_states)} 个游戏状态/输入对象。")

    # 写入输出 python 文件
    try:
        with open(output_py_file_path, 'w', encoding='utf-8') as f:
            f.write("# -*- coding: utf-8 -*-\n")
            f.write("# 此文件由 create_test_cases.py 自动生成，请勿手动编辑。\n")
            f.write("# 包含了从 process.txt 中提取的游戏状态，用于测试。\n\n")
            f.write("test_cases = " + pprint.pformat(game_states, indent=4, width=120) + "\n")
        print(f"测试用例已成功写入: {output_py_file_path}")
    except Exception as e:
        print(f"写入输出文件失败: {e}")

if __name__ == '__main__':
    # 获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建相对于脚本位置的完整文件路径
    log_file = os.path.join(script_dir, 'process.txt')
    output_file = os.path.join(script_dir, 'game_state_test_cases.py')

    create_test_cases_from_log(log_file, output_file)