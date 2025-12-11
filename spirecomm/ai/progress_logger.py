import os
import datetime
import sys
import torch

class ProgressLogger:
    """
    记录并可视化一局游戏内AI的每一步决策过程。
    """
    def __init__(self, log_dir="progress"):
        # 使用绝对路径，避免相对路径混淆导致与其它 logger 冲突
        self.log_dir = os.path.abspath(log_dir)
        self.log_file_path = None
        self.file_handle = None
        self.start_time = None
        self.step_count = 0
        
        # 确保日志目录存在（使用 exist_ok=True 更稳健）
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def __del__(self):
        """析构函数，确保在对象销毁时文件能被正确关闭并安全重命名为 INCOMPLETE（带异常保护）。"""
        try:
            if self.file_handle and not self.file_handle.closed:
                try:
                    self.file_handle.close()
                except Exception as e:
                    raise RuntimeError(f"无法关闭日志文件: {e}")
                # 只有在原文件存在时才尝试重命名，重命名也要捕获异常
                try:
                    if self.log_file_path and os.path.exists(self.log_file_path):
                        final_path = self.log_file_path.replace(".txt", "_INCOMPLETE.txt")
                        # 避免覆盖已有文件
                        if not os.path.exists(final_path):
                            os.rename(self.log_file_path, final_path)
                except Exception as e:
                    raise RuntimeError(f"无法重命名日志文件: {e}")
        except Exception as e:
            # 最后防护：确保 __del__ 不抛异常
            raise RuntimeError(f"析构函数出错: {e}")

    def start_episode(self):
        """在一局游戏开始时调用。"""
        self.step_count = 0
        self.start_time = datetime.datetime.now()
        # 再次确保目录存在（并发/外部修改时防护）
        os.makedirs(self.log_dir, exist_ok=True)
        # 生成更唯一的文件名，包含微秒和进程 id，降低与其他 logger 文件名冲突或被误当目录的概率
        fname = f"game_{self.start_time.strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}.txt"
        self.log_file_path = os.path.join(self.log_dir, fname)
        # 打开文件写入
        self.file_handle = open(self.log_file_path, 'w', encoding='utf-8')
        
    def log_step(self, step_info):
        """记录一步的详细信息。"""
        if not self.file_handle:
            return
            
        self.step_count += 1
        log_entry = f"--- Step {self.step_count} ---\n"

        # 记录可选动作及其Q值
        log_entry += "[Available Actions & Q-Values]:\n"
        if 'q_values' in step_info and step_info['q_values']:
            for action, q_val in step_info['q_values'].items():
                log_entry += f"  - Action: {action:<30} | Q-Value: {q_val:.4f}\n"
        else:
            log_entry += "  - (No Q-values available for this step)\n"
        
        # 记录选择的动作
        log_entry += f"\n[Chosen Action]: {step_info.get('chosen_action', 'N/A')}\n"

        # 记录状态变化
        log_entry += "\n[State Change]:\n"
        prev_player = step_info.get('prev_player', {})
        next_player = step_info.get('next_player', {})
        prev_monsters = step_info.get('prev_monsters', [])
        next_monsters = step_info.get('next_monsters', [])

        # 玩家状态
        log_entry += f"  - Player HP:    {prev_player.get('current_hp', 'N/A')} -> {next_player.get('current_hp', 'N/A')}\n"
        log_entry += f"  - Player Block: {prev_player.get('block', 'N/A')} -> {next_player.get('block', 'N/A')}\n"
        
        # 怪物状态
        for i, monster in enumerate(next_monsters):
            prev_monster_hp = "New"
            prev_monster_block = "N/A"
            if i < len(prev_monsters):
                prev_monster_hp = prev_monsters[i].get('current_hp', 'N/A')
                prev_monster_block = prev_monsters[i].get('block', 'N/A')
            log_entry += f"  - Monster {i} ({monster.get('name')}): HP {prev_monster_hp} -> {monster.get('current_hp')}, Block {prev_monster_block} -> {monster.get('block')}\n"

        # 能力变化 (简单对比)
        prev_powers = {p['id'] for p in prev_player.get('powers', [])}
        next_powers = {p['id'] for p in next_player.get('powers', [])}
        gained_powers = next_powers - prev_powers
        if gained_powers:
            log_entry += f"  - Gained Powers: {', '.join(gained_powers)}\n"

        # 记录奖励
        log_entry += f"\n[Reward]: {step_info.get('reward', 0.0):.4f}\n"
        log_entry += "=" * 40 + "\n\n"

        self.file_handle.write(log_entry)

    def end_episode(self):
        """在一局游戏结束时调用。"""
        if not self.file_handle:
            return

        end_time = datetime.datetime.now()
        duration = end_time - self.start_time
        self.file_handle.write(f"\n--- Episode Finished ---\n")
        self.file_handle.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file_handle.write(f"Total Duration: {duration}\n")
        self.file_handle.close()

        # 重置状态
        self.log_file_path = None
        self.file_handle = None
        self.start_time = None