# 朝绝对路径进行txt内容的写入
import datetime
from math import log
import os
import sys
import json
from enum import Enum

class LogType(Enum):
    """
    日志类型枚举类，用于指定日志的级别。
    """
    PROGRESS = 1
    REWARD = 2
    STATE = 3

class AbsoluteLogger:
    """
    一个简单的日志记录器，用于将字符串内容写入指定路径的文件。
    """
    def __init__(self, log_type:LogType = LogType.PROGRESS):
        # 如果是windows系统，换一个绝对路径D:\Projects\spireAI
        if log_type == LogType.PROGRESS:
            if os.name == 'nt':
                log_file_path = "D:/Projects/spireAI/log/"
            else:
                log_file_path = "/Users/duang/Projects/spireAI/log/"
        elif log_type == LogType.REWARD:
            if os.name == 'nt':
                log_file_path = "D:/Projects/spireAI/reward_log/"
            else:
                log_file_path = "/Users/duang/Projects/spireAI/reward_log/"
        elif log_type == LogType.STATE:
            if os.name == 'nt':
                log_file_path = "D:/Projects/spireAI/state_log/"
            else:
                log_file_path = "/Users/duang/Projects/spireAI/state_log/"
        else:
            raise ValueError(f"未知的日志类型: {log_type}")

        self.log_file_path = log_file_path
        self.file_handle = None
        self._ensure_dir_exists()

    def start_episode(self):
        """在一局游戏开始时调用。"""
        self.step_count = 0
        self.start_time = datetime.datetime.now()
        self.log_file_path = os.path.join(self.log_file_path, f"{self.start_time.strftime('%Y.%m.%d')}{self.start_time.strftime('_%H.%M.%S')}.txt")
        self.file_handle = open(self.log_file_path, 'w', encoding='utf-8')
    
    def _ensure_dir_exists(self):
        """确保日志文件所在的目录存在。"""
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def open(self, mode='w', encoding='utf-8'):
        """打开日志文件。"""
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.close()
        self.file_handle = open(self.log_file_path, mode, encoding=encoding)

    def write(self, content):
        """向日志文件写入内容。"""
        if self.file_handle:
            if isinstance(content, dict):
                # 如果内容是字典，则转换为格式化的JSON字符串
                self.file_handle.write(json.dumps(content, indent=4, ensure_ascii=False) + "\n")
            else:
                self.file_handle.write(str(content))
        else:
            print(f"Warning: Log file not open. Cannot write to {self.log_file_path}", file=sys.stderr)

    def close(self):
        """关闭日志文件。"""
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.close()
            self.file_handle = None

    def __enter__(self):
        """支持 with 语句。"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句，在退出时自动关闭文件。"""
        self.close()

    def __del__(self):
        """析构函数，确保在对象销毁时文件能被正确关闭。"""
        self.close()
