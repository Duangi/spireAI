# 朝绝对路径进行txt内容的写入
import datetime
from math import log
import os
import sys
import json
from enum import Enum
from spirecomm.utils.path import get_root_dir
class LogType(Enum):
    """
    日志类型枚举类，用于指定日志的级别。
    """
    PROGRESS = 1
    REWARD = 2
    STATE = 3
    QVALUE = 4
# 获取当前项目的绝对路径的根目
PROJECT_ROOT = get_root_dir()
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
# 将以下类改为使用绝对路径
class AbsoluteLogger:
    """
    一个简单的日志记录器，用于将字符串内容写入指定路径的文件。
    """
    def __init__(self, log_type:LogType = LogType.PROGRESS, max_file_num = 10):
        self.max_file_num = max_file_num
        self.step_count = 0
        # 如果是windows系统，换一个绝对路径D:\Projects\spireAI
        if log_type == LogType.PROGRESS:
            if os.name == 'nt':
                log_file_path = os.path.join(LOGS_DIR, "progress_log")
            else:
                log_file_path = "/Users/duang/Projects/spireAI/log/"
        elif log_type == LogType.REWARD:
            if os.name == 'nt':
                log_file_path = os.path.join(LOGS_DIR, "reward_log")
            else:
                log_file_path = "/Users/duang/Projects/spireAI/reward_log/"
        elif log_type == LogType.STATE:
            if os.name == 'nt':
                log_file_path = os.path.join(LOGS_DIR, "state_log")
            else:
                log_file_path = "/Users/duang/Projects/spireAI/state_log/"
        elif log_type == LogType.QVALUE:
            if os.name == 'nt':
                log_file_path = os.path.join(LOGS_DIR, "qvalue_log")
            else:
                log_file_path = "/Users/duang/Projects/spireAI/qvalue_log/"
        else:
            raise ValueError(f"未知的日志类型: {log_type}")

        self.log_file_path = log_file_path
        self.file_handle = None
        self._ensure_dir_exists()

    def start_episode(self, filename_suffix=""):
        """在一局游戏开始时调用。"""
        self.step_count = 0
        self.start_time = datetime.datetime.now()
        filename = f"{self.start_time.strftime('%Y.%m.%d')}{self.start_time.strftime('_%H.%M.%S')}{filename_suffix}.txt"
        self.log_file_path = os.path.join(self.log_file_path, filename)
        # 确保目标目录存在（防止在并发/不同启动方式下出错）
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.file_handle = open(self.log_file_path, 'w', encoding='utf-8')
    
    def _ensure_dir_exists(self):
        """确保日志文件所在的目录存在。"""
        # 如果当前 log_file_path 指向的是目录（初始化时通常为目录），直接确保该目录存在；
        # 如果它已经是文件路径（start_episode 之后），则取其父目录。
        path = self.log_file_path
        # 判断 path 是否更像是文件：有扩展名或以 ".txt" 结尾
        is_file_like = os.path.splitext(path)[1] != "" or path.lower().endswith('.txt')
        log_dir = os.path.dirname(path) if is_file_like else path
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
 
    def open(self, mode='w', encoding='utf-8'):
        """打开日志文件。"""
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.close()
        # 在打开文件前确保目录存在（兜底）
        try:
            log_dir = os.path.dirname(self.log_file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        except Exception:
            # 如果目录创建失败也不阻塞写入操作（open 会报错）
            pass
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
            raise RuntimeError("日志文件未打开，无法写入内容。")

        # 写完之后，查看是否超出最大文件数，若超出则删除最旧的文件
        log_dir = os.path.dirname(self.log_file_path)
        self.step_count += 1
        if self.step_count % 10 == 0:  # 每10次写入检查一次，减少文件系统操作
            try:
                all_files = [f for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))]
                if len(all_files) > self.max_file_num:
                    # 按修改时间排序，删除最旧的文件
                    all_files.sort(key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))
                    files_to_delete = all_files[:len(all_files) - self.max_file_num]
                    for f in files_to_delete:
                        os.remove(os.path.join(log_dir, f))
            except Exception:
                # 如果删除文件失败，不阻塞正常写入
                pass
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
