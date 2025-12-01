# 朝绝对路径进行txt内容的写入
import datetime
import os
import sys
import json

class AbsoluteLogger:
    """
    一个简单的日志记录器，用于将字符串内容写入指定路径的文件。
    """
    def __init__(self, log_file_path = "/Users/duang/Projects/spireAI/log/"):
        # 如果是windows系统，换一个绝对路径D:\Projects\spireAI
        if os.name == 'nt':
            log_file_path = "D:/Projects/spireAI/log/"
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

# 测试
if __name__ == "__main__":
    # 创建一个AbsoluteLogger对象，并指定日志文件的路径
    logger = AbsoluteLogger("/Users/duang/Projects/spireAI/log/")
    logger.start_episode()
    # 使用with语句打开日志文件，并写入内容
    # with logger:
    #     logger.write("This is a test log entry.\n")
    #     logger.write("Another log entry.\n")