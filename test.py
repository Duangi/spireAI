import os


def get_root_dir():
    return os.path.abspath(os.path.dirname(__file__))

print(get_root_dir())