import os
import glob
from spirecomm.utils.path import get_root_dir

MEMORY_DIR = os.path.join(get_root_dir(), "data", "memory")
print(f"MEMORY_DIR: {MEMORY_DIR}")

pattern = os.path.join(MEMORY_DIR, "**", "*.pt")
print(f"Pattern: {pattern}")

files = glob.glob(pattern, recursive=True)
print(f"Found files: {files}")

# Test manual path
manual_pattern = "data/memory/**/*.pt"
print(f"Manual Pattern: {manual_pattern}")
files_manual = glob.glob(manual_pattern, recursive=True)
print(f"Found files (manual): {files_manual}")
