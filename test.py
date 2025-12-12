import re
filename = "step_563_45_20251212_090249.pt"
match = re.search(r'step_(\d+)', filename)
if match:
    print(int(match.group(1)))
else:
    print(float('inf'))