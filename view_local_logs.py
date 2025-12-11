import os
import json
import pandas as pd
import glob

def view_latest_log():
    # 1. Find the latest wandb run directory
    wandb_dir = os.path.join(os.path.dirname(__file__), "wandb")
    if not os.path.exists(wandb_dir):
        return

    # Get all directories starting with 'run-' or 'offline-run-'
    runs = glob.glob(os.path.join(wandb_dir, "*run-*"))
    if not runs:
        return

    # Sort by modification time, newest first
    latest_run = max(runs, key=os.path.getmtime)

    # 2. Find the table JSON file
    # Path pattern: wandb/run-XXX/files/media/table/Episode_Trace_*.table.json
    table_dir = os.path.join(latest_run, "files", "media", "table")
    if not os.path.exists(table_dir):
        return

    json_files = glob.glob(os.path.join(table_dir, "*.json"))
    if not json_files:
        return

    # Pick the latest json file (usually Episode_Trace)
    latest_json = max(json_files, key=os.path.getmtime)

    # 3. Load and Display
    try:
        with open(latest_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        columns = data.get("columns", [])
        rows = data.get("data", [])
        
        if not rows:
            return

        df = pd.DataFrame(rows, columns=columns)
        
        # Clean up HTML tags for terminal display
        # Replace <br> with \n, remove <b> tags
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace("<br>", "\n").str.replace("<b>", "").str.replace("</b>", "")

        # Set pandas display options to show full content
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 50)


    except Exception as e:
        raise RuntimeError(f"无法加载或显示日志文件: {e}")

if __name__ == "__main__":
    view_latest_log()
