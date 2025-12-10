import os
import json
import pandas as pd
import glob

def view_latest_log():
    # 1. Find the latest wandb run directory
    wandb_dir = os.path.join(os.path.dirname(__file__), "wandb")
    if not os.path.exists(wandb_dir):
        print("No 'wandb' directory found.")
        return

    # Get all directories starting with 'run-' or 'offline-run-'
    runs = glob.glob(os.path.join(wandb_dir, "*run-*"))
    if not runs:
        print("No run directories found in 'wandb/'.")
        return

    # Sort by modification time, newest first
    latest_run = max(runs, key=os.path.getmtime)
    print(f"Checking latest run: {os.path.basename(latest_run)}")

    # 2. Find the table JSON file
    # Path pattern: wandb/run-XXX/files/media/table/Episode_Trace_*.table.json
    table_dir = os.path.join(latest_run, "files", "media", "table")
    if not os.path.exists(table_dir):
        print(f"No table directory found at: {table_dir}")
        print("Did the run finish successfully and commit the table?")
        return

    json_files = glob.glob(os.path.join(table_dir, "*.json"))
    if not json_files:
        print("No .json table files found.")
        return

    # Pick the latest json file (usually Episode_Trace)
    latest_json = max(json_files, key=os.path.getmtime)
    print(f"Reading table file: {os.path.basename(latest_json)}\n")

    # 3. Load and Display
    try:
        with open(latest_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        columns = data.get("columns", [])
        rows = data.get("data", [])
        
        if not rows:
            print("Table is empty.")
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

        print("="*80)
        print(f"Total Steps Recorded: {len(df)}")
        print("="*80)
        
        # Show the last 5 steps
        print("Last 5 Steps of Training Data:")
        print(df.tail(5).to_string())
        print("="*80)
        print("\nTip: You can modify this script to show more rows or specific columns.")

    except Exception as e:
        print(f"Error reading/parsing JSON: {e}")

if __name__ == "__main__":
    view_latest_log()
