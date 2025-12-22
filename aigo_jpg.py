import os
import sys
import time

# ================= 1. 代理设置 =================
PROXY_URL = "http://127.0.0.1:7890" 
os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL
# ==============================================

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
import matplotlib.patches as patches
import pandas as pd
import wandb

# ================= 2. 字体与外观 =================
FONT_PATH = "C:/Windows/Fonts/msyh.ttc" 
try:
    if os.path.exists(FONT_PATH):
        MY_FONT = font_manager.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = MY_FONT.get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        MY_FONT = None
except:
    MY_FONT = None
plt.rcParams['axes.unicode_minus'] = False 

# ================= 3. 配置 =================
MODELS_DIR = r"D:\Projects\spireAI\models" # 请确认本地模型路径

WIDTH_PX = 1480
HEIGHT_PX = 720
DPI = 120 

THEME = {
    "bg": "#ffffff",
    "panel_bg": "#f8f9fa",
    "grid": "#eaecf0",
    "text_main": "#101828",
    "text_sub": "#667085",
    "col_max_q":  "#007AFF", 
    "col_reward": "#FF3B30", 
    "col_avg_q":  "#34C759", 
}

WANDB_ENTITY = "duang"          
WANDB_PROJECT = "spire-ai-trainer"

KEYS = {
    "reward": "avg_reward",
    "max_q": "max_q_value",
    "avg_q": "avg_q_value",
    "temp": "temperature",
    "speed_min": "train/steps_per_min"
}

# ================= 4. 数据逻辑 =================
def get_latest_model_path(player_class=None):
    target_dir = MODELS_DIR
    if player_class:
        class_dir = os.path.join(MODELS_DIR, str(player_class))
        if os.path.exists(class_dir):
            target_dir = class_dir
    if not os.path.exists(target_dir): return None, 0
    try:
        model_files = [f for f in os.listdir(target_dir) if f.startswith("step_") and f.endswith(".pth")]
    except: return None, 0
    
    latest_step = 0
    latest_path = None
    if not model_files: return None, 0
    for f in model_files:
        try:
            step_num = int(f[len("step_"):-len(".pth")])
            if step_num > latest_step:
                latest_step = step_num
                latest_path = os.path.join(target_dir, f)
        except: continue
    return latest_path, latest_step

def get_latest_run_data():
    try:
        api = wandb.Api()
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", order="-created_at", per_page=1)
        if not runs: return None, "No Runs", 0
        run = runs[0]
        all_keys = list(KEYS.values()) + ["_step"]
        # 保持 10万 采样以确保稀疏数据能被搜到
        df = run.history(keys=all_keys, samples=100000, pandas=True)
        if df.empty: return None, run, 0
        return df, run, run.lastHistoryStep
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None, None, 0

def get_robust_scalar(run, df, key_id):
    key = KEYS[key_id]
    
    # 逻辑优化：查找最后一个非零值
    # 1. 先看 History (df)
    if key in df.columns:
        valid = df[key].dropna()
        # 过滤掉 0 (假设速度/温度不会真的是0)
        valid = valid[valid > 0.0001]
        
        if not valid.empty:
            val = valid.iloc[-1]
            if val > 100: return f"{int(val):,}"
            return f"{val:.4f}"

    # 2. 如果 History 里全是0或空，再看 Summary
    if key in run.summary:
        val = run.summary[key]
        if isinstance(val, (int, float)) and val > 0.0001: 
            if val > 100: return f"{int(val):,}"
            return f"{val:.4f}"
            
    return "-"

def get_metric_trend(df, key_id):
    key = KEYS[key_id]
    if key not in df.columns: return None, None
    s = df[key].dropna()
    if s.empty: return None, None
    steps = df.loc[s.index, "_step"]
    return steps, s

# ================= 5. 绘图主逻辑 =================
def generate_dashboard():
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 开始生成看板...", end="\n", flush=True)
    
    df, run, _ = get_latest_run_data()
    if df is None: 
        print(" ❌ 无数据")
        return

    _, local_model_step = get_latest_model_path()
    
    plt.style.use('default')
    fig = plt.figure(figsize=(WIDTH_PX/DPI, HEIGHT_PX/DPI), dpi=DPI)
    fig.patch.set_facecolor(THEME["bg"])
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                           left=0.06, right=0.96, top=0.92, bottom=0.08,
                           wspace=0.18, hspace=0.28)

    # --- 左上角: 信息面板 ---
    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.set_axis_off()
    rect = patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0", ec="none", fc=THEME["panel_bg"], transform=ax_info.transAxes, zorder=0)
    ax_info.add_patch(rect)

    # 获取数据 (使用优化后的非0逻辑)
    val_temp = get_robust_scalar(run, df, "temp")
    val_speed_min = get_robust_scalar(run, df, "speed_min")
    val_local_step = f"{local_model_step:,}" if local_model_step > 0 else "None"

    # 字体
    f_title = MY_FONT.copy(); f_title.set_size(24); f_title.set_weight('bold')
    f_label = MY_FONT.copy(); f_label.set_size(9); f_label.set_weight('bold')
    # 三个数字统一大小
    f_val_lg = MY_FONT.copy(); f_val_lg.set_size(22); f_val_lg.set_weight('bold')

    # A. 标题区
    ax_info.text(0.06, 0.78, "杀戮尖塔 AI 训练", color=THEME["text_main"], fontproperties=f_title, transform=ax_info.transAxes)
    ax_info.plot([0.06, 0.94], [0.65, 0.65], color="#d0d5dd", linewidth=1, transform=ax_info.transAxes)

    # B. 数据区 - 三列横向排布 (Equal Size)
    
    # Column 1: 温度 (x=0.06)
    ax_info.text(0.06, 0.48, "EXPLORE / 温度", color="#F5A623", fontproperties=f_label, transform=ax_info.transAxes)
    ax_info.text(0.06, 0.25, val_temp, color=THEME["text_main"], fontproperties=f_val_lg, transform=ax_info.transAxes)
    
    # Column 2: 速度 (x=0.38)
    ax_info.text(0.38, 0.48, "SPEED (steps/min)", color="#17a2b8", fontproperties=f_label, transform=ax_info.transAxes)
    ax_info.text(0.38, 0.25, val_speed_min, color=THEME["text_main"], fontproperties=f_val_lg, transform=ax_info.transAxes)

    # Column 3: 模型 (x=0.70)
    ax_info.text(0.70, 0.48, "LATEST MODEL(steps)", color="#9013FE", fontproperties=f_label, transform=ax_info.transAxes)
    ax_info.text(0.70, 0.25, val_local_step, color=THEME["text_main"], fontproperties=f_val_lg, transform=ax_info.transAxes)

    # --- 绘图函数 ---
    def draw_chart(loc, key_id, title_cn, title_en, color):
        ax = fig.add_subplot(loc)
        ax.set_facecolor(THEME["bg"])
        
        f_head = MY_FONT.copy(); f_head.set_size(11); f_head.set_weight('bold')
        f_curr = MY_FONT.copy(); f_curr.set_size(14); f_curr.set_weight('bold')
        
        steps, s = get_metric_trend(df, key_id)
        
        if s is not None:
            ax.plot(steps, s, color=color, alpha=0.1, linewidth=0.8)
            
            if len(s) > 20:
                smooth = s.ewm(alpha=0.1).mean()
                ax.plot(steps, smooth, color=color, linewidth=2)
                ax.fill_between(steps, smooth, smooth.min(), color=color, alpha=0.08)
                
                # Max Q 取最大值，其他取最新值
                if key_id == "max_q":
                    display_val = s.max()
                else:
                    display_val = smooth.iloc[-1]
            else:
                display_val = s.max() if key_id == "max_q" else s.iloc[-1]
                ax.plot(steps, s, color=color, linewidth=1.5)

            ax.set_title(f"{title_cn} | {title_en}", loc='left', color=THEME["text_sub"], fontproperties=f_head, pad=10)
            ax.text(1.0, 1.05, f"{display_val:.2f}", transform=ax.transAxes, ha='right', va='bottom', color=color, fontproperties=f_curr)
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', color="#ccc")

        ax.grid(True, linestyle=':', color=THEME["grid"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(THEME["grid"])
        ax.tick_params(axis='both', colors="#999", labelsize=8)

    # 绘图
    draw_chart(gs[0, 1], "max_q", "最大价值", "MAX Q-VALUE", THEME["col_max_q"])
    draw_chart(gs[1, 0], "reward", "平均奖励", "AVG REWARD", THEME["col_reward"])
    draw_chart(gs[1, 1], "avg_q", "平均价值", "AVG Q-VALUE", THEME["col_avg_q"])

    try:
        plt.savefig("wandb_spire_dashboard.png", dpi=DPI, facecolor=THEME["bg"])
    except Exception as e:
        print(f" ⚠️ 保存失败: {e}")
    finally:
        plt.close()
    
    print(f" ✅ 完成! ⏱️ {time.time() - t_start:.2f}s")

if __name__ == "__main__":
    while True:
        generate_dashboard()
        print("Waiting 30s...\n")
        time.sleep(30)