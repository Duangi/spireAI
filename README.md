# spireAI
基于DQN的杀戮尖塔ai

## 环境需求
### 操作系统
windows / mac
目前仅在win11 N卡环境 / macbook air m4上测试过可行性
### 游戏相关
1. （必需）steam 杀戮尖塔游戏本体
2. （必需）相关mod:
    - Communication Mod（自动化关键）
    - BaseMod（前置mod）
    - StSLib（前置mod）
    - ModTheSpire（前置mod）

### 环境相关
1. （必需）python3.8+ （CommuncationMod刚需3.5+，再考虑到
2. （必需）pytorch（应该来说是和自己的python
3. （可选）CUDA

- 当前已测试通过的环境：

- 操作系统：Macbook air m4
    - Python 版本：3.8.20
    - Torch 版本：2.4.1
    - TorchVision 版本：0.19.1
- 操作系统：Win11
    - Python 版本：3.10.0
    - Torch 版本：2.9.1+cu130
    - TorchVision 版本：0.24.1+cu130
    - CUDA 版本：13.0

## 安装步骤
（前提1）下载好并安装好杀戮尖塔游戏本体，并安装好上述mod
（前提2）安装好python3.8+环境，并配置好pip
1. 克隆本项目到本地
```
git clone https://github.com/Duangi/spireAI.git
```
2. 进入项目目录，安装依赖
```
cd spireAI
pip install -r requirements.txt
```
3. （重要）配置Communication Mod
    - 下载mod并运行一次游戏使mod生成配置文件
    - 关闭游戏，找到系统对应的文件
        - **Windows:** `%LOCALAPPDATA%\ModTheSpire\`
        - **Linux:** `~/.config/ModTheSpire/`
        - **Mac:** `~/Library/Preferences/ModTheSpire/`
    - 保存文件
    - 修改目录下的CommunicationMod/config.properties
    - **Windows:**`当前项目所在位置\\spireAI\\train.py`
    - **Mac:** `当前项目所在位置/spireAI/train.py`