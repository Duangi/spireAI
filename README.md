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
1. **克隆本项目到本地**
```
git clone https://github.com/Duangi/spireAI.git
```
2. （可选）通过conda创建虚拟环境
```
conda init
conda create -n spireAI python=3.10.0
conda activate spireAI
```
ps:windows可能会遇到的问题：终端左侧没有出现(base)
- 解决方法：先解除限制，再重新初始化，最后重启终端
```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
conda init powershell
```
3. **进入项目目录，安装依赖**
```
cd spireAI
pip install -r requirements.txt
```
4. **（重要）配置Communication Mod**
    - 下载mod并运行一次游戏使mod生成配置文件
    - 关闭游戏，找到系统对应的文件
        - **Windows:** `%LOCALAPPDATA%\ModTheSpire\`
        - **Linux:** `~/.config/ModTheSpire/`
        - **Mac:** `~/Library/Preferences/ModTheSpire/`
    - 保存文件
    - 修改目录下的CommunicationMod/config.properties
    - **Windows:**`当前项目所在位置\\spireAI\\train.py`
    ```python
    # 以下为我的实际配置路径示例
    command=D\:\\Projects\\spireAI\\train.py
    # 如果使用了虚拟环境，这里需要加上虚拟环境的python解释器路径
    command=C\:\\Users\\Admin\\miniconda3\\envs\\spire310\\python D\:\\Projects\\spireAI\\train.py
    ```
    - 如何查找python解释器路径：
    ```bash
    python -c "import sys; print(sys.executable)"
    ```
    - **Mac:** `当前项目所在位置/spireAI/train.py`
    ```python
    command=/Users/duang/Projects/spireAI/train.py
    # 如果使用了虚拟环境,和windows不一样的是，需要在train.py文件最前方加上虚拟环境
    # 以下是我的实际路径示例
    #!/opt/miniconda3/envs/spire/bin/python3
    ```
5. 打开游戏，进入mod界面，点击Communication Mod，在里面点击左侧的按钮就可以启动训练了。