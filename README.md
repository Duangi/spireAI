# spireAI 安装配置指南（基于DQN的杀戮尖塔AI）

# 一、项目简介

spireAI 是基于 DQN（Deep Q-Network）算法实现的《杀戮尖塔》AI 项目，通过游戏 Mod 与 Python 脚本联动，实现 AI 自动训练与决策。

# 二、环境需求

## 2.1 操作系统兼容性

支持系统：Windows / Mac
已验证环境：

- Windows 11（N 卡环境）

- MacBook Air M4

## 2.2 游戏及必备 Mod

必需组件（缺一不可）：

1. Steam 平台《杀戮尖塔》游戏本体

2. 核心功能 Mod：Communication Mod（自动化联动关键）

3. BaseMod（前置mod）

4. StSLib（前置mod）

5. ModTheSpire（Mod 管理器）

## 2.3 开发环境依赖

1. 必需：Python 3.8+（Communication Mod 最低要求 3.5+，推荐 3.8+ 兼容更优）

2. 必需：PyTorch（需与 Python 版本匹配，建议参考下方验证环境）

3. 可选：CUDA（Windows 环境建议安装，加速模型训练；Mac 自带了mps加速，无需下载）

## 2.4 已验证通过的环境配置

|操作系统|Python 版本|Torch 版本|TorchVision 版本|CUDA 版本|
|---|---|---|---|---|
|MacBook Air M4|3.8.20|2.4.1|0.19.1|默认自带mps|
|Windows 11|3.10.0|2.9.1+cu130|0.24.1+cu130|13.0|
# 三、安装步骤

前置条件：已完成《杀戮尖塔》本体安装 + 所有必需 Mod 安装；已安装 Python 3.8+ 并配置好 pip 环境。

## 3.1 克隆项目到本地

打开终端（Windows：CMD/PowerShell；Mac：终端），执行以下命令：

```bash

git clone https://github.com/Duangi/spireAI.git
```

## 3.2 （可选）创建并激活 Conda 虚拟环境

建议使用虚拟环境隔离依赖，避免版本冲突：

```bash

# 初始化 Conda（首次使用需执行）
conda init

# 创建虚拟环境（指定 Python 版本，示例为 3.10.0，可按需调整）
conda create -n spireAI python=3.10.0

# 激活虚拟环境
conda activate spireAI
```

### Windows 常见问题：激活后终端左侧未显示 (spireAI) 标识

原因：PowerShell 执行策略限制，导致 Conda 初始化失效，解决步骤：

```powershell

# 1. 解除当前用户的执行策略限制（仅需执行一次）
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. 重新初始化 Conda 对 PowerShell 的支持
conda init powershell

# 3. 关闭所有终端窗口，重新打开后再激活环境
conda activate spireAI
```

## 3.3 安装项目依赖包

进入项目根目录，通过 requirements.txt 安装所有依赖：

```bash

# 进入项目目录（请替换为你的实际克隆路径）
cd spireAI

# 安装依赖
pip install -r requirements.txt
```

## 3.4 关键配置：Communication Mod 联动设置

该步骤是 AI 脚本与游戏联动的核心，需手动修改 Mod 配置文件，指定 Python 脚本路径：

### 步骤 1：生成 Mod 配置文件

1. 安装好所有 Mod 后，启动一次《杀戮尖塔》游戏（无需进入对局）；
2. 直接关闭游戏，Mod 会自动生成配置文件到系统对应目录。

### 步骤 2：定位配置文件路径

- Windows：`%LOCALAPPDATA%\ModTheSpire\`（可直接复制到文件资源管理器地址栏打开）

- Linux：`~/.config/ModTheSpire/`

- Mac：`~/Library/Preferences/ModTheSpire/`

### 步骤 3：修改 CommunicationMod 配置

在上述目录中，找到 `CommunicationMod/config.properties` 文件，用记事本/文本编辑器打开，修改 `command` 字段：

#### Windows 配置示例

核心：指定虚拟环境的 Python 解释器路径 + train.py 脚本路径（避免使用中文路径）：

```properties

# 格式：[虚拟环境Python路径] [train.py脚本路径]
command=C\:\\Users\\Admin\\miniconda3\\envs\\spire310\\python D\:\\Projects\\spireAI\\train.py
```

#### Mac 配置示例

核心：两种方式任选（推荐方式 1，更稳定）：

1. 方式 1：在 train.py 头部添加虚拟环境 Python 路径（推荐）
        `# 编辑 train.py 文件，在第一行添加以下内容（替换为你的虚拟环境 Python 路径）
#!/opt/miniconda3/envs/spire/bin/python3

# 然后在 config.properties 中指定 train.py 路径
command=/Users/duang/Projects/spireAI/train.py`

2. 方式 2：直接在 command 中指定完整路径
        `command=/opt/miniconda3/envs/spire/bin/python3 /Users/duang/Projects/spireAI/train.py`

### 如何获取虚拟环境的 Python 解释器路径？

激活虚拟环境后，在终端执行以下命令，直接输出路径：

```bash

python -c "import sys; print(sys.executable)"
```

## 3.5 启动 AI 训练

1. 打开《杀戮尖塔》游戏，进入 Mod 管理界面；

2. 确保 Communication Mod 已启用，点击该 Mod 进入配置界面；

3. 点击左侧对应按钮，启动 AI 训练（此时会自动调用 Python 脚本）。

# 四、常见问题排查

- 问题 1：启动训练后无响应 → 检查 config.properties 中 command 路径是否正确（避免中文/空格）、Python 依赖是否安装完整；

- 问题 2：PyTorch 导入失败 → 确认 Torch 版本与 Python 版本匹配，Windows 环境需检查 CUDA 与 Torch 兼容性；

- 问题 3：Mod 加载失败 → 检查 BaseMod/StSLib/ModTheSpire 是否为最新版本，按顺序安装前置 Mod。
> （注：文档部分内容可能由 AI 生成）