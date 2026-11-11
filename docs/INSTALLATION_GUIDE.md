# 依赖安装指南

## 快速安装（推荐）

### 方式1：使用 pip 安装（最简单）

```bash
# 进入项目目录
cd "<项目根目录>"

# 安装依赖（不包含TA-Lib）
pip install -r requirements.txt
```

**注意**：TA-Lib 已被注释掉，因为它安装较复杂。如果您不需要高级技术分析功能，可以跳过。

## TA-Lib 安装（可选）

TA-Lib 是一个技术分析库，但安装较为复杂。**如果您只使用ROE对比功能，可以不安装。**

### Windows 用户安装 TA-Lib

#### 方法1：使用预编译的 wheel 文件（推荐）

1. 访问：https://github.com/cgohlke/talib-build/releases
2. 下载对应您Python版本的 `.whl` 文件，例如：
   - Python 3.10: `TA_Lib-0.4.28-cp310-cp310-win_amd64.whl`
   - Python 3.11: `TA_Lib-0.4.28-cp311-cp311-win_amd64.whl`
   - Python 3.12: `TA_Lib-0.4.28-cp312-cp312-win_amd64.whl`

3. 安装下载的文件：
```bash
pip install TA_Lib-0.4.28-cp310-cp310-win_amd64.whl
```

#### 方法2：使用 conda 安装

```bash
conda install -c conda-forge ta-lib
```

### Linux/Mac 用户安装 TA-Lib

```bash
# 安装 C 库
# Ubuntu/Debian
sudo apt-get install ta-lib

# macOS
brew install ta-lib

# 然后安装 Python 包
pip install TA-Lib
```

## 检查 Python 版本

```bash
python --version
```

## 完整安装步骤

### 1. 创建虚拟环境（推荐）

```bash
# 使用 venv
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 2. 升级 pip

```bash
python -m pip install --upgrade pip
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python -c "import tushare; import pandas; import flask; print('安装成功！')"
```

## 常见问题

### Q1: pip 安装速度慢？

使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 某个包安装失败？

尝试单独安装：

```bash
pip install tushare
pip install pandas
pip install flask
pip install flask-cors
pip install matplotlib
pip install loguru
```

### Q3: 提示权限错误？

使用 `--user` 参数：

```bash
pip install -r requirements.txt --user
```

### Q4: conda 和 pip 混用问题？

建议统一使用 pip：

```bash
# 如果在 conda 环境中
conda deactivate

# 创建新的 venv 环境
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 最小依赖（仅运行 ROE 对比功能）

如果您只想使用 ROE 对比功能，只需安装以下核心包：

```bash
pip install tushare pandas flask flask-cors matplotlib loguru openpyxl
```

## 验证 ROE 对比功能

安装完成后，测试 ROE 对比功能：

```bash
# 启动 Web 系统
python start.py

# 或运行测试脚本
python scripts/test_roe_compare.py
```

## 配置 Tushare Token

安装完成后，需要配置 Tushare Token：

1. 注册 Tushare 账号：https://tushare.pro/register
2. 获取 Token
3. 编辑 `config/api_config.py`：

```python
TUSHARE_TOKEN = "your_token_here"
```

## 完整依赖列表

当前项目依赖：

- **tushare** (>=1.2.89) - 金融数据接口
- **pandas** (>=1.5.0) - 数据处理
- **numpy** (>=1.23.0) - 数值计算
- **matplotlib** (>=3.6.0) - 图表绘制
- **seaborn** (>=0.12.0) - 统计可视化
- **flask** (>=2.3.0) - Web 框架
- **flask-cors** (>=4.0.0) - 跨域支持
- **loguru** (>=0.7.0) - 日志记录
- **openpyxl** (>=3.1.0) - Excel 读写
- **xlrd** (>=2.0.0) - Excel 读取
- **plotly** (>=5.14.0) - 交互式图表（可选）
- **pyecharts** (>=2.0.0) - ECharts 图表（可选）
- **requests** (>=2.28.0) - HTTP 请求
- **python-dateutil** (>=2.8.0) - 日期处理

## 下一步

安装完成后，您可以：

1. **启动 Web 系统**：
```bash
python start.py
```

2. **访问系统**：
http://localhost:5000

3. **体验 ROE 对比功能**：
   - 搜索公司
   - 切换到多选对比模式
   - 勾选多个公司
   - 点击 ROE 对比分析

祝您使用愉快！🚀





