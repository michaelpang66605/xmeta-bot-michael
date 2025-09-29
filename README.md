

# XMeta Bot - pzm

用于 XMeta 资产的高级实时价格追踪、预测与分析。

## 快速开始

```bash
python main.py
```

选择：
- **选项 1**：交互式 Streamlit 仪表盘（网页应用）
- **选项 2**：命令行分析

## 文件说明

- `main.py` - 主程序（仪表盘 + 命令行）
- `README.md` - 本文件

## 环境搭建

1. **安装依赖：**
   ```bash
   pip install streamlit plotly numpy pandas requests
   ```

2. **添加 XMeta API 凭证**（在 `main.py` 中）：
   ```python
   XMETA_API_KEY = "输入API"
   XMETA_TOKEN = "如果有输入TOKEN"
   XMETA_USER_ID = "USER_ID"
   ```

3. **运行：**
   ```bash
   python main.py
   ```

## 主要功能

- **实时 XMeta 价格追踪**（API 集成）
- **蒙特卡洛模拟**，预测未来 30 天价格
- **风险指标：**
  - 年化波动率
  - 夏普比率
  - 价值风险（VaR，95%）
- **价格预警邮件**，当价格突破 $60 或 $70 时发送
- **历史价格记录**（CSV 文件）
- **交互式图表**（Plotly）
- **Streamlit 仪表盘**，可视化与分析

## 分析与函数

- **技术指标：**
  - 简单/指数移动平均线（SMA, EMA）
  - 相对强弱指数（RSI）
  - MACD 指标
  - 布林带
  - 类 ATR 波动率
- **高级分析：**
  - 与基准相关性与贝塔值（支持 CSV 上传）
  - 季节性检测
  - 市场状态检测（基于波动率）
  - 场景分析（冲击模拟）
- **预测功能：**
  - 蒙特卡洛模拟（1000 次，30 天）
  - 预测均值与置信区间（5%，95%）

## 仪表盘

选项 1 会在浏览器打开：`http://localhost:8501`

## 常见问题

- **API 失败**：请在 `main.py` 中添加有效的 XMeta 凭证
- **浏览器未自动打开**：请手动访问 `http://localhost:8501`
- **邮件预警**：使用 Gmail 应用密码（可在代码中修改）
- **加载缓慢**：可能是 API 超时或网络问题

## 备注

- 会自动创建 `xmeta_price_log.csv` 用于历史数据记录
- 所有分析与指标均在 Python 本地计算，无需外部服务
- 命令行与仪表盘模式共用同一代码

---
**作者：** 庞子鸣
