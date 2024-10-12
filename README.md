# 因子投资组合优化应用

## **概述**

这是一个基于因子的投资组合优化应用，旨在通过均值-方差优化方法，构建符合特定 β 敞口和约束条件的投资组合。

## **功能**

1. **加载因子收益数据**：包括市场风险溢价（`Mkt_RF`）、规模因子（`SMB`）、价值因子（`HML`）、动量因子（`Momentum`）和无风险利率（`RF`）。
2. **加载股票数据**：包括股票代码、日期、月度收益和收盘价。
3. **回归分析**：对每只股票进行回归分析，估算其对各因子的 β 敞口。
4. **计算预期收益**：基于历史平均收益计算每只股票的预期收益。
5. **均值-方差优化**：在满足以下约束的情况下，优化投资组合：
   - 持仓股数为整数。
   - 每只股票的持仓不超过 4%。
   - 总持仓不超过 200 只。
   - 目标投资组合的 β 敞口符合用户指定。
   - 初始资金1000000
   - 净敞口 1000000
   - 总敞口 2000000
6. **输出结果**：生成持仓结果和投资组合 β 敞口的 CSV 文件。
---

## **安装和配置**

### **1. 克隆项目**

```bash
git clone https://github.com/yourusername/factor_investment_project.git
cd factor_investment_project
```


### ***2. 安装依赖 ###
使用以下命令安装所需的 Python 依赖：
```bash
pip install -r requirements.txt
```


### **3. 数据准备**

确保有以下数据文件：

1. **因子收益数据**：CSV 文件（例如：`F-F_Research_Data_Factors.CSV`），其中包含：
   - `Date`
   - `Mkt_RF`
   - `SMB`
   - `HML`
   - `Momentum`
   - `RF`

2. **股票数据**：SQLite 数据库（例如：`russell3000.db`），包含以下表：

   - **`stock_data`**：记录股票的收益和价格数据。
     - 列：`Ticker`, `Date`, `Return`, `Close`

## **4. 运行参数**

以下是所有支持的命令行参数：

```bash
python main.py \
    --factor_file F-F_Research_Data_Factors.CSV \
    --db_name russell3000.db \
    --start_date 2022-01-01 \
    --end_date 2023-01-01 \
    --industries "Technology,Healthcare" \
    --market_cap_min 5000000000 \
    --avg_volume_min 1000000 \
    --top_n 100 \
    --desired_beta "1.0,0.2,-0.15,0.3"
```

参数详情：
--db_name：
指定 SQLite 数据库的名称（默认：russell3000.db）。
--start_date：
回测的开始日期（格式：YYYY-MM-DD）。必须提供。
--end_date：
回测的结束日期（格式：YYYY-MM-DD）。必须提供。
--industries：
限制选股的行业（用逗号分隔），例如 "Technology,Healthcare"。默认使用所有行业。
--market_cap_min：
限制股票的最小市值（单位：美元）。默认不设置。
--avg_volume_min：
限制股票的最小平均交易量（单位：股）。默认不设置。
--top_n：
选择预期收益最高的前 N 只股票（默认：500）。
--desired_beta：
指定目标投资组合的 β 敞口（用逗号分隔），例如 "1.0,0.2,-0.15,0.3"。如果不指定，则通过均值-方差优化计算组合的 β。

## **5. 运行示例**

### **示例 1：不指定目标 β**

```bash
python main.py \
    --db_name russell3000.db \
    --start_date 2022-01-01 \
    --end_date 2023-01-01 \
    --industries "Technology,Healthcare" \
    --market_cap_min 5000000000 \
    --avg_volume_min 1000000 \
    --top_n 100
```
在这个示例中，应用会选择符合要求的前 100 只股票，并基于其历史收益构建投资组合。

### **示例 1：指定目标 β**
```bash
python main.py \
    --db_name russell3000.db \
    --start_date 2022-01-01 \
    --end_date 2023-01-01 \
    --industries "Finance,RealEstate" \
    --top_n 50 \
    --desired_beta "1.0,0.3,0.1,0.2"
```
在这个示例中，应用会选择金融和房地产行业的前 50 只股票，并优化投资组合以实现指定的 β 敞口。

## **6. 输出结果**

### **持仓结果 (`holdings.csv`)**

文件示例：

| Ticker | Holdings |
|--------|----------|
| AAPL   | 50       |
| MSFT   | 30       |
| AMZN   | 40       |
| GOOGL  | 20       |
| META   | 60       |

在 `holdings.csv` 文件中，每一行表示股票代码及其对应的持仓数量。

---

### **投资组合摘要 (`portfolio_summary.txt`)**

文件内容示例：
#### **解释：**

1. **投资组合预期收益**：  
   该数值表示根据历史数据和权重计算出的组合平均收益（每月或每年）。

2. **投资组合的风险（标准差）**：  
   风险是指组合收益的波动性，以标准差的形式表示。较低的标准差表示收益更稳定。

3. **投资组合的 β 敞口**：  
   如果用户未指定目标 β，应用会通过优化过程计算组合的实际 β 敞口。该例子显示的 β 敞口为 `[1.02, 0.28, -0.12, 0.31]`，分别对应于市场超额收益（`Mkt_RF`）、规模因子（`SMB`）、价值因子（`HML`）和动量因子（`Momentum`）。

这些输出文件将保存在项目的根目录下，可用于进一步分析和审阅投资组合的表现。

## **7. 日志**

应用会将运行过程中的信息输出到控制台，帮助用户了解每一步的执行情况和可能的错误。

### **日志格式示例：**

```text
[Main] INFO: 股票数据加载成功。
[Regression] INFO: 股票 AAPL 的回归分析完成。
[Regression] WARNING: 股票 XYZ 的数据点不足（10 个），跳过回归。
[Optimization] DEBUG: 当前目标函数值 = 66281.8243
[Optimization] INFO: 优化成功，持仓已保存到 holdings.csv。
[Main] INFO: 投资组合摘要已保存到 portfolio_summary.txt。
[Main] ERROR: 数据库连接失败：无法访问 russell3000.db。
日志级别说明：
```

INFO：
表示正常的应用流程信息。例如，数据加载成功、文件保存成功等。
WARNING：
表示非致命的警告信息。例如，某只股票的回归分析因数据不足被跳过。
DEBUG：
提供详细的调试信息。例如，优化过程中每次迭代的目标函数值。
ERROR：
表示严重错误，可能导致程序终止。例如，数据库连接失败或文件路径错误。

## **8. 贡献**

欢迎提交 Issue 和 Pull Request 来改进此项目。

如果在使用过程中发现任何问题，或有新的功能建议，请前往项目的 GitHub 仓库提交 Issue。

### **贡献指南：**

1. **Fork 项目**  
   点击项目页面右上角的 **Fork** 按钮，将仓库复制到自己的 GitHub 账号下。

2. **克隆仓库**  
   使用以下命令将 Fork 后的仓库克隆到本地：

   ```bash
   git clone https://github.com/yourusername/factor_investment_project.git
   cd factor_investment_project
   ```