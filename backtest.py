# portfolio_optimization.py
import pandas as pd
import numpy as np
import cvxpy as cp
from sqlalchemy import create_engine

def load_factors(db_name='russell3000.db'):
    engine = create_engine(f'sqlite:///{db_name}')
    factors = pd.read_sql('SELECT * FROM factors', engine, parse_dates=['Date'], index_col='Date')
    return factors

def load_stock_data(db_name='russell3000.db'):
    engine = create_engine(f'sqlite:///{db_name}')
    df = pd.read_sql('SELECT * FROM stock_data', engine)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def calculate_asset_exposures(factors, db_name='russell3000.db'):
    df = load_stock_data(db_name)
    df = df.dropna(subset=['Market_Cap', 'Book_Value', 'Close', 'Volume'])
    df['Book_to_Market'] = df['Book_Value'] / df['Market_Cap']
    df['Return'] = df.groupby('Ticker')['Close'].pct_change()
    df['Momentum'] = df.groupby('Ticker')['Return'].transform(lambda x: x.rolling(window=126).sum())
    df = df.dropna(subset=['Momentum'])
    
    # 选择最新日期的数据
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date]
    
    # 初始化暴露 DataFrame
    exposures = pd.DataFrame(index=latest_data['Ticker'].unique())
    
    # SMB 因子暴露
    group = latest_data.sort_values('Market_Cap', ascending=True)
    median_cap = group['Market_Cap'].median()
    small_cap = group[group['Market_Cap'] <= median_cap]
    big_cap = group[group['Market_Cap'] > median_cap]
    
    small_high_bm = small_cap[small_cap['Book_to_Market'] >= small_cap['Book_to_Market'].quantile(0.9)]
    small_low_bm = small_cap[small_cap['Book_to_Market'] <= small_cap['Book_to_Market'].quantile(0.1)]
    big_high_bm = big_cap[big_cap['Book_to_Market'] >= big_cap['Book_to_Market'].quantile(0.9)]
    big_low_bm = big_cap[big_cap['Book_to_Market'] <= big_cap['Book_to_Market'].quantile(0.1)]
    
    exposures['SMB'] = 0
    exposures.loc[small_high_bm['Ticker'], 'SMB'] += 0.5
    exposures.loc[small_low_bm['Ticker'], 'SMB'] += 0.5
    exposures.loc[big_high_bm['Ticker'], 'SMB'] -= 0.5
    exposures.loc[big_low_bm['Ticker'], 'SMB'] -= 0.5
    
    # HML 因子暴露
    high_bm_large = big_cap[big_cap['Book_to_Market'] >= big_cap['Book_to_Market'].quantile(0.9)]
    low_bm_large = big_cap[big_cap['Book_to_Market'] <= big_cap['Book_to_Market'].quantile(0.1)]
    high_bm_small = small_cap[small_cap['Book_to_Market'] >= small_cap['Book_to_Market'].quantile(0.9)]
    low_bm_small = small_cap[small_cap['Book_to_Market'] <= small_cap['Book_to_Market'].quantile(0.1)]
    
    hml_large = high_bm_large['Return'].mean() - low_bm_large['Return'].mean()
    hml_small = high_bm_small['Return'].mean() - low_bm_small['Return'].mean()
    
    hml_return = 0.5 * hml_large + 0.5 * hml_small
    exposures['HML'] = hml_return
    
    # Momentum 因子暴露
    winners = latest_data[latest_data['Momentum'] >= latest_data['Momentum'].quantile(0.9)]
    losers = latest_data[latest_data['Momentum'] <= latest_data['Momentum'].quantile(0.1)]
    exposures['Momentum'] = winners['Return'].mean() - losers['Return'].mean()
    
    # 替代 RM_RF（市场风险溢价）
    if latest_date in factors.index:
        exposures['RM_RF'] = factors.loc[latest_date, 'RM_RF']
    else:
        exposures['RM_RF'] = 0
    
    # 填充缺失值
    exposures = exposures.fillna(0)
    
    return exposures

def mean_variance_optimization(factors, gamma=5.0):
    # 标准化因子数据（z-score）
    factors = (factors - factors.mean()) / factors.std()
    
    E = factors.mean().values  # 预期收益向量
    Sigma = factors.cov().values  # 因子协方差矩阵
    
    n_factors = len(E)
    
    # 定义优化变量：因子权重
    w_factors = cp.Variable(n_factors)
    
    # 定义效用函数：U = w^T E - (gamma / 2) * w^T Sigma w
    utility = E @ w_factors - (gamma / 2) * cp.quad_form(w_factors, Sigma)
    
    # 定义优化问题
    prob = cp.Problem(cp.Maximize(utility))
    
    # 求解优化问题
    prob.solve()
    
    if w_factors.value is None:
        raise ValueError("因子权重优化未能收敛，请检查数据和参数设置。")
    
    factor_weights = pd.Series(w_factors.value, index=factors.columns)
    return factor_weights

def assign_weights_to_securities(factor_weights, asset_exposures, initial_capital=1000000, gamma=5.0):
    # 计算每只股票的预期收益
    # E[r] = exposures * factor_weights
    E_r = asset_exposures.values @ factor_weights.values  # 预期收益向量
    
    # 加载股票价格
    stock_data = load_stock_data()
    latest_date = stock_data['Date'].max()
    latest_prices = stock_data[stock_data['Date'] == latest_date].set_index('Ticker')['Close']
    
    tickers = asset_exposures.index.tolist()
    prices = latest_prices.loc[tickers].values  # 确保价格顺序与tickers一致
    
    # 定义优化变量：持仓股数（整数）
    h = cp.Variable(len(tickers), integer=True)
    
    # 计算投资组合预期收益
    portfolio_return = E_r @ h
    
    # 计算投资组合风险（方差）
    # 这里简化为 w^T Sigma w，假设 Sigma 是因子协方差矩阵
    # 如果有更详细的股票协方差矩阵，可以替换为更精确的风险度量
    Sigma_factors = asset_exposures.multiply(factor_weights, axis=1).cov().values
    portfolio_risk = cp.quad_form(h, Sigma_factors)
    
    # 定义效用函数：U = E[r] - (gamma / 2) * Var(R_p)
    utility = portfolio_return - (gamma / 2) * portfolio_risk
    
    # 设定目标：最大化效用
    objective = cp.Maximize(utility)
    
    # 约束条件
    constraints = [
        cp.sum(cp.multiply(prices, h)) == 0,  # 总投资净值为0（长短仓组合）
        h >= -0.05 * (initial_capital / prices),  # 每只股票最小持仓
        h <= 0.05 * (initial_capital / prices)   # 每只股票最大持仓
    ]
    
    # 定义优化问题
    prob = cp.Problem(objective, constraints)
    
    # 求解优化问题
    # 使用适合MIQP的求解器，如 GUROBI 或 SCIP
    try:
        prob.solve(solver=cp.GUROBI, MIPGap=0.01)  # 设置 MIPGap 以控制求解精度
    except:
        # 如果GUROBI不可用，尝试使用CBC
        prob.solve(solver=cp.CBC, MIPGap=0.01)
    
    if h.value is None:
        raise ValueError("资产权重优化未能收敛，请检查数据和参数设置。")
    
    # 将持仓转化为整数股数
    holdings = pd.Series(np.round(h.value).astype(int), index=tickers)
    
    # 打印持仓
    print("\n具体股票的持仓股数:")
    for ticker, quantity in holdings.items():
        print(f"{ticker}: {quantity}")
    
    # 将持仓保存到 CSV
    holdings.to_csv('holdings.csv', header=['Holdings'])
    
    return holdings

def main():
    # 参数设定
    gamma = 5.0
    initial_capital = 1000000
    
    # 加载因子数据
    factors = load_factors()
    
    # 进行因子优化，获取因子权重
    factor_weights = mean_variance_optimization(factors, gamma=gamma)
    print("因子权重：")
    print(factor_weights)
    
    # 计算资产暴露
    asset_exposures = calculate_asset_exposures(factors)
    print("\n资产暴露：")
    print(asset_exposures.head())
    
    # 分配证券权重，获取持仓股数
    holdings = assign_weights_to_securities(factor_weights, asset_exposures, initial_capital=initial_capital, gamma=gamma)
    
    # 保存资产权重
    holdings.to_csv('asset_weights.csv', header=['Holdings'])
    print("\n资产权重已保存到 asset_weights.csv")
    print("持仓股数已保存到 holdings.csv")

if __name__ == "__main__":
    main()
