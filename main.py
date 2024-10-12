import argparse
import sys
import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
from portfolio_optimization_app import (
    load_factor_returns,
    perform_regression,
    mean_variance_optimization,
    setup_logger,
    construct_momentum,
    calculate_momentum_returns,
    get_risk_free_rate_from_db
)
from portfolio_optimization_app.regression import perform_regression  # 导入回归函数


def main():
    logger = setup_logger('Main')

    parser = argparse.ArgumentParser(description='Factor-Based Portfolio Optimization App')
    parser.add_argument('--factor_file', type=str, default='F-F_Research_Data_Factors.CSV', help='Path to factor returns CSV file')
    parser.add_argument('--db_name', type=str, default='russell3000.db', help='SQLite database name')
    parser.add_argument('--desired_beta', type=str, default=None, help='Desired portfolio beta as comma-separated values (e.g., "1.0,0.2,-0.15,0.3")')
    parser.add_argument('--start_date', type=str, required=True, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--industries', type=str, default=None, help='Comma-separated list of industries to include')
    parser.add_argument('--market_cap_min', type=float, default=None, help='Minimum market cap to include')
    parser.add_argument('--avg_volume_min', type=float, default=None, help='Minimum average volume to include')
    parser.add_argument('--top_n', type=int, default=500, help='Number of top stocks to select based on expected returns')
    parser.add_argument('--initial_capital', type=float, default=1000000, help='Initial capital for backtest')

    args = parser.parse_args()

    # 解析 desired_beta
    if args.desired_beta:
        try:
            desired_beta = [float(x) for x in args.desired_beta.split(',')]
            logger.info(f"目标β敞口: {desired_beta}")
        except:
            logger.error("解析 desired_beta 时出错。请提供以逗号分隔的数值，例如 '1.0,0.2,-0.15,0.3'。")
            sys.exit(1)
    else:
        desired_beta = None
        logger.info("未提供目标β敞口，将执行标准均值-方差优化并计算β。")

    # 加载因子收益数据
    factors = load_factor_returns()

    # 连接到 SQLite 数据库
    try:
        conn = sqlite3.connect(args.db_name)
        logger.info(f"已连接到数据库 {args.db_name}。")
    except Exception as e:
        logger.error(f"无法连接到数据库 {args.db_name}:{e}")
        sys.exit(1)

    # 加载股票数据
    stock_df = pd.read_sql_query("SELECT * FROM stock_data", conn)
    logger.info("股票数据加载成功。")

    # 确保日期列为 datetime 类型
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    logger.info("股票数据的日期列已转换为 datetime 格式。")

    # 获取10年期美债无风险利率数据
    #risk_free_rate = get_risk_free_rate_from_db(conn, args.start_date, args.end_date)


    # 检查是否存在 'Return' 列，如果不存在则计算
    if 'Return' not in stock_df.columns:
        if 'Adj Close' in stock_df.columns:
            stock_df.sort_values(by=['Ticker', 'Date'], inplace=True)
            stock_df['Return'] = stock_df.groupby('Ticker')['Adj Close'].pct_change()
            stock_df.dropna(subset=['Return'], inplace=True)
            logger.info("'Return' 列已成功计算。")
        else:
            logger.error("股票数据中缺少 'Adj Close' 列，无法计算 'Return'。")
            sys.exit(1)

    # 获取股票代码列表
    tickers = stock_df['Ticker'].unique().tolist()
    logger.info(f"共有 {len(tickers)} 只股票需要检查行业信息。")

    # 检查数据库中是否已有股票信息表 'stock_info'
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_info';")
    table_exists = cursor.fetchone()

    if table_exists:
        # 从数据库中加载股票信息
        stock_info_df = pd.read_sql_query("SELECT * FROM stock_info", conn)
        logger.info("已从数据库中加载股票信息。")
    else:
        # 如果没有 'stock_info' 表，则创建一个新的表
        logger.info("数据库中未找到股票信息表 'stock_info'，将从 Yahoo Finance 获取信息。")
        stock_info_df = pd.DataFrame(columns=['Ticker', 'Sector', 'Industry', 'MarketCap', 'AvgVolume'])

    # 找出缺少信息的股票
    existing_tickers = stock_info_df['Ticker'].tolist()
    missing_tickers = list(set(tickers) - set(existing_tickers))
    logger.info(f"共有 {len(missing_tickers)} 只股票需要从 Yahoo Finance 获取信息。")

    # 从 Yahoo Finance 获取缺失的股票信息
    if missing_tickers:
        new_stock_info_list = []
        for ticker in missing_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                stock_info = {
                    'Ticker': ticker,
                    'Sector': info.get('sector', 'N/A'),
                    'Industry': info.get('industry', 'N/A'),
                    'MarketCap': info.get('marketCap', np.nan),
                    'AvgVolume': info.get('averageVolume', np.nan)
                }
                new_stock_info_list.append(stock_info)
            except Exception as e:
                logger.warning(f"无法获取 {ticker} 的信息：{e}")
                continue

        # 将新获取的信息添加到 DataFrame
        new_stock_info_df = pd.DataFrame(new_stock_info_list)
        stock_info_df = pd.concat([stock_info_df, new_stock_info_df], ignore_index=True)

        # 将股票信息保存到数据库
        stock_info_df.to_sql('stock_info', conn, if_exists='replace', index=False)
        logger.info("已将股票信息保存到数据库表 'stock_info'。")

    # 合并股票数据和股票信息
    stock_df = stock_df.merge(stock_info_df, on='Ticker', how='left')
    logger.info("股票数据和行业信息已合并。")

    # 检查是否有股票缺少行业信息
    missing_industry = stock_df['Industry'].isnull().sum()
    if missing_industry > 0:
        logger.warning(f"有 {missing_industry} 条股票数据缺少行业信息，将被删除。")
        stock_df = stock_df.dropna(subset=['Industry'])

    # 提取所有独特的行业并保存到文本文件
    all_industries = stock_df['Industry'].dropna().unique()
    all_industries.sort()
    with open('industries.txt', 'w') as f:
        for industry in all_industries:
            f.write(f"{industry}\n")
    logger.info("所有可选行业已保存到 industries.txt。")

    # 解析行业列表
    if args.industries:
        selected_industries = [x.strip() for x in args.industries.split(',')]
        logger.info(f"将限制选股范围到以下行业：{selected_industries}")
    else:
        # 未提供 --industries 参数，默认使用所有行业
        selected_industries = all_industries.tolist()
        logger.info("未指定行业，将使用所有可用行业的股票。")

    # 检查用户输入的行业是否在可用行业列表中
    valid_industries = set(all_industries)
    invalid_industries = [ind for ind in selected_industries if ind not in valid_industries]
    if invalid_industries:
        logger.error(f"以下行业无效或不存在于数据中：{invalid_industries}")
        sys.exit(1)
    # 筛选出属于指定行业的股票
    stock_df = stock_df[stock_df['Industry'].isin(selected_industries)]
    logger.info(f"已筛选出指定行业的股票，行业列表：{selected_industries}")

    # 根据市值和流动性筛选股票
    if args.market_cap_min is not None:
        stock_df = stock_df[stock_df['MarketCap'] >= args.market_cap_min]
        logger.info(f"已筛选出市值大于 {args.market_cap_min} 的股票。")

    if args.avg_volume_min is not None:
        stock_df = stock_df[stock_df['AvgVolume'] >= args.avg_volume_min]
        logger.info(f"已筛选出平均交易量大于 {args.avg_volume_min} 的股票。")

    # 解析开始和结束日期
    try:
        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
        if start_date >= end_date:
            logger.error("开始日期必须早于结束日期。")
            sys.exit(1)
    except:
        logger.error("解析开始日期或结束日期时出错。请提供有效的日期格式，例如 '2023-01-01'。")
        sys.exit(1)
    
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    logger.info("股票数据的日期列已转换为 datetime 格式。")
    
    # 过滤回测日期范围内的数据
    stock_df_filtered = stock_df[(stock_df['Date'] >= start_date) & (stock_df['Date'] <= end_date)].copy()
    logger.info(f"已过滤出从 {start_date.date()} 到 {end_date.date()} 的股票数据。")

    # 移除重复的 (Date, Ticker) 组合
    #if 'Date' not in stock_df_filtered.columns or 'Ticker' not in stock_df_filtered.columns:
        #logger.error("缺少 'Date' 或 'Ticker' 列，无法检查重复组合。")
        #sys.exit(1)
    stock_df_filtered.reset_index(drop=False, inplace=True)
    #print(stock_df_filtered)
    #print(stock_df_filtered.columns)
    stock_df_filtered['Date'] = pd.to_datetime(stock_df_filtered['Date'], errors='coerce')
    stock_df_filtered.dropna(subset=['Date', 'Ticker'], inplace=True)

    duplicates = stock_df_filtered.duplicated(subset=['Date', 'Ticker'])
    if duplicates.any():
        logger.warning(f"存在 {duplicates.sum()} 个重复的 Date-Ticker 组合，正在删除。")
        stock_df_filtered = stock_df_filtered[~duplicates]
        logger.info("重复的 Date-Ticker 组合已删除。")

    # 计算预期收益
    expected_returns = stock_df_filtered.groupby('Ticker')['Return'].mean()
    logger.info("已计算出预期收益。")

    # 选择预期收益最高的前 N 只股票
    top_n = args.top_n
    expected_returns = expected_returns.sort_values(ascending=False).head(top_n)
    selected_tickers = expected_returns.index.tolist()
    logger.info(f"已选择预期收益最高的前 {top_n} 只股票。")

    # 更新 stock_df_filtered，只保留选定的股票
    stock_df_filtered = stock_df_filtered[stock_df_filtered['Ticker'].isin(selected_tickers)]

    # 更新 latest_prices，以匹配筛选后的股票
    latest_prices = stock_df_filtered.groupby('Ticker')['Adj Close'].last()
    latest_prices = latest_prices.loc[selected_tickers]

    # 计算协方差矩阵
    try:
        returns_pivot = stock_df_filtered.pivot(index='Date', columns='Ticker', values='Return')

        # 移除缺失数据过多的股票（至少70%的数据非缺失）
        required_non_nan = 0.7 * returns_pivot.shape[0]
        returns_pivot = returns_pivot.dropna(axis=1, thresh=required_non_nan)
        logger.info(f"协方差矩阵前，剩余股票数量: {returns_pivot.shape[1]}")

        if returns_pivot.shape[1] == 0:
            logger.error("所有股票在协方差矩阵计算中被删除，可能是缺失值过多。请检查数据或降低缺失值阈值。")
            sys.exit(1)

        # 更新 expected_returns 和 latest_prices 以匹配筛选后的股票
        expected_returns = expected_returns.loc[returns_pivot.columns]
        latest_prices = latest_prices.loc[returns_pivot.columns]

        cov_matrix = returns_pivot.cov().values
        logger.info("协方差矩阵已成功计算。")
    except Exception as e:
        logger.error(f"计算协方差矩阵时出错: {e}")
        sys.exit(1)

    # 执行回归分析以获取每只股票的β敞口
    logger.info("开始对每只股票进行回归分析以估算β敞口。")
    betas_df = perform_regression(stock_df_filtered, factors)
    logger.info("回归分析已完成。")

    # 确保 betas_df 只包含在协方差矩阵中的股票
    betas_df = betas_df.loc[returns_pivot.columns]

    # 执行均值-方差优化
    holdings = mean_variance_optimization(
        betas=betas_df,  # 传递回归得到的β敞口
        expected_returns=expected_returns,
        desired_beta=desired_beta,
        latest_prices=latest_prices.values,
        cov_matrix=cov_matrix,
        initial_capital=args.initial_capital,
        max_holding_percent=0.04,
        max_holdings=args.top_n,
        gamma=5.0
    )

    # 如果没有目标 β，则通过投资组合权重估算β
    if not desired_beta:
        portfolio_beta = np.dot(betas_df.T, holdings.values / args.initial_capital)
        logger.info(f"通过投资组合权重估算的投资组合 β 敞口: {portfolio_beta}")

    # 保存持仓结果到 CSV
    holdings.to_csv('holdings.csv', header=['Holdings'])
    logger.info("持仓结果已保存到 holdings.csv。")

    # 计算投资组合预期收益和风险
    portfolio_return = expected_returns.loc[holdings.index].dot(holdings.values)
    portfolio_risk = np.sqrt(holdings.values.T @ cov_matrix @ holdings.values)
    with open('portfolio_summary.txt', 'w') as f:
        f.write(f"投资组合预期收益: {portfolio_return:.4f}\n")
        f.write(f"投资组合风险 (标准差): {portfolio_risk:.4f}\n")
        if not desired_beta:
            f.write(f"投资组合 β 敞口: {portfolio_beta}\n")
    logger.info("投资组合摘要已保存到 portfolio_summary.txt。")

    # 关闭数据库连接
    conn.close()
    logger.info("数据库连接已关闭。")


if __name__ == "__main__":
    main()
