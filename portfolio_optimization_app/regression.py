import yfinance as yf
import pandas as pd
import sqlite3
import logging
import statsmodels.api as sm
from tqdm.rich import tqdm

def create_risk_free_rate_table_if_not_exists(conn):
    """
    如果数据库中没有 risk_free_rate 表，则创建该表。
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS risk_free_rate (
        Date TEXT PRIMARY KEY,
        RF REAL
    );
    """
    conn.execute(create_table_query)
    conn.commit()

def get_risk_free_rate_from_db(conn, start_date, end_date):
    """
    从 SQLite 数据库中获取 10 年期美债的无风险利率数据。
    如果数据库中没有数据，则从 Yahoo Finance 下载并存储到数据库中。
    """
    # 首先确保 risk_free_rate 表已存在
    create_risk_free_rate_table_if_not_exists(conn)
    
    query = f"""
    SELECT Date, RF FROM risk_free_rate
    WHERE Date BETWEEN '{start_date}' AND '{end_date}';
    """
    
    # 尝试从数据库中读取数据
    rf_data = pd.read_sql_query(query, conn, parse_dates=['Date'])
    
    if rf_data.empty:
        # 如果数据库中没有数据，则从 Yahoo Finance 下载
        logger.info("数据库中无10年期美债数据，正在从Yahoo Finance下载...")
        tnx_data = yf.download('^TNX', start=start_date, end=end_date)
        tnx_data['RF'] = tnx_data['Adj Close'] / 100
        tnx_data = tnx_data[['RF']]
        tnx_data.index.name = 'Date'
        
        # 将数据写入 SQLite 数据库
        tnx_data.to_sql('risk_free_rate', conn, if_exists='append', index=True)
        logger.info("已将10年期美债无风险利率数据存储到数据库。")
        
        return tnx_data
    else:
        logger.info("已从数据库中加载10年期美债无风险利率数据。")
        return rf_data.set_index('Date')


def perform_regression(stock_df, factors):
    """
    对每只股票进行回归分析，估算 β 敞口，并使用从 factors 中的 RF 作为无风险利率。
    在此过程中，计算每只股票的月度收益率。
    """
    tickers = stock_df['Ticker'].unique()
    betas = {}
    #print(factors)
    # 确保所有数据的日期列是 datetime 格式
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    factors.reset_index(inplace=True)
    factors['Date'] = pd.to_datetime(factors['Date'])

    # 计算每只股票的月度收益率
    stock_df['Month'] = stock_df['Date'].dt.to_period('M')  # 提取月份

    for ticker in tqdm(tickers):
        ticker_data = stock_df[stock_df['Ticker'] == ticker].copy()

        # 获取每个月的第一天和最后一天
        first_day = ticker_data.groupby('Month').first()
        last_day = ticker_data.groupby('Month').last()

        # 计算月度收益率
        monthly_return = (last_day['Close'] - first_day['Close']) / first_day['Close']
        monthly_return = monthly_return.reset_index().rename(columns={0: 'Return'})
        monthly_return['Date'] = first_day.reset_index()['Date']
        monthly_return.rename(columns={'Close': 'Return'}, inplace=True)
        monthly_return = monthly_return[['Date', 'Return']]

        # 使用 first_day 来提供月度的日期信息
        monthly_return['Date'] = first_day.reset_index()['Date']

        # 合并月度收益率与因子数据
        monthly_data = pd.merge(monthly_return, factors, on='Date', how='inner')

        #x w wprint(monthly_data)
        # 确保 RF 列存在并没有缺失值
        if 'RF' not in monthly_data.columns or monthly_data['RF'].isnull().any():
            logger.warning(f"股票 {ticker} 数据中缺少无风险利率信息，跳过回归。")
            continue

        # 计算超额收益
        monthly_data['Excess_Return'] = monthly_data['Return'] - monthly_data['RF']

        # 准备回归变量
        X = monthly_data[['Mkt_RF', 'SMB', 'HML', 'WML']]
        X = sm.add_constant(X)  # 添加截距
        y = monthly_data['Excess_Return']

        # 检查是否有足够的数据点
        if len(monthly_data) < 6:  # 如果数据不足6个月，则跳过
            logger.warning(f"股票 {ticker} 的数据点不足，跳过回归。")
            continue

        # 执行 OLS 回归
        try:
            model = sm.OLS(y, X).fit()
            betas[ticker] = {
                'params': model.params[1:],  # 因子系数
                'rsquared': model.rsquared,  # R-squared 值
                'pvalues': model.pvalues[1:]  # 各系数的 p-value，排除截距
            }
            logger.info(f"股票 {ticker} 的回归分析完成。")
        except Exception as e:
            logger.error(f"对股票 {ticker} 进行回归时出错: {e}")
            continue

    # 将 betas 转化为 DataFrame，返回更详细的信息
    betas_df = pd.DataFrame.from_dict(betas, orient='index')
    return betas_df


def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 设置日志级别
        logger.setLevel(logging.INFO)
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # 创建格式化器并添加到处理器
        formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        # 将处理器添加到记录器
        logger.addHandler(ch)
    return logger

# 初始化日志记录
logger = setup_logger('Regression')
