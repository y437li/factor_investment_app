import pandas as pd
import logging

# factor_builder.py

def construct_momentum(df, window=6):
    # 确保按照日期排序
    df = df.sort_values(by=['Ticker', 'Date'])
    # 计算动量分数，使用 transform 确保结果与原 DataFrame 对齐
    df['Momentum'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.pct_change(periods=window))
    # 删除缺失值
    df.dropna(subset=['Momentum'], inplace=True)
    return df

def calculate_momentum_returns(df, long_percent=0.1, short_percent=0.1):
    logger = setup_logger('MomentumReturns')

    # 确保日期为 datetime 类型，并设置为索引
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 只保留需要的列
    df = df[['Ticker', 'Return', 'Momentum']]

    # 按月进行分组
    monthly_returns = []
    months = df.index.to_period('M').unique()

    for month in months:
        month_data = df[df.index.to_period('M') == month]

        # 获取当月的股票列表及其动量分数
        momentum_scores = month_data.groupby('Ticker')['Momentum'].last()

        # 排序并选择高动量和低动量股票
        sorted_scores = momentum_scores.sort_values(ascending=False)
        n_long = max(int(len(sorted_scores) * long_percent), 1)
        n_short = max(int(len(sorted_scores) * short_percent), 1)

        long_tickers = sorted_scores.head(n_long).index
        short_tickers = sorted_scores.tail(n_short).index

        # 计算当月高动量组和低动量组的平均收益
        next_month = (month + 1).to_timestamp()
        next_month_data = df[df.index.to_period('M') == next_month.to_period('M')]

        if next_month_data.empty:
            continue

        # 获取多头和空头股票的下个月收益
        long_returns = next_month_data[next_month_data['Ticker'].isin(long_tickers)]['Return']
        short_returns = next_month_data[next_month_data['Ticker'].isin(short_tickers)]['Return']

        # 计算平均收益
        long_return = long_returns.mean()
        short_return = short_returns.mean()

        # 动量策略收益
        momentum_return = long_return - short_return

        monthly_returns.append({
            'Date': month.to_timestamp(),
            'LongReturn': long_return,
            'ShortReturn': short_return,
            'Momentum': momentum_return})

    # 将结果转换为 DataFrame
    momentum_returns_df = pd.DataFrame(monthly_returns)
    return momentum_returns_df

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
