# portfolio_optimization_app/data_loader.py

import pandas as pd
from sqlalchemy import create_engine
import sys
import logging

def setup_logger():
    """设置日志记录"""
    logger = logging.getLogger('DataLoader')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

logger = setup_logger()

def load_factor_returns():
    path = './F-F_Research_Data_Factors.CSV'
    path_2 = './Developed_MOM_Factor.CSV'
    """
    加载并处理因子收益数据。
    """
    try:
        # 指定列名，包括 'Date'
        columns_1 = ['Date', 'Mkt_RF', 'SMB', 'HML', 'RF']
        columns_2 = ['Date', 'WML']

        # 读取第一个因子数据文件
        factors_1 = pd.read_csv(
            path,
            names=columns_1,
            header=0,  # 跳过第一行标题
            on_bad_lines='skip',  # 跳过无法解析的行
            skipinitialspace=True  # 跳过字段前的空格
        )

        # 读取第二个动量因子数据文件
        factors_2 = pd.read_csv(
            path_2,
            names=columns_2,
            header=0,  # 跳过第一行标题
            on_bad_lines='skip',  # 跳过无法解析的行
            skipinitialspace=True  # 跳过字段前的空格
        )

        factors_2['Date'] = factors_2['Date'].astype(str)
        factors_1['Date'] = factors_1['Date'].astype(str)
        # 合并两个因子数据
        factors = pd.merge(factors_1, factors_2, on='Date', how='inner')
        logger.info("因子收益数据加载成功。")
        
        factor_columns = ['Mkt_RF', 'SMB', 'HML', 'RF', 'WML']
        factors[factor_columns] = factors[factor_columns].apply(pd.to_numeric, errors='coerce')
        factors[factor_columns] = factors[factor_columns] / 100
        logger.info("因子列已成功除以 100。")

    except Exception as e:
        logger.error(f"加载因子收益数据时出错: {e}")
        sys.exit(1)

    # 将 'Date' 转换为 datetime 格式，使用 errors='coerce' 将无法解析的日期设置为 NaT
    try:
        initial_count = len(factors)
        factors['Date'] = pd.to_datetime(factors['Date'].astype(str), format='%Y%m', errors='coerce')
        factors = factors.dropna(subset=['Date'])  # 删除无法解析的日期行
        final_count = len(factors)
        logger.info(f"已删除 {initial_count - final_count} 行无法解析的日期。")
    except Exception as e:
        logger.error(f"解析 'Date' 列时出错: {e}")
        sys.exit(1)

    # 将 'Date' 列设置为索引
    factors.set_index('Date', inplace=True)

    return factors

def load_stock_data(db_name='russell3000.db'):
    """
    从SQLite数据库加载股票数据。如果缺少 'Return' 列，则计算它。
    """
    engine = create_engine(f'sqlite:///{db_name}')
    try:
        stock_df = pd.read_sql('SELECT * FROM stock_data', engine)
        logger.info("股票数据加载成功。")
    except Exception as e:
        logger.error(f"加载股票数据时出错: {e}")
        sys.exit(1)
    
    # 处理日期
    try:
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        logger.info("股票数据的日期列已转换为datetime格式。")
    except Exception as e:
        logger.error(f"解析 'Date' 列时出错: {e}")
        sys.exit(1)
    
    # 检查是否存在 'Return' 列
    if 'Return' not in stock_df.columns:
        logger.warning("'Return' 列不存在，正在计算 'Return'。")
        try:
            # 按照 'Ticker' 分组并按日期排序，然后计算月度收益率
            stock_df = stock_df.sort_values(['Ticker', 'Date'])
            stock_df['Return'] = stock_df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change())
            # 删除第一行（由于pct_change导致的NaN值）
            initial_rows = len(stock_df)
            stock_df = stock_df.dropna(subset=['Return'])
            final_rows = len(stock_df)
            logger.info(f"'Return' 列已成功计算，删除了 {initial_rows - final_rows} 个缺失的 'Return' 值。")
        except Exception as e:
            logger.error(f"计算 'Return' 列时出错: {e}")
            sys.exit(1)
    else:
        logger.info("'Return' 列已存在。")
    
    return stock_df
