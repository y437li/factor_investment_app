# data_collection.py
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import datetime
import time
from tqdm.rich import tqdm
from rich import print

def get_russell3000_tickers():
    # Russell 3000 成分股列表需要从可靠来源获取
    # 这里假设您有一个 CSV 文件包含所有成分股的符号
    tickers = pd.read_excel('grid1_tgshydpp.xlsx')['Ticker'].tolist()
    return tickers

def download_stock_data(tickers, start_date='1960-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    data = []
    for ticker in tqdm(tickers):
        try:
            stock = yf.Ticker(ticker)
            hist = yf.download(ticker,start=start_date, end=end_date)
            print(hist)
            if hist.empty:
                continue
            hist.reset_index(inplace=True)
            hist['Ticker'] = ticker
            # 获取财务数据
            info = stock.info
            hist['Market_Cap'] = info.get('marketCap', None)
            hist['Book_Value'] = info.get('bookValue', None)
            data.append(hist)
            print(f"下载成功: {ticker}")
            time.sleep(0.1)  # 避免请求过快
        except Exception as e:
            print(f"下载 {ticker} 时出错: {e}")
    combined_data = pd.concat(data, ignore_index=True)
    return combined_data

def save_to_sqlite(df, db_name='russell3000.db'):
    engine = create_engine(f'sqlite:///{db_name}')
    df.to_sql('stock_data', engine, if_exists='replace', index=False)

def main():
    tickers = get_russell3000_tickers()
    stock_data = download_stock_data(tickers)
    save_to_sqlite(stock_data)

if __name__ == "__main__":
    main()
