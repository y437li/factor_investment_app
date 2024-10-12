# dashboard.py
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimization import mean_variance_optimization, assign_weights_to_assets, calculate_asset_exposures
from factor_construction import load_factors, load_stock_data
from backtest import calculate_portfolio_returns, save_logs, main as backtest_main
import cvxpy as cp
import numpy as np
import os

def plot_holdings_log(holdings_log_df):
    plt.figure(figsize=(12, 8))
    # 为了避免图表过于复杂，选择前10个股票展示持仓变化
    sample_tickers = list(holdings_log_df['Holdings'].iloc[0].keys())[:10]
    for ticker in sample_tickers:
        holdings = holdings_log_df['Holdings'].apply(lambda x: x.get(ticker, 0))
        plt.plot(holdings_log_df['Date'], holdings, label=ticker)
    plt.title('持仓股数变化（前10个股票）')
    plt.xlabel('日期')
    plt.ylabel('持仓股数')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('holdings_log.png')
    plt.close()
    return 'holdings_log.png'

# 其他绘图函数...

def adjust_exposure(start_date, end_date, factor1, factor2, factor3, factor4):
    try:
        # 重新运行因子构建
        from factor_construction import main as factor_main
        factor_main(start_date, end_date)

        # 重新运行投资组合优化
        from portfolio_optimization import main as portfolio_main
        portfolio_main()

        # 重新运行回测
        backtest_main(start_date, end_date)

        # 重新绘制图表
        if os.path.exists('asset_weights.csv'):
            latest_weights = pd.read_csv('asset_weights.csv', index_col=0).squeeze()
            asset_weights_plot = plot_asset_weights(latest_weights)
        else:
            asset_weights_plot = 'asset_weights.png'

        factors = load_factors()
        factor_weights = mean_variance_optimization(factors, lambda_param=1.0)
        factor_weights_plot = plot_beta_exposure(factor_weights)

        if os.path.exists('portfolio_returns.csv'):
            portfolio_returns = pd.read_csv('portfolio_returns.csv', index_col=0, parse_dates=True)
            backtest_plot = plot_backtest(portfolio_returns['Portfolio_Return'])
        else:
            backtest_plot = 'backtest.png'

        if os.path.exists('transaction_log.csv'):
            transaction_log_df = pd.read_csv('transaction_log.csv', parse_dates=['Date'])
            transaction_log_plot = plot_transaction_log(transaction_log_df)
        else:
            transaction_log_plot = 'transaction_log.png'

        if os.path.exists('position_log.csv'):
            position_log_df = pd.read_csv('position_log.csv', parse_dates=['Date'])
            position_log_plot = plot_position_log(position_log_df)
        else:
            position_log_plot = 'position_log.png'

        if os.path.exists('holdings_log.csv'):
            holdings_log_df = pd.read_csv('holdings_log.csv', parse_dates=['Date'])
            holdings_log_plot = plot_holdings_log(holdings_log_df)
        else:
            holdings_log_plot = 'holdings_log.png'

        return {
            "资产仓位": asset_weights_plot,
            "因子暴露": factor_weights_plot,
            "回测累计收益": backtest_plot,
            "交易信号": transaction_log_plot,
            "仓位变化": position_log_plot,
            "持仓股数变化": holdings_log_plot
        }
    except Exception as e:
        return f"调整过程中出现错误: {e}"

def create_dashboard():
    with gr.Blocks() as demo:
        gr.Markdown("# 因子投资仪表板")

        with gr.Tab("资产仓位"):
            if os.path.exists('asset_weights.csv'):
                latest_weights = pd.read_csv('asset_weights.csv', index_col=0).squeeze()
                img = gr.Image(plot_asset_weights(latest_weights))
            else:
                img = gr.Image("asset_weights.png")

        with gr.Tab("因子暴露"):
            if os.path.exists('factors.csv'):
                factors = load_factors()
                factor_weights = mean_variance_optimization(factors, lambda_param=1.0)
                img2 = gr.Image(plot_beta_exposure(factor_weights))
            else:
                img2 = gr.Image("beta_exposure.png")

        with gr.Tab("回测收益"):
            if os.path.exists('portfolio_returns.csv'):
                portfolio_returns = pd.read_csv('portfolio_returns.csv', index_col=0, parse_dates=True)
                img3 = gr.Image(plot_backtest(portfolio_returns['Portfolio_Return']))
            else:
                img3 = gr.Image("backtest.png")

        with gr.Tab("交易日志"):
            if os.path.exists('transaction_log.csv'):
                transaction_log_df = pd.read_csv('transaction_log.csv', parse_dates=['Date'])
                img4 = gr.Image(plot_transaction_log(transaction_log_df))
            else:
                img4 = gr.Image("transaction_log.png")

        with gr.Tab("仓位变化"):
            if os.path.exists('position_log.csv'):
                position_log_df = pd.read_csv('position_log.csv', parse_dates=['Date'])
                img5 = gr.Image(plot_position_log(position_log_df))
            else:
                img5 = gr.Image("position_log.png")

        with gr.Tab("持仓股数变化"):
            if os.path.exists('holdings_log.csv'):
                holdings_log_df = pd.read_csv('holdings_log.csv', parse_dates=['Date'])
                img6 = gr.Image(plot_holdings_log(holdings_log_df))
            else:
                img6 = gr.Image("holdings_log.png")

        with gr.Tab("调整因子和回测周期"):
            start_date = gr.DatePicker(label="回测起始日期", value="2010-01-01")
            end_date = gr.DatePicker(label="回测结束日期", value="2023-12-31")
            factor1 = gr.Slider(0, 1, value=0.25, label="SMB")
            factor2 = gr.Slider(0, 1, value=0.25, label="HML")
            factor3 = gr.Slider(0, 1, value=0.25, label="Momentum")
            factor4 = gr.Slider(0, 1, value=0.25, label="RM_RF")
            submit = gr.Button("提交调整")
            output = gr.Gallery(label="更新后的图表").style(grid=[2])

            submit.click(
                fn=adjust_exposure,
                inputs=[start_date, end_date, factor1, factor2, factor3, factor4],
                outputs=[output]
            )

    demo.launch()

if __name__ == "__main__":
    create_dashboard()
