import numpy as np
import pandas as pd
import logging
import sys
from scipy.optimize import minimize

def setup_logger():
    """设置日志记录"""
    logger = logging.getLogger('Optimization')
    logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[Optimization] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    # 避免重复添加处理器
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

logger = setup_logger()

def mean_variance_optimization(
    betas=None,
    expected_returns=None,
    desired_beta=None,
    latest_prices=None,
    cov_matrix=None,
    initial_capital=1000000,
    max_holding_percent=0.04,
    max_holdings=None,  # 由于无法直接实现，该参数暂时忽略
    gamma=5.0,
    l1_lambda=0.01  # L1正则化系数，用于鼓励稀疏性
):
    """
    使用 scipy.optimize 进行均值-方差优化，加入新的敞口约束和4%的个股持仓限制。
    如果没有提供目标β敞口，使用优化后的组合权重估算组合β敞口。
    """
    tickers = expected_returns.index.tolist()
    n = len(tickers)
    
    # 获取预期收益
    E_r = expected_returns.values  # (n,)
    
    if betas is not None and desired_beta is not None:
        k = betas.shape[1]
        if len(desired_beta) != k:
            logger.error(f"期望的β长度 {len(desired_beta)} 与因子数量 {k} 不匹配。")
            sys.exit(1)
        desired_beta = np.array(desired_beta)
        B = betas.loc[tickers].values  # (n x k)
    else:
        B = None
    
    if cov_matrix is None:
        logger.error("协方差矩阵不能为空。")
        sys.exit(1)
    Sigma_stock = cov_matrix  # 使用提供的协方差矩阵
    logger.info("使用提供的协方差矩阵进行均值-方差优化。")
    
    # 强制协方差矩阵对称
    Sigma_stock = (Sigma_stock + Sigma_stock.T) / 2
    
    # 检查协方差矩阵的对称性和缺失值
    logger.info(f"协方差矩阵的形状: {Sigma_stock.shape}")
    logger.info(f"协方差矩阵是否对称: {np.allclose(Sigma_stock, Sigma_stock.T)}")
    logger.info(f"协方差矩阵中是否存在NaN值: {np.isnan(Sigma_stock).any()}")
    
    if not np.allclose(Sigma_stock, Sigma_stock.T):
        logger.error("协方差矩阵不对称，优化过程无法继续。")
        sys.exit(1)
    
    if np.isnan(Sigma_stock).any():
        logger.error("协方差矩阵中存在NaN值，优化过程无法继续。")
        sys.exit(1)
    
    # 定义目标函数，包含 L1 正则化项
    def objective(h):
        portfolio_return = E_r @ h
        portfolio_risk = h.T @ Sigma_stock @ h
        #l1_penalty = l1_lambda * np.sum(np.abs(h))
        #utility = portfolio_return - (gamma / 2) * portfolio_risk - l1_penalty
        utility = portfolio_return - (gamma / 2) * portfolio_risk
        return -utility  # scipy.optimize.minimize 进行最小化
    
    # 定义约束条件
    constraints = []
    
    # 净敞口约束：净投资等于初始资金
    def net_exposure_constraint(h):
        return np.dot(latest_prices, h) - initial_capital
    constraints.append({'type': 'eq', 'fun': net_exposure_constraint})
    logger.info("已添加净敞口约束：净投资等于初始资金。")
    
    # 总敞口约束：总投资等于2倍的初始资金
    def gross_exposure_constraint(h):
        return np.dot(latest_prices, np.abs(h)) - 2 * initial_capital
    constraints.append({'type': 'eq', 'fun': gross_exposure_constraint})
    logger.info("已添加总敞口约束：总投资等于2倍的初始资金。")
    
    # 如果提供了目标β敞口，添加β敞口约束
    if B is not None and desired_beta is not None:
        def beta_constraint(h):
            portfolio_beta = np.dot(B.T, h) / initial_capital  # (k,)
            return portfolio_beta - desired_beta
        for i in range(len(desired_beta)):
            constraints.append({'type': 'eq', 'fun': lambda h, i=i: beta_constraint(h)[i]})
        logger.info("已添加投资组合β敞口约束。")
    
    # 设置变量的上下界（4%的个股持仓限制）
    max_holding_shares = max_holding_percent * initial_capital / latest_prices  # (n,)
        # 将持仓上下限作为约束条件
    for i in range(n):
        # 添加不等式约束： h[i] >= -max_holding_shares[i]
        constraints.append({
            'type': 'ineq',
            'fun': lambda h, i=i: h[i] + max_holding_shares[i]
        })
        # 添加不等式约束： h[i] <= max_holding_shares[i]
        constraints.append({
            'type': 'ineq',
            'fun': lambda h, i=i: max_holding_shares[i] - h[i]
        })
    logger.info("已设置每只股票的持仓上下界,确保不超过4%的持仓限制。")
    
    # 初始猜测值
    x0 = np.zeros(n)
    
    # 定义回调函数
    iteration = [0]
    def callback(h):
        iteration[0] += 1
        current_value = objective(h)
        logger.debug(f"迭代 {iteration[0]}：当前目标函数值 = {current_value:.6f}")
    
    # 执行优化
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=None,
        constraints=constraints,
        callback=callback,
        options={'maxiter': 10000, 'ftol': 1e-2, 'disp': True}
    )
    
    if not result.success:
        logger.error(f"优化未收敛：{result.message}")
        sys.exit(1)
    
    # 获取优化结果
    h_optimal = result.x
    
    # 如果没有提供目标β，计算投资组合β敞口
    if B is not None and desired_beta is None:
        portfolio_beta = np.dot(B.T, h_optimal / initial_capital)
        logger.info(f"通过优化计算得到的投资组合 β 敞口为: {portfolio_beta}")
    
    # 将持仓数量取整
    holdings = np.round(h_optimal).astype(int)
    holdings_series = pd.Series(holdings, index=tickers)
    
    # 验证持仓比例
    holding_values = holdings_series.values * latest_prices
    holding_percentages = np.abs(holding_values) / initial_capital  # 取绝对值
    exceeded = holding_percentages > max_holding_percent + 1e-6
    if np.any(exceeded):
        tickers_exceeded = holdings_series.index[exceeded]
        logger.warning(f"以下股票的持仓超过了{max_holding_percent*100}%的限制: {tickers_exceeded.tolist()}")
    else:
        logger.info("所有股票的持仓均在4%的限制以内。")
    
    logger.info("均值-方差优化完成。")
    return holdings_series
