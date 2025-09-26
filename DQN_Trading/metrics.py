import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from typing import List, Dict, Any
import os


def sharpe_ratio(returns, risk_free_rate=0.0):
    returns = np.array(returns)
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free_rate) / returns.std()

def calculate_cumulative_return(returns):
    """
    Calculate cumulative return from a list of returns.
    
    Args:
        returns: List of returns (e.g., [0.01, -0.02, 0.03, ...])
        
    Returns:
        Cumulative return as a percentage
    """
    returns = np.array(returns)
    cumulative_return = np.prod(1 + returns) - 1
    return cumulative_return * 100  # Convert to percentage

def calculate_maximum_drawdown(returns):
    """
    Calculate maximum drawdown from a list of returns.
    
    Args:
        returns: List of returns
        
    Returns:
        Maximum drawdown as a percentage
    """
    returns = np.array(returns)
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    return max_drawdown * 100  # Convert to percentage

def calculate_roi(returns, initial_investment=10000):
    """
    Calculate Return on Investment (ROI).
    
    Args:
        returns: List of returns
        initial_investment: Initial investment amount (default: $10,000)
        
    Returns:
        ROI as a percentage and absolute dollar amount
    """
    cumulative_return = calculate_cumulative_return(returns) / 100  # Convert back to decimal
    roi_percentage = cumulative_return * 100
    roi_dollars = initial_investment * cumulative_return
    return roi_percentage, roi_dollars

def calculate_periodic_returns(returns, period_length=252):
    """
    Calculate periodic returns (e.g., annualized) from daily returns.
    
    Args:
        returns: List of daily returns
        period_length: Number of periods in a year (252 for daily returns)
        
    Returns:
        Annualized return as a percentage
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    # Calculate total return
    total_return = calculate_cumulative_return(returns) / 100
    
    # Annualize the return
    years = len(returns) / period_length
    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return * 100
    else:
        return 0.0

def create_performance_visualizations(cfg, dqn_returns, buy_hold_returns, individual_returns, stocks, final_results):
    """Create comprehensive performance visualizations"""
    
    # 1. Cumulative Returns Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # DQN cumulative returns
    dqn_cumulative = np.cumprod(1 + np.array(dqn_returns)) - 1
    buy_hold_cumulative = np.cumprod(1 + np.array(buy_hold_returns)) - 1
    
    ax1.plot(dqn_cumulative * 100, label='DQN Strategy', linewidth=2, color='blue')
    ax1.plot(buy_hold_cumulative * 100, label='Buy & Hold', linewidth=2, color='red')
    ax1.set_title('Cumulative Returns Comparison')
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown Analysis
    def calculate_drawdown_series(returns):
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown * 100
    
    dqn_drawdown = calculate_drawdown_series(dqn_returns)
    buy_hold_drawdown = calculate_drawdown_series(buy_hold_returns)
    
    ax2.plot(dqn_drawdown, label='DQN Strategy', linewidth=2, color='blue')
    ax2.plot(buy_hold_drawdown, label='Buy & Hold', linewidth=2, color='red')
    ax2.set_title('Drawdown Analysis')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(cfg.result_path+'/cumulative_returns_and_drawdown.jpg')
    plt.close()
    
    # 3. Individual Stock Performance
    fig, ax = plt.subplots(figsize=(14, 8))
    
    stock_cum_returns = []
    stock_names = []
    for stock in stocks:
        stock_returns = individual_returns[stock]
        cum_return = calculate_cumulative_return(stock_returns)
        stock_cum_returns.append(cum_return)
        stock_names.append(stock)
    
    bars = ax.bar(stock_names, stock_cum_returns, color='skyblue', alpha=0.7, edgecolor='black')  # type: ignore
    
    # Add value labels on bars
    for bar, value in zip(bars, stock_cum_returns):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1.5),
                f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')  # type: ignore
    
    ax.set_title('Individual Stock Cumulative Returns (DQN Strategy)')  # type: ignore
    ax.set_ylabel('Cumulative Return (%)')  # type: ignore
    ax.set_xlabel('Stocks')  # type: ignore
    ax.grid(axis='y', alpha=0.3)  # type: ignore
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(cfg.result_path+'/individual_stock_performance.jpg')
    plt.close()
    
    # 4. Performance Metrics Summary
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = ['Sharpe Ratio', 'Cumulative Return (%)', 'Max Drawdown (%)', 'Annualized Return (%)']
    dqn_values = [
        final_results['dqn_test_sharpe'],
        final_results['performance_metrics']['dqn']['cumulative_return'],
        final_results['performance_metrics']['dqn']['max_drawdown'],
        final_results['performance_metrics']['dqn']['annualized_return']
    ]
    buy_hold_values = [
        final_results['buy_hold_sharpe'],
        final_results['performance_metrics']['buy_hold']['cumulative_return'],
        final_results['performance_metrics']['buy_hold']['max_drawdown'],
        final_results['performance_metrics']['buy_hold']['annualized_return']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dqn_values, width, label='DQN Strategy', color='lightblue', alpha=0.8)  # type: ignore
    bars2 = ax.bar(x + width/2, buy_hold_values, width, label='Buy & Hold', color='lightcoral', alpha=0.8)  # type: ignore
    
    ax.set_xlabel('Performance Metrics')  # type: ignore
    ax.set_ylabel('Value')  # type: ignore
    ax.set_title('Performance Metrics Comparison')  # type: ignore
    ax.set_xticks(x)  # type: ignore
    ax.set_xticklabels(metrics, rotation=45)  # type: ignore
    ax.legend()  # type: ignore
    ax.grid(axis='y', alpha=0.3)  # type: ignore
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                    f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)  # type: ignore
    
    plt.tight_layout()
    plt.savefig(cfg.result_path+'/performance_metrics_comparison.jpg')
    plt.close()
    
    print(f"\nPerformance visualizations saved to: {cfg.result_path}")
    print("Files created:")
    print("  - cumulative_returns_and_drawdown.jpg")
    print("  - individual_stock_performance.jpg")
    print("  - performance_metrics_comparison.jpg")

