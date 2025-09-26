#!/usr/bin/env python3
"""
Standalone script to run DQN training and calculate comprehensive performance metrics
"""

import sys
import os
sys.path.append('..')
from DQN_Trading.main import *
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

def main():
    print("Starting DQN Trading Performance Analysis...")
    print("="*60)
    
    # Download data
    print("\n--- Downloading Training Data ---")
    train_data = download_data_with_cache(train_tickers, train_start_date, train_end_date)
    
    print("\n--- Downloading Testing Data ---")
    test_data = download_data_with_cache(test_tickers, test_start_date, test_end_date)
    
    print(f'Data Downloaded: {list(test_data.keys())}')
    
    # Initialize configuration
    cfg = Config()
    
    # Training
    print("\n--- Starting Training ---")
    env, agent = env_agent_config(train_data, cfg, 'train')
    rewards, ma_rewards, episode_rewards, train_sharpe_ratios, test_sharpe_ratios = train_with_testing(cfg, env, agent, test_data)
    
    # Create output directories
    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs(cfg.model_path, exist_ok=True)
    agent.save(path=cfg.model_path)
    
    # Testing with comprehensive metrics
    print("\n--- Starting Comprehensive Testing ---")
    all_data = {**train_data, **test_data}
    env, agent = env_agent_config(all_data, cfg, 'test')
    agent.load(path=cfg.model_path)
    
    stocks, rewards, individual_returns, cumulative_returns = test(cfg, env, agent)
    buy_and_hold_rewards = [sum(all_data[stock]) for stock in stocks]
    
    # Calculate comprehensive performance metrics
    print("\n" + "="*60)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Aggregate all DQN returns for overall metrics
    all_dqn_returns = []
    for stock_returns in individual_returns.values():
        all_dqn_returns.extend(stock_returns)
    
    # Calculate metrics
    dqn_test_sharpe = sharpe_ratio(rewards)
    buy_hold_sharpe = sharpe_ratio(buy_and_hold_rewards)
    dqn_train_sharpe = sharpe_ratio(episode_rewards)
    
    dqn_cumulative_return = calculate_cumulative_return(all_dqn_returns)
    dqn_max_drawdown = calculate_maximum_drawdown(all_dqn_returns)
    dqn_roi_percentage, dqn_roi_dollars = calculate_roi(all_dqn_returns)
    dqn_annualized_return = calculate_periodic_returns(all_dqn_returns)
    
    buy_hold_cumulative_return = calculate_cumulative_return(buy_and_hold_rewards)
    buy_hold_max_drawdown = calculate_maximum_drawdown(buy_and_hold_rewards)
    buy_hold_roi_percentage, buy_hold_roi_dollars = calculate_roi(buy_and_hold_rewards)
    buy_hold_annualized_return = calculate_periodic_returns(buy_and_hold_rewards)
    
    # Print results
    print(f"\nDQN MODEL PERFORMANCE:")
    print(f"  Sharpe Ratio: {dqn_test_sharpe:.3f}")
    print(f"  Cumulative Return: {dqn_cumulative_return:.2f}%")
    print(f"  Maximum Drawdown: {dqn_max_drawdown:.2f}%")
    print(f"  ROI: {dqn_roi_percentage:.2f}% (${dqn_roi_dollars:,.2f})")
    print(f"  Annualized Return: {dqn_annualized_return:.2f}%")
    
    print(f"\nBUY & HOLD PERFORMANCE:")
    print(f"  Sharpe Ratio: {buy_hold_sharpe:.3f}")
    print(f"  Cumulative Return: {buy_hold_cumulative_return:.2f}%")
    print(f"  Maximum Drawdown: {buy_hold_max_drawdown:.2f}%")
    print(f"  ROI: {buy_hold_roi_percentage:.2f}% (${buy_hold_roi_dollars:,.2f})")
    print(f"  Annualized Return: {buy_hold_annualized_return:.2f}%")
    
    print(f"\nPERFORMANCE COMPARISON:")
    print(f"  DQN vs Buy & Hold Sharpe: {dqn_test_sharpe:.3f} vs {buy_hold_sharpe:.3f}")
    print(f"  DQN vs Buy & Hold Return: {dqn_cumulative_return:.2f}% vs {buy_hold_cumulative_return:.2f}%")
    print(f"  DQN vs Buy & Hold Max DD: {dqn_max_drawdown:.2f}% vs {buy_hold_max_drawdown:.2f}%")
    print(f"  DQN vs Buy & Hold ROI: ${dqn_roi_dollars:,.2f} vs ${buy_hold_roi_dollars:,.2f}")
    
    # Individual stock performance
    print(f"\nINDIVIDUAL STOCK PERFORMANCE:")
    print(f"{'Stock':<8} {'Cum Return':<12} {'Max DD':<10} {'Sharpe':<8}")
    print("-" * 40)
    for stock in stocks:
        stock_returns = individual_returns[stock]
        stock_cum_return = calculate_cumulative_return(stock_returns)
        stock_max_dd = calculate_maximum_drawdown(stock_returns)
        stock_sharpe = sharpe_ratio(stock_returns)
        print(f"{stock:<8} {stock_cum_return:>10.2f}% {stock_max_dd:>8.2f}% {stock_sharpe:>6.3f}")
    
    # Save results to file
    results_summary = {
        'timestamp': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'train_episodes': cfg.train_eps,
            'state_dim': cfg.state_space_dim,
            'learning_rate': cfg.lr,
            'batch_size': cfg.batch_size,
            'memory_capacity': cfg.memory_capacity
        },
        'performance_metrics': {
            'dqn': {
                'sharpe_ratio': dqn_test_sharpe,
                'cumulative_return': dqn_cumulative_return,
                'max_drawdown': dqn_max_drawdown,
                'roi_percentage': dqn_roi_percentage,
                'roi_dollars': dqn_roi_dollars,
                'annualized_return': dqn_annualized_return
            },
            'buy_hold': {
                'sharpe_ratio': buy_hold_sharpe,
                'cumulative_return': buy_hold_cumulative_return,
                'max_drawdown': buy_hold_max_drawdown,
                'roi_percentage': buy_hold_roi_percentage,
                'roi_dollars': buy_hold_roi_dollars,
                'annualized_return': buy_hold_annualized_return
            }
        },
        'individual_stocks': {
            stock: {
                'cumulative_return': calculate_cumulative_return(individual_returns[stock]),
                'max_drawdown': calculate_maximum_drawdown(individual_returns[stock]),
                'sharpe_ratio': sharpe_ratio(individual_returns[stock])
            }
            for stock in stocks
        }
    }
    
    # Save to JSON file
    import json
    results_file = os.path.join(cfg.result_path, 'performance_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    print("\nPerformance analysis completed!")

if __name__ == "__main__":
    main() 