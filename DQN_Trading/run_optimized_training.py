import sys
import os
import json
import datetime as dt
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

sys.path.append('..')
from DQN_Trading.finrl_trading_env import FinRLTradingEnv_v1
from data_manager import DataManager
from dqn import DQN, univariate_trimmed_mean, compute_G_t
from metrics import *
from trading_env import TradingSystem_v0
from main import train_with_testing, test, env_agent_config, download_data_with_cache, save_training_results

curr_path = os.path.dirname(__file__)
curr_time = dt.datetime.now().strftime("%Y%m%d-%H%M")

# Configuration
train_tickers = ['ZM', 'X', 'META', 'MTCH', 'GOOG', 'PINS', 'SNAP', 'ETSY']
test_tickers = train_tickers

train_start_date = dt.date(2015, 1, 1)
train_end_date = dt.date(2022, 12, 31)
test_start_date = dt.date(2023, 1, 1)
test_end_date = dt.date(2024, 12, 31)

def set_seed(seed):
    """
    Set seeds for all random number generators to ensure reproducibility.
    
    Args:
        seed (int): Seed value for all random number generators
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Seeds set to {seed} for reproducibility")

class OptimizedConfig:
    def __init__(self, best_params=None):
        # Environment parameters
        self.algo_name = 'DQN'
        self.env_name = 'TradingSystem_v0'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42  # Changed to a more standard seed
        self.train_eps = 200  # Full training episodes
        self.state_space_dim = 50
        self.lookback_window = 20
        self.action_space_dim = 3
        
        # Algorithm parameters (default values)
        self.gamma = 0.87
        self.epsilon_start = 0.90
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.0001
        self.memory_capacity = 5000
        self.batch_size = 64
        self.target_update = 3
        self.hidden_dim = 128
        self.use_robust_reward = True  
        self.rho = 0.1  
        self.corruption_type = 'random'
        self.R_threshold = 22.0
        self.C_threshold = 22.0
        self.delta_1 = 0.3
        
        # Override with best parameters if provided
        if best_params:
            for param, value in best_params.items():
                if hasattr(self, param):
                    setattr(self, param, value)
        
        # Paths
        self.result_path = os.path.join(curr_path, "outputs", "optimized_training", f'{curr_time}', "results")
        self.model_path = os.path.join(curr_path, "outputs", "optimized_training", f'{curr_time}', "models")
        self.save = True

def load_best_hyperparameters():
    """Load the best hyperparameters from the tuning results."""
    results_path = os.path.join(curr_path, "outputs", "hyperparameter_tuning", "study_results", "best_hyperparameters.json")
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results['best_params']
    else:
        print("No hyperparameter tuning results found. Using default parameters.")
        return None

def main():
    """Main function to run optimized training."""
    print("=== DQN Trading with Optimized Hyperparameters ===")
    
    # Load best hyperparameters
    best_params = load_best_hyperparameters()
    
    # Create optimized configuration
    cfg = OptimizedConfig(best_params)
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Download data
    print("\n--- Downloading Training Data ---")
    train_data = download_data_with_cache(train_tickers, train_start_date, train_end_date)
    
    print("\n--- Downloading Testing Data ---")
    test_data = download_data_with_cache(test_tickers, test_start_date, test_end_date)
    
    # Create output directories
    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs(cfg.model_path, exist_ok=True)
    
    # Training
    print("\n--- Starting Optimized Training ---")
    env, agent = env_agent_config(train_data, cfg, 'train')
    rewards, ma_rewards, episode_rewards, train_sharpe_ratios, test_sharpe_ratios = train_with_testing(cfg, env, agent, test_data)
    
    # Save model
    agent.save(path=cfg.model_path)
    
    # Testing
    print("\n--- Starting Testing ---")
    test_env, test_agent = env_agent_config(test_data, cfg, 'test')
    test_agent.load(path=cfg.model_path)
    stocks, rewards, individual_returns, cumulative_returns = test(cfg, test_env, test_agent)
    
    # Calculate performance metrics
    buy_and_hold_rewards = [sum(test_data[stock]) for stock in stocks]
    dqn_test_sharpe = sharpe_ratio(rewards)
    buy_hold_sharpe = sharpe_ratio(buy_and_hold_rewards)
    dqn_train_sharpe = sharpe_ratio(episode_rewards)
    
    # Aggregate all DQN returns for overall metrics
    all_dqn_returns = []
    for stock_returns in individual_returns.values():
        all_dqn_returns.extend(stock_returns)
    
    dqn_cumulative_return = calculate_cumulative_return(all_dqn_returns)
    dqn_max_drawdown = calculate_maximum_drawdown(all_dqn_returns)
    dqn_roi_percentage, dqn_roi_dollars = calculate_roi(all_dqn_returns)
    dqn_annualized_return = calculate_periodic_returns(all_dqn_returns)
    
    buy_hold_cumulative_return = calculate_cumulative_return(buy_and_hold_rewards)
    buy_hold_max_drawdown = calculate_maximum_drawdown(buy_and_hold_rewards)
    buy_hold_roi_percentage, buy_hold_roi_dollars = calculate_roi(buy_and_hold_rewards)
    buy_hold_annualized_return = calculate_periodic_returns(buy_and_hold_rewards)
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZED DQN PERFORMANCE RESULTS")
    print("="*60)
    
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
    
    # Prepare final test results
    final_test_results = {
        'dqn_train_sharpe': dqn_train_sharpe,
        'dqn_test_sharpe': dqn_test_sharpe,
        'buy_hold_sharpe': buy_hold_sharpe,
        'stocks': stocks,
        'dqn_rewards': rewards,
        'buy_hold_rewards': buy_and_hold_rewards,
        'individual_returns': individual_returns,
        'cumulative_returns': cumulative_returns,
        'performance_metrics': {
            'dqn': {
                'cumulative_return': dqn_cumulative_return,
                'max_drawdown': dqn_max_drawdown,
                'roi_percentage': dqn_roi_percentage,
                'roi_dollars': dqn_roi_dollars,
                'annualized_return': dqn_annualized_return
            },
            'buy_hold': {
                'cumulative_return': buy_hold_cumulative_return,
                'max_drawdown': buy_hold_max_drawdown,
                'roi_percentage': buy_hold_roi_percentage,
                'roi_dollars': buy_hold_roi_dollars,
                'annualized_return': buy_hold_annualized_return
            }
        }
    }
    
    # Save results
    save_training_results(cfg, rewards, ma_rewards, episode_rewards, train_sharpe_ratios, 
                         test_sharpe_ratios, final_test_results, cfg.result_path)
    
    # Create plots
    fig, ax = plt.subplots(figsize=(12, 8))
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()[0]
    
    train_epochs = list(range(1, len(train_sharpe_ratios) + 1))
    ax.plot(train_epochs, train_sharpe_ratios, 'b-o', label='Training Sharpe Ratio', 
            linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Optimized Training Sharpe Ratio vs Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_path, 'optimized_sharpe_vs_epochs.jpg'))
    plt.close()
    
    # Final comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()[0]
    
    strategies = ['DQN Training', 'DQN Testing', 'Buy & Hold']
    sharpe_values = [dqn_train_sharpe, dqn_test_sharpe, buy_hold_sharpe]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    bars = ax.bar(strategies, sharpe_values, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, value in zip(bars, sharpe_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Optimized DQN vs Buy & Hold Performance')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_path, 'optimized_final_comparison.jpg'))
    plt.close()
    
    print("\n=== Optimized Training Complete ===")

if __name__ == "__main__":
    main() 