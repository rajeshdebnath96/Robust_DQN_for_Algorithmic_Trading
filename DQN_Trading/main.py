import sys
import os

sys.path.append('..')
from DQN_Trading.finrl_trading_env import FinRLTradingEnv_v1
from data_manager import DataManager
curr_path = os.path.dirname(__file__)

import gym
import torch
import numpy as np
import random
import yfinance as yf
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import json
from dqn import DQN, univariate_trimmed_mean, compute_G_t
from metrics import *
from trading_env import TradingSystem_v0

curr_time = dt.datetime.now().strftime("%Y%m%d-%H%M")

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

# Dow Jones Industrial Average (DJIA) tickers as of 2024
# Source: TradingView, Nasdaq, etc.
dow_jones_tickers = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'DOW', 'GS',
    'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
    'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'CRM'
]
# train_tickers = dow_jones_tickers[:5]
# Use the same for test, or split if you want to do out-of-sample
# For now, use all for both train and test as per user request
# test_tickers = dow_jones_tickers[:5]
train_tickers = ['ZM', 'X', 'META', 'MTCH', 'GOOG', 'PINS', 'SNAP', 'ETSY']
# test_tickers = ['IAC', 'TTWO', 'BMBL', 'SOCL']
test_tickers = train_tickers

# Define the tickers and the time range
train_start_date = dt.date(2015, 1, 1)
train_end_date = dt.date(2022, 12, 31)
# end_date = dt.datetime.today().strftime ('%Y-%m-%d')
test_start_date = dt.date(2023, 1, 1)
test_end_date = dt.date(2024, 12, 31)
# test_start_date, test_end_date = train_start_date, train_end_date

class Config:
    '''
    hyperparameters
    '''

    def __init__(self):
        ################################## env hyperparameters ###################################
        self.algo_name = 'DQN' # algorithmic name
        self.env_name = 'TradingSystem_v0' # environment name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # examine GPU
        self.seed = 42 # random seed
        self.train_eps = 500 if self.env_name == 'FinRLTradingEnv_v1' else 100
        self.state_space_dim = 5 # state space size (K-value)
        self.lookback_window = 20
        self.action_space_dim = 3 # action space size (short: 0, neutral: 1, long: 2)
        ################################################################################

        ################################## algo hyperparameters ###################################
        self.gamma = 0.95  # discount factor
        self.epsilon_start = 0.90  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01  # end epsilon of e-greedy policy
        self.epsilon_decay = 500  # attenuation rate of epsilon in e-greedy policy
        self.lr = 0.001206357403773036 # learning rate
        self.memory_capacity = 6000  # capacity of experience replay
        self.batch_size = 64  # size of mini-batch SGD
        self.target_update = 2  # update frequency of target network
        self.hidden_dim = 128  # dimension of hidden layer
        self.use_robust_reward = False  # whether to use robust reward estimation
        self.rho = 0  # probability to observe a corrupted reward
        self.corruption_type = 'random'  # flip the sign of the true reward when corrupted
        ################################################################################
        # Parameters for thresholding the robust reward
        self.R_threshold = 10.0   # expected reward scale (e.g., maximum absolute reward)
        self.C_threshold = 10.0   # constant multiplier for threshold
        self.delta_1 = 0.1       # delta parameter for threshold function

        ################################# save path ##############################
        self.result_path = os.path.join(curr_path, "outputs", self.env_name, f'{curr_time}_rho_{self.rho}_corruption_{self.corruption_type}', "results")
        self.model_path = os.path.join(curr_path, "outputs", self.env_name, f'{curr_time}_rho_{self.rho}_corruption_{self.corruption_type}', "models")
        self.save = True  # whether to save the image
        ################################################################################

def save_training_results(cfg, rewards, ma_rewards, episode_rewards, train_sharpe_ratios, 
                         test_sharpe_ratios, final_test_results, save_path):
    """
    Save training results to JSON files for later analysis and plotting.
    
    Args:
        cfg: Configuration object
        rewards: List of episode rewards
        ma_rewards: List of moving average rewards
        episode_rewards: List of individual episode rewards
        train_sharpe_ratios: List of training Sharpe ratios
        test_sharpe_ratios: List of testing Sharpe ratios
        final_test_results: Dictionary with final test results
        save_path: Path to save the results
    """
    # Create results directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare training metrics
    training_metrics = {
        'config': {
            'algo_name': cfg.algo_name,
            'env_name': cfg.env_name,
            'train_eps': cfg.train_eps,
            'state_space_dim': cfg.state_space_dim,
            'action_space_dim': cfg.action_space_dim,
            'gamma': cfg.gamma,
            'epsilon_start': cfg.epsilon_start,
            'epsilon_end': cfg.epsilon_end,
            'epsilon_decay': cfg.epsilon_decay,
            'lr': cfg.lr,
            'memory_capacity': cfg.memory_capacity,
            'batch_size': cfg.batch_size,
            'target_update': cfg.target_update,
            'hidden_dim': cfg.hidden_dim,
            'use_robust_reward': cfg.use_robust_reward,
            'rho': cfg.rho,
            'corruption_type': cfg.corruption_type,
            'R_threshold': cfg.R_threshold,
            'C_threshold': cfg.C_threshold,
            'delta_1': cfg.delta_1
        },
        'training': {
            'rewards': rewards,
            'ma_rewards': ma_rewards,
            'episode_rewards': episode_rewards,
            'train_sharpe_ratios': train_sharpe_ratios,
            'test_sharpe_ratios': test_sharpe_ratios,   
        },
        'final_results': final_test_results,
        'metadata': {
            'timestamp': curr_time,
            'train_start_date': train_start_date.strftime('%Y-%m-%d'),
            'train_end_date': train_end_date.strftime('%Y-%m-%d'),
            'test_start_date': test_start_date.strftime('%Y-%m-%d'),
            'test_end_date': test_end_date.strftime('%Y-%m-%d'),
            'tickers': train_tickers
        }
    }
    
    # Save to JSON file
    results_file = os.path.join(save_path, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(training_metrics, f, indent=2, default=str)
    
    print(f"Training results saved to: {results_file}")
    
    # Also save a summary file for quick reference
    summary = {
        'run_id': f'{curr_time}_rho_{cfg.rho}_corruption_{cfg.corruption_type}',
        'final_train_sharpe': train_sharpe_ratios[-1] if train_sharpe_ratios else None,
        'final_test_sharpe': test_sharpe_ratios[-1] if test_sharpe_ratios else None,
        'final_dqn_test_sharpe': final_test_results.get('dqn_test_sharpe'),
        'final_buy_hold_sharpe': final_test_results.get('buy_hold_sharpe'),
        'config': {
            'rho': cfg.rho,
            'corruption_type': cfg.corruption_type,
            'use_robust_reward': cfg.use_robust_reward
        }
    }
    
    summary_file = os.path.join(save_path, 'run_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Run summary saved to: {summary_file}")


def env_agent_config(data, cfg, mode):
    ''' create environment and agent
    '''
    if cfg.env_name == 'FinRLTradingEnv_v1':
        env = FinRLTradingEnv_v1(data, cfg.lookback_window, rho=cfg.rho, corruption_type=cfg.corruption_type, mode=mode, use_technical_indicators=True)
    elif cfg.env_name == 'TradingSystem_v0':
        env = TradingSystem_v0(data, cfg.state_space_dim, rho=cfg.rho, corruption_type=cfg.corruption_type, mode=mode)
    else:
        raise ValueError(f"Invalid environment name: {cfg.env_name}")
    agent = DQN(cfg.state_space_dim, cfg.action_space_dim, cfg)
    if cfg.seed != 0:  # set random seeds
        set_seed(cfg.seed)
    return env, agent


def test_during_training(cfg, env, agent, test_data):
    """Test the current model during training to get Sharpe ratio"""
    print('Testing current model...')
    # Temporarily set epsilon to 0 for testing
    original_epsilon_start = cfg.epsilon_start
    original_epsilon_end = cfg.epsilon_end
    cfg.epsilon_start = 0.0
    cfg.epsilon_end = 0.0
    
    # Create test environment
    if cfg.env_name == 'FinRLTradingEnv_v1':
        test_env = FinRLTradingEnv_v1(test_data, cfg.lookback_window, rho=cfg.rho, corruption_type=cfg.corruption_type, mode='test', use_technical_indicators=True)
    elif cfg.env_name == 'TradingSystem_v0':
        test_env = TradingSystem_v0(test_data, cfg.state_space_dim, rho=cfg.rho, corruption_type=cfg.corruption_type, mode='test')
    else:
        raise ValueError(f"Invalid environment name: {cfg.env_name}")
    test_stocks = test_env.tickers
    test_rewards = []
    
    for i_ep in range(len(test_stocks)):
        ep_reward = 0
        state = test_env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = test_env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        test_rewards.append(ep_reward)
    
    # Restore original epsilon values
    cfg.epsilon_start = original_epsilon_start
    cfg.epsilon_end = original_epsilon_end
    
    # Calculate Sharpe ratio    
    test_sharpe = sharpe_ratio(test_rewards)
    return test_sharpe

def train_with_testing(cfg, env, agent, test_data):
    ''' training with periodic testing
    '''
    print('Start Training with Testing!')
    print(f'Environment：{cfg.env_name}, Algorithm：{cfg.algo_name}, Device：{cfg.device}')
    rewards = []  # record total rewards
    ma_rewards = []  # record moving average total rewards
    episode_rewards = []  # record individual episode rewards for Sharpe calculation
    train_sharpe_ratios = []  # record training Sharpe ratios
    test_sharpe_ratios = []  # record testing Sharpe ratios
    
    
    for i_ep in range(cfg.train_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            if cfg.use_robust_reward:
                # --- Robust Reward Estimation using Univariate Trimmed Mean ---
                # Use a (state, action) key to maintain reward buffers.
                key = (tuple(state), action)
                if key not in agent.Z:
                    agent.Z[key] = []
                    agent.Z_tilde[key] = []
                agent.Z[key].append(reward)
                agent.Z_tilde[key].append(reward)
                # Truncate buffers to maintain fixed size.
                if len(agent.Z[key]) > agent.robust_buffer_len:
                    agent.Z[key] = agent.Z[key][-agent.robust_buffer_len:]
                    agent.Z_tilde[key] = agent.Z_tilde[key][-agent.robust_buffer_len:]
                # Compute robust reward only when the buffer is full; otherwise, fallback to raw reward.
                if len(agent.Z[key]) == agent.robust_buffer_len:
                    robust_reward = univariate_trimmed_mean(agent.Z[key], agent.Z_tilde[key],
                                                            agent.robust_epsilon, agent.robust_delta)
                    # Use threshold function to clip the robust reward.
                    t_val = len(agent.Z[key])
                    G_threshold = compute_G_t(t_val, agent.R_threshold, agent.C_threshold,
                                            agent.robust_epsilon, agent.delta_1)
                    reward = np.clip(robust_reward, -G_threshold, G_threshold)
                else:
                    reward = reward

            # -------------------------------------------------------------
            agent.memory.push(state, action, reward, next_state, done)  # save transition
            state = next_state
            agent.update()
            ep_reward += reward
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # update target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        episode_rewards.append(ep_reward)  # Store for Sharpe calculation
                
        current_train_sharpe = sharpe_ratio(episode_rewards)
        train_sharpe_ratios.append(current_train_sharpe)
        
        # current_test_sharpe = test_during_training(cfg, env, agent, test_data)
        current_test_sharpe = 0
        # test_sharpe_ratios.append(current_test_sharpe)
        print('Episode：{}/{}, Train Sharpe：{:.3f}, Test Sharpe：{:.3f}'.format(
            i_ep + 1, cfg.train_eps, current_train_sharpe, current_test_sharpe))
        
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    
    print('Finish Training!')
    return rewards, ma_rewards, episode_rewards, train_sharpe_ratios, test_sharpe_ratios


def test(cfg, env, agent):
    print('Start Testing!')
    print(f'Environment：{cfg.env_name}, Algorithm：{cfg.algo_name}, Device：{cfg.device}')
    ############# Test does not use e-greedy policy, so we set epsilon to 0 ###############
    cfg.epsilon_start = 0.0
    cfg.epsilon_end = 0.0
    ################################################################################
    stocks = env.tickers
    rewards = []  # record total rewards
    individual_returns = {}  # record individual returns for each stock
    cumulative_returns = {}  # record cumulative returns for each stock
    
    for i_ep in range(len(stocks)):
        ep_reward = 0
        stock_returns = []  # track returns for this stock
        state = env.reset()
        step_count = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward
            stock_returns.append(reward)  # Track individual returns
            step_count += 1
            
            if done:
                break
        
        rewards.append(ep_reward)
        individual_returns[stocks[i_ep]] = stock_returns
        cumulative_returns[stocks[i_ep]] = calculate_cumulative_return(stock_returns)
        
        print(f"Episode：{i_ep + 1}/{len(stocks)}，Stock: {stocks[i_ep]}，Reward：{ep_reward:.4f}，Cumulative Return: {cumulative_returns[stocks[i_ep]]:.2f}%")
    
    print('Finish Testing!')
    return stocks, rewards, individual_returns, cumulative_returns

def download_data_with_cache(tickers, start_date, end_date, cache_dir="data_cache"):
    """
    Download data with caching support.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        cache_dir: Directory for caching
        
    Returns:
        Dictionary mapping tickers to their returns data
    """
    print(f"Downloading data for {len(tickers)} tickers with caching...")
    
    # Convert dates to string format
    start_date_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
    end_date_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
    
    # Use the data manager for downloading with caching
    data_manager = DataManager(cache_dir)
    
    # Download data with caching
    data_dict = data_manager.batch_download_data(
        tickers=tickers,
        start_date=start_date_str,
        end_date=end_date_str,
        force_download=False,  # Use cache if available
        max_cache_age_days=7   # Cache is valid for 7 days
    )
    
    print(f"Successfully downloaded data for {len(data_dict)} tickers")
    print(f"Available tickers: {list(data_dict.keys())}")
    
    return data_dict

if __name__ == "__main__":

    print("\n--- Downloading Training Data ---")
    train_data = download_data_with_cache(train_tickers, train_start_date, train_end_date)
    
    # Download testing data with caching
    print("\n--- Downloading Testing Data ---")
    test_data = download_data_with_cache(test_tickers, test_start_date, test_end_date)
    
    print('Data Downloaded:"', test_data.keys())
    cfg = Config()
    # training
    env, agent = env_agent_config(train_data, cfg, 'train')
    rewards, ma_rewards, episode_rewards, train_sharpe_ratios, test_sharpe_ratios = train_with_testing(cfg, env, agent, test_data)
    os.makedirs(cfg.result_path, exist_ok=True)  # create output folders
    os.makedirs(cfg.model_path, exist_ok=True)
    agent.save(path=cfg.model_path)  # save model
    
    # Plot Sharpe ratio vs epochs
    fig, ax = plt.subplots(figsize=(12, 8))
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()[0]
    
    # Filter out None values and get corresponding epochs for training Sharpe
    train_epochs = []
    train_sharpe_filtered = []
    for i, sharpe in enumerate(train_sharpe_ratios):
        if sharpe is not None:
            train_epochs.append((i + 1))  # Every 10 episodes
            train_sharpe_filtered.append(sharpe)
    
    # Plot training Sharpe ratio
    ax.plot(train_epochs, train_sharpe_ratios, 'b-o', label='Training Sharpe Ratio', 
            linewidth=2, markersize=6, alpha=0.8)
    
    # Plot testing Sharpe ratio
    if test_sharpe_ratios:
        ax.plot(train_epochs, test_sharpe_ratios, 'r-s', label='Testing Sharpe Ratio', 
                linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Training vs Testing Sharpe Ratio vs Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.result_path+'/sharpe_vs_epochs.jpg')
    plt.close()
    
    # Original training rewards plot
    fig, ax = plt.subplots(figsize=(10, 7))   # plot the training result
    # Ensure ax is a single Axes object (not ndarray)
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()[0]
    ax.plot(list(range(1, cfg.train_eps+1)), rewards, color='blue', label='rewards')
    ax.plot(list(range(1, cfg.train_eps+1)), ma_rewards, color='green', label='ma_rewards')
    ax.legend()
    ax.set_xlabel('Episode')
    plt.savefig(cfg.result_path+'/train.jpg')

    # testing
    # all_data = {**train_data, **test_data}
    env, agent = env_agent_config(test_data, cfg, 'test')
    agent.load(path=cfg.model_path)  # load model
    stocks, rewards, individual_returns, cumulative_returns = test(cfg, env, agent)
    buy_and_hold_rewards = [sum(test_data[stock]) for stock in stocks]
    # Calculate Sharpe ratios
    dqn_test_sharpe = sharpe_ratio(rewards)
    buy_hold_sharpe = sharpe_ratio(buy_and_hold_rewards)
    dqn_train_sharpe = sharpe_ratio(episode_rewards)
    
    # Calculate comprehensive performance metrics for DQN
    print("\n" + "="*60)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Aggregate all DQN returns for overall metrics
    all_dqn_returns = []
    for stock_returns in individual_returns.values():
        all_dqn_returns.extend(stock_returns)
    
    # Calculate overall DQN metrics
    dqn_cumulative_return = calculate_cumulative_return(all_dqn_returns)
    dqn_max_drawdown = calculate_maximum_drawdown(all_dqn_returns)
    dqn_roi_percentage, dqn_roi_dollars = calculate_roi(all_dqn_returns)
    dqn_annualized_return = calculate_periodic_returns(all_dqn_returns)
    
    # Calculate Buy & Hold metrics
    buy_hold_cumulative_return = calculate_cumulative_return(buy_and_hold_rewards)
    buy_hold_max_drawdown = calculate_maximum_drawdown(buy_and_hold_rewards)
    buy_hold_roi_percentage, buy_hold_roi_dollars = calculate_roi(buy_and_hold_rewards)
    buy_hold_annualized_return = calculate_periodic_returns(buy_and_hold_rewards)
    
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
    
    print(f"\nDQN Training Sharpe Ratio: {dqn_train_sharpe:.3f}")
    print(f"DQN Testing Sharpe Ratio: {dqn_test_sharpe:.3f}")
    print(f"Buy & Hold Sharpe Ratio: {buy_hold_sharpe:.3f}")
    
    # Prepare final test results for saving
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
    
    # Save training results for later analysis
    save_training_results(cfg, rewards, ma_rewards, episode_rewards, train_sharpe_ratios, 
                         test_sharpe_ratios, final_test_results, cfg.result_path)
    
    # Create comprehensive performance visualization
    create_performance_visualizations(cfg, all_dqn_returns, buy_and_hold_rewards, 
                                    individual_returns, stocks, final_test_results)
    
    # Plot final comparison of all strategies
    fig, ax = plt.subplots(figsize=(10, 6))
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()[0]
    
    # Create bar plot for final Sharpe ratios
    strategies = ['DQN Training', 'DQN Testing', 'Buy & Hold']
    sharpe_values = [dqn_train_sharpe, dqn_test_sharpe, buy_hold_sharpe]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    bars = ax.bar(strategies, sharpe_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, sharpe_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Final Sharpe Ratio Comparison')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.result_path+'/final_sharpe_comparison.jpg')
    plt.close()
    
    # Original test results plot
    fig, ax = plt.subplots(figsize=(14, 8))  # plot the test result
    # Ensure ax is a single Axes object (not ndarray)
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()[0]
    width = 0.3
    x = np.arange(len(stocks))
    ax.bar(x, rewards, width=width, color='salmon', label='DQN')
    ax.bar(x+width, buy_and_hold_rewards, width=width, color='orchid', label='Buy and Hold')
    ax.set_xticks(x+width/2)
    ax.set_xticklabels(stocks, fontsize=10, rotation=45)
    ax.legend()
    ax.set_title(f"Test Results (Sharpe: DQN={dqn_test_sharpe:.2f}, Buy&Hold={buy_hold_sharpe:.2f})")
    plt.tight_layout()
    plt.savefig(cfg.result_path+'/test.jpg')
    plt.close()







