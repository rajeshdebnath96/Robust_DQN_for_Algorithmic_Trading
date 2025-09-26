import sys
import os
curr_path = os.path.dirname(__file__)

import gym
import torch
import numpy as np
import random
import yfinance as yf
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from dqn import DQN
from trading_env import TradingSystem_v0

# Import the data manager for caching
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('..')
from data_manager import DataManager, download_multiple_tickers

curr_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
random.seed(11)

# Define the tickers and the time range
start_date = dt.date(2020, 1, 1)
# end_date = dt.datetime.today().strftime('%Y-%m-%d')
end_date = dt.date(2025, 6, 24)
train_tickers = ["ZM",'X','META','MTCH','GOOG','PINS','SNAP','ETSY']
test_tickers = ['IAC','TTWO','BMBL','SOCL']


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
        self.seed = 11 # random seed
        self.train_eps = 200 # training episodes
        self.state_space_dim = 5 # state space size (K-value)
        self.action_space_dim = 3 # action space size (short: 0, neutral: 1, long: 2)
        ################################################################################

        ################################## algo hyperparameters ###################################
        self.gamma = 0.95  # discount factor
        self.epsilon_start = 0.90  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01  # end epsilon of e-greedy policy
        self.epsilon_decay = 500  # attenuation rate of epsilon in e-greedy policy
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 1000  # capacity of experience replay
        self.batch_size = 64  # size of mini-batch SGD
        self.target_update = 4  # update frequency of target network
        self.hidden_dim = 128  # dimension of hidden layer
        ################################################################################

        ################################# save path ##############################
        self.result_path = os.path.join(curr_path, "outputs", self.env_name, curr_time, "results")
        self.model_path = os.path.join(curr_path, "outputs", self.env_name, curr_time, "models")
        self.save = True  # whether to save the image
        ################################################################################


def env_agent_config(data, cfg, mode):
    ''' create environment and agent
    '''
    env = TradingSystem_v0(data, cfg.state_space_dim, mode)
    agent = DQN(cfg.state_space_dim, cfg.action_space_dim, cfg)
    if cfg.seed != 0:  # set random seeds
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    return env, agent


def train(cfg, env, agent):
    ''' training
    '''
    print('Start Training!')
    print(f'Environment：{cfg.env_name}, Algorithm：{cfg.algo_name}, Device：{cfg.device}')
    rewards = []  # record total rewards
    ma_rewards = []  # record moving average total rewards
    for i_ep in range(cfg.train_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            # print(len(state), f'type: {type(env.r_ts)}', env.r_ts.shape, action)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)  # save transition
            state = next_state
            agent.update()
            ep_reward += reward
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # update target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print('Episode：{}/{}, Reward：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
    print('Finish Training!')
    return rewards, ma_rewards


def test(cfg, env, agent):
    print('Start Testing!')
    print(f'Environment：{cfg.env_name}, Algorithm：{cfg.algo_name}, Device：{cfg.device}')
    ############# Test does not use e-greedy policy, so we set epsilon to 0 ###############
    cfg.epsilon_start = 0.0
    cfg.epsilon_end = 0.0
    ################################################################################
    stocks = env.tickers
    rewards = []  # record total rewards
    for i_ep in range(len(stocks)):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            ep_reward += reward.item()
            if done:
                break
        rewards.append(ep_reward)
        print(f"Episode：{i_ep + 1}/{len(stocks)}，Reward：{ep_reward:.1f}")
    print('Finish Testing!')
    return stocks, rewards


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
    print("=== DQN Trading with Cached Data Download ===")
    
    # Download training data with caching
    print("\n--- Downloading Training Data ---")
    train_data = download_data_with_cache(train_tickers, start_date, end_date)
    
    # Download testing data with caching
    print("\n--- Downloading Testing Data ---")
    test_data = download_data_with_cache(test_tickers, start_date, end_date)
    
    # Check if we have enough data
    if len(train_data) == 0:
        print("❌ No training data available. Exiting.")
        sys.exit(1)
    
    if len(test_data) == 0:
        print("⚠️  No testing data available. Using only training data for testing.")
        test_data = {}
    
    print(f'\nData Downloaded - Training: {len(train_data)} tickers, Testing: {len(test_data)} tickers')
    print(f'Training tickers: {list(train_data.keys())}')
    print(f'Testing tickers: {list(test_data.keys())}')
    print(train_data["GOOG"][:5])  # Print first 5 returns for a sample ticker
    
    # Create configuration
    cfg = Config()
    
    # Create output directories
    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs(cfg.model_path, exist_ok=True)
    
    # Training
    print("\n--- Starting Training ---")
    env, agent = env_agent_config(train_data, cfg, 'train')
    rewards, ma_rewards = train(cfg, env, agent)
    
    # Save model
    agent.save(path=cfg.model_path)
    print(f"✅ Model saved to {cfg.model_path}")
    
    # Plot training results
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(list(range(1, cfg.train_eps+1)), rewards, color='blue', label='rewards')
    ax.plot(list(range(1, cfg.train_eps+1)), ma_rewards, color='green', label='ma_rewards')
    ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Results')
    plt.savefig(os.path.join(cfg.result_path, 'train.jpg'))
    print(f"✅ Training plot saved to {cfg.result_path}/train.jpg")

    # Testing
    print("\n--- Starting Testing ---")
    all_data = {**train_data, **test_data}
    env, agent = env_agent_config(all_data, cfg, 'test')
    agent.load(path=cfg.model_path)
    stocks, rewards = test(cfg, env, agent)
    
    # Calculate buy and hold rewards
    buy_and_hold_rewards = [sum(all_data[stock]).item() for stock in stocks]
    
    # Plot testing results
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    width = 0.3
    x = np.arange(len(stocks))
    ax.bar(x, rewards, width=width, color='salmon', label='DQN')
    ax.bar(x+width, buy_and_hold_rewards, width=width, color='orchid', label='Buy and Hold')
    ax.set_xticks(x+width/2)
    ax.set_xticklabels(stocks, fontsize=12)
    ax.set_ylabel('Total Reward')
    ax.set_title('Testing Results: DQN vs Buy and Hold')
    ax.legend()
    plt.savefig(os.path.join(cfg.result_path, 'test.jpg'))
    print(f"✅ Testing plot saved to {cfg.result_path}/test.jpg")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Training completed with {len(train_data)} tickers")
    print(f"Testing completed with {len(stocks)} tickers")
    print(f"Results saved to: {cfg.result_path}")
    print(f"Model saved to: {cfg.model_path}")
    
    # Show cache info
    data_manager = DataManager()
    cache_info = data_manager.get_cache_info()
    print(f"Cache info: {cache_info['total_files']} files, {cache_info['cache_size_mb']:.2f} MB") 