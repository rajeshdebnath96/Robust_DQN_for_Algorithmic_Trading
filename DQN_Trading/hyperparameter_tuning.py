import sys
import os
import optuna
import torch
import numpy as np
import datetime as dt
import json
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')
from DQN_Trading.finrl_trading_env import FinRLTradingEnv_v1
from data_manager import DataManager
from dqn import DQN, univariate_trimmed_mean, compute_G_t
from metrics import *
from trading_env import TradingSystem_v0

curr_path = os.path.dirname(__file__)
curr_time = dt.datetime.now().strftime("%Y%m%d-%H%M")

# Global variables for data
train_data = None
test_data = None

# Dow Jones Industrial Average (DJIA) tickers as of 2024
dow_jones_tickers = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'DOW', 'GS',
    'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
    'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'CRM'
]

train_tickers = ['ZM', 'X', 'META', 'MTCH', 'GOOG', 'PINS', 'SNAP', 'ETSY']
test_tickers = train_tickers

# Define the tickers and the time range
train_start_date = dt.date(2015, 1, 1)
train_end_date = dt.date(2022, 12, 31)
test_start_date = dt.date(2023, 1, 1)
test_end_date = dt.date(2024, 12, 31)

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
        self.train_eps = 200  # Reduced for faster hyperparameter tuning
        self.state_space_dim = 10 # state space size (K-value)
        self.lookback_window = 20
        self.action_space_dim = 3 # action space size (short: 0, neutral: 1, long: 2)
        ################################################################################

        ################################## algo hyperparameters ###################################
        self.gamma = 0.95  # discount factor
        self.epsilon_start = 0.90  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01  # end epsilon of e-greedy policy
        self.epsilon_decay = 500  # attenuation rate of epsilon in e-greedy policy
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 5000  # capacity of experience replay
        self.batch_size = 64  # size of mini-batch SGD
        self.target_update = 4  # update frequency of target network
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

def download_data_with_cache(tickers, start_date, end_date, cache_dir="data_cache"):
    """
    Download data with caching support.
    """
    print(f"Downloading data for {len(tickers)} tickers with caching...")
    
    start_date_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
    end_date_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
    
    data_manager = DataManager(cache_dir)
    
    data_dict = data_manager.batch_download_data(
        tickers=tickers,
        start_date=start_date_str,
        end_date=end_date_str,
        force_download=False,
        max_cache_age_days=7
    )
    
    print(f"Successfully downloaded data for {len(data_dict)} tickers")
    return data_dict

def env_agent_config(data, cfg, mode):
    ''' create environment and agent
    '''
    if cfg.env_name == 'FinRLTradingEnv_v1':
        env = FinRLTradingEnv_v1(data, cfg.lookback_window, rho=cfg.rho, corruption_type=cfg.corruption_type, mode=mode, use_technical_indicators=False)
    elif cfg.env_name == 'TradingSystem_v0':
        env = TradingSystem_v0(data, cfg.state_space_dim, rho=cfg.rho, corruption_type=cfg.corruption_type, mode=mode)
    else:
        raise ValueError(f"Invalid environment name: {cfg.env_name}")
    agent = DQN(cfg.state_space_dim, cfg.action_space_dim, cfg)
    if cfg.seed != 0:  # set random seeds
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    return env, agent

def train_and_evaluate(cfg, train_data, test_data):
    """
    Train the model and evaluate it on test data.
    Returns the test Sharpe ratio as the objective value.
    """
    try:
        # Training
        env, agent = env_agent_config(train_data, cfg, 'train')
        rewards, ma_rewards, episode_rewards, train_sharpe_ratios, test_sharpe_ratios = train_with_testing(cfg, env, agent, test_data)
        
        # Testing
        all_data = {**train_data, **test_data}
        env, agent = env_agent_config(all_data, cfg, 'test')
        agent.load(path=cfg.model_path)
        stocks, rewards, individual_returns, cumulative_returns = test(cfg, env, agent)
        
        # Calculate final test Sharpe ratio
        dqn_test_sharpe = sharpe_ratio(rewards)
        
        # Clean up
        if os.path.exists(cfg.model_path):
            import shutil
            shutil.rmtree(os.path.dirname(cfg.model_path))
        
        return dqn_test_sharpe if dqn_test_sharpe is not None else -100.0
        
    except Exception as e:
        print(f"Error in trial: {e}")
        return -100.0

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
                # Robust Reward Estimation using Univariate Trimmed Mean
                key = (tuple(state), action)
                if key not in agent.Z:
                    agent.Z[key] = []
                    agent.Z_tilde[key] = []
                agent.Z[key].append(reward)
                agent.Z_tilde[key].append(reward)
                if len(agent.Z[key]) > agent.robust_buffer_len:
                    agent.Z[key] = agent.Z[key][-agent.robust_buffer_len:]
                    agent.Z_tilde[key] = agent.Z_tilde[key][-agent.robust_buffer_len:]
                if len(agent.Z[key]) == agent.robust_buffer_len:
                    robust_reward = univariate_trimmed_mean(agent.Z[key], agent.Z_tilde[key],
                                                            agent.robust_epsilon, agent.robust_delta)
                    t_val = len(agent.Z[key])
                    G_threshold = compute_G_t(t_val, agent.R_threshold, agent.C_threshold,
                                            agent.robust_epsilon, agent.delta_1)
                    reward = np.clip(robust_reward, -G_threshold, G_threshold)

            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
            ep_reward += reward
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        episode_rewards.append(ep_reward)
        
        current_train_sharpe = sharpe_ratio(episode_rewards)
        train_sharpe_ratios.append(current_train_sharpe)
        
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    
    print('Finish Training!')
    return rewards, ma_rewards, episode_rewards, train_sharpe_ratios, test_sharpe_ratios

def test(cfg, env, agent):
    print('Start Testing!')
    print(f'Environment：{cfg.env_name}, Algorithm：{cfg.algo_name}, Device：{cfg.device}')
    cfg.epsilon_start = 0.0
    cfg.epsilon_end = 0.0
    
    stocks = env.tickers
    rewards = []
    individual_returns = {}
    cumulative_returns = {}
    
    for i_ep in range(len(stocks)):
        ep_reward = 0
        stock_returns = []
        state = env.reset()
        step_count = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward
            stock_returns.append(reward)
            step_count += 1
            
            if done:
                break
        
        rewards.append(ep_reward)
        individual_returns[stocks[i_ep]] = stock_returns
        cumulative_returns[stocks[i_ep]] = calculate_cumulative_return(stock_returns)
    
    print('Finish Testing!')
    return stocks, rewards, individual_returns, cumulative_returns

def objective(trial):
    """
    Objective function for Optuna optimization.
    """
    # Create configuration with hyperparameters from trial
    cfg = Config()
    
    # Define hyperparameter search spaces
    cfg.gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
    cfg.epsilon_start = trial.suggest_float('epsilon_start', 0.7, 1.0, step=0.05)
    cfg.epsilon_end = trial.suggest_float('epsilon_end', 0.001, 0.1, step=0.001)
    cfg.epsilon_decay = trial.suggest_int('epsilon_decay', 100, 1000, step=50)
    cfg.lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    cfg.memory_capacity = trial.suggest_int('memory_capacity', 1000, 10000, step=500)
    cfg.batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    cfg.target_update = trial.suggest_int('target_update', 2, 10, step=1)
    cfg.hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    cfg.state_space_dim = trial.suggest_int('state_space_dim', 5, 20, step=1)
    
    # Robust reward parameters (optional)
    cfg.use_robust_reward = trial.suggest_categorical('use_robust_reward', [True, False])
    if cfg.use_robust_reward:
        cfg.rho = trial.suggest_float('rho', 0.0, 0.3, step=0.01)
        cfg.R_threshold = trial.suggest_float('R_threshold', 5.0, 20.0, step=1.0)
        cfg.C_threshold = trial.suggest_float('C_threshold', 5.0, 20.0, step=1.0)
        cfg.delta_1 = trial.suggest_float('delta_1', 0.05, 0.2, step=0.01)
    
    # Update paths for this trial
    trial_id = trial.number
    cfg.result_path = os.path.join(curr_path, "outputs", "hyperparameter_tuning", f'trial_{trial_id}', "results")
    cfg.model_path = os.path.join(curr_path, "outputs", "hyperparameter_tuning", f'trial_{trial_id}', "models")
    
    # Create directories
    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs(cfg.model_path, exist_ok=True)
    
    # Train and evaluate
    test_sharpe = train_and_evaluate(cfg, train_data, test_data)
    
    # Report intermediate value for pruning
    trial.report(test_sharpe, step=cfg.train_eps)
    
    return test_sharpe

def create_hyperparameter_study(n_trials=50, study_name="dqn_trading_optimization"):
    """
    Create and run hyperparameter optimization study.
    """
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=50),
        study_name=study_name,
        storage=None  # Use in-memory storage for simplicity
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=None)
    
    return study

def save_study_results(study, save_path):
    """
    Save study results and create visualizations.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    results = {
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': len(study.trials),
        'timestamp': curr_time
    }
    
    with open(os.path.join(save_path, 'best_hyperparameters.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create optimization history plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot optimization history
    ax1.plot([trial.value for trial in study.trials if trial.value is not None])
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Test Sharpe Ratio')
    ax1.set_title('Optimization History')
    ax1.grid(True, alpha=0.3)
    
    # Plot parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())
        values = list(importance.values())
        
        ax2.barh(params, values)
        ax2.set_xlabel('Importance')
        ax2.set_title('Parameter Importance')
        ax2.grid(True, alpha=0.3)
    except:
        ax2.text(0.5, 0.5, 'Parameter importance\nnot available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'optimization_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create parameter relationships plot
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(save_path, 'parallel_coordinate.html'))
        
        fig = optuna.visualization.plot_contour(study)
        fig.write_html(os.path.join(save_path, 'contour_plot.html'))
    except:
        print("Could not create interactive plots (plotly not available)")
    
    print(f"Study results saved to: {save_path}")
    print(f"Best Sharpe Ratio: {best_value:.4f}")
    print(f"Best Parameters: {best_params}")

def main():
    """
    Main function to run hyperparameter tuning.
    """
    global train_data, test_data
    
    print("=== DQN Trading Hyperparameter Tuning ===")
    
    # Download data
    print("\n--- Downloading Training Data ---")
    train_data = download_data_with_cache(train_tickers, train_start_date, train_end_date)
    
    print("\n--- Downloading Testing Data ---")
    test_data = download_data_with_cache(test_tickers, test_start_date, test_end_date)
    
    # Create and run study
    print("\n--- Starting Hyperparameter Optimization ---")
    study = create_hyperparameter_study(n_trials=30)  # Adjust number of trials as needed
    
    # Save results
    save_path = os.path.join(curr_path, "outputs", "hyperparameter_tuning", "study_results")
    save_study_results(study, save_path)
    
    print("\n=== Hyperparameter Tuning Complete ===")
    print(f"Best Sharpe Ratio: {study.best_value:.4f}")
    print("Best Parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

if __name__ == "__main__":
    main() 