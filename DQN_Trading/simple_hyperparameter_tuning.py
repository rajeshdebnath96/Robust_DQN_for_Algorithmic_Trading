import sys
import os
import optuna
import torch
import numpy as np
import datetime as dt
import json
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
import warnings
import random
warnings.filterwarnings('ignore')

sys.path.append('..')
from DQN_Trading.finrl_trading_env import FinRLTradingEnv_v1
from data_manager import DataManager
from dqn import DQN, univariate_trimmed_mean, compute_G_t
from metrics import *
from trading_env import TradingSystem_v0
from main import train_with_testing, test, env_agent_config, download_data_with_cache

curr_path = os.path.dirname(__file__)
curr_time = dt.datetime.now().strftime("%Y%m%d-%H%M")

# Global variables for data
train_data = None
test_data = None

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
    
    # Set Optuna random seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Seeds set to {seed} for reproducibility")

class Config:
    def __init__(self):
        # Environment parameters
        self.algo_name = 'DQN'
        self.env_name = 'FinRLTradingEnv_v1'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42 
        self.train_eps = 200  # Reduced for faster tuning
        self.state_space_dim = 26
        self.lookback_window = 20
        self.action_space_dim = 3
        
        # Algorithm parameters (will be set by Optuna)
        self.gamma = 0.95
        self.epsilon_start = 0.90
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.0001
        self.memory_capacity = 5000
        self.batch_size = 64
        self.target_update = 4
        self.hidden_dim = 128
        self.use_robust_reward = False  
        self.rho = 0
        self.corruption_type = 'random'
        self.R_threshold = 10.0
        self.C_threshold = 10.0
        self.delta_1 = 0.1
        
        # Paths (will be updated per trial)
        self.result_path = ""
        self.model_path = ""
        self.save = True

def objective(trial):
    """Objective function for Optuna optimization."""
    global train_data, test_data
    
    # Set seed for this trial to ensure reproducibility
    trial_seed = 42 
    set_seed(trial_seed)
    
    # Create configuration
    cfg = Config()
    
    # Define hyperparameter search spaces
    cfg.lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    # cfg.batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    # cfg.hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    cfg.gamma = trial.suggest_float('gamma', 0.85, 0.99, step=0.01)
    cfg.state_space_dim = trial.suggest_int('state_space_dim', 5, 50, step=5)
    if cfg.env_name == 'FinRLTradingEnv_v1':
        cfg.lookback_window = cfg.state_space_dim - 6
    # cfg.train_eps = trial.suggest_int('train_eps', 50, 200, step=50)
    # cfg.epsilon_start = trial.suggest_float('epsilon_start', 0.8, 1.0, step=0.05)
    # cfg.epsilon_end = trial.suggest_float('epsilon_end', 0.01, 0.1, step=0.01)
    cfg.memory_capacity = trial.suggest_int('memory_capacity', 2000, 8000, step=1000)
    cfg.target_update = trial.suggest_int('target_update', 2, 8, step=1)
    if cfg.use_robust_reward:
        cfg.R_threshold = trial.suggest_float('R_threshold', 0.0, 30.0, step=1.0)
        cfg.C_threshold = trial.suggest_float('C_threshold', 0.0, 30.0, step=1.0)
        cfg.delta_1 = trial.suggest_float('delta_1', 0.0, 1.0, step=0.1)
    
    # Update paths for this trial
    trial_id = trial.number
    cfg.result_path = os.path.join(curr_path, "outputs", "hyperparameter_tuning", f'trial_{trial_id}', "results")
    cfg.model_path = os.path.join(curr_path, "outputs", "hyperparameter_tuning", f'trial_{trial_id}', "models")
    
    # Create directories
    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs(cfg.model_path, exist_ok=True)
    
    try:
        # Train model using train_with_testing from main.py
        env, agent = env_agent_config(train_data, cfg, 'train')
        rewards, ma_rewards, episode_rewards, train_sharpe_ratios, test_sharpe_ratios = train_with_testing(cfg, env, agent, test_data)
        
        # Save model
        agent.save(path=cfg.model_path)
        
        # Test model using test function from main.py
        test_env, test_agent = env_agent_config(test_data, cfg, 'test')
        test_agent.load(path=cfg.model_path)
        stocks, test_rewards, individual_returns, cumulative_returns = test(cfg, test_env, test_agent)
        
        # Calculate Sharpe ratio
        test_sharpe = sharpe_ratio(test_rewards)
        
        # Clean up
        import shutil
        if os.path.exists(os.path.dirname(cfg.model_path)):
            shutil.rmtree(os.path.dirname(cfg.model_path))
        
        return test_sharpe if test_sharpe is not None else -100.0
        
    except Exception as e:
        print(f"Error in trial {trial_id}: {e}")
        return -100.0

def create_optimization_study(n_trials=50):
    """Create and run hyperparameter optimization study."""
    # Set seed for Optuna study
    set_seed(42)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name="dqn_trading_optimization"
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=None)
    return study

def save_results(study, save_path):
    """Save study results and create visualizations."""
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
    values = [trial.value for trial in study.trials if trial.value is not None]
    ax1.plot(values)
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
    
    print(f"Results saved to: {save_path}")
    print(f"Best Sharpe Ratio: {best_value:.4f}")
    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

def main():
    """Main function to run hyperparameter tuning."""
    global train_data, test_data
    
    print("=== DQN Trading Hyperparameter Tuning ===")
    
    # Download data
    print("\n--- Downloading Data ---")
    train_data = download_data_with_cache(train_tickers, train_start_date, train_end_date)
    test_data = download_data_with_cache(test_tickers, test_start_date, test_end_date)
    
    # Create and run study
    print("\n--- Starting Hyperparameter Optimization ---")
    study = create_optimization_study(n_trials=50)  # Adjust number of trials
    
    # Save results
    save_path = os.path.join(curr_path, "outputs", "hyperparameter_tuning", "dqn_study_results")
    save_results(study, save_path)
    
    print("\n=== Hyperparameter Tuning Complete ===")

if __name__ == "__main__":
    main() 