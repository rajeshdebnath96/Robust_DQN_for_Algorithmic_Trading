#!/usr/bin/env python3
"""
Script to plot combined training MA and Sharpe ratio data from optimized training results.
Uses rho from config as labels for each run.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import glob

def load_optimized_training_results(results_dir: str) -> Dict:
    """
    Load optimized training results from a JSON file.
    
    Args:
        results_dir: Directory containing optimized_training_results.json
        
    Returns:
        Dictionary with training results
    """
    results_file = os.path.join(results_dir, 'optimized_training_results.json')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def find_optimized_training_runs(base_output_dir: str) -> List[Dict]:
    """
    Find all optimized training runs in the output directory.
    
    Args:
        base_output_dir: Base directory containing all run outputs
        
    Returns:
        List of dictionaries with run information
    """
    runs = []
    
    # Look for directories matching the pattern: YYYYMMDD-HHMM
    # pattern = os.path.join(base_output_dir, "*-*")
    # run_dirs = glob.glob(pattern)

    # selected_dirs = ["20250706-1557", "20250706-1633", "20250714-0456"]
    selected_dirs = ["20250706-1557", "20250706-1609", "20250706-1633"]
    run_dirs = [os.path.join(base_output_dir, d) for d in selected_dirs]
    
    for run_dir in run_dirs:
        results_dir = os.path.join(run_dir, "results")
        if os.path.exists(results_dir):
            try:
                results = load_optimized_training_results(results_dir)
                
                # Extract run ID from directory name
                run_id = os.path.basename(run_dir)
                
                run_info = {
                    'run_dir': run_dir,
                    'results_dir': results_dir,
                    'run_id': run_id,
                    'results': results
                }
                runs.append(run_info)
                print(f"Loaded run: {run_id}")
            except Exception as e:
                print(f"Failed to load run from {run_dir}: {e}")
    
    return runs

def plot_combined_training_ma_and_sharpe(runs: List[Dict], save_path: Optional[str] = None):
    """
    Plot combined training MA and Sharpe ratio data for multiple runs.
    
    Args:
        runs: List of run dictionaries
        save_path: Path to save the plot
    """
    if not runs:
        print("No runs to plot")
        return
    # run_label = {"20250706-1557": "Vanilla DQN", "20250706-1633": "ρ=0.01 (Corruption Fraction)", "20250714-0456": "Robust DQN"}
    run_label = {"20250706-1557": "Vanilla DQN", "20250706-1609": "ρ=0.1 (Corruption Fraction)", "20250706-1633": "ρ=0.01 (Corruption Fraction)"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Use a colormap for better color distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    ep_limit = 75
    
    for i, run in enumerate(runs):
        results = run['results']
        config = results['config']
        training_data = results['training']
        run_id = run['run_id']
        
        # Get rho value for labeling
        rho = config['rho']
        corruption_type = config['corruption_type']
        
        # Create label with rho and corruption type
        label = run_label[run_id]
        
        # Plot training MA rewards
        ma_rewards = training_data['ma_rewards'][:ep_limit]
        episodes = list(range(1, len(ma_rewards) + 1))[:ep_limit]
        
        ax1.plot(episodes, ma_rewards, color=colors[i], alpha=0.8, 
                label=label, linewidth=2, marker='o', markersize=3)
        
        # Plot training Sharpe ratios
        train_sharpe = training_data['train_sharpe_ratios'][:ep_limit]
        if train_sharpe:
            ax2.plot(episodes, train_sharpe, color=colors[i], alpha=0.8, 
                    label=label, linewidth=2, marker='s', markersize=3)
    
    # Configure MA rewards plot
    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('Moving Average Reward')
    ax1.set_title('Training Moving Average Rewards - All Optimized Runs')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Configure Sharpe ratios plot
    ax2.set_xlabel('Training Episodes')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Training Sharpe Ratios - All Optimized Runs')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # plt.suptitle('Combined Training Performance: MA Rewards and Sharpe Ratios', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined training plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_individual_training_curves(runs: List[Dict], save_path: Optional[str] = None):
    """
    Plot individual training curves for each run in separate subplots.
    
    Args:
        runs: List of run dictionaries
        save_path: Path to save the plot
    """
    if not runs:
        print("No runs to plot")
        return
    
    # Calculate grid dimensions
    n_runs = len(runs)
    n_cols = min(3, n_runs)  # Max 3 columns
    n_rows = (n_runs + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Handle single subplot case
    if n_runs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_runs))
    
    for i, run in enumerate(runs):
        results = run['results']
        config = results['config']
        training_data = results['training']
        
        # Get rho value for labeling
        rho = config['rho']
        corruption_type = config['corruption_type']
        
        # Determine subplot position
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
        
        # Plot MA rewards
        ma_rewards = training_data['ma_rewards']
        episodes = list(range(1, len(ma_rewards) + 1))
        
        ax.plot(episodes, ma_rewards, color=colors[i], alpha=0.8, 
                label='MA Rewards', linewidth=2, marker='o', markersize=3)
        
        # Plot Sharpe ratios on secondary y-axis
        ax2 = ax.twinx()
        train_sharpe = training_data['train_sharpe_ratios']
        if train_sharpe:
            ax2.plot(episodes, train_sharpe, color='red', alpha=0.8, 
                    label='Sharpe Ratio', linewidth=2, marker='s', markersize=3)
        
        # Configure subplot
        ax.set_xlabel('Training Episodes')
        ax.set_ylabel('Moving Average Reward', color=colors[i])
        ax2.set_ylabel('Sharpe Ratio', color='red')
        ax.set_title(f'ρ={rho} ({corruption_type})')
        ax.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Hide empty subplots
    for i in range(n_runs, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
        ax.set_visible(False)
    
    plt.suptitle('Individual Training Curves: MA Rewards and Sharpe Ratios', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Individual training curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_training_summary_table(runs: List[Dict], save_path: Optional[str] = None):
    """
    Create a summary table of training performance for all runs.
    
    Args:
        runs: List of run dictionaries
        save_path: Path to save the table
    """
    data = []
    
    for run in runs:
        run_id = run['run_id']
        results = run['results']
        config = results['config']
        training_data = results['training']
        final_results = results['final_results']
        
        # Calculate final MA reward
        ma_rewards = training_data['ma_rewards']
        final_ma_reward = ma_rewards[-1] if ma_rewards else 0
        
        # Calculate final Sharpe ratio
        train_sharpe = training_data['train_sharpe_ratios']
        final_train_sharpe = train_sharpe[-1] if train_sharpe else 0
        
        data.append({
            'Run ID': run_id,
            'ρ (Corruption Prob)': config['rho'],
            'Corruption Type': config['corruption_type'],
            'Final MA Reward': final_ma_reward,
            'Final Train Sharpe': final_train_sharpe,
            'Final Test Sharpe': final_results['dqn_test_sharpe'],
            'Buy & Hold Sharpe': final_results['buy_hold_sharpe'],
            'Train Episodes': config['train_eps'],
            'Learning Rate': config['lr'],
            'Gamma': config['gamma']
        })
    
    df = pd.DataFrame(data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Training summary table saved to: {save_path}")
    
    print("\n=== Training Summary Table ===")
    print(df.to_string(index=False))
    
    return df

def main():
    """
    Main function to load and plot combined optimized training results.
    """
    # Base directory containing all run outputs
    base_output_dir = "outputs/optimized_training"
    
    if not os.path.exists(base_output_dir):
        print(f"Base output directory not found: {base_output_dir}")
        return
    
    # Find all runs
    runs = find_optimized_training_runs(base_output_dir)
    
    if not runs:
        print("No optimized training runs found!")
        return
    
    print(f"\nFound {len(runs)} optimized training runs")
    
    # Create output directory for combined plots
    combined_output_dir = "combined_optimized_analysis"
    os.makedirs(combined_output_dir, exist_ok=True)
    
    # Generate combined plots
    plot_combined_training_ma_and_sharpe(runs, 
                                       os.path.join(combined_output_dir, "combined_training_ma_sharpe.png"))
    # plot_individual_training_curves(runs, 
    #                               os.path.join(combined_output_dir, "individual_training_curves.png"))
    create_training_summary_table(runs, 
                                  os.path.join(combined_output_dir, "training_summary_table.csv"))
    
    print(f"\nAll combined optimized training analysis saved to: {combined_output_dir}")

if __name__ == "__main__":
    main() 