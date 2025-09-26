#!/usr/bin/env python3
"""
Standalone script to generate combined plots from existing sharpe data files.
This script searches recursively through all experiment folders and creates combined plots.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import glob

def load_sharpe_data_recursive(search_path: str, run_name: Optional[str] = None) -> List[Dict]:
    """
    Load Sharpe ratio data recursively from all subdirectories.
    
    Args:
        search_path: Root path to search for sharpe data files
        run_name: Specific run name to load, or None to load all
    
    Returns:
        List of data dictionaries
    """
    data_files = []
    
    def find_sharpe_files(directory):
        """Recursively find all sharpe_data_*.json files in directory and subdirectories."""
        files = []
        try:
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    if filename.startswith('sharpe_data_') and filename.endswith('.json'):
                        files.append(os.path.join(root, filename))
        except Exception as e:
            print(f"Error searching directory {directory}: {e}")
        return files
    
    if run_name:
        # Search for specific run name recursively
        all_files = find_sharpe_files(search_path)
        for file in all_files:
            if f'sharpe_data_{run_name}.json' in file:
                data_files.append(file)
                break
    else:
        # Load all sharpe data files recursively
        data_files = find_sharpe_files(search_path)
    
    print(f"Found {len(data_files)} sharpe data files:")
    for file in data_files:
        print(f"  - {file}")
    
    loaded_data = []
    for filename in data_files:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                # Add source file information to the data
                data['source_file'] = filename
                # Extract experiment info from path
                path_parts = filename.split(os.sep)
                if 'outputs' in path_parts:
                    outputs_idx = path_parts.index('outputs')
                    if outputs_idx + 2 < len(path_parts):
                        env_name = path_parts[outputs_idx + 1]
                        experiment_name = path_parts[outputs_idx + 2]
                        data['experiment_info'] = f"{env_name}/{experiment_name}"
                loaded_data.append(data)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return loaded_data

def plot_combined_sharpe_data_single_plot(data_list: List[Dict], save_path: str, title: str = "Combined Sharpe Ratio Comparison"):
    """
    Plot combined Sharpe ratio data from multiple runs in a single plot.
    
    Args:
        data_list: List of data dictionaries from load_sharpe_data
        save_path: Path to save the combined plot
        title: Title for the plot
    """
    if not data_list:
        print("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Ensure ax is a single Axes object (not ndarray)
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()[0]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 
              'darkblue', 'darkred', 'darkgreen', 'darkorange', 'darkviolet', 'saddlebrown', 'deeppink', 'dimgray', 'darkolivegreen', 'darkcyan']
    
    for i, data in enumerate(data_list):
        run_name = data['run_name']
        train_sharpe = data['train_sharpe_ratios']
        test_sharpe = data['test_sharpe_ratios']
        episodes = list(range(1, len(train_sharpe) + 1))
        
        # Use experiment info if available for better labeling
        label_prefix = data.get('experiment_info', run_name)
        
        # Plot training Sharpe ratios
        ax.plot(episodes, train_sharpe, 'o-', label=f'{label_prefix} (Train)', 
                color=colors[i % len(colors)], linewidth=2, markersize=4, alpha=0.8)
        
        # Plot testing Sharpe ratios
        if test_sharpe:
            ax.plot(episodes, test_sharpe, 's--', label=f'{label_prefix} (Test)', 
                    color=colors[i % len(colors)], linewidth=2, markersize=4, alpha=0.8)
    
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved to {save_path}")

def plot_combined_sharpe_data_separate_plots(data_list: List[Dict], save_path: str, title: str = "Combined Sharpe Ratio Comparison"):
    """
    Plot combined Sharpe ratio data from multiple runs in separate training and testing plots.
    
    Args:
        data_list: List of data dictionaries from load_sharpe_data
        save_path: Path to save the combined plot
        title: Title for the plot
    """
    if not data_list:
        print("No data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
              'darkblue', 'darkred', 'darkgreen', 'darkorange', 'darkviolet', 'saddlebrown', 'deeppink', 'dimgray', 'darkolivegreen', 'darkcyan']
    
    for i, data in enumerate(data_list):
        run_name = data['run_name']
        train_sharpe = data['train_sharpe_ratios']
        test_sharpe = data['test_sharpe_ratios']
        episodes = list(range(1, len(train_sharpe) + 1))
        
        # Use experiment info if available for better labeling
        label_prefix = data.get('experiment_info', run_name)
        
        # Plot training Sharpe ratios
        ax1.plot(episodes, train_sharpe, 'o-', label=f'{label_prefix}', 
                color=colors[i % len(colors)], linewidth=2, markersize=4, alpha=0.8)
        
        # Plot testing Sharpe ratios
        if test_sharpe:
            ax2.plot(episodes, test_sharpe, 's-', label=f'{label_prefix}', 
                    color=colors[i % len(colors)], linewidth=2, markersize=4, alpha=0.8)
    
    # Configure training plot
    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Training Sharpe Ratios - All Experiments')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Configure testing plot
    ax2.set_xlabel('Training Episodes')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Testing Sharpe Ratios - All Experiments')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved to {save_path}")

def load_training_results(results_dir: str) -> Dict:
    """
    Load training results from a JSON file.
    
    Args:
        results_dir: Directory containing training_results.json
        
    Returns:
        Dictionary with training results
    """
    results_file = os.path.join(results_dir, 'optimized_training_results.json')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def load_run_summary(results_dir: str) -> Dict:
    """
    Load run summary from a JSON file.
    
    Args:
        results_dir: Directory containing run_summary.json
        
    Returns:
        Dictionary with run summary
    """
    summary_file = os.path.join(results_dir, 'run_summary.json')
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    return summary

def find_all_runs(base_output_dir: str) -> List[Dict]:
    """
    Find all training runs in the output directory.
    
    Args:
        base_output_dir: Base directory containing all run outputs
        
    Returns:
        List of dictionaries with run information
    """
    runs = []
    
    # Look for directories matching the pattern: YYYYMMDD_rho_X_corruption_Y
    pattern = os.path.join(base_output_dir, "*-*")
    run_dirs = glob.glob(pattern)
    
    for run_dir in run_dirs:
        results_dir = os.path.join(run_dir, "results")
        if os.path.exists(results_dir):
            try:
                # summary = load_run_summary(results_dir)
                results = load_training_results(results_dir)
                
                run_info = {
                    'run_dir': run_dir,
                    'results_dir': results_dir,
                    # 'summary': summary,
                    'results': results
                }
                runs.append(run_info)
                # print(f"Loaded run: {summary['run_id']}")
            except Exception as e:
                print(f"Failed to load run from {run_dir}: {e}")
    
    return runs

def plot_combined_training_curves(runs: List[Dict], save_path: str = None):
    """
    Plot combined training curves for multiple runs.
    
    Args:
        runs: List of run dictionaries
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    
    for i, run in enumerate(runs):
        results = run['results']
        # summary = run['summary']
        
        # Plot training rewards
        rewards = results['training']['rewards']
        episodes = list(range(1, len(rewards) + 1))
        
        # label = f"{summary['run_id']} (ρ={results['config']['rho']}, {results['config']['corruption_type']})"
        label = 'ρ=0.001'
        ax1.plot(episodes, rewards, color=colors[i], alpha=0.7, label=label, linewidth=1)
        
        # Plot Sharpe ratios
        train_sharpe = results['training']['train_sharpe_ratios']
        if train_sharpe:
            ax2.plot(episodes, train_sharpe, color=colors[i], alpha=0.7, label=label, linewidth=1)
    
    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Rewards Comparison')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Training Episodes')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Training Sharpe Ratios Comparison')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined training curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_final_performance_comparison(runs: List[Dict], save_path: str = None):
    """
    Plot final performance comparison for all runs.
    
    Args:
        runs: List of run dictionaries
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data
    run_ids = []
    train_sharpes = []
    test_sharpes = []
    buy_hold_sharpes = []
    rhos = []
    corruption_types = []
    
    for run in runs:
        summary = run['summary']
        results = run['results']
        
        run_ids.append(summary['run_id'])
        train_sharpes.append(summary['final_train_sharpe'])
        test_sharpes.append(summary['final_dqn_test_sharpe'])
        buy_hold_sharpes.append(summary['final_buy_hold_sharpe'])
        rhos.append(results['config']['rho'])
        corruption_types.append(results['config']['corruption_type'])
    
    # Create bar plot for Sharpe ratios
    x = np.arange(len(run_ids))
    width = 0.25
    
    bars1 = ax1.bar(x - width, train_sharpes, width, label='DQN Training', alpha=0.8)
    bars2 = ax1.bar(x, test_sharpes, width, label='DQN Testing', alpha=0.8)
    bars3 = ax1.bar(x + width, buy_hold_sharpes, width, label='Buy & Hold', alpha=0.8)
    
    ax1.set_xlabel('Runs')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Final Sharpe Ratios Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"ρ={r}\n{c}" for r, c in zip(rhos, corruption_types)], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height is not None:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Create scatter plot: Training vs Testing Sharpe
    ax2.scatter(train_sharpes, test_sharpes, c=range(len(run_ids)), cmap='viridis', s=100, alpha=0.7)
    
    # Add labels for each point
    for i, (train_sh, test_sh, rho, corr_type) in enumerate(zip(train_sharpes, test_sharpes, rhos, corruption_types)):
        ax2.annotate(f"ρ={rho}\n{corr_type}", (train_sh, test_sh), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Training Sharpe Ratio')
    ax2.set_ylabel('Testing Sharpe Ratio')
    ax2.set_title('Training vs Testing Sharpe Ratios')
    ax2.grid(True, alpha=0.3)
    
    # Add diagonal line for reference
    min_val = min(min(train_sharpes), min(test_sharpes))
    max_val = max(max(train_sharpes), max(test_sharpes))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Final performance comparison saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_robustness_analysis(runs: List[Dict], save_path: str = None):
    """
    Plot robustness analysis comparing different corruption levels and types.
    
    Args:
        runs: List of run dictionaries
        save_path: Path to save the plot
    """
    # Group runs by corruption type
    corruption_groups = {}
    for run in runs:
        corr_type = run['results']['config']['corruption_type']
        if corr_type not in corruption_groups:
            corruption_groups[corr_type] = []
        corruption_groups[corr_type].append(run)
    
    fig, axes = plt.subplots(1, len(corruption_groups), figsize=(5*len(corruption_groups), 6))
    if len(corruption_groups) == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (corr_type, corr_runs) in enumerate(corruption_groups.items()):
        ax = axes[i]
        
        # Sort runs by rho
        corr_runs.sort(key=lambda x: x['results']['config']['rho'])
        
        rhos = [run['results']['config']['rho'] for run in corr_runs]
        train_sharpes = [run['summary']['final_train_sharpe'] for run in corr_runs]
        test_sharpes = [run['summary']['final_dqn_test_sharpe'] for run in corr_runs]
        buy_hold_sharpes = [run['summary']['final_buy_hold_sharpe'] for run in corr_runs]
        
        ax.plot(rhos, train_sharpes, 'o-', label='DQN Training', color=colors[0], linewidth=2, markersize=6)
        ax.plot(rhos, test_sharpes, 's-', label='DQN Testing', color=colors[1], linewidth=2, markersize=6)
        ax.plot(rhos, buy_hold_sharpes, '^-', label='Buy & Hold', color=colors[2], linewidth=2, markersize=6)
        
        ax.set_xlabel('Corruption Probability (ρ)')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(f'Robustness Analysis: {corr_type.capitalize()} Corruption')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Robustness analysis saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_summary_table(runs: List[Dict], save_path: str = None):
    """
    Create a summary table of all runs.
    
    Args:
        runs: List of run dictionaries
        save_path: Path to save the table
    """
    data = []
    
    for run in runs:
        summary = run['summary']
        results = run['results']
        config = results['config']
        
        data.append({
            'Run ID': summary['run_id'],
            'ρ (Corruption Prob)': config['rho'],
            'Corruption Type': config['corruption_type'],
            'Use Robust Reward': config['use_robust_reward'],
            'Final Train Sharpe': summary['final_train_sharpe'],
            'Final Test Sharpe': summary['final_dqn_test_sharpe'],
            'Buy & Hold Sharpe': summary['final_buy_hold_sharpe'],
            'Train Episodes': config['train_eps'],
            'Learning Rate': config['lr'],
            'Gamma': config['gamma']
        })
    
    df = pd.DataFrame(data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Summary table saved to: {save_path}")
    
    print("\n=== Summary Table ===")
    print(df.to_string(index=False))
    
    return df

def main():
    """
    Main function to load and plot combined results.
    """
    # Base directory containing all run outputs
    base_output_dir = "outputs/optimized_training"
    
    if not os.path.exists(base_output_dir):
        print(f"Base output directory not found: {base_output_dir}")
        return
    
    # Find all runs
    runs = find_all_runs(base_output_dir)
    
    if not runs:
        print("No training runs found!")
        return
    
    print(f"\nFound {len(runs)} training runs")
    
    # Create output directory for combined plots
    combined_output_dir = "combined_analysis"
    os.makedirs(combined_output_dir, exist_ok=True)
    
    # Generate combined plots
    plot_combined_training_curves(runs, os.path.join(combined_output_dir, "combined_training_curves.png"))
    plot_final_performance_comparison(runs, os.path.join(combined_output_dir, "final_performance_comparison.png"))
    plot_robustness_analysis(runs, os.path.join(combined_output_dir, "robustness_analysis.png"))
    create_summary_table(runs, os.path.join(combined_output_dir, "summary_table.csv"))
    
    print(f"\nAll combined analysis saved to: {combined_output_dir}")

if __name__ == "__main__":
    main() 