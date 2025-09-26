import random
import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
except ImportError:
    print("FinRL not properly installed. Please install with: pip install finrl")
    raise


class FinRLTradingEnv(gym.Env):
    """
    A trading environment built on top of FinRL that maintains compatibility
    with the original TradingSystem_v0 interface while leveraging FinRL's robust framework.
    """
    
    def __init__(self, returns_data: Dict[str, List[float]], k_value: int, mode: str, 
                 rho: float = 0.1, corruption_type: str = 'random', 
                 initial_amount: float = 10000.0, transaction_fee_pct: float = 0.001):
        """
        Initialize the FinRL-based trading environment.
        
        Args:
            returns_data: Dictionary mapping ticker symbols to their returns data
            k_value: Number of lookback periods for state representation
            mode: 'train' or 'test' mode
            rho: Probability of reward corruption
            corruption_type: Type of reward corruption ('random', 'flipped', 'scaled')
            initial_amount: Initial portfolio value
            transaction_fee_pct: Transaction fee as percentage
        """
        super(FinRLTradingEnv, self).__init__()
        
        self.mode = mode
        self.returns_data = returns_data
        self.tickers = list(returns_data.keys())
        self.k = k_value
        self.rho = rho
        self.corruption_type = corruption_type
        self.initial_amount = initial_amount
        self.transaction_fee_pct = transaction_fee_pct
        
        # Current episode state
        self.current_stock_idx = 0
        self.current_stock = self.tickers[self.current_stock_idx]
        self.current_returns = np.array(self.returns_data[self.current_stock])
        self.current_step = 0
        self.total_steps = len(self.current_returns) - self.k
        
        # Portfolio state
        self.portfolio_value = initial_amount
        self.cash = initial_amount
        self.shares = 0
        self.current_price = 100.0  # Base price for simulation
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: sell, 1: hold, 2: buy
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(k_value + 3,),  # k returns + cash, shares, portfolio_value
            dtype=np.float32
        )
        
        # Episode state
        self.state = None
        self.reward = 0.0
        self.is_terminal = False
        
        # Reset to initialize state
        self.reset()
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        Returns:
            State array containing k returns + portfolio information
        """
        # Get the k most recent returns
        returns_window = self.current_returns[self.current_step:self.current_step + self.k]
        
        # Normalize portfolio values for better training
        normalized_cash = self.cash / self.initial_amount
        normalized_shares = self.shares / 100  # Normalize by typical position size
        normalized_portfolio = self.portfolio_value / self.initial_amount
        
        # Combine returns and portfolio state
        state = np.concatenate([
            returns_window,
            # [normalized_cash, normalized_shares, normalized_portfolio]
        ], dtype=np.float32)
        
        return state
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate the reward based on action and next return.
        
        Args:
            action: Trading action (0: sell, 1: hold, 2: buy)
            
        Returns:
            Reward value
        """
        if self.current_step + self.k >= len(self.current_returns):
            return 0.0
        
        # Get the next return (the one we're trading on)
        next_return = self.current_returns[self.current_step + self.k]
        
        # Calculate position-based reward
        position = action - 1  # -1: short, 0: neutral, 1: long
        true_reward = position * next_return
        
        # Apply transaction costs if action changes position
        if action != 1:  # Not hold
            transaction_cost = abs(position) * self.transaction_fee_pct
            true_reward -= transaction_cost
        
        # Apply reward corruption with probability rho
        if random.random() < self.rho:
            return self._corrupt_reward(true_reward)
        else:
            return true_reward
    
    def _corrupt_reward(self, true_reward: float) -> float:
        """
        Corrupt the reward based on corruption type.
        
        Args:
            true_reward: The true reward value
            
        Returns:
            Corrupted reward value
        """
        corruption_methods = {
            'random': lambda: random.uniform(-1, 1),
            'flipped': lambda: -3 * true_reward,
            'scaled': lambda: true_reward * random.uniform(0.5, 1.5)
        }
        return corruption_methods.get(self.corruption_type, lambda: true_reward)()
    
    def _update_portfolio(self, action: int):
        """
        Update portfolio based on action.
        
        Args:
            action: Trading action (0: sell, 1: hold, 2: buy)
        """
        if self.current_step + self.k >= len(self.current_returns):
            return
        
        # Get the next return to simulate price movement
        next_return = self.current_returns[self.current_step + self.k]
        price_change = 1 + next_return
        
        # Update current price
        self.current_price *= price_change
        
        # Execute trading action
        if action == 0:  # Sell
            if self.shares > 0:
                sell_value = self.shares * self.current_price * (1 - self.transaction_fee_pct)
                self.cash += sell_value
                self.shares = 0
        elif action == 2:  # Buy
            if self.cash > 0:
                max_shares = self.cash / (self.current_price * (1 + self.transaction_fee_pct))
                self.shares += max_shares
                self.cash = 0
        
        # Update portfolio value
        self.portfolio_value = self.cash + (self.shares * self.current_price)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Trading action (0: sell, 1: hold, 2: buy)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Calculate reward before updating portfolio
        reward = self._calculate_reward(action)
        
        # Update portfolio
        self._update_portfolio(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        self.is_terminal = self.current_step >= self.total_steps
        
        # Update state
        self.state = self._get_state()
        
        # Prepare info dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'shares': self.shares,
            'current_price': self.current_price,
            'current_stock': self.current_stock,
            'step': self.current_step
        }
        
        return self.state, reward, self.is_terminal, info
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial state
        """
        # Select stock for this episode
        if self.mode == 'train':
            self.current_stock = random.choice(self.tickers)
        else:
            self.current_stock = self.tickers[self.current_stock_idx]
            self.current_stock_idx = (self.current_stock_idx + 1) % len(self.tickers)
        
        # Reset episode state
        self.current_returns = np.array(self.returns_data[self.current_stock])
        self.total_steps = len(self.current_returns) - self.k
        self.current_step = 0
        
        # Reset portfolio
        self.portfolio_value = self.initial_amount
        self.cash = self.initial_amount
        self.shares = 0
        self.current_price = 100.0
        
        # Reset episode flags
        self.is_terminal = False
        self.reward = 0.0
        
        # Get initial state
        self.state = self._get_state()
        
        return self.state
    
    def render(self, mode='human'):
        """Render the environment (not implemented for this version)."""
        pass
    
    def close(self):
        """Close the environment."""
        pass


class FinRLTradingEnv_v1(FinRLTradingEnv):
    """
    Enhanced version of FinRLTradingEnv with additional features.
    """
    
    def __init__(self, returns_data: Dict[str, List[float]], k_value: int, mode: str, 
                 rho: float = 0.1, corruption_type: str = 'random', 
                 initial_amount: float = 10000.0, transaction_fee_pct: float = 0.001,
                 use_technical_indicators: bool = True):
        """
        Initialize enhanced FinRL trading environment.
        
        Args:
            use_technical_indicators: Whether to include technical indicators in state
        """
        self.use_technical_indicators = use_technical_indicators
        
        # Adjust observation space if using technical indicators
        if use_technical_indicators:
            # Additional features: moving averages, volatility, momentum
            additional_features = 6  # 2 MAs + volatility + momentum + RSI + MACD
        else:
            additional_features = 0
        
        super().__init__(returns_data, k_value, mode, rho, corruption_type, 
                        initial_amount, transaction_fee_pct)
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(k_value + 3 + additional_features,),
            dtype=np.float32
        )
    
    def _calculate_technical_indicators(self, returns_window: np.ndarray) -> np.ndarray:
        """
        Calculate technical indicators from returns window.
        
        Args:
            returns_window: Window of returns
            
        Returns:
            Array of technical indicators
        """
        if not self.use_technical_indicators:
            return np.array([])
        
        # Convert returns to prices for technical analysis
        prices = np.cumprod(1 + returns_window)
        
        # Simple Moving Averages
        ma_short = np.mean(prices[-5:]) if len(prices) >= 5 else np.mean(prices)
        ma_long = np.mean(prices[-10:]) if len(prices) >= 10 else np.mean(prices)
        
        # Volatility (rolling standard deviation)
        volatility = np.std(returns_window[-10:]) if len(returns_window) >= 10 else np.std(returns_window)
        
        # Momentum (price change over last 5 periods)
        momentum = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
        
        # Simple RSI approximation
        gains = np.sum(returns_window[returns_window > 0])
        losses = -np.sum(returns_window[returns_window < 0])
        rsi = gains / (gains + losses) if (gains + losses) > 0 else 0.5
        
        # MACD approximation (difference between short and long MA)
        macd = ma_short - ma_long
        
        return np.array([ma_short, ma_long, volatility, momentum, rsi, macd], dtype=np.float32)
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation with technical indicators.
        
        Returns:
            Enhanced state array
        """
        # Get base state
        returns_window = self.current_returns[self.current_step:self.current_step + self.k]
        
        # Normalize portfolio values
        normalized_cash = self.cash / self.initial_amount
        normalized_shares = self.shares / 100
        normalized_portfolio = self.portfolio_value / self.initial_amount
        
        # Calculate technical indicators
        technical_indicators = self._calculate_technical_indicators(returns_window)
        
        # Combine all features
        state = np.concatenate([
            returns_window,
            # [normalized_cash, normalized_shares, normalized_portfolio],
            technical_indicators
        ], dtype=np.float32)
        
        return state


# Backward compatibility class
class TradingSystem_v0(FinRLTradingEnv):
    """
    Backward compatibility class that maintains the original interface.
    """
    
    def __init__(self, returns_data, k_value, mode, rho=0.1, corruption_type='random'):
        super().__init__(returns_data, k_value, mode, rho, corruption_type)
    
    def step(self, action):
        """
        Maintain original step interface.
        
        Returns:
            Tuple of (state, reward, done) instead of (state, reward, done, info)
        """
        state, reward, done, info = super().step(action)
        return state, reward, done 