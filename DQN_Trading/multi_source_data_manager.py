#!/usr/bin/env python3
"""
Multi-source data manager that can use different APIs and fall back to synthetic data
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import pickle
import datetime as dt
import time
import random
import requests
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiSourceDataManager:
    """Manages downloading data from multiple sources with fallback options."""
    
    def __init__(self, cache_dir: str = "data_cache"):
        """
        Initialize MultiSourceDataManager.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        self._ensure_cache_directory()
        
        # API keys (you can add your own keys here)
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY', 'demo')
        self.polygon_key = os.getenv('POLYGON_KEY', 'demo')
        
        # Rate limiting settings
        self.yfinance_delay = 2.0
        self.alpha_vantage_delay = 12.0  # 5 requests per minute
        self.polygon_delay = 12.0  # 5 requests per minute
    
    def _ensure_cache_directory(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")
    
    def _get_cache_filename(self, ticker: str, start_date: str, end_date: str) -> str:
        """Generate cache filename for a ticker and date range."""
        start_clean = start_date.replace('-', '').replace('/', '')
        end_clean = end_date.replace('-', '').replace('/', '')
        return f"{ticker}_{start_clean}_{end_clean}.pkl"
    
    def _get_cache_path(self, ticker: str, start_date: str, end_date: str) -> str:
        """Get full cache path for a ticker and date range."""
        filename = self._get_cache_filename(ticker, start_date, end_date)
        return os.path.join(self.cache_dir, filename)
    
    def _is_cache_valid(self, cache_path: str, max_age_days: int = 7) -> bool:
        """Check if cached data is still valid."""
        if not os.path.exists(cache_path):
            return False
        
        file_time = dt.datetime.fromtimestamp(os.path.getmtime(cache_path))
        current_time = dt.datetime.now()
        age_days = (current_time - file_time).days
        
        return age_days <= max_age_days
    
    def save_to_cache(self, data: pd.DataFrame, cache_path: str):
        """Save data to cache file."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}: {e}")
    
    def load_from_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """Load data from cache file."""
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_path}: {e}")
            return None
    
    def download_from_yfinance(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Download data from Yahoo Finance."""
        try:
            logger.info(f"Downloading {ticker} from Yahoo Finance...")
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
            
            if data is None or data.empty:
                logger.warning(f"No data received for {ticker} from Yahoo Finance")
                return None
            
            logger.info(f"Downloaded {ticker} from Yahoo Finance: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to download {ticker} from Yahoo Finance: {e}")
            return None
    
    def download_from_alpha_vantage(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Download data from Alpha Vantage."""
        try:
            logger.info(f"Downloading {ticker} from Alpha Vantage...")
            
            # Alpha Vantage daily adjusted endpoint
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                return None
            
            if 'Time Series (Daily)' not in data:
                logger.error(f"No time series data for {ticker} from Alpha Vantage")
                return None
            
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df_data = []
            
            for date, values in time_series.items():
                if start_date <= date <= end_date:
                    df_data.append({
                        'Date': date,
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Adj Close': float(values['5. adjusted close']),
                        'Volume': int(values['6. volume'])
                    })
            
            if not df_data:
                logger.warning(f"No data in date range for {ticker} from Alpha Vantage")
                return None
            
            df = pd.DataFrame(df_data)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"Downloaded {ticker} from Alpha Vantage: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download {ticker} from Alpha Vantage: {e}")
            return None
    
    def generate_synthetic_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic stock data for testing."""
        logger.info(f"Generating synthetic data for {ticker}...")
        
        # Parse dates
        start = dt.datetime.strptime(start_date, '%Y-%m-%d')
        end = dt.datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate business days
        business_days = pd.bdate_range(start=start, end=end)
        n_days = len(business_days)
        
        # Generate synthetic price data
        np.random.seed(hash(ticker) % 2**32)
        
        # Different characteristics for different tickers
        ticker_params = {
            'AAPL': {'volatility': 0.025, 'trend': 0.0002, 'start_price': 100},
            'MSFT': {'volatility': 0.022, 'trend': 0.0003, 'start_price': 80},
            'GOOGL': {'volatility': 0.028, 'trend': 0.0001, 'start_price': 120},
            'AMZN': {'volatility': 0.032, 'trend': 0.0004, 'start_price': 90},
            'TSLA': {'volatility': 0.045, 'trend': 0.0005, 'start_price': 200},
            'NVDA': {'volatility': 0.035, 'trend': 0.0006, 'start_price': 150},
            'META': {'volatility': 0.030, 'trend': 0.0002, 'start_price': 180},
            'NFLX': {'volatility': 0.040, 'trend': 0.0003, 'start_price': 300},
            'CRM': {'volatility': 0.026, 'trend': 0.0001, 'start_price': 160},
            'WMT': {'volatility': 0.018, 'trend': 0.0001, 'start_price': 70},
        }
        
        params = ticker_params.get(ticker, {'volatility': 0.025, 'trend': 0.0002, 'start_price': 100})
        
        # Generate price series
        returns = np.random.normal(params['trend'], params['volatility'], n_days)
        returns = np.clip(returns, -0.2, 0.2)  # Max 20% daily move
        
        # Add some autocorrelation
        for i in range(1, n_days):
            returns[i] += 0.1 * returns[i-1]
        
        # Convert to prices
        prices = [params['start_price']]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Adj Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=business_days)
        
        # Ensure High >= Low
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        logger.info(f"Generated synthetic data for {ticker}: {df.shape}")
        return df
    
    def get_stock_data(self, ticker: str, start_date: str, end_date: str, 
                      force_download: bool = False, max_cache_age_days: int = 7) -> Optional[pd.DataFrame]:
        """
        Get stock data from multiple sources with fallback.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date string
            end_date: End date string
            force_download: Force download even if cache exists
            max_cache_age_days: Maximum age of cache in days
            
        Returns:
            DataFrame with stock data or None if failed
        """
        cache_path = self._get_cache_path(ticker, start_date, end_date)
        
        # Try to load from cache first
        if not force_download and self._is_cache_valid(cache_path, max_cache_age_days):
            try:
                data = self.load_from_cache(cache_path)
                if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                    logger.info(f"Loaded {ticker} data from cache: {data.shape}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load {ticker} from cache: {e}")
        
        # Try different data sources in order
        data_sources = [
            ('Yahoo Finance', self.download_from_yfinance),
            ('Alpha Vantage', self.download_from_alpha_vantage),
            ('Synthetic Data', self.generate_synthetic_data)
        ]
        
        for source_name, download_func in data_sources:
            try:
                logger.info(f"Trying {source_name} for {ticker}...")
                data = download_func(ticker, start_date, end_date)
                
                if data is not None and not data.empty:
                    # Cache the data
                    self.save_to_cache(data, cache_path)
                    logger.info(f"Successfully downloaded {ticker} from {source_name}: {data.shape}")
                    return data
                
                # Add delay between sources
                if source_name != 'Synthetic Data':
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Failed to get {ticker} from {source_name}: {e}")
                continue
        
        logger.error(f"Failed to get data for {ticker} from all sources")
        return None
    
    def get_returns_data(self, ticker: str, start_date: str, end_date: str,
                        force_download: bool = False, max_cache_age_days: int = 7) -> Optional[List[float]]:
        """
        Get returns data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date string
            end_date: End date string
            force_download: Force download even if cache exists
            max_cache_age_days: Maximum age of cache in days
            
        Returns:
            List of returns or None if failed
        """
        data = self.get_stock_data(ticker, start_date, end_date, force_download, max_cache_age_days)
        
        if data is None or data.empty:
            return None
        
        # Extract returns from Adj Close or Close
        if 'Adj Close' in data.columns:
            returns = data['Adj Close'].pct_change().dropna().values.tolist()
        elif 'Close' in data.columns:
            returns = data['Close'].pct_change().dropna().values.tolist()
        else:
            logger.error(f"No price data found for {ticker}")
            return None
        
        return returns
    
    def batch_download_data(self, tickers: List[str], start_date: str, end_date: str,
                           force_download: bool = False, max_cache_age_days: int = 7) -> Dict[str, List[float]]:
        """
        Download data for multiple tickers with multiple sources.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date string
            end_date: End date string
            force_download: Force download even if cache exists
            max_cache_age_days: Maximum age of cache in days
            
        Returns:
            Dictionary mapping tickers to their returns data
        """
        results = {}
        
        for i, ticker in enumerate(tickers):
            logger.info(f"Processing {ticker}... ({i+1}/{len(tickers)})")
            
            # Add delay between requests
            if i > 0:
                delay = 2.0 + random.uniform(0, 1.0)
                logger.info(f"Waiting {delay:.1f} seconds before next request...")
                time.sleep(delay)
            
            returns = self.get_returns_data(ticker, start_date, end_date, force_download, max_cache_age_days)
            
            if returns is not None:
                results[ticker] = returns
                logger.info(f"Successfully processed {ticker}: {len(returns)} data points")
            else:
                logger.warning(f"Failed to get data for {ticker}")
        
        return results

# Convenience function
def download_data_with_multiple_sources(tickers: List[str], start_date: str, end_date: str, 
                                      cache_dir: str = "data_cache") -> Dict[str, List[float]]:
    """
    Convenience function to download data using multiple sources.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date string
        end_date: End date string
        cache_dir: Directory for caching
        
    Returns:
        Dictionary mapping tickers to their returns data
    """
    dm = MultiSourceDataManager(cache_dir)
    return dm.batch_download_data(tickers, start_date, end_date) 