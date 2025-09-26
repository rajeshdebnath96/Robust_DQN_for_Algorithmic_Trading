import yfinance as yf
import pandas as pd
import os
import pickle
import datetime as dt
import random
import time
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """Manages downloading, caching, and loading of financial data."""
    
    def __init__(self, cache_dir: str = "data_cache"):
        """
        Initialize DataManager.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        self._ensure_cache_directory()
    
    def _ensure_cache_directory(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")
    
    def _get_cache_filename(self, ticker: str, start_date: str, end_date: str) -> str:
        """
        Generate cache filename for a ticker and date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Cache filename
        """
        # Clean dates for filename
        start_clean = start_date.replace('-', '').replace('/', '')
        end_clean = end_date.replace('-', '').replace('/', '')
        return f"{ticker}_{start_clean}_{end_clean}.pkl"
    
    def _get_cache_path(self, ticker: str, start_date: str, end_date: str) -> str:
        """
        Get full cache path for a ticker and date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Full cache file path
        """
        filename = self._get_cache_filename(ticker, start_date, end_date)
        return os.path.join(self.cache_dir, filename)
    
    def _is_cache_valid(self, cache_path: str, max_age_days: int = 7) -> bool:
        """
        Check if cached data is still valid (not too old).
        
        Args:
            cache_path: Path to cache file
            max_age_days: Maximum age of cache in days
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not os.path.exists(cache_path):
            return False
        
        # Check file modification time
        file_time = dt.datetime.fromtimestamp(os.path.getmtime(cache_path))
        current_time = dt.datetime.now()
        age_days = (current_time - file_time).days
        
        return age_days <= max_age_days
    
    def download_and_cache_data(self, 
                               ticker: str, 
                               start_date: str, 
                               end_date: str,
                               force_download: bool = False,
                               max_cache_age_days: int = 7) -> Optional[pd.DataFrame]:
        """
        Download data for a ticker and cache it, or load from cache if available.
        
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
        
        # Try to load from cache first (unless force_download is True)
        if not force_download and self._is_cache_valid(cache_path, max_cache_age_days):
            try:
                data = self.load_from_cache(cache_path)
                if data is not None and not data.empty:
                    logger.info(f"Loaded {ticker} data from cache: {data.shape}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load {ticker} from cache: {e}")
        
        # Download data from yfinance
        try:
            logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            
            if data.empty:
                logger.warning(f"No data received for {ticker}")
                return None
            
            logger.info(f"Downloaded {ticker} data: {data.shape}")
            
            # Cache the downloaded data
            self.save_to_cache(data, cache_path)
            logger.info(f"Cached {ticker} data to {cache_path}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to download data for {ticker}: {e}")
            
            # Try to load from cache as fallback (even if old)
            if os.path.exists(cache_path):
                try:
                    logger.info(f"Attempting to load {ticker} from old cache as fallback")
                    data = self.load_from_cache(cache_path)
                    if data is not None and not data.empty:
                        logger.info(f"Successfully loaded {ticker} from old cache: {data.shape}")
                        return data
                except Exception as cache_error:
                    logger.error(f"Failed to load {ticker} from cache: {cache_error}")
            
            return None
    
    def save_to_cache(self, data: pd.DataFrame, cache_path: str):
        """
        Save data to cache file.
        
        Args:
            data: DataFrame to cache
            cache_path: Path where to save the cache
        """
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}: {e}")
    
    def load_from_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache file.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            DataFrame from cache or None if failed
        """
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_path}: {e}")
            return None
    
    def get_returns_data(self, 
                        ticker: str, 
                        start_date: str, 
                        end_date: str,
                        force_download: bool = False,
                        max_cache_age_days: int = 7) -> Optional[List[float]]:
        """
        Get returns data for a ticker (with caching).
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date string
            end_date: End date string
            force_download: Force download even if cache exists
            max_cache_age_days: Maximum age of cache in days
            
        Returns:
            List of returns or None if failed
        """
        data = self.download_and_cache_data(ticker, start_date, end_date, 
                                          force_download, max_cache_age_days)
        
        if data is None or data.empty:
            return None
        
        # Extract returns from Adj Close or Close
        if 'Adj Close' in data.columns:
            returns = data['Adj Close'].pct_change().dropna().values.flatten().tolist()
        elif 'Close' in data.columns:
            returns = data['Close'].pct_change().dropna().values.flatten().tolist()
        else:
            logger.error(f"No price data found for {ticker}")
            return None
        
        return returns
    
    def batch_download_data(self, 
                           tickers: List[str], 
                           start_date: str, 
                           end_date: str,
                           force_download: bool = False,
                           max_cache_age_days: int = 7,
                           delay_between_requests: float = 1.0) -> Dict[str, List[float]]:
        """
        Download data for multiple tickers with caching and rate limiting.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date string
            end_date: End date string
            force_download: Force download even if cache exists
            max_cache_age_days: Maximum age of cache in days
            delay_between_requests: Delay between requests in seconds
            
        Returns:
            Dictionary mapping tickers to their returns data
        """
        results = {}
        
        for i, ticker in enumerate(tickers):
            logger.info(f"Processing {ticker}... ({i+1}/{len(tickers)})")
            
            # Add delay between requests to avoid rate limiting
            if i > 0:
                delay = delay_between_requests + random.uniform(0, 0.5)  # Add some randomness
                logger.info(f"Waiting {delay:.1f} seconds before next request...")
                time.sleep(delay)
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    returns = self.get_returns_data(ticker, start_date, end_date, 
                                                  force_download, max_cache_age_days)
                    
                    if returns is not None:
                        results[ticker] = returns
                        logger.info(f"Successfully processed {ticker}: {len(returns)} data points")
                        break
                    else:
                        logger.warning(f"Failed to get data for {ticker} (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    logger.error(f"Error processing {ticker} (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"Failed to process {ticker} after {max_retries} attempts")
        
        return results
    
    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear cache files.
        
        Args:
            ticker: Specific ticker to clear cache for, or None to clear all
        """
        if ticker:
            # Clear cache for specific ticker
            pattern = f"{ticker}_*.pkl"
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(f"{ticker}_"):
                    filepath = os.path.join(self.cache_dir, filename)
                    os.remove(filepath)
                    logger.info(f"Removed cache file: {filepath}")
        else:
            # Clear all cache
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(self.cache_dir, filename)
                    os.remove(filepath)
                    logger.info(f"Removed cache file: {filepath}")
    
    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        
        info = {
            'total_files': len(cache_files),
            'cache_size_mb': sum(os.path.getsize(os.path.join(self.cache_dir, f)) 
                               for f in cache_files) / (1024 * 1024)
        }
        
        return info


# Convenience functions for easy usage
def download_ticker_data(ticker: str, start_date: str, end_date: str, 
                        cache_dir: str = "data_cache") -> Optional[List[float]]:
    """
    Convenience function to download and cache data for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date string
        end_date: End date string
        cache_dir: Directory for caching
        
    Returns:
        List of returns or None if failed
    """
    dm = DataManager(cache_dir)
    return dm.get_returns_data(ticker, start_date, end_date)


def download_multiple_tickers(tickers: List[str], start_date: str, end_date: str,
                            cache_dir: str = "data_cache") -> Dict[str, List[float]]:
    """
    Convenience function to download and cache data for multiple tickers.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date string
        end_date: End date string
        cache_dir: Directory for caching
        
    Returns:
        Dictionary mapping tickers to their returns data
    """
    dm = DataManager(cache_dir)
    return dm.batch_download_data(tickers, start_date, end_date) 