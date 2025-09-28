#!/usr/bin/env python3
"""
NSE Trading Assistant v3.0 - REAL DATA + LLM INTEGRATION
Professional trading assistant with live market data and AI analysis
Target: rpwarade2@gmail.com | Schedule: Daily 8:00 AM IST

NEW: Real NSE data + GPT-4/Gemini analysis + intelligent API management
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import time
import yaml
import os
import warnings
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Data source imports
import yfinance as yf
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

@dataclass
class StockData:
    """Data structure for stock information"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float]
    historical_data: pd.DataFrame
    technical_indicators: Dict
    news_data: List[Dict]
    timestamp: datetime

@dataclass 
class LLMAnalysis:
    """Data structure for LLM analysis results"""
    symbol: str
    analysis: str
    confidence: float
    score: int
    rationale: str
    risk_factors: List[str]
    opportunities: List[str]
    model_used: str
    timestamp: datetime

class RateLimiter:
    """Smart rate limiter for API calls"""
    def __init__(self):
        self.call_history = {}
        self.circuit_breakers = {}
    
    def can_make_call(self, api_name: str, limit_per_minute: int) -> bool:
        """Check if API call is within rate limits"""
        now = datetime.now()
        minute_key = now.replace(second=0, microsecond=0)
        
        if api_name not in self.call_history:
            self.call_history[api_name] = {}
        
        # Clean old entries
        self.call_history[api_name] = {
            k: v for k, v in self.call_history[api_name].items() 
            if now - k < timedelta(minutes=2)
        }
        
        current_calls = self.call_history[api_name].get(minute_key, 0)
        return current_calls < limit_per_minute
    
    def record_call(self, api_name: str):
        """Record an API call"""
        now = datetime.now()
        minute_key = now.replace(second=0, microsecond=0)
        
        if api_name not in self.call_history:
            self.call_history[api_name] = {}
        
        self.call_history[api_name][minute_key] = self.call_history[api_name].get(minute_key, 0) + 1
    
    def wait_if_needed(self, api_name: str, limit_per_minute: int):
        """Wait if rate limit reached"""
        while not self.can_make_call(api_name, limit_per_minute):
            time.sleep(1)

class DataCache:
    """Simple in-memory cache for API responses"""
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str, ttl_seconds: int) -> Optional[any]:
        """Get cached data if not expired"""
        if key in self.cache:
            if datetime.now() - self.timestamps[key] < timedelta(seconds=ttl_seconds):
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: any):
        """Cache data"""
        self.cache[key] = value
        self.timestamps[key] = datetime.now()

class DataProvider(ABC):
    """Abstract base class for data providers"""
    @abstractmethod
    def get_stock_data(self, symbol: str) -> Optional[StockData]:
        pass
    
    @abstractmethod
    def get_stock_list(self) -> List[str]:
        pass

class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider"""
    def __init__(self, api_key: str, rate_limiter: RateLimiter, cache: DataCache):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limiter = rate_limiter
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    def get_stock_data(self, symbol: str) -> Optional[StockData]:
        """Fetch real stock data from Alpha Vantage"""
        try:
            # Check cache first
            cache_key = f"av_stock_{symbol}"
            cached_data = self.cache.get(cache_key, 300)  # 5 minute cache
            if cached_data:
                return cached_data
            
            # Rate limiting
            self.rate_limiter.wait_if_needed('alpha_vantage', 5)
            
            # Get daily data
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': f'{symbol}.BSE',  # NSE stocks
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            self.rate_limiter.record_call('alpha_vantage')
            
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                dates = list(time_series.keys())
                
                if not dates:
                    return None
                
                # Latest data
                latest_date = dates[0]
                latest_data = time_series[latest_date]
                
                current_price = float(latest_data['4. close'])
                prev_price = float(time_series[dates[1]]['4. close']) if len(dates) > 1 else current_price
                change = current_price - prev_price
                change_percent = (change / prev_price * 100) if prev_price != 0 else 0
                
                # Create historical DataFrame
                historical_data = []
                for date_str in dates[:100]:  # Last 100 days
                    day_data = time_series[date_str]
                    historical_data.append({
                        'Date': pd.to_datetime(date_str),
                        'Open': float(day_data['1. open']),
                        'High': float(day_data['2. high']),
                        'Low': float(day_data['3. low']),
                        'Close': float(day_data['4. close']),
                        'Volume': int(day_data['5. volume'])
                    })
                
                df = pd.DataFrame(historical_data)
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                
                # Calculate technical indicators
                technical_indicators = self.calculate_technical_indicators(df)
                
                stock_data = StockData(
                    symbol=symbol,
                    price=current_price,
                    change=change,
                    change_percent=change_percent,
                    volume=int(latest_data['5. volume']),
                    market_cap=None,  # Not available in this call
                    historical_data=df,
                    technical_indicators=technical_indicators,
                    news_data=[],  # Will be populated separately
                    timestamp=datetime.now()
                )
                
                # Cache the result
                self.cache.set(cache_key, stock_data)
                return stock_data
                
            else:
                self.logger.warning(f"No data available for {symbol} from Alpha Vantage")
                return None
                
        except Exception as e:
            self.logger.error(f"Alpha Vantage error for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators from price data"""
        indicators = {}
        
        try:
            # Moving averages
            indicators['SMA_20'] = df['Close'].rolling(20).mean().iloc[-1]
            indicators['SMA_50'] = df['Close'].rolling(50).mean().iloc[-1]
            indicators['EMA_12'] = df['Close'].ewm(span=12).mean().iloc[-1]
            indicators['EMA_26'] = df['Close'].ewm(span=26).mean().iloc[-1]
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['RSI'] = rsi.iloc[-1]
            
            # MACD
            macd_line = indicators['EMA_12'] - indicators['EMA_26']
            indicators['MACD'] = macd_line
            
            # Volume indicators
            indicators['Volume_SMA'] = df['Volume'].rolling(20).mean().iloc[-1]
            indicators['Volume_Ratio'] = df['Volume'].iloc[-1] / indicators['Volume_SMA']
            
            # Bollinger Bands
            bb_middle = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            indicators['BB_Upper'] = (bb_middle + bb_std * 2).iloc[-1]
            indicators['BB_Lower'] = (bb_middle - bb_std * 2).iloc[-1]
            indicators['BB_Position'] = (df['Close'].iloc[-1] - indicators['BB_Lower']) / (indicators['BB_Upper'] - indicators['BB_Lower'])
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            
        return indicators
    
    def get_stock_list(self) -> List[str]:
        """Get NSE stock universe - limited implementation for Alpha Vantage"""
        # Alpha Vantage doesn't provide a direct stock listing API
        # Return a curated list of major NSE stocks
        return [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ITC', 'SBIN',
            'BHARTIARTL', 'KOTAKBANK', 'LT', 'ASIANPAINT', 'MARUTI', 'AXISBANK',
            'ICICIBANK', 'TITAN', 'NESTLEIND', 'ULTRACEMCO', 'BAJFINANCE',
            'WIPRO', 'ONGC', 'NTPC', 'POWERGRID', 'COALINDIA', 'GRASIM',
            'DRREDDY', 'CIPLA', 'SUNPHARMA', 'DIVISLAB', 'TECHM', 'HCLTECH'
        ]

class TwelveDataProvider(DataProvider):
    """Twelve Data API provider"""
    def __init__(self, api_key: str, rate_limiter: RateLimiter, cache: DataCache):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.rate_limiter = rate_limiter
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    def get_stock_data(self, symbol: str) -> Optional[StockData]:
        """Fetch stock data from Twelve Data"""
        try:
            cache_key = f"td_stock_{symbol}"
            cached_data = self.cache.get(cache_key, 300)
            if cached_data:
                return cached_data
            
            self.rate_limiter.wait_if_needed('twelve_data', 8)
            
            # Get time series data
            params = {
                'symbol': symbol,
                'interval': '1day',
                'outputsize': 100,
                'apikey': self.api_key,
                'exchange': 'NSE'
            }
            
            response = requests.get(f"{self.base_url}/time_series", params=params, timeout=30)
            data = response.json()
            
            self.rate_limiter.record_call('twelve_data')
            
            if 'values' in data and data['values']:
                values = data['values']
                
                # Latest data
                latest = values[0]
                prev = values[1] if len(values) > 1 else latest
                
                current_price = float(latest['close'])
                prev_price = float(prev['close'])
                change = current_price - prev_price
                change_percent = (change / prev_price * 100) if prev_price != 0 else 0
                
                # Create DataFrame
                historical_data = []
                for item in values:
                    historical_data.append({
                        'Date': pd.to_datetime(item['datetime']),
                        'Open': float(item['open']),
                        'High': float(item['high']),
                        'Low': float(item['low']),
                        'Close': float(item['close']),
                        'Volume': int(item['volume']) if item['volume'] else 0
                    })
                
                df = pd.DataFrame(historical_data)
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                
                technical_indicators = self.calculate_technical_indicators(df)
                
                stock_data = StockData(
                    symbol=symbol,
                    price=current_price,
                    change=change,
                    change_percent=change_percent,
                    volume=int(latest['volume']) if latest['volume'] else 0,
                    market_cap=None,
                    historical_data=df,
                    technical_indicators=technical_indicators,
                    news_data=[],
                    timestamp=datetime.now()
                )
                
                self.cache.set(cache_key, stock_data)
                return stock_data
                
        except Exception as e:
            self.logger.error(f"Twelve Data error for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators - same as Alpha Vantage"""
        indicators = {}
        
        try:
            indicators['SMA_20'] = df['Close'].rolling(20).mean().iloc[-1]
            indicators['SMA_50'] = df['Close'].rolling(50).mean().iloc[-1]
            indicators['EMA_12'] = df['Close'].ewm(span=12).mean().iloc[-1]
            indicators['EMA_26'] = df['Close'].ewm(span=26).mean().iloc[-1]
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['RSI'] = rsi.iloc[-1]
            
            # MACD
            indicators['MACD'] = indicators['EMA_12'] - indicators['EMA_26']
            
            # Volume
            indicators['Volume_SMA'] = df['Volume'].rolling(20).mean().iloc[-1]
            indicators['Volume_Ratio'] = df['Volume'].iloc[-1] / indicators['Volume_SMA'] if indicators['Volume_SMA'] > 0 else 1
            
            # Bollinger Bands
            bb_middle = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            indicators['BB_Upper'] = (bb_middle + bb_std * 2).iloc[-1]
            indicators['BB_Lower'] = (bb_middle - bb_std * 2).iloc[-1]
            indicators['BB_Position'] = (df['Close'].iloc[-1] - indicators['BB_Lower']) / (indicators['BB_Upper'] - indicators['BB_Lower'])
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            
        return indicators
    
    def get_stock_list(self) -> List[str]:
        """Get stock list from Twelve Data"""
        try:
            self.rate_limiter.wait_if_needed('twelve_data', 8)
            
            params = {
                'exchange': 'NSE',
                'apikey': self.api_key,
                'type': 'stock'
            }
            
            response = requests.get(f"{self.base_url}/stocks", params=params, timeout=30)
            data = response.json()
            
            self.rate_limiter.record_call('twelve_data')
            
            if 'data' in data:
                return [stock['symbol'] for stock in data['data'][:200]]  # Limit to 200 stocks
            
        except Exception as e:
            self.logger.error(f"Error fetching stock list from Twelve Data: {e}")
        
        # Fallback list
        return AlphaVantageProvider.get_stock_list(self)

class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider"""
    def __init__(self, rate_limiter: RateLimiter, cache: DataCache):
        self.rate_limiter = rate_limiter
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    def get_stock_data(self, symbol: str) -> Optional[StockData]:
        """Fetch stock data from Yahoo Finance"""
        try:
            cache_key = f"yf_stock_{symbol}"
            cached_data = self.cache.get(cache_key, 300)
            if cached_data:
                return cached_data
            
            # Yahoo Finance rate limiting
            self.rate_limiter.wait_if_needed('yfinance', 30)  # 30 calls per minute to be safe
            
            # NSE stocks have .NS suffix
            yf_symbol = f"{symbol}.NS"
            ticker = yf.Ticker(yf_symbol)
            
            # Get historical data
            hist = ticker.history(period="100d")
            if hist.empty:
                return None
            
            # Get current info
            info = ticker.info
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            change = current_price - prev_price
            change_percent = (change / prev_price * 100) if prev_price != 0 else 0
            
            # Technical indicators
            technical_indicators = self.calculate_technical_indicators(hist)
            
            stock_data = StockData(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(hist['Volume'].iloc[-1]),
                market_cap=info.get('marketCap'),
                historical_data=hist,
                technical_indicators=technical_indicators,
                news_data=[],
                timestamp=datetime.now()
            )
            
            self.rate_limiter.record_call('yfinance')
            self.cache.set(cache_key, stock_data)
            return stock_data
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        indicators = {}
        
        try:
            indicators['SMA_20'] = df['Close'].rolling(20).mean().iloc[-1]
            indicators['SMA_50'] = df['Close'].rolling(50).mean().iloc[-1]
            indicators['EMA_12'] = df['Close'].ewm(span=12).mean().iloc[-1]
            indicators['EMA_26'] = df['Close'].ewm(span=26).mean().iloc[-1]
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['RSI'] = rsi.iloc[-1]
            
            # MACD
            indicators['MACD'] = indicators['EMA_12'] - indicators['EMA_26']
            
            # Volume
            indicators['Volume_SMA'] = df['Volume'].rolling(20).mean().iloc[-1]
            indicators['Volume_Ratio'] = df['Volume'].iloc[-1] / indicators['Volume_SMA'] if indicators['Volume_SMA'] > 0 else 1
            
            # Bollinger Bands
            bb_middle = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            indicators['BB_Upper'] = (bb_middle + bb_std * 2).iloc[-1]
            indicators['BB_Lower'] = (bb_middle - bb_std * 2).iloc[-1]
            indicators['BB_Position'] = (df['Close'].iloc[-1] - indicators['BB_Lower']) / (indicators['BB_Upper'] - indicators['BB_Lower'])
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            
        return indicators
    
    def get_stock_list(self) -> List[str]:
        """Get NSE stock list - curated major stocks"""
        return [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ITC', 'SBIN',
            'BHARTIARTL', 'KOTAKBANK', 'LT', 'ASIANPAINT', 'MARUTI', 'AXISBANK',
            'ICICIBANK', 'TITAN', 'NESTLEIND', 'ULTRACEMCO', 'BAJFINANCE',
            'WIPRO', 'ONGC', 'NTPC', 'POWERGRID', 'COALINDIA', 'GRASIM',
            'DRREDDY', 'CIPLA', 'SUNPHARMA', 'DIVISLAB', 'TECHM', 'HCLTECH',
            'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'BAJAJ-AUTO', 'BAJAJFINSV',
            'BRITANNIA', 'EICHERMOT', 'HEROMOTOCO', 'HINDALCO', 'JSWSTEEL',
            'M&M', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'UPL'
        ]

class LLMAnalyzer:
    """LLM-powered stock analysis"""
    def __init__(self, api_config: Dict):
        self.config = api_config
        self.logger = logging.getLogger(__name__)
        self.cache = DataCache()
        
        # Initialize OpenAI
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.openai_available = True
        else:
            self.openai_available = False

        self.logger.error("OpenAI_Available : " + self.openai_available)
            
        # Initialize Google AI
        if GOOGLE_AI_AVAILABLE and os.getenv('GOOGLE_AI_API_KEY'):
            genai.configure(api_key=os.getenv('GOOGLE_AI_API_KEY'))
            self.gemini_model = genai.GenerativeModel('gemini-pro')
            self.google_available = True
        else:
            self.google_available = False
         self.logger.error("Google_AI_Available : " + self.google_available)
    
    def analyze_stock_with_llm(self, stock_data: StockData) -> Optional[LLMAnalysis]:
        """Comprehensive LLM analysis of stock"""
        try:
            # Check cache first
            cache_key = f"llm_analysis_{stock_data.symbol}_{stock_data.timestamp.date()}"
            cached_analysis = self.cache.get(cache_key, 3600)  # 1 hour cache
            if cached_analysis:
                return cached_analysis
            
            # Prepare data for LLM
            analysis_prompt = self._create_analysis_prompt(stock_data)
            
            # Try primary model (GPT-4) first
            analysis = None
            model_used = "none"
            
            if self.openai_available:
                analysis = self._analyze_with_openai(analysis_prompt, stock_data.symbol)
                model_used = "gpt-4"
            
            # Fallback to Gemini if OpenAI fails
            if not analysis and self.google_available:
                analysis = self._analyze_with_gemini(analysis_prompt, stock_data.symbol)
                model_used = "gemini-pro"
            
            if analysis:
                self.cache.set(cache_key, analysis)
                analysis.model_used = model_used
                return analysis
            
            return None
            
        except Exception as e:
            self.logger.error(f"LLM analysis error for {stock_data.symbol}: {e}")
            return None
    
    def _create_analysis_prompt(self, stock_data: StockData) -> str:
        """Create comprehensive analysis prompt for LLM"""
        indicators = stock_data.technical_indicators
        
        prompt = f"""
        As a professional stock analyst, analyze {stock_data.symbol} based on the following data:

        CURRENT DATA:
        - Price: ₹{stock_data.price:.2f}
        - Change: {stock_data.change:+.2f} ({stock_data.change_percent:+.2f}%)
        - Volume: {stock_data.volume:,}
        - Market Cap: {stock_data.market_cap or 'N/A'}

        TECHNICAL INDICATORS:
        - RSI: {indicators.get('RSI', 'N/A'):.1f}
        - SMA 20: ₹{indicators.get('SMA_20', 0):.2f}
        - SMA 50: ₹{indicators.get('SMA_50', 0):.2f}
        - MACD: {indicators.get('MACD', 0):.3f}
        - Volume Ratio: {indicators.get('Volume_Ratio', 0):.2f}x
        - Bollinger Band Position: {indicators.get('BB_Position', 0):.2f}

        PRICE CONTEXT:
        - Current price vs 20-day SMA: {'Above' if stock_data.price > indicators.get('SMA_20', 0) else 'Below'}
        - Current price vs 50-day SMA: {'Above' if stock_data.price > indicators.get('SMA_50', 0) else 'Below'}
        - RSI Status: {'Oversold' if indicators.get('RSI', 50) < 30 else 'Overbought' if indicators.get('RSI', 50) > 70 else 'Neutral'}

        Please provide:
        1. Overall investment recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
        2. Confidence score (0-100)
        3. Detailed technical analysis
        4. Key risk factors (minimum 2)
        5. Key opportunities (minimum 2)
        6. Suggested entry price range
        7. Stop loss recommendation
        8. Target price (short and medium term)
        9. Position sizing suggestion (% of portfolio)

        Format your response as JSON:
        {{
            "recommendation": "Buy/Sell/Hold/etc",
            "confidence": 85,
            "score": 75,
            "analysis": "Detailed analysis here...",
            "risk_factors": ["Risk 1", "Risk 2"],
            "opportunities": ["Opportunity 1", "Opportunity 2"],
            "entry_range": "2400-2450",
            "stop_loss": "2300",
            "target_short": "2600",
            "target_medium": "2800",
            "position_size": "2-3%"
        }}
        """
        
        return prompt
    
    def _analyze_with_openai(self, prompt: str, symbol: str) -> Optional[LLMAnalysis]:
        """Analyze using OpenAI GPT-4"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional stock analyst with expertise in Indian stock markets and NSE trading."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config['llm']['openai']['max_tokens'],
                temperature=self.config['llm']['openai']['temperature'],
                timeout=60
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            analysis_data = json.loads(content)
            
            return LLMAnalysis(
                symbol=symbol,
                analysis=analysis_data.get('analysis', ''),
                confidence=analysis_data.get('confidence', 50) / 100,
                score=analysis_data.get('score', 50),
                rationale=analysis_data.get('analysis', ''),
                risk_factors=analysis_data.get('risk_factors', []),
                opportunities=analysis_data.get('opportunities', []),
                model_used="gpt-4",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI analysis error for {symbol}: {e}")
            return None
    
    def _analyze_with_gemini(self, prompt: str, symbol: str) -> Optional[LLMAnalysis]:
        """Analyze using Google Gemini"""
        try:
            response = self.gemini_model.generate_content(prompt)
            content = response.text.strip()
            
            # Parse JSON response
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            analysis_data = json.loads(content)
            
            return LLMAnalysis(
                symbol=symbol,
                analysis=analysis_data.get('analysis', ''),
                confidence=analysis_data.get('confidence', 50) / 100,
                score=analysis_data.get('score', 50),
                rationale=analysis_data.get('analysis', ''),
                risk_factors=analysis_data.get('risk_factors', []),
                opportunities=analysis_data.get('opportunities', []),
                model_used="gemini-pro",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Gemini analysis error for {symbol}: {e}")
            return None

class NSETradingAssistant:
    """Main NSE Trading Assistant with real data and LLM integration"""
    
    def __init__(self, config_path="config.yaml", api_config_path="api-config.yaml"):
        # Load configurations
        self.config = self.load_config(config_path)
        self.api_config = self.load_config(api_config_path)
        
        # Initialize components
        self.setup_logging()
        self.rate_limiter = RateLimiter()
        self.cache = DataCache()
        
        # Initialize data providers
        self.data_providers = self._initialize_data_providers()
        self.llm_analyzer = LLMAnalyzer(self.api_config)
        
        self.logger.info("🚀 NSE Trading Assistant v3.0 - Real Data + LLM Integration")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        else:
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            'email': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': os.getenv('EMAIL_SENDER', ''),
                'sender_password': os.getenv('EMAIL_PASSWORD', ''),
                'recipient_email': 'rpwarade2@gmail.com'
            },
            'technical_analysis': {
                'score_thresholds': {'strong_buy': 85, 'buy': 70, 'watch': 50}
            }
        }
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('nse_trading_assistant_v3.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_data_providers(self) -> List[DataProvider]:
        """Initialize data providers with failover"""
        providers = []
        
        # Alpha Vantage
        if os.getenv('ALPHAVANTAGE_API_KEY'):
            providers.append(AlphaVantageProvider(
                os.getenv('ALPHAVANTAGE_API_KEY'),
                self.rate_limiter,
                self.cache
            ))
            self.logger.info("✅ Alpha Vantage provider initialized")
        
        # Twelve Data
        if os.getenv('TWELVE_DATA_API_KEY'):
            providers.append(TwelveDataProvider(
                os.getenv('TWELVE_DATA_API_KEY'),
                self.rate_limiter,
                self.cache
            ))
            self.logger.info("✅ Twelve Data provider initialized")
        
        # Yahoo Finance (always available)
        providers.append(YFinanceProvider(self.rate_limiter, self.cache))
        self.logger.info("✅ Yahoo Finance provider initialized")
        
        return providers
    
    def get_stock_data_with_failover(self, symbol: str) -> Optional[StockData]:
        """Get stock data with intelligent failover between providers"""
        for i, provider in enumerate(self.data_providers):
            try:
                provider_name = provider.__class__.__name__
                self.logger.debug(f"Trying {provider_name} for {symbol}")
                
                stock_data = provider.get_stock_data(symbol)
                if stock_data:
                    self.logger.info(f"✅ Got data for {symbol} from {provider_name}")
                    return stock_data
                    
            except Exception as e:
                self.logger.warning(f"⚠️ {provider.__class__.__name__} failed for {symbol}: {e}")
                continue
        
        self.logger.error(f"❌ All providers failed for {symbol}")
        return None
    
    def analyze_stock_comprehensive(self, symbol: str) -> Optional[Dict]:
        """Comprehensive stock analysis combining real data + LLM"""
        try:
            # Get real market data
            stock_data = self.get_stock_data_with_failover(symbol)
            if not stock_data:
                return None
            
            # Get LLM analysis
            llm_analysis = self.llm_analyzer.analyze_stock_with_llm(stock_data)
            
            # Combine traditional + LLM analysis
            combined_analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                
                # Market data
                'current_price': stock_data.price,
                'price_change': stock_data.change,
                'price_change_percent': stock_data.change_percent,
                'volume': stock_data.volume,
                'market_cap': stock_data.market_cap,
                
                # Technical indicators
                'technical_indicators': stock_data.technical_indicators,
                
                # LLM analysis
                'llm_recommendation': llm_analysis.analysis if llm_analysis else "Analysis unavailable",
                'llm_confidence': llm_analysis.confidence if llm_analysis else 0.5,
                'llm_score': llm_analysis.score if llm_analysis else 50,
                'risk_factors': llm_analysis.risk_factors if llm_analysis else [],
                'opportunities': llm_analysis.opportunities if llm_analysis else [],
                'model_used': llm_analysis.model_used if llm_analysis else "none",
                
                # Combined scoring
                'final_score': self._calculate_combined_score(stock_data, llm_analysis),
                'suggested_action': self._determine_action(stock_data, llm_analysis),
                
                # Risk management
                'position_size_pct': self._calculate_position_size(stock_data, llm_analysis),
                'entry_price': stock_data.price,
                'stop_loss': stock_data.price * 0.95,  # 5% stop loss
                'target_1': stock_data.price * 1.08,   # 8% target
                'target_2': stock_data.price * 1.15    # 15% target
            }
            
            return combined_analysis
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return None
    
    def _calculate_combined_score(self, stock_data: StockData, llm_analysis: Optional[LLMAnalysis]) -> int:
        """Calculate combined score from technical + LLM analysis"""
        try:
            # Technical score (50% weight)
            technical_score = 50  # Base score
            indicators = stock_data.technical_indicators
            
            # Trend scoring
            if stock_data.price > indicators.get('SMA_20', 0):
                technical_score += 10
            if indicators.get('SMA_20', 0) > indicators.get('SMA_50', 0):
                technical_score += 10
            
            # RSI scoring
            rsi = indicators.get('RSI', 50)
            if 30 <= rsi <= 70:
                technical_score += 10
            elif rsi < 30:
                technical_score += 5  # Oversold potential
            
            # Volume scoring
            vol_ratio = indicators.get('Volume_Ratio', 1)
            if vol_ratio > 1.5:
                technical_score += 10
            elif vol_ratio > 1.2:
                technical_score += 5
            
            # LLM score (50% weight)
            llm_score = llm_analysis.score if llm_analysis else 50
            
            # Weighted combination
            final_score = int((technical_score * 0.5) + (llm_score * 0.5))
            return max(0, min(100, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating combined score: {e}")
            return 50
    
    def _determine_action(self, stock_data: StockData, llm_analysis: Optional[LLMAnalysis]) -> str:
        """Determine trading action based on combined analysis"""
        final_score = self._calculate_combined_score(stock_data, llm_analysis)
        
        thresholds = self.config.get('technical_analysis', {}).get('score_thresholds', {
            'strong_buy': 85, 'buy': 70, 'watch': 50
        })
        
        if final_score >= thresholds.get('strong_buy', 85):
            return 'Strong Buy'
        elif final_score >= thresholds.get('buy', 70):
            return 'Buy'
        elif final_score >= thresholds.get('watch', 50):
            return 'Watch'
        else:
            return 'Avoid'
    
    def _calculate_position_size(self, stock_data: StockData, llm_analysis: Optional[LLMAnalysis]) -> float:
        """Calculate recommended position size"""
        base_position = 2.0  # 2% base position
        confidence = llm_analysis.confidence if llm_analysis else 0.5
        
        # Adjust based on confidence
        adjusted_position = base_position * (0.5 + confidence)
        
        # Cap at 5% maximum
        return min(5.0, adjusted_position)
    
    def scan_nse_universe(self) -> List[Dict]:
        """Scan NSE universe with real data and LLM analysis"""
        try:
            self.logger.info("🚀 Starting comprehensive NSE universe scan with real data + LLM")
            start_time = datetime.now()
            
            # Get stock universe from primary provider
            stock_universe = []
            for provider in self.data_providers:
                try:
                    stocks = provider.get_stock_list()
                    if stocks:
                        stock_universe = stocks
                        self.logger.info(f"📊 Got {len(stocks)} stocks from {provider.__class__.__name__}")
                        break
                except Exception as e:
                    self.logger.warning(f"Failed to get stock list from {provider.__class__.__name__}: {e}")
                    continue
            
            if not stock_universe:
                self.logger.error("❌ Failed to get stock universe from any provider")
                return []
            
            results = []
            processed = 0
            
            # Process stocks in batches
            batch_size = self.api_config.get('performance', {}).get('batch_size', 20)
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit batch of analysis tasks
                futures = []
                for stock in stock_universe[:50]:  # Limit to 50 stocks for execution time
                    future = executor.submit(self.analyze_stock_comprehensive, stock)
                    futures.append((stock, future))
                
                # Collect results as they complete
                for stock, future in futures:
                    try:
                        result = future.result(timeout=120)  # 2 minute timeout per stock
                        if result and result['final_score'] >= 45:  # Quality threshold
                            results.append(result)
                            processed += 1
                            
                            if processed % 5 == 0:
                                self.logger.info(f"✅ Processed {processed} quality stocks")
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Analysis failed for {stock}: {e}")
                        continue
            
            # Sort by final score
            results.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Limit to top 25 results
            final_results = results[:25]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"✅ Universe scan completed in {execution_time:.1f} seconds")
            self.logger.info(f"📊 Found {len(final_results)} quality opportunities from {processed} stocks")
            
            if final_results:
                top_score = final_results[0]['final_score']
                avg_score = sum(r['final_score'] for r in final_results) / len(final_results)
                self.logger.info(f"🏆 Top score: {top_score} | Average: {avg_score:.1f}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in universe scan: {e}")
            return []
    
    def generate_professional_email(self, results: List[Dict]) -> str:
        """Generate professional HTML email with LLM insights"""
        if not results:
            return "<html><body><h1>No quality opportunities found today</h1></body></html>"
        
        # Statistics
        strong_buy = sum(1 for r in results if r['suggested_action'] == 'Strong Buy')
        buy = sum(1 for r in results if r['suggested_action'] == 'Buy')
        watch = sum(1 for r in results if r['suggested_action'] == 'Watch')
        avg_score = sum(r['final_score'] for r in results) / len(results)
        llm_analyzed = sum(1 for r in results if r['model_used'] != 'none')
        
        css = """
        body { font-family: 'Segoe UI', -apple-system, sans-serif; margin: 0; padding: 20px; background: #f8fafc; }
        .container { max-width: 1100px; margin: 0 auto; background: white; border-radius: 16px; box-shadow: 0 4px 25px rgba(0,0,0,0.08); overflow: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 35px; text-align: center; }
        .header h1 { margin: 0; font-size: 32px; font-weight: 300; }
        .header .subtitle { margin: 15px 0 0 0; font-size: 16px; opacity: 0.9; }
        .ai-badge { background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 20px; font-size: 12px; margin-top: 10px; display: inline-block; }
        
        .summary { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 30px; color: #2d3748; }
        .summary h3 { margin: 0 0 20px 0; font-size: 22px; font-weight: 600; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 20px; margin-top: 20px; }
        .stat { background: rgba(255,255,255,0.8); padding: 20px; border-radius: 12px; text-align: center; backdrop-filter: blur(10px); }
        .stat-number { font-size: 28px; font-weight: bold; color: #2b6cb0; margin-bottom: 5px; }
        .stat-label { font-size: 13px; color: #4a5568; font-weight: 500; }
        
        .table-section { padding: 0; }
        .section-header { background: #f7fafc; padding: 25px; font-size: 20px; font-weight: 600; color: #2d3748; border-left: 5px solid #4299e1; }
        .table-container { overflow-x: auto; }
        .table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .table th { background: #2d3748; color: white; padding: 15px 10px; text-align: left; font-weight: 500; white-space: nowrap; }
        .table td { padding: 12px 10px; border-bottom: 1px solid #e2e8f0; }
        .table tr:nth-child(even) { background: #f8fafc; }
        .table tr:hover { background: #ebf8ff; transition: background 0.2s; }
        
        .action-strong-buy { background: #c6f6d5; color: #22543d; padding: 5px 10px; border-radius: 15px; font-size: 11px; font-weight: 600; }
        .action-buy { background: #bee3f8; color: #2a4365; padding: 5px 10px; border-radius: 15px; font-size: 11px; font-weight: 600; }
        .action-watch { background: #faf089; color: #744210; padding: 5px 10px; border-radius: 15px; font-size: 11px; font-weight: 600; }
        .action-avoid { background: #fed7e2; color: #702459; padding: 5px 10px; border-radius: 15px; font-size: 11px; font-weight: 600; }
        
        .score-high { color: #38a169; font-weight: bold; font-size: 15px; }
        .score-medium { color: #d69e2e; font-weight: bold; font-size: 15px; }
        .score-low { color: #e53e3e; font-weight: bold; font-size: 15px; }
        
        .ai-insights { padding: 30px; background: #f7fafc; }
        .ai-insights h3 { margin: 0 0 25px 0; font-size: 20px; color: #2d3748; }
        .insight-card { margin: 20px 0; padding: 25px; background: white; border-left: 5px solid #4299e1; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
        .insight-header { font-weight: 600; color: #2b6cb0; margin-bottom: 12px; font-size: 16px; }
        .insight-meta { color: #718096; font-size: 13px; margin-bottom: 10px; }
        .insight-analysis { line-height: 1.6; color: #4a5568; margin-bottom: 15px; }
        .risk-opp { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px; }
        .risk-list, .opp-list { padding: 0; margin: 0; }
        .risk-item, .opp-item { padding: 5px 0; font-size: 13px; }
        .risk-item { color: #e53e3e; }
        .opp-item { color: #38a169; }
        
        .disclaimer { background: #fff5d6; border: 1px solid #f6e05e; color: #744210; padding: 25px; margin: 25px; border-radius: 12px; }
        .footer { background: #2d3748; color: #a0aec0; padding: 30px; text-align: center; font-size: 12px; }
        .footer strong { color: #e2e8f0; }
        """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSE Trading Analysis</title>
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 NSE AI Trading Analysis</h1>
            <div class="subtitle">{datetime.now().strftime('%A, %B %d, %Y')} • 8:00 AM IST • Real Market Data</div>
            <div class="ai-badge">🤖 AI-Powered Analysis • {llm_analyzed} stocks analyzed with LLM</div>
        </div>
        
        <div class="summary">
            <h3>📊 Professional Market Intelligence</h3>
            <p>Real-time NSE data analysis powered by advanced AI models • <strong>{len(results)}</strong> premium opportunities identified • Combined Score Average: <strong>{avg_score:.1f}/100</strong></p>
            <div class="stats">
                <div class="stat">
                    <div class="stat-number">{strong_buy}</div>
                    <div class="stat-label">Strong Buy Signals</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{buy}</div>
                    <div class="stat-label">Buy Recommendations</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{watch}</div>
                    <div class="stat-label">Watch List</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{llm_analyzed}</div>
                    <div class="stat-label">AI Analyzed</div>
                </div>
            </div>
        </div>
        
        <div class="table-section">
            <div class="section-header">🚀 Real-Time Trading Opportunities</div>
            <div class="table-container">
                <table class="table">
                    <thead>
                        <tr>
                            <th>Rank</th><th>Stock</th><th>Price</th><th>Change</th><th>Action</th><th>Score</th>
                            <th>RSI</th><th>Volume</th><th>Entry</th><th>Stop Loss</th><th>Target</th><th>Position</th><th>AI Model</th>
                        </tr>
                    </thead>
                    <tbody>"""
        
        # Add top 20 stocks
        for i, stock in enumerate(results[:20], 1):
            action_class = f"action-{stock['suggested_action'].lower().replace(' ', '-')}"
            score_class = 'score-high' if stock['final_score'] >= 75 else ('score-medium' if stock['final_score'] >= 60 else 'score-low')
            
            # Format change with color
            change_color = '#38a169' if stock['price_change'] > 0 else '#e53e3e'
            change_text = f"<span style='color: {change_color};'>{stock['price_change']:+.2f} ({stock['price_change_percent']:+.1f}%)</span>"
            
            html += f"""
                    <tr>
                        <td><strong>#{i}</strong></td>
                        <td><strong>{stock['symbol']}</strong></td>
                        <td>₹{stock['current_price']:.2f}</td>
                        <td>{change_text}</td>
                        <td><span class="{action_class}">{stock['suggested_action']}</span></td>
                        <td><span class="{score_class}">{stock['final_score']}</span></td>
                        <td>{stock['technical_indicators'].get('RSI', 0):.0f}</td>
                        <td>{stock['technical_indicators'].get('Volume_Ratio', 0):.1f}x</td>
                        <td>₹{stock['entry_price']:.2f}</td>
                        <td>₹{stock['stop_loss']:.2f}</td>
                        <td>₹{stock['target_1']:.2f}</td>
                        <td>{stock['position_size_pct']:.1f}%</td>
                        <td><small>{stock['model_used']}</small></td>
                    </tr>"""
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="ai-insights">
            <h3>🤖 AI-Powered Insights - Top 5 Opportunities</h3>"""
        
        # Add detailed AI analysis for top 5
        for i, stock in enumerate(results[:5], 1):
            if stock['llm_recommendation'] and stock['llm_recommendation'] != "Analysis unavailable":
                html += f"""
                <div class="insight-card">
                    <div class="insight-header">#{i}. {stock['symbol']} - {stock['suggested_action']} (Combined Score: {stock['final_score']}/100)</div>
                    <div class="insight-meta">
                        💰 ₹{stock['current_price']:.2f} | 📊 RSI: {stock['technical_indicators'].get('RSI', 0):.0f} | 
                        📈 Change: {stock['price_change']:+.2f}% | 🤖 AI Confidence: {stock['llm_confidence']*100:.0f}% | 
                        Model: {stock['model_used']}
                    </div>
                    <div class="insight-analysis">{stock['llm_recommendation'][:300]}...</div>
                    <div class="risk-opp">
                        <div>
                            <strong style="color: #e53e3e;">⚠️ Key Risks:</strong>
                            <ul class="risk-list">
                                {(''.join([f'<li class="risk-item">{risk}</li>' for risk in stock['risk_factors'][:2]]))}
                            </ul>
                        </div>
                        <div>
                            <strong style="color: #38a169;">✅ Opportunities:</strong>
                            <ul class="opp-list">
                                {(''.join([f'<li class="opp-item">{opp}</li>' for opp in stock['opportunities'][:2]]))}
                            </ul>
                        </div>
                    </div>
                </div>"""
        
        html += f"""
        </div>
        
        <div class="disclaimer">
            <strong>⚠️ IMPORTANT DISCLAIMER:</strong> This analysis uses real NSE market data and advanced AI models for educational purposes only. 
            It does not constitute financial advice. The AI analysis is based on current market data and technical indicators but should not be the sole basis for investment decisions. 
            Always conduct thorough research, consider your risk tolerance, and consult qualified financial advisors before making investment decisions. 
            Trading involves substantial risk of loss. Past performance does not guarantee future results.
        </div>
        
        <div class="footer">
            <p><strong>NSE Trading Assistant v3.0 - Real Data + AI Integration</strong></p>
            <p>Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')} | Target: rpwarade2@gmail.com</p>
            <p>Data Sources: Alpha Vantage, Twelve Data, Yahoo Finance | AI Models: GPT-4, Gemini Pro</p>
            <p>Real-time NSE data with professional AI analysis • Next scan: Tomorrow 8:00 AM IST</p>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def generate_enhanced_csv(self, results: List[Dict]) -> str:
        """Generate comprehensive CSV with all analysis data"""
        try:
            headers = [
                'rank', 'symbol', 'price', 'change', 'change_percent', 'volume', 'market_cap',
                'final_score', 'llm_score', 'suggested_action', 'llm_confidence',
                'rsi', 'sma_20', 'sma_50', 'macd', 'volume_ratio', 'bb_position',
                'entry_price', 'stop_loss', 'target_1', 'target_2', 'position_size_pct',
                'ai_model', 'llm_analysis', 'risk_factors', 'opportunities'
            ]
            
            csv_rows = [headers]
            
            for i, stock in enumerate(results, 1):
                indicators = stock['technical_indicators']
                csv_rows.append([
                    str(i), str(stock['symbol']),
                    f"{stock['current_price']:.2f}", f"{stock['price_change']:.2f}", 
                    f"{stock['price_change_percent']:.2f}%", str(stock['volume']),
                    str(stock['market_cap'] or 'N/A'), str(stock['final_score']),
                    str(stock['llm_score']), str(stock['suggested_action']),
                    f"{stock['llm_confidence']*100:.0f}%", f"{indicators.get('RSI', 0):.1f}",
                    f"{indicators.get('SMA_20', 0):.2f}", f"{indicators.get('SMA_50', 0):.2f}",
                    f"{indicators.get('MACD', 0):.3f}", f"{indicators.get('Volume_Ratio', 0):.2f}",
                    f"{indicators.get('BB_Position', 0):.2f}", f"{stock['entry_price']:.2f}",
                    f"{stock['stop_loss']:.2f}", f"{stock['target_1']:.2f}",
                    f"{stock['target_2']:.2f}", f"{stock['position_size_pct']:.1f}%",
                    str(stock['model_used']), 
                    str(stock['llm_recommendation'][:200].replace(',', ';') if stock['llm_recommendation'] else ''),
                    str(';'.join(stock['risk_factors'])), str(';'.join(stock['opportunities']))
                ])
            
            return '\n'.join([','.join(row) for row in csv_rows])
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced CSV: {e}")
            return "symbol,price,score,action,error\n"
    
    def send_professional_email(self, html_content: str, csv_data: str) -> bool:
        """Send professional email with analysis"""
        try:
            config = self.config['email']
            sender = os.getenv('EMAIL_SENDER') or config.get('sender_email', '')
            password = os.getenv('EMAIL_PASSWORD') or config.get('sender_password', '')
            recipient = config['recipient_email']
            
            if not sender or not password:
                self.logger.error("❌ Email credentials not found")
                return False
            
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = f"NSE AI Analysis — {datetime.now().strftime('%Y-%m-%d')} — Real Data + LLM Insights (8:00 IST)"
            
            msg.attach(MIMEText(html_content, 'html'))
            
            # Enhanced CSV attachment
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(csv_data.encode())
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', f'attachment; filename="nse_ai_analysis_{datetime.now().strftime("%Y%m%d")}.csv"')
            msg.attach(attachment)
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"✅ Professional email sent successfully to {recipient}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Email error: {e}")
            return False
    
    def run_comprehensive_analysis(self):
        """Main execution with real data + LLM integration"""
        try:
            self.logger.info("🚀 Starting NSE Trading Assistant v3.0 - Real Data + AI Analysis")
            start_time = datetime.now()
            
            # Run comprehensive scan
            results = self.scan_nse_universe()
            
            if not results:
                self.logger.warning("⚠️ No quality opportunities found")
                return
            
            # Generate professional content
            html_content = self.generate_professional_email(results)
            csv_data = self.generate_enhanced_csv(results)
            
            # Send professional email
            email_sent = self.send_professional_email(html_content, csv_data)
            
            # Save comprehensive results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(f'nse_ai_analysis_{timestamp}.json', 'w') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            # Execution summary
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"✅ Comprehensive analysis completed in {duration:.1f} seconds")
            self.logger.info(f"📊 {len(results)} quality opportunities with real data + AI insights")
            self.logger.info(f"📧 Professional email: {'✅ Sent' if email_sent else '❌ Failed'}")
            
            # Top recommendations summary
            if results:
                self.logger.info("🏆 Top 5 AI-powered recommendations:")
                for i, stock in enumerate(results[:5], 1):
                    model_info = f" (AI: {stock['model_used']})" if stock['model_used'] != 'none' else ""
                    self.logger.info(f"  {i}. {stock['symbol']}: {stock['suggested_action']} "
                                   f"(Score: {stock['final_score']}/100){model_info}")
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in comprehensive analysis: {e}")
            raise

def main():
    print("🚀 NSE Trading Assistant v3.0")
    print("=" * 70)
    print("✨ REAL MARKET DATA + AI INTEGRATION")
    print("📊 Live NSE data from Alpha Vantage, Twelve Data, Yahoo Finance")
    print("🤖 AI analysis powered by GPT-4 and Gemini Pro")
    print("📧 Professional reports to rpwarade2@gmail.com")
    print("⚡ Intelligent failover and rate limiting")
    print("=" * 70)
    
    try:
        assistant = NSETradingAssistant()
        assistant.run_comprehensive_analysis()
        print("✅ Real data + AI analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
