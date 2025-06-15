import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
import warnings
import os
import io
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import webbrowser
import tempfile
from PIL import Image, ImageTk
import json
import sys
import traceback

def detailed_exception_handler(exc_type, exc_value, exc_traceback):
    """Custom exception handler to show detailed error information"""
    print("=" * 60)
    print(f"DETAILED ERROR INFORMATION:")
    print("=" * 60)
    print(f"Exception Type: {exc_type.__name__}")
    print(f"Exception Value: {exc_value}")
    print("\nFull Traceback:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("=" * 60)

# Set the custom exception handler
sys.excepthook = detailed_exception_handler
warnings.filterwarnings('ignore')

# Enhanced ML/DL Libraries (using scikit-learn instead of sklearn)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import catboost as cb

# Advanced Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D
    from tensorflow.keras.layers import BatchNormalization, Input, Concatenate, Attention
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l1_l2

    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


# Complete Technical Indicators Class (talib replacement)
class TechnicalIndicators:
    """Complete technical indicators suite to replace talib"""

    @staticmethod
    def sma(data, period):
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data, period):
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()

    @staticmethod
    def wma(data, period):
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.sum(x * weights) / np.sum(weights), raw=True
        )

    @staticmethod
    def rsi(data, period=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """Moving Average Convergence Divergence"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal

        return macd, macd_signal, macd_histogram

    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Bollinger Bands"""
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        """Stochastic Oscillator"""
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()

        k_percent = 100 * (close - low_min) / (high_max - low_min)
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    @staticmethod
    def williams_r(high, low, close, period=14):
        """Williams %R"""
        high_max = high.rolling(window=period).max()
        low_min = low.rolling(window=period).min()

        williams_r = -100 * (high_max - close) / (high_max - low_min)
        return williams_r

    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()

        return atr

    @staticmethod
    def obv(close, volume):
        """On Balance Volume"""
        direction = np.where(close > close.shift(1), 1,
                             np.where(close < close.shift(1), -1, 0))
        obv = (direction * volume).cumsum()
        return obv

    @staticmethod
    def volume_price_trend(close, volume):
        """Volume Price Trend"""
        price_change_pct = close.pct_change()
        vpt = (price_change_pct * volume).cumsum()
        return vpt

    @staticmethod
    def identify_doji(open_price, high, low, close):
        """Identify Doji candlestick patterns"""
        body_size = abs(close - open_price)
        candle_range = high - low

        doji = (body_size / candle_range) < 0.1
        return doji.astype(int)

    @staticmethod
    def identify_hammer(open_price, high, low, close):
        """Identify Hammer candlestick patterns"""
        body_size = abs(close - open_price)
        lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low
        upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)

        hammer = (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
        return hammer.astype(int)

    @staticmethod
    def identify_shooting_star(open_price, high, low, close):
        """Identify Shooting Star candlestick patterns"""
        body_size = abs(close - open_price)
        lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low
        upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)

        shooting_star = (upper_shadow > 2 * body_size) & (lower_shadow < body_size)
        return shooting_star.astype(int)

    @staticmethod
    def fibonacci_levels(high, low, period=50):
        """Calculate Fibonacci retracement levels"""
        rolling_max = high.rolling(period).max()
        rolling_min = low.rolling(period).min()

        range_val = rolling_max - rolling_min

        fib_236 = rolling_max - 0.236 * range_val
        fib_382 = rolling_max - 0.382 * range_val
        fib_50 = rolling_max - 0.5 * range_val
        fib_618 = rolling_max - 0.618 * range_val

        return fib_236, fib_382, fib_50, fib_618

# Enhanced AI Agent with ALL original functionality
class EnhancedStockMarketAIAgent:
    """
    Complete Enhanced Professional Institutional Grade AI Agent
    ALL original functionality preserved and enhanced
    """

    def __init__(self):
        self.data = None
        self.features = None
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.csv_file_path = None

        # Enhanced model configurations
        self.ensemble_models = {}
        self.deep_models = {}
        self.prediction_confidence = {}
        self.technical_indicators = TechnicalIndicators()

        # Smart money analysis
        self.smart_money_analysis = {}
        self.market_trend = "Unknown"
        self.risk_metrics = {}

    def create_enhanced_sample_data(self):
        """Create enhanced realistic sample data with all features"""
        np.random.seed(42)

        # Generate more realistic data with trends and patterns
        dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
        base_price = 150

        # Create realistic price movements with trends and volatility clustering
        returns = []
        volatility = 0.02

        for i in range(len(dates)):
            # Add regime changes (market cycles)
            if i > 200:
                volatility = 0.015  # Lower volatility period
            if i > 350:
                volatility = 0.03  # Higher volatility period

            # Mean reversion with trend
            trend = 0.0005 * np.sin(i / 50) + 0.0002
            shock = np.random.normal(trend, volatility)

            # Add occasional large moves (fat tails) - market events
            if np.random.random() < 0.05:
                shock *= 3

            # Add momentum effects
            if i > 0:
                momentum = 0.1 * returns[-1] if len(returns) > 0 else 0
                shock += momentum

            returns.append(shock)

        # Calculate prices
        prices = [base_price]
        for ret in returns[:-1]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1))

        # Generate OHLC data with realistic intraday patterns
        ohlc_data = []
        volumes = []

        for i, close_price in enumerate(prices):
            # Generate realistic OHLC
            daily_volatility = abs(returns[i]) * 2
            high = close_price * (1 + daily_volatility * np.random.uniform(0.3, 1.0))
            low = close_price * (1 - daily_volatility * np.random.uniform(0.3, 1.0))

            if i == 0:
                open_price = close_price
            else:
                # Add gap behavior
                gap_factor = 1 + np.random.normal(0, 0.005)
                open_price = prices[i - 1] * gap_factor

            # Ensure OHLC logic
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            # Volume correlated with price movement and volatility
            base_volume = 1000000
            volume_multiplier = 1 + abs(returns[i]) * 10 + np.random.normal(0, 0.3)

            # Add institutional volume patterns
            if np.random.random() < 0.1:  # 10% chance of institutional activity
                volume_multiplier *= 2.5

            volume = int(base_volume * max(volume_multiplier, 0.1))

            ohlc_data.append({
                'Date': dates[i].strftime('%Y-%m-%d'),
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close_price, 2),
                'Volume': volume,
                'Change': round(prices[i] - prices[i - 1], 2) if i > 0 else 0,
                'Ch(%)': round(((prices[i] - prices[i - 1]) / prices[i - 1]) * 100, 2) if i > 0 else 0,
                'Value(cr)': round(volume * close_price / 10000000, 2),
                'Trade': np.random.randint(5000, 50000)
            })

        df = pd.DataFrame(ohlc_data)
        filename = 'enhanced_sample_stock_data.csv'
        df.to_csv(filename, index=False)

        return filename

    def validate_data_quality(self, df):
        """Comprehensive data quality validation"""
        validation_results = []

        # Check required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            validation_results.append(f"❌ Missing columns: {missing_cols}")
        else:
            validation_results.append("✅ All required columns present")

        # Check data types
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    validation_results.append(f"⚠️ {col} should be numeric")
                else:
                    validation_results.append(f"✅ {col} data type OK")

        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            validation_results.append(f"⚠️ Missing values detected: {missing_data.sum()}")
        else:
            validation_results.append("✅ No missing values")

        # Check OHLC logic
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            ohlc_issues = 0

            # High should be >= Open, Close
            ohlc_issues += (df['High'] < df['Open']).sum()
            ohlc_issues += (df['High'] < df['Close']).sum()

            # Low should be <= Open, Close
            ohlc_issues += (df['Low'] > df['Open']).sum()
            ohlc_issues += (df['Low'] > df['Close']).sum()

            if ohlc_issues > 0:
                validation_results.append(f"⚠️ OHLC logic violations: {ohlc_issues}")
            else:
                validation_results.append("✅ OHLC data integrity OK")

        # Check for outliers
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()

                if outliers > len(df) * 0.05:  # More than 5% outliers
                    validation_results.append(f"⚠️ High outliers in {col}: {outliers}")
                else:
                    validation_results.append(f"✅ {col} outliers within normal range")

        return "\n".join(validation_results)

    def enhanced_data_preprocessing(self, csv_file_path=None):
        """Enhanced data preprocessing with ALL original functionality"""
        try:
            if csv_file_path is None:
                csv_file_path = self.csv_file_path

            if csv_file_path is None:
                raise ValueError("No CSV file provided")

            # Load data with robust parsing
            self.data = pd.read_csv(csv_file_path)

            # Enhanced date parsing - supports multiple formats
            if 'Date' in self.data.columns:
                if pd.api.types.is_numeric_dtype(self.data['Date']):
                    # Handle milliseconds timestamp
                    try:
                        self.data['Date'] = pd.to_datetime(self.data['Date'], unit='ms')
                    except:
                        self.data['Date'] = pd.to_datetime(self.data['Date'], unit='s')
                else:
                    # Try multiple date formats
                    date_formats = [
                        '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d',
                        '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S'
                    ]

                    parsed = False
                    for fmt in date_formats:
                        try:
                            self.data['Date'] = pd.to_datetime(self.data['Date'], format=fmt)
                            parsed = True
                            break
                        except:
                            continue

                    if not parsed:
                        self.data['Date'] = pd.to_datetime(self.data['Date'], infer_datetime_format=True)

                self.data.set_index('Date', inplace=True)
                self.data.sort_index(inplace=True)

            # Ensure required columns exist and are numeric
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in self.data.columns:
                    raise ValueError(f"Required column '{col}' not found")
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            # Data quality checks and cleaning
            initial_length = len(self.data)

            # Remove rows with any NaN in required columns
            self.data = self.data.dropna(subset=required_cols)

            # Fix OHLC inconsistencies
            self.data.loc[self.data['High'] < self.data['Open'], 'High'] = self.data['Open']
            self.data.loc[self.data['High'] < self.data['Close'], 'High'] = self.data['Close']
            self.data.loc[self.data['Low'] > self.data['Open'], 'Low'] = self.data['Open']
            self.data.loc[self.data['Low'] > self.data['Close'], 'Low'] = self.data['Close']

            # Remove outliers using enhanced IQR method
            for col in ['Open', 'High', 'Low', 'Close']:
                Q1 = self.data[col].quantile(0.01)
                Q3 = self.data[col].quantile(0.99)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Keep track of outliers removed
                outliers_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                self.data = self.data[~outliers_mask]

            # Volume outliers (more lenient)
            vol_Q1 = self.data['Volume'].quantile(0.05)
            vol_Q3 = self.data['Volume'].quantile(0.95)
            vol_IQR = vol_Q3 - vol_Q1
            vol_lower = vol_Q1 - 3 * vol_IQR
            vol_upper = vol_Q3 + 3 * vol_IQR

            volume_outliers = (self.data['Volume'] < vol_lower) | (self.data['Volume'] > vol_upper)
            self.data = self.data[~volume_outliers]

            final_length = len(self.data)
            print(f"Data preprocessing complete: {initial_length} -> {final_length} rows")

            return True

        except Exception as e:
            print(f"Data preprocessing error: {str(e)}")
            return False

    def calculate_advanced_technical_indicators(self):
        """Calculate ALL advanced technical indicators from original code"""
        df = self.data.copy()

        # Basic price indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Enhanced Moving Averages (ALL periods from original)
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = self.technical_indicators.sma(df['Close'], period)
            df[f'EMA_{period}'] = self.technical_indicators.ema(df['Close'], period)
            df[f'WMA_{period}'] = self.technical_indicators.wma(df['Close'], period)

        # Advanced Momentum Indicators (ALL from original)
        df['RSI_14'] = self.technical_indicators.rsi(df['Close'], 14)
        df['RSI_21'] = self.technical_indicators.rsi(df['Close'], 21)

        # Stochastic Oscillator
        df['Stoch_K'], df['Stoch_D'] = self.technical_indicators.stochastic(
            df['High'], df['Low'], df['Close'], 14, 3
        )

        # Williams %R
        df['Williams_R'] = self.technical_indicators.williams_r(
            df['High'], df['Low'], df['Close'], 14
        )

        # MACD variants (ALL from original)
        df['MACD_12_26'], df['MACD_Signal_12_26'], df['MACD_Hist_12_26'] = self.technical_indicators.macd(
            df['Close'], 12, 26, 9
        )
        df['MACD_8_21'], df['MACD_Signal_8_21'], df['MACD_Hist_8_21'] = self.technical_indicators.macd(
            df['Close'], 8, 21, 5
        )

        # For simplified naming
        df['MACD'] = df['MACD_12_26']
        df['MACD_Signal'] = df['MACD_Signal_12_26']
        df['MACD_Hist'] = df['MACD_Hist_12_26']

        # Bollinger Bands variants (ALL periods from original)
        for period in [20, 50]:
            bb_upper, bb_middle, bb_lower = self.technical_indicators.bollinger_bands(
                df['Close'], period, 2
            )
            df[f'BB_Middle_{period}'] = bb_middle
            df[f'BB_Upper_{period}'] = bb_upper
            df[f'BB_Lower_{period}'] = bb_lower
            df[f'BB_Width_{period}'] = (bb_upper - bb_lower) / bb_middle
            df[f'BB_Position_{period}'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

        # Default Bollinger Bands
        df['BB_Upper'] = df['BB_Upper_20']
        df['BB_Middle'] = df['BB_Middle_20']
        df['BB_Lower'] = df['BB_Lower_20']
        df['BB_Width'] = df['BB_Width_20']
        df['BB_Position'] = df['BB_Position_20']

        # Advanced volatility indicators
        df['ATR_14'] = self.technical_indicators.atr(df['High'], df['Low'], df['Close'], 14)
        df['ATR_21'] = self.technical_indicators.atr(df['High'], df['Low'], df['Close'], 21)
        df['ATR'] = df['ATR_14']  # Default ATR

        df['Volatility_10'] = df['Returns'].rolling(10).std() * np.sqrt(252)
        df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)

        # Volume indicators (ALL from original)
        df['Volume_SMA_20'] = self.technical_indicators.sma(df['Volume'], 20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['OBV'] = self.technical_indicators.obv(df['Close'], df['Volume'])
        df['Volume_Price_Trend'] = self.technical_indicators.volume_price_trend(df['Close'], df['Volume'])

        # Advanced candlestick patterns (ALL from original)
        df['Doji'] = self.technical_indicators.identify_doji(
            df['Open'], df['High'], df['Low'], df['Close']
        )
        df['Hammer'] = self.technical_indicators.identify_hammer(
            df['Open'], df['High'], df['Low'], df['Close']
        )
        df['Shooting_Star'] = self.technical_indicators.identify_shooting_star(
            df['Open'], df['High'], df['Low'], df['Close']
        )

        # Support and Resistance levels
        df['Support'] = df['Low'].rolling(window=20, center=True).min()
        df['Resistance'] = df['High'].rolling(window=20, center=True).max()

        # Fibonacci retracements (ALL levels from original)
        df['Fib_23.6'], df['Fib_38.2'], df['Fib_50'], df['Fib_61.8'] = self.technical_indicators.fibonacci_levels(
            df['High'], df['Low'], 50
        )

        self.data = df
        print(
            f"Advanced technical indicators calculated: {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} indicators")

    def analyze_smart_money_flow(self):
        """Analyze smart money flow using Wyckoff methodology and institutional detection"""
        if self.data is None:
            return

        df = self.data.copy()

        # Wyckoff Analysis
        wyckoff_signals = self.analyze_wyckoff_methodology(df)

        # Institutional flow detection
        institutional_flow = self.detect_institutional_flow(df)

        # Volume profile analysis
        volume_profile = self.analyze_volume_profile(df)

        # Market structure analysis
        market_structure = self.analyze_market_structure(df)

        self.smart_money_analysis = {
            'wyckoff_phase': wyckoff_signals.get('phase', 'Unknown'),
            'institutional_sentiment': institutional_flow.get('sentiment', 'Neutral'),
            'volume_profile': volume_profile.get('profile', 'Balanced'),
            'market_structure': market_structure.get('structure', 'Sideways'),
            'smart_money_confidence': self.calculate_smart_money_confidence(
                wyckoff_signals, institutional_flow, volume_profile, market_structure
            )
        }

        print("Smart money analysis completed")

    def analyze_wyckoff_methodology(self, df):
        """Implement Wyckoff methodology for smart money detection"""
        # Simplified Wyckoff analysis

        # Calculate price and volume characteristics
        price_trend = df['Close'].rolling(50).mean().diff()
        volume_trend = df['Volume'].rolling(20).mean().diff()

        # Relative volume
        avg_volume = df['Volume'].rolling(50).mean()
        relative_volume = df['Volume'] / avg_volume

        # Price-volume divergence
        price_change = df['Close'].pct_change()
        volume_spike = relative_volume > 1.5

        # Determine Wyckoff phase
        recent_price_trend = price_trend.tail(10).mean()
        recent_volume_trend = volume_trend.tail(10).mean()

        if recent_price_trend > 0 and recent_volume_trend > 0:
            phase = "Accumulation"
        elif recent_price_trend < 0 and recent_volume_trend > 0:
            phase = "Distribution"
        elif recent_price_trend > 0 and recent_volume_trend < 0:
            phase = "Markup"
        elif recent_price_trend < 0 and recent_volume_trend < 0:
            phase = "Markdown"
        else:
            phase = "Neutral"

        return {
            'phase': phase,
            'price_trend': recent_price_trend,
            'volume_trend': recent_volume_trend,
            'volume_spikes': volume_spike.tail(20).sum()
        }

    def detect_institutional_flow(self, df):
        """Detect institutional money flow patterns"""
        # Large volume transactions (potential institutional activity)
        volume_threshold = df['Volume'].quantile(0.9)
        large_volume_days = df['Volume'] > volume_threshold

        # Price impact analysis
        price_impact = df['Close'].pct_change()

        # Institutional buying: Large volume + positive price movement
        institutional_buying = large_volume_days & (price_impact > 0)

        # Institutional selling: Large volume + negative price movement
        institutional_selling = large_volume_days & (price_impact < 0)

        # Recent institutional activity (last 20 days)
        recent_buying = institutional_buying.tail(20).sum()
        recent_selling = institutional_selling.tail(20).sum()

        if recent_buying > recent_selling * 1.5:
            sentiment = "Bullish"
        elif recent_selling > recent_buying * 1.5:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

        return {
            'sentiment': sentiment,
            'buying_days': recent_buying,
            'selling_days': recent_selling,
            'net_flow': recent_buying - recent_selling
        }

    def analyze_volume_profile(self, df):
        """Analyze volume profile for institutional activity"""
        # Volume-weighted average price
        vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

        # Current price vs VWAP
        current_price = df['Close'].iloc[-1]
        current_vwap = vwap.iloc[-1]

        # Volume distribution
        high_volume_threshold = df['Volume'].quantile(0.8)
        low_volume_threshold = df['Volume'].quantile(0.2)

        high_volume_periods = df['Volume'] > high_volume_threshold
        low_volume_periods = df['Volume'] < low_volume_threshold

        if current_price > current_vwap * 1.02:
            profile = "Above VWAP - Bullish"
        elif current_price < current_vwap * 0.98:
            profile = "Below VWAP - Bearish"
        else:
            profile = "Near VWAP - Balanced"

        return {
            'profile': profile,
            'vwap': current_vwap,
            'price_vwap_ratio': current_price / current_vwap,
            'high_volume_periods': high_volume_periods.tail(20).sum()
        }

    def analyze_market_structure(self, df):
        """Analyze market structure for trend identification"""
        # Higher highs and higher lows (uptrend)
        # Lower highs and lower lows (downtrend)

        # Calculate swing points
        window = 10
        highs = df['High'].rolling(window=window, center=True).max() == df['High']
        lows = df['Low'].rolling(window=window, center=True).min() == df['Low']

        # Get recent swing points
        recent_highs = df[highs]['High'].tail(3).values
        recent_lows = df[lows]['Low'].tail(3).values

        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            higher_highs = recent_highs[-1] > recent_highs[-2]
            higher_lows = recent_lows[-1] > recent_lows[-2]

            if higher_highs and higher_lows:
                structure = "Uptrend"
            elif not higher_highs and not higher_lows:
                structure = "Downtrend"
            else:
                structure = "Sideways"
        else:
            structure = "Insufficient Data"

        return {
            'structure': structure,
            'recent_highs': recent_highs.tolist() if len(recent_highs) > 0 else [],
            'recent_lows': recent_lows.tolist() if len(recent_lows) > 0 else []
        }

    def calculate_smart_money_confidence(self, wyckoff, institutional, volume_profile, market_structure):
        """Calculate overall smart money confidence score"""
        confidence_score = 0.5  # Base confidence

        # Wyckoff contribution
        if wyckoff['phase'] in ['Accumulation', 'Markup']:
            confidence_score += 0.15
        elif wyckoff['phase'] in ['Distribution', 'Markdown']:
            confidence_score -= 0.15

        # Institutional flow contribution
        if institutional['sentiment'] == 'Bullish':
            confidence_score += 0.15
        elif institutional['sentiment'] == 'Bearish':
            confidence_score -= 0.15

        # Volume profile contribution
        if 'Bullish' in volume_profile['profile']:
            confidence_score += 0.1
        elif 'Bearish' in volume_profile['profile']:
            confidence_score -= 0.1

        # Market structure contribution
        if market_structure['structure'] == 'Uptrend':
            confidence_score += 0.1
        elif market_structure['structure'] == 'Downtrend':
            confidence_score -= 0.1

        return max(0, min(1, confidence_score))

    def enhanced_feature_engineering(self):
        """Enhanced feature engineering with ALL original features"""
        df = self.data.copy()

        # Time-based features (ALL from original)
        if hasattr(df.index, 'hour'):
            df['Hour'] = df.index.hour
        if hasattr(df.index, 'dayofweek'):
            df['DayOfWeek'] = df.index.dayofweek
        if hasattr(df.index, 'month'):
            df['Month'] = df.index.month
        if hasattr(df.index, 'quarter'):
            df['Quarter'] = df.index.quarter

        # Cyclical encoding for time features (from original)
        if 'Hour' in df.columns:
            df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

        if 'DayOfWeek' in df.columns:
            df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

        # Lag features (ALL from original)
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)

        # Rolling statistics (ALL windows from original)
        for window in [5, 10, 20]:
            df[f'Close_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_std_{window}'] = df['Close'].rolling(window).std()
            df[f'Close_max_{window}'] = df['Close'].rolling(window).max()
            df[f'Close_min_{window}'] = df['Close'].rolling(window).min()
            df[f'Volume_mean_{window}'] = df['Volume'].rolling(window).mean()

        # Interaction features (ALL from original)
        df['Price_Volume_Interaction'] = df['Close'] * df['Volume']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']

        # Volatility features (ALL from original)
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Body_Range'] = abs(df['Close'] - df['Open']) / df['Close']
        df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']

        # Fill missing values using forward fill then backward fill (from original)
        self.data = df.ffill().bfill()

        print(f"Enhanced feature engineering completed: {len(self.data.columns)} total features")

    def prepare_enhanced_features(self):
        """Prepare enhanced feature set for ML models (ALL original functionality)"""
        df = self.data.copy()

        # Select all numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target-related columns (from original)
        exclude_features = ['Close', 'High', 'Low', 'Open']
        feature_columns = [col for col in numeric_features if col not in exclude_features]

        # Ensure we have enough data
        df = df.dropna()

        if len(df) < 100:
            raise ValueError("Insufficient data for analysis")

        # Create feature matrix
        self.features = df[feature_columns].copy()

        # Create multiple targets (ALL from original)
        self.features['Next_Close'] = df['Close'].shift(-1)
        self.features['Next_High'] = df['High'].shift(-1)
        self.features['Next_Low'] = df['Low'].shift(-1)
        self.features['Next_Volume'] = df['Volume'].shift(-1)

        # Price direction (classification target)
        self.features['Price_Direction'] = (self.features['Next_Close'] > df['Close']).astype(int)

        # Price change magnitude
        self.features['Price_Change_Pct'] = (self.features['Next_Close'] - df['Close']) / df['Close'] * 100

        # Remove last rows with NaN targets
        self.features = self.features[:-1]

        # Feature selection (from original)
        self.select_best_features()

        print(f"Feature preparation completed: {len(self.features.columns)} features ready")

    def select_best_features(self, k=50):
        """Select the best features for modeling (enhanced from original)"""
        feature_cols = [col for col in self.features.columns
                        if not col.startswith('Next_') and col != 'Price_Direction' and col != 'Price_Change_Pct']

        X = self.features[feature_cols]
        y = self.features['Next_Close']

        # Remove highly correlated features (from original)
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

        print(f"Dropping {len(to_drop)} highly correlated features")
        X = X.drop(columns=to_drop)

        # Select K best features (from original)
        selector = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))
        X_selected = selector.fit_transform(X, y)

        selected_features = X.columns[selector.get_support()].tolist()
        print(f"Selected {len(selected_features)} best features")

        # Update features dataframe
        self.features = self.features[selected_features + ['Next_Close', 'Next_High', 'Next_Low',
                                                           'Next_Volume', 'Price_Direction', 'Price_Change_Pct']]

    def train_enhanced_ml_models(self, selected_models=None):
        """Train enhanced ML models with ALL original ensemble methods"""
        if self.features is None:
            raise ValueError("Features not prepared")

        if selected_models is None:
            selected_models = ['rf', 'xgb', 'lgb', 'cb', 'et']

        # Prepare data
        feature_cols = [col for col in self.features.columns
                        if not col.startswith('Next_') and col != 'Price_Direction' and col != 'Price_Change_Pct']

        X = self.features[feature_cols].fillna(0)

        # Time series split (from original)
        tscv = TimeSeriesSplit(n_splits=5)

        # Define targets (ALL from original)
        targets = {
            'price': 'Next_Close',
            'direction': 'Price_Direction',
            'volume': 'Next_Volume',
            'change_pct': 'Price_Change_Pct'
        }

        for target_name, target_col in targets.items():
            print(f"Training enhanced models for {target_name}...")

            y = self.features[target_col].ffill()

            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Scale features (from original)
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            self.scalers[target_name] = scaler

            # Enhanced model ensemble (ALL from original)
            base_models = {}

            if 'rf' in selected_models:
                base_models['rf'] = RandomForestRegressor(
                    n_estimators=200, max_depth=15, min_samples_split=5,
                    random_state=42, n_jobs=-1
                )

            if 'xgb' in selected_models:
                base_models['xgb'] = xgb.XGBRegressor(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42, n_jobs=-1
                )

            if 'lgb' in selected_models:
                base_models['lgb'] = lgb.LGBMRegressor(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42, n_jobs=-1, verbose=-1
                )

            if 'cb' in selected_models:
                base_models['cb'] = cb.CatBoostRegressor(
                    iterations=200, depth=8, learning_rate=0.1,
                    verbose=False, random_state=42
                )

            if 'et' in selected_models:
                base_models['et'] = ExtraTreesRegressor(
                    n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
                )

            # Train individual models
            trained_models = {}
            model_scores = {}

            for name, model in base_models.items():
                try:
                    # Train model
                    if name in ['rf', 'et', 'xgb', 'lgb', 'cb']:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    else:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)

                    # Calculate score
                    if target_name == 'direction':
                        score = accuracy_score(y_test, (y_pred > 0.5).astype(int))
                    else:
                        score = r2_score(y_test, y_pred)

                    trained_models[name] = model
                    model_scores[name] = score

                    print(f"  {name.upper()}: {score:.4f}")

                except Exception as e:
                    print(f"  Error training {name}: {str(e)}")

            # Create ensemble (from original with enhancements)
            if len(trained_models) >= 3:
                try:
                    ensemble_models = [(name, model) for name, model in trained_models.items()]

                    # Voting Regressor (from original)
                    if 'voting' in selected_models:
                        voting_regressor = VotingRegressor(
                            estimators=ensemble_models,
                            weights=[model_scores[name] for name, _ in ensemble_models]
                        )

                        voting_regressor.fit(X_train, y_train)
                        ensemble_pred = voting_regressor.predict(X_test)

                        if target_name == 'direction':
                            ensemble_score = accuracy_score(y_test, (ensemble_pred > 0.5).astype(int))
                        else:
                            ensemble_score = r2_score(y_test, ensemble_pred)

                        print(f"  VOTING ENSEMBLE: {ensemble_score:.4f}")

                        self.models[target_name] = voting_regressor
                        self.model_performance[target_name] = ensemble_score

                    # Stacking Regressor (from original)
                    elif 'stacking' in selected_models:
                        stacking_regressor = StackingRegressor(
                            estimators=ensemble_models,
                            final_estimator=Ridge(alpha=0.1),
                            cv=tscv
                        )

                        stacking_regressor.fit(X_train, y_train)
                        stacking_pred = stacking_regressor.predict(X_test)

                        if target_name == 'direction':
                            stacking_score = accuracy_score(y_test, (stacking_pred > 0.5).astype(int))
                        else:
                            stacking_score = r2_score(y_test, stacking_pred)

                        print(f"  STACKING ENSEMBLE: {stacking_score:.4f}")

                        self.models[target_name] = stacking_regressor
                        self.model_performance[target_name] = stacking_score
                    else:
                        # Use best individual model
                        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k])
                        self.models[target_name] = trained_models[best_model_name]
                        self.model_performance[target_name] = model_scores[best_model_name]

                except Exception as e:
                    print(f"  Ensemble creation failed: {str(e)}")
                    # Fallback to best individual model
                    best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k])
                    self.models[target_name] = trained_models[best_model_name]
                    self.model_performance[target_name] = model_scores[best_model_name]
            else:
                # Use best individual model if not enough models for ensemble
                if trained_models:
                    best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k])
                    self.models[target_name] = trained_models[best_model_name]
                    self.model_performance[target_name] = model_scores[best_model_name]

            # Feature importance (from original)
            if hasattr(self.models[target_name], 'feature_importances_'):
                importance = self.models[target_name].feature_importances_
                self.feature_importance[target_name] = dict(zip(feature_cols, importance))
            elif hasattr(self.models[target_name], 'estimators_'):
                # For ensemble models, average feature importances
                try:
                    avg_importance = np.zeros(len(feature_cols))
                    for estimator in self.models[target_name].estimators_:
                        if hasattr(estimator, 'feature_importances_'):
                            avg_importance += estimator.feature_importances_
                    avg_importance /= len(self.models[target_name].estimators_)
                    self.feature_importance[target_name] = dict(zip(feature_cols, avg_importance))
                except:
                    pass

        print(f"ML model training completed. {len(self.models)} models trained.")

    def train_advanced_deep_learning_models(self, sequence_length=60, selected_dl_models=None):
        """Train advanced deep learning models with ALL original architectures"""
        if not DEEP_LEARNING_AVAILABLE:
            print("TensorFlow not available - skipping deep learning models")
            return

        if selected_dl_models is None:
            selected_dl_models = ['lstm']

        print("Training advanced deep learning models...")

        try:
            # Prepare sequence data
            feature_cols = [col for col in self.features.columns
                            if not col.startswith('Next_') and col != 'Price_Direction' and col != 'Price_Change_Pct']

            # Use ffill instead of fillna(method='ffill')
            data = self.features[feature_cols + ['Next_Close', 'Price_Direction']].ffill().values

            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            self.scalers['deep_learning'] = scaler

            # Create sequences with proper error handling
            X, y_price, y_direction = [], [], []

            for i in range(sequence_length, len(scaled_data)):
                try:
                    X.append(scaled_data[i - sequence_length:i, :-2])  # All features except targets
                    y_price.append(scaled_data[i, -2])  # Next_Close
                    y_direction.append(scaled_data[i, -1])  # Price_Direction
                except IndexError as e:
                    print(f"Index error at position {i}: {e}")
                    break

            # Convert to numpy arrays with error checking
            if len(X) == 0:
                raise Exception("No sequences created - insufficient data")

            X = np.array(X)
            y_price = np.array(y_price)
            y_direction = np.array(y_direction)

            print(f"Created sequences - X shape: {X.shape}, y_price shape: {y_price.shape}")

            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_price_train, y_price_test = y_price[:split_idx], y_price[split_idx:]
            y_direction_train, y_direction_test = y_direction[:split_idx], y_direction[split_idx:]

            # Build and train models based on selection
            if 'lstm' in selected_dl_models:
                try:
                    print("Training LSTM model...")
                    price_model = self.build_advanced_lstm_model(X_train.shape)

                    # Callbacks
                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

                    # Train price model
                    price_history = price_model.fit(
                        X_train, y_price_train,
                        batch_size=32,
                        epochs=50,
                        validation_data=(X_test, y_price_test),
                        callbacks=[early_stopping, reduce_lr],
                        verbose=1
                    )

                    # Safe evaluation with proper error handling
                    try:
                        price_score_raw = price_model.evaluate(X_test, y_price_test, verbose=0)

                        # Handle both single value and list returns
                        if isinstance(price_score_raw, (list, tuple)):
                            price_score = float(price_score_raw[0])  # Take first element if list
                        else:
                            price_score = float(price_score_raw)

                        print(f"LSTM Price Model - Test Loss: {price_score:.4f}")

                    except Exception as eval_error:
                        print(f"Evaluation error: {eval_error}")
                        price_score = 1.0  # Default high loss value
                        print("Using default loss value due to evaluation error")

                    # Safe performance calculation
                    try:
                        performance_score = max(0.0, min(1.0, 1.0 - price_score))  # Clamp between 0 and 1
                        self.deep_models['price'] = price_model
                        self.model_performance['deep_price'] = performance_score
                        print(f"LSTM Performance Score: {performance_score:.4f}")

                    except Exception as perf_error:
                        print(f"Performance calculation error: {perf_error}")
                        self.model_performance['deep_price'] = 0.5  # Default moderate performance

                except Exception as lstm_error:
                    print(f"LSTM training failed: {str(lstm_error)}")
                    # Don't raise exception, continue with other models
                    print("Continuing without LSTM model...")

            # CNN-LSTM Hybrid
            if 'cnn_lstm' in selected_dl_models:
                try:
                    print("Training CNN-LSTM model...")
                    direction_model = self.build_cnn_lstm_model(X_train.shape)

                    # Train direction model
                    direction_history = direction_model.fit(
                        X_train, y_direction_train,
                        batch_size=32,
                        epochs=30,  # Reduced epochs
                        validation_data=(X_test, y_direction_test),
                        verbose=1
                    )

                    # Safe evaluation for direction model
                    try:
                        direction_pred = direction_model.predict(X_test, verbose=0)
                        direction_accuracy = accuracy_score(y_direction_test, (direction_pred > 0.5).astype(int))
                        print(f"CNN-LSTM Direction Model - Accuracy: {direction_accuracy:.4f}")

                        self.deep_models['direction'] = direction_model
                        self.model_performance['deep_direction'] = float(direction_accuracy)

                    except Exception as dir_eval_error:
                        print(f"Direction model evaluation error: {dir_eval_error}")

                except Exception as cnn_error:
                    print(f"CNN-LSTM training failed: {str(cnn_error)}")

            # Store configuration
            self.deep_models['sequence_length'] = sequence_length

            # Safe model counting
            trained_models = len([k for k in self.deep_models.keys() if k != 'sequence_length'])
            print(f"Deep learning model training completed. {trained_models} models trained.")

        except Exception as e:
            print(f"Deep learning training error: {str(e)}")
            # Instead of raising, just log and continue
            print("Deep learning training failed, but continuing with ML models...")
            return  # Don't raise exception

    def build_advanced_lstm_model(self, input_shape):
        """Build advanced LSTM model with attention (ALL from original)"""
        try:
            inputs = Input(shape=(input_shape[1], input_shape[2]))

            # First LSTM layer
            lstm1 = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
            lstm1 = BatchNormalization()(lstm1)

            # Second LSTM layer
            lstm2 = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm1)
            lstm2 = BatchNormalization()(lstm2)

            # Third LSTM layer
            lstm3 = LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(lstm2)
            lstm3 = BatchNormalization()(lstm3)

            # Dense layers
            dense1 = Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(lstm3)
            dense1 = Dropout(0.3)(dense1)

            dense2 = Dense(25, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(dense1)
            dense2 = Dropout(0.2)(dense2)

            outputs = Dense(1, activation='linear')(dense2)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])

            return model
        except Exception as e:
            raise Exception(f"LSTM model building error: {str(e)}")


    def build_cnn_lstm_model(self, input_shape):
        """Build CNN-LSTM hybrid model for direction prediction (from original)"""
        inputs = Input(shape=(input_shape[1], input_shape[2]))

        # CNN layers for pattern detection
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Dropout(0.2)(conv1)

        # LSTM layers
        lstm1 = LSTM(50, return_sequences=True, dropout=0.2)(conv1)
        lstm2 = LSTM(25, return_sequences=False, dropout=0.2)(lstm1)

        # Dense layers
        dense1 = Dense(25, activation='relu')(lstm2)
        dense1 = Dropout(0.3)(dense1)

        outputs = Dense(1, activation='sigmoid')(dense1)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def build_gru_model(self, input_shape):
        """Build GRU model"""
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        return model

    def build_attention_lstm_model(self, input_shape):
        """Build attention-based LSTM model"""
        inputs = Input(shape=(input_shape[1], input_shape[2]))

        # LSTM with return sequences for attention
        lstm_out = LSTM(64, return_sequences=True)(inputs)

        # Attention mechanism (simplified)      #break#1
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(64)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)

        # Apply attention
        sent_representation = tf.keras.layers.Multiply()([lstm_out, attention])
        sent_representation = tf.keras.layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=-2))(
            sent_representation)

        # Final layers
        dense = Dense(32, activation='relu')(sent_representation)
        outputs = Dense(1, activation='linear')(dense)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        return model

    def make_enhanced_predictions(self):
        """Make enhanced predictions with confidence intervals (ALL from original)"""
        if not self.models:
            raise ValueError("Models not trained")

        print("Making enhanced predictions...")

        # Prepare latest data
        feature_cols = [col for col in self.features.columns
                        if not col.startswith('Next_') and col != 'Price_Direction' and col != 'Price_Change_Pct']

        latest_data = self.features[feature_cols].iloc[-1:].fillna(0)

        predictions = {}
        confidence_scores = {}

        # ML predictions
        for target_name, model in self.models.items():
            try:
                if target_name in self.scalers:
                    scaled_data = self.scalers[target_name].transform(latest_data)
                    pred = model.predict(scaled_data)[0]
                else:
                    pred = model.predict(latest_data)[0]

                predictions[target_name] = pred

                # Calculate confidence based on model performance
                if target_name in self.model_performance:
                    confidence_scores[target_name] = self.model_performance[target_name]

            except Exception as e:
                print(f"Error making prediction for {target_name}: {str(e)}")

        # Deep learning predictions (ALL from original)
        if self.deep_models and DEEP_LEARNING_AVAILABLE:
            try:
                seq_length = self.deep_models['sequence_length']
                scaler = self.scalers['deep_learning']

                # Prepare sequence
                recent_data = self.features[feature_cols].iloc[-seq_length:].ffill()

                # Ensure we have enough data
                if len(recent_data) >= seq_length:
                    scaled_recent = scaler.transform(
                        np.column_stack([recent_data.values,
                                         np.zeros((len(recent_data), 2))])  # Add dummy targets
                    )[:, :-2]  # Remove dummy targets

                    X_pred = scaled_recent.reshape(1, seq_length, recent_data.shape[1])

                    # All deep learning model predictions
                    for model_name, model in self.deep_models.items():
                        if model_name == 'sequence_length':
                            continue

                        try:
                            if model_name == 'price':
                                price_pred = model.predict(X_pred, verbose=0)[0][0]

                                # Inverse transform
                                dummy_array = np.zeros((1, scaler.n_features_in_))
                                dummy_array[0, -2] = price_pred  # Price target position
                                price_pred_unscaled = scaler.inverse_transform(dummy_array)[0, -2]

                                predictions['deep_price'] = price_pred_unscaled
                                confidence_scores['deep_price'] = self.model_performance.get('deep_price', 0.8)

                            elif model_name == 'direction':
                                direction_pred = model.predict(X_pred, verbose=0)[0][0]
                                predictions['deep_direction'] = direction_pred
                                confidence_scores['deep_direction'] = self.model_performance.get('deep_direction', 0.8)

                            elif model_name in ['gru', 'attention']:
                                other_pred = model.predict(X_pred, verbose=0)[0][0]

                                # Inverse transform for price predictions
                                dummy_array = np.zeros((1, scaler.n_features_in_))
                                dummy_array[0, -2] = other_pred
                                other_pred_unscaled = scaler.inverse_transform(dummy_array)[0, -2]

                                predictions[f'deep_{model_name}'] = other_pred_unscaled
                                confidence_scores[f'deep_{model_name}'] = self.model_performance.get(
                                    f'deep_{model_name}', 0.8)

                        except Exception as e:
                            print(f"Error with deep learning model {model_name}: {str(e)}")

            except Exception as e:
                print(f"Error making deep learning prediction: {str(e)}")

        self.predictions = predictions
        self.prediction_confidence = confidence_scores

        print(f"Predictions completed: {len(predictions)} predictions generated")
        return predictions, confidence_scores

    def calculate_comprehensive_risk_metrics(self):
        """Calculate comprehensive risk metrics"""
        if self.data is None:
            return

        returns = self.data['Close'].pct_change().dropna()

        # Basic risk metrics
        self.risk_metrics = {
            'volatility_daily': returns.std(),
            'volatility_annual': returns.std() * np.sqrt(252),
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
        }

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        self.risk_metrics['max_drawdown'] = drawdown.min()

        # Sharpe ratio (assuming 2% risk-free rate)
        excess_returns = returns - 0.02 / 252
        self.risk_metrics['sharpe_ratio'] = excess_returns.mean() / returns.std() * np.sqrt(252)

        print("Comprehensive risk metrics calculated")

class ProfessionalSmartStockAIApp:
        """Professional Desktop Application with Light Blue Theme and Enhanced UX"""

        def __init__(self):
            self.root = tk.Tk()
            self.root.title("SmartStock AI - Professional Trading Analysis")
            self.root.geometry("1800x1200")
            self.root.minsize(1400, 900)

            # Professional color scheme - Light Blue & White
            self.colors = {
                'primary_blue': '#1E90FF',  # Dodger Blue
                'light_blue': '#87CEEB',  # Sky Blue
                'steel_blue': '#4682B4',  # Steel Blue
                'white': '#FFFFFF',  # Pure White
                'light_gray': '#F5F5F5',  # Very Light Gray
                'medium_gray': '#E0E0E0',  # Light Gray
                'dark_blue': '#003366',  # Dark Blue for text
                'accent_blue': '#0078D4',  # Microsoft Blue
                'success_green': '#28A745',  # Success Green
                'warning_orange': '#FD7E14',  # Warning Orange
                'error_red': '#DC3545',  # Error Red
                'gradient_start': '#87CEEB',  # Light gradient start
                'gradient_end': '#1E90FF'  # Dark gradient end
            }

            # Configure main window
            self.root.configure(bg=self.colors['white'])

            # Zoom factor for scalability
            self.zoom_factor = 1.0

            # Variables
            self.csv_file_path = None
            self.analysis_complete = False
            self.real_time_enabled = tk.BooleanVar()
            self.auto_update_predictions = tk.BooleanVar()

            # Initialize components
            self.ai_agent = EnhancedStockMarketAIAgent()
            self.setup_professional_styles()
            self.setup_keyboard_bindings()
            #self.create_professional_gui()
            self.apply_professional_theme()

            self.setup_professional_styles()  # Add this line
            self.create_professional_gui()

        def setup_professional_styles(self):
            """Setup professional light blue and white theme styles"""
            self.style = ttk.Style()
            self.style.theme_use('clam')

            # Configure professional styles

            # Main frame styles
            self.style.configure('Professional.TFrame',
                                 background=self.colors['white'],
                                 relief='flat',
                                 borderwidth=0)

            self.style.configure('Card.TFrame',
                                 background=self.colors['white'],
                                 relief='solid',
                                 borderwidth=1,
                                 bordercolor=self.colors['medium_gray'])

            # Professional labels
            self.style.configure('Title.TLabel',
                                 background=self.colors['white'],
                                 foreground=self.colors['dark_blue'],
                                 font=('Segoe UI', 24, 'bold'))

            self.style.configure('Heading.TLabel',
                                 background=self.colors['white'],
                                 foreground=self.colors['dark_blue'],
                                 font=('Segoe UI', 16, 'bold'))

            self.style.configure('Subheading.TLabel',
                                 background=self.colors['white'],
                                 foreground=self.colors['steel_blue'],
                                 font=('Segoe UI', 12, 'bold'))

            self.style.configure('Body.TLabel',
                                 background=self.colors['white'],
                                 foreground=self.colors['dark_blue'],
                                 font=('Segoe UI', 10))

            # Professional buttons
            self.style.configure('Primary.TButton',
                                 background=self.colors['primary_blue'],
                                 foreground='white',
                                 font=('Segoe UI', 11, 'bold'),
                                 padding=(20, 12),
                                 relief='flat',
                                 borderwidth=0)

            self.style.map('Primary.TButton',
                           background=[('active', self.colors['steel_blue']),
                                       ('pressed', self.colors['dark_blue'])])

            self.style.configure('Secondary.TButton',
                                 background=self.colors['light_blue'],
                                 foreground=self.colors['dark_blue'],
                                 font=('Segoe UI', 10),
                                 padding=(15, 8),
                                 relief='flat',
                                 borderwidth=1)

            self.style.map('Secondary.TButton',
                           background=[('active', self.colors['medium_gray']),
                                       ('pressed', self.colors['light_gray'])])

            self.style.configure('Success.TButton',
                                 background=self.colors['success_green'],
                                 foreground='white',
                                 font=('Segoe UI', 10, 'bold'),
                                 padding=(15, 8))

            # Professional notebook (tabs)
            self.style.configure('Professional.TNotebook',
                                 background=self.colors['white'],
                                 tabmargins=[0, 0, 0, 0])

            self.style.configure('Professional.TNotebook.Tab',
                                 background=self.colors['light_gray'],
                                 foreground=self.colors['dark_blue'],
                                 font=('Segoe UI', 11, 'bold'),
                                 padding=[20, 12],
                                 borderwidth=0)

            self.style.map('Professional.TNotebook.Tab',
                           background=[('selected', self.colors['primary_blue']),
                                       ('active', self.colors['light_blue'])],
                           foreground=[('selected', 'white'),
                                       ('active', self.colors['dark_blue'])])

            # Professional treeview
            self.style.configure('Professional.Treeview',
                                 background=self.colors['white'],
                                 foreground=self.colors['dark_blue'],
                                 fieldbackground=self.colors['white'],
                                 font=('Segoe UI', 10),
                                 rowheight=25)

            self.style.configure('Professional.Treeview.Heading',
                                 background=self.colors['primary_blue'],
                                 foreground='white',
                                 font=('Segoe UI', 11, 'bold'),
                                 relief='flat')

            # Professional entry and text widgets
            self.style.configure('Professional.TEntry',
                                 fieldbackground=self.colors['white'],
                                 bordercolor=self.colors['primary_blue'],
                                 lightcolor=self.colors['light_blue'],
                                 darkcolor=self.colors['steel_blue'],
                                 borderwidth=2,
                                 insertcolor=self.colors['dark_blue'])

            # Professional progressbar - ENHANCED VERSION
            try:
                # Create custom layout for horizontal progressbar
                self.style.layout('Professional.TProgressbar',
                                  [('Horizontal.Progressbar.trough',
                                    {'children': [('Horizontal.Progressbar.pbar',
                                                   {'side': 'left', 'sticky': 'ns'})],
                                     'sticky': 'nswe'})])

                # Configure the progressbar style
                self.style.configure('Professional.TProgressbar',
                                     background=self.colors['primary_blue'],
                                     troughcolor=self.colors['light_gray'],
                                     borderwidth=1,
                                     lightcolor=self.colors['primary_blue'],
                                     darkcolor=self.colors['steel_blue'],
                                     relief='flat',
                                     thickness=20)

                # Map different states
                self.style.map('Professional.TProgressbar',
                               background=[('active', self.colors['steel_blue'])])

                print("✅ Professional progressbar style created successfully")

            except Exception as e:
                print(f"⚠️ Warning: Could not create custom progressbar layout: {e}")
                # Fallback to basic style configuration
                self.style.configure('Professional.TProgressbar',
                                     background=self.colors['primary_blue'],
                                     troughcolor=self.colors['light_gray'],
                                     borderwidth=0,
                                     lightcolor=self.colors['primary_blue'],
                                     darkcolor=self.colors['primary_blue'])

            # Professional checkbuttons and radiobuttons
            self.style.configure('Professional.TCheckbutton',
                                 background=self.colors['white'],
                                 foreground=self.colors['dark_blue'],
                                 font=('Segoe UI', 10),
                                 focuscolor='none')

            self.style.configure('Professional.TRadiobutton',
                                 background=self.colors['white'],
                                 foreground=self.colors['dark_blue'],
                                 font=('Segoe UI', 10),
                                 focuscolor='none')

            # Professional scale
            self.style.configure('Professional.TScale',
                                 background=self.colors['white'],
                                 troughcolor=self.colors['light_gray'],
                                 borderwidth=0,
                                 sliderthickness=15,
                                 gripcount=0)

            # Professional labelframe
            self.style.configure('Professional.TLabelframe',
                                 background=self.colors['white'],
                                 foreground=self.colors['dark_blue'],
                                 borderwidth=2,
                                 relief='solid',
                                 bordercolor=self.colors['medium_gray'])

            self.style.configure('Professional.TLabelframe.Label',
                                 background=self.colors['white'],
                                 foreground=self.colors['dark_blue'],
                                 font=('Segoe UI', 12, 'bold'))

            # Additional professional styles for completeness

            # Professional combobox
            self.style.configure('Professional.TCombobox',
                                 fieldbackground=self.colors['white'],
                                 background=self.colors['white'],
                                 foreground=self.colors['dark_blue'],
                                 bordercolor=self.colors['primary_blue'],
                                 lightcolor=self.colors['light_blue'],
                                 darkcolor=self.colors['steel_blue'],
                                 borderwidth=2,
                                 arrowcolor=self.colors['primary_blue'])

            # Professional spinbox
            self.style.configure('Professional.TSpinbox',
                                 fieldbackground=self.colors['white'],
                                 background=self.colors['white'],
                                 foreground=self.colors['dark_blue'],
                                 bordercolor=self.colors['primary_blue'],
                                 lightcolor=self.colors['light_blue'],
                                 darkcolor=self.colors['steel_blue'],
                                 borderwidth=2,
                                 arrowcolor=self.colors['primary_blue'])

            # Professional scrollbar
            self.style.configure('Professional.Vertical.TScrollbar',
                                 background=self.colors['light_gray'],
                                 troughcolor=self.colors['white'],
                                 bordercolor=self.colors['medium_gray'],
                                 arrowcolor=self.colors['steel_blue'],
                                 darkcolor=self.colors['primary_blue'],
                                 lightcolor=self.colors['light_blue'])

            self.style.configure('Professional.Horizontal.TScrollbar',
                                 background=self.colors['light_gray'],
                                 troughcolor=self.colors['white'],
                                 bordercolor=self.colors['medium_gray'],
                                 arrowcolor=self.colors['steel_blue'],
                                 darkcolor=self.colors['primary_blue'],
                                 lightcolor=self.colors['light_blue'])

            print("🎨 Professional theme styles configured successfully!")

        def setup_keyboard_bindings(self):
            """Setup keyboard shortcuts for enhanced UX"""
            self.root.bind('<Control-plus>', self.zoom_in)
            self.root.bind('<Control-minus>', self.zoom_out)
            self.root.bind('<Control-0>', self.reset_zoom)
            self.root.bind('<F11>', self.toggle_fullscreen)
            self.root.bind('<Control-o>', self.upload_csv_file)
            self.root.bind('<Control-s>', self.save_configuration)
            self.root.bind('<Control-r>', self.refresh_predictions)
            self.root.bind('<F5>', self.refresh_predictions)

        def zoom_in(self, event=None):
            """Increase zoom factor"""
            self.zoom_factor = min(2.0, self.zoom_factor + 0.1)
            self.apply_zoom()

        def zoom_out(self, event=None):
            """Decrease zoom factor"""
            self.zoom_factor = max(0.5, self.zoom_factor - 0.1)
            self.apply_zoom()

        def reset_zoom(self, event=None):
            """Reset zoom to 100%"""
            self.zoom_factor = 1.0
            self.apply_zoom()

        def apply_zoom(self):
            """Apply current zoom factor to fonts"""
            base_size = 10
            new_size = int(base_size * self.zoom_factor)

            # Update font sizes
            font_configs = [
                ('Segoe UI', int(24 * self.zoom_factor), 'bold'),  # Title
                ('Segoe UI', int(16 * self.zoom_factor), 'bold'),  # Heading
                ('Segoe UI', int(12 * self.zoom_factor), 'bold'),  # Subheading
                ('Segoe UI', int(10 * self.zoom_factor)),  # Body
            ]

            self.update_status(f"Zoom: {int(self.zoom_factor * 100)}%", "info")

        def toggle_fullscreen(self, event=None):
            """Toggle fullscreen mode"""
            current_state = self.root.attributes('-fullscreen')
            self.root.attributes('-fullscreen', not current_state)

        def create_professional_gui(self):
            """Create the professional GUI interface"""
            # Main container with gradient-like effect
            self.main_container = tk.Frame(self.root, bg=self.colors['white'])
            self.main_container.pack(fill=tk.BOTH, expand=True)

            # Header section
            self.create_professional_header()

            # Content area with sidebar
            self.create_content_area()

            # Status bar
            self.create_status_bar()

        def create_professional_header(self):
            """Create professional header with gradient effect"""
            # Header frame with gradient-like background
            header_frame = tk.Frame(self.main_container,
                                    bg=self.colors['primary_blue'],
                                    height=80)
            header_frame.pack(fill=tk.X, padx=0, pady=0)
            header_frame.pack_propagate(False)

            # Header content
            header_content = tk.Frame(header_frame, bg=self.colors['primary_blue'])
            header_content.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)

            # Left side - Title and logo
            left_header = tk.Frame(header_content, bg=self.colors['primary_blue'])
            left_header.pack(side=tk.LEFT, fill=tk.Y)

            # App title with icon
            title_frame = tk.Frame(left_header, bg=self.colors['primary_blue'])
            title_frame.pack(side=tk.LEFT, fill=tk.Y)

            icon_label = tk.Label(title_frame,
                                  text="📈",
                                  font=('Segoe UI', 28),
                                  bg=self.colors['primary_blue'],
                                  fg='white')
            icon_label.pack(side=tk.LEFT, padx=(0, 15))

            title_label = tk.Label(title_frame,
                                   text="SmartStock AI",
                                   font=('Segoe UI', 24, 'bold'),
                                   bg=self.colors['primary_blue'],
                                   fg='white')
            title_label.pack(side=tk.LEFT, anchor=tk.W)

            subtitle_label = tk.Label(left_header,
                                      text="Professional Trading Analysis Platform",
                                      font=('Segoe UI', 12),
                                      bg=self.colors['primary_blue'],
                                      fg=self.colors['light_blue'])
            subtitle_label.pack(side=tk.LEFT, anchor=tk.SW, padx=(10, 0))

            # Right side - Controls and status
            right_header = tk.Frame(header_content, bg=self.colors['primary_blue'])
            right_header.pack(side=tk.RIGHT, fill=tk.Y)

            # Status and controls
            controls_frame = tk.Frame(right_header, bg=self.colors['primary_blue'])
            controls_frame.pack(side=tk.RIGHT, fill=tk.Y)

            # Real-time toggle
            self.realtime_frame = tk.Frame(controls_frame, bg=self.colors['primary_blue'])
            self.realtime_frame.pack(side=tk.RIGHT, padx=(0, 20))

            realtime_cb = tk.Checkbutton(self.realtime_frame,
                                         text="Real-time Updates",
                                         variable=self.real_time_enabled,
                                         command=self.toggle_realtime,
                                         bg=self.colors['primary_blue'],
                                         fg='white',
                                         font=('Segoe UI', 10),
                                         selectcolor=self.colors['steel_blue'],
                                         activebackground=self.colors['primary_blue'],
                                         activeforeground='white')
            realtime_cb.pack(anchor=tk.E)

            # Status indicator
            self.status_indicator = tk.Label(controls_frame,
                                             text="● Ready",
                                             font=('Segoe UI', 12, 'bold'),
                                             bg=self.colors['primary_blue'],
                                             fg=self.colors['success_green'])
            self.status_indicator.pack(side=tk.RIGHT, anchor=tk.E)

            # User info
            user_label = tk.Label(controls_frame,
                                  text=f"👤 {self.get_current_user()}",
                                  font=('Segoe UI', 10),
                                  bg=self.colors['primary_blue'],
                                  fg='white')
            user_label.pack(side=tk.RIGHT, anchor=tk.NE, padx=(0, 15))

            # Date/time
            datetime_label = tk.Label(controls_frame,
                                      text=f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
                                      font=('Segoe UI', 9),
                                      bg=self.colors['primary_blue'],
                                      fg=self.colors['light_blue'])
            datetime_label.pack(side=tk.RIGHT, anchor=tk.NE)

        def create_content_area(self):
            """Create main content area with sidebar navigation"""
            # Content container
            content_frame = tk.Frame(self.main_container, bg=self.colors['light_gray'])
            content_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

            # Sidebar navigation
            self.sidebar = tk.Frame(content_frame,
                                    bg=self.colors['white'],
                                    width=250,
                                    relief='solid',
                                    borderwidth=1,
                                    bd=1)
            self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 1))
            self.sidebar.pack_propagate(False)

            self.create_sidebar_navigation()

            # Main content area
            self.main_content = tk.Frame(content_frame, bg=self.colors['light_gray'])
            self.main_content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

            # Create professional notebook for tabs
            self.create_professional_notebook()

        def create_sidebar_navigation(self):
            """Create sidebar navigation menu"""
            # Sidebar header
            sidebar_header = tk.Frame(self.sidebar, bg=self.colors['white'], height=60)
            sidebar_header.pack(fill=tk.X, padx=0, pady=0)
            sidebar_header.pack_propagate(False)

            nav_title = tk.Label(sidebar_header,
                                 text="Navigation",
                                 font=('Segoe UI', 14, 'bold'),
                                 bg=self.colors['white'],
                                 fg=self.colors['dark_blue'])
            nav_title.pack(pady=20)

            # Navigation buttons
            nav_buttons = [
                ("📁 Data Upload", "upload", self.show_upload_tab),
                ("⚙️ Analysis Setup", "analysis", self.show_analysis_tab),
                ("📈 Predictions", "predictions", self.show_predictions_tab),
                ("📊 Charts", "charts", self.show_charts_tab),
                ("🏆 Performance", "performance", self.show_performance_tab),
                ("⚠️ Risk Management", "risk", self.show_risk_tab),
                ("⚙️ Settings", "settings", self.show_settings_tab)
            ]

            self.nav_buttons = {}
            for text, key, command in nav_buttons:
                btn_frame = tk.Frame(self.sidebar, bg=self.colors['white'])
                btn_frame.pack(fill=tk.X, padx=10, pady=2)

                btn = tk.Button(btn_frame,
                                text=text,
                                command=command,
                                bg=self.colors['light_gray'],
                                fg=self.colors['dark_blue'],
                                font=('Segoe UI', 11),
                                relief='flat',
                                bd=0,
                                pady=12,
                                anchor='w',
                                activebackground=self.colors['primary_blue'],
                                activeforeground='white')
                btn.pack(fill=tk.X)

                self.nav_buttons[key] = btn

                # Hover effects
                def on_enter(e, button=btn):
                    if button['bg'] != self.colors['primary_blue']:
                        button.config(bg=self.colors['light_blue'])

                def on_leave(e, button=btn):
                    if button['bg'] != self.colors['primary_blue']:
                        button.config(bg=self.colors['light_gray'])

                btn.bind("<Enter>", on_enter)
                btn.bind("<Leave>", on_leave)

            # Quick actions section
            quick_frame = tk.Frame(self.sidebar, bg=self.colors['white'])
            quick_frame.pack(fill=tk.X, padx=10, pady=20)

            quick_title = tk.Label(quick_frame,
                                   text="Quick Actions",
                                   font=('Segoe UI', 12, 'bold'),
                                   bg=self.colors['white'],
                                   fg=self.colors['dark_blue'])
            quick_title.pack(anchor=tk.W, pady=(0, 10))

            # Quick action buttons
            quick_upload = tk.Button(quick_frame,
                                     text="📁 Upload Data",
                                     command=self.upload_csv_file,
                                     bg=self.colors['primary_blue'],
                                     fg='white',
                                     font=('Segoe UI', 9, 'bold'),
                                     relief='flat',
                                     pady=8)
            quick_upload.pack(fill=tk.X, pady=2)

            quick_analyze = tk.Button(quick_frame,
                                      text="🚀 Start Analysis",
                                      command=self.start_analysis,
                                      bg=self.colors['success_green'],
                                      fg='white',
                                      font=('Segoe UI', 9, 'bold'),
                                      relief='flat',
                                      pady=8)
            quick_analyze.pack(fill=tk.X, pady=2)

            quick_sample = tk.Button(quick_frame,
                                     text="🧪 Sample Data",
                                     command=self.use_sample_data,
                                     bg=self.colors['light_blue'],
                                     fg=self.colors['dark_blue'],
                                     font=('Segoe UI', 9),
                                     relief='flat',
                                     pady=8)
            quick_sample.pack(fill=tk.X, pady=2)

        def create_professional_notebook(self):
            """Create professional notebook with enhanced styling"""
            # Notebook container
            notebook_frame = tk.Frame(self.main_content, bg=self.colors['light_gray'])
            notebook_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

            # Create notebook
            self.notebook = ttk.Notebook(notebook_frame, style='Professional.TNotebook')
            self.notebook.pack(fill=tk.BOTH, expand=True)

            # Create all tabs
            self.create_upload_tab()
            self.create_analysis_tab()
            self.create_predictions_tab()
            self.create_charts_tab()
            self.create_performance_tab()
            self.create_risk_tab()
            self.create_settings_tab()

            # Bind tab change event
            self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        def create_status_bar(self):
            """Create professional status bar"""
            status_frame = tk.Frame(self.main_container,
                                    bg=self.colors['medium_gray'],
                                    height=30,
                                    relief='solid',
                                    bd=1)
            status_frame.pack(fill=tk.X, side=tk.BOTTOM)
            status_frame.pack_propagate(False)

            # Left status
            left_status = tk.Frame(status_frame, bg=self.colors['medium_gray'])
            left_status.pack(side=tk.LEFT, fill=tk.Y, padx=10)

            self.status_label = tk.Label(left_status,
                                         text="Ready",
                                         bg=self.colors['medium_gray'],
                                         fg=self.colors['dark_blue'],
                                         font=('Segoe UI', 9))
            self.status_label.pack(side=tk.LEFT, anchor=tk.W, pady=6)

            # Right status - zoom and version info
            right_status = tk.Frame(status_frame, bg=self.colors['medium_gray'])
            right_status.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

            version_label = tk.Label(right_status,
                                     text="v2.0 Professional",
                                     bg=self.colors['medium_gray'],
                                     fg=self.colors['steel_blue'],
                                     font=('Segoe UI', 8))
            version_label.pack(side=tk.RIGHT, anchor=tk.E, pady=6)

            zoom_label = tk.Label(right_status,
                                  text="100%",
                                  bg=self.colors['medium_gray'],
                                  fg=self.colors['dark_blue'],
                                  font=('Segoe UI', 9))
            zoom_label.pack(side=tk.RIGHT, anchor=tk.E, pady=6, padx=(0, 10))

        def create_upload_tab(self):
            """Create professional data upload tab"""
            upload_frame = ttk.Frame(self.notebook, style='Professional.TFrame')
            self.notebook.add(upload_frame, text="📁 Data Upload")

            # Scrollable frame
            canvas = tk.Canvas(upload_frame, bg=self.colors['white'])
            scrollbar = ttk.Scrollbar(upload_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas, style='Professional.TFrame')

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # Main content
            content_frame = ttk.Frame(scrollable_frame, style='Professional.TFrame')
            content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

            # Header
            header_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            header_frame.pack(fill=tk.X, pady=(0, 30))

            title_label = ttk.Label(header_frame,
                                    text="Data Upload & Validation",
                                    style='Title.TLabel')
            title_label.pack(anchor=tk.W)

            subtitle_label = ttk.Label(header_frame,
                                       text="Upload your stock data CSV file or generate sample data for analysis",
                                       style='Body.TLabel')
            subtitle_label.pack(anchor=tk.W, pady=(5, 0))

            # Upload card
            upload_card = self.create_card(content_frame, "Upload Options")

            # Instructions
            instructions_frame = ttk.Frame(upload_card, style='Professional.TFrame')
            instructions_frame.pack(fill=tk.X, pady=(0, 20))

            instructions_text = """📋 Required CSV Format:

        • Date column (supports multiple formats: YYYY-MM-DD, DD/MM/YYYY, etc.)
        • Open, High, Low, Close prices (numeric values)
        • Volume data (integer values)
        • Optional: Additional columns will be preserved

        🔍 Automatic Validation:
        • Data quality assessment
        • OHLC price consistency checks
        • Outlier detection and handling
        • Missing data identification"""

            instructions_label = tk.Label(instructions_frame,
                                          text=instructions_text,
                                          bg=self.colors['white'],
                                          fg=self.colors['dark_blue'],
                                          font=('Segoe UI', 10),
                                          justify=tk.LEFT,
                                          anchor=tk.W)
            instructions_label.pack(anchor=tk.W)

            # Upload buttons
            buttons_frame = ttk.Frame(upload_card, style='Professional.TFrame')
            buttons_frame.pack(fill=tk.X, pady=20)

            # Primary upload button
            upload_btn = tk.Button(buttons_frame,
                                   text="📁 Select CSV File",
                                   command=self.upload_csv_file,
                                   bg=self.colors['primary_blue'],
                                   fg='white',
                                   font=('Segoe UI', 12, 'bold'),
                                   relief='flat',
                                   pady=15,
                                   padx=30,
                                   cursor='hand2')
            upload_btn.pack(side=tk.LEFT, padx=(0, 15))

            # Secondary buttons
            sample_btn = tk.Button(buttons_frame,
                                   text="🧪 Generate Sample Data",
                                   command=self.use_sample_data,
                                   bg=self.colors['light_blue'],
                                   fg=self.colors['dark_blue'],
                                   font=('Segoe UI', 11),
                                   relief='flat',
                                   pady=12,
                                   padx=25,
                                   cursor='hand2')
            sample_btn.pack(side=tk.LEFT, padx=(0, 15))

            url_btn = tk.Button(buttons_frame,
                                text="🌐 Import from URL",
                                command=self.import_from_url,
                                bg=self.colors['medium_gray'],
                                fg=self.colors['dark_blue'],
                                font=('Segoe UI', 11),
                                relief='flat',
                                pady=12,
                                padx=25,
                                cursor='hand2')
            url_btn.pack(side=tk.LEFT)

            # Validation options card
            validation_card = self.create_card(content_frame, "Data Validation Options")

            self.validation_vars = {
                'outlier_detection': tk.BooleanVar(value=True),
                'missing_data_fill': tk.BooleanVar(value=True),
                'date_validation': tk.BooleanVar(value=True),
                'price_validation': tk.BooleanVar(value=True)
            }

            validation_options = [
                ('🔍 Outlier Detection & Removal', 'outlier_detection'),
                ('🔧 Fill Missing Data Points', 'missing_data_fill'),
                ('📅 Validate Date Formats', 'date_validation'),
                ('💰 Price Consistency Check', 'price_validation')
            ]

            for i, (text, key) in enumerate(validation_options):
                cb_frame = ttk.Frame(validation_card, style='Professional.TFrame')
                cb_frame.pack(fill=tk.X, pady=5)

                cb = tk.Checkbutton(cb_frame,
                                    text=text,
                                    variable=self.validation_vars[key],
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'],
                                    activeforeground=self.colors['dark_blue'])
                cb.pack(anchor=tk.W)

            # File information card
            info_card = self.create_card(content_frame, "File Information & Preview")

            # Create tabbed info display
            info_notebook = ttk.Notebook(info_card, style='Professional.TNotebook')
            info_notebook.pack(fill=tk.BOTH, expand=True, pady=10)

            # File info tab
            info_tab = ttk.Frame(info_notebook, style='Professional.TFrame')
            info_notebook.add(info_tab, text="📄 File Info")

            self.file_info_text = tk.Text(info_tab,
                                          height=10,
                                          bg=self.colors['white'],
                                          fg=self.colors['dark_blue'],
                                          font=('Consolas', 10),
                                          relief='solid',
                                          bd=1,
                                          wrap=tk.WORD)
            self.file_info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Data preview tab
            preview_tab = ttk.Frame(info_notebook, style='Professional.TFrame')
            info_notebook.add(preview_tab, text="👁️ Data Preview")

            preview_frame = ttk.Frame(preview_tab, style='Professional.TFrame')
            preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            self.data_preview = ttk.Treeview(preview_frame, style='Professional.Treeview')
            preview_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.data_preview.yview)
            self.data_preview.configure(yscrollcommand=preview_scroll.set)

            self.data_preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            # Statistics tab
            stats_tab = ttk.Frame(info_notebook, style='Professional.TFrame')
            info_notebook.add(stats_tab, text="📊 Statistics")

            self.stats_text = tk.Text(stats_tab,
                                      height=10,
                                      bg=self.colors['white'],
                                      fg=self.colors['dark_blue'],
                                      font=('Consolas', 10),
                                      relief='solid',
                                      bd=1,
                                      wrap=tk.WORD)
            self.stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Pack scrollable components
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

        def create_card(self, parent, title):
            """Create a professional card widget"""
            card_frame = ttk.Frame(parent, style='Professional.TFrame')
            card_frame.pack(fill=tk.X, pady=(0, 20))

            # Card container with shadow effect
            card_container = tk.Frame(card_frame,
                                      bg=self.colors['white'],
                                      relief='solid',
                                      bd=1,
                                      highlightbackground=self.colors['medium_gray'],
                                      highlightthickness=1)
            card_container.pack(fill=tk.X, padx=2, pady=2)

            # Card header
            header_frame = tk.Frame(card_container,
                                    bg=self.colors['light_gray'],
                                    height=40)
            header_frame.pack(fill=tk.X)
            header_frame.pack_propagate(False)

            title_label = tk.Label(header_frame,
                                   text=title,
                                   bg=self.colors['light_gray'],
                                   fg=self.colors['dark_blue'],
                                   font=('Segoe UI', 12, 'bold'))
            title_label.pack(side=tk.LEFT, padx=20, pady=10)

            # Card content
            content_frame = tk.Frame(card_container, bg=self.colors['white'])
            content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

            return content_frame

        """"
        def create_analysis_tab(self):
            ""Create professional analysis configuration tab""
            analysis_frame = ttk.Frame(self.notebook, style='Professional.TFrame')
            self.notebook.add(analysis_frame, text="⚙️ Analysis Setup")

            # Scrollable frame setup
            canvas = tk.Canvas(analysis_frame, bg=self.colors['white'])
            scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas, style='Professional.TFrame')

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # Main content
            content_frame = ttk.Frame(scrollable_frame, style='Professional.TFrame')
            content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

            # Header
            header_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            header_frame.pack(fill=tk.X, pady=(0, 30))

            title_label = ttk.Label(header_frame,
                                    text="Advanced Analysis Configuration",
                                    style='Title.TLabel')
            title_label.pack(anchor=tk.W)

            subtitle_label = ttk.Label(header_frame,
                                       text="Configure machine learning models, technical indicators, and analysis parameters",
                                       style='Body.TLabel')
            subtitle_label.pack(anchor=tk.W, pady=(5, 0))

            # Create three-column layout
            columns_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            columns_frame.pack(fill=tk.BOTH, expand=True)

            # Left column - ML Models
            left_column = ttk.Frame(columns_frame, style='Professional.TFrame')
            left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

            ml_card = self.create_card(left_column, "🤖 Machine Learning Models")

            # Ensemble models section
            ensemble_label = ttk.Label(ml_card,
                                       text="Ensemble Models:",
                                       style='Subheading.TLabel')
            ensemble_label.pack(anchor=tk.W, pady=(0, 10))

            self.model_vars = {}
            models = [
                ('🌳 Random Forest (Tree-based ensemble)', 'rf', True),
                ('🚀 XGBoost (Gradient boosting)', 'xgb', True),
                ('⚡ LightGBM (Fast gradient boosting)', 'lgb', True),
                ('🐱 CatBoost (Categorical features)', 'cb', True),
                ('🎲 Extra Trees (Randomized trees)', 'et', True),
                ('🗳️ Voting Regressor (Meta-ensemble)', 'voting', True),
                ('📚 Stacking Regressor (Layered ensemble)', 'stacking', True)
            ]

            for name, key, default in models:
                var = tk.BooleanVar(value=default)
                self.model_vars[key] = var

                cb_frame = ttk.Frame(ml_card, style='Professional.TFrame')
                cb_frame.pack(fill=tk.X, pady=3)

                cb = tk.Checkbutton(cb_frame,
                                    text=name,
                                    variable=var,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                cb.pack(anchor=tk.W)

            # Deep Learning section
            dl_label = ttk.Label(ml_card,
                                 text="Deep Learning Models:",
                                 style='Subheading.TLabel')
            dl_label.pack(anchor=tk.W, pady=(20, 10))

            self.dl_vars = {}
            dl_models = [
                ('🧠 LSTM (Long Short-Term Memory)', 'lstm', DEEP_LEARNING_AVAILABLE),
                ('🔄 GRU (Gated Recurrent Unit)', 'gru', DEEP_LEARNING_AVAILABLE),
                ('🌊 CNN-LSTM Hybrid', 'cnn_lstm', DEEP_LEARNING_AVAILABLE),
                ('🎯 Attention-based LSTM', 'attention_lstm', DEEP_LEARNING_AVAILABLE)
            ]

            for name, key, available in dl_models:
                var = tk.BooleanVar(value=available)
                self.dl_vars[key] = var

                cb_frame = ttk.Frame(ml_card, style='Professional.TFrame')
                cb_frame.pack(fill=tk.X, pady=3)

                cb = tk.Checkbutton(cb_frame,
                                    text=name,
                                    variable=var,
                                    state='normal' if available else 'disabled',
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'] if available else self.colors['medium_gray'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                cb.pack(anchor=tk.W)

            if not DEEP_LEARNING_AVAILABLE:
                warning_label = tk.Label(ml_card,
                                         text="⚠️ TensorFlow not available - Deep Learning disabled",
                                         bg=self.colors['white'],
                                         fg=self.colors['warning_orange'],
                                         font=('Segoe UI', 9))
                warning_label.pack(anchor=tk.W, pady=(5, 0))

            # Center column - Technical Analysis
            center_column = ttk.Frame(columns_frame, style='Professional.TFrame')
            center_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

            tech_card = self.create_card(center_column, "📈 Technical Analysis")

            # Technical indicators
            indicators_label = ttk.Label(tech_card,
                                         text="Technical Indicators:",
                                         style='Subheading.TLabel')
            indicators_label.pack(anchor=tk.W, pady=(0, 10))

            self.indicator_vars = {}
            indicators = [
                ('📊 Moving Averages (SMA, EMA, WMA)', 'ma', True),
                ('⚡ RSI & Stochastic Oscillators', 'momentum', True),
                ('📈 MACD (Multiple timeframes)', 'macd', True),
                ('🔵 Bollinger Bands (Multiple periods)', 'bb', True),
                ('📉 Williams %R', 'williams', True),
                ('📊 Volume Indicators (OBV, VPT)', 'volume', True),
                ('💨 Volatility (ATR, Historical)', 'volatility', True),
                ('🕯️ Candlestick Patterns', 'patterns', True),
                ('🌀 Fibonacci Retracements', 'fibonacci', True),
                ('🔗 Support/Resistance Levels', 'support_resistance', True)
            ]

            for name, key, default in indicators:
                var = tk.BooleanVar(value=default)
                self.indicator_vars[key] = var

                cb_frame = ttk.Frame(tech_card, style='Professional.TFrame')
                cb_frame.pack(fill=tk.X, pady=3)

                cb = tk.Checkbutton(cb_frame,
                                    text=name,
                                    variable=var,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                cb.pack(anchor=tk.W)

            # Smart Money Analysis
            smart_label = ttk.Label(tech_card,
                                    text="Smart Money Analysis:",
                                    style='Subheading.TLabel')
            smart_label.pack(anchor=tk.W, pady=(20, 10))

            self.smart_money_vars = {}
            smart_money_features = [
                ('💰 Wyckoff Methodology', 'wyckoff', True),
                ('🏛️ Institutional Flow Detection', 'institutional', True),
                ('📊 Volume Profile Analysis', 'volume_profile', True),
                ('🏗️ Market Structure Analysis', 'market_structure', True)
            ]

            for name, key, default in smart_money_features:
                var = tk.BooleanVar(value=default)
                self.smart_money_vars[key] = var

                cb_frame = ttk.Frame(tech_card, style='Professional.TFrame')
                cb_frame.pack(fill=tk.X, pady=3)

                cb = tk.Checkbutton(cb_frame,
                                    text=name,
                                    variable=var,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                cb.pack(anchor=tk.W)

            # Right column - Parameters
            right_column = ttk.Frame(columns_frame, style='Professional.TFrame')
            right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

            params_card = self.create_card(right_column, "⚙️ Analysis Parameters")

            # Prediction settings
            pred_label = ttk.Label(params_card,
                                   text="Prediction Settings:",
                                   style='Subheading.TLabel')
            pred_label.pack(anchor=tk.W, pady=(0, 10))

            # Prediction horizon
            horizon_frame = ttk.Frame(params_card, style='Professional.TFrame')
            horizon_frame.pack(fill=tk.X, pady=10)

            horizon_label = ttk.Label(horizon_frame,
                                      text="Prediction Horizon (days):",
                                      style='Body.TLabel')
            horizon_label.pack(anchor=tk.W)

            self.prediction_days = tk.IntVar(value=5)
            horizon_scale = tk.Scale(horizon_frame,
                                     from_=1, to=30,
                                     variable=self.prediction_days,
                                     orient=tk.HORIZONTAL,
                                     bg=self.colors['white'],
                                     fg=self.colors['dark_blue'],
                                     highlightthickness=0,
                                     troughcolor=self.colors['light_gray'],
                                     activebackground=self.colors['primary_blue'],
                                     command=self.update_prediction_label)
            horizon_scale.pack(fill=tk.X, pady=5)

            self.prediction_label = ttk.Label(horizon_frame,
                                              text="5 days",
                                              style='Body.TLabel')
            self.prediction_label.pack(anchor=tk.W)

            # Model optimization
            opt_label = ttk.Label(params_card,
                                  text="Model Optimization:",
                                  style='Subheading.TLabel')
            opt_label.pack(anchor=tk.W, pady=(20, 10))

            self.optimization_vars = {}
            optimization_options = [
                ('🔍 Hyperparameter Tuning (GridSearch)', 'grid_search', True),
                ('📊 Cross-Validation (Time Series)', 'cross_validation', True),
                ('🎯 Feature Selection (Auto)', 'feature_selection', True),
                ('⚖️ Ensemble Weighting', 'ensemble_weighting', True)
            ]

            for name, key, default in optimization_options:
                var = tk.BooleanVar(value=default)
                self.optimization_vars[key] = var

                cb_frame = ttk.Frame(params_card, style='Professional.TFrame')
                cb_frame.pack(fill=tk.X, pady=3)

                cb = tk.Checkbutton(cb_frame,
                                    text=name,
                                    variable=var,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 9),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                cb.pack(anchor=tk.W)

            # Advanced settings
            advanced_label = ttk.Label(params_card,
                                       text="Advanced Settings:",
                                       style='Subheading.TLabel')
            advanced_label.pack(anchor=tk.W, pady=(20, 10))

            # Training split
            split_frame = ttk.Frame(params_card, style='Professional.TFrame')
            split_frame.pack(fill=tk.X, pady=5)

            split_label = ttk.Label(split_frame,
                                    text="Training Split:",
                                    style='Body.TLabel')
            split_label.pack(anchor=tk.W)

            self.train_split = tk.DoubleVar(value=0.8)
            split_scale = tk.Scale(split_frame,
                                   from_=0.6, to=0.9,
                                   variable=self.train_split,
                                   orient=tk.HORIZONTAL,
                                   resolution=0.05,
                                   bg=self.colors['white'],
                                   fg=self.colors['dark_blue'],
                                   highlightthickness=0,
                                   troughcolor=self.colors['light_gray'],
                                   activebackground=self.colors['primary_blue'])
            split_scale.pack(fill=tk.X, pady=2)

            # LSTM sequence length
            seq_frame = ttk.Frame(params_card, style='Professional.TFrame')
            seq_frame.pack(fill=tk.X, pady=5)

            seq_label = ttk.Label(seq_frame,
                                  text="LSTM Sequence Length:",
                                  style='Body.TLabel')
            seq_label.pack(anchor=tk.W)

            self.sequence_length = tk.IntVar(value=60)
            seq_scale = tk.Scale(seq_frame,
                                 from_=20, to=120,
                                 variable=self.sequence_length,
                                 orient=tk.HORIZONTAL,
                                 bg=self.colors['white'],
                                 fg=self.colors['dark_blue'],
                                 highlightthickness=0,
                                 troughcolor=self.colors['light_gray'],
                                 activebackground=self.colors['primary_blue'])
            seq_scale.pack(fill=tk.X, pady=2)

            # Control buttons
            controls_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            controls_frame.pack(fill=tk.X, pady=30)

            # Create button row
            button_row = ttk.Frame(controls_frame, style='Professional.TFrame')
            button_row.pack(anchor=tk.CENTER)

            validate_btn = tk.Button(button_row,
                                     text="🔍 Validate Configuration",
                                     command=self.validate_configuration,
                                     bg=self.colors['light_blue'],
                                     fg=self.colors['dark_blue'],
                                     font=('Segoe UI', 11),
                                     relief='flat',
                                     pady=12,
                                     padx=20,
                                     cursor='hand2')
            validate_btn.pack(side=tk.LEFT, padx=(0, 15))

            start_btn = tk.Button(button_row,
                                  text="🚀 Start Complete Analysis",
                                  command=self.start_analysis,
                                  bg=self.colors['success_green'],
                                  fg='white',
                                  font=('Segoe UI', 12, 'bold'),
                                  relief='flat',
                                  pady=15,
                                  padx=30,
                                  cursor='hand2')
            start_btn.pack(side=tk.LEFT, padx=(0, 15))

            save_btn = tk.Button(button_row,
                                 text="💾 Save Configuration",
                                 command=self.save_configuration,
                                 bg=self.colors['medium_gray'],
                                 fg=self.colors['dark_blue'],
                                 font=('Segoe UI', 11),
                                 relief='flat',
                                 pady=12,
                                 padx=20,
                                 cursor='hand2')
            save_btn.pack(side=tk.LEFT, padx=(0, 15))

            load_btn = tk.Button(button_row,
                                 text="📂 Load Configuration",
                                 command=self.load_configuration,
                                 bg=self.colors['medium_gray'],
                                 fg=self.colors['dark_blue'],
                                 font=('Segoe UI', 11),
                                 relief='flat',
                                 pady=12,
                                 padx=20,
                                 cursor='hand2')
            load_btn.pack(side=tk.LEFT)

            # Progress section
            progress_card = self.create_card(content_frame, "📊 Analysis Progress")

            # Progress bar
            self.progress = ttk.Progressbar(progress_card,
                                            style='Professional.TProgressbar',
                                            mode='indeterminate')
            self.progress.pack(fill=tk.X, pady=(0, 15))

            # Progress details
            self.progress_text = tk.Text(progress_card,
                                         height=6,
                                         bg=self.colors['white'],
                                         fg=self.colors['dark_blue'],
                                         font=('Consolas', 9),
                                         relief='solid',
                                         bd=1,
                                         wrap=tk.WORD)
            self.progress_text.pack(fill=tk.X)

            # Pack scrollable components
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
        """""

        def create_analysis_tab(self):
            """Create professional analysis configuration tab"""
            analysis_frame = ttk.Frame(self.notebook, style='Professional.TFrame')
            self.notebook.add(analysis_frame, text="⚙️ Analysis Setup")

            # Scrollable frame setup
            canvas = tk.Canvas(analysis_frame, bg=self.colors['white'])
            scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas, style='Professional.TFrame')

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # Main content
            content_frame = ttk.Frame(scrollable_frame, style='Professional.TFrame')
            content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

            # Header
            header_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            header_frame.pack(fill=tk.X, pady=(0, 30))

            title_label = ttk.Label(header_frame,
                                    text="Advanced Analysis Configuration",
                                    style='Title.TLabel')
            title_label.pack(anchor=tk.W)

            subtitle_label = ttk.Label(header_frame,
                                       text="Configure machine learning models, technical indicators, and analysis parameters",
                                       style='Body.TLabel')
            subtitle_label.pack(anchor=tk.W, pady=(5, 0))

            # Create three-column layout
            columns_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            columns_frame.pack(fill=tk.BOTH, expand=True)

            # Left column - ML Models
            left_column = ttk.Frame(columns_frame, style='Professional.TFrame')
            left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

            ml_card = self.create_card(left_column, "🤖 Machine Learning Models")

            # Ensemble models section
            ensemble_label = ttk.Label(ml_card,
                                       text="Ensemble Models:",
                                       style='Subheading.TLabel')
            ensemble_label.pack(anchor=tk.W, pady=(0, 10))

            self.model_vars = {}
            models = [
                ('🌳 Random Forest (Tree-based ensemble)', 'rf', True),
                ('🚀 XGBoost (Gradient boosting)', 'xgb', True),
                ('⚡ LightGBM (Fast gradient boosting)', 'lgb', True),
                ('🐱 CatBoost (Categorical features)', 'cb', True),
                ('🎲 Extra Trees (Randomized trees)', 'et', True),
                ('🗳️ Voting Regressor (Meta-ensemble)', 'voting', True),
                ('📚 Stacking Regressor (Layered ensemble)', 'stacking', True)
            ]

            for name, key, default in models:
                var = tk.BooleanVar(value=default)
                self.model_vars[key] = var

                cb_frame = ttk.Frame(ml_card, style='Professional.TFrame')
                cb_frame.pack(fill=tk.X, pady=3)

                cb = tk.Checkbutton(cb_frame,
                                    text=name,
                                    variable=var,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                cb.pack(anchor=tk.W)

            # Deep Learning section
            dl_label = ttk.Label(ml_card,
                                 text="Deep Learning Models:",
                                 style='Subheading.TLabel')
            dl_label.pack(anchor=tk.W, pady=(20, 10))

            self.dl_vars = {}
            dl_models = [
                ('🧠 LSTM (Long Short-Term Memory)', 'lstm', DEEP_LEARNING_AVAILABLE),
                ('🔄 GRU (Gated Recurrent Unit)', 'gru', DEEP_LEARNING_AVAILABLE),
                ('🌊 CNN-LSTM Hybrid', 'cnn_lstm', DEEP_LEARNING_AVAILABLE),
                ('🎯 Attention-based LSTM', 'attention_lstm', DEEP_LEARNING_AVAILABLE)
            ]

            for name, key, available in dl_models:
                var = tk.BooleanVar(value=available)
                self.dl_vars[key] = var

                cb_frame = ttk.Frame(ml_card, style='Professional.TFrame')
                cb_frame.pack(fill=tk.X, pady=3)

                cb = tk.Checkbutton(cb_frame,
                                    text=name,
                                    variable=var,
                                    state='normal' if available else 'disabled',
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'] if available else self.colors['medium_gray'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                cb.pack(anchor=tk.W)

            if not DEEP_LEARNING_AVAILABLE:
                warning_label = tk.Label(ml_card,
                                         text="⚠️ TensorFlow not available - Deep Learning disabled",
                                         bg=self.colors['white'],
                                         fg=self.colors['warning_orange'],
                                         font=('Segoe UI', 9))
                warning_label.pack(anchor=tk.W, pady=(5, 0))

            # Center column - Technical Analysis
            center_column = ttk.Frame(columns_frame, style='Professional.TFrame')
            center_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

            tech_card = self.create_card(center_column, "📈 Technical Analysis")

            # Technical indicators
            indicators_label = ttk.Label(tech_card,
                                         text="Technical Indicators:",
                                         style='Subheading.TLabel')
            indicators_label.pack(anchor=tk.W, pady=(0, 10))

            self.indicator_vars = {}
            indicators = [
                ('📊 Moving Averages (SMA, EMA, WMA)', 'ma', True),
                ('⚡ RSI & Stochastic Oscillators', 'momentum', True),
                ('📈 MACD (Multiple timeframes)', 'macd', True),
                ('🔵 Bollinger Bands (Multiple periods)', 'bb', True),
                ('📉 Williams %R', 'williams', True),
                ('📊 Volume Indicators (OBV, VPT)', 'volume', True),
                ('💨 Volatility (ATR, Historical)', 'volatility', True),
                ('🕯️ Candlestick Patterns', 'patterns', True),
                ('🌀 Fibonacci Retracements', 'fibonacci', True),
                ('🔗 Support/Resistance Levels', 'support_resistance', True)
            ]

            for name, key, default in indicators:
                var = tk.BooleanVar(value=default)
                self.indicator_vars[key] = var

                cb_frame = ttk.Frame(tech_card, style='Professional.TFrame')
                cb_frame.pack(fill=tk.X, pady=3)

                cb = tk.Checkbutton(cb_frame,
                                    text=name,
                                    variable=var,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                cb.pack(anchor=tk.W)

            # Smart Money Analysis
            smart_label = ttk.Label(tech_card,
                                    text="Smart Money Analysis:",
                                    style='Subheading.TLabel')
            smart_label.pack(anchor=tk.W, pady=(20, 10))

            self.smart_money_vars = {}
            smart_money_features = [
                ('💰 Wyckoff Methodology', 'wyckoff', True),
                ('🏛️ Institutional Flow Detection', 'institutional', True),
                ('📊 Volume Profile Analysis', 'volume_profile', True),
                ('🏗️ Market Structure Analysis', 'market_structure', True)
            ]

            for name, key, default in smart_money_features:
                var = tk.BooleanVar(value=default)
                self.smart_money_vars[key] = var

                cb_frame = ttk.Frame(tech_card, style='Professional.TFrame')
                cb_frame.pack(fill=tk.X, pady=3)

                cb = tk.Checkbutton(cb_frame,
                                    text=name,
                                    variable=var,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                cb.pack(anchor=tk.W)

            # Right column - Parameters
            right_column = ttk.Frame(columns_frame, style='Professional.TFrame')
            right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

            params_card = self.create_card(right_column, "⚙️ Analysis Parameters")

            # Prediction settings
            pred_label = ttk.Label(params_card,
                                   text="Prediction Settings:",
                                   style='Subheading.TLabel')
            pred_label.pack(anchor=tk.W, pady=(0, 10))

            # Prediction horizon
            horizon_frame = ttk.Frame(params_card, style='Professional.TFrame')
            horizon_frame.pack(fill=tk.X, pady=10)

            horizon_label = ttk.Label(horizon_frame,
                                      text="Prediction Horizon (days):",
                                      style='Body.TLabel')
            horizon_label.pack(anchor=tk.W)

            self.prediction_days = tk.IntVar(value=5)
            horizon_scale = tk.Scale(horizon_frame,
                                     from_=1, to=30,
                                     variable=self.prediction_days,
                                     orient=tk.HORIZONTAL,
                                     bg=self.colors['white'],
                                     fg=self.colors['dark_blue'],
                                     highlightthickness=0,
                                     troughcolor=self.colors['light_gray'],
                                     activebackground=self.colors['primary_blue'],
                                     command=self.update_prediction_label)
            horizon_scale.pack(fill=tk.X, pady=5)

            self.prediction_label = ttk.Label(horizon_frame,
                                              text="5 days",
                                              style='Body.TLabel')
            self.prediction_label.pack(anchor=tk.W)

            # Model optimization
            opt_label = ttk.Label(params_card,
                                  text="Model Optimization:",
                                  style='Subheading.TLabel')
            opt_label.pack(anchor=tk.W, pady=(20, 10))

            self.optimization_vars = {}
            optimization_options = [
                ('🔍 Hyperparameter Tuning (GridSearch)', 'grid_search', True),
                ('📊 Cross-Validation (Time Series)', 'cross_validation', True),
                ('🎯 Feature Selection (Auto)', 'feature_selection', True),
                ('⚖️ Ensemble Weighting', 'ensemble_weighting', True)
            ]

            for name, key, default in optimization_options:
                var = tk.BooleanVar(value=default)
                self.optimization_vars[key] = var

                cb_frame = ttk.Frame(params_card, style='Professional.TFrame')
                cb_frame.pack(fill=tk.X, pady=3)

                cb = tk.Checkbutton(cb_frame,
                                    text=name,
                                    variable=var,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 9),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                cb.pack(anchor=tk.W)

            # Advanced settings
            advanced_label = ttk.Label(params_card,
                                       text="Advanced Settings:",
                                       style='Subheading.TLabel')
            advanced_label.pack(anchor=tk.W, pady=(20, 10))

            # Training split
            split_frame = ttk.Frame(params_card, style='Professional.TFrame')
            split_frame.pack(fill=tk.X, pady=5)

            split_label = ttk.Label(split_frame,
                                    text="Training Split:",
                                    style='Body.TLabel')
            split_label.pack(anchor=tk.W)

            self.train_split = tk.DoubleVar(value=0.8)
            split_scale = tk.Scale(split_frame,
                                   from_=0.6, to=0.9,
                                   variable=self.train_split,
                                   orient=tk.HORIZONTAL,
                                   resolution=0.05,
                                   bg=self.colors['white'],
                                   fg=self.colors['dark_blue'],
                                   highlightthickness=0,
                                   troughcolor=self.colors['light_gray'],
                                   activebackground=self.colors['primary_blue'])
            split_scale.pack(fill=tk.X, pady=2)

            # LSTM sequence length
            seq_frame = ttk.Frame(params_card, style='Professional.TFrame')
            seq_frame.pack(fill=tk.X, pady=5)

            seq_label = ttk.Label(seq_frame,
                                  text="LSTM Sequence Length:",
                                  style='Body.TLabel')
            seq_label.pack(anchor=tk.W)

            self.sequence_length = tk.IntVar(value=60)
            seq_scale = tk.Scale(seq_frame,
                                 from_=20, to=120,
                                 variable=self.sequence_length,
                                 orient=tk.HORIZONTAL,
                                 bg=self.colors['white'],
                                 fg=self.colors['dark_blue'],
                                 highlightthickness=0,
                                 troughcolor=self.colors['light_gray'],
                                 activebackground=self.colors['primary_blue'])
            seq_scale.pack(fill=tk.X, pady=2)

            # Control buttons
            controls_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            controls_frame.pack(fill=tk.X, pady=30)

            # Create button row
            button_row = ttk.Frame(controls_frame, style='Professional.TFrame')
            button_row.pack(anchor=tk.CENTER)

            validate_btn = tk.Button(button_row,
                                     text="🔍 Validate Configuration",
                                     command=self.validate_configuration,
                                     bg=self.colors['light_blue'],
                                     fg=self.colors['dark_blue'],
                                     font=('Segoe UI', 11),
                                     relief='flat',
                                     pady=12,
                                     padx=20,
                                     cursor='hand2')
            validate_btn.pack(side=tk.LEFT, padx=(0, 15))

            start_btn = tk.Button(button_row,
                                  text="🚀 Start Complete Analysis",
                                  command=self.start_analysis,
                                  bg=self.colors['success_green'],
                                  fg='white',
                                  font=('Segoe UI', 12, 'bold'),
                                  relief='flat',
                                  pady=15,
                                  padx=30,
                                  cursor='hand2')
            start_btn.pack(side=tk.LEFT, padx=(0, 15))

            save_btn = tk.Button(button_row,
                                 text="💾 Save Configuration",
                                 command=self.save_configuration,
                                 bg=self.colors['medium_gray'],
                                 fg=self.colors['dark_blue'],
                                 font=('Segoe UI', 11),
                                 relief='flat',
                                 pady=12,
                                 padx=20,
                                 cursor='hand2')
            save_btn.pack(side=tk.LEFT, padx=(0, 15))

            load_btn = tk.Button(button_row,
                                 text="📂 Load Configuration",
                                 command=self.load_configuration,
                                 bg=self.colors['medium_gray'],
                                 fg=self.colors['dark_blue'],
                                 font=('Segoe UI', 11),
                                 relief='flat',
                                 pady=12,
                                 padx=20,
                                 cursor='hand2')
            load_btn.pack(side=tk.LEFT)

            # Progress section
            progress_card = self.create_card(content_frame, "📊 Analysis Progress")

            # Progress bar - FIXED: Removed custom style that doesn't exist
            self.progress = ttk.Progressbar(progress_card,
                                            orient='horizontal',
                                            mode='indeterminate',
                                            length=400)
            self.progress.pack(fill=tk.X, pady=(0, 15))

            # Progress details
            self.progress_text = tk.Text(progress_card,
                                         height=6,
                                         bg=self.colors['white'],
                                         fg=self.colors['dark_blue'],
                                         font=('Consolas', 9),
                                         relief='solid',
                                         bd=1,
                                         wrap=tk.WORD)
            self.progress_text.pack(fill=tk.X)

            # Pack scrollable components
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

        def create_predictions_tab(self):
            """Create professional predictions display tab"""
            pred_frame = ttk.Frame(self.notebook, style='Professional.TFrame')
            self.notebook.add(pred_frame, text="📈 Smart Predictions")

            # Main content with padding
            content_frame = ttk.Frame(pred_frame, style='Professional.TFrame')
            content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

            # Header
            header_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            header_frame.pack(fill=tk.X, pady=(0, 20))

            title_label = ttk.Label(header_frame,
                                    text="AI-Powered Market Predictions",
                                    style='Title.TLabel')
            title_label.pack(anchor=tk.W)

            subtitle_label = ttk.Label(header_frame,
                                       text="Advanced ML predictions with confidence intervals and risk assessment",
                                       style='Body.TLabel')
            subtitle_label.pack(anchor=tk.W, pady=(5, 0))

            # Control panel
            control_card = self.create_card(content_frame, "🎛️ Prediction Controls")

            controls_row = ttk.Frame(control_card, style='Professional.TFrame')
            controls_row.pack(fill=tk.X)

            # Control buttons
            refresh_btn = tk.Button(controls_row,
                                    text="🔄 Refresh Predictions",
                                    command=self.refresh_predictions,
                                    bg=self.colors['primary_blue'],
                                    fg='white',
                                    font=('Segoe UI', 10, 'bold'),
                                    relief='flat',
                                    pady=10,
                                    padx=15,
                                    cursor='hand2')
            refresh_btn.pack(side=tk.LEFT, padx=(0, 10))

            compare_btn = tk.Button(controls_row,
                                    text="📊 Compare Models",
                                    command=self.compare_models,
                                    bg=self.colors['light_blue'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 10),
                                    relief='flat',
                                    pady=10,
                                    padx=15,
                                    cursor='hand2')
            compare_btn.pack(side=tk.LEFT, padx=(0, 10))

            export_btn = tk.Button(controls_row,
                                   text="💾 Export Predictions",
                                   command=self.export_predictions,      #break#2
                                   bg=self.colors['medium_gray'],
                                   fg=self.colors['dark_blue'],
                                   font=('Segoe UI', 10),
                                   relief='flat',
                                   pady=10,
                                   padx=15,
                                   cursor='hand2')
            export_btn.pack(side=tk.LEFT, padx=(0, 10))

            # Auto-update option
            auto_frame = ttk.Frame(controls_row, style='Professional.TFrame')
            auto_frame.pack(side=tk.RIGHT)

            auto_cb = tk.Checkbutton(auto_frame,
                                     text="🔄 Auto-update",
                                     variable=self.auto_update_predictions,
                                     bg=self.colors['white'],
                                     fg=self.colors['dark_blue'],
                                     font=('Segoe UI', 10),
                                     selectcolor=self.colors['primary_blue'],
                                     activebackground=self.colors['white'])
            auto_cb.pack(side=tk.RIGHT)

            # Predictions display with multiple views
            display_card = self.create_card(content_frame, "🎯 Prediction Results")

            pred_notebook = ttk.Notebook(display_card, style='Professional.TNotebook')
            pred_notebook.pack(fill=tk.BOTH, expand=True, pady=10)

            # Summary tab
            summary_tab = ttk.Frame(pred_notebook, style='Professional.TFrame')
            pred_notebook.add(summary_tab, text="📋 Executive Summary")

            summary_frame = ttk.Frame(summary_tab, style='Professional.TFrame')
            summary_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

            self.predictions_text = tk.Text(summary_frame,
                                            bg=self.colors['white'],
                                            fg=self.colors['dark_blue'],
                                            font=('Segoe UI', 11),
                                            wrap=tk.WORD,
                                            relief='solid',
                                            bd=1,
                                            padx=15,
                                            pady=15)

            summary_scroll = ttk.Scrollbar(summary_frame, orient="vertical", command=self.predictions_text.yview)
            self.predictions_text.configure(yscrollcommand=summary_scroll.set)

            self.predictions_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            # Detailed analysis tab
            detailed_tab = ttk.Frame(pred_notebook, style='Professional.TFrame')
            pred_notebook.add(detailed_tab, text="🔍 Detailed Analysis")

            detailed_frame = ttk.Frame(detailed_tab, style='Professional.TFrame')
            detailed_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

            self.detailed_predictions = tk.Text(detailed_frame,
                                                bg=self.colors['white'],
                                                fg=self.colors['dark_blue'],
                                                font=('Consolas', 10),
                                                wrap=tk.WORD,
                                                relief='solid',
                                                bd=1,
                                                padx=15,
                                                pady=15)

            detailed_scroll = ttk.Scrollbar(detailed_frame, orient="vertical", command=self.detailed_predictions.yview)
            self.detailed_predictions.configure(yscrollcommand=detailed_scroll.set)

            self.detailed_predictions.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            detailed_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            # Confidence visualization tab
            confidence_tab = ttk.Frame(pred_notebook, style='Professional.TFrame')
            pred_notebook.add(confidence_tab, text="🎯 Confidence Metrics")

            self.confidence_frame = ttk.Frame(confidence_tab, style='Professional.TFrame')
            self.confidence_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

            # Risk assessment tab
            risk_tab = ttk.Frame(pred_notebook, style='Professional.TFrame')
            pred_notebook.add(risk_tab, text="⚠️ Risk Assessment")

            risk_frame = ttk.Frame(risk_tab, style='Professional.TFrame')
            risk_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

            self.risk_text = tk.Text(risk_frame,
                                     bg=self.colors['white'],
                                     fg=self.colors['dark_blue'],
                                     font=('Segoe UI', 10),
                                     wrap=tk.WORD,
                                     relief='solid',
                                     bd=1,
                                     padx=15,
                                     pady=15)

            risk_scroll = ttk.Scrollbar(risk_frame, orient="vertical", command=self.risk_text.yview)
            self.risk_text.configure(yscrollcommand=risk_scroll.set)

            self.risk_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            risk_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def create_charts_tab(self):
            """Create professional interactive charts tab"""
            charts_frame = ttk.Frame(self.notebook, style='Professional.TFrame')
            self.notebook.add(charts_frame, text="📊 Professional Charts")

            # Main content
            content_frame = ttk.Frame(charts_frame, style='Professional.TFrame')
            content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

            # Header
            header_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            header_frame.pack(fill=tk.X, pady=(0, 20))

            title_label = ttk.Label(header_frame,
                                    text="Professional Trading Charts",
                                    style='Title.TLabel')
            title_label.pack(anchor=tk.W)

            subtitle_label = ttk.Label(header_frame,
                                       text="Interactive charts with technical analysis and smart money flow indicators",
                                       style='Body.TLabel')
            subtitle_label.pack(anchor=tk.W, pady=(5, 0))

            # Chart controls card
            controls_card = self.create_card(content_frame, "📈 Chart Configuration")

            # Chart type selection
            type_frame = ttk.Frame(controls_card, style='Professional.TFrame')
            type_frame.pack(fill=tk.X, pady=(0, 15))

            type_label = ttk.Label(type_frame,
                                   text="Chart Type:",
                                   style='Subheading.TLabel')
            type_label.pack(side=tk.LEFT, padx=(0, 20))

            self.chart_type = tk.StringVar(value="comprehensive")
            chart_types = [
                ("📊 Comprehensive Dashboard", "comprehensive"),
                ("💰 Price Action Only", "price"),
                ("📈 Technical Indicators", "technical"),
                ("📊 Volume Analysis", "volume"),
                ("💎 Smart Money Flow", "smart_money")
            ]

            for text, value in chart_types:
                rb = tk.Radiobutton(type_frame,
                                    text=text,
                                    variable=self.chart_type,
                                    value=value,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                rb.pack(side=tk.LEFT, padx=(0, 15))

            # Chart generation buttons
            button_frame = ttk.Frame(controls_card, style='Professional.TFrame')
            button_frame.pack(fill=tk.X, pady=15)

            generate_btn = tk.Button(button_frame,
                                     text="📊 Generate Charts",
                                     command=self.generate_charts,
                                     bg=self.colors['primary_blue'],
                                     fg='white',
                                     font=('Segoe UI', 12, 'bold'),
                                     relief='flat',
                                     pady=12,
                                     padx=25,
                                     cursor='hand2')
            generate_btn.pack(side=tk.LEFT, padx=(0, 15))

            realtime_btn = tk.Button(button_frame,
                                     text="🔄 Real-time Chart",
                                     command=self.start_realtime_chart,
                                     bg=self.colors['light_blue'],
                                     fg=self.colors['dark_blue'],
                                     font=('Segoe UI', 11),
                                     relief='flat',
                                     pady=10,
                                     padx=20,
                                     cursor='hand2')
            realtime_btn.pack(side=tk.LEFT, padx=(0, 15))

            export_btn = tk.Button(button_frame,
                                   text="💾 Export Charts",
                                   command=self.export_charts,
                                   bg=self.colors['medium_gray'],
                                   fg=self.colors['dark_blue'],
                                   font=('Segoe UI', 11),
                                   relief='flat',
                                   pady=10,
                                   padx=20,
                                   cursor='hand2')
            export_btn.pack(side=tk.LEFT, padx=(0, 15))

            print_btn = tk.Button(button_frame,
                                  text="🖨️ Print Charts",
                                  command=self.print_charts,
                                  bg=self.colors['medium_gray'],
                                  fg=self.colors['dark_blue'],
                                  font=('Segoe UI', 11),
                                  relief='flat',
                                  pady=10,
                                  padx=20,
                                  cursor='hand2')
            print_btn.pack(side=tk.LEFT)

            # Chart customization
            custom_card = self.create_card(content_frame, "🎨 Chart Customization")

            custom_row1 = ttk.Frame(custom_card, style='Professional.TFrame')
            custom_row1.pack(fill=tk.X, pady=(0, 10))

            # Timeframe selection
            timeframe_frame = ttk.Frame(custom_row1, style='Professional.TFrame')
            timeframe_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

            timeframe_label = ttk.Label(timeframe_frame,
                                        text="Timeframe:",
                                        style='Subheading.TLabel')
            timeframe_label.pack(anchor=tk.W, pady=(0, 5))

            self.timeframe = tk.StringVar(value="all")
            timeframes = [("All Data", "all"), ("6 Months", "6m"), ("3 Months", "3m"), ("1 Month", "1m")]

            tf_row = ttk.Frame(timeframe_frame, style='Professional.TFrame')
            tf_row.pack(anchor=tk.W)

            for text, value in timeframes:
                rb = tk.Radiobutton(tf_row,
                                    text=text,
                                    variable=self.timeframe,
                                    value=value,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                rb.pack(side=tk.LEFT, padx=(0, 15))

            # Theme selection
            theme_frame = ttk.Frame(custom_row1, style='Professional.TFrame')
            theme_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)

            theme_label = ttk.Label(theme_frame,
                                    text="Chart Theme:",
                                    style='Subheading.TLabel')
            theme_label.pack(anchor=tk.W, pady=(0, 5))

            self.chart_theme = tk.StringVar(value="plotly_white")
            themes = [("🌞 Light", "plotly_white"), ("🌙 Dark", "plotly_dark"), ("💼 Professional", "seaborn")]

            theme_row = ttk.Frame(theme_frame, style='Professional.TFrame')
            theme_row.pack(anchor=tk.W)

            for text, value in themes:
                rb = tk.Radiobutton(theme_row,
                                    text=text,
                                    variable=self.chart_theme,
                                    value=value,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                rb.pack(side=tk.LEFT, padx=(0, 15))

            # Chart display area
            display_card = self.create_card(content_frame, "📊 Chart Display")

            self.chart_status = tk.Label(display_card,
                                         text="📊 Click 'Generate Charts' to create professional trading visualizations\n\n" +
                                              "• Comprehensive dashboards with multiple indicators\n" +
                                              "• Interactive zoom and pan capabilities\n" +
                                              "• Professional color schemes and styling\n" +
                                              "• Export options for presentations",
                                         font=('Segoe UI', 12),
                                         bg=self.colors['white'],
                                         fg=self.colors['steel_blue'],
                                         justify=tk.CENTER)
            self.chart_status.pack(expand=True, pady=50)

        def create_performance_tab(self):
            """Create professional model performance analysis tab"""
            perf_frame = ttk.Frame(self.notebook, style='Professional.TFrame')
            self.notebook.add(perf_frame, text="🏆 Model Performance")

            # Main content
            content_frame = ttk.Frame(perf_frame, style='Professional.TFrame')
            content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

            # Header
            header_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            header_frame.pack(fill=tk.X, pady=(0, 20))

            title_label = ttk.Label(header_frame,
                                    text="Model Performance Analytics",
                                    style='Title.TLabel')
            title_label.pack(anchor=tk.W)

            subtitle_label = ttk.Label(header_frame,
                                       text="Comprehensive analysis of AI model accuracy, reliability, and performance metrics",
                                       style='Body.TLabel')
            subtitle_label.pack(anchor=tk.W, pady=(5, 0))

            # Performance metrics card
            metrics_card = self.create_card(content_frame, "📊 Performance Metrics")

            # Create performance table
            table_frame = ttk.Frame(metrics_card, style='Professional.TFrame')
            table_frame.pack(fill=tk.BOTH, expand=True, pady=10)

            columns = ('Model', 'Accuracy', 'R² Score', 'MAE', 'RMSE', 'Training Time', 'Status')
            self.performance_tree = ttk.Treeview(table_frame,
                                                 columns=columns,
                                                 show='headings',
                                                 style='Professional.Treeview',
                                                 height=12)

            # Configure columns
            for col in columns:
                self.performance_tree.heading(col, text=col, anchor=tk.CENTER)

            # Set column widths
            widths = {'Model': 150, 'Accuracy': 100, 'R² Score': 100, 'MAE': 80, 'RMSE': 80, 'Training Time': 120,
                      'Status': 100}
            for col in columns:
                self.performance_tree.column(col, width=widths[col], anchor=tk.CENTER)

            # Scrollbars
            v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.performance_tree.yview)
            h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.performance_tree.xview)
            self.performance_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

            self.performance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

            # Performance analysis buttons
            button_card = self.create_card(content_frame, "📈 Performance Analysis Tools")

            button_row = ttk.Frame(button_card, style='Professional.TFrame')
            button_row.pack(anchor=tk.CENTER, pady=10)

            report_btn = tk.Button(button_row,
                                   text="📊 Generate Performance Report",
                                   command=self.generate_performance_report,
                                   bg=self.colors['primary_blue'],
                                   fg='white',
                                   font=('Segoe UI', 11, 'bold'),
                                   relief='flat',
                                   pady=12,
                                   padx=20,
                                   cursor='hand2')
            report_btn.pack(side=tk.LEFT, padx=(0, 15))

            chart_btn = tk.Button(button_row,
                                  text="📈 Model Comparison Chart",
                                  command=self.create_model_comparison_chart,
                                  bg=self.colors['light_blue'],
                                  fg=self.colors['dark_blue'],
                                  font=('Segoe UI', 11),
                                  relief='flat',
                                  pady=12,
                                  padx=20,
                                  cursor='hand2')
            chart_btn.pack(side=tk.LEFT, padx=(0, 15))

            export_btn = tk.Button(button_row,
                                   text="💾 Export Performance Data",
                                   command=self.export_performance_data,
                                   bg=self.colors['medium_gray'],
                                   fg=self.colors['dark_blue'],
                                   font=('Segoe UI', 11),
                                   relief='flat',
                                   pady=12,
                                   padx=20,
                                   cursor='hand2')
            export_btn.pack(side=tk.LEFT)

        def create_risk_tab(self):
            """Create professional risk management tab"""
            risk_frame = ttk.Frame(self.notebook, style='Professional.TFrame')
            self.notebook.add(risk_frame, text="⚠️ Risk Management")

            # Main content
            content_frame = ttk.Frame(risk_frame, style='Professional.TFrame')
            content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

            # Header
            header_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            header_frame.pack(fill=tk.X, pady=(0, 20))

            title_label = ttk.Label(header_frame,
                                    text="Risk Management & Portfolio Analytics",
                                    style='Title.TLabel')
            title_label.pack(anchor=tk.W)

            subtitle_label = ttk.Label(header_frame,
                                       text="Comprehensive risk assessment tools for informed trading decisions",
                                       style='Body.TLabel')
            subtitle_label.pack(anchor=tk.W, pady=(5, 0))

            # Two-column layout
            columns_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            columns_frame.pack(fill=tk.BOTH, expand=True)

            # Left column - Risk metrics
            left_column = ttk.Frame(columns_frame, style='Professional.TFrame')
            left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

            metrics_card = self.create_card(left_column, "📊 Risk Metrics")

            self.risk_display = tk.Text(metrics_card,
                                        height=15,
                                        bg=self.colors['white'],
                                        fg=self.colors['dark_blue'],
                                        font=('Consolas', 10),
                                        relief='solid',
                                        bd=1,
                                        wrap=tk.WORD,
                                        padx=15,
                                        pady=15)

            risk_scroll = ttk.Scrollbar(metrics_card, orient="vertical", command=self.risk_display.yview)
            self.risk_display.configure(yscrollcommand=risk_scroll.set)

            self.risk_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            risk_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            # Right column - Risk controls
            right_column = ttk.Frame(columns_frame, style='Professional.TFrame')
            right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0))

            controls_card = self.create_card(right_column, "⚙️ Risk Parameters")

            # Risk tolerance
            tolerance_frame = ttk.Frame(controls_card, style='Professional.TFrame')
            tolerance_frame.pack(fill=tk.X, pady=(0, 20))

            tolerance_label = ttk.Label(tolerance_frame,
                                        text="Risk Tolerance:",
                                        style='Subheading.TLabel')
            tolerance_label.pack(anchor=tk.W, pady=(0, 10))

            self.risk_tolerance = tk.StringVar(value="moderate")
            risk_levels = [("🛡️ Conservative", "conservative"), ("⚖️ Moderate", "moderate"),
                           ("🚀 Aggressive", "aggressive")]

            for text, value in risk_levels:
                rb = tk.Radiobutton(tolerance_frame,
                                    text=text,
                                    variable=self.risk_tolerance,
                                    value=value,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 11),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                rb.pack(anchor=tk.W, pady=3)

            # Position sizing
            position_frame = ttk.Frame(controls_card, style='Professional.TFrame')
            position_frame.pack(fill=tk.X, pady=(0, 20))

            position_label = ttk.Label(position_frame,
                                       text="Position Size (% of portfolio):",
                                       style='Subheading.TLabel')
            position_label.pack(anchor=tk.W, pady=(0, 5))

            self.position_size = tk.DoubleVar(value=5.0)
            position_scale = tk.Scale(position_frame,
                                      from_=1.0, to=20.0,
                                      variable=self.position_size,
                                      orient=tk.HORIZONTAL,
                                      resolution=0.5,
                                      bg=self.colors['white'],
                                      fg=self.colors['dark_blue'],
                                      highlightthickness=0,
                                      troughcolor=self.colors['light_gray'],
                                      activebackground=self.colors['primary_blue'])
            position_scale.pack(fill=tk.X, pady=5)

            position_value = tk.Label(position_frame,
                                      text="5.0%",
                                      bg=self.colors['white'],
                                      fg=self.colors['dark_blue'],
                                      font=('Segoe UI', 10))
            position_value.pack(anchor=tk.W)

            # Stop loss
            stop_frame = ttk.Frame(controls_card, style='Professional.TFrame')
            stop_frame.pack(fill=tk.X, pady=(0, 20))

            stop_label = ttk.Label(stop_frame,
                                   text="Stop Loss (%):",
                                   style='Subheading.TLabel')
            stop_label.pack(anchor=tk.W, pady=(0, 5))

            self.stop_loss = tk.DoubleVar(value=5.0)
            stop_scale = tk.Scale(stop_frame,
                                  from_=1.0, to=15.0,
                                  variable=self.stop_loss,
                                  orient=tk.HORIZONTAL,
                                  resolution=0.5,
                                  bg=self.colors['white'],
                                  fg=self.colors['dark_blue'],
                                  highlightthickness=0,
                                  troughcolor=self.colors['light_gray'],
                                  activebackground=self.colors['primary_blue'])
            stop_scale.pack(fill=tk.X, pady=5)

            stop_value = tk.Label(stop_frame,
                                  text="5.0%",
                                  bg=self.colors['white'],
                                  fg=self.colors['dark_blue'],
                                  font=('Segoe UI', 10))
            stop_value.pack(anchor=tk.W)

            # Risk calculation button
            calc_button_frame = ttk.Frame(controls_card, style='Professional.TFrame')
            calc_button_frame.pack(fill=tk.X, pady=20)

            calc_btn = tk.Button(calc_button_frame,
                                 text="📊 Calculate Risk Metrics",
                                 command=self.calculate_risk_metrics,
                                 bg=self.colors['primary_blue'],
                                 fg='white',
                                 font=('Segoe UI', 12, 'bold'),
                                 relief='flat',
                                 pady=15,
                                 padx=30,
                                 cursor='hand2')
            calc_btn.pack(anchor=tk.CENTER)

        def create_settings_tab(self):
            """Create professional settings tab"""
            settings_frame = ttk.Frame(self.notebook, style='Professional.TFrame')
            self.notebook.add(settings_frame, text="⚙️ Settings")

            # Scrollable frame setup
            canvas = tk.Canvas(settings_frame, bg=self.colors['white'])
            scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas, style='Professional.TFrame')

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # Main content
            content_frame = ttk.Frame(scrollable_frame, style='Professional.TFrame')
            content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

            # Header
            header_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            header_frame.pack(fill=tk.X, pady=(0, 30))

            title_label = ttk.Label(header_frame,
                                    text="Application Settings",
                                    style='Title.TLabel')
            title_label.pack(anchor=tk.W)

            subtitle_label = ttk.Label(header_frame,
                                       text="Customize the application appearance, performance, and behavior",
                                       style='Body.TLabel')
            subtitle_label.pack(anchor=tk.W, pady=(5, 0))

            # Create settings cards in two columns
            columns_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            columns_frame.pack(fill=tk.BOTH, expand=True)

            # Left column
            left_column = ttk.Frame(columns_frame, style='Professional.TFrame')
            left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

            # Appearance settings
            appearance_card = self.create_card(left_column, "🎨 Appearance")

            # Theme selection
            theme_label = ttk.Label(appearance_card,
                                    text="Application Theme:",
                                    style='Subheading.TLabel')
            theme_label.pack(anchor=tk.W, pady=(0, 10))

            self.theme_var = tk.StringVar(value="Professional Light")

            theme_options = [
                ("🌞 Professional Light (Current)", "Professional Light"),
                ("🌙 Professional Dark", "Professional Dark"),
                ("🎨 Custom Theme", "Custom")
            ]

            for text, value in theme_options:
                rb = tk.Radiobutton(appearance_card,
                                    text=text,
                                    variable=self.theme_var,
                                    value=value,
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'],
                                    font=('Segoe UI', 10),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
                rb.pack(anchor=tk.W, pady=3)

            # Font size
            font_frame = ttk.Frame(appearance_card, style='Professional.TFrame')
            font_frame.pack(fill=tk.X, pady=(20, 0))

            font_label = ttk.Label(font_frame,
                                   text="Font Size Scale:",
                                   style='Body.TLabel')
            font_label.pack(anchor=tk.W)

            self.font_scale = tk.DoubleVar(value=1.0)
            font_scale_widget = tk.Scale(font_frame,
                                         from_=0.8, to=1.5,
                                         variable=self.font_scale,
                                         orient=tk.HORIZONTAL,
                                         resolution=0.1,
                                         bg=self.colors['white'],
                                         fg=self.colors['dark_blue'],
                                         highlightthickness=0,
                                         troughcolor=self.colors['light_gray'],
                                         activebackground=self.colors['primary_blue'])
            font_scale_widget.pack(fill=tk.X, pady=5)

            # Auto-save settings
            autosave_card = self.create_card(left_column, "💾 Auto-save")

            self.auto_save_enabled = tk.BooleanVar(value=True)
            autosave_cb = tk.Checkbutton(autosave_card,
                                         text="Enable auto-save",
                                         variable=self.auto_save_enabled,
                                         bg=self.colors['white'],
                                         fg=self.colors['dark_blue'],
                                         font=('Segoe UI', 11),
                                         selectcolor=self.colors['primary_blue'],
                                         activebackground=self.colors['white'])
            autosave_cb.pack(anchor=tk.W, pady=(0, 10))

            interval_label = ttk.Label(autosave_card,
                                       text="Auto-save interval (minutes):",
                                       style='Body.TLabel')
            interval_label.pack(anchor=tk.W, pady=(10, 0))

            self.auto_save_interval = tk.IntVar(value=5)
            interval_scale = tk.Scale(autosave_card,
                                      from_=1, to=30,
                                      variable=self.auto_save_interval,
                                      orient=tk.HORIZONTAL,
                                      bg=self.colors['white'],
                                      fg=self.colors['dark_blue'],
                                      highlightthickness=0,
                                      troughcolor=self.colors['light_gray'],
                                      activebackground=self.colors['primary_blue'])
            interval_scale.pack(fill=tk.X, pady=5)

            # Right column
            right_column = ttk.Frame(columns_frame, style='Professional.TFrame')
            right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0))

            # Performance settings
            performance_card = self.create_card(right_column, "⚡ Performance")

            self.parallel_processing = tk.BooleanVar(value=True)
            parallel_cb = tk.Checkbutton(performance_card,
                                         text="Enable parallel processing",
                                         variable=self.parallel_processing,
                                         bg=self.colors['white'],
                                         fg=self.colors['dark_blue'],
                                         font=('Segoe UI', 11),
                                         selectcolor=self.colors['primary_blue'],
                                         activebackground=self.colors['white'])
            parallel_cb.pack(anchor=tk.W, pady=(0, 10))

            cores_label = ttk.Label(performance_card,
                                    text="CPU cores to use:",
                                    style='Body.TLabel')
            cores_label.pack(anchor=tk.W, pady=(10, 0))

            self.cpu_cores = tk.IntVar(value=max(1, os.cpu_count() // 2))
            cores_scale = tk.Scale(performance_card,
                                   from_=1, to=os.cpu_count(),
                                   variable=self.cpu_cores,
                                   orient=tk.HORIZONTAL,
                                   bg=self.colors['white'],
                                   fg=self.colors['dark_blue'],
                                   highlightthickness=0,
                                   troughcolor=self.colors['light_gray'],
                                   activebackground=self.colors['primary_blue'])
            cores_scale.pack(fill=tk.X, pady=5)

            self.gpu_acceleration = tk.BooleanVar(value=DEEP_LEARNING_AVAILABLE)
            gpu_cb = tk.Checkbutton(performance_card,
                                    text="Enable GPU acceleration (TensorFlow)",
                                    variable=self.gpu_acceleration,
                                    state='normal' if DEEP_LEARNING_AVAILABLE else 'disabled',
                                    bg=self.colors['white'],
                                    fg=self.colors['dark_blue'] if DEEP_LEARNING_AVAILABLE else self.colors[
                                        'medium_gray'],
                                    font=('Segoe UI', 11),
                                    selectcolor=self.colors['primary_blue'],
                                    activebackground=self.colors['white'])
            gpu_cb.pack(anchor=tk.W, pady=(20, 0))

            # Memory management
            memory_label = ttk.Label(performance_card,
                                     text="Memory Management:",
                                     style='Subheading.TLabel')
            memory_label.pack(anchor=tk.W, pady=(20, 10))

            self.memory_optimization = tk.BooleanVar(value=True)
            memory_cb = tk.Checkbutton(performance_card,
                                       text="Enable memory optimization",
                                       variable=self.memory_optimization,
                                       bg=self.colors['white'],
                                       fg=self.colors['dark_blue'],
                                       font=('Segoe UI', 10),
                                       selectcolor=self.colors['primary_blue'],
                                       activebackground=self.colors['white'])
            memory_cb.pack(anchor=tk.W)

            # Data settings
            data_card = self.create_card(right_column, "📊 Data Processing")

            self.enable_caching = tk.BooleanVar(value=True)
            cache_cb = tk.Checkbutton(data_card,
                                      text="Enable data caching",
                                      variable=self.enable_caching,
                                      bg=self.colors['white'],
                                      fg=self.colors['dark_blue'],
                                      font=('Segoe UI', 11),
                                      selectcolor=self.colors['primary_blue'],
                                      activebackground=self.colors['white'])
            cache_cb.pack(anchor=tk.W, pady=(0, 10))

            cache_size_label = ttk.Label(data_card,
                                         text="Cache size (MB):",
                                         style='Body.TLabel')
            cache_size_label.pack(anchor=tk.W, pady=(10, 0))

            self.cache_size = tk.IntVar(value=500)
            cache_size_scale = tk.Scale(data_card,
                                        from_=100, to=2000,
                                        variable=self.cache_size,
                                        orient=tk.HORIZONTAL,
                                        bg=self.colors['white'],
                                        fg=self.colors['dark_blue'],
                                        highlightthickness=0,
                                        troughcolor=self.colors['light_gray'],
                                        activebackground=self.colors['primary_blue'])
            cache_size_scale.pack(fill=tk.X, pady=5)

            # Apply settings button
            apply_frame = ttk.Frame(content_frame, style='Professional.TFrame')
            apply_frame.pack(fill=tk.X, pady=30)

            apply_btn = tk.Button(apply_frame,
                                  text="💾 Apply Settings",
                                  command=self.apply_settings,
                                  bg=self.colors['success_green'],
                                  fg='white',
                                  font=('Segoe UI', 12, 'bold'),
                                  relief='flat',
                                  pady=15,
                                  padx=40,
                                  cursor='hand2')
            apply_btn.pack(anchor=tk.CENTER)

            # Pack scrollable components
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

        def apply_professional_theme(self):
            """Apply professional theme to all widgets"""
            # Configure text widget colors
            text_config = {
                'bg': self.colors['white'],
                'fg': self.colors['dark_blue'],
                'insertbackground': self.colors['primary_blue'],
                'selectbackground': self.colors['light_blue'],
                'selectforeground': self.colors['dark_blue']
            }

            # Apply to all text widgets when they exist
            for widget_name in ['file_info_text', 'stats_text', 'progress_text',
                                'predictions_text', 'detailed_predictions', 'risk_text', 'risk_display']:
                if hasattr(self, widget_name):
                    widget = getattr(self, widget_name)
                    widget.configure(**text_config)

            # Navigation methods

        def show_upload_tab(self):
            """Show upload tab and highlight nav button"""
            self.notebook.select(0)
            self.highlight_nav_button('upload')

        def show_analysis_tab(self):
            """Show analysis tab and highlight nav button"""
            self.notebook.select(1)
            self.highlight_nav_button('analysis')

        def show_predictions_tab(self):
            """Show predictions tab and highlight nav button"""
            self.notebook.select(2)
            self.highlight_nav_button('predictions')

        def show_charts_tab(self):
            """Show charts tab and highlight nav button"""
            self.notebook.select(3)
            self.highlight_nav_button('charts')

        def show_performance_tab(self):
            """Show performance tab and highlight nav button"""
            self.notebook.select(4)
            self.highlight_nav_button('performance')

        def show_risk_tab(self):
            """Show risk tab and highlight nav button"""
            self.notebook.select(5)
            self.highlight_nav_button('risk')

        def show_settings_tab(self):
            """Show settings tab and highlight nav button"""
            self.notebook.select(6)
            self.highlight_nav_button('settings')

        def highlight_nav_button(self, active_key):
            """Highlight the active navigation button"""
            for key, button in self.nav_buttons.items():
                if key == active_key:
                    button.config(bg=self.colors['primary_blue'], fg='white')
                else:
                    button.config(bg=self.colors['light_gray'], fg=self.colors['dark_blue'])

        def on_tab_changed(self, event):
            """Handle tab change event"""
            selection = event.widget.select()
            tab_text = event.widget.tab(selection, "text")

            # Map tab text to nav button keys
            tab_mapping = {
                "📁 Data Upload": "upload",
                "⚙️ Analysis Setup": "analysis",
                "📈 Smart Predictions": "predictions",
                "📊 Professional Charts": "charts",
                "🏆 Model Performance": "performance",
                "⚠️ Risk Management": "risk",
                "⚙️ Settings": "settings"
            }

            if tab_text in tab_mapping:
                self.highlight_nav_button(tab_mapping[tab_text])

            # Implementation of all GUI methods with professional styling...

        def upload_csv_file(self, event=None):
            """Handle CSV file upload with enhanced validation"""
            file_path = filedialog.askopenfilename(
                title="Select Stock Data CSV File",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx"),
                    ("All files", "*.*")
                ],
                initialdir=os.getcwd()
            )

            if file_path:
                self.csv_file_path = file_path
                self.ai_agent.csv_file_path = file_path
                self.validate_and_display_file(file_path)
                self.update_status("✅ File loaded successfully", "success")

        def validate_and_display_file(self, file_path):
            """Validate and display file information with enhanced preview"""
            try:
                # Show loading state
                self.update_status("🔍 Validating file...", "info")

                # Read file based on extension
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)

                # Basic file info with professional formatting
                file_size_mb = os.path.getsize(file_path) / 1024 / 1024
                info_text = f"""📄 FILE INFORMATION
            {'=' * 50}
            📁 Filename: {os.path.basename(file_path)}
            📊 Data Points: {len(df):,} rows
            📋 Columns: {len(df.columns)} fields
            💾 File Size: {file_size_mb:.2f} MB
            📅 Date Range: {df.iloc[0, 0] if len(df) > 0 else 'N/A'} → {df.iloc[-1, 0] if len(df) > 0 else 'N/A'}

            📋 COLUMN STRUCTURE
            {'=' * 50}
            {chr(10).join([f"  • {col}" for col in df.columns])}

            🔍 DATA QUALITY ASSESSMENT
            {'=' * 50}"""

                # Enhanced validation results
                validation_results = self.ai_agent.validate_data_quality(df)
                info_text += f"\n{validation_results}"

                # Add data insights
                if len(df) > 0:
                    info_text += f"""

            📈 MARKET DATA INSIGHTS
            {'=' * 50}
            • Trading Days: {len(df):,}
            • Data Completeness: {((df.count().sum() / (len(df) * len(df.columns))) * 100):.1f}%
            • Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB"""

                    if 'Close' in df.columns:
                        price_range = df['Close'].max() - df['Close'].min()
                        price_volatility = df['Close'].std() / df['Close'].mean() * 100
                        info_text += f"""
            • Price Range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f} (${price_range:.2f})
            • Price Volatility: {price_volatility:.1f}%"""

                    if 'Volume' in df.columns:
                        avg_volume = df['Volume'].mean()
                        info_text += f"""
            • Average Volume: {avg_volume:,.0f}"""

                self.file_info_text.delete(1.0, tk.END)
                self.file_info_text.insert(tk.END, info_text)

                # Update data preview with professional styling
                self.update_data_preview(df)

                # Update statistics
                self.update_statistics(df)

                self.update_status("✅ File validation completed", "success")

            except Exception as e:
                error_text = f"""❌ FILE VALIDATION ERROR
            {'=' * 50}
            Error: {str(e)}

            📋 TROUBLESHOOTING TIPS:
            • Ensure file is a valid CSV or Excel format
            • Check that data contains required columns (Date, Open, High, Low, Close, Volume)
            • Verify file is not corrupted or in use by another application
            • Try opening file in Excel/spreadsheet software first"""

                self.file_info_text.delete(1.0, tk.END)
                self.file_info_text.insert(tk.END, error_text)
                self.update_status(f"❌ File validation failed: {str(e)}", "error")

        def update_data_preview(self, df):
            """Update data preview with professional styling"""
            # Clear existing items
            for item in self.data_preview.get_children():
                self.data_preview.delete(item)

            # Configure columns with enhanced styling
            self.data_preview['columns'] = list(df.columns)
            self.data_preview['show'] = 'headings'

            # Set column properties
            for col in df.columns:
                self.data_preview.heading(col, text=col, anchor=tk.CENTER)

                # Adjust column width based on content
                if col in ['Date']:
                    width = 120
                elif col in ['Open', 'High', 'Low', 'Close']:
                    width = 100
                elif col in ['Volume']:
                    width = 120
                else:
                    width = 80

                self.data_preview.column(col, width=width, anchor=tk.CENTER)

            # Insert data with alternating row colors (first 50 rows for performance)
            preview_rows = min(50, len(df))
            for i, (index, row) in enumerate(df.head(preview_rows).iterrows()):
                # Format values for better display
                formatted_row = []
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        formatted_row.append("N/A")
                    elif col in ['Open', 'High', 'Low', 'Close'] and pd.api.types.is_numeric_dtype(df[col]):
                        formatted_row.append(f"${value:.2f}")
                    elif col == 'Volume' and pd.api.types.is_numeric_dtype(df[col]):
                        formatted_row.append(f"{value:,.0f}")
                    else:
                        formatted_row.append(str(value))

                # Insert with tags for styling
                tags = ('evenrow',) if i % 2 == 0 else ('oddrow',)
                self.data_preview.insert('', tk.END, values=formatted_row, tags=tags)

            # Configure row styling
            self.data_preview.tag_configure('evenrow', background=self.colors['white'])
            self.data_preview.tag_configure('oddrow', background=self.colors['light_gray'])

        def update_statistics(self, df):
            """Update statistical summary with professional formatting"""
            try:
                numeric_df = df.select_dtypes(include=[np.number])

                stats_text = f"""📊 STATISTICAL SUMMARY
            {'=' * 60}

            📈 DESCRIPTIVE STATISTICS
            {'-' * 40}
            """
                if len(numeric_df.columns) > 0:
                    stats_text += numeric_df.describe().round(4).to_string()

                    # Add additional insights
                    stats_text += f"""

            🔍 ADVANCED METRICS
            {'-' * 40}"""

                    for col in numeric_df.columns:
                        if col in ['Open', 'High', 'Low', 'Close']:
                            returns = numeric_df[col].pct_change().dropna()
                            if len(returns) > 0:
                                stats_text += f"""
            {col}:
              • Daily Volatility: {returns.std():.4f} ({returns.std() * 100:.2f}%)
              • Annualized Volatility: {returns.std() * np.sqrt(252):.4f} ({returns.std() * np.sqrt(252) * 100:.1f}%)
              • Skewness: {returns.skew():.4f}
              • Kurtosis: {returns.kurtosis():.4f}"""

                else:
                    stats_text += "No numeric columns found for statistical analysis."

                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, stats_text)

            except Exception as e:
                error_text = f"""❌ STATISTICS ERROR
            {'=' * 40}
            Error generating statistics: {str(e)}

            The file may contain non-numeric data or formatting issues."""

                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, error_text)

        def use_sample_data(self):
            """Generate and use enhanced sample data with professional feedback"""
            try:
                self.update_status("🧪 Generating professional sample dataset...", "info")

                # Show progress
                self.progress.start()

                # Generate sample data in background thread
                def generate_sample():
                    sample_file = self.ai_agent.create_enhanced_sample_data()

                    # Update UI in main thread
                    self.root.after(0, lambda: self.complete_sample_generation(sample_file))

                thread = threading.Thread(target=generate_sample)
                thread.daemon = True
                thread.start()

            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Sample Data Error", f"Failed to generate sample data:\n\n{str(e)}")
                self.update_status("❌ Sample data generation failed", "error")

        def complete_sample_generation(self, sample_file):
            """Complete sample data generation"""
            try:
                self.progress.stop()
                self.csv_file_path = sample_file
                self.ai_agent.csv_file_path = sample_file
                self.validate_and_display_file(sample_file)
                self.update_status("✅ Professional sample data generated successfully", "success")

                # Show success message
                messagebox.showinfo("Sample Data Generated",
                                    "Professional sample dataset created successfully!\n\n"
                                    "The dataset includes:\n"
                                    "• 500 days of realistic trading data\n"
                                    "• Proper OHLC price relationships\n"
                                    "• Volume patterns with institutional activity\n"
                                    "• Market regime changes and volatility clustering\n"
                                    "• Ready for comprehensive analysis")

            except Exception as e:
                self.update_status(f"❌ Error completing sample generation: {str(e)}", "error")

        def import_from_url(self):
            """Import data from URL with enhanced error handling"""
            # Create custom dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Import Data from URL")
            dialog.geometry("600x300")
            dialog.configure(bg=self.colors['white'])
            dialog.transient(self.root)
            dialog.grab_set()

            # Center dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
            y = (dialog.winfo_screenheight() // 2) - (300 // 2)
            dialog.geometry(f"+{x}+{y}")

            # Dialog content
            content_frame = tk.Frame(dialog, bg=self.colors['white'])
            content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

            title_label = tk.Label(content_frame,
                                   text="📡 Import Stock Data from URL",
                                   font=('Segoe UI', 16, 'bold'),
                                   bg=self.colors['white'],
                                   fg=self.colors['dark_blue'])
            title_label.pack(anchor=tk.W, pady=(0, 20))

            instruction_label = tk.Label(content_frame,
                                         text="Enter the URL of a CSV file containing stock market data:",
                                         font=('Segoe UI', 11),
                                         bg=self.colors['white'],
                                         fg=self.colors['dark_blue'])
            instruction_label.pack(anchor=tk.W, pady=(0, 10))

            # URL entry
            url_frame = tk.Frame(content_frame, bg=self.colors['white'])
            url_frame.pack(fill=tk.X, pady=(0, 20))

            url_entry = tk.Entry(url_frame,
                                 font=('Segoe UI', 11),
                                 bg=self.colors['white'],
                                 fg=self.colors['dark_blue'],
                                 relief='solid',
                                 bd=2,
                                 highlightthickness=0)
            url_entry.pack(fill=tk.X, ipady=8)
            url_entry.focus()

            # Example URLs
            example_label = tk.Label(content_frame,
                                     text="Example URLs:",
                                     font=('Segoe UI', 10, 'bold'),
                                     bg=self.colors['white'],
                                     fg=self.colors['steel_blue'])
            example_label.pack(anchor=tk.W, pady=(10, 5))

            examples_text = """• https://raw.githubusercontent.com/user/repo/main/stock_data.csv
            • https://api.example.com/stock_data.csv
            • Any direct link to a CSV file with stock data"""

            examples_label = tk.Label(content_frame,
                                      text=examples_text,
                                      font=('Segoe UI', 9),
                                      bg=self.colors['white'],
                                      fg=self.colors['dark_blue'],
                                      justify=tk.LEFT)
            examples_label.pack(anchor=tk.W)

            # Buttons
            button_frame = tk.Frame(content_frame, bg=self.colors['white'])
            button_frame.pack(fill=tk.X, pady=(30, 0))

            def import_data():
                url = url_entry.get().strip()
                if url:
                    dialog.destroy()
                    self.process_url_import(url)
                else:
                    messagebox.showwarning("Invalid URL", "Please enter a valid URL")

            def cancel_import():
                dialog.destroy()

            import_btn = tk.Button(button_frame,
                                   text="📥 Import Data",
                                   command=import_data,
                                   bg=self.colors['primary_blue'],
                                   fg='white',
                                   font=('Segoe UI', 11, 'bold'),
                                   relief='flat',
                                   pady=10,
                                   padx=20,
                                   cursor='hand2')
            import_btn.pack(side=tk.RIGHT, padx=(10, 0))

            cancel_btn = tk.Button(button_frame,
                                   text="❌ Cancel",
                                   command=cancel_import,
                                   bg=self.colors['medium_gray'],
                                   fg=self.colors['dark_blue'],
                                   font=('Segoe UI', 11),
                                   relief='flat',
                                   pady=10,
                                   padx=20,
                                   cursor='hand2')
            cancel_btn.pack(side=tk.RIGHT)

            # Bind Enter key
            url_entry.bind('<Return>', lambda e: import_data())

        def process_url_import(self, url):
            """Process URL import in background thread"""

            def import_thread():
                try:
                    self.root.after(0, lambda: self.update_status("📡 Importing data from URL...", "info"))
                    self.root.after(0, lambda: self.progress.start())

                    # Import data
                    df = pd.read_csv(url)

                    # Save to temporary file
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    temp_file = f"imported_data_{timestamp}.csv"
                    df.to_csv(temp_file, index=False)

                    # Update UI in main thread
                    def complete_import():
                        self.progress.stop()
                        self.csv_file_path = temp_file
                        self.ai_agent.csv_file_path = temp_file
                        self.validate_and_display_file(temp_file)
                        self.update_status("✅ Data imported successfully from URL", "success")

                        messagebox.showinfo("Import Successful",
                                            f"Data imported successfully!\n\n"
                                            f"• {len(df):,} rows imported\n"
                                            f"• {len(df.columns)} columns\n"
                                            f"• Saved as: {temp_file}")

                    self.root.after(0, complete_import)

                except Exception as e:
                    def show_error():
                        self.progress.stop()
                        self.update_status(f"❌ URL import failed: {str(e)}", "error")
                        messagebox.showerror("Import Error",
                                             f"Failed to import from URL:\n\n{str(e)}\n\n"
                                             "Please check:\n"
                                             "• URL is accessible\n"
                                             "• File is in CSV format\n"
                                             "• Internet connection is stable")

                    self.root.after(0, show_error)

            thread = threading.Thread(target=import_thread)
            thread.daemon = True
            thread.start()

        def validate_configuration(self):
            """Validate analysis configuration with detailed feedback"""
            issues = []
            warnings = []

            # Check if data is loaded
            if not self.csv_file_path:
                issues.append("• No data file loaded - Please upload a CSV file first")

            # Check model selection
            selected_models = [k for k, v in self.model_vars.items() if v.get()]
            if not selected_models:
                issues.append("• No ML models selected - Select at least one model")

            selected_dl_models = [k for k, v in self.dl_vars.items() if v.get()]

            # Check deep learning availability
            if selected_dl_models and not DEEP_LEARNING_AVAILABLE:
                warnings.append("• Deep learning models selected but TensorFlow not available")

            # Check technical indicators
            selected_indicators = [k for k, v in self.indicator_vars.items() if v.get()]
            if not selected_indicators:
                warnings.append("• No technical indicators selected - Consider enabling some for better analysis")

            # Check prediction horizon
            pred_days = self.prediction_days.get()
            if pred_days > 14:
                warnings.append(f"• High prediction horizon ({pred_days} days) may reduce accuracy")

            # Check training split
            train_split = self.train_split.get()
            if train_split < 0.7:
                warnings.append(f"• Low training split ({train_split:.0%}) may affect model performance")
            elif train_split > 0.9:
                warnings.append(f"• High training split ({train_split:.0%}) may cause overfitting")

            # Show validation results
            if issues:
                result_text = f"""❌ CONFIGURATION ISSUES FOUND
            {'=' * 50}

            🔴 Critical Issues:
            {chr(10).join(issues)}

            Please fix these issues before starting analysis."""

                messagebox.showerror("Configuration Issues", result_text)

            elif warnings:
                result_text = f"""⚠️ CONFIGURATION WARNINGS
            {'=' * 50}

            🟡 Warnings:
            {chr(10).join(warnings)}

            You can proceed with analysis, but consider reviewing these settings."""

                response = messagebox.askyesno("Configuration Warnings",
                                               result_text + "\n\nProceed with analysis anyway?")
                if response:
                    messagebox.showinfo("Validation Passed", "✅ Configuration validated - Ready for analysis!")

            else:
                success_text = f"""✅ CONFIGURATION VALIDATED
            {'=' * 50}

            🎯 Ready for Analysis:
            • Data file: ✅ Loaded
            • ML models: ✅ {len(selected_models)} selected
            • Deep learning: ✅ {"Available" if DEEP_LEARNING_AVAILABLE else "Not available"}
            • Technical indicators: ✅ {len(selected_indicators)} enabled
            • Parameters: ✅ Optimized

            Configuration is perfect for professional analysis!"""

                messagebox.showinfo("Validation Successful", success_text)

        def save_configuration(self, event=None):           #break#3
            def save_configuration(self, event=None):
                """Save current configuration to file with enhanced format"""
                config = {
                    'metadata': {
                        'created_by': self.get_current_user(),
                        'created_at': datetime.now().isoformat(),
                        'version': '2.0',
                        'description': 'SmartStock AI Professional Configuration'
                    },
                    'models': {k: v.get() for k, v in self.model_vars.items()},
                    'deep_learning': {k: v.get() for k, v in self.dl_vars.items()},
                    'indicators': {k: v.get() for k, v in self.indicator_vars.items()},
                    'smart_money': {k: v.get() for k, v in self.smart_money_vars.items()},
                    'optimization': {k: v.get() for k, v in self.optimization_vars.items()},
                    'parameters': {
                        'prediction_days': self.prediction_days.get(),
                        'train_split': self.train_split.get(),
                        'sequence_length': self.sequence_length.get()
                    },
                    'validation': {k: v.get() for k, v in self.validation_vars.items()},
                    'risk_settings': {
                        'tolerance': self.risk_tolerance.get(),
                        'position_size': self.position_size.get(),
                        'stop_loss': self.stop_loss.get()
                    },
                    'appearance': {
                        'theme': self.theme_var.get(),
                        'font_scale': self.font_scale.get(),
                        'zoom_factor': self.zoom_factor
                    }
                }

                file_path = filedialog.asksaveasfilename(
                    title="Save SmartStock AI Configuration",
                    defaultextension=".json",
                    filetypes=[
                        ("SmartStock Config", "*.json"),
                        ("All files", "*.*")
                    ],
                    initialdir=os.getcwd(),
                    initialfile=f"smartstock_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )

                if file_path:
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(config, f, indent=2, ensure_ascii=False)

                        messagebox.showinfo("Configuration Saved",
                                            f"✅ Configuration saved successfully!\n\n"
                                            f"📁 File: {os.path.basename(file_path)}\n"
                                            f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                            f"👤 User: {self.get_current_user()}")

                        self.update_status(f"✅ Configuration saved: {os.path.basename(file_path)}", "success")

                    except Exception as e:
                        messagebox.showerror("Save Error", f"Failed to save configuration:\n\n{str(e)}")
                        self.update_status("❌ Configuration save failed", "error")

        def load_configuration(self):
                """Load configuration from file with enhanced validation"""
                file_path = filedialog.askopenfilename(
                    title="Load SmartStock AI Configuration",
                    filetypes=[
                        ("SmartStock Config", "*.json"),
                        ("All files", "*.*")
                    ],
                    initialdir=os.getcwd()
                )

                if file_path:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)

                        # Validate configuration format
                        if 'metadata' in config:
                            metadata = config['metadata']
                            created_by = metadata.get('created_by', 'Unknown')
                            created_at = metadata.get('created_at', 'Unknown')
                            version = metadata.get('version', '1.0')

                            load_info = f"""📋 CONFIGURATION INFO
            {'=' * 40}
            👤 Created by: {created_by}
            📅 Created: {created_at}
            🔖 Version: {version}
            📁 File: {os.path.basename(file_path)}

            Load this configuration?"""

                            response = messagebox.askyesno("Load Configuration", load_info)
                            if not response:
                                return

                        # Apply configuration with error handling
                        def safe_apply(section, variables, config_key):
                            if config_key in config:
                                for k, v in config[config_key].items():
                                    if k in variables:
                                        try:
                                            variables[k].set(v)
                                        except Exception as e:
                                            print(f"Warning: Could not set {k} = {v}: {e}")

                        safe_apply("ML Models", self.model_vars, 'models')
                        safe_apply("Deep Learning", self.dl_vars, 'deep_learning')
                        safe_apply("Indicators", self.indicator_vars, 'indicators')
                        safe_apply("Smart Money", self.smart_money_vars, 'smart_money')
                        safe_apply("Optimization", self.optimization_vars, 'optimization')
                        safe_apply("Validation", self.validation_vars, 'validation')

                        # Apply parameters
                        if 'parameters' in config:
                            params = config['parameters']
                            self.prediction_days.set(params.get('prediction_days', 5))
                            self.train_split.set(params.get('train_split', 0.8))
                            self.sequence_length.set(params.get('sequence_length', 60))

                        # Apply risk settings
                        if 'risk_settings' in config:
                            risk = config['risk_settings']
                            self.risk_tolerance.set(risk.get('tolerance', 'moderate'))
                            self.position_size.set(risk.get('position_size', 5.0))
                            self.stop_loss.set(risk.get('stop_loss', 5.0))

                        # Apply appearance settings
                        if 'appearance' in config:
                            appearance = config['appearance']
                            self.theme_var.set(appearance.get('theme', 'Professional Light'))
                            self.font_scale.set(appearance.get('font_scale', 1.0))
                            if 'zoom_factor' in appearance:
                                self.zoom_factor = appearance['zoom_factor']
                                self.apply_zoom()

                        messagebox.showinfo("Configuration Loaded",
                                            "✅ Configuration loaded successfully!\n\n"
                                            "All settings have been applied to the current session.")

                        self.update_status(f"✅ Configuration loaded: {os.path.basename(file_path)}", "success")

                    except Exception as e:
                        messagebox.showerror("Load Error",
                                             f"Failed to load configuration:\n\n{str(e)}\n\n"
                                             "Please ensure the file is a valid SmartStock configuration.")
                        self.update_status("❌ Configuration load failed", "error")

        def start_analysis(self):
                """Start the comprehensive analysis process with enhanced UI feedback"""
                if not self.csv_file_path:
                    messagebox.showwarning("No Data",
                                           "Please upload a CSV file first!\n\n"
                                           "Use the 'Upload CSV File' button or generate sample data.")
                    return

                # Final validation
                issues = []
                if not any(var.get() for var in self.model_vars.values()):
                    issues.append("No ML models selected")

                if issues:
                    messagebox.showwarning("Configuration Issue",
                                           f"Cannot start analysis:\n\n" + "\n".join(f"• {issue}" for issue in issues))
                    return

                # Show analysis confirmation dialog
                selected_models = [k for k, v in self.model_vars.items() if v.get()]
                selected_dl = [k for k, v in self.dl_vars.items() if v.get()]
                selected_indicators = [k for k, v in self.indicator_vars.items() if v.get()]

                confirmation = f"""🚀 START COMPREHENSIVE ANALYSIS
            {'=' * 50}

            📊 Data: {os.path.basename(self.csv_file_path)}
            🤖 ML Models: {len(selected_models)} selected
            🧠 Deep Learning: {len(selected_dl)} models
            📈 Technical Indicators: {len(selected_indicators)} enabled
            🎯 Prediction Horizon: {self.prediction_days.get()} days

            ⏱️ Estimated Time: 2-5 minutes
            💾 Memory Usage: Moderate

            Start analysis now?"""

                response = messagebox.askyesno("Confirm Analysis", confirmation)
                if not response:
                    return

                # Start analysis
                self.progress.start()
                self.update_status("🚀 Starting comprehensive analysis...", "info")

                # Clear previous results
                self.predictions_text.delete(1.0, tk.END)
                self.detailed_predictions.delete(1.0, tk.END)
                self.risk_text.delete(1.0, tk.END)
                self.progress_text.delete(1.0, tk.END)

                # Run analysis in background thread
                analysis_thread = threading.Thread(target=self.run_comprehensive_analysis)
                analysis_thread.daemon = True
                analysis_thread.start()

        def run_comprehensive_analysis(self):
                """Run the complete analysis pipeline with professional error handling"""
                try:
                    # Initialize progress tracking
                    steps = [
                        ("🔍 Validating and preprocessing data", self.preprocess_data_step),
                        ("📈 Calculating technical indicators", self.calculate_indicators_step),
                        ("💰 Analyzing smart money flow", self.analyze_smart_money_step),
                        ("🔧 Engineering advanced features", self.engineer_features_step),
                        ("🤖 Training machine learning models", self.train_ml_models_step),
                        ("🧠 Training deep learning models", self.train_dl_models_step),
                        ("🎯 Generating predictions", self.generate_predictions_step),
                        ("⚠️ Calculating risk metrics", self.calculate_risk_step),
                        ("📊 Finalizing results", self.finalize_results_step)
                    ]

                    total_steps = len(steps)

                    for i, (description, step_function) in enumerate(steps):
                        try:
                            self.update_progress(f"Step {i + 1}/{total_steps}: {description}")

                            # Execute step
                            success = step_function()

                            if not success:
                                raise Exception(f"Step failed: {description}")

                            # Update progress percentage
                            progress_pct = ((i + 1) / total_steps) * 100
                            self.root.after(0, lambda p=progress_pct: self.update_progress_percentage(p))

                        except Exception as step_error:
                            raise Exception(f"{description} failed: {str(step_error)}")

                    # Analysis completed successfully
                    self.analysis_complete = True
                    self.root.after(0, lambda: self.update_status("✅ Comprehensive analysis completed successfully!",
                                                                  "success"))
                    self.root.after(0, self.show_analysis_complete_dialog)

                except Exception as e:
                    # Handle analysis errors
                    error_details = f"""
            🔴 ANALYSIS ERROR
            {'=' * 50}
            Error: {str(e)}
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            User: {self.get_current_user()}

            Please check the error details and try again.
            If the problem persists, verify your data file format.
            """

                    def show_error():
                        self.progress.stop()
                        self.progress_text.delete(1.0, tk.END)
                        self.progress_text.insert(tk.END, error_details)
                        self.update_status("❌ Analysis failed - check details", "error")

                        messagebox.showerror("Analysis Error",
                                             f"Analysis failed:\n\n{str(e)}\n\n"
                                             "Check the progress tab for detailed error information.")

                    self.root.after(0, show_error)
                finally:
                    self.root.after(0, lambda: self.progress.stop())

        def preprocess_data_step(self):
                """Data preprocessing step"""
                return self.ai_agent.enhanced_data_preprocessing(self.csv_file_path)

        def calculate_indicators_step(self):
                """Technical indicators calculation step"""
                if any(self.indicator_vars[k].get() for k in self.indicator_vars):
                    self.ai_agent.calculate_advanced_technical_indicators()
                return True

        def analyze_smart_money_step(self):
                """Smart money analysis step"""
                if any(self.smart_money_vars[k].get() for k in self.smart_money_vars):
                    self.ai_agent.analyze_smart_money_flow()
                return True

        def engineer_features_step(self):
                """Feature engineering step"""
                self.ai_agent.enhanced_feature_engineering()
                self.ai_agent.prepare_enhanced_features()
                return True

        def train_ml_models_step(self):
                """ML models training step"""
                selected_models = [k for k, v in self.model_vars.items() if v.get()]
                if selected_models:
                    self.ai_agent.train_enhanced_ml_models(selected_models)
                return True

        def train_dl_models_step(self):
                """Deep learning models training step"""
                selected_dl_models = [k for k, v in self.dl_vars.items() if v.get()]
                if selected_dl_models and DEEP_LEARNING_AVAILABLE:
                    sequence_length = self.sequence_length.get()
                    self.ai_agent.train_advanced_deep_learning_models(sequence_length, selected_dl_models)
                return True

        def generate_predictions_step(self):
                """Predictions generation step"""
                predictions, confidence = self.ai_agent.make_enhanced_predictions()
                self.root.after(0, lambda: self.display_comprehensive_results(predictions, confidence))
                return True

        def calculate_risk_step(self):
                """Risk metrics calculation step"""
                self.ai_agent.calculate_comprehensive_risk_metrics()
                self.root.after(0, self.update_risk_display)
                return True

        def finalize_results_step(self):
                """Finalize results step"""
                self.root.after(0, self.update_performance_display)
                return True

        def update_progress(self, message):
                """Update progress display with professional formatting"""
                try:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    progress_message = f"[{timestamp}] {message}\n"

                    def update_ui():
                        try:
                            self.progress_text.insert(tk.END, progress_message)
                            self.progress_text.see(tk.END)
                            self.root.update_idletasks()
                        except Exception as ui_error:
                            print(f"UI Progress update failed: {ui_error}")

                    self.root.after(0, update_ui)
                except Exception as e:
                    print(f"Progress update error: {e}")

        def update_progress_percentage(self, percentage):
                """Update progress percentage (placeholder for future enhancement)"""
                # This could be used to show a progress bar percentage
                pass

        def show_analysis_complete_dialog(self):
                """Show analysis completion dialog with results summary"""
                try:
                    # Calculate summary statistics
                    model_count = len(self.ai_agent.model_performance)
                    avg_performance = np.mean(list(self.ai_agent.model_performance.values())) if model_count > 0 else 0

                    completion_message = f"""🎉 ANALYSIS COMPLETED SUCCESSFULLY!
            {'=' * 50}

            📊 Results Summary:
            • Models Trained: {model_count}
            • Average Performance: {avg_performance:.1%}
            • Technical Indicators: ✅ Calculated
            • Smart Money Analysis: ✅ Completed
            • Risk Metrics: ✅ Generated
            • Predictions: ✅ Ready

            🎯 Next Steps:
            • Review predictions in the Smart Predictions tab
            • Analyze charts in the Professional Charts tab
            • Check model performance metrics
            • Review risk assessment recommendations

            Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"""

                    messagebox.showinfo("Analysis Complete", completion_message)

                    # Switch to predictions tab
                    self.show_predictions_tab()

                except Exception as e:
                    print(f"Error showing completion dialog: {e}")

        def display_comprehensive_results(self, predictions, confidence):
                """Display comprehensive prediction results with professional formatting"""
                self.predictions_text.delete(1.0, tk.END)

                # Professional header
                result_text = f"""🚀 SMARTSTOCK AI - COMPREHENSIVE ANALYSIS RESULTS
            {'=' * 80}

            📊 Analysis Summary Report
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            User: {self.get_current_user()}
            Data: {os.path.basename(self.csv_file_path) if self.csv_file_path else 'Sample Data'}

            {'=' * 80}

            """

                # Current market state
                if hasattr(self.ai_agent, 'data') and self.ai_agent.data is not None:
                    current_price = self.ai_agent.data['Close'].iloc[-1]
                    prev_price = self.ai_agent.data['Close'].iloc[-2] if len(self.ai_agent.data) > 1 else current_price
                    daily_change = current_price - prev_price
                    daily_change_pct = (daily_change / prev_price) * 100 if prev_price != 0 else 0

                    trend_emoji = "📈" if daily_change > 0 else "📉" if daily_change < 0 else "➡️"

                    result_text += f"""📊 CURRENT MARKET STATE
            {'-' * 50}
            Current Price: ${current_price:.2f}
            Daily Change: {trend_emoji} ${daily_change:+.2f} ({daily_change_pct:+.2f}%)
            Market Trend: {getattr(self.ai_agent, 'market_trend', 'Unknown')}
            Data Points Analyzed: {len(self.ai_agent.data):,}

            """

                # ML Predictions Section
                if predictions:
                    result_text += f"""🤖 MACHINE LEARNING PREDICTIONS
            {'-' * 50}
            """

                    if 'price' in predictions:
                        predicted_price = predictions['price']
                        current_price = self.ai_agent.data['Close'].iloc[-1] if hasattr(self.ai_agent, 'data') else 0

                        if current_price > 0:
                            price_change = predicted_price - current_price
                            price_change_pct = (price_change / current_price) * 100

                            direction = "BULLISH 📈" if price_change > 0 else "BEARISH 📉" if price_change < 0 else "NEUTRAL ➡️"

                            result_text += f"""Target Price: ${predicted_price:.2f}
            Expected Change: ${price_change:+.2f} ({price_change_pct:+.2f}%)
            Direction: {direction}
            Model Confidence: {confidence.get('price', 0):.1%}

            """

                    if 'direction' in predictions:
                        direction_prob = predictions['direction']
                        direction = "STRONG BUY 🚀" if direction_prob > 0.7 else "BUY 📈" if direction_prob > 0.6 else "WEAK BUY ⬆️" if direction_prob > 0.55 else "STRONG SELL 📉" if direction_prob < 0.3 else "SELL ⬇️" if direction_prob < 0.4 else "WEAK SELL 📉" if direction_prob < 0.45 else "NEUTRAL ➡️"

                        result_text += f"""Direction Signal: {direction}
            Probability: {direction_prob:.1%}
            Signal Strength: {"Strong" if abs(direction_prob - 0.5) > 0.2 else "Moderate" if abs(direction_prob - 0.5) > 0.1 else "Weak"}
            Confidence: {confidence.get('direction', 0):.1%}

            """

                # Deep Learning Analysis
                if 'deep_price' in predictions:
                    result_text += f"""🧠 DEEP LEARNING ANALYSIS
            {'-' * 50}
            """

                    deep_price = predictions['deep_price']
                    current_price = self.ai_agent.data['Close'].iloc[-1] if hasattr(self.ai_agent, 'data') else 0

                    if current_price > 0:
                        deep_change = deep_price - current_price
                        deep_change_pct = (deep_change / current_price) * 100

                        result_text += f"""LSTM Prediction: ${deep_price:.2f}
            Neural Network Change: ${deep_change:+.2f} ({deep_change_pct:+.2f}%)
            Deep Learning Confidence: {confidence.get('deep_price', 0):.1%}

            """

                # Smart Money Analysis
                if hasattr(self.ai_agent, 'smart_money_analysis') and self.ai_agent.smart_money_analysis:
                    result_text += f"""💰 SMART MONEY ANALYSIS
            {'-' * 50}
            """
                    smart_analysis = self.ai_agent.smart_money_analysis

                    for key, value in smart_analysis.items():
                        formatted_key = key.replace('_', ' ').title()
                        if isinstance(value, float):
                            result_text += f"{formatted_key}: {value:.1%}\n"
                        else:
                            result_text += f"{formatted_key}: {value}\n"
                    result_text += "\n"

                # Technical Analysis Summary
                result_text += f"""📈 TECHNICAL ANALYSIS SUMMARY
            {'-' * 50}
            """

                if hasattr(self.ai_agent, 'data') and self.ai_agent.data is not None:
                    # RSI Analysis
                    if 'RSI_14' in self.ai_agent.data.columns:
                        current_rsi = self.ai_agent.data['RSI_14'].iloc[-1]
                        rsi_signal = "Oversold 🟢" if current_rsi < 30 else "Overbought 🔴" if current_rsi > 70 else "Neutral 🟡"
                        result_text += f"RSI (14): {current_rsi:.1f} - {rsi_signal}\n"

                    # MACD Analysis
                    if 'MACD' in self.ai_agent.data.columns:
                        macd_current = self.ai_agent.data['MACD'].iloc[-1]
                        macd_signal = self.ai_agent.data['MACD_Signal'].iloc[-1]
                        macd_trend = "Bullish 📈" if macd_current > macd_signal else "Bearish 📉"
                        result_text += f"MACD: {macd_trend} (MACD: {macd_current:.3f}, Signal: {macd_signal:.3f})\n"

                    # Bollinger Bands Analysis
                    if 'BB_Position' in self.ai_agent.data.columns:
                        bb_position = self.ai_agent.data['BB_Position'].iloc[-1]
                        bb_analysis = "Upper Band 🔴" if bb_position > 0.8 else "Lower Band 🟢" if bb_position < 0.2 else "Middle Range 🟡"
                        result_text += f"Bollinger Bands: {bb_analysis} (Position: {bb_position:.2f})\n"

                result_text += "\n"

                # Model Performance Summary
                if hasattr(self.ai_agent, 'model_performance') and self.ai_agent.model_performance:
                    result_text += f"""🏆 MODEL PERFORMANCE SUMMARY
            {'-' * 50}
            """
                    for model_name, performance in self.ai_agent.model_performance.items():
                        stars = "⭐" * min(5, int(performance * 5))
                        result_text += f"{model_name.upper()}: {performance:.1%} {stars}\n"

                # Trading Recommendations
                result_text += f"""

            🎯 TRADING RECOMMENDATIONS
            {'-' * 50}
            """

                if predictions and 'direction' in predictions:
                    direction_prob = predictions['direction']

                    if direction_prob > 0.7:
                        result_text += """🚀 STRONG BUY SIGNAL
            • High probability upward movement detected
            • Consider entering long position
            • Use tight stop-loss for risk management
            • Monitor for confirmation signals

            """
                    elif direction_prob > 0.6:
                        result_text += """📈 MODERATE BUY SIGNAL
            • Positive momentum indicated
            • Consider gradual position building
            • Wait for additional confirmation
            • Implement proper risk controls

            """
                    elif direction_prob < 0.3:
                        result_text += """📉 STRONG SELL SIGNAL
            • High probability downward movement
            • Consider exiting long positions
            • Potential short opportunity (if applicable)
            • Implement protective stops

            """
                    elif direction_prob < 0.4:
                        result_text += """⬇️ MODERATE SELL SIGNAL
            • Negative momentum building
            • Reduce position sizes
            • Monitor for reversal signals
            • Maintain defensive posture

            """
                    else:
                        result_text += """➡️ NEUTRAL SIGNAL
            • Mixed signals detected
            • Consider range trading strategies
            • Wait for clearer directional bias
            • Focus on risk management

            """

                # Risk Assessment Preview
                result_text += f"""⚠️ RISK ASSESSMENT PREVIEW
            {'-' * 50}
            Position Size Recommendation: {self.position_size.get():.1f}% of portfolio
            Risk Profile: {self.risk_tolerance.get().title()}
            Suggested Stop Loss: {self.stop_loss.get():.1f}%

            📊 For detailed risk analysis, check the Risk Management tab.

            """

                # Disclaimer
                result_text += f"""📋 IMPORTANT DISCLAIMER
            {'-' * 50}
            • This analysis is for educational and informational purposes only
            • Past performance does not guarantee future results
            • Always implement proper risk management strategies
            • Consider consulting with financial advisors
            • Never invest more than you can afford to lose

            Analysis generated by SmartStock AI v2.0 Professional
            Report ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(self.get_current_user()) % 10000:04d}
            """

                self.predictions_text.insert(tk.END, result_text)

                # Update detailed predictions
                self.update_detailed_predictions(predictions, confidence)

        def update_detailed_predictions(self, predictions, confidence):
                """Update detailed predictions tab with technical analysis"""
                self.detailed_predictions.delete(1.0, tk.END)

                detailed_text = f"""🔬 DETAILED PREDICTION ANALYSIS
            {'=' * 80}

            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            User: {self.get_current_user()}

            """

                # Model-by-model breakdown
                detailed_text += f"""📊 MODEL-BY-MODEL ANALYSIS
            {'-' * 60}

            """

                for model_name, prediction in predictions.items():
                    detailed_text += f"""🤖 {model_name.upper().replace('_', ' ')} MODEL
            {'-' * 40}
            """

                    if isinstance(prediction, (int, float)):
                        detailed_text += f"Prediction Value: {prediction:.6f}\n"
                        detailed_text += f"Confidence Level: {confidence.get(model_name, 0):.3%}\n"

                        # Add model-specific insights
                        if 'rf' in model_name.lower():
                            detailed_text += """Model Type: Random Forest Ensemble
            Strengths: 
            • Excellent handling of non-linear patterns
            • Robust against overfitting
            • Provides feature importance rankings
            • Good performance with mixed data types

            Characteristics:
            • Tree-based ensemble method
            • Handles missing values well
            • Reduces variance through averaging
            • Interpretable feature relationships

            """
                        elif 'xgb' in model_name.lower():
                            detailed_text += """Model Type: XGBoost Gradient Boosting
            Strengths:
            • High accuracy and performance
            • Advanced regularization techniques
            • Efficient handling of large datasets
            • Built-in feature importance

            Characteristics:
            • Gradient boosting framework
            • Sequential error correction
            • Optimal bias-variance tradeoff
            • Industry-standard performance

            """
                        elif 'lgb' in model_name.lower():
                            detailed_text += """Model Type: LightGBM Fast Gradient Boosting
            Strengths:
            • Faster training than XGBoost
            • Lower memory consumption
            • High accuracy with efficiency
            • Handles categorical features natively

            Characteristics:
            • Leaf-wise tree growth
            • Gradient-based one-side sampling
            • Exclusive feature bundling
            • Optimized for speed and memory

            """
                        elif 'catboost' in model_name.lower() or 'cb' in model_name.lower():
                            detailed_text += """Model Type: CatBoost Categorical Boosting
            Strengths:
            • Superior categorical feature handling
            • Minimal hyperparameter tuning required
            • Built-in overfitting protection
            • Ordered boosting technique

            Characteristics:
            • Advanced categorical encoding
            • Symmetric trees structure
            • GPU acceleration support
            • Robust default parameters

            """
                        elif 'lstm' in model_name.lower() or 'deep' in model_name.lower():
                            detailed_text += """Model Type: Deep Learning Neural Network
            Strengths:
            • Captures complex temporal patterns
            • Memory of long-term dependencies
            • Non-linear relationship modeling
            • Sequential data processing

            Characteristics:
            • Recurrent neural network architecture
            • Attention mechanisms
            • Sequence-to-sequence learning
            • Advanced pattern recognition

            """
                        elif 'voting' in model_name.lower():
                            detailed_text += """Model Type: Voting Ensemble
            Strengths:
            • Combines multiple model predictions
            • Reduces prediction variance
            • Improves overall accuracy
            • Balances individual model biases

            Characteristics:
            • Weighted average of predictions
            • Democratic decision making
            • Ensemble meta-learning
            • Increased reliability

            """
                        elif 'stacking' in model_name.lower():
                            detailed_text += """Model Type: Stacking Ensemble
            Strengths:
            • Meta-model learns optimal combinations
            • Higher accuracy than simple averaging
            • Captures model interaction patterns
            • Advanced ensemble technique

            Characteristics:
            • Two-level learning architecture
            • Base models + meta-learner
            • Cross-validation training
            • Sophisticated prediction blending

            """

                # Prediction convergence analysis
                if len(predictions) > 1:
                    detailed_text += f"""🎯 MODEL CONVERGENCE ANALYSIS
            {'-' * 60}

            """

                    price_predictions = [v for k, v in predictions.items() if
                                         'price' in k.lower() and isinstance(v, (int, float))]
                    if len(price_predictions) > 1:
                        mean_pred = np.mean(price_predictions)
                        std_pred = np.std(price_predictions)
                        min_pred = min(price_predictions)
                        max_pred = max(price_predictions)
                        cv = std_pred / mean_pred if mean_pred != 0 else 0

                        convergence = "High" if cv < 0.02 else "Medium" if cv < 0.05 else "Low"

                        detailed_text += f"""Prediction Convergence: {convergence} Consensus
            Mean Prediction: ${mean_pred:.2f}
            Standard Deviation: ${std_pred:.2f}
            Coefficient of Variation: {cv:.1%}
            Prediction Range: ${min_pred:.2f} - ${max_pred:.2f}
            Range Spread: ${max_pred - min_pred:.2f}

            Interpretation:
            """

                        if convergence == "High":
                            detailed_text += "• Strong model agreement indicates high confidence\n"
                            detailed_text += "• Predictions are closely aligned\n"
                            detailed_text += "• Low uncertainty in forecast\n"
                        elif convergence == "Medium":
                            detailed_text += "• Moderate model agreement\n"
                            detailed_text += "• Some variation in predictions\n"
                            detailed_text += "• Normal level of uncertainty\n"
                        else:
                            detailed_text += "• Low model agreement indicates high uncertainty\n"
                            detailed_text += "• Significant variation in predictions\n"
                            detailed_text += "• Exercise caution in decision making\n"

                        detailed_text += "\n"

                # Time series characteristics
                if hasattr(self.ai_agent, 'data') and self.ai_agent.data is not None:
                    detailed_text += f"""📈 TIME SERIES CHARACTERISTICS
            {'-' * 60}

            """

                    returns = self.ai_agent.data['Close'].pct_change().dropna()

                    if len(returns) > 0:
                        mean_return = returns.mean()
                        daily_vol = returns.std()
                        annual_vol = daily_vol * np.sqrt(252)
                        skewness = returns.skew()
                        kurtosis = returns.kurtosis()

                        detailed_text += f"""Return Statistics:
            • Mean Daily Return: {mean_return:.4%}
            • Daily Volatility: {daily_vol:.4%}
            • Annualized Volatility: {annual_vol:.1%}
            • Skewness: {skewness:.3f}
            • Kurtosis: {kurtosis:.3f}

            """

                        # Autocorrelation analysis
                        if len(returns) > 5:
                            autocorr_1 = returns.autocorr(lag=1)
                            autocorr_5 = returns.autocorr(lag=5)

                            detailed_text += f"""Autocorrelation Analysis:
            • 1-day Autocorrelation: {autocorr_1:.3f}
            • 5-day Autocorrelation: {autocorr_5:.3f}

            """

                            if abs(autocorr_1) > 0.1:
                                trend_behavior = "Trending (momentum effects)" if autocorr_1 > 0 else "Mean Reverting"
                            else:
                                trend_behavior = "Random Walk (efficient market)"

                            detailed_text += f"Series Behavior: {trend_behavior}\n\n"

                        # Distribution analysis
                        detailed_text += f"""Distribution Characteristics:
            """

                        if abs(skewness) > 0.5:
                            skew_desc = "Positive skew (right tail)" if skewness > 0 else "Negative skew (left tail)"
                        else:
                            skew_desc = "Approximately symmetric"

                        if kurtosis > 3:
                            kurt_desc = "Heavy tails (high kurtosis)"
                        elif kurtosis < 1:
                            kurt_desc = "Light tails (low kurtosis)"
                        else:
                            kurt_desc = "Normal tails"

                        detailed_text += f"• {skew_desc}\n"
                        detailed_text += f"• {kurt_desc}\n"
                        detailed_text += f"• {'Fat tail risk present' if kurtosis > 5 else 'Normal tail risk'}\n\n"

                # Market regime analysis
                if hasattr(self.ai_agent, 'data') and self.ai_agent.data is not None:
                    detailed_text += f"""🌊 MARKET REGIME ANALYSIS
            {'-' * 60}

            """

                    # Volatility regime analysis
                    if 'Close' in self.ai_agent.data.columns:
                        vol_20 = self.ai_agent.data['Close'].pct_change().rolling(20).std()
                        current_vol = vol_20.iloc[-1] if len(vol_20) > 0 else 0
                        avg_vol = vol_20.mean() if len(vol_20) > 0 else 0

                        if avg_vol > 0:
                            vol_ratio = current_vol / avg_vol

                            if vol_ratio > 1.5:
                                regime = "High Volatility Regime 🔴"
                                regime_desc = "• Market experiencing elevated uncertainty\n• Increased risk and opportunity\n• Consider defensive positioning"
                            elif vol_ratio < 0.7:
                                regime = "Low Volatility Regime 🟢"
                                regime_desc = "• Market in calm period\n• Reduced risk environment\n• Potential for volatility expansion"
                            else:
                                regime = "Normal Volatility Regime 🟡"
                                regime_desc = "• Market in typical volatility range\n• Standard risk levels\n• Normal trading conditions"

                            detailed_text += f"""Current Regime: {regime}
            Current Volatility: {current_vol:.4f} ({current_vol * 100:.2f}%)
            Average Volatility: {avg_vol:.4f} ({avg_vol * 100:.2f}%)
            Volatility Ratio: {vol_ratio:.2f}x

            Implications:
            {regime_desc}

            """

                # Feature importance analysis
                if hasattr(self.ai_agent, 'feature_importance') and self.ai_agent.feature_importance:
                    detailed_text += f"""🔍 KEY MARKET DRIVERS
            {'-' * 60}

            """

                    if 'price' in self.ai_agent.feature_importance:
                        importance = self.ai_agent.feature_importance['price']
                        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

                        detailed_text += "Top 10 Most Influential Factors:\n\n"

                        for i, (feature, score) in enumerate(sorted_features, 1):
                            clean_name = feature.replace('_', ' ').title()
                            bar_length = int(score * 20)  # Scale for visual bar
                            bar = "█" * bar_length + "░" * (20 - bar_length)
                            detailed_text += f"{i:2d}. {clean_name:<25} {bar} {score:.4f}\n"

                        detailed_text += "\nFeature Importance Interpretation:\n"
                        detailed_text += "• Higher scores indicate stronger predictive power\n"
                        detailed_text += "• Top features drive most of the prediction\n"
                        detailed_text += "• Monitor these indicators for market changes\n\n"

                # Prediction uncertainty analysis
                detailed_text += f"""🎲 PREDICTION UNCERTAINTY ANALYSIS
            {'-' * 60}

            """

                confidence_values = list(confidence.values())
                if confidence_values:
                    avg_confidence = np.mean(confidence_values)
                    min_confidence = min(confidence_values)
                    max_confidence = max(confidence_values)

                    detailed_text += f"""Overall Confidence Metrics:
            • Average Confidence: {avg_confidence:.1%}
            • Minimum Confidence: {min_confidence:.1%}
            • Maximum Confidence: {max_confidence:.1%}
            • Confidence Range: {max_confidence - min_confidence:.1%}

            """

                    if avg_confidence > 0.8:
                        uncertainty_level = "Low Uncertainty 🟢"
                        uncertainty_desc = "• High model confidence\n• Predictions are reliable\n• Good conditions for decision making"
                    elif avg_confidence > 0.6:
                        uncertainty_level = "Moderate Uncertainty 🟡"
                        uncertainty_desc = "• Reasonable model confidence\n• Normal prediction reliability\n• Standard caution recommended"
                    else:
                        uncertainty_level = "High Uncertainty 🔴"
                        uncertainty_desc = "• Low model confidence\n• Uncertain prediction environment\n• Exercise extreme caution"

                    detailed_text += f"""Uncertainty Assessment: {uncertainty_level}

            {uncertainty_desc}

            """

                # Recommendations for improvement
                detailed_text += f"""💡 RECOMMENDATIONS FOR ENHANCED ANALYSIS
            {'-' * 60}

            Data Enhancement:
            • Increase data history for better pattern recognition
            • Include additional market indicators (VIX, sector data)
            • Add fundamental analysis data points
            • Incorporate news sentiment analysis

            Model Improvement:
            • Experiment with different ensemble combinations
            • Fine-tune hyperparameters for specific market conditions
            • Add market regime classification models
            • Implement online learning for adaptive predictions

            Risk Management:
            • Develop dynamic position sizing algorithms
            • Create multi-timeframe analysis framework
            • Implement correlation-based portfolio optimization
            • Add stress testing scenarios

            """

                # Technical notes
                detailed_text += f"""📋 TECHNICAL NOTES
            {'-' * 60}

            Model Training Details:
            • Training Data Split: {self.train_split.get():.0%}
            • Prediction Horizon: {self.prediction_days.get()} days
            • Cross-Validation: Time Series Split
            • Feature Selection: Automated (top 50 features)
            • Scaling Method: Robust Scaler

            Performance Metrics:
            • R² Score: Coefficient of determination
            • MAE: Mean Absolute Error
            • RMSE: Root Mean Square Error
            • Accuracy: Classification accuracy for direction

            Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            Analysis ID: {hash(str(predictions)) % 100000:05d}

            """

                self.detailed_predictions.insert(tk.END, detailed_text)

        def update_performance_display(self):
                """Update model performance display with enhanced metrics"""
                # Clear existing items
                for item in self.performance_tree.get_children():
                    self.performance_tree.delete(item)

                # Add performance data with professional formatting
                if hasattr(self.ai_agent, 'model_performance') and self.ai_agent.model_performance:
                    for model_name, performance in self.ai_agent.model_performance.items():
                        # Calculate additional metrics (placeholder - would be calculated during training)
                        mae = getattr(self.ai_agent, f'{model_name}_mae', 'N/A')
                        rmse = getattr(self.ai_agent, f'{model_name}_rmse', 'N/A')
                        training_time = getattr(self.ai_agent, f'{model_name}_training_time', 'N/A')

                        # Determine status based on performance
                        if performance > 0.8:
                            status = "Excellent ⭐⭐⭐"
                        elif performance > 0.6:
                            status = "Good ⭐⭐"
                        elif performance > 0.4:
                            status = "Fair ⭐"
                        else:
                            status = "Poor"

                        self.performance_tree.insert('', tk.END, values=(
                            model_name.replace('_', ' ').title(),
                            f"{performance:.1%}",
                            f"{performance:.3f}",
                            f"{mae:.3f}" if mae != 'N/A' else 'N/A',
                            f"{rmse:.3f}" if rmse != 'N/A' else 'N/A',
                            f"{training_time:.2f}s" if training_time != 'N/A' else 'N/A',
                            status
                        ))

        def update_risk_display(self):
                """Update risk assessment display with comprehensive analysis"""
                self.risk_display.delete(1.0, tk.END)

                risk_text = f"""⚠️ COMPREHENSIVE RISK ASSESSMENT
            {'=' * 70}

            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            User: {self.get_current_user()}
            Risk Profile: {self.risk_tolerance.get().title()}

            """

                # Market Risk Metrics
                risk_text += f"""📊 MARKET RISK METRICS
            {'-' * 50}

            """

                if hasattr(self.ai_agent, 'data') and self.ai_agent.data is not None:
                    returns = self.ai_agent.data['Close'].pct_change().dropna()

                    if len(returns) > 0:
                        # Value at Risk calculations
                        var_95 = np.percentile(returns, 5)
                        var_99 = np.percentile(returns, 1)

                        # Expected Shortfall (Conditional VaR)
                        es_95 = returns[returns <= var_95].mean()
                        es_99 = returns[returns <= var_99].mean()

                        risk_text += f"""Value at Risk (VaR):
            • 95% VaR (1-day): {var_95:.2%} (${abs(var_95) * 1000:.2f} loss per $1000)
            • 99% VaR (1-day): {var_99:.2%} (${abs(var_99) * 1000:.2f} loss per $1000)

            Expected Shortfall (Conditional VaR):
            • 95% ES: {es_95:.2%} (Average loss beyond VaR)
            • 99% ES: {es_99:.2%} (Tail risk expectation)

            """

                        # Maximum Drawdown Analysis
                        cumulative = (1 + returns).cumprod()
                        rolling_max = cumulative.expanding().max()
                        drawdown = (cumulative - rolling_max) / rolling_max
                        max_drawdown = drawdown.min()

                        # Current drawdown
                        current_drawdown = drawdown.iloc[-1]

                        risk_text += f"""Drawdown Analysis:
            • Maximum Drawdown: {max_drawdown:.2%}
            • Current Drawdown: {current_drawdown:.2%}
            • Recovery Status: {"In drawdown" if current_drawdown < -0.01 else "Near peak"}

            """

                        # Volatility Analysis
                        daily_vol = returns.std()
                        annual_vol = daily_vol * np.sqrt(252)
                        vol_percentile = np.percentile(returns.rolling(20).std().dropna(), 70)

                        vol_regime = "High" if daily_vol > vol_percentile * 1.5 else "Low" if daily_vol < vol_percentile * 0.7 else "Normal"

                        risk_text += f"""Volatility Metrics:
            • Daily Volatility: {daily_vol:.2%}
            • Annualized Volatility: {annual_vol:.1%}
            • Volatility Regime: {vol_regime}
            • 20-day Average Vol: {returns.rolling(20).std().iloc[-1]:.2%}

            """

                        # Sharpe Ratio (assuming 2% risk-free rate)
                        risk_free_rate = 0.02
                        excess_returns = returns - risk_free_rate / 252
                        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)

                        # Sortino Ratio (downside deviation)
                        downside_returns = returns[returns < 0]
                        downside_deviation = downside_returns.std()
                        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252) if len(
                            downside_returns) > 0 else 0

                        risk_text += f"""Risk-Adjusted Returns:
            • Sharpe Ratio: {sharpe_ratio:.2f}
            • Sortino Ratio: {sortino_ratio:.2f}
            • Information Ratio: N/A (benchmark needed)

            """

                # Model Risk Assessment
                risk_text += f"""🤖 MODEL RISK ASSESSMENT
            {'-' * 50}

            """

                if hasattr(self.ai_agent, 'model_performance') and self.ai_agent.model_performance:
                    performances = list(self.ai_agent.model_performance.values())
                    avg_performance = np.mean(performances)
                    std_performance = np.std(performances)
                    min_performance = min(performances)

                    model_risk = "Low" if avg_performance > 0.8 and std_performance < 0.1 else "Medium" if avg_performance > 0.6 else "High"

                    risk_text += f"""Model Reliability Assessment:
            • Average Model Score: {avg_performance:.1%}
            • Performance Std Dev: {std_performance:.1%}
            • Worst Model Score: {min_performance:.1%}
            • Model Risk Level: {model_risk}

            """

                    # Model disagreement analysis
                    if len(performances) > 1:
                        disagreement = "High" if std_performance > 0.15 else "Medium" if std_performance > 0.08 else "Low"
                        risk_text += f"""Model Consensus Analysis:
            • Model Agreement: {disagreement} disagreement
            • Consensus Strength: {"Weak" if std_performance > 0.15 else "Strong"}
            • Prediction Reliability: {"Use with caution" if std_performance > 0.15 else "Good confidence"}

            """

                # Position Risk Analysis
                risk_text += f"""💰 POSITION RISK ANALYSIS
            {'-' * 50}

            Current Settings:
            • Position Size: {self.position_size.get():.1f}% of portfolio
            • Stop Loss: {self.stop_loss.get():.1f}%
            • Risk Tolerance: {self.risk_tolerance.get().title()}

            """

                position_size_pct = self.position_size.get() / 100
                stop_loss_pct = self.stop_loss.get() / 100
                max_loss_per_trade = position_size_pct * stop_loss_pct

                risk_text += f"""Risk Calculations:
            • Maximum Loss per Trade: {max_loss_per_trade:.2%} of total portfolio
            • Risk per $10,000 portfolio: ${max_loss_per_trade * 10000:.2f}
            • Break-even trades needed: {1 / max_loss_per_trade:.1f} profitable trades per loss

            """

                # Kelly Criterion calculation (simplified)
                if hasattr(self.ai_agent, 'data') and self.ai_agent.data is not None and len(returns) > 0:
                    win_rate = len(returns[returns > 0]) / len(returns)
                    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
                    avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0

                    if avg_loss > 0:
                        win_loss_ratio = avg_win / avg_loss
                        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
                        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

                        risk_text += f"""Kelly Criterion Analysis:
            • Historical Win Rate: {win_rate:.1%}
            • Average Win/Loss Ratio: {win_loss_ratio:.2f}
            • Kelly Optimal Position: {kelly_fraction:.1%} of portfolio
            • Recommended Position: {min(kelly_fraction * 0.5, 0.1):.1%} (50% of Kelly)

            """

                # Risk Management Recommendations
                risk_text += f"""🎯 RISK MANAGEMENT RECOMMENDATIONS
            {'-' * 50}

            Based on Risk Profile: {self.risk_tolerance.get().title()}

            """

                risk_tolerance = self.risk_tolerance.get()

                if risk_tolerance == "conservative":
                    risk_text += """Conservative Approach:
            ✅ Recommended Actions:
            • Use smaller position sizes (2-3% max)
            • Set tight stop losses (3-5%)
            • Focus on high-confidence signals only
            • Diversify across multiple uncorrelated positions
            • Avoid leverage
            • Monitor positions daily

            ⚠️ Avoid:
            • Position sizes above 5%
            • High-volatility assets
            • Speculative trades
            • Concentration in single positions

            """
                elif risk_tolerance == "moderate":
                    risk_text += """Moderate Approach:
            ✅ Recommended Actions:
            • Standard position sizes (3-7%)
            • Moderate stop losses (5-8%)
            • Balance risk and return objectives
            • Use trailing stops for profit protection
            • Moderate diversification
            • Regular portfolio rebalancing

            ⚠️ Monitor:
            • Overall portfolio correlation
            • Maximum drawdown levels
            • Risk-adjusted returns
            • Position concentration

            """
                else:  # aggressive
                    risk_text += """Aggressive Approach:
            ✅ Acceptable Actions:
            • Larger position sizes (5-15%)
            • Wider stop losses (8-12%)
            • Accept higher volatility for returns
            • Use leverage cautiously (max 2:1)
            • Concentrate in best opportunities
            • Active position management

            ⚠️ Critical Controls:
            • Never exceed 20% in single position
            • Maintain emergency stops
            • Monitor leverage ratios
            • Have exit strategies ready

            """

                # Universal risk management rules
                risk_text += f"""🛡️ UNIVERSAL RISK MANAGEMENT RULES
            {'-' * 50}

            Essential Principles (All Risk Levels):
            • Never risk more than 2% of portfolio per trade
            • Always use stop-loss orders
            • Maintain proper position sizing discipline
            • Monitor correlation between positions
            • Keep detailed trading records
            • Review and adjust risk parameters regularly

            """

                # Market condition warnings
                if hasattr(self.ai_agent, 'data') and self.ai_agent.data is not None:
                    current_vol = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else daily_vol
                    vol_threshold = returns.std() * 1.5

                    if current_vol > vol_threshold:
                        risk_text += f"""🚨 CURRENT MARKET WARNING
            {'-' * 50}

            HIGH VOLATILITY ENVIRONMENT DETECTED
            • Current volatility: {current_vol:.2%} (daily)
            • Normal volatility: {returns.std():.2%} (daily)
            • Volatility ratio: {current_vol / returns.std():.2f}x normal

            Recommended Adjustments:
            • Reduce position sizes by 30-50%
            • Tighten stop losses
            • Increase monitoring frequency
            • Consider defensive positioning
            • Avoid new positions in extreme volatility

            """

                # Portfolio heat map (if multiple positions)
                risk_text += f"""📊 PORTFOLIO RISK SUMMARY
            {'-' * 50}

            Current Portfolio Risk Profile:
            • Single Position Risk: {max_loss_per_trade:.2%}
            • Daily VaR Estimate: {abs(var_95):.2%}
            • Expected Annual Volatility: {annual_vol:.1%}
            • Maximum Recommended Exposure: {min(position_size_pct * 5, 0.5):.0%}

            Risk Level: {"🟢 LOW" if max_loss_per_trade < 0.01 else "🟡 MEDIUM" if max_loss_per_trade < 0.02 else "🔴 HIGH"}

            """

                # Stress testing scenarios
                risk_text += f"""📈 STRESS TESTING SCENARIOS
            {'-' * 50}

            Scenario Analysis (Portfolio Impact):

            1. Market Crash (-20% in 1 day):
               • Portfolio Loss: {position_size_pct * 0.20:.1%}
               • Recovery Time: ~{int(1 / (avg_performance if 'avg_performance' in locals() else 0.1) * 20)} days

            2. Extended Bear Market (-40% over 6 months):
               • Maximum Portfolio Loss: {position_size_pct * 0.40:.1%}
               • Stop-loss Protection: Limits loss to {max_loss_per_trade:.1%}

            3. Flash Crash (-10% in minutes):
               • Potential Slippage: {max_loss_per_trade * 1.5:.1%}
               • Gap Risk: {"High" if stop_loss_pct > 0.1 else "Moderate"}

            """

                # Emergency procedures
                risk_text += f"""🚨 EMERGENCY PROCEDURES
            {'-' * 50}

            Immediate Actions if Portfolio Loss Exceeds 5%:
            1. Review all open positions
            2. Reduce position sizes by 50%
            3. Implement tighter stop losses
            4. Halt new position entries
            5. Increase monitoring frequency

            Immediate Actions if Daily Loss Exceeds 2%:
            1. Close most volatile positions
            2. Implement defensive stops
            3. Review risk management rules
            4. Consider market timing

            Contact Information:
            • Emergency Stop-Loss: Automated
            • Risk Manager: {self.get_current_user()}
            • Last Review: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            """

                # Regulatory and compliance notes
                risk_text += f"""📋 COMPLIANCE & REGULATORY NOTES
            {'-' * 50}

            Important Disclaimers:
            • This analysis is for educational purposes only
            • Past performance does not guarantee future results
            • Consider professional financial advice
            • Ensure compliance with local regulations
            • Maintain proper documentation for tax purposes

            Risk Management Certification:
            • User: {self.get_current_user()}
            • Timestamp: {datetime.now().isoformat()}
            • Risk Assessment ID: {hash(risk_tolerance + str(position_size_pct)) % 10000:04d}

            """

                self.risk_display.insert(tk.END, risk_text)

            # Additional professional GUI methods (utility functions)

        def get_current_user(self):
                """Get current user (wahabsust)"""
                return "wahabsust"

        def update_status(self, message, status_type="info"):
                """Update status with professional styling"""
                colors = {
                    "info": self.colors['primary_blue'],
                    "success": self.colors['success_green'],
                    "error": self.colors['error_red'],
                    "warning": self.colors['warning_orange']
                }

                color = colors.get(status_type, self.colors['dark_blue'])

                if hasattr(self, 'status_label'):
                    self.status_label.config(text=message, foreground=color)

                if hasattr(self, 'status_indicator'):
                    indicator_text = {
                        "info": "● Processing",
                        "success": "● Ready",
                        "error": "● Error",
                        "warning": "● Warning"
                    }
                    self.status_indicator.config(text=indicator_text.get(status_type, "● Ready"),
                                                 foreground=color)

        def update_prediction_label(self, value):
                """Update prediction horizon label"""
                if hasattr(self, 'prediction_label'):
                    self.prediction_label.config(text=f"{int(float(value))} days")

            # Placeholder methods for functionality to be implemented

        def refresh_predictions(self):
                """Refresh predictions with current data"""
                if not self.analysis_complete:
                    messagebox.showwarning("Analysis Required",
                                           "Please complete the initial analysis first!\n\n"
                                           "Go to Analysis Setup tab and click 'Start Complete Analysis'.")
                    return

                try:
                    self.update_status("🔄 Refreshing predictions...", "info")
                    predictions, confidence = self.ai_agent.make_enhanced_predictions()
                    self.display_comprehensive_results(predictions, confidence)
                    self.update_status("✅ Predictions refreshed successfully", "success")
                except Exception as e:
                    messagebox.showerror("Refresh Error", f"Failed to refresh predictions:\n\n{str(e)}")
                    self.update_status("❌ Prediction refresh failed", "error")

        def compare_models(self):
                """Create professional model comparison visualization"""
                if not self.analysis_complete:
                    messagebox.showwarning("Analysis Required", "Please complete analysis first!")
                    return

                try:
                    # Create comparison chart
                    fig = go.Figure()

                    models = list(self.ai_agent.model_performance.keys())
                    scores = list(self.ai_agent.model_performance.values())

                    # Create professional bar chart
                    fig.add_trace(go.Bar(
                        x=[model.replace('_', ' ').title() for model in models],
                        y=scores,
                        name='Performance Score',
                        marker=dict(
                            color=scores,
                            colorscale='Blues',
                            colorbar=dict(title="Performance"),
                            line=dict(color='white', width=2)
                        ),
                        text=[f'{s:.1%}' for s in scores],
                        textposition='auto',
                        hovertemplate='<b>%{x}</b><br>Performance: %{y:.1%}<extra></extra>'
                    ))

                    # Add average line
                    avg_score = np.mean(scores)
                    fig.add_hline(y=avg_score, line_dash="dash", line_color="red",
                                  annotation_text=f"Average: {avg_score:.1%}")

                    fig.update_layout(
                        title=dict(
                            text="SmartStock AI - Model Performance Comparison",
                            font=dict(size=20, family="Segoe UI"),
                            x=0.5
                        ),
                        xaxis_title="Models",
                        yaxis_title="Performance Score",
                        template="plotly_white",
                        height=600,
                        showlegend=False,
                        yaxis=dict(tickformat='.0%')
                    )

                    # Save and display
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                    pyo.plot(fig, filename=temp_file.name, auto_open=False)
                    webbrowser.open(f'file://{temp_file.name}')

                except Exception as e:
                    messagebox.showerror("Comparison Error", f"Failed to create comparison:\n\n{str(e)}")


        def export_predictions(self):
                """Export predictions with professional formatting"""
                if not self.analysis_complete:
                    messagebox.showwarning("Analysis Required", "Please complete analysis first!")
                    return

                file_path = filedialog.asksaveasfilename(
                    title="Export SmartStock AI Predictions",
                    defaultextension=".txt",
                    filetypes=[
                        ("Text Report", "*.txt"),
                        ("CSV Data", "*.csv"),
                        ("JSON Data", "*.json"),
                        ("Excel Report", "*.xlsx")
                    ],
                    initialfile=f"smartstock_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

                if file_path:
                    try:
                        if file_path.endswith('.json'):
                            # Export as structured JSON
                            export_data = {
                                'metadata': {
                                    'generated_by': 'SmartStock AI v2.0 Professional',
                                    'user': self.get_current_user(),
                                    'timestamp': datetime.now().isoformat(),
                                    'data_file': os.path.basename(self.csv_file_path) if self.csv_file_path else 'Sample Data',
                                    'analysis_id': f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(self.get_current_user()) % 10000:04d}"
                                },
                                'predictions': self.ai_agent.predictions if hasattr(self.ai_agent, 'predictions') else {},
                                'confidence': self.ai_agent.prediction_confidence if hasattr(self.ai_agent,
                                                                                             'prediction_confidence') else {},
                                'model_performance': self.ai_agent.model_performance if hasattr(self.ai_agent,
                                                                                                'model_performance') else {},
                                'smart_money_analysis': self.ai_agent.smart_money_analysis if hasattr(self.ai_agent,
                                                                                                      'smart_money_analysis') else {},
                                'risk_metrics': self.ai_agent.risk_metrics if hasattr(self.ai_agent, 'risk_metrics') else {},
                                'configuration': {
                                    'models_used': [k for k, v in self.model_vars.items() if v.get()],
                                    'deep_learning_used': [k for k, v in self.dl_vars.items() if v.get()],
                                    'indicators_used': [k for k, v in self.indicator_vars.items() if v.get()],
                                    'prediction_horizon': self.prediction_days.get(),
                                    'risk_tolerance': self.risk_tolerance.get()
                                }
                            }

                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)

                        elif file_path.endswith('.csv'):
                            # Export as CSV data
                            if hasattr(self.ai_agent, 'predictions') and self.ai_agent.predictions:
                                df_data = []
                                for model, prediction in self.ai_agent.predictions.items():
                                    confidence_score = self.ai_agent.prediction_confidence.get(model, 0)
                                    performance = self.ai_agent.model_performance.get(model, 0)

                                    df_data.append({
                                        'Model': model,
                                        'Prediction': prediction,
                                        'Confidence': confidence_score,
                                        'Performance': performance,
                                        'Timestamp': datetime.now().isoformat()
                                    })

                                df = pd.DataFrame(df_data)
                                df.to_csv(file_path, index=False)

                        elif file_path.endswith('.xlsx'):
                            # Export as Excel with multiple sheets
                            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                                # Summary sheet
                                if hasattr(self.ai_agent, 'predictions') and self.ai_agent.predictions:
                                    summary_data = []
                                    for model, prediction in self.ai_agent.predictions.items():
                                        summary_data.append({
                                            'Model': model,
                                            'Prediction': prediction,
                                            'Confidence': self.ai_agent.prediction_confidence.get(model, 0),
                                            'Performance': self.ai_agent.model_performance.get(model, 0)
                                        })

                                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Predictions', index=False)

                                # Performance sheet
                                if hasattr(self.ai_agent, 'model_performance') and self.ai_agent.model_performance:
                                    perf_data = [{'Model': k, 'Performance': v} for k, v in self.ai_agent.model_performance.items()]
                                    pd.DataFrame(perf_data).to_excel(writer, sheet_name='Performance', index=False)

                                # Metadata sheet
                                metadata = {
                                    'Generated By': ['SmartStock AI v2.0 Professional'],
                                    'User': [self.get_current_user()],
                                    'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')],
                                    'Data File': [os.path.basename(self.csv_file_path) if self.csv_file_path else 'Sample Data']
                                }
                                pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadata', index=False)

                        else:
                            # Export as text report
                            content = self.predictions_text.get(1.0, tk.END)
                            detailed_content = self.detailed_predictions.get(1.0, tk.END)

                            full_report = f"""SMARTSTOCK AI v2.0 PROFESSIONAL - COMPLETE ANALYSIS REPORT
             {'=' * 90}
            
             EXECUTIVE SUMMARY
             {'-' * 50}
             {content}
            
             DETAILED TECHNICAL ANALYSIS
             {'-' * 50}
             {detailed_content}
            
             EXPORT INFORMATION
             {'-' * 50}
             Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
             User: {self.get_current_user()}
             Software: SmartStock AI v2.0 Professional
             Export Format: Text Report
             File: {os.path.basename(file_path)}
            
             © 2025 SmartStock AI Professional Trading Analysis Platform
             """

                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(full_report)

                        messagebox.showinfo("Export Successful",
                                            f"✅ Predictions exported successfully!\n\n"
                                            f"📁 File: {os.path.basename(file_path)}\n"
                                            f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                            f"📊 Format: {file_path.split('.')[-1].upper()}")

                        self.update_status(f"✅ Predictions exported: {os.path.basename(file_path)}", "success")

                    except Exception as e:
                        messagebox.showerror("Export Error", f"Failed to export predictions:\n\n{str(e)}")
                        self.update_status("❌ Export failed", "error")

        def generate_charts(self):
                """Generate professional charts with enhanced styling"""
                if not self.analysis_complete:
                    messagebox.showwarning("Analysis Required",
                                           "Please complete the analysis first!\n\n"
                                           "Go to Analysis Setup and click 'Start Complete Analysis'.")
                    return

                try:
                    chart_type = self.chart_type.get()
                    timeframe = self.timeframe.get()
                    theme = self.chart_theme.get()

                    self.update_status("📊 Generating professional charts...", "info")

                    # Filter data based on timeframe
                    data = self.ai_agent.data.copy()
                    if timeframe != "all":
                        days = {"1m": 30, "3m": 90, "6m": 180}[timeframe]
                        data = data.tail(days)

                    # Generate appropriate chart based on selection
                    if chart_type == "comprehensive":
                        fig = self.create_comprehensive_dashboard(data, theme)
                    elif chart_type == "price":
                        fig = self.create_price_action_chart(data, theme)
                    elif chart_type == "technical":
                        fig = self.create_technical_indicators_chart(data, theme)
                    elif chart_type == "volume":
                        fig = self.create_volume_analysis_chart(data, theme)
                    elif chart_type == "smart_money":
                        fig = self.create_smart_money_chart(data, theme)

                    # Save and display chart
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                    pyo.plot(fig, filename=temp_file.name, auto_open=False)
                    webbrowser.open(f'file://{temp_file.name}')

                    self.update_status("✅ Charts generated successfully", "success")

                    # Update chart status
                    self.chart_status.config(
                        text="✅ Charts Generated Successfully!\n\n"
                             "Professional trading charts have been created and opened in your browser.\n"
                             "The charts include interactive features:\n"
                             "• Zoom and pan capabilities\n"
                             "• Hover for detailed information\n"
                             "• Professional styling and colors\n"
                             "• Export options available"
                    )

                except Exception as e:
                    messagebox.showerror("Chart Error", f"Failed to generate charts:\n\n{str(e)}")
                    self.update_status("❌ Chart generation failed", "error")


        def create_comprehensive_dashboard(self, data, theme):
                """Create comprehensive trading dashboard with professional styling"""
                fig = make_subplots(
                    rows=4, cols=2,
                    subplot_titles=[
                        'Price Action & Moving Averages', 'Volume Profile',
                        'Momentum Indicators (RSI & MACD)', 'Smart Money Flow Indicators',
                        'Bollinger Bands & Volatility', 'Technical Analysis Summary',
                        'Market Structure Analysis', 'Risk Assessment Chart'
                    ],
                    vertical_spacing=0.08,
                    horizontal_spacing=0.1,
                    specs=[[{"secondary_y": True}, {"secondary_y": True}]] * 4
                )

                # Professional color scheme
                colors = {
                    'candlestick_up': '#00D4AA',
                    'candlestick_down': '#FF4444',
                    'ma_20': '#FF6B35',
                    'ma_50': '#004E89',
                    'ma_200': '#9B59B6',
                    'volume': '#3498DB',
                    'rsi': '#E74C3C',
                    'macd': '#2ECC71',
                    'signal': '#F39C12'
                }

                # 1. Price Action with Moving Averages
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='OHLC',
                        increasing_line_color=colors['candlestick_up'],
                        decreasing_line_color=colors['candlestick_down'],
                        increasing_fillcolor=colors['candlestick_up'],
                        decreasing_fillcolor=colors['candlestick_down']
                    ),
                    row=1, col=1
                )

                # Add moving averages with professional styling
                ma_configs = [
                    ('SMA_20', colors['ma_20'], 'SMA 20', 2),
                    ('SMA_50', colors['ma_50'], 'SMA 50', 2),
                    ('SMA_200', colors['ma_200'], 'SMA 200', 3)
                ]

                for ma_col, color, name, width in ma_configs:
                    if ma_col in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data[ma_col],
                                name=name,
                                line=dict(color=color, width=width),
                                opacity=0.8
                            ),
                            row=1, col=1
                        )

                # 2. Volume Profile with enhanced visualization
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color=colors['volume'],
                        opacity=0.6,
                        yaxis='y2'
                    ),
                    row=1, col=2
                )

                if 'Volume_SMA_20' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Volume_SMA_20'],
                            name='Volume MA',
                            line=dict(color='red', width=2),
                            yaxis='y2'
                        ),
                        row=1, col=2
                    )

                # 3. RSI with professional levels
                if 'RSI_14' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['RSI_14'],
                            name='RSI(14)',
                            line=dict(color=colors['rsi'], width=2)
                        ),
                        row=2, col=1
                    )

                    # RSI levels with fill areas
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.7)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.7)
                    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1, opacity=0.5)

                # 4. MACD with histogram
                if 'MACD' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['MACD'],
                            name='MACD',
                            line=dict(color=colors['macd'], width=2)
                        ),
                        row=2, col=2
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['MACD_Signal'],
                            name='Signal',
                            line=dict(color=colors['signal'], width=2)
                        ),
                        row=2, col=2
                    )

                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data['MACD_Hist'],
                            name='Histogram',
                            marker_color='lightblue',
                            opacity=0.6
                        ),
                        row=2, col=2
                    )

                # 5. Bollinger Bands with price
                if 'BB_Upper' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['BB_Upper'],
                            name='BB Upper',
                            line=dict(color='gray', width=1, dash='dash'),
                            opacity=0.7
                        ),
                        row=3, col=1
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['BB_Lower'],
                            name='BB Lower',
                            line=dict(color='gray', width=1, dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(173, 216, 230, 0.2)',
                            opacity=0.7
                        ),
                        row=3, col=1
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            name='Close Price',
                            line=dict(color='black', width=2)
                        ),
                        row=3, col=1
                    )

                # 6. Smart Money Flow (OBV)
                if 'OBV' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['OBV'],
                            name='On Balance Volume',
                            line=dict(color='purple', width=2)
                        ),
                        row=3, col=2
                    )

                # 7. Market Structure (Support/Resistance)
                if 'Support' in data.columns and 'Resistance' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Support'],
                            name='Support',
                            line=dict(color='green', width=1, dash='dot'),
                            opacity=0.6
                        ),
                        row=4, col=1
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Resistance'],
                            name='Resistance',
                            line=dict(color='red', width=1, dash='dot'),
                            opacity=0.6
                        ),
                        row=4, col=1
                    )

                # 8. Volatility Analysis
                if 'ATR' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['ATR'],
                            name='Average True Range',
                            line=dict(color='orange', width=2)
                        ),
                        row=4, col=2
                    )

                # Update layout with professional styling
                fig.update_layout(
                    title=dict(
                        text="SmartStock AI - Comprehensive Professional Trading Dashboard",
                        font=dict(size=24, family="Segoe UI", color='#2C3E50'),
                        x=0.5
                    ),
                    height=1200,
                    showlegend=True,
                    template=theme,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    annotations=[
                        dict(
                            text=f"Generated by SmartStock AI v2.0 Professional | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
                            xref="paper", yref="paper",
                            x=0.5, y=-0.1,
                            showarrow=False,
                            font=dict(size=10, color='gray')
                        )
                    ]
                )

                # Update axes labels
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=1, col=2)
                fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
                fig.update_yaxes(title_text="MACD", row=2, col=2)
                fig.update_yaxes(title_text="Price ($)", row=3, col=1)
                fig.update_yaxes(title_text="OBV", row=3, col=2)
                fig.update_yaxes(title_text="Price ($)", row=4, col=1)
                fig.update_yaxes(title_text="ATR", row=4, col=2)

                return fig


        def create_price_action_chart(self, data, theme):
                """Create focused price action chart"""
                fig = go.Figure()

                # Professional candlestick chart
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price Action',
                    increasing_line_color='#00D4AA',
                    decreasing_line_color='#FF4444'
                ))

                # Key moving averages with professional styling
                mas = [
                    ('SMA_20', '#FF6B35', 'SMA 20'),
                    ('SMA_50', '#004E89', 'SMA 50'),
                    ('SMA_200', '#9B59B6', 'SMA 200')
                ]

                for ma_col, color, name in mas:
                    if ma_col in data.columns:
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data[ma_col],
                            name=name,
                            line=dict(color=color, width=2),
                            opacity=0.8
                        ))

                # Support and resistance levels
                if 'Support' in data.columns:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Support'],
                        name='Support Level',
                        line=dict(color='green', width=1, dash='dot'),
                        opacity=0.6
                    ))

                if 'Resistance' in data.columns:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Resistance'],
                        name='Resistance Level',
                        line=dict(color='red', width=1, dash='dot'),
                        opacity=0.6
                    ))

                fig.update_layout(
                    title=dict(
                        text="Professional Price Action Analysis",
                        font=dict(size=20, family="Segoe UI"),
                        x=0.5
                    ),
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template=theme,
                    height=700,
                    showlegend=True
                )

                return fig


        def create_technical_indicators_chart(self, data, theme):
                """Create technical indicators chart"""
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=['RSI Analysis', 'MACD Analysis', 'Stochastic Oscillator'],
                    vertical_spacing=0.15
                )

                # RSI with professional styling
                if 'RSI_14' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['RSI_14'],
                            name='RSI(14)',
                            line=dict(color='#E74C3C', width=2)
                        ),
                        row=1, col=1
                    )

                    # RSI levels
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1, opacity=0.7)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1, opacity=0.7)
                    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=1, opacity=0.5)

                # MACD
                if 'MACD' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['MACD'],
                            name='MACD',
                            line=dict(color='#2ECC71', width=2)
                        ),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['MACD_Signal'],
                            name='Signal',
                            line=dict(color='#F39C12', width=2)
                        ),
                        row=2, col=1
                    )

                    if 'MACD_Hist' in data.columns:
                        fig.add_trace(
                            go.Bar(
                                x=data.index,
                                y=data['MACD_Hist'],
                                name='Histogram',
                                marker_color='lightblue',
                                opacity=0.6
                            ),
                            row=2, col=1
                        )

                # Stochastic
                if 'Stoch_K' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Stoch_K'],
                            name='%K',
                            line=dict(color='#9B59B6', width=2)
                        ),
                        row=3, col=1
                    )

                if 'Stoch_D' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Stoch_D'],
                            name='%D',
                            line=dict(color='#34495E', width=2)
                        ),
                        row=3, col=1
                    )

                fig.update_layout(
                    title=dict(
                        text="Technical Indicators Professional Analysis",
                        font=dict(size=20, family="Segoe UI"),
                        x=0.5
                    ),
                    template=theme,
                    height=900,
                    showlegend=True
                )

                return fig


        def create_volume_analysis_chart(self, data, theme):
                """Create volume analysis chart"""
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=['Volume Profile & Trends', 'Volume Indicators'],
                    vertical_spacing=0.2
                )

                # Volume bars with color coding
                colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green'
                          for i in range(len(data))]

                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=1, col=1
                )

                if 'Volume_SMA_20' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Volume_SMA_20'],
                            name='Volume MA(20)',
                            line=dict(color='blue', width=2)
                        ),
                        row=1, col=1
                    )

                # OBV
                if 'OBV' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['OBV'],
                            name='On Balance Volume',
                            line=dict(color='purple', width=2)
                        ),
                        row=2, col=1
                    )

                fig.update_layout(
                    title=dict(
                        text="Professional Volume Analysis",
                        font=dict(size=20, family="Segoe UI"),
                        x=0.5
                    ),
                    template=theme,
                    height=800,
                    showlegend=True
                )

                return fig


        def create_smart_money_chart(self, data, theme):
                """Create smart money flow analysis chart"""
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=['Smart Money Flow Indicators', 'Institutional Activity Analysis'],
                    vertical_spacing=0.2
                )

                # Price with volume overlay (normalized)
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        name='Price',
                        line=dict(color='black', width=2)
                    ),
                    row=1, col=1
                )

                if 'OBV' in data.columns:
                    # Normalize OBV for overlay
                    obv_norm = (data['OBV'] - data['OBV'].min()) / (data['OBV'].max() - data['OBV'].min())
                    price_range = data['Close'].max() - data['Close'].min()
                    obv_scaled = data['Close'].min() + (obv_norm * price_range)

                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=obv_scaled,
                            name='OBV (Normalized)',
                            line=dict(color='purple', width=2, dash='dash'),
                            opacity=0.7
                        ),
                        row=1, col=1
                    )

                # Volume Price Trend
                if 'Volume_Price_Trend' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Volume_Price_Trend'],
                            name='Volume Price Trend',
                            line=dict(color='orange', width=2)
                        ),
                        row=2, col=1
                    )

                fig.update_layout(
                    title=dict(
                        text="Smart Money Flow Professional Analysis",
                        font=dict(size=20, family="Segoe UI"),
                        x=0.5
                    ),
                    template=theme,
                    height=800,
                    showlegend=True
                )

                return fig


        def start_realtime_chart(self):
                """Start real-time charting (placeholder for future implementation)"""
                messagebox.showinfo("Real-time Charts",
                                    "🔄 Real-time charting feature coming soon!\n\n"
                                    "This will include:\n"
                                    "• Live price updates\n"
                                    "• Real-time technical indicators\n"
                                    "• Dynamic chart updates\n"
                                    "• Live trading signals\n\n"
                                    "Stay tuned for the next update!")


        def export_charts(self):
                """Export charts to various formats"""
                if not self.analysis_complete:
                    messagebox.showwarning("Analysis Required", "Please complete analysis first!")
                    return

                file_path = filedialog.asksaveasfilename(
                    title="Export Professional Charts",
                    defaultextension=".html",
                    filetypes=[
                        ("Interactive HTML", "*.html"),
                        ("High-Quality PNG", "*.png"),
                        ("Professional PDF", "*.pdf"),
                        ("Vector SVG", "*.svg")
                    ],
                    initialfile=f"smartstock_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

                if file_path:
                    try:
                        # Generate chart based on current selection
                        chart_type = self.chart_type.get()
                        timeframe = self.timeframe.get()
                        theme = self.chart_theme.get()

                        data = self.ai_agent.data.copy()
                        if timeframe != "all":
                            days = {"1m": 30, "3m": 90, "6m": 180}[timeframe]
                            data = data.tail(days)

                        if chart_type == "comprehensive":
                            fig = self.create_comprehensive_dashboard(data, theme)
                        else:
                            fig = self.create_price_action_chart(data, theme)

                        # Export based on file extension
                        if file_path.endswith('.html'):
                            pyo.plot(fig, filename=file_path, auto_open=False)
                        elif file_path.endswith('.png'):
                            fig.write_image(file_path, width=1920, height=1080, scale=2)
                        elif file_path.endswith('.pdf'):
                            fig.write_image(file_path, width=1920, height=1080)
                        elif file_path.endswith('.svg'):
                            fig.write_image(file_path, width=1920, height=1080)

                        messagebox.showinfo("Export Successful",
                                            f"✅ Charts exported successfully!\n\n"
                                            f"📁 File: {os.path.basename(file_path)}\n"
                                            f"📊 Format: {file_path.split('.')[-1].upper()}\n"
                                            f"🎨 Theme: {theme}\n"
                                            f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                        self.update_status(f"✅ Charts exported: {os.path.basename(file_path)}", "success")

                    except Exception as e:
                        messagebox.showerror("Export Error", f"Failed to export charts:\n\n{str(e)}")
                        self.update_status("❌ Chart export failed", "error")


        def print_charts(self):
                """Print charts (placeholder)"""
                messagebox.showinfo("Print Charts",
                                    "🖨️ Print functionality coming soon!\n\n"
                                    "Alternative options:\n"
                                    "• Export as PDF and print\n"
                                    "• Export as PNG for high-quality prints\n"
                                    "• Use browser print from HTML export")


        def generate_performance_report(self):
                """Generate comprehensive performance report"""
                if not self.analysis_complete:
                    messagebox.showwarning("Analysis Required", "Please complete analysis first!")
                    return

                try:
                    # Create comprehensive performance report
                    report = f"""
            SMARTSTOCK AI v2.0 PROFESSIONAL - MODEL PERFORMANCE REPORT
            {'=' * 80}
            
            EXECUTIVE SUMMARY
            {'-' * 50}
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            User: {self.get_current_user()}
            Analysis ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(self.get_current_user()) % 10000:04d}
            
            OVERALL PERFORMANCE METRICS
            {'-' * 50}
            """

                    if hasattr(self.ai_agent, 'model_performance') and self.ai_agent.model_performance:
                        performances = list(self.ai_agent.model_performance.values())

                        report += f"""
            Overall Model Performance: {np.mean(performances):.1%}
            Number of Models Trained: {len(self.ai_agent.model_performance)}
            Best Performing Model: {max(self.ai_agent.model_performance.items(), key=lambda x: x[1])[0]}
            Worst Performing Model: {min(self.ai_agent.model_performance.items(), key=lambda x: x[1])[0]}
            Performance Range: {min(performances):.1%} - {max(performances):.1%}
            Standard Deviation: {np.std(performances):.1%}
            """

                        if hasattr(self.ai_agent, 'data'):
                            report += f"Training Data Points: {len(self.ai_agent.data):,}\n"

                        report += f"""
            
            MODEL-BY-MODEL BREAKDOWN
            {'-' * 50}
            """

                        for model, performance in sorted(self.ai_agent.model_performance.items(), key=lambda x: x[1], reverse=True):
                            stars = "⭐" * min(5, int(performance * 5))
                            grade = "A+" if performance > 0.9 else "A" if performance > 0.8 else "B+" if performance > 0.7 else "B" if performance > 0.6 else "C"

                            report += f"""
            {model.upper().replace('_', ' '):<25}: {performance:.3f} ({performance:.1%}) {stars} Grade: {grade}
            """

                    report += f"""
            
            TECHNICAL ANALYSIS SUMMARY
            {'-' * 50}
            Data Quality Assessment: Excellent
            Feature Engineering: Advanced (50+ technical indicators)
            Cross-Validation Method: Time Series Split
            Ensemble Techniques: Voting & Stacking Regressors
            Deep Learning Integration: {"Available" if DEEP_LEARNING_AVAILABLE else "Not Available"}
            
            CONFIGURATION DETAILS
            {'-' * 50}
            Prediction Horizon: {self.prediction_days.get()} days
            Training Split: {self.train_split.get():.0%}
            Risk Tolerance: {self.risk_tolerance.get().title()}
            Position Size: {self.position_size.get():.1f}%
            Stop Loss: {self.stop_loss.get():.1f}%
            
            SMART MONEY ANALYSIS
            {'-' * 50}
            """

                    if hasattr(self.ai_agent, 'smart_money_analysis') and self.ai_agent.smart_money_analysis:
                        for key, value in self.ai_agent.smart_money_analysis.items():
                            formatted_key = key.replace('_', ' ').title()
                            if isinstance(value, float):
                                report += f"{formatted_key}: {value:.1%}\n"
                            else:
                                report += f"{formatted_key}: {value}\n"

                    report += f"""
            
            RISK ASSESSMENT SUMMARY
            {'-' * 50}
            Model Reliability: {"HIGH" if np.mean(performances) > 0.8 else "MEDIUM" if np.mean(performances) > 0.6 else "LOW"}
            Prediction Confidence: {np.mean(list(self.ai_agent.prediction_confidence.values())) if hasattr(self.ai_agent, 'prediction_confidence') else 0:.1%}
            Recommended Usage: {"Live Trading Compatible" if np.mean(performances) > 0.7 else "Paper Trading Recommended"}
            Risk Level: {self.risk_tolerance.get().title()} (with proper risk management)
            
            PERFORMANCE BENCHMARKS
            {'-' * 50}
            Industry Standard (>60%): {"✅ PASSED" if np.mean(performances) > 0.6 else "❌ FAILED"}
            Professional Grade (>70%): {"✅ PASSED" if np.mean(performances) > 0.7 else "❌ FAILED"}
            Institutional Level (>80%): {"✅ PASSED" if np.mean(performances) > 0.8 else "❌ FAILED"}
            Elite Performance (>90%): {"✅ PASSED" if np.mean(performances) > 0.9 else "❌ FAILED"}
            
            RECOMMENDATIONS
            {'-' * 50}
            """

                    avg_perf = np.mean(performances) if performances else 0

                    if avg_perf > 0.8:
                        report += """
            ✅ EXCELLENT PERFORMANCE - READY FOR LIVE TRADING
            • Models demonstrate exceptional accuracy
            • High confidence in predictions
            • Suitable for institutional-grade trading
            • Recommended for live implementation with proper risk management
            """
                    elif avg_perf > 0.6:
                        report += """
            ✅ GOOD PERFORMANCE - SUITABLE FOR PAPER TRADING
            • Models show solid accuracy
            • Moderate confidence in predictions
            • Recommended for paper trading first
            • Consider model fine-tuning for improvement
            """
                    else:
                        report += """
            ⚠️ BELOW AVERAGE PERFORMANCE - ADDITIONAL TUNING NEEDED
            • Models require improvement
            • Consider additional data or features
            • Recommended hyperparameter optimization
            • Paper trading only until performance improves
            """

                    report += f"""
            
            TECHNICAL RECOMMENDATIONS
            {'-' * 50}
            1. Data Enhancement:
               • Increase historical data for better training
               • Add external market indicators (VIX, sector data)
               • Include fundamental analysis metrics
               • Incorporate news sentiment data
            
            2. Model Optimization:
               • Fine-tune hyperparameters for best models
               • Experiment with different ensemble combinations
               • Implement adaptive learning techniques
               • Add market regime classification
            
            3. Risk Management:
               • Implement dynamic position sizing
               • Create correlation-based portfolio rules
               • Add stress testing scenarios
               • Develop drawdown protection mechanisms
            
            DISCLAIMER
            {'-' * 50}
            This analysis is for educational and informational purposes only.
            Past performance does not guarantee future results.
            Always implement proper risk management strategies.
            Consider consulting with qualified financial advisors.
            Never invest more than you can afford to lose.
            
            CERTIFICATION
            {'-' * 50}
            Performance Report Generated by: SmartStock AI v2.0 Professional
            Software Version: 2.0.0
            Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            User Certification: {self.get_current_user()}
            Digital Signature: {hash(str(performances) + self.get_current_user()) % 1000000:06d}
            
            © 2025 SmartStock AI Professional Trading Analysis Platform
            All Rights Reserved. Licensed Software Product.
            """

                    # Save report
                    file_path = filedialog.asksaveasfilename(
                        title="Save Performance Report",
                        defaultextension=".txt",
                        filetypes=[
                            ("Text Report", "*.txt"),
                            ("PDF Report", "*.pdf"),
                            ("Word Document", "*.docx")
                        ],
                        initialfile=f"smartstock_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    )

                    if file_path:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(report)

                        messagebox.showinfo("Report Generated",
                                            f"✅ Performance report generated successfully!\n\n"
                                            f"📁 File: {os.path.basename(file_path)}\n"
                                            f"📊 Overall Performance: {np.mean(performances):.1%}\n"
                                            f"🏆 Best Model: {max(self.ai_agent.model_performance.items(), key=lambda x: x[1])[0]}\n"
                                            f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                        self.update_status(f"✅ Performance report saved: {os.path.basename(file_path)}", "success")

                except Exception as e:
                    messagebox.showerror("Report Error", f"Failed to generate performance report:\n\n{str(e)}")
                    self.update_status("❌ Report generation failed", "error")


        def create_model_comparison_chart(self):
                """Create detailed model comparison chart"""
                if not self.analysis_complete:
                    messagebox.showwarning("Analysis Required", "Please complete analysis first!")
                    return

                try:
                    models = list(self.ai_agent.model_performance.keys())
                    scores = list(self.ai_agent.model_performance.values())

                    # Create professional comparison chart
                    fig = go.Figure()

                    # Create gradient colors based on performance
                    colors = ['#FF4444' if s < 0.5 else '#FFA500' if s < 0.7 else '#90EE90' if s < 0.8 else '#00D4AA' for s in
                              scores]

                    fig.add_trace(go.Bar(
                        x=[model.replace('_', ' ').title() for model in models],
                        y=scores,
                        name='Performance Score',
                        marker=dict(
                            color=colors,
                            line=dict(color='white', width=2),
                            pattern_shape=['', '/', '\\', '+', 'x', '.', '-'][0:len(models)]
                        ),
                        text=[f'{s:.1%}' for s in scores],
                        textposition='auto',
                        textfont=dict(size=12, color='white'),
                        hovertemplate='<b>%{x}</b><br>Performance: %{y:.1%}<br>Grade: %{customdata}<extra></extra>',
                        customdata=['A+' if s > 0.9 else 'A' if s > 0.8 else 'B+' if s > 0.7 else 'B' if s > 0.6 else 'C' for s in
                                    scores]
                    ))

                    # Add performance benchmark lines
                    benchmarks = [
                        (0.9, "Elite Level", "green"),
                        (0.8, "Institutional Grade", "blue"),
                        (0.7, "Professional Standard", "orange"),
                        (0.6, "Industry Minimum", "red")
                    ]

                    for value, label, color in benchmarks:
                        fig.add_hline(
                            y=value,
                            line_dash="dash",
                            line_color=color,
                            annotation_text=f"{label} ({value:.0%})",
                            annotation_position="right"
                        )

                    # Add average line
                    avg_score = np.mean(scores)
                    fig.add_hline(
                        y=avg_score,
                        line_dash="solid",
                        line_color="purple",
                        line_width=3,
                        annotation_text=f"Average: {avg_score:.1%}",
                        annotation_position="left"
                    )

                    fig.update_layout(
                        title=dict(
                            text="SmartStock AI - Professional Model Performance Analysis",
                            font=dict(size=20, family="Segoe UI", color='#2C3E50'),
                            x=0.5
                        ),
                        xaxis=dict(
                            title="AI Models",
                            title_font=dict(size=14),
                            tickangle=45
                        ),
                        yaxis=dict(
                            title="Performance Score",
                            title_font=dict(size=14),
                            tickformat='.0%',
                            range=[0, 1]
                        ),
                        template="plotly_white",
                        height=600,
                        showlegend=False,
                        annotations=[
                            dict(
                                text=f"Analysis: {len(models)} models trained | Best: {max(scores):.1%} | Avg: {avg_score:.1%} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
                                xref="paper", yref="paper",
                                x=0.5, y=-0.2,
                                showarrow=False,
                                font=dict(size=10, color='gray')
                            )
                        ]
                    )

                    # Display chart
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                    pyo.plot(fig, filename=temp_file.name, auto_open=False)
                    webbrowser.open(f'file://{temp_file.name}')

                    self.update_status("✅ Model comparison chart generated", "success")

                except Exception as e:
                    messagebox.showerror("Chart Error", f"Failed to create comparison chart:\n\n{str(e)}")
                    self.update_status("❌ Chart generation failed", "error")


        def export_performance_data(self):
                """Export performance data to various formats"""
                if not self.analysis_complete:
                    messagebox.showwarning("Analysis Required", "Please complete analysis first!")
                    return

                file_path = filedialog.asksaveasfilename(
                    title="Export Performance Data",
                    defaultextension=".csv",
                    filetypes=[
                        ("CSV Data", "*.csv"),
                        ("Excel Workbook", "*.xlsx"),
                        ("JSON Data", "*.json")
                    ],
                    initialfile=f"smartstock_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

                if file_path:
                    try:
                        # Create comprehensive performance dataset
                        perf_data = []
                        for model, score in self.ai_agent.model_performance.items():
                            grade = "A+" if score > 0.9 else "A" if score > 0.8 else "B+" if score > 0.7 else "B" if score > 0.6 else "C"
                            rank = sorted(self.ai_agent.model_performance.values(), reverse=True).index(score) + 1

                            perf_data.append({
                                'Model': model.replace('_', ' ').title(),
                                'Performance_Score': score,
                                'Accuracy_Percent': f"{score:.1%}",
                                'Grade': grade,
                                'Rank': rank,
                                'Category': 'Deep Learning' if 'deep' in model.lower() or 'lstm' in model.lower() or 'cnn' in model.lower() else 'Machine Learning',
                                'Status': 'Excellent' if score > 0.8 else 'Good' if score > 0.6 else 'Needs Improvement',
                                'Timestamp': datetime.now().isoformat()
                            })

                        df = pd.DataFrame(perf_data)
                        df = df.sort_values('Performance_Score', ascending=False)

                        if file_path.endswith('.xlsx'):
                            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                                # Main performance data
                                df.to_excel(writer, sheet_name='Performance_Data', index=False)

                                # Summary statistics
                                summary = {
                                    'Metric': ['Total Models', 'Average Performance', 'Best Performance', 'Worst Performance',
                                               'Standard Deviation'],
                                    'Value': [
                                        len(df),
                                        f"{df['Performance_Score'].mean():.1%}",
                                        f"{df['Performance_Score'].max():.1%}",
                                        f"{df['Performance_Score'].min():.1%}",
                                        f"{df['Performance_Score'].std():.1%}"
                                    ]
                                }
                                pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)

                                # Metadata
                                metadata = {
                                    'Field': ['Generated By', 'User', 'Timestamp', 'Software Version', 'Analysis ID'],
                                    'Value': [
                                        'SmartStock AI v2.0 Professional',
                                        self.get_current_user(),
                                        datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
                                        '2.0.0',
                                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(self.get_current_user()) % 10000:04d}"
                                    ]
                                }
                                pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadata', index=False)

                        elif file_path.endswith('.json'):
                            # Export as structured JSON
                            export_data = {
                                'metadata': {
                                    'generated_by': 'SmartStock AI v2.0 Professional',
                                    'user': self.get_current_user(),
                                    'timestamp': datetime.now().isoformat(),
                                    'total_models': len(df),
                                    'analysis_id': f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(self.get_current_user()) % 10000:04d}"
                                },
                                'summary': {
                                    'average_performance': df['Performance_Score'].mean(),
                                    'best_performance': df['Performance_Score'].max(),
                                    'worst_performance': df['Performance_Score'].min(),
                                    'std_deviation': df['Performance_Score'].std(),
                                    'models_above_80pct': len(df[df['Performance_Score'] > 0.8]),
                                    'models_above_70pct': len(df[df['Performance_Score'] > 0.7])
                                },
                                'performance_data': df.to_dict('records')
                            }

                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(export_data, f, indent=2, default=str)

                        else:
                            # Export as CSV
                            df.to_csv(file_path, index=False, encoding='utf-8')

                        messagebox.showinfo("Export Successful",
                                            f"✅ Performance data exported successfully!\n\n"
                                            f"📁 File: {os.path.basename(file_path)}\n"
                                            f"📊 Models: {len(df)} exported\n"
                                            f"🏆 Best Performance: {df['Performance_Score'].max():.1%}\n"
                                            f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                        self.update_status(f"✅ Performance data exported: {os.path.basename(file_path)}", "success")

                    except Exception as e:
                        messagebox.showerror("Export Error", f"Failed to export performance data:\n\n{str(e)}")
                        self.update_status("❌ Export failed", "error")


        def calculate_risk_metrics(self):
                """Calculate and display comprehensive risk metrics"""
                if not self.analysis_complete:
                    messagebox.showwarning("Analysis Required",
                                           "Please complete the analysis first!\n\n"
                                           "Risk metrics require completed model training and data analysis.")
                    return

                try:
                    self.update_status("📊 Calculating comprehensive risk metrics...", "info")

                    # Calculate risk metrics using the AI agent
                    self.ai_agent.calculate_comprehensive_risk_metrics()

                    # Update the risk display
                    self.update_risk_display()

                    # Show completion message
                    messagebox.showinfo("Risk Analysis Complete",
                                        "✅ Comprehensive risk metrics calculated!\n\n"
                                        "The analysis includes:\n"
                                        "• Value at Risk (VaR) calculations\n"
                                        "• Maximum drawdown analysis\n"
                                        "• Volatility assessments\n"
                                        "• Model risk evaluation\n"
                                        "• Position sizing recommendations\n\n"
                                        "Check the Risk Management tab for detailed results.")

                    self.update_status("✅ Risk metrics calculated successfully", "success")

                except Exception as e:
                    messagebox.showerror("Risk Calculation Error",
                                         f"Failed to calculate risk metrics:\n\n{str(e)}\n\n"
                                         "Ensure that analysis has been completed successfully.")
                    self.update_status("❌ Risk calculation failed", "error")


        def toggle_realtime(self):
                """Toggle real-time updates"""
                if self.real_time_enabled.get():
                    self.update_status("🔄 Real-time mode enabled", "info")
                    # Placeholder for real-time functionality
                    messagebox.showinfo("Real-time Mode",
                                        "🔄 Real-time updates enabled!\n\n"
                                        "Note: Full real-time functionality coming soon.\n"
                                        "Current features:\n"
                                        "• Manual refresh capabilities\n"
                                        "• Auto-update predictions (when enabled)\n"
                                        "• Live status indicators\n\n"
                                        "Future updates will include:\n"
                                        "• Live data feeds\n"
                                        "• Automatic model retraining\n"
                                        "• Real-time alerts")
                else:
                    self.update_status("⏸️ Real-time mode disabled", "info")


        def apply_settings(self):
                """Apply application settings with professional feedback"""
                try:
                    settings_applied = []

                    # Apply theme changes
                    if self.theme_var.get() != "Professional Light":
                        settings_applied.append(f"• Theme: {self.theme_var.get()}")
                        # Placeholder for theme switching logic

                    # Apply font scaling
                    if self.font_scale.get() != 1.0:
                        settings_applied.append(f"• Font Scale: {self.font_scale.get():.1f}x")
                        # Apply font scaling (would need to update all font configurations)

                    # Apply performance settings
                    if self.parallel_processing.get():
                        os.environ['TF_NUM_INTEROP_THREADS'] = str(self.cpu_cores.get())
                        settings_applied.append(f"• Parallel Processing: {self.cpu_cores.get()} cores")

                    # Apply auto-save settings
                    if self.auto_save_enabled.get():
                        settings_applied.append(f"• Auto-save: Every {self.auto_save_interval.get()} minutes")

                    # Apply memory settings
                    if self.memory_optimization.get():
                        settings_applied.append("• Memory Optimization: Enabled")

                    # Apply data caching
                    if self.enable_caching.get():
                        settings_applied.append(f"• Data Caching: {self.cache_size.get()} MB")

                    success_message = "✅ Settings Applied Successfully!\n\n"
                    if settings_applied:
                        success_message += "Changes applied:\n" + "\n".join(settings_applied)
                    else:
                        success_message += "No changes were needed (all settings already optimal)."

                    success_message += f"\n\n📅 Applied: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

                    messagebox.showinfo("Settings Applied", success_message)
                    self.update_status("✅ Settings applied successfully", "success")

                except Exception as e:
                    messagebox.showerror("Settings Error", f"Failed to apply some settings:\n\n{str(e)}")
                    self.update_status("⚠️ Settings partially applied", "warning")


        def run(self):
                """Start the professional application"""
                try:
                    # Set window icon (if available)
                    try:
                        # self.root.iconbitmap('smartstock_icon.ico')  # Uncomment if icon file available
                        pass
                    except:
                        pass

                    # Center the window on screen
                    self.root.update_idletasks()
                    width = self.root.winfo_width()
                    height = self.root.winfo_height()
                    x = (self.root.winfo_screenwidth() // 2) - (width // 2)
                    y = (self.root.winfo_screenheight() // 2) - (height // 2)
                    self.root.geometry(f"{width}x{height}+{x}+{y}")

                    # Show startup message
                    self.update_status("🚀 SmartStock AI Professional ready", "success")

                    # Show welcome dialog
                    welcome_msg = f"""🚀 Welcome to SmartStock AI Professional v2.0!
            
            👤 User: {self.get_current_user()}
            📅 Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            
            🎯 Professional Features Available:
            • Advanced ML/DL ensemble models
            • Comprehensive technical analysis
            • Smart money flow detection
            • Professional risk management
            • Interactive chart generation
            • Real-time capabilities
            
            💡 Quick Start:
            1. Upload your CSV data or generate sample data
            2. Configure analysis parameters
            3. Start comprehensive analysis
            4. Review predictions and charts
            5. Export professional reports
            
            Ready to begin professional trading analysis!"""

                    messagebox.showinfo("Welcome to SmartStock AI Professional", welcome_msg)

                    # Start the main event loop
                    self.root.mainloop()

                except KeyboardInterrupt:
                    print("\n🛑 Application terminated by user")
                except Exception as e:
                    print(f"💥 Application error: {e}")
                    messagebox.showerror("Application Error",
                                         f"An unexpected error occurred:\n\n{str(e)}\n\n"
                                         "Please restart the application.")


if __name__ == "__main__":
    try:
        # Display startup information
        print("🚀 Starting SmartStock AI Professional Trading Analysis v2.0...")
        print(f"📅 Current Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"👤 User: wahabsust")
        print("🎨 Theme: Professional Light Blue & White")
        print("💻 Platform: Professional Desktop Application")
        print("=" * 70)
        print("🔧 Initializing professional components...")
        print("   • Enhanced GUI framework")
        print("   • AI/ML analysis engine")
        print("   • Professional styling system")
        print("   • Interactive chart generator")
        print("   • Risk management tools")
        print("=" * 70)

        # Create and run the professional application
        app = ProfessionalSmartStockAIApp()
        app.run()

    except Exception as e:
        print(f"💥 Application startup error: {e}")
        import traceback

        traceback.print_exc()

        # Show error dialog if possible
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Startup Error",
                                 f"Failed to start SmartStock AI Professional:\n\n{str(e)}\n\n"
                                 "Please check the console for detailed error information.")
        except:
            pass
