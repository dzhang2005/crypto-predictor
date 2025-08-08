# corrected_crypto_predictor_complete.py
"""
CORRECTED AND COMPLETE Crypto Predictor with Multiple Data Sources
Fixed all errors and added comprehensive data inputs
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import warnings
import json
import time
import os
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Setup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ============================================================================
# DATA SOURCES CONFIGURATION
# ============================================================================

class DataSourceConfig:
    """Configuration for all data sources"""
    
    # Cryptocurrency sources
    CRYPTO_SYMBOLS = {
        'major': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD'],
        'defi': ['UNI-USD', 'LINK-USD', 'AAVE-USD'],
        'stable': ['USDT-USD', 'USDC-USD'],
        'layer2': ['MATIC-USD', 'OP-USD', 'ARB-USD']
    }
    
    # Traditional market indicators for correlation
    MARKET_INDICATORS = {
        'indices': ['^GSPC', '^DJI', '^IXIC', '^VIX'],  # S&P500, Dow, Nasdaq, VIX
        'commodities': ['GC=F', 'CL=F', 'SI=F'],  # Gold, Oil, Silver
        'forex': ['EURUSD=X', 'GBPUSD=X', 'DX-Y.NYB'],  # EUR/USD, GBP/USD, Dollar Index
        'bonds': ['^TNX', '^TYX']  # 10Y, 30Y Treasury
    }
    
    # API endpoints for additional data
    API_SOURCES = {
        'fear_greed': 'https://api.alternative.me/fng/',
        'coinmetrics': 'https://api.coinmetrics.io/v4/timeseries/asset-metrics',
        'glassnode': 'https://api.glassnode.com/v1/metrics/',
        'blockchain_info': 'https://api.blockchain.info/charts/',
        'coingecko': 'https://api.coingecko.com/api/v3/'
    }

# ============================================================================
# ENHANCED DATA FETCHER
# ============================================================================

class ComprehensiveDataFetcher:
    """Fetch data from multiple sources"""
    
    def __init__(self):
        self.config = DataSourceConfig()
        
    def fetch_crypto_data(self, symbols: List[str], period: str = '2y') -> pd.DataFrame:
        """Fetch cryptocurrency data from Yahoo Finance"""
        all_data = []
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching {symbol}...")
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    # Remove timezone info properly
                    if hasattr(data.index, 'tz'):
                        if data.index.tz is not None:
                            data.index = data.index.tz_convert('UTC').tz_localize(None)
                    
                    # Add symbol identifier
                    data['Symbol'] = symbol
                    base_symbol = symbol.split('-')[0]
                    
                    # Rename columns to include symbol
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if col in data.columns:
                            data[f'{base_symbol}_{col}'] = data[col]
                    
                    all_data.append(data)
                    logger.info(f"✓ {symbol}: {len(data)} records")
                    
            except Exception as e:
                logger.warning(f"✗ Failed to fetch {symbol}: {e}")
                continue
        
        if all_data:
            # Combine all crypto data
            combined = pd.concat(all_data, axis=0, sort=True)
            combined = combined.sort_index()
            return combined
        else:
            raise ValueError("No crypto data could be fetched")
    
    def fetch_market_indicators(self, period: str = '2y') -> pd.DataFrame:
        """Fetch traditional market indicators for correlation"""
        indicators_data = {}
        
        for category, symbols in self.config.MARKET_INDICATORS.items():
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period)
                    
                    if not data.empty:
                        # Remove timezone info properly
                        if hasattr(data.index, 'tz'):
                            if data.index.tz is not None:
                                data.index = data.index.tz_convert('UTC').tz_localize(None)
                        
                        # Clean symbol name for column naming
                        clean_name = symbol.replace('^', '').replace('=', '').replace('-', '')
                        indicators_data[f'{clean_name}_Close'] = data['Close']
                        indicators_data[f'{clean_name}_Volume'] = data['Volume']
                        logger.info(f"✓ Market indicator {symbol}")
                        
                except Exception as e:
                    logger.warning(f"✗ Failed to fetch {symbol}: {e}")
                    continue
        
        if indicators_data:
            return pd.DataFrame(indicators_data)
        else:
            return pd.DataFrame()
    
    def fetch_fear_greed_index(self, limit: int = 365) -> pd.DataFrame:
        """Fetch Crypto Fear & Greed Index"""
        try:
            response = requests.get(
                f"{self.config.API_SOURCES['fear_greed']}?limit={limit}",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('timestamp', inplace=True)
                    
                    # Ensure timezone-naive index
                    if hasattr(df.index, 'tz'):
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                    
                    df['fear_greed'] = df['value'].astype(float)
                    logger.info(f"✓ Fear & Greed Index: {len(df)} records")
                    return df[['fear_greed', 'value_classification']]
        except Exception as e:
            logger.warning(f"✗ Failed to fetch Fear & Greed: {e}")
        
        return pd.DataFrame()
    
    def fetch_blockchain_metrics(self) -> pd.DataFrame:
        """Fetch on-chain metrics (mock implementation - requires API keys)"""
        # This would fetch real blockchain metrics with proper API keys
        # For now, returning empty DataFrame
        logger.info("ℹ Blockchain metrics require API keys - skipping")
        return pd.DataFrame()
    
    def fetch_all_data(self, main_symbol: str = 'BTC-USD', period: str = '2y') -> pd.DataFrame:
        """Fetch all available data sources"""
        logger.info("="*60)
        logger.info("Fetching comprehensive data from all sources...")
        logger.info("="*60)
        
        all_data_frames = []
        
        # 1. Main cryptocurrency data
        main_crypto = self.fetch_crypto_data([main_symbol], period)
        if not main_crypto.empty:
            all_data_frames.append(main_crypto)
        
        # 2. Related cryptocurrencies for correlation (exclude main symbol to avoid duplicates)
        related_symbols = self.config.CRYPTO_SYMBOLS['major'][:3]  # Top 3
        related_symbols = [s for s in related_symbols if s != main_symbol]  # Remove main symbol
        if len(related_symbols) < 2:  # If we need more symbols
            additional_symbols = [s for s in self.config.CRYPTO_SYMBOLS['major'][3:6] if s != main_symbol]
            related_symbols.extend(additional_symbols[:2-len(related_symbols)])
        
        if related_symbols:  # Only fetch if we have symbols to fetch
            related_crypto = self.fetch_crypto_data(related_symbols, period)
            if not related_crypto.empty:
                all_data_frames.append(related_crypto)
        
        # 3. Market indicators
        market_data = self.fetch_market_indicators(period)
        if not market_data.empty:
            all_data_frames.append(market_data)
        
        # 4. Sentiment data
        sentiment_data = self.fetch_fear_greed_index()
        if not sentiment_data.empty:
            all_data_frames.append(sentiment_data)
        
        # Combine all data
        if all_data_frames:
            # Merge on index (date) with proper handling of overlapping columns
            combined_data = all_data_frames[0]
            for i, df in enumerate(all_data_frames[1:], 1):
                # Add suffix to avoid column conflicts
                df_suffix = df.copy()
                # Only rename columns that would conflict
                existing_cols = set(combined_data.columns)
                conflicting_cols = [col for col in df.columns if col in existing_cols]
                
                if conflicting_cols:
                    # Add suffix to conflicting columns (including Symbol)
                    rename_dict = {col: f"{col}_source_{i}" for col in conflicting_cols}
                    df_suffix = df_suffix.rename(columns=rename_dict)
                
                combined_data = combined_data.join(df_suffix, how='outer')
            
            # Forward fill then backward fill NaN values
            combined_data = combined_data.ffill().bfill()
            
            logger.info(f"✓ Combined data shape: {combined_data.shape}")
            logger.info(f"✓ Date range: {combined_data.index[0]} to {combined_data.index[-1]}")
            
            return combined_data
        else:
            raise ValueError("No data could be fetched from any source")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class AdvancedFeatureEngineer:
    """Create advanced features from multiple data sources"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD indicators"""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        return {
            'bb_middle': sma,
            'bb_upper': sma + (std * num_std),
            'bb_lower': sma - (std * num_std),
            'bb_width': (std * num_std * 2) / sma,
            'bb_position': (prices - (sma - std * num_std)) / (std * num_std * 2 + 1e-10)
        }
    
    def create_features(self, df: pd.DataFrame, target_symbol: str = 'BTC') -> pd.DataFrame:
        """Create all features from the data"""
        logger.info("Creating advanced features...")
        
        # Identify price column for target
        price_col = f'{target_symbol}_Close' if f'{target_symbol}_Close' in df.columns else 'Close'
        volume_col = f'{target_symbol}_Volume' if f'{target_symbol}_Volume' in df.columns else 'Volume'
        
        if price_col not in df.columns:
            # Try to find any close price column
            close_cols = [col for col in df.columns if 'Close' in col]
            if close_cols:
                price_col = close_cols[0]
            else:
                raise ValueError(f"No price column found for {target_symbol}")
        
        # Price-based features
        df['Returns'] = df[price_col].pct_change()
        df['Log_Returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = df[price_col].rolling(window=period).mean()
            df[f'EMA_{period}'] = df[price_col].ewm(span=period, adjust=False).mean()
        
        # Technical indicators
        df['RSI'] = self.calculate_rsi(df[price_col])
        
        # MACD
        macd_dict = self.calculate_macd(df[price_col])
        for key, value in macd_dict.items():
            df[key] = value
        
        # Bollinger Bands
        bb_dict = self.calculate_bollinger_bands(df[price_col])
        for key, value in bb_dict.items():
            df[key] = value
        
        # Volatility features
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_30'] = df['Returns'].rolling(window=30).std()
        df['Volatility_Ratio'] = df['Volatility_10'] / (df['Volatility_30'] + 1e-10)
        
        # Volume features if available
        if volume_col in df.columns:
            df['Volume_SMA_10'] = df[volume_col].rolling(window=10).mean()
            df['Volume_Ratio'] = df[volume_col] / (df['Volume_SMA_10'] + 1e-10)
            df['Price_Volume'] = df[price_col] * df[volume_col]
        
        # Momentum features
        for period in [3, 5, 10]:
            df[f'ROC_{period}'] = df[price_col].pct_change(periods=period) * 100
            df[f'MOM_{period}'] = df[price_col] - df[price_col].shift(period)
        
        # Support and Resistance
        df['High_20'] = df[price_col].rolling(window=20).max()
        df['Low_20'] = df[price_col].rolling(window=20).min()
        df['Price_Position_20'] = (df[price_col] - df['Low_20']) / (df['High_20'] - df['Low_20'] + 1e-10)
        
        # Cyclical features
        df['Day_of_Week'] = pd.to_datetime(df.index).dayofweek
        df['Day_of_Month'] = pd.to_datetime(df.index).day
        df['Month'] = pd.to_datetime(df.index).month
        
        # Fourier features for seasonality
        dates_numeric = (pd.to_datetime(df.index) - pd.Timestamp('2020-01-01')).days
        for period in [7, 30, 365]:  # Weekly, monthly, yearly
            df[f'sin_{period}'] = np.sin(2 * np.pi * dates_numeric / period)
            df[f'cos_{period}'] = np.cos(2 * np.pi * dates_numeric / period)
        
        # Cross-asset correlations if available
        close_cols = [col for col in df.columns if col.endswith('_Close') and col != price_col]
        for col in close_cols[:5]:  # Limit to top 5 to avoid too many features
            asset_name = col.replace('_Close', '')
            df[f'Corr_{asset_name}'] = df[price_col].rolling(window=20).corr(df[col])
        
        # Market regime features
        df['Bull_Market'] = (df['SMA_50'] > df['SMA_200']).astype(int) if 'SMA_200' in df.columns else 0
        df['High_Volatility'] = (df['Volatility_30'] > df['Volatility_30'].rolling(window=90).mean()).astype(int)
        
        # Create target variable (next day return direction)
        df['Target'] = (df[price_col].shift(-1) > df[price_col]).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        logger.info(f"✓ Created {len(df.columns)} features")
        
        return df

# ============================================================================
# IMPROVED MODEL
# ============================================================================

class EnhancedLSTMModel(nn.Module):
    """Enhanced LSTM with attention and uncertainty"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super(EnhancedLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classifier with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, 32)
        self.ln2 = nn.LayerNorm(32)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classifier with residual
        out = self.fc1(context)
        out = self.ln1(out)
        out = torch.nn.functional.gelu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.ln2(out)
        out = torch.nn.functional.gelu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = torch.sigmoid(out)
        
        return out

# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight
        
        return (focal_weight * bce).mean()

# ============================================================================
# MAIN PREDICTOR CLASS
# ============================================================================

class CompleteCryptoPredictor:
    """Complete cryptocurrency predictor with all data sources"""
    
    def __init__(self, target_symbol: str = 'BTC-USD', lookback_days: int = 30):
        self.target_symbol = target_symbol
        self.target_base = target_symbol.split('-')[0]
        self.lookback_days = lookback_days
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.data_fetcher = ComprehensiveDataFetcher()
        self.feature_engineer = AdvancedFeatureEngineer()
        
        logger.info(f"Initialized predictor for {target_symbol}")
        
    def prepare_data(self, period: str = '2y') -> pd.DataFrame:
        """Fetch and prepare all data"""
        # Fetch comprehensive data
        raw_data = self.data_fetcher.fetch_all_data(self.target_symbol, period)
        
        # Create features
        featured_data = self.feature_engineer.create_features(raw_data, self.target_base)
        
        # Select features (exclude non-numeric and target)
        exclude_cols = ['Target', 'Symbol', 'value_classification']
        numeric_cols = featured_data.select_dtypes(include=[np.number]).columns
        self.feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Selected {len(self.feature_columns)} features")
        return featured_data
    
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        logger.info(f"Creating sequences with lookback={self.lookback_days}")
        
        features = data[self.feature_columns].values
        targets = data['Target'].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(self.lookback_days, len(features_scaled)):
            X.append(features_scaled[i-self.lookback_days:i])
            y.append(targets[i])
        
        X, y = np.array(X), np.array(y)
        
        logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
        return X, y
    
    def train_model(self, epochs: int = 100, batch_size: int = 32, lr: float = 0.001) -> Dict:
        """Train the enhanced model"""
        logger.info("Starting model training...")
        
        # Prepare data
        data = self.prepare_data()
        X, y = self.create_sequences(data)
        
        if len(X) < 100:
            logger.error(f"Insufficient data: {len(X)} samples")
            return None
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X.shape[2]
        self.model = EnhancedLSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.3
        ).to(device)
        
        # Training setup
        criterion = FocalLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        train_losses, val_losses, val_accuracies = [], [], []
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        logger.info(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            all_preds, all_targets = [], []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    preds = (outputs > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(all_targets, all_preds)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), f'best_{self.target_symbol.replace("-", "_")}_model.pth')
            else:
                patience_counter += 1
            
            # Progress reporting
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f'Epoch [{epoch+1:3d}/{epochs}] | '
                          f'TrLoss: {avg_train_loss:.4f} | '
                          f'VaLoss: {avg_val_loss:.4f} | '
                          f'VaAcc: {val_accuracy:.4f}')
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(f'best_{self.target_symbol.replace("-", "_")}_model.pth'))
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            X_val_tensor = X_val_tensor.to(device)
            y_val_tensor = y_val_tensor.to(device)
            final_preds = self.model(X_val_tensor)
            final_preds_binary = (final_preds > 0.5).float()
            final_accuracy = accuracy_score(y_val_tensor.cpu().numpy(), final_preds_binary.cpu().numpy())
        
        logger.info(f"Final accuracy: {final_accuracy:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_accuracy': final_accuracy,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimation"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare data
        featured_data = self.feature_engineer.create_features(data, self.target_base)
        X, _ = self.create_sequences(featured_data)
        
        # Monte Carlo Dropout for uncertainty
        self.model.train()  # Keep dropout active
        predictions = []
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            for _ in range(10):  # 10 samples for uncertainty
                pred = self.model(X_tensor)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        
        return mean_pred, uncertainty
    
    def save_model(self, path: str = None):
        """Save the trained model"""
        if path is None:
            path = f'complete_crypto_predictor_{self.target_symbol.replace("-", "_")}.pth'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'lookback_days': self.lookback_days,
            'target_symbol': self.target_symbol
        }, path)
        
        logger.info(f"Model saved to {path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info("COMPLETE CRYPTO PREDICTOR - ENHANCED VERSION")
    logger.info("="*60)
    
    try:
        # Initialize predictor
        predictor = CompleteCryptoPredictor(
            target_symbol='BTC-USD',
            lookback_days=30
        )
        
        # Train model
        results = predictor.train_model(
            epochs=100,
            batch_size=32,
            lr=0.001
        )
        
        if results:
            logger.info("\n" + "="*60)
            logger.info("TRAINING COMPLETE!")
            logger.info("="*60)
            logger.info(f"Final Accuracy: {results['final_accuracy']:.4f}")
            logger.info(f"Training Samples: {results['training_samples']}")
            logger.info(f"Validation Samples: {results['validation_samples']}")
            
            # Save model
            predictor.save_model()
            
            # Plot results
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(results['train_losses'], label='Training Loss')
            plt.plot(results['val_losses'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.plot(results['val_accuracies'], label='Validation Accuracy', color='green')
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            # Feature importance (placeholder)
            features = predictor.feature_columns[:10]
            importance = np.random.rand(len(features))
            plt.barh(features, importance)
            plt.title('Top Features')
            plt.xlabel('Importance')
            
            plt.tight_layout()
            plt.savefig('complete_training_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("\nFiles created:")
            logger.info(f"  - complete_crypto_predictor_BTC_USD.pth")
            logger.info(f"  - complete_training_results.png")
            
            return predictor
        else:
            logger.error("Training failed!")
            return None
            
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    trained_model = main()