# crypto_predictor_pytorch.py
# Complete PyTorch-based cryptocurrency price prediction system
# Optimized for CUDA 12.9 + RTX 3090 + Python 3.12.10
# Fixed: Windows encoding issues + PyTorch API compatibility

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import warnings
import os
import logging
import time
import sys
warnings.filterwarnings('ignore')

# Set up logging with proper Windows encoding
log_handlers = []

# Console handler with UTF-8 encoding
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# File handler with UTF-8 encoding
try:
    file_handler = logging.FileHandler('crypto_predictor.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    log_handlers.append(file_handler)
except:
    # Fallback if UTF-8 not supported
    file_handler = logging.FileHandler('crypto_predictor.log')
    file_handler.setLevel(logging.INFO)
    log_handlers.append(file_handler)

log_handlers.append(console_handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"ROCKET Using device: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU Device: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
    logger.info(f"PyTorch CUDA: Compatible with CUDA 12.9")

class LSTMPredictor(nn.Module):
    """
    LSTM-based cryptocurrency price prediction model with attention mechanism
    """
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        logger.info(f"BUILDING Initializing LSTM model: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism (simple)
        self.attention = nn.Linear(hidden_size, 1)
        
        # Classification head with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MODEL STATS: {total_params:,} total parameters, {trainable_params:,} trainable")
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention to LSTM outputs
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(context)
        return output

class CryptoPredictorPyTorch:
    """
    Main cryptocurrency prediction system using PyTorch and LSTM
    """
    def __init__(self, coin_symbol='BTC-USD', lookback_days=60, hidden_size=128):
        logger.info(f"INIT Initializing CryptoPredictorPyTorch for {coin_symbol}")
        
        self.coin_symbol = coin_symbol
        self.lookback_days = lookback_days
        self.hidden_size = hidden_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.device = device
        self.feature_columns = []
        
        logger.info(f"CONFIG: lookback_days={lookback_days}, hidden_size={hidden_size}")
        
        # Windows optimizations for PyTorch
        if torch.cuda.is_available():
            logger.info("CUDA OPTS: Applying CUDA optimizations...")
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("CUDA OPTS: Applied successfully")
        
    def fetch_price_data(self, period='2y'):
        """Fetch historical price data using yfinance"""
        try:
            logger.info(f"FETCH PRICE: Getting data for {self.coin_symbol} (period: {period})...")
            start_time = time.time()
            
            ticker = yf.Ticker(self.coin_symbol)
            data = ticker.history(period=period)
            
            fetch_time = time.time() - start_time
            
            if data.empty:
                logger.error(f"ERROR: No data found for {self.coin_symbol}")
                return None
            
            # Log timezone information
            logger.info(f"DATA TIMEZONE: {data.index.tz}")
            logger.info(f"DATA RANGE: {data.index[0]} to {data.index[-1]}")
            logger.info(f"SUCCESS: Retrieved {len(data)} days of price data in {fetch_time:.2f}s")
            
            # Ensure timezone-naive for consistency
            if data.index.tz is not None:
                logger.info("TIMEZONE FIX: Converting to timezone-naive (UTC)")
                data.index = data.index.tz_convert('UTC').tz_localize(None)
            
            return data[['Close', 'Volume', 'High', 'Low', 'Open']]
            
        except Exception as e:
            logger.error(f"ERROR FETCH PRICE: {e}")
            return None
    
    def fetch_fear_greed_index(self):
        """Get crypto fear & greed index as sentiment proxy"""
        try:
            logger.info("FETCH SENTIMENT: Getting Fear & Greed Index...")
            start_time = time.time()
            
            url = "https://api.alternative.me/fng/?limit=200"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"WARNING: Fear & Greed API returned status {response.status_code}")
                return None
            
            data = response.json()
            
            if 'data' not in data:
                logger.warning("WARNING: Unexpected Fear & Greed API response format")
                return None
            
            fear_greed_df = pd.DataFrame(data['data'])
            
            # Convert timestamp and ensure timezone-naive
            logger.info("PROCESS SENTIMENT: Processing timestamps...")
            fear_greed_df['timestamp'] = pd.to_datetime(fear_greed_df['timestamp'], unit='s', utc=True)
            fear_greed_df['timestamp'] = fear_greed_df['timestamp'].dt.tz_localize(None)  # Make timezone-naive
            fear_greed_df['value'] = fear_greed_df['value'].astype(int)
            fear_greed_df = fear_greed_df.set_index('timestamp')
            
            fetch_time = time.time() - start_time
            logger.info(f"SENTIMENT RANGE: {fear_greed_df.index[0]} to {fear_greed_df.index[-1]}")
            logger.info(f"SUCCESS SENTIMENT: Retrieved {len(fear_greed_df)} readings in {fetch_time:.2f}s")
            
            return fear_greed_df[['value']].rename(columns={'value': 'fear_greed'})
            
        except Exception as e:
            logger.warning(f"WARNING SENTIMENT: Could not fetch data: {e}")
            logger.info("FALLBACK: Will use neutral sentiment (50) for all periods")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        logger.info("TECH INDICATORS: Calculating...")
        start_time = time.time()
        
        df = data.copy()
        initial_rows = len(df)
        
        logger.info(f"INPUT DATA: {initial_rows} rows, columns: {list(df.columns)}")
        
        # Moving averages
        logger.info("CALCULATING: Moving averages...")
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_21'] = df['Close'].rolling(window=21).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        logger.info("CALCULATING: MACD...")
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        logger.info("CALCULATING: RSI...")
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # Bollinger Bands
        logger.info("CALCULATING: Bollinger Bands...")
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volatility measures
        logger.info("CALCULATING: Volatility indicators...")
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['ATR'] = self.calculate_atr(df)  # Average True Range
        
        # Price change features
        logger.info("CALCULATING: Price change features...")
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_2d'] = df['Close'].pct_change(2)
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        
        # Support/Resistance levels
        logger.info("CALCULATING: Support/resistance levels...")
        df['Price_Position'] = (df['Close'] - df['Low'].rolling(20).min()) / (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        
        # Momentum indicators
        logger.info("CALCULATING: Momentum indicators...")
        df['ROC'] = df['Close'].pct_change(periods=10) * 100  # Rate of Change
        df['Williams_R'] = self.calculate_williams_r(df)
        
        # Volume indicators
        logger.info("CALCULATING: Volume indicators...")
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        calc_time = time.time() - start_time
        logger.info(f"SUCCESS TECH: Calculated in {calc_time:.2f}s")
        logger.info(f"FINAL SHAPE: {df.shape}")
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    def calculate_williams_r(self, df, window=14):
        """Calculate Williams %R"""
        highest_high = df['High'].rolling(window=window).max()
        lowest_low = df['Low'].rolling(window=window).min()
        return ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100
    
    def prepare_features(self, data):
        """Engineer features and create target variable"""
        logger.info("FEATURE ENG: Starting feature engineering...")
        start_time = time.time()
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(data)
        
        # Add sentiment data with proper timezone handling
        logger.info("JOIN SENTIMENT: Joining sentiment data...")
        sentiment_data = self.fetch_fear_greed_index()
        
        if sentiment_data is not None:
            try:
                # Ensure both DataFrames have timezone-naive indexes
                logger.info(f"PRICE INDEX: type={type(df.index)}, timezone={getattr(df.index, 'tz', 'None')}")
                logger.info(f"SENTIMENT INDEX: type={type(sentiment_data.index)}, timezone={getattr(sentiment_data.index, 'tz', 'None')}")
                
                # Join with overlap information
                before_join = len(df)
                df = df.join(sentiment_data, how='left')
                after_join = len(df)
                
                logger.info(f"JOIN RESULT: {before_join} -> {after_join} rows")
                
                # Forward fill missing sentiment values, then use neutral value
                missing_before = df['fear_greed'].isna().sum()
                df['fear_greed'] = df['fear_greed'].fillna(method='ffill').fillna(50)
                missing_after = df['fear_greed'].isna().sum()
                
                logger.info(f"SENTIMENT FILL: {missing_before} missing values filled, {missing_after} remaining")
                logger.info(f"SENTIMENT RANGE: {df['fear_greed'].min():.1f} to {df['fear_greed'].max():.1f}")
                
            except Exception as e:
                logger.error(f"ERROR JOIN SENTIMENT: {e}")
                logger.info("FALLBACK: Using neutral sentiment (50) for all periods")
                df['fear_greed'] = 50
        else:
            logger.info("NEUTRAL SENTIMENT: Using value 50 for all periods")
            df['fear_greed'] = 50  # Neutral sentiment
        
        # Create target variable
        logger.info("TARGET: Creating target variable...")
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        target_distribution = df['Target'].value_counts()
        logger.info(f"TARGET DIST: {dict(target_distribution)}")
        
        # Define feature columns
        self.feature_columns = [
            'Close', 'Volume', 'High', 'Low', 'Open',
            'SMA_7', 'SMA_21', 'SMA_50', 'EMA_12', 'EMA_26', 
            'MACD', 'MACD_signal', 'MACD_histogram',
            'RSI', 'BB_width', 'BB_position', 'Volatility', 'ATR',
            'Price_Change', 'Price_Change_2d', 'Volume_Change', 
            'High_Low_Ratio', 'Price_Position', 'ROC', 'Williams_R',
            'Volume_Ratio', 'fear_greed'
        ]
        
        logger.info(f"FEATURE LIST: {len(self.feature_columns)} features defined")
        for i, feature in enumerate(self.feature_columns):
            logger.info(f"   {i+1:2d}. {feature}")
        
        # Remove rows with NaN values
        before_clean = len(df)
        df_clean = df.dropna()
        after_clean = len(df_clean)
        
        logger.info(f"DATA CLEAN: {before_clean} -> {after_clean} rows ({before_clean - after_clean} removed)")
        
        # Check for any remaining issues
        missing_features = []
        for feature in self.feature_columns:
            if feature not in df_clean.columns:
                missing_features.append(feature)
            else:
                missing_count = df_clean[feature].isna().sum()
                if missing_count > 0:
                    logger.warning(f"WARNING: {feature} has {missing_count} missing values")
        
        if missing_features:
            logger.error(f"ERROR: Missing features: {missing_features}")
            return None
        
        prep_time = time.time() - start_time
        logger.info(f"SUCCESS FEATURE ENG: Complete in {prep_time:.2f}s: {len(self.feature_columns)} features, {len(df_clean)} samples")
        
        return df_clean[self.feature_columns + ['Target']]
    
    def create_sequences(self, data):
        """Create LSTM sequences for training"""
        logger.info(f"CREATE SEQ: Creating sequences with lookback period: {self.lookback_days} days")
        start_time = time.time()
        
        # Separate features and targets
        features = data[self.feature_columns].values
        targets = data['Target'].values
        
        logger.info(f"INPUT SHAPES: features={features.shape}, targets={targets.shape}")
        
        # Normalize features
        logger.info("NORMALIZE: Scaling features...")
        features_scaled = self.scaler.fit_transform(features)
        
        # Log normalization statistics
        logger.info(f"SCALING: min={features_scaled.min():.4f}, max={features_scaled.max():.4f}")
        
        X, y = [], []
        for i in range(self.lookback_days, len(features_scaled)):
            X.append(features_scaled[i-self.lookback_days:i])
            y.append(targets[i])
        
        X, y = np.array(X), np.array(y)
        
        seq_time = time.time() - start_time
        logger.info(f"SUCCESS SEQ: Created {len(X)} sequences in {seq_time:.2f}s")
        logger.info(f"SEQUENCE SHAPES: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def create_data_loaders(self, X, y, test_size=0.2, batch_size=64):
        """Create PyTorch data loaders with proper time series split"""
        logger.info(f"DATA LOADERS: Creating (test_size={test_size}, batch_size={batch_size})...")
        
        # Split data (time series - no shuffling to maintain temporal order)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"DATA SPLIT: Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"TRAIN TARGETS: {np.bincount(y_train.astype(int))}")
        logger.info(f"TEST TARGETS: {np.bincount(y_test.astype(int))}")
        
        # Convert to tensors
        logger.info("TENSOR CONVERT: Converting to PyTorch tensors...")
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        logger.info(f"TENSOR SHAPES: X_train={X_train_tensor.shape}, y_train={y_train_tensor.shape}")
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"LOADERS READY: {len(train_loader)} train batches, {len(test_loader)} test batches")
        
        return train_loader, test_loader, (X_test_tensor, y_test_tensor)
    
    def train(self, epochs=100, batch_size=128, lr=0.001, patience=15):
        """Train the PyTorch model with optimizations for RTX 3090"""
        logger.info(f"TRAINING START: {self.coin_symbol}")
        logger.info("=" * 60)
        logger.info(f"TRAINING PARAMS: epochs={epochs}, batch_size={batch_size}, lr={lr}, patience={patience}")
        
        training_start_time = time.time()
        
        # Get price data
        price_data = self.fetch_price_data()
        if price_data is None:
            logger.error("FAILED: Could not fetch price data")
            return None
        
        # Prepare features
        featured_data = self.prepare_features(price_data)
        if featured_data is None or len(featured_data) < self.lookback_days + 50:
            logger.error(f"INSUFFICIENT DATA: {len(featured_data) if featured_data is not None else 0} samples")
            return None
        
        # Create sequences
        X, y = self.create_sequences(featured_data)
        if len(X) == 0:
            logger.error("FAILED: Could not create sequences")
            return None
        
        # Create data loaders with larger batch size for RTX 3090
        train_loader, test_loader, (X_test, y_test) = self.create_data_loaders(
            X, y, batch_size=batch_size
        )
        
        # Initialize model with optimized architecture for RTX 3090
        input_size = X.shape[2]
        logger.info(f"MODEL INIT: Initializing with input_size={input_size}")
        
        self.model = LSTMPredictor(
            input_size=input_size, 
            hidden_size=self.hidden_size,
            num_layers=3,  # Increased for better performance
            dropout=0.3
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Fixed: Remove verbose parameter that doesn't exist in this PyTorch version
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=7, factor=0.5
        )
        
        logger.info(f"OPTIMIZER: AdamW with lr={lr}, weight_decay=1e-4")
        logger.info(f"LOSS: BCELoss")
        logger.info(f"SCHEDULER: ReduceLROnPlateau (patience=7, factor=0.5)")
        
        # Training tracking
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"TRAINING: Starting on {self.device}")
        logger.info(f"BATCHES: {len(train_loader)} train, {len(test_loader)} test")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    preds = (outputs > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            train_accuracy = train_correct / train_total
            val_accuracy = accuracy_score(all_targets, all_preds)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                model_path = f'best_{self.coin_symbol.replace("-", "_")}_model.pth'
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"BEST MODEL: Saved {model_path} (val_loss: {avg_val_loss:.4f})")
            else:
                patience_counter += 1
            
            epoch_time = time.time() - epoch_start_time
            
            # Progress reporting
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f'Epoch [{epoch+1:3d}/{epochs}] | '
                          f'TrLoss: {avg_train_loss:.4f} | '
                          f'VaLoss: {avg_val_loss:.4f} | '
                          f'TrAcc: {train_accuracy:.4f} | '
                          f'VaAcc: {val_accuracy:.4f} | '
                          f'LR: {optimizer.param_groups[0]["lr"]:.6f} | '
                          f'Time: {epoch_time:.1f}s')
            
            if patience_counter >= patience:
                logger.info(f"EARLY STOP: At epoch {epoch+1} (patience reached)")
                break
        
        # Load best model
        best_model_path = f'best_{self.coin_symbol.replace("-", "_")}_model.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        logger.info(f"LOADED: Best model from {best_model_path}")
        
        # Final evaluation
        logger.info("\n" + "=" * 60)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 60)
        
        with torch.no_grad():
            self.model.eval()
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            predictions = self.model(X_test)
            final_preds = (predictions > 0.5).float()
            final_accuracy = accuracy_score(y_test.cpu().numpy(), final_preds.cpu().numpy())
            
            logger.info(f"FINAL ACCURACY: {final_accuracy:.4f}")
            
            # Classification report
            target_names = ['Down', 'Up']
            class_report = classification_report(
                y_test.cpu().numpy(), 
                final_preds.cpu().numpy(), 
                target_names=target_names,
                digits=4
            )
            logger.info(f"CLASSIFICATION REPORT:\n{class_report}")
        
        total_training_time = time.time() - training_start_time
        logger.info(f"TRAINING TIME: {total_training_time:.1f}s ({total_training_time/60:.1f} minutes)")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_accuracy': final_accuracy,
            'best_epoch': len(train_losses) - patience_counter,
            'training_time': total_training_time
        }
    
    def predict_next_movement(self):
        """Predict next day's price movement"""
        if self.model is None:
            logger.error("ERROR: Model not trained yet!")
            return None
        
        try:
            logger.info(f"PREDICT: Making prediction for {self.coin_symbol}...")
            pred_start_time = time.time()
            
            # Get latest data (more recent period for prediction)
            latest_data = self.fetch_price_data(period='120d')
            if latest_data is None:
                return None
                
            featured_data = self.prepare_features(latest_data)
            if featured_data is None or len(featured_data) < self.lookback_days:
                logger.error(f"INSUFFICIENT PRED DATA: {len(featured_data) if featured_data is not None else 0} rows")
                return None
            
            # Get last sequence
            last_sequence = featured_data[self.feature_columns].tail(self.lookback_days).values
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.device)
            logger.info(f"INPUT TENSOR: shape={input_tensor.shape}")
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(input_tensor)
                confidence = float(prediction[0][0])
            
            direction = "UP" if confidence > 0.5 else "DOWN"
            confidence_score = abs(confidence - 0.5) * 2  # Convert to 0-1 confidence
            
            # Determine signal strength
            if confidence_score > 0.8:
                signal_strength = "Very Strong"
            elif confidence_score > 0.6:
                signal_strength = "Strong"
            elif confidence_score > 0.4:
                signal_strength = "Moderate"
            else:
                signal_strength = "Weak"
            
            # Get current price for context
            current_price = latest_data['Close'].iloc[-1]
            
            pred_time = time.time() - pred_start_time
            
            result = {
                'direction': direction,
                'confidence': confidence_score,
                'raw_prediction': confidence,
                'signal_strength': signal_strength,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'prediction_time': pred_time
            }
            
            logger.info(f"PREDICTION RESULT: {direction} (confidence: {confidence_score:.2%}) in {pred_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"PREDICTION ERROR: {e}")
            return None
    
    def save_model(self, path=None):
        """Save the trained model and scaler"""
        if self.model is not None:
            if path is None:
                path = f'crypto_predictor_{self.coin_symbol.replace("-", "_")}.pth'
            
            logger.info(f"SAVE MODEL: Saving to {path}...")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'lookback_days': self.lookback_days,
                'hidden_size': self.hidden_size,
                'coin_symbol': self.coin_symbol
            }, path)
            logger.info(f"SAVE SUCCESS: Model saved")
    
    def load_model(self, path):
        """Load a trained model"""
        try:
            logger.info(f"LOAD MODEL: Loading from {path}...")
            checkpoint = torch.load(path, map_location=self.device)
            
            # Restore configuration
            self.scaler = checkpoint['scaler']
            self.feature_columns = checkpoint['feature_columns']
            self.lookback_days = checkpoint['lookback_days']
            self.hidden_size = checkpoint['hidden_size']
            self.coin_symbol = checkpoint['coin_symbol']
            
            # Initialize and load model
            input_size = len(self.feature_columns)
            self.model = LSTMPredictor(
                input_size=input_size, 
                hidden_size=self.hidden_size
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"LOAD SUCCESS: Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"LOAD ERROR: {e}")
            return False

def main():
    """Main function to train and test the crypto predictor"""
    logger.info("CRYPTO PREDICTOR - PyTorch Edition")
    logger.info("Optimized for CUDA 12.9 + RTX 3090 + Python 3.12.10")
    logger.info("=" * 60)
    
    main_start_time = time.time()
    
    # Initialize predictor with optimal settings for RTX 3090
    predictor = CryptoPredictorPyTorch(
        coin_symbol='BTC-USD',
        lookback_days=60,
        hidden_size=256  # Increased for RTX 3090's 24GB VRAM
    )
    
    # Train model with optimal batch size for RTX 3090
    logger.info("MAIN: Starting model training...")
    history = predictor.train(
        epochs=150,         # More epochs for better performance
        batch_size=128,     # Large batch size for 24GB VRAM
        lr=0.0008,          # Slightly lower learning rate
        patience=20         # More patience for complex model
    )
    
    if history:
        logger.info(f"\nTRAINING COMPLETE!")
        logger.info(f"BEST ACCURACY: {history['final_accuracy']:.4f}")
        logger.info(f"TRAINING TIME: {history['training_time']:.1f}s ({history['training_time']/60:.1f} minutes)")
        
        # Save the model
        predictor.save_model()
        
        # Test prediction
        logger.info("\nTEST PREDICTION: Testing prediction capability...")
        prediction = predictor.predict_next_movement()
        
        if prediction:
            logger.info(f"\nSAMPLE PREDICTION for {predictor.coin_symbol}:")
            logger.info(f"   Direction: {prediction['direction']}")
            logger.info(f"   Confidence: {prediction['confidence']:.2%}")
            logger.info(f"   Signal Strength: {prediction['signal_strength']}")
            logger.info(f"   Current Price: ${prediction['current_price']:.2f}")
            logger.info(f"   Prediction Time: {prediction['prediction_time']:.3f}s")
        
        # Plot training history
        logger.info("\nPLOTTING: Generating training plots...")
        try:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(history['train_losses'], label='Training Loss', alpha=0.8)
            plt.plot(history['val_losses'], label='Validation Loss', alpha=0.8)
            plt.title('Model Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 2)
            plt.plot(history['val_accuracies'], label='Validation Accuracy', color='green')
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
            plt.title('Model Accuracy Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 3)
            # GPU utilization info
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                plt.bar(['GPU Memory Used', 'GPU Memory Total'], 
                       [gpu_memory_used, gpu_memory_total],
                       color=['orange', 'lightblue'])
                plt.title('GPU Memory Usage (GB)')
                plt.ylabel('Memory (GB)')
            
            plt.tight_layout()
            plot_filename = f'training_results_{predictor.coin_symbol.replace("-", "_")}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"PLOT SAVED: {plot_filename}")
            
        except Exception as e:
            logger.warning(f"PLOT WARNING: Could not generate plots: {e}")
        
        # Summary
        total_time = time.time() - main_start_time
        logger.info(f"\nCOMPLETE: Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"FILES SAVED:")
        logger.info(f"   MODEL: crypto_predictor_{predictor.coin_symbol.replace('-', '_')}.pth")
        logger.info(f"   PLOT: training_results_{predictor.coin_symbol.replace('-', '_')}.png")
        logger.info(f"   LOG: crypto_predictor.log")
        logger.info(f"READY FOR DEPLOYMENT!")
        
        return predictor
    else:
        logger.error("TRAINING FAILED!")
        return None

if __name__ == "__main__":
    # Run the main training function
    try:
        trained_model = main()
        
        if trained_model:
            logger.info("\nSUCCESS! Your crypto predictor is ready!")
            logger.info("\nNEXT STEPS:")
            logger.info("   1. Run the Flask web app: python flask_api_pytorch.py")
            logger.info("   2. Deploy to your .ai domain")
            logger.info("   3. Start making predictions!")
        else:
            logger.error("\nSETUP INCOMPLETE. Please check the error messages above.")
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        raise